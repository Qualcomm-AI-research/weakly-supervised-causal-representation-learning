# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import torch
from torch import nn as nn
from torch.nn import functional as F
from itertools import chain

from ws_crl.encoder.vae import gaussian_encode
from ws_crl.nets import make_mlp, make_elementwise_mlp
from ws_crl.utils import generate_permutation


def add_coords(x):
    """
    Adds coordinate encodings to a tensor.

    Parameters:
    -----------
    x: torch.Tensor of shape (b, c, h, w)
        Input tensor

    Returns:
    --------
    augmented_x: torch.Tensor of shape (b, c+2, h, w)
        Input tensor augmented with two new channels with positional encodings
    """

    b, c, h, w = x.shape
    coords_h = torch.linspace(-1, 1, h, device=x.device)[:, None].expand(b, 1, h, w)
    coords_w = torch.linspace(-1, 1, w, device=x.device).expand(b, 1, h, w)
    return torch.cat([x, coords_h, coords_w], 1)


class CoordConv2d(nn.Module):
    """
    Conv2d that adds coordinate encodings to the input
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(add_coords(x))


class ResNetDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=2,
        batchnorm=True,
        batchnorm_epsilon=0.01,
        conv_class=nn.Conv2d,
    ):
        super(ResNetDown, self).__init__()

        self.conv1 = conv_class(in_features, out_features // 2, 3, padding=1)
        self.conv2 = conv_class(out_features // 2, out_features, 3, padding=1)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_features // 2, eps=batchnorm_epsilon)
            self.bn2 = nn.BatchNorm2d(out_features, eps=batchnorm_epsilon)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.point_wise = conv_class(in_features, out_features, 1)
        self.pool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.point_wise(self.pool(x))

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x + skip)
        return x


class ResNetUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=2,
        batchnorm=True,
        batchnorm_epsilon=0.01,
        conv_class=nn.Conv2d,
    ):
        super(ResNetUp, self).__init__()

        self.conv1 = conv_class(in_features, out_features // 2, 3, padding=1)
        self.conv2 = conv_class(out_features // 2, out_features, 3, padding=1)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_features // 2, eps=batchnorm_epsilon)
            self.bn2 = nn.BatchNorm2d(out_features, eps=batchnorm_epsilon)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.point_wise = conv_class(in_features, out_features, 1)
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x):
        skip = self.point_wise(self.upsample(x))

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = self.bn2(self.conv2(x))

        x = F.leaky_relu(x + skip)
        return x


def vector_to_gaussian(x, min_std=0.0, fix_std=False):
    """
    Map network output to mean and stddev (via softplus) of Gaussian.

    [b, 2*d, ...] -> 2*[b, d, ...]
    """

    if fix_std:
        mu = x
        std = min_std * torch.ones_like(x)
    else:
        d = x.shape[1] // 2
        mu, std_param = x[:, :d], x[:, d:]
        std = F.softplus(std_param) + min_std

    return mu, std


class ImageEncoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z
    """

    def __init__(
        self,
        in_features,
        out_features,
        in_resolution=64,
        hidden_features=64,
        batchnorm=True,
        batchnorm_epsilon=0.1,
        conv_class=nn.Conv2d,
        mlp_layers=0,
        mlp_hidden=64,
        min_std=0.0,
        elementwise_layers=1,
        elementwise_hidden=16,
        permutation=0,
    ):
        super().__init__()

        self.net = self._make_conv_net(
            batchnorm,
            batchnorm_epsilon,
            conv_class,
            hidden_features,
            in_features,
            mlp_hidden,
            mlp_layers,
            out_features,
            in_resolution,
        )

        hidden_units = [mlp_hidden] * mlp_layers + [2 * out_features]
        self.mlp = make_mlp(hidden_units, activation="leaky_relu", initial_activation="leaky_relu")
        self.register_buffer("min_std", torch.tensor(min_std))

        self.elementwise = make_elementwise_mlp(
            [elementwise_hidden] * elementwise_layers, activation="leaky_relu"
        )

        if permutation == 0:
            self.permutation = None
        else:
            self.permutation = generate_permutation(out_features, permutation, inverse=False)

    def _make_conv_net(
        self,
        batchnorm,
        batchnorm_epsilon,
        conv_class,
        hidden_features,
        in_features,
        mlp_hidden,
        mlp_layers,
        out_features,
        in_resolution,
    ):

        net_out_features = mlp_hidden if mlp_layers > 0 else 2 * out_features
        kwargs = {
            "batchnorm": batchnorm,
            "batchnorm_epsilon": batchnorm_epsilon,
            "conv_class": conv_class,
        }

        if in_resolution == 64:
            net = nn.Sequential(
                ResNetDown(in_features, hidden_features, **kwargs),
                ResNetDown(hidden_features, 2 * hidden_features, **kwargs),
                ResNetDown(2 * hidden_features, 4 * hidden_features, **kwargs),
                ResNetDown(4 * hidden_features, 8 * hidden_features, **kwargs),
                ResNetDown(8 * hidden_features, 8 * hidden_features, **kwargs),
                ResNetDown(8 * hidden_features, 8 * hidden_features, **kwargs),
                conv_class(8 * hidden_features, net_out_features, 1),
            )
        elif in_resolution == 512:
            net = nn.Sequential(
                ResNetDown(in_features, hidden_features, **kwargs),
                ResNetDown(hidden_features, 2 * hidden_features, **kwargs),
                ResNetDown(2 * hidden_features, 4 * hidden_features, **kwargs),
                ResNetDown(4 * hidden_features, 8 * hidden_features, **kwargs),
                ResNetDown(8 * hidden_features, 16 * hidden_features, **kwargs),
                ResNetDown(16 * hidden_features, 32 * hidden_features, **kwargs),
                ResNetDown(32 * hidden_features, 32 * hidden_features, **kwargs),
                ResNetDown(32 * hidden_features, 32 * hidden_features, **kwargs),
                ResNetDown(32 * hidden_features, 32 * hidden_features, **kwargs),
                conv_class(32 * hidden_features, net_out_features, 1),
            )

        else:
            raise NotImplementedError(
                f"Haven't implemented convolutional encoder for resolution {in_resolution}"
            )

        return net

    def forward(
        self,
        x,
        eval_likelihood_at=None,
        deterministic=False,
        return_mean=False,
        return_std=False,
        full=True,
        reduction="sum",
    ):
        """
        Encode image, returns Gaussian

        [b, in_channels, 64, 64] -> Gaussian over [b, out_features]
        See gaussian_encode for parameters and return type.
        """

        mean, std = self.mean_std(x)
        return gaussian_encode(
            mean,
            std,
            eval_likelihood_at,
            deterministic,
            return_mean=return_mean,
            return_std=return_std,
            full=full,
            reduction=reduction,
        )

    def mean_std(self, x):
        """Encode image, return mean and std"""
        hidden = self.net(x).squeeze(3).squeeze(2)
        hidden = self.mlp(hidden)
        mean, std = vector_to_gaussian(hidden, min_std=self.min_std)

        mean = self.elementwise(mean)

        if self.permutation is not None:
            mean = mean[:, self.permutation]
            std = std[:, self.permutation]

        return mean, std

    def freeze(self):
        """Freeze convolutional net and MLP, but not elementwise transformation"""
        for parameter in chain(self.mlp.parameters(), self.net.parameters()):
            parameter.requires_grad = False

    def freezable_parameters(self):
        """Returns parameters that should be frozen during training"""
        return chain(self.mlp.parameters(), self.net.parameters())

    def unfreezable_parameters(self):
        """Returns parameters that should not be frozen during training"""
        return self.elementwise.parameters()


class ImageDecoder(nn.Module):
    """
    Decoder block
    """

    def __init__(
        self,
        in_features,
        out_features,
        out_resolution=64,
        hidden_features=64,
        batchnorm=True,
        batchnorm_epsilon=0.1,
        conv_class=nn.Conv2d,
        fix_std=False,
        min_std=1e-3,
        mlp_layers=2,
        mlp_hidden=64,
        elementwise_layers=1,
        elementwise_hidden=16,
        permutation=0,
    ):
        super().__init__()

        if permutation == 0:
            self.permutation = None
        else:
            self.permutation = generate_permutation(in_features, permutation, inverse=True)

        self.elementwise = make_elementwise_mlp(
            [elementwise_hidden] * elementwise_layers, activation="leaky_relu"
        )

        hidden_units = [in_features] + [mlp_hidden] * mlp_layers
        self.mlp = make_mlp(hidden_units, activation="leaky_relu", final_activation="leaky_relu")

        self.net = self._create_conv_net(
            batchnorm,
            batchnorm_epsilon,
            conv_class,
            fix_std,
            hidden_features,
            in_features,
            mlp_hidden,
            mlp_layers,
            out_features,
            out_resolution,
        )
        self.fix_std = fix_std
        self.register_buffer("min_std", torch.tensor(min_std))

    def _create_conv_net(
        self,
        batchnorm,
        batchnorm_epsilon,
        conv_class,
        fix_std,
        hidden_features,
        in_features,
        mlp_hidden,
        mlp_layers,
        out_features,
        out_resolution,
    ):
        net_in_features = mlp_hidden if mlp_layers > 0 else in_features
        feature_multiplier = 1 if fix_std else 2
        kwargs = {
            "batchnorm": batchnorm,
            "batchnorm_epsilon": batchnorm_epsilon,
            "conv_class": conv_class,
        }

        if out_resolution == 64:
            net = nn.Sequential(
                ResNetUp(net_in_features, hidden_features * 8, **kwargs),
                ResNetUp(hidden_features * 8, hidden_features * 8, **kwargs),
                ResNetUp(hidden_features * 8, hidden_features * 4, **kwargs),
                ResNetUp(hidden_features * 4, hidden_features * 2, **kwargs),
                ResNetUp(hidden_features * 2, hidden_features, **kwargs),
                ResNetUp(hidden_features, hidden_features // 2, **kwargs),
                conv_class(hidden_features // 2, feature_multiplier * out_features, 1),
            )
        elif out_resolution == 512:
            net = nn.Sequential(
                ResNetUp(net_in_features, hidden_features * 32, **kwargs),
                ResNetUp(hidden_features * 32, hidden_features * 32, **kwargs),
                ResNetUp(hidden_features * 32, hidden_features * 32, **kwargs),
                ResNetUp(hidden_features * 32, hidden_features * 16, **kwargs),
                ResNetUp(hidden_features * 16, hidden_features * 8, **kwargs),
                ResNetUp(hidden_features * 8, hidden_features * 4, **kwargs),
                ResNetUp(hidden_features * 4, hidden_features * 2, **kwargs),
                ResNetUp(hidden_features * 2, hidden_features, **kwargs),
                ResNetUp(hidden_features, hidden_features // 2, **kwargs),
                conv_class(hidden_features // 2, feature_multiplier * out_features, 1),
            )
        else:
            raise NotImplementedError(
                f"Haven't implemented convolutional decoder for resolution {out_resolution}"
            )

        return net

    def forward(
        self,
        x,
        eval_likelihood_at=None,
        deterministic=False,
        return_mean=False,
        return_std=False,
        full=True,
        reduction="sum",
    ):
        """
        Decodes latent into image, returns Gaussian

        [b, in_channels] -> Gaussian over [b, out_features, 64, 64]
        See gaussian_encode for parameters and return type.
        """

        mean, std = self.mean_std(x)

        return gaussian_encode(
            mean,
            std,
            eval_likelihood_at,
            deterministic,
            return_mean=return_mean,
            return_std=return_std,
            full=full,
            reduction=reduction,
        )

    def mean_std(self, x):
        """Given latent, compute mean and std"""
        if self.permutation is not None:
            x = x[:, self.permutation]

        hidden = self.elementwise(x)
        hidden = self.mlp(hidden)
        hidden = self.net(hidden[:, :, None, None])
        mean, std = vector_to_gaussian(hidden, fix_std=self.fix_std, min_std=self.min_std)

        return mean, std

    def freezable_parameters(self):
        """Returns parameters that should be frozen during training"""
        return chain(self.mlp.parameters(), self.net.parameters())

    def unfreezable_parameters(self):
        """Returns parameters that should not be frozen during training"""
        return self.elementwise.parameters()

    def freeze(self):
        """Freeze convolutional net and MLP, but not elementwise transformation"""
        for parameter in chain(self.mlp.parameters(), self.net.parameters()):
            parameter.requires_grad = False
