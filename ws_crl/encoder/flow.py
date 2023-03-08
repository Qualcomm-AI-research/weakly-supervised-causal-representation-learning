# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import nflows
import torch
from torch import nn

from ws_crl.encoder.base import Encoder
from ws_crl.transforms import make_scalar_transform


class InvertibleEncoder(Encoder):
    """Base class for deterministic invertible encoders based on a nflows.transforms.Transform"""

    def __init__(self, input_features=2, output_features=2, **kwargs):
        super().__init__(input_features, output_features)
        assert self.input_features == self.output_features
        self.transform = self._make_transform(**kwargs)

    def forward(self, inputs, deterministic=False):
        """Given observed data, returns latent representation; i.e. encoding."""
        return self.transform(inputs)

    def inverse(self, inputs, deterministic=False):
        """Given latent representation, returns observed version; i.e. decoding."""
        return self.transform.inverse(inputs)

    def _make_transform(self, **kwargs):
        raise NotImplementedError


class SONEncoder(Encoder):
    """Deterministic SO(n) encoder / decoder"""

    def __init__(self, coeffs=None, input_features=2, output_features=2, coeff_std=0.05):
        super().__init__(input_features, output_features)

        # Coefficients
        d = self.output_features * (self.output_features - 1) // 2
        if coeffs is None:
            self.coeffs = nn.Parameter(torch.zeros((d,)))  # (d)
            nn.init.normal_(self.coeffs, std=coeff_std)
        else:
            assert coeffs.shape == (d,)
            self.coeffs = nn.Parameter(coeffs)

        # Generators
        self.generators = torch.zeros((d, self.output_features, self.output_features))

    def forward(self, inputs, deterministic=False):
        """Given observed data, returns latent representation; i.e. encoding."""
        z = torch.einsum("ij,bj->bi", self._rotation_matrix(), inputs)
        logdet = torch.zeros([])
        return z, logdet

    def inverse(self, inputs, deterministic=False):
        """Given latent representation, returns observed version; i.e. decoding."""
        x = torch.einsum("ij,bj->bi", self._rotation_matrix(inverse=True), inputs)
        logdet = torch.zeros([])
        return x, logdet

    def _rotation_matrix(self, inverse=False):
        """
        Low-level function to generate an element of SO(n) by exponentiating the Lie algebra
        (skew-symmetric matrices)
        """

        o = torch.zeros(self.output_features, self.output_features, device=self.coeffs.device)
        i, j = torch.triu_indices(self.output_features, self.output_features, offset=1)
        if inverse:
            o[i, j] = -self.coeffs
            o.T[i, j] = self.coeffs
        else:
            o[i, j] = self.coeffs
            o.T[i, j] = -self.coeffs
        a = torch.matrix_exp(o)
        return a


class FlowEncoder(InvertibleEncoder):
    """
    Deterministic invertible encoder based on an affine coupling flow, mapping R^n to R^n or to
    [0, 1]^n
    """

    def __init__(
        self,
        layers=3,
        hidden=10,
        transform_blocks=1,
        sigmoid=False,
        input_features=2,
        output_features=2,
    ):
        super().__init__(
            input_features,
            output_features,
            layers=layers,
            hidden=hidden,
            transform_blocks=transform_blocks,
            sigmoid=sigmoid,
        )

    def _make_transform(self, layers=3, hidden=10, transform_blocks=1, sigmoid=False):
        return make_scalar_transform(
            self.output_features,
            layers=layers,
            hidden=hidden,
            transform_blocks=transform_blocks,
            sigmoid=sigmoid,
        )


class LULinearEncoder(InvertibleEncoder):
    """
    Deterministic linear invertible encoder using an LU decomposition to ensure invertibility
    (note that the det > 0 and det < 0 pieces are disconnected in the parameterization)
    """

    def _make_transform(self, **kwargs):
        # noinspection PyUnresolvedReferences
        return nflows.transforms.LULinear(self.output_features, **kwargs)


class NaiveLinearEncoder(InvertibleEncoder):
    """
    Deterministic linear invertible encoder based on a naive matrix parameterization (inversion is
    slow in higher dimensions)
    """

    def __init__(self, input_features=2, output_features=2, matrix=None, **kwargs):
        super().__init__(input_features, output_features, matrix=matrix)

    def _make_transform(self, matrix, **kwargs):
        # noinspection PyUnresolvedReferences
        transform = nflows.transforms.NaiveLinear(self.output_features, **kwargs)

        if matrix is not None:
            assert matrix.shape == (self.output_features, self.output_features)
            with torch.no_grad():
                transform._weight.copy_(matrix)

        return transform
