# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import nflows.transforms
import nflows.utils
import torch
from torch import nn

from ws_crl.nets import make_mlp


class MaskedSolutionTransform(nn.Module):
    """Transform wrapper around the solution function in an SCM"""

    def __init__(self, scm, scm_component):
        super().__init__()
        self.scm = scm
        self.scm_component = scm_component

    def forward(self, inputs, context):
        masked_context = self.scm.get_masked_context(
            self.scm_component, epsilon=context, adjacency_matrix=None
        )
        return self.scm.solution_functions[self.scm_component](inputs, context=masked_context)

    def inverse(self, inputs, context):
        masked_context = self.scm.get_masked_context(
            self.scm_component, epsilon=context, adjacency_matrix=None
        )
        return self.scm.solution_functions[self.scm_component](inputs, context=masked_context)


def make_scalar_transform(
    n_features,
    layers=3,
    hidden=10,
    transform_blocks=1,
    sigmoid=False,
    transform="affine",
    conditional_features=0,
    bins=10,
    tail_bound=10.0,
):
    """Utility function that constructs an invertible transformation for unstructured data"""

    def transform_net_factory_fn(in_features, out_features):
        # noinspection PyUnresolvedReferences
        return nflows.nn.nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden,
            context_features=conditional_features,
            num_blocks=transform_blocks,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    transforms = []
    for i in range(layers):
        transforms.append(nflows.transforms.RandomPermutation(features=n_features))
        if transform == "affine":
            transforms.append(
                nflows.transforms.AffineCouplingTransform(
                    mask=nflows.utils.create_alternating_binary_mask(n_features, even=(i % 2 == 0)),
                    transform_net_create_fn=transform_net_factory_fn,
                )
            )
        elif transform == "piecewise_linear":
            transforms.append(
                nflows.transforms.PiecewiseLinearCouplingTransform(
                    mask=nflows.utils.create_alternating_binary_mask(n_features, even=(i % 2 == 0)),
                    transform_net_create_fn=transform_net_factory_fn,
                    tail_bound=tail_bound,
                    num_bins=bins,
                    tails="linear",
                )
            )
        else:
            raise ValueError(transform)
    transforms.append(nflows.transforms.RandomPermutation(features=n_features))
    if sigmoid:
        transforms.append(nflows.transforms.Sigmoid())

    return nflows.transforms.CompositeTransform(transforms)


class ConditionalAffineScalarTransform(nflows.transforms.Transform):
    """
    Computes X = X * scale(context) + shift(context), where (scale, shift) are given by
    param_net(context). param_net takes as input the context with shape (batchsize,
    context_features) or None, its output has to have shape (batchsize, 2).
    """

    def __init__(self, param_net=None, features=None, conditional_std=True, min_scale=None):
        super().__init__()

        self.conditional_std = conditional_std
        self.param_net = param_net

        if self.param_net is None:
            assert features is not None
            self.shift = torch.zeros(features)
            torch.nn.init.normal_(self.shift)
            self.shift = torch.nn.Parameter(self.shift)
        else:
            self.shift = None

        if self.param_net is None or not conditional_std:
            self.log_scale = torch.zeros(features)
            torch.nn.init.normal_(self.log_scale)
            self.log_scale = torch.nn.Parameter(self.log_scale)
        else:
            self.log_scale = None

        if min_scale is None:
            self.min_scale = None
        else:
            self.register_buffer("min_scale", torch.tensor(min_scale))

    def get_scale_and_shift(self, context):
        if self.param_net is None:
            shift = self.shift.unsqueeze(1)
            log_scale = self.log_scale.unsqueeze(1)
        elif not self.conditional_std:
            shift = self.param_net(context)
            log_scale = self.log_scale.unsqueeze(1)
        else:
            log_scale_and_shift = self.param_net(context)
            log_scale = log_scale_and_shift[:, 0].unsqueeze(1)
            shift = log_scale_and_shift[:, 1].unsqueeze(1)

        scale = torch.exp(log_scale)
        if self.min_scale is not None:
            scale = scale + self.min_scale

        num_dims = torch.prod(torch.tensor([1]), dtype=torch.float)
        logabsdet = torch.log(scale) * num_dims

        return scale, shift, logabsdet

    def forward(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = inputs * scale + shift
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale, shift, logabsdet = self.get_scale_and_shift(context)
        outputs = (inputs - shift) / scale
        return outputs, -logabsdet


def make_intervention_transform(homoskedastic, enhance_causal_effects, min_std=None):
    """
    Utility function that constructs an invertible transformation for interventional distributions
    in SCMs
    """

    trf = ConditionalAffineScalarTransform(
        param_net=None, features=1, conditional_std=not homoskedastic, min_scale=min_std
    )
    torch.nn.init.normal_(trf.shift, mean=0.0, std=1.0 if enhance_causal_effects else 1.0e-3)
    torch.nn.init.normal_(trf.log_scale, mean=0.0, std=1.0e-3)

    return trf


def make_mlp_structure_transform(
    dim_z,
    hidden_layers,
    hidden_units,
    homoskedastic,
    min_std,
    concat_masks_to_parents=True,
    initialization="default",
):
    """
    Utility function that constructs an invertible transformation for causal mechanisms
    in SCMs
    """
    input_factor = 2 if concat_masks_to_parents else 1
    features = (
        [input_factor * dim_z]
        + [hidden_units for _ in range(hidden_layers)]
        + [1 if homoskedastic else 2]
    )
    param_net = make_mlp(features)

    if initialization == "default":
        # param_net outputs mean and log std parameters of a Gaussian (log std only if
        # homoskedastic = False), as a function of the causal parents.
        # We usually want to initialize param_net such that:
        #  - log std is very close to zero
        #  - mean is reasonably close to zero, but may already have some nontrivial dependence on
        #    the parents
        mean_bias_std = 1.0e-3
        mean_weight_std = 0.1
        log_std_bias_std = 1.0e-6
        log_std_weight_std = 1.0e-3
        log_std_bias_mean = 0.0
    elif initialization == "strong_effects":
        # However, when creating a GT model as an initialized neural SCM, we want slightly more
        # interesting initializations, with pronounced causal effects. That's what the
        # enhance_causal_effects keyword is for. When that's True, we would like the Gaussian mean
        # to depend quite strongly on the parents, and also would appreciate some non-trivial
        # heteroskedasticity (log std depending on the parents).
        mean_bias_std = 0.2
        mean_weight_std = 1.5
        log_std_bias_std = 1.0e-6
        log_std_weight_std = 0.1
        log_std_bias_mean = 0.0
    elif initialization == "broad":
        # For noise-centric models we want that the typical initial standard deviation in p(e2 | e1)
        # is large, around 10
        mean_bias_std = 1.0e-3
        mean_weight_std = 0.1
        log_std_bias_std = 1.0e-6
        log_std_weight_std = 1.0e-3
        log_std_bias_mean = 2.3
    else:
        raise ValueError(f"Unknown initialization scheme {initialization}")

    last_layer = list(param_net._modules.values())[-1]
    if homoskedastic:
        nn.init.normal_(last_layer.bias, mean=0.0, std=mean_bias_std)
        nn.init.normal_(last_layer.weight, mean=0.0, std=mean_weight_std)
    else:
        nn.init.normal_(last_layer.bias[0], mean=log_std_bias_mean, std=log_std_bias_std)
        nn.init.normal_(last_layer.weight[0, :], mean=0.0, std=log_std_weight_std)
        nn.init.normal_(last_layer.bias[1], mean=0.0, std=mean_bias_std)
        nn.init.normal_(last_layer.weight[1, :], mean=0.0, std=mean_weight_std)

    structure_trf = ConditionalAffineScalarTransform(
        param_net=param_net, features=1, conditional_std=not homoskedastic, min_scale=min_std
    )

    return structure_trf
