# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

"""
General-purpose neural networks
"""

import torch
from torch import nn


class Quadratic(nn.Module):
    """Neural network layer that implements an elementwise quadratic transform"""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.xavier_normal_(self.weight)

    def forward(self, inputs):
        outputs = inputs**2 @ self.weight[0].T + inputs @ self.weight[1].T
        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs


def get_activation(key):
    """Utility function that returns an activation function given its name"""

    if key == "relu":
        return nn.ReLU()
    elif key == "leaky_relu":
        return nn.LeakyReLU()
    elif key == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation {key}")


def make_mlp(features, activation="relu", final_activation=None, initial_activation=None):
    """Utility function that constructs a simple MLP from specs"""

    if len(features) >= 2:
        layers = []

        if initial_activation is not None:
            layers.append(get_activation(initial_activation))

        for in_, out in zip(features[:-2], features[1:-1]):
            layers.append(nn.Linear(in_, out))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(features[-2], features[-1]))
        if final_activation is not None:
            layers.append(get_activation(final_activation))

        net = nn.Sequential(*layers)

    else:
        net = nn.Identity()

    return net


class ElementwiseResNet(torch.nn.Module):
    """Elementwise ResNet"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        in_shape = inputs.shape
        hidden = inputs.reshape((-1, 1))
        hidden = self.net(hidden)
        residuals = hidden.reshape(in_shape)
        outputs = inputs + residuals
        return outputs


def make_elementwise_mlp(
    hidden_features, activation="relu", final_activation=None, initial_activation=None
):
    """Utility function that constructs an elementwise MLP"""

    if not hidden_features:
        return nn.Identity()

    features = [1] + hidden_features + [1]
    net = make_mlp(
        features,
        activation=activation,
        final_activation=final_activation,
        initial_activation=initial_activation,
    )
    return ElementwiseResNet(net)
