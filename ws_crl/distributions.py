# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Probability distributions based on nflows.distributions.base.Distribution """

import numpy as np
import torch

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class IIDZeroMeanNormal(Distribution):
    """IID normal distribution with zero mean and fixed variance"""

    def __init__(self, shape, std=1.0):
        super().__init__()

        self.register_buffer("mean_", torch.zeros(shape).reshape(1, -1))
        self.register_buffer("log_std_", torch.log(std * torch.ones(shape)).reshape(1, -1))
        self.register_buffer(
            "_log_z",
            torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64),
            persistent=False,
        )
        self._shape = torch.Size(shape)

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, inputs.shape[1:])
            )

        # Compute parameters.
        means = self.mean_
        log_stds = self.log_std_

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs**2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        return torch.exp(self.log_std_) * torch.randn(
            num_samples, *self._shape, device=self._log_z.device
        )

    def _mean(self, context):
        return self.mean
