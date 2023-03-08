# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Generative models in the weakly supervised setting """

import torch
from torch import nn

from ws_crl.causal.interventions import InterventionPrior
from ws_crl.encoder.base import Inverse
from ws_crl.utils import logmeanexp

MEAN_VARS = {}
LOGMEANEXP_VARS = {}


class BaseLCM(nn.Module):
    """Base class for a generative model"""

    def __init__(
        self,
        causal_model,
        encoder,
        decoder=None,
        dim_z=2,
        intervention_prior=None,
        intervention_set="atomic_or_none",
        **kwargs,
    ):
        super().__init__()
        self.dim_z = dim_z

        self.scm = causal_model

        if intervention_prior is None:
            intervention_prior = InterventionPrior(
                0, dim_z=dim_z, intervention_set=intervention_set
            )
        self.intervention_prior = intervention_prior
        self.n_interventions = self.intervention_prior.n_interventions
        self.register_buffer("_interventions", self.intervention_prior._masks.to(torch.float))

        self.encoder = encoder
        if decoder is None:
            self.decoder = Inverse(self.encoder)
        else:
            self.decoder = decoder

        self.register_buffer(
            "dummy", torch.zeros([1])
        )  # Sole purpose is to track the device for self.sample()

    def sample(self, n, additional_noise=None, device=None):
        """
        Samples from the data-generating process in the weakly supervised setting.

        Returns:
        --------
        x1 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (after a uniformly sampled intervention)
        intervention_labels : torch.Tensor of shape (batchsize,), dtype torch.int
            Obfuscated intervention label
        interventions : torch.Tensor of shape (batchsize, self.dim_z), dtype torch.bool
            Intervention masks
        """

        # Sample intervention
        interventions, intervention_labels = self.intervention_prior.sample(n)

        # Sample causal variables
        z1, z2 = self.scm.sample_weakly_supervised(n, interventions)

        # Push to data space
        x1, _ = self.decoder(z1)
        x2, _ = self.decoder(z2)

        # Optionally, add a small amount of observation noise to avoid numerical issues with
        # submanifolds
        if additional_noise:
            x1 += additional_noise * torch.randn(x1.size(), device=x1.device)
            x2 += additional_noise * torch.randn(x2.size(), device=x2.device)

        return x1, x2, z1, z2, intervention_labels, interventions

    def forward(self, x1, x2, interventions=None, **kwargs):
        """
        Forward pass during training. Needs to be implemented by subclasses.

        Arguments:
        ----------
        x1 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (after the intervention)
        interventions : None or torch.Tensor of shape (batchsize, DIM_Z,), dtype torch.float
            If not None, specifies the interventions

        Returns:
        --------
        log_prob : torch.Tensor of shape (batchsize, 1), dtype torch.float
            If `interventions` is not None: Conditional log likelihood
            `log p(x1, x2 | interventions)` or lower bound.
            If `interventions` is None: Marginal log likelihood `log p(x1, x2)` or lower bound..
        outputs : dict with str keys and torch.Tensor values
            Detailed breakdown of the model outputs and internals.
        """

        raise NotImplementedError

    def log_likelihood(self, x1, x2, interventions=None, **kwargs):
        """
        Evaluates the log likelihood of a data pair at test time.

        This may be the same as forward() (in a flow) or different (in a VAE).

        Arguments:
        ----------
        x1 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Observed data point (after the intervention)
        interventions : None or torch.Tensor of shape (batchsize, DIM_Z,), dtype torch.float
            If not None, specifies the interventions

        Returns:
        --------
        log_prob : torch.Tensor of shape (batchsize, 1), dtype torch.float
            If `interventions` is not None: Conditional log likelihood
            `log p(x1, x2 | interventions)`.
            If `interventions` is None: Marginal log likelihood `log p(x1, x2)`.
        """

        raise NotImplementedError

    def encode_to_noise(self, x, deterministic=False):
        """
        Given data x, returns the noise encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        epsilon : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise encoding phi_epsilon(x)
        """
        raise NotImplementedError

    def encode_to_causal(self, x, deterministic=False):
        """
        Given data x, returns the causal encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal-variable encoding phi_z(x)
        """
        raise NotImplementedError

    def decode_noise(self, epsilon, deterministic=False):
        """
        Given noise encoding epsilon, returns data x.

        Arguments:
        ----------
        epsilon : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise-encoded data.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """
        raise NotImplementedError

    def decode_causal(self, z, deterministic=False):
        """
        Given causal latents inputs, returns data x.

        Arguments:
        ----------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal latent variables.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """
        raise NotImplementedError

    def encode_decode(self, x, deterministic=True):
        """Autoencode data and return reconstruction"""
        z = self.encode_to_causal(x, deterministic=deterministic)
        x_reco = self.decode_causal(z, deterministic=deterministic)

        return x_reco

    def _evaluate_prior(
        self,
        z1,
        z2,
        interventions,
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        noise_centric=False,
        include_nonintervened=True,
    ):
        """
        Evaluates prior p(z1, z2) or p(epsilon1, epsilon2).

        If interventions is not None, explicitly marginalizes over all possible interventions with
        a uniform prior.
        """

        # Sample adjacency matrices
        z1 = self._expand(z1, repeats=graph_samples)
        z2 = self._expand(z2, repeats=graph_samples)
        adjacency_matrices = self._sample_adjacency_matrices(
            z1.shape[0], mode=graph_mode, temperature=graph_temperature
        )

        # If interventions are not specified, enumerate them
        if interventions is None:
            z1 = self._expand(z1)
            z2 = self._expand(z2)
            adjacency_matrices_ = self._expand(adjacency_matrices)
            interventions_ = self._enumerate_interventions(z1, z2)
        else:
            adjacency_matrices_ = adjacency_matrices
            interventions_ = interventions

        # Evaluate prior p(z1, z2|interventions)
        log_prob, outputs = self._evaluate_intervention_conditional_prior(
            z1,
            z2,
            interventions_,
            adjacency_matrices_,
            noise_centric=noise_centric,
            include_nonintervened=include_nonintervened,
        )

        # Marginalize over interventions
        if interventions is None:
            outputs = self._contract_dict(outputs)
            log_prob = self._contract(log_prob, mode="logmeanexp")  # Marginalize likelihood

        # Marginalize over adjacency matrices
        log_prob = self._contract(log_prob, repeats=graph_samples, mode="mean")
        outputs = self._contract_dict(outputs, repeats=graph_samples)

        return log_prob, outputs

    def _evaluate_intervention_conditional_prior(
        self,
        z1,
        z2,
        interventions,
        adjacency_matrices,
        noise_centric=False,
        include_nonintervened=True,
    ):
        """Evaluates conditional prior p(z1, z2|I)"""

        # Check inputs
        interventions = self._sanitize_interventions(interventions)

        # Evaluate conditional prior
        if noise_centric:
            log_prob, outputs = self.scm.log_prob_noise_weakly_supervised(
                z1,
                z2,
                interventions,
                adjacency_matrix=adjacency_matrices,
                include_nonintervened=include_nonintervened,
            )
        else:
            log_prob, outputs = self.scm.log_prob_weakly_supervised(
                z1, z2, interventions, adjacency_matrix=adjacency_matrices
            )

        return log_prob, outputs

    def _sanitize_interventions(self, interventions):
        """Ensures correct dtype of interventions"""
        assert interventions.shape[1] == self.dim_z
        return interventions.to(torch.float)

    def _enumerate_interventions(self, z1, z2):
        """Generates interventions"""
        n = (
            z1.shape[0] // self._interventions.shape[0]
        )  # z1 has shape (n_interventions * batchsize, DIM_Z) already
        return self._interventions.repeat_interleave(n, dim=0)

    def _expand(self, x, repeats=None):
        """
        Given x with shape (batchsize, components), repeats elements and returns a tensor of shape
        (batchsize * repeats, components)
        """

        if x is None:
            return None
        if repeats is None:
            repeats = len(self._interventions)

        unaffected_dims = tuple(1 for _ in range(1, len(x.shape)))
        x_expanded = x.repeat(repeats, *unaffected_dims)

        return x_expanded

    def _contract(self, x, mode="mean", repeats=None):
        """
        Given x with shape (batchsize * repeats, components), returns either
         - the mean over repeats, with shape (batchsize, components),
         - the logmeanexp over repeats, with shape (batchsize, components), or
         - the reshaped version with shape (batchsize, repeats, components).
        """

        assert mode in ["sum", "mean", "reshape", "logmeanexp"]

        if x is None:
            return None
        if len(x.shape) == 1:
            return self._contract(x.unsqueeze(1), mode, repeats)

        if repeats is None:
            repeats = len(self._interventions)

        # assert len(x.shape) == 2, x.shape
        y = x.reshape([repeats, -1] + list(x.shape[1:]))
        if mode == "sum":
            return torch.sum(y, dim=0)
        elif mode == "mean":
            return torch.mean(y, dim=0)
        elif mode == "logmeanexp":
            return logmeanexp(y, 0)
        elif mode == "reshape":
            return y.transpose(0, 1)
        else:
            raise ValueError(mode)

    def _contract_dict(self, data, repeats=None):
        """Given a dict of data, contracts each data variable approproately (see `_contract`)"""

        contracted_dict = {}
        for key, val in data.items():
            if key in MEAN_VARS:
                mode = "mean"
            elif key in LOGMEANEXP_VARS:
                mode = "logmeanexp"
            else:
                mode = "reshape"
            contracted_dict[key] = self._contract(val, mode, repeats)

        return contracted_dict

    def _sample_adjacency_matrices(self, *args, **kwargs):
        if self.scm.graph is None:
            return None

        return self.scm.graph.sample_adjacency_matrices(*args, **kwargs)
