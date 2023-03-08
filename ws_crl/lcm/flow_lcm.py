# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

from ws_crl.lcm.base import BaseLCM


class FlowLCM(BaseLCM):
    """
    Top-level class for generative models with
    - an SCM with a fixed causal graph
    - an invertible encoder (the inverse is used as decoder) outputting the causal variables
    - no inference module for interventions
    """

    def __init__(
        self,
        causal_model,
        encoder,
        intervention_prior=None,
        dim_z=2,
        intervention_set="atomic_or_none",
    ):
        super().__init__(
            causal_model,
            encoder,
            decoder=None,
            intervention_prior=intervention_prior,
            dim_z=dim_z,
            intervention_set=intervention_set,
        )

    def forward(self, x1, x2, interventions=None, **kwargs):
        """
        Evaluates an observed data pair.

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
            `log p(x1, x2 | interventions)`.
            If `interventions` is None: Marginal log likelihood `log p(x1, x2)`.
        outputs : dict with str keys and torch.Tensor values
            Detailed breakdown of the model outputs and internals.
        """

        # Push to latent space
        z1, logdet_x1 = self.encoder(x1)
        z2, logdet_x2 = self.encoder(x2)

        # Evaluate likelihood / evidence in latent space
        log_prob_z, outputs = self._evaluate_prior_marginalized_over_interventions(
            z1, z2, interventions
        )

        # Put together to log p(x1, x2)
        log_prob = log_prob_z + logdet_x1 + logdet_x2
        outputs["log_det_j"] = logdet_x1 + logdet_x2

        return log_prob, outputs

    def log_likelihood(self, x1, x2, interventions=None, **kwargs):
        """Given data pair, compute weakly supervised log likelihood"""
        return self.forward(x1, x2, interventions, **kwargs)[0]

    def encode_to_noise(self, x, deterministic=True):
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

        z, _ = self.encoder(x, deterministic=deterministic)
        epsilon = self.scm.causal_to_noise(z)
        return epsilon

    def encode_to_causal(self, x, deterministic=True):
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

        z, _ = self.encoder(x, deterministic=deterministic)
        return z

    def decode_noise(self, epsilon, deterministic=True):
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

        z = self.scm.noise_to_causal(epsilon)
        x, _ = self.encoder.inverse(z)
        return x

    def decode_causal(self, z, deterministic=True):
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

        x, _ = self.encoder.inverse(z)
        return x
