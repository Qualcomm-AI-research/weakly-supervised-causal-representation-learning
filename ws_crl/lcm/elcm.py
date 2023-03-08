# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import numpy as np
import torch

from ws_crl.lcm.base import BaseLCM


class ELCM(BaseLCM):
    """
    Top-level class for explicit LCMs, generative models with
    - an SCM with a learned or fixed causal graph
    - separate encoder and decoder (i.e. a VAE) outputting causal variables
    - no inference module for interventions
    """

    def forward(
        self,
        x1,
        x2,
        interventions=None,
        beta=1.0,
        beta_intervention_target=None,
        pretrain_beta=None,
        full_likelihood=True,
        likelihood_reduction="sum",
        z_regularization=None,
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        pretrain=False,
        **kwargs,
    ):
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

        # Check inputs
        if pretrain_beta is None:
            pretrain_beta = beta

        feature_dims = list(range(1, len(x1.shape)))
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2))

        # Encode and compute posterior
        z1, log_posterior1, encoder_std1 = self.encoder(x1, deterministic=False, return_std=True)
        z2, log_posterior2, encoder_std2 = self.encoder(x2, deterministic=False, return_std=True)
        log_posterior = log_posterior1 + log_posterior2

        # Decode and compute log likelihood / reconstruction error
        x1_reco, log_likelihood1, decoder_std1 = self.decoder(
            z1,
            eval_likelihood_at=x1,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        x2_reco, log_likelihood2, decoder_std2 = self.decoder(
            z2,
            eval_likelihood_at=x2,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        log_likelihood = log_likelihood1 + log_likelihood2

        # Regularization terms: e norm, MSE, inverse consistency MSE (|z - encode(decode(z))|^2)
        z_norm = torch.sum(z1**2, 1, keepdim=True) + torch.sum(z2**2, 1, keepdim=True)
        mse = torch.sum((x1_reco - x1) ** 2, feature_dims).unsqueeze(1)
        mse += torch.sum((x2_reco - x2) ** 2, feature_dims).unsqueeze(1)
        consistency_mse = mse
        z1_reencoded = self.encode_to_causal(x1_reco, deterministic=False)
        z2_reencoded = self.encode_to_causal(x2_reco, deterministic=False)
        inverse_consistency_mse = torch.sum((z1_reencoded - z1) ** 2, 1, keepdim=True)
        inverse_consistency_mse += torch.sum((z2_reencoded - z2) ** 2, 1, keepdim=True)
        encoder_std = 0.5 * torch.mean(encoder_std1 + encoder_std2, dim=1, keepdim=True)

        # Pretraining
        if pretrain:
            return self.forward_pretrain(
                z1,
                z2,
                log_likelihood=log_likelihood,
                log_posterior=log_posterior,
                z_norm=z_norm,
                consistency_mse=consistency_mse,
                encoder_std=encoder_std,
                beta=pretrain_beta,
            )

        # Evaluate prior p(z, z') [or p(z, z'|interventions)]
        log_prior, outputs = self._evaluate_prior(
            z1,
            z2,
            interventions,
            graph_mode=graph_mode,
            graph_temperature=graph_temperature,
            graph_samples=graph_samples,
        )

        # Compute intervention posterior p(I|z, z') from log_prior
        # Note that outputs["scm_log_prior"] has shape (batchsize, graphs, intervention, 1)
        if "scm_log_prior" in outputs:
            intervention_posterior = torch.softmax(
                torch.mean(outputs["scm_log_prior"], dim=1).squeeze(), 1
            )
        else:
            intervention_posterior = None

        # Put together to compute
        # ELBO = log p(x) - KL[q(inputs|x)|p(inputs|x)]
        #      = E_z[log p(x|inputs) + log p(inputs) - log q(inputs|x)]
        #      <= log p(x)
        # Note that this has the opposite sign from the VAE loss: the larger ELBO, the better
        # To train, we'll actually use a beta-VAE loss
        kl = log_posterior - log_prior
        elbo = log_likelihood - kl
        beta_vae_loss = -log_likelihood + beta * kl

        # Track individual components
        outputs["elbo"] = elbo
        outputs["beta_vae_loss"] = beta_vae_loss
        outputs["kl"] = kl
        outputs["log_likelihood"] = log_likelihood
        outputs["log_posterior"] = log_posterior
        outputs["log_prior"] = log_prior
        outputs["mse"] = mse
        outputs["consistency_mse"] = consistency_mse
        outputs["inverse_consistency_mse"] = inverse_consistency_mse
        outputs["z_regularization"] = z_norm
        outputs["encoder_std"] = encoder_std

        if self.scm.graph is not None:
            outputs["edges"] = self.scm.graph.num_edges
            outputs["cyclicity"] = self.scm.graph.acyclicity_regularizer

        if intervention_posterior is not None:
            outputs["intervention_posterior"] = intervention_posterior

        return beta_vae_loss, outputs

    def forward_pretrain(
        self,
        z1,
        z2,
        log_likelihood,
        log_posterior,
        consistency_mse,
        encoder_std,
        z_norm,
        beta,
    ):
        """Forward mode for pretraining (with trivial prior)"""
        # Compute prior and beta-VAE loss
        log_prior1 = torch.sum(
            self.scm.base_density.log_prob(z1.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )
        log_prior2 = torch.sum(
            self.scm.base_density.log_prob(z2.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )
        beta_vae_loss = -log_likelihood + beta * (log_posterior - log_prior1 - log_prior2)

        # Pretraining loss
        outputs = dict(
            z_regularization=z_norm, consistency_mse=consistency_mse, encoder_std=encoder_std
        )

        return beta_vae_loss, outputs

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
        x, _ = self.decoder(z, deterministic=deterministic)
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

        x, _ = self.decoder(z, deterministic=deterministic)
        return x

    def log_likelihood(self, x1, x2, interventions=None, n_latent_samples=20, **kwargs):
        """
        Computes estimate of the log likelihood using importance weighting, like in IWAE.

        `log p(x) = log E_{inputs ~ q(inputs|x)} [ p(x|inputs) p(inputs) / q(inputs|x) ]`
        """

        # Copy each sample n_latent_samples times
        x1_ = self._expand(x1, repeats=n_latent_samples)
        x2_ = self._expand(x2, repeats=n_latent_samples)
        interventions_ = self._expand(interventions, repeats=n_latent_samples)

        # Evaluate ELBO
        negative_elbo, _ = self.forward(x1_, x2_, interventions_, beta=1.0)

        # Compute importance-weighted estimate of log likelihood
        log_likelihood = self._contract(-negative_elbo, mode="logmeanexp", repeats=n_latent_samples)

        return log_likelihood

    def encode_decode_pair(
        self, x1, x2, deterministic=True, graph_mode="deterministic", graph_temperature=1.0
    ):
        """Auto-encode data pair and return latent representations and reconstructions"""
        # Push to latent space
        z1 = self.encode_to_causal(x1, deterministic=deterministic)
        z2 = self.encode_to_causal(x2, deterministic=deterministic)

        # Sample adjacency matrices
        z1 = self._expand(z1, repeats=1)
        z2 = self._expand(z2, repeats=1)
        if self.scm.graph is None:
            adjacency_matrices = None
        else:
            adjacency_matrices = self.scm.graph.sample_adjacency_matrices(
                z1.shape[0], mode=graph_mode, temperature=graph_temperature
            )

        # Infer intervention target
        z1_ = self._expand(z1)
        z2_ = self._expand(z2)
        if self.scm.graph is None:
            adjacency_matrices_ = None
        else:
            adjacency_matrices_ = self._expand(adjacency_matrices)
        interventions = self._enumerate_interventions(z1_, z2_)
        log_prob_z1_z2_given_interventions, _ = self._evaluate_intervention_conditional_prior(
            z1_, z2_, interventions=interventions, adjacency_matrices=adjacency_matrices_
        )
        log_prob_z1_z2_given_interventions = self._contract(
            log_prob_z1_z2_given_interventions, "reshape"
        )
        log_prob_interventions = torch.log(torch.softmax(log_prob_z1_z2_given_interventions, 1))
        most_likely_intervention = torch.argmax(log_prob_interventions, dim=1).flatten()
        intervention = self._interventions[most_likely_intervention]

        # Project back to data space
        x1_reco = self.decode_causal(z1, deterministic=deterministic)
        x2_reco = self.decode_causal(z2, deterministic=deterministic)

        return (
            x1_reco,
            x2_reco,
            z1,
            z2,
            z1,
            z2,
            torch.exp(log_prob_interventions),
            most_likely_intervention,
            intervention,
        )

    def infer_intervention(
        self,
        x1,
        x2,
        deterministic_encoder=True,
        sharp_manifold=True,
        graph_mode="deterministic",
        graph_temperature=1.0,
    ):
        """Given data pair, infer intervention"""

        # Push to latent space
        z1 = self.encode_to_causal(x1, deterministic=deterministic_encoder)
        z2 = self.encode_to_causal(x2, deterministic=deterministic_encoder)

        # Sample adjacency matrices
        z1 = self._expand(z1, repeats=1)
        z2 = self._expand(z2, repeats=1)
        adjacency_matrices, _ = self.scm.graph.sample_adjacency_matrices(
            z1.shape[0], mode=graph_mode, temperature=graph_temperature
        )

        # Infer intervention target
        z1_ = self._expand(z1)
        z2_ = self._expand(z2)
        interventions = self._enumerate_interventions(z1_, z2_)
        log_prob_z1_z2_given_interventions, _ = self._evaluate_intervention_conditional_prior(
            z1_, z2_, interventions=interventions, adjacency_matrices=adjacency_matrices
        )
        log_prob_z1_z2_given_interventions = self._contract(
            log_prob_z1_z2_given_interventions, "reshape"
        )
        log_prob_interventions = torch.log(torch.softmax(log_prob_z1_z2_given_interventions, 1))
        most_likely_intervention = torch.argmax(log_prob_interventions, dim=1).flatten()
        intervention_mask = self._interventions[most_likely_intervention]

        # Resample z2 for this intervention
        z2_resampled = self.scm.generate_similar_intervention(
            z1, z2, intervention=intervention_mask, sharp_manifold=sharp_manifold
        )
        x2_resampled = self.decode_causal(z2_resampled)

        return most_likely_intervention, log_prob_interventions, x2_resampled

    def mcmc(
        self,
        x1,
        x2,
        z1_init,
        z2_init,
        interventions=None,
        n_steps=1000,
        initial_step_size=0.05,
        final_step_size=1.0e-4,
    ):
        """Sampling from posterior p(z1, z2 | x1, x2) with Metropolis-Hastings"""

        with torch.no_grad():
            chain = []
            z = (z1_init, z2_init)
            log_joint = self._compute_log_joint_density(x1, x2, *z, interventions=interventions)

            for step in range(n_steps):
                stepsize = np.exp(
                    np.log(initial_step_size)
                    + (np.log(final_step_size) - np.log(initial_step_size)) * step / (n_steps - 1)
                )
                z_proposed = self._mh_proposal(*z, stepsize)
                log_joint_proposed = self._compute_log_joint_density(x1, x2, *z_proposed)

                acceptance_ratio = torch.exp(log_joint_proposed - log_joint).item()
                u = torch.rand([])
                if acceptance_ratio >= u:
                    z = z_proposed
                    log_joint = log_joint_proposed

                chain.append(z)

        return chain

    @staticmethod
    def _mh_proposal(z1, z2, step_size):
        z1_proposal = z1 + step_size * torch.randn(z1.shape)
        z2_proposal = z2 + step_size * torch.randn(z2.shape)
        return z1_proposal, z2_proposal

    def _compute_log_joint_density(self, x1, x2, z1, z2, interventions=None):
        """Computes p(x1 | z1) p(x2 | z2) p(z1, z2)"""

        # Likelihood
        _, (log_likelihood1, _) = self.decoder(z1, eval_likelihood_at=x1)
        _, (log_likelihood2, _) = self.decoder(z2, eval_likelihood_at=x2)
        log_likelihood = log_likelihood1 + log_likelihood2

        # Prior
        log_prob_z, _ = self._evaluate_prior(z1, z2, interventions=interventions)

        # Return joint log density
        return log_prob_z + log_likelihood
