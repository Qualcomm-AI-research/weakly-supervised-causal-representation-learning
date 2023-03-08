# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Loss functions, metrics, training utilities """

import itertools

import torch
from torch import nn

LOG_MEAN_VARS = {
    "elbo",
    "kl",
    "kl_epsilon",
    "kl_intervention_target",
    "mse",
    "consistency_mse",
    "inverse_consistency_mse",
    "log_prior",
    "log_prior_observed",
    "log_prior_intervened",
    "log_prior_nonintervened",
    "log_likelihood",
    "log_posterior",
    "z_regularization",
    "edges",
    "cyclicity",
    "encoder_std",
}


class VAEMetrics(nn.Module):
    """Metrics for generative training (maximizing the marginal likelihood / ELBO)"""

    def __init__(self, dim_z=2):
        super().__init__()
        self.dim_z = dim_z

    def forward(
        self,
        loss,
        true_intervention_labels,
        solution_std=None,
        intervention_posterior=None,
        eps=1.0e-9,
        z_regularization_amount=0.0,
        consistency_regularization_amount=0.0,
        inverse_consistency_regularization_amount=0.0,
        edge_regularization_amount=0.0,
        cyclicity_regularization_amount=0.0,
        intervention_entropy_regularization_amount=0.0,
        **model_outputs,
    ):
        metrics = {}
        batchsize = loss.shape[0]

        # beta-VAE loss
        loss = torch.mean(loss)

        # Regularization term
        loss = self._regulate(
            batchsize,
            consistency_regularization_amount,
            eps,
            intervention_entropy_regularization_amount,
            intervention_posterior,
            inverse_consistency_regularization_amount,
            loss,
            metrics,
            model_outputs,
            z_regularization_amount,
            edge_regularization_amount,
            cyclicity_regularization_amount,
        )

        assert torch.isfinite(loss)
        metrics["loss"] = loss.item()

        # Additional logged metrics (non-differentiable)
        with torch.no_grad():
            # Intervention posterior
            self._evaluate_intervention_posterior(
                eps, metrics, true_intervention_labels, intervention_posterior
            )

            # Mean std in p(epsilon2|epsilon1)
            if solution_std is not None:
                for i in range(solution_std.shape[-1]):
                    metrics[f"solution_std_{i}"] = torch.mean(solution_std[..., i]).item()

            # For most other quantities logged, just keep track of the mean
            for key in LOG_MEAN_VARS:
                if key in model_outputs:
                    try:
                        metrics[key] = torch.mean(model_outputs[key].to(torch.float)).item()
                    except AttributeError:
                        metrics[key] = float(model_outputs[key])

        return loss, metrics

    def _regulate(
        self,
        batchsize,
        consistency_regularization_amount,
        eps,
        intervention_entropy_regularization_amount,
        intervention_posterior,
        inverse_consistency_regularization_amount,
        loss,
        metrics,
        model_outputs,
        z_regularization_amount,
        edge_regularization_amount,
        cyclicity_regularization_amount,
    ):
        if edge_regularization_amount is not None and "edges" in model_outputs:
            loss += edge_regularization_amount * torch.mean(model_outputs["edges"])

        if cyclicity_regularization_amount is not None and "cyclicity" in model_outputs:
            try:
                loss += cyclicity_regularization_amount * torch.mean(model_outputs["cyclicity"])
            except TypeError:  # some models return a float
                loss += cyclicity_regularization_amount * model_outputs["cyclicity"]

        if z_regularization_amount is not None and "z_regularization" in model_outputs:
            loss += z_regularization_amount * torch.mean(model_outputs["z_regularization"])

        if consistency_regularization_amount is not None and "consistency_mse" in model_outputs:
            loss += consistency_regularization_amount * torch.mean(model_outputs["consistency_mse"])

        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )

        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )

        if (
            intervention_entropy_regularization_amount is not None
            and intervention_posterior is not None
        ):
            aggregate_posterior = torch.mean(intervention_posterior, dim=0)
            intervention_entropy = -torch.sum(
                aggregate_posterior * torch.log(aggregate_posterior + eps)
            )
            loss -= (
                intervention_entropy_regularization_amount * intervention_entropy
            )  # note minus sign: maximize entropy!
            metrics["intervention_entropy"] = intervention_entropy.item()

            # Let's also log the entropy corresponding to the determinstic (argmax) intervention
            # encoder
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            det_posterior = torch.zeros_like(intervention_posterior)
            det_posterior[torch.arange(batchsize), most_likely_intervention] = 1.0
            aggregate_det_posterior = torch.mean(det_posterior, dim=0)
            det_intervention_entropy = -torch.sum(
                aggregate_det_posterior * torch.log(aggregate_det_posterior + eps)
            )
            metrics["intervention_entropy_deterministic"] = det_intervention_entropy.item()

        return loss

    @torch.no_grad()
    def _evaluate_intervention_posterior(
        self, eps, metrics, true_intervention_labels, intervention_posterior
    ):
        # We don't really want to iterate over all permutations of 32 latent variables
        if self.dim_z > 5:
            return

        # Some methods don't compute an intervention posterior
        if intervention_posterior is None:
            return

        batchsize = true_intervention_labels.shape[0]
        idx = torch.arange(batchsize)

        for i in range(intervention_posterior.shape[1]):
            metrics[f"intervention_posterior_{i}"] = torch.mean(intervention_posterior[:, i]).item()

        # Find all permutations of dim_z variables, and evaluate probability of true intervention
        # + accuracy
        true_int_prob, log_true_int_prob, int_accuracy = -float("inf"), -float("inf"), -float("inf")
        for permutation in itertools.permutations(list(range(1, self.dim_z + 1))):
            permutation = [0] + list(permutation)
            intervention_probs_permuted = intervention_posterior[:, permutation]
            predicted_intervention_permuted = torch.zeros_like(intervention_probs_permuted)
            predicted_intervention_permuted[
                idx, torch.argmax(intervention_probs_permuted, dim=1)
            ] = 1.0

            # log p(I*)
            log_true_int_prob_ = torch.mean(
                torch.log(
                    intervention_probs_permuted[idx, true_intervention_labels.flatten()] + eps
                )
            ).item()
            log_true_int_prob = max(log_true_int_prob, log_true_int_prob_)

            # p(I*)
            true_int_prob_ = torch.mean(
                intervention_probs_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            true_int_prob = max(true_int_prob, true_int_prob_)

            # Accuracy
            int_accuracy_ = torch.mean(
                predicted_intervention_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            int_accuracy = max(int_accuracy, int_accuracy_)

        metrics[f"intervention_correct_log_posterior"] = log_true_int_prob
        metrics[f"intervention_correct_posterior"] = true_int_prob
        metrics[f"intervention_accuracy"] = int_accuracy
