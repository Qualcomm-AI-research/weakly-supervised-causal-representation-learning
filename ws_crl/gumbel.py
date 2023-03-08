# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Functions for differentiable Gumbel-something sampling """

import torch


def gumbel_bernouilli(logits, tau=1.0, hard=False):
    """Gumbel-Bernouilli distribution"""

    logits_ = logits.unsqueeze(-1)
    logits_ = torch.cat((logits_, torch.zeros_like(logits_)), dim=-1)

    # GS sampling
    # noinspection PyUnresolvedReferences
    existence = torch.nn.functional.gumbel_softmax(logits_, tau=tau, hard=hard)[..., 0]

    # Log probability, based on hard Bernouilli distribution
    p_existence = torch.exp(logits) / (1.0 + torch.exp(logits))
    log_prob = torch.where(existence >= 0.5, torch.log(p_existence), torch.log(1.0 - p_existence))

    return existence, log_prob


def soft_sort(scores, tau=1.0, hard=False, power=1.0):
    """
    SoftSort based on "SoftSort: A Continuous Relaxation for the argsort Operator", ICML 2020.

    Inputs:
    -------
    scores : torch.Tensor of shape (batchsize, n), dtype torch.float
        Parameters to be sorted
    tau : float
        Temperature parameter
    hard : bool
        If True, returns STE
    power : float
    """

    scores = scores.unsqueeze(-1)
    sorted_scores = scores.sort(descending=True, dim=1)[0]

    pairwise_diffs = (scores.transpose(1, 2) - sorted_scores).abs().pow(power).neg() / tau
    soft_permutation = pairwise_diffs.softmax(-1)

    if hard:
        hard_permutation = torch.zeros_like(soft_permutation, device=soft_permutation.device)
        hard_permutation.scatter_(-1, soft_permutation.topk(1, -1)[1], value=1)
        soft_permutation = (hard_permutation - soft_permutation).detach() + soft_permutation

    return soft_permutation


def sample_permutation(logits, tau=1.0, mode="hard"):
    """
    Samples a permutation matrix using Gumbel-Top-k and SoftSort, following "Differentiable DAG
    Sampling" (ICLR 2022)
    """

    assert mode in {"hard", "soft", "deterministic"}

    # Deterministic behaviour (test time)
    if mode == "deterministic":
        permutation = soft_sort(logits, tau=1.0e-9, hard=True)

    # Non-deterministic behaviour (for training)
    else:
        # Sample Gumbel variables
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )

        # SoftSort
        permutation = soft_sort(logits + gumbels, tau=tau, hard=mode == "hard")

    return permutation
