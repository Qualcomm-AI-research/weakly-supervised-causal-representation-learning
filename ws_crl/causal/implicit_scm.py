# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Noise-centric SCM representations """

from collections import defaultdict
import nflows.distributions
import nflows.nn.nets
import torch
from torch import nn

from ws_crl.causal.graph import ENCOLearnedGraph, DDSLearnedGraph, FixedOrderLearnedGraph
from ws_crl.transforms import make_mlp_structure_transform, MaskedSolutionTransform
from ws_crl.utils import mask, clean_and_clamp

DEFAULT_BASE_DENSITY = nflows.distributions.StandardNormal((1,))


class ImplicitSCM(nn.Module):
    """
    Implicit causal model, centered around noise encoding and solution functions.

    Parameters:
    -----------
    graph: Graph or None
        Causal graph. If None, the full graph is assumed (no masking)
    solution_functions: list of self.dim_z Transforms
        The i-th element in this list is a diffeo that models `p(e'_i|e)` with noise encodings `e`
        like a flow
    """

    def __init__(
        self, graph, solution_functions, base_density, manifold_thickness, dim_z, causal_structure
    ):
        super().__init__()
        self.dim_z = dim_z

        self.solution_functions = torch.nn.ModuleList(solution_functions)
        self.base_density = base_density
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))
        self.register_buffer("_mask_values", torch.zeros(dim_z))
        self.register_buffer("topological_order", torch.zeros(dim_z, dtype=torch.long))

        self.set_causal_structure(graph, causal_structure)

    def sample(self, n, intervention=None, graph_mode="hard", graph_temperature=1.0):
        """Samples a single latent vector, either observed or under an intervention"""

        raise NotImplementedError

    def sample_weakly_supervised(self, n, intervention, graph_mode="hard", graph_temperature=1.0):
        """Samples in the weakly supervised setting for a given intervention"""

        raise NotImplementedError

    def sample_noise_weakly_supervised(self, n, intervention, adjacency_matrix=None):
        """Samples in the weakly supervised setting for a given intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        # Sample pre-intervention noise encodings
        epsilon1 = self._sample_noise(n)  # noise variables used for the data pre intervention

        # Sample intervention process for targets
        intervention_noise = self._sample_noise(n)  # noise used for the intervened-upon variables
        epsilon2 = (
            intervention
            * self._inverse(intervention_noise, epsilon1, adjacency_matrix=adjacency_matrix)[0]
        )

        # Counterfactual consistency noise for non-intervened variables
        cf_noise = self._sample_noise(n, True)  # noise used for the non-intervened-upon variables
        epsilon2 += (1.0 - intervention) * (epsilon1 + cf_noise)

        return epsilon1, epsilon2

    def log_prob_weakly_supervised(self, z1, z2, intervention, adjacency_matrix):
        """
        Given weakly supervised causal variables and the intervention mask, computes the
        corresponding noise variables and log likelihoods.
        """

        raise NotImplementedError

    def log_prob_noise_weakly_supervised(
        self,
        epsilon1,
        epsilon2,
        intervention,
        adjacency_matrix,
        include_intervened=True,
        include_nonintervened=True,
    ):
        """
        Given weakly supervised as noise encodings epsilon1, epsilon2 and the intervention mask,
        computes the corresponding causal variables and log likelihoods.
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, epsilon1.shape[0])
        assert torch.all(torch.isfinite(epsilon1))
        assert torch.all(torch.isfinite(epsilon2))

        # Observed likelihood
        logprob_observed = self._compute_logprob_observed(epsilon1)
        logprob = logprob_observed

        # Intervention likelihood
        if include_intervened:
            log_det, logprob_intervened = self._compute_logprob_intervened(
                adjacency_matrix, epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_intervened
        else:
            logprob_intervened = torch.zeros_like(logprob_observed)
            log_det = torch.zeros((epsilon1.shape[0], 1), device=epsilon1.device)

        # Counterfactual discrepancy for not-intervened-upon variables
        if include_nonintervened:
            logprob_nonintervened = self._compute_logprob_nonintervened(
                epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_nonintervened
        else:
            logprob_nonintervened = torch.zeros_like(logprob_intervened)

        # Package outputs
        assert torch.all(torch.isfinite(logprob))
        outputs = dict(
            log_prior_observed=logprob_observed,
            log_prior_intervened=logprob_intervened,
            log_prior_nonintervened=logprob_nonintervened,
            solution_std=torch.exp(
                -log_det
            ),  # log_det is log(std) from noise encoding -> z transform
        )

        return logprob, outputs

    def _compute_logprob_nonintervened(self, epsilon1, epsilon2, intervention):
        cf_noise = (epsilon2 - epsilon1) / self.manifold_thickness
        assert torch.all(torch.isfinite(cf_noise))
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened -= torch.log(self.manifold_thickness)
        logprob_nonintervened = clean_and_clamp(logprob_nonintervened)
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened  # (batchsize, self.dim_z)
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)  # (batchsize, 1)
        return logprob_nonintervened

    def _compute_logprob_intervened(self, adjacency_matrix, epsilon1, epsilon2, intervention):
        z_intervened, log_det = self._solve(
            epsilon=epsilon2, conditioning_epsilon=epsilon1, adjacency_matrix=adjacency_matrix
        )
        assert torch.all(torch.isfinite(z_intervened))
        logprob_intervened = self.base_density.log_prob(z_intervened.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_intervened += log_det
        logprob_intervened = intervention * logprob_intervened  # (batchsize, self.dim_z)
        logprob_intervened = clean_and_clamp(logprob_intervened)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)  # (batchsize, 1)
        return log_det, logprob_intervened

    def _compute_logprob_observed(self, epsilon1):
        logprob_observed = self.base_density.log_prob(epsilon1.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed = clean_and_clamp(logprob_observed)
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)  # (batchsize, 1)
        return logprob_observed

    def noise_to_causal(self, epsilon, adjacency_matrix=None):
        """Given noise encoding, returns causal encoding"""

        return self._solve(epsilon, epsilon, adjacency_matrix=adjacency_matrix)[0]

    def causal_to_noise(self, z, adjacency_matrix=None):
        """Given causal latents, returns noise encoding"""

        assert self.topological_order is not None

        conditioning_epsilon = z.clone()
        epsilons = {}

        for i in self.topological_order:
            i = i.item()

            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)
            epsilon, _ = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)

            epsilons[i] = epsilon
            conditioning_epsilon[:, i : i + 1] = epsilon

        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)

        return epsilon

    @property
    def manifold_thickness(self):
        """Returns counterfactual manifold thickness (only here for legacy reasons)"""
        return self._manifold_thickness

    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        """Sets counterfactual manifold thickness (only here for legacy reasons)"""
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))

    @torch.no_grad()
    def get_scm_parameters(self):
        """Returns key parameters of causal model for logging purposes"""
        # Manifold thickness
        parameters = {"manifold_thickness": self.manifold_thickness}

        return parameters

    def generate_similar_intervention(
        self, z1, z2_example, intervention, adjacency_matrix, sharp_manifold=True
    ):
        """Infers intervention and "fakes" it in the model"""
        raise NotImplementedError

    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)

        return intervention

    @torch.no_grad()
    def get_masked_solution_function(self, i):
        """Returns solution function where inputs are masked to conform to topological order"""
        return MaskedSolutionTransform(self, i)

    def _solve(self, epsilon, conditioning_epsilon, adjacency_matrix):
        """
        Given noise encodings, compute causal variables (= base variables in counterfactual flow).
        """

        zs = []
        logdets = []

        for i, transform in enumerate(self.solution_functions):
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)

            z, logdet = transform.inverse(epsilon[:, i : i + 1], context=masked_epsilon)
            zs.append(z)
            logdets.append(logdet)

        z = torch.cat(zs, 1)
        logdet = torch.cat(logdets, 1)

        return z, logdet

    def _inverse(self, z, conditioning_epsilon, adjacency_matrix=None, order=None):
        if order is None:
            assert self.topological_order is not None
            order = self.topological_order

        epsilons = {}
        logdets = {}

        for i in order:
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)

            epsilon, logdet = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)
            epsilons[i] = epsilon
            logdets[i] = logdet

        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)
        logdet = torch.cat([logdets[i] for i in range(self.dim_z)], 1)

        return epsilon, logdet

    def get_masked_context(self, i, epsilon, adjacency_matrix):
        """Masks the input to a solution function to conform to topological order"""
        mask_ = self._get_ancestor_mask(
            i, adjacency_matrix, device=epsilon.device, n=epsilon.shape[0]
        )
        dummy_data = self._mask_values.unsqueeze(0)
        dummy_data[:, i] = 0.0
        masked_epsilon = mask(epsilon, mask_, mask_data=dummy_data)
        return masked_epsilon

    def _get_ancestor_mask(self, i, adjacency_matrix, device, n=1):
        if self.graph is None:
            if self.causal_structure == "fixed_order":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
                ancestor_mask[..., self.ancestor_idx[i]] = 1.0
            elif self.causal_structure == "trivial":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
            else:
                ancestor_mask = torch.ones((n, self.dim_z), device=device)
                ancestor_mask[..., i] = 0.0

        else:
            # Rather than the adjacency matrix, we're computing the
            # non-descendancy matrix: the probability of j not being a descendant of i
            # The idea is that this has to be gradient-friendly, soft adjacency-friendly way.
            # 1 - anc = (1 - adj) * (1 - adj^2) * (1 - adj^3) * ... * (1 - adj^(n-1))
            non_ancestor_matrix = torch.ones_like(adjacency_matrix)
            for n in range(1, self.dim_z):
                non_ancestor_matrix *= 1.0 - torch.linalg.matrix_power(adjacency_matrix, n)

            ancestor_mask = 1.0 - non_ancestor_matrix[..., i]

        return ancestor_mask

    def _sample_noise(self, n, sample_consistency_noise=False):
        """Samples noise"""
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)

    def set_causal_structure(
        self, graph, causal_structure, topological_order=None, mask_values=None
    ):
        """Fixes causal structure, usually to a given topoloigical order"""
        if graph is None:
            assert causal_structure in ["none", "fixed_order", "trivial"]

        if topological_order is None:
            topological_order = list(range(self.dim_z))

        if mask_values is None:
            mask_values = torch.zeros(self.dim_z, device=self._manifold_thickness.device)

        self.graph = graph
        self.causal_structure = causal_structure
        self.topological_order.copy_(torch.LongTensor(topological_order))
        self._mask_values.copy_(mask_values)

        self._compute_ancestors()

    def _compute_ancestors(self):
        # Construct ancestor index dict
        ancestor_idx = defaultdict(list)
        descendants = set(range(self.dim_z))
        for i in self.topological_order:
            i = i.item()
            descendants.remove(i)
            for j in descendants:
                ancestor_idx[j].append(i)

        self.ancestor_idx = ancestor_idx

    def load_state_dict(self, state_dict, strict=True):
        """Overloading the state dict loading so we can compute ancestor structure"""
        super().load_state_dict(state_dict, strict)
        self._compute_ancestors()


class MLPImplicitSCM(ImplicitSCM):
    """MLP-based implementation of ILCMs"""

    def __init__(
        self,
        graph_parameterization,
        manifold_thickness,
        dim_z,
        hidden_layers=1,
        hidden_units=100,
        base_density=DEFAULT_BASE_DENSITY,
        homoskedastic=True,
        min_std=None,
    ):
        solution_functions = []

        # Initialize graph
        causal_structure = None
        assert graph_parameterization in {
            "enco",
            "dds",
            "fixed_order",
            None,
            "none",
            "none_fixed_order",
            "none_trivial",
        }
        if graph_parameterization == "enco":
            graph = ENCOLearnedGraph(dim_z)
        elif graph_parameterization == "dds":
            graph = DDSLearnedGraph(dim_z)
        elif graph_parameterization == "fixed_order":
            graph = FixedOrderLearnedGraph(dim_z)
        elif graph_parameterization == "none_fixed_order":
            graph = None
            causal_structure = "fixed_order"
        elif graph_parameterization == "none_trivial":
            graph = None
            causal_structure = "trivial"
        else:
            graph = None
            causal_structure = "none"

        # Initialize transforms for p(e'|e)
        for _ in range(dim_z):
            solution_functions.append(
                make_mlp_structure_transform(
                    dim_z,
                    hidden_layers,
                    hidden_units,
                    homoskedastic,
                    min_std=min_std,
                    initialization="broad",
                )
            )

        super().__init__(
            graph,
            solution_functions,
            base_density,
            manifold_thickness,
            dim_z=dim_z,
            causal_structure=causal_structure,
        )
