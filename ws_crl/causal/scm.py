# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Graphs and structural causal models """

import nflows.distributions
import nflows.nn.nets
import torch
from torch import nn

from ws_crl import transforms
from ws_crl.causal.graph import (
    ENCOLearnedGraph,
    DDSLearnedGraph,
    FixedOrderLearnedGraph,
)
from ws_crl.nets import Quadratic
from ws_crl.transforms import make_intervention_transform, make_mlp_structure_transform
from ws_crl.utils import topological_sort, mask, clean_and_clamp

DEFAULT_BASE_DENSITY = nflows.distributions.StandardNormal((1,))


class FixedOrderSCM(nn.Module):
    """
    Structural causal model with a learnable graph.

    We follow the convention that each causal variable has exactly one associated noise variable
    and fix its distribution to a standard normal.

    In addition to the structure functions and the graph structure, this class also contains an
    interventional distribution for each causal variable. These are parameterized through functions
    from a base noise distribution (again Uniform(0,1)) to the causal variable, similar to the
    structure functions.

    Parameters:
    -----------
    graph: Graph
        Causal graph.
    structure_transforms: list of self.dim_z Transforms
        The i-th element in this list is the structure function for the i-th causal variable,
        `z_i = f_i(epsilon_i, inputs)`. It is implemented as a flow-like transform of a single noise
        variable conditional on all latent variables inputs (the non-parents will be masked out =
        fixed to zero).
    intervention_transforms: list of self.dim_z Transforms
        The i-th element in this list parameterizes the interventional distribution for the i-th
        causal variable, such that under each intervention that targets (a superset of) this
        variable we have `inputs'_i = g_i(epsilon'_i)` with `epsilon'_i ~ N(0,1)` (or whatever
        `base_density` is).
    """

    def __init__(
        self,
        graph,
        structure_transforms,
        intervention_transforms,
        base_density,
        manifold_thickness,
        dim_z,
        concatenate_mask_in_context=True,
    ):
        super().__init__()
        self.dim_z = dim_z

        self.graph = graph
        self.structure_transforms = torch.nn.ModuleList(structure_transforms)
        self.intervention_transforms = torch.nn.ModuleList(intervention_transforms)
        self.base_density = base_density
        self._concat_mask = concatenate_mask_in_context
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))

    def sample(self, n, intervention=None, graph_mode="hard", graph_temperature=1.0):
        """Samples a single latent vector, either observed or under an intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        # Sample graphs
        adjacency_matrices, _ = self.graph.sample_adjacency_matrices(
            n, mode=graph_mode, temperature=graph_temperature
        )

        # Sample noise
        noise = self._sample_noise(n)
        intervention_noise = self._sample_noise(n)

        # Push through solution function
        z, _ = self._solve(
            noise,
            intervention_noise=intervention_noise,
            intervention=intervention,
            adjacency_matrix=adjacency_matrices,
        )

        return z

    def sample_weakly_supervised(self, n, intervention, graph_mode="hard", graph_temperature=1.0):
        """Samples in the weakly supervised setting for a given intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        # Sample graphs
        adjacency_matrices, _ = self.graph.sample_adjacency_matrices(
            n, mode=graph_mode, temperature=graph_temperature
        )

        # Sample noise
        noise = self._sample_noise(n)  # noise variables used for the data pre intervention
        intervention_noise = self._sample_noise(n)  # noise used for the intervened-upon variables
        cf_noise = self._sample_noise(n, True)  # noise used for the non-intervened-upon variables

        # Push through solution function
        z1, _ = self._solve(noise, adjacency_matrix=adjacency_matrices)
        z2, _ = self._solve(
            noise,
            intervention=intervention,
            intervention_noise=intervention_noise,
            cf_noise=cf_noise,
            adjacency_matrix=adjacency_matrices,
        )

        return z1, z2

    def log_prob_weakly_supervised(self, z1, z2, intervention, adjacency_matrix):
        """
        Given weakly supervised causal variables and the intervention mask, computes the
        corresponding noise variables
        and log likelihoods.
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, z1.shape[0])
        assert torch.all(torch.isfinite(z1))
        assert torch.all(torch.isfinite(z2))

        # Observed data point
        observation_noise, logdet1 = self._inverse(z1, adjacency_matrix=adjacency_matrix)
        assert torch.all(torch.isfinite(observation_noise))
        logprob_observed = self.base_density.log_prob(observation_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed += logdet1  # (batchsize, self.dim_z)
        logprob_observed = clean_and_clamp(logprob_observed)
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)  # (batchsize, 1)

        # After intervention: density for intervened-upon variables
        intervention_noise, logdet2 = self._inverse(
            z2, intervention=intervention, adjacency_matrix=adjacency_matrix
        )
        assert torch.all(torch.isfinite(intervention_noise))
        logprob_intervened = self.base_density.log_prob(
            intervention_noise.reshape((-1, 1))
        ).reshape((-1, self.dim_z))
        logprob_intervened += logdet2
        logprob_intervened = clean_and_clamp(logprob_intervened)
        logprob_intervened = intervention * logprob_intervened  # (batchsize, self.dim_z)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)  # (batchsize, 1)

        # Counterfactual discrepancy
        z_counterfactual = self._solve_counterfactual(
            z2,
            intervention=intervention,
            noise=observation_noise,
            adjacency_matrix=adjacency_matrix,
        )
        assert torch.all(torch.isfinite(z_counterfactual))
        cf_noise = (z2 - z_counterfactual) / self.manifold_thickness
        assert torch.all(torch.isfinite(cf_noise))
        logdet_cf = -torch.log(self.manifold_thickness)
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened += logdet_cf
        logprob_nonintervened = clean_and_clamp(logprob_nonintervened)
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened  # (batchsize, self.dim_z)
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)  # (batchsize, 1)

        # Package outputs
        logprob = logprob_observed + logprob_intervened + logprob_nonintervened
        assert torch.all(torch.isfinite(logprob))
        outputs = dict(
            scm_log_prior=logprob,
            scm_log_prior_observed=logprob_observed,
            scm_log_prior_intervened=logprob_intervened,
            scm_log_prior_nonintervened=logprob_nonintervened,
        )

        return logprob, outputs

    def log_prob_noise_weakly_supervised(self, epsilon1, epsilon2, intervention, adjacency_matrix):
        """
        Given weakly supervised as noise encodings epsilon1, epsilon2 and the intervention mask,
        computes the corresponding causal variables and log likelihoods.
        """

        raise NotImplementedError

    def noise_to_causal(self, epsilon, adjacency_matrix):
        """Given noise encoding, returns causal encoding"""
        return self._solve(epsilon, adjacency_matrix=adjacency_matrix)[0]

    def causal_to_noise(self, z, adjacency_matrix):
        """Given causal latents, returns noise encoding"""
        return self._inverse(z, adjacency_matrix=adjacency_matrix)[0]

    @property
    def manifold_thickness(self):
        """
        Returns "thickness" (standard deviation of corresponding likelihood) of counterfactual
        manifold
        """
        return self._manifold_thickness

    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        """
        Sets "thickness" (standard deviation of corresponding likelihood) of counterfactual
        manifold
        """
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))

    @torch.no_grad()
    def get_scm_parameters(self):
        """Gets SCM parameters for logging purposes"""
        # Manifold thickness
        parameters = {"manifold_thickness": self.manifold_thickness}

        # Adjacency matrix: edge probabilities
        adj = self.graph.adjacency_matrix.cpu().detach()
        for i in range(self.dim_z):
            for j in range(self.dim_z):
                if i != j:
                    parameters[f"adjacency_matrix_{i}_{j}"] = adj[i, j]

        # Internal parameters
        parameters.update(self.graph.get_graph_parameters())

        return parameters

    def generate_similar_intervention(
        self, z1, z2_example, intervention, adjacency_matrix, sharp_manifold=True
    ):
        """Infers intervention from data and "imitates" it"""
        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, z1.shape[0])

        # To noise variables
        observation_noise, _ = self._inverse(z1, adjacency_matrix=adjacency_matrix)
        intervention_noise, _ = self._inverse(
            z2_example, intervention=intervention, adjacency_matrix=adjacency_matrix
        )
        z_counterfactual = self._solve_counterfactual(
            z2_example, intervention=intervention, z=z1, adjacency_matrix=adjacency_matrix
        )

        # Set counterfactual noise to zero (sharp manifold)
        if sharp_manifold:
            return z_counterfactual

        cf_noise = (z2_example - z_counterfactual) / self.manifold_thickness
        z2, _ = self._solve(
            noise=observation_noise,
            intervention=intervention,
            intervention_noise=intervention_noise,
            cf_noise=cf_noise,
            adjacency_matrix=adjacency_matrix,
        )

        return z2

    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)

        return intervention

    def _solve(
        self,
        noise,
        adjacency_matrix,
        intervention=None,
        intervention_noise=None,
        cf_noise=None,
        order=None,
    ):
        """
        Given SCM noise variables (and optionally an intervention target and intervention noise),
        computes the causal variables inputs.
        """

        # Order of causal variables
        if order is None:
            order = self.graph.order

        # Prepare output
        n = noise.shape[0]
        z = torch.zeros((n, self.dim_z), device=noise.device)
        logdet = torch.zeros((n, self.dim_z), device=z.device)

        # We need to treat the components as separate tensors, otherwise autograd has issues
        z_columns = [column.unsqueeze(1) for column in z.T]
        logdet_columns = [column.unsqueeze(1) for column in logdet.T]

        # Go through graph in topological order
        for i in order:
            # Interventions
            if intervention is not None:
                z_, logdet_ = self.intervention_transforms[i](
                    intervention_noise[:, i : i + 1], context=None
                )
                z_columns[i] = intervention[:, i : i + 1] * z_
                logdet_columns[i] = intervention[:, i : i + 1] * logdet_

            # Concatenate list to tensor again
            z = torch.cat(z_columns, dim=1)

            # Non-interventions
            non_intervention_mask = (
                1.0 if intervention is None else (1.0 - intervention[:, i : i + 1])
            )
            parent_mask = adjacency_matrix[:, :, i]
            masked_z = mask(z, parent_mask, concat_mask=self._concat_mask)
            z_, logdet_ = self.structure_transforms[i](noise[:, i : i + 1], context=masked_z)
            z_columns[i] = z_columns[i] + non_intervention_mask * z_
            logdet_columns[i] = logdet_columns[i] + non_intervention_mask * logdet_

            # "Counterfactual noise": deviations from exact counterfactual manifold
            if cf_noise is not None:
                z_columns[i] = z_columns[i] + non_intervention_mask * cf_noise[:, i : i + 1]

        # Concatenate list to tensor again
        z = torch.cat(z_columns, dim=1)
        logdet = torch.cat(logdet_columns, dim=1)

        return z, logdet

    def _solve_counterfactual(self, z_query, adjacency_matrix, intervention, z=None, noise=None):
        """
        Given (SCM noise variables or causal variables) and an counterfactual query, computes the
        answer to the counterfactual query.
        """

        # You can call this with either noise or inputs (but not both)
        assert (noise is None) != (
            z is None
        ), "For counterfactual queries, either inputs or noise has to be provided, but not both."
        if noise is None:
            noise, _ = self._inverse(z, adjacency_matrix=adjacency_matrix)

        # Counterfactual z is initialized to the intervention target values given in the query
        z_cf = intervention * z_query.clone()

        # Recompute variables that are not intervention targets in topological order
        for i in self.graph.order:

            # We need to treat the components as separate tensors, otherwise autograd has issues
            z_cf_columns = [column.unsqueeze(1) for column in z_cf.T]

            parent_mask = adjacency_matrix[:, :, i]
            masked_z = mask(z_cf, parent_mask, concat_mask=self._concat_mask)
            z_cf_columns[i] = (
                z_cf_columns[i]
                + (1.0 - intervention[:, i : i + 1])
                * self.structure_transforms[i](noise[:, i : i + 1], context=masked_z)[0]
            )

            # Concatenate list to tensor again
            z_cf = torch.cat(z_cf_columns, dim=1)

        return z_cf

    def _inverse(self, z, adjacency_matrix, intervention=None):
        """
        Given causal variables (and optionally intervention targets), computes the corresponding
        noise variables and the log det of the Jacobian.
        """

        # Prepare output
        n = z.shape[0]
        noise = torch.zeros((n, self.dim_z), device=z.device)
        logdet = torch.zeros((n, self.dim_z), device=z.device)

        # Interventions
        if intervention is not None:
            for i in self.graph.order:
                noise_, logdet_ = self.intervention_transforms[i].inverse(
                    z[:, i : i + 1], context=None
                )
                noise[:, i : i + 1] = intervention[:, i : i + 1] * noise_
                logdet[:, i : i + 1] = intervention[:, i : i + 1] * logdet_

        # Non-interventions
        for i in self.graph.order:
            non_intervention_mask = (
                1.0 if intervention is None else (1.0 - intervention[:, i : i + 1])
            )
            parent_mask = adjacency_matrix[:, :, i]
            masked_z = mask(z, parent_mask, concat_mask=self._concat_mask)

            noise_, logdet_ = self.structure_transforms[i].inverse(
                z[:, i : i + 1], context=masked_z
            )

            noise[:, i : i + 1] += non_intervention_mask * noise_
            logdet[:, i : i + 1] += non_intervention_mask * logdet_

        return noise, logdet

    def _sample_noise(self, n, sample_consistency_noise=False):
        """Samples noise"""
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)


class MLPFixedOrderSCM(FixedOrderSCM):
    """SCM implementation based on a fixed topological order and MLP mechanisms"""

    def __init__(
        self,
        manifold_thickness,
        dim_z,
        hidden_layers=1,
        hidden_units=100,
        base_density=DEFAULT_BASE_DENSITY,
        homoskedastic=True,
        enhance_causal_effects_at_init=False,
        min_std=None,
    ):
        structure_transforms = []
        intervention_transforms = []

        # Initialize graph
        graph = FixedOrderLearnedGraph(dim_z)

        for _ in range(dim_z):
            # Initialize structural assignments
            structure_transforms.append(
                make_mlp_structure_transform(
                    dim_z,
                    hidden_layers,
                    hidden_units,
                    homoskedastic,
                    min_std,
                    initialization="strong_effects"
                    if enhance_causal_effects_at_init
                    else "default",
                )
            )

            # Initialize interventional distributions
            intervention_transforms.append(
                make_intervention_transform(
                    homoskedastic, enhance_causal_effects_at_init, min_std=min_std
                )
            )

        super().__init__(
            graph,
            structure_transforms,
            intervention_transforms,
            base_density,
            manifold_thickness,
            dim_z=dim_z,
        )


class VariableOrderSCM(FixedOrderSCM):
    """
    Structural causal model with a learnable graph, NOT assuming the same topological order between
    variables.

    We follow the convention that each causal variable has exactly one associated noise variable and
    fix its distribution to a standard normal.

    In addition to the structure functions and the graph structure, this class also contains an
    interventional distribution for each causal variable. These are parameterized through functions
    from a base noise distribution (again Uniform(0,1)) to the causal variable, similar to the
    structure functions.

    Parameters:
    -----------
    graph: Graph
        Causal graph.
    structure_transforms: list of self.dim_z Transforms
        The i-th element in this list is the structure function for the i-th causal variable,
        `z_i = f_i(epsilon_i, inputs)`. It is implemented as a flow-like transform of a single noise
        variable conditional on all latent variables inputs (the non-parents will be masked out =
        fixed to zero).
    intervention_transforms: list of self.dim_z Transforms
        The i-th element in this list parameterizes the interventional distribution for the i-th
        causal variable, such that under each intervention that targets (a superset of) this
        variable we have `inputs'_i = g_i(epsilon'_i)` with `epsilon'_i ~ N(0,1)` (or whatever
        `base_density` is).
    """

    def _solve(
        self,
        noise,
        adjacency_matrix,
        intervention=None,
        intervention_noise=None,
        cf_noise=None,
        order=None,
    ):
        """
        Given SCM noise variables (and optionally an intervention target and intervention noise),
        computes the causal variables inputs.
        """

        # Here the topological order is not fixed, and for sampling we need to know it. We solve
        # this by generating samples one by one, finding the topological order for each sample.
        # This is highly inefficient! But it was easy to implement.

        z, logdet = [], []

        for i in range(noise.shape[0]):
            intervention_ = None if intervention is None else intervention[i : i + 1]
            intervention_noise_ = (
                None if intervention_noise is None else intervention_noise[i : i + 1]
            )
            cf_noise_ = None if cf_noise is None else cf_noise[i : i + 1]

            order = topological_sort(adjacency_matrix[i])

            z_, logdet_ = super()._solve(
                noise[i : i + 1],
                adjacency_matrix[i : i + 1],
                intervention_,
                intervention_noise_,
                cf_noise_,
                order=order,
            )

            z.append(z_)
            logdet.append(logdet_)

        z = torch.cat(z, dim=0)
        logdet = torch.cat(logdet, dim=0)

        return z, logdet

    def _solve_counterfactual(self, z_query, adjacency_matrix, intervention, z=None, noise=None):
        """
        Given (SCM noise variables or causal variables) and an counterfactual query, computes the
        answer to the counterfactual query.
        """

        # You can call this with either noise or inputs (but not both)
        assert (noise is None) != (
            z is None
        ), "For counterfactual queries, either inputs or noise has to be provided, but not both."
        if noise is None:
            noise, _ = self._inverse(z, adjacency_matrix=adjacency_matrix)

        # Prepare outputs
        z_cf = torch.empty_like(z_query)

        # Go over causal variables and compute expected value based on noise and parent values after
        # intervention. The order doesn't really matter.
        for i in range(self.dim_z):
            parent_mask = adjacency_matrix[:, :, i]
            masked_z = mask(
                z_query, parent_mask
            )  # z after intervention, non-parents are masked to zero
            cf_value = self.structure_transforms[i](noise[:, i : i + 1], context=masked_z)[
                0
            ].squeeze(1)
            z_cf[:, i] = intervention[:, i] * z_query[:, i] + (1.0 - intervention[:, i]) * cf_value

        return z_cf


class MLPVariableOrderCausalModel(VariableOrderSCM):
    """SCM implementation based on a variable topological order and MLP mechanisms"""

    def __init__(
        self,
        manifold_thickness,
        dim_z,
        graph_parameterization="enco",
        hidden_layers=1,
        hidden_units=100,
        base_density=DEFAULT_BASE_DENSITY,
        homoskedastic=True,
        enhance_causal_effects_at_init=False,
        min_std=None,
    ):
        structure_transforms = []
        intervention_transforms = []

        # Initialize graph
        assert graph_parameterization in {"enco", "dds"}
        if graph_parameterization == "enco":
            graph = ENCOLearnedGraph(dim_z)
        else:
            graph = DDSLearnedGraph(dim_z)

        for _ in range(dim_z):
            # Initialize structural assignments
            structure_transforms.append(
                make_mlp_structure_transform(
                    dim_z,
                    hidden_layers,
                    hidden_units,
                    homoskedastic,
                    min_std,
                    initialization="strong_effects"
                    if enhance_causal_effects_at_init
                    else "default",
                )
            )

            # Initialize interventional distributions
            intervention_transforms.append(
                make_intervention_transform(
                    homoskedastic, enhance_causal_effects_at_init, min_std=min_std
                )
            )

        super().__init__(
            graph,
            structure_transforms,
            intervention_transforms,
            base_density,
            manifold_thickness,
            dim_z=dim_z,
        )


class FixedGraphCausalModel(nn.Module):
    """
    Structural causal model with fixed graph.

    We follow the convention that each causal variable has exactly one associated noise variable and
    fix its distribution to a standard normal.

    In addition to the structure functions and the graph structure, this class also contains an
    interventional distribution for each causal variable. These are parameterized through functions
    from a base noise distribution (again Uniform(0,1)) to the causal variable, similar to the
    structure functions.

    Parameters:
    -----------
    graph: Graph
        Causal graph.
    structure_transforms: list of self.dim_z Transforms
        The i-th element in this list is the structure function for the i-th causal variable,
        z_i = f_i(epsilon_i, inputs). It is implemented as a transform of a single noise variable
        (for which we assume a uniform distribution) conditional on all latent variables inputs
        (the non-parents will be masked out = fixed to zero).
    intervention_transforms: list of self.dim_z Transforms
        The i-th element in this list parameterizes the interventional distribution for the i-th
        causal variable, such that under each intervention that targets (a superset of) this
        variable we have inputs'_i = g_i(epsilon'_i) with epsilon'_i ~ Uniform(0,1).
    """

    def __init__(
        self,
        graph,
        structure_transforms,
        intervention_transforms,
        base_density,
        manifold_thickness,
        dim_z,
    ):
        super().__init__()
        self.dim_z = dim_z

        self.graph = graph
        self.structure_transforms = torch.nn.ModuleList(structure_transforms)
        self.intervention_transforms = torch.nn.ModuleList(intervention_transforms)
        self.base_density = base_density
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))

    def sample(self, n, intervention=None):
        """Samples a single latent vector, either observed or under an intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        noise = self._sample_noise(n)
        intervention_noise = self._sample_noise(n)
        z, _ = self._solve(noise, intervention_noise=intervention_noise, intervention=intervention)

        return z

    def sample_weakly_supervised(self, n, intervention):
        """Samples in the weakly supervised setting for a given intervention"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, n)

        noise = self._sample_noise(n)  # noise variables used for the data pre intervention
        intervention_noise = self._sample_noise(n)  # noise used for the intervened-upon variables
        cf_noise = self._sample_noise(n, True)  # noise used for the non-intervened-upon variables

        z1, _ = self._solve(noise)
        z2, _ = self._solve(noise, intervention, intervention_noise, cf_noise)

        return z1, z2

    def log_prob_weakly_supervised(self, z1, z2, intervention, **kwargs):
        """
        Given weakly supervised causal variables and the intervention mask, computes the
        corresponding noise variables and log likelihoods.
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, z1.shape[0])

        # Observed data point
        observation_noise, logdet1 = self._inverse(z1)
        logprob_observed = self.base_density.log_prob(observation_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed += logdet1  # (batchsize, self.dim_z)
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)  # (batchsize, 1)

        # After intervention: density for intervened-upon variables
        intervention_noise, logdet2 = self._inverse(z2, intervention=intervention)
        logprob_intervened = self.base_density.log_prob(
            intervention_noise.reshape((-1, 1))
        ).reshape((-1, self.dim_z))
        logprob_intervened += logdet2
        logprob_intervened = intervention * logprob_intervened  # (batchsize, self.dim_z)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)  # (batchsize, 1)

        # Counterfactual discrepancy
        z_counterfactual = self._solve_counterfactual(
            z2, intervention=intervention, noise=observation_noise
        )
        cf_noise = (z2 - z_counterfactual) / self.manifold_thickness
        logdet_cf = -torch.log(self.manifold_thickness)
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened += logdet_cf
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened  # (batchsize, self.dim_z)
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)  # (batchsize, 1)

        # Package outputs
        logprob = logprob_observed + logprob_intervened + logprob_nonintervened
        outputs = dict(
            scm_log_prob_z=logprob,
            scm_log_prob_z_observed=logprob_observed,
            scm_log_prob_z_intervened=logprob_intervened,
            scm_log_prob_z_nonintervened=logprob_nonintervened,
            scm_obs_noise=observation_noise,
            scm_intervention_noise=intervention_noise,
            scm_cf_noise=cf_noise,
            scm_interventions=intervention,
        )

        return logprob, outputs

    def log_prob_noise_weakly_supervised(self, epsilon1, epsilon2, intervention):
        """
        Given weakly supervised as noise encodings epsilon1, epsilon2 and the intervention mask,
        computes the corresponding causal variables and log likelihoods.
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, epsilon1.shape[0])

        # Transform noise encoding to generative noise variables
        observation_noise = epsilon1  # Without interventions: noise encoding = generative noise var
        z2, logdet2_1 = self._solve(epsilon2)
        intervention_noise, logdet2_2 = self._inverse(z2, intervention=intervention)

        # Observed likelihood
        logprob_observed = self.base_density.log_prob(observation_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)  # (batchsize, 1)

        # Intervention likelihood
        logprob_intervened = self.base_density.log_prob(
            intervention_noise.reshape((-1, 1))
        ).reshape((-1, self.dim_z))
        logprob_intervened += logdet2_2 + logdet2_1
        logprob_intervened = intervention * logprob_intervened  # (batchsize, self.dim_z)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)  # (batchsize, 1)

        # Counterfactual discrepancy for not-intervened-upon variables
        z_counterfactual = self._solve_counterfactual(
            z2, intervention=intervention, noise=observation_noise
        )
        cf_noise = (z2 - z_counterfactual) / self.manifold_thickness
        logdet_cf = -torch.log(self.manifold_thickness)
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened += logdet2_1 + logdet_cf
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened  # (batchsize, self.dim_z)
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)  # (batchsize, 1)

        # Package outputs
        logprob = logprob_observed + logprob_intervened + logprob_nonintervened
        outputs = dict(
            scm_log_prob_z=logprob,
            scm_log_prob_z_observed=logprob_observed,
            scm_log_prob_z_intervened=logprob_intervened,
            scm_log_prob_z_nonintervened=logprob_nonintervened,
            scm_obs_noise=observation_noise,
            scm_intervention_noise=intervention_noise,
            scm_cf_noise=cf_noise,
            scm_interventions=intervention,
        )

        return logprob, outputs

    def noise_to_causal(self, epsilon):
        """Given noise encoding, returns causal encoding"""
        return self._solve(epsilon)[0]

    def causal_to_noise(self, z):
        """Given causal latents, returns noise encoding"""
        return self._inverse(z)[0]

    @property
    def manifold_thickness(self):
        """Returns the counterfactual manifold thickness"""
        return self._manifold_thickness

    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        """Sets the counterfactual manifold thickness"""
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))

    def get_scm_parameters(self):
        """Returns key parameters for logging purposes"""
        return {"manifold_thickness": self.manifold_thickness}

    def generate_similar_intervention(self, z1, z2_example, intervention, sharp_manifold=True):
        """Infers intervention from data and "imitates" it"""

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, z1.shape[0])

        # To noise variables
        observation_noise, _ = self._inverse(z1)
        intervention_noise, _ = self._inverse(z2_example, intervention=intervention)
        z_counterfactual = self._solve_counterfactual(z2_example, intervention=intervention, z=z1)

        # Set counterfactual noise to zero (sharp manifold)
        if sharp_manifold:
            return z_counterfactual

        cf_noise = (z2_example - z_counterfactual) / self.manifold_thickness
        z2, _ = self._solve(observation_noise, intervention, intervention_noise, cf_noise)

        return z2

    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)

        return intervention

    def _solve(self, noise, intervention=None, intervention_noise=None, cf_noise=None):
        """
        Given SCM noise variables (and optionally an intervention target and intervention noise),
        computes the causal variables inputs.
        """

        # Prepare output
        n = noise.shape[0]
        z = torch.zeros((n, self.dim_z), device=noise.device)
        logdet = torch.zeros((n, self.dim_z), device=z.device)

        # Interventions
        if intervention is not None:
            for i in self.graph.order:
                z_, logdet_ = self.intervention_transforms[i](
                    intervention_noise[:, i : i + 1], context=None
                )
                z[:, i : i + 1] = intervention[:, i : i + 1] * z_
                logdet[:, i : i + 1] = intervention[:, i : i + 1] * logdet_

        # Non-interventions
        for i in self.graph.order:
            non_intervention_mask = (
                1.0 if intervention is None else (1.0 - intervention[:, i : i + 1])
            )
            masked_z = self.graph.parent_masks(i).unsqueeze(0) * z

            z_, logdet_ = self.structure_transforms[i](noise[:, i : i + 1], context=masked_z)
            z[:, i : i + 1] += non_intervention_mask * z_
            logdet[:, i : i + 1] += non_intervention_mask * logdet_

        if cf_noise is not None:
            non_intervention_mask = 1.0 if intervention is None else (1.0 - intervention[:])
            z += non_intervention_mask * cf_noise

        return z, logdet

    def _solve_counterfactual(self, z_query, intervention, z=None, noise=None):
        """
        Given (SCM noise variables or causal variables) and an counterfactual query, computes the
        answer to the counterfactual query.
        """

        # You can call this with either noise or inputs (but not both)
        assert (noise is None) != (
            z is None
        ), "For counterfactual queries, either inputs or noise has to be provided, but not both."
        if noise is None:
            noise, _ = self._inverse(z)

        # Counterfactual z is initialized to the intervention target values given in the query
        z_cf = torch.zeros_like(z_query) + intervention * z_query

        # Non-interventions
        for i in self.graph.order:
            masked_z = self.graph.parent_masks(i).unsqueeze(0) * z_cf
            z_cf[:, i : i + 1] += (1.0 - intervention[:, i : i + 1]) * self.structure_transforms[i](
                noise[:, i : i + 1], context=masked_z
            )[0]

        return z_cf

    def _inverse(self, z, intervention=None):
        """
        Given causal variables (and optionally intervention targets), computes the corresponding
        noise variables and the log det of the Jacobian.
        """

        # Prepare output
        n = z.shape[0]
        noise = torch.zeros((n, self.dim_z), device=z.device)
        logdet = torch.zeros((n, self.dim_z), device=z.device)

        # Interventions
        if intervention is not None:
            for i in self.graph.order:
                noise_, logdet_ = self.intervention_transforms[i].inverse(
                    z[:, i : i + 1], context=None
                )
                noise[:, i : i + 1] = intervention[:, i : i + 1] * noise_
                logdet[:, i : i + 1] = intervention[:, i : i + 1] * logdet_

        # Non-interventions
        for i in self.graph.order:
            non_intervention_mask = (
                1.0 if intervention is None else (1.0 - intervention[:, i : i + 1])
            )
            masked_z = self.graph.parent_masks(i).unsqueeze(0) * z

            noise_, logdet_ = self.structure_transforms[i].inverse(
                z[:, i : i + 1], context=masked_z
            )
            noise[:, i : i + 1] += non_intervention_mask * noise_
            logdet[:, i : i + 1] += non_intervention_mask * logdet_

        return noise, logdet

    def _sample_noise(self, n, sample_consistency_noise=False):
        """Samples noise"""
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)


class FixedGraphLinearANM(FixedOrderSCM):
    """SCM implementation for a fixed graph and a linear additive-noise model"""

    def __init__(
        self,
        graph,
        dim_z,
        manifold_thickness=1.0e-9,
        base_density=DEFAULT_BASE_DENSITY,
        initialization="standard",
    ):
        # Check inputs
        assert initialization in ["standard", "bimodal"]

        # Set up structure functions
        structure_transforms = []
        intervention_transforms = []
        for _ in range(dim_z):
            structure_transforms.append(
                transforms.ConditionalAffineScalarTransform(
                    param_net=torch.nn.Linear(dim_z, 1), features=1, conditional_std=False
                )
            )
            intervention_transforms.append(
                transforms.ConditionalAffineScalarTransform(param_net=None, features=1)
            )

        super().__init__(
            graph,
            structure_transforms,
            intervention_transforms,
            base_density,
            manifold_thickness,
            dim_z,
            concatenate_mask_in_context=False,
        )

        # Initialize causal effects
        self._canonical_init(initialization)

    @torch.no_grad()
    def _canonical_init(self, initialization="standard"):
        for trf in self.structure_transforms:
            # Std of each causal variable is 1
            trf.log_scale.copy_(torch.zeros(1))

            # Parent-independent mean of each causal variable is 0
            trf.param_net.bias.copy_(torch.zeros(1))

            # Sample causal effects from standard Gaussian
            if initialization == "standard":
                trf.param_net.weight.copy_(torch.randn((1, self.dim_z)))
            elif initialization == "bimodal":
                bimodal = torch.sign(torch.randn((1, self.dim_z))) * (
                    1.0 + 0.3 * torch.randn((1, self.dim_z))
                )
                trf.param_net.weight.copy_(bimodal)
            else:
                raise ValueError(f"Unknown initialization {initialization}")

        # Intervention distribution has std 1 and mean 0
        for trf in self.intervention_transforms:
            trf.shift.copy_(torch.zeros(1))
            trf.log_scale.copy_(torch.zeros(1))

    def get_scm_parameters(self):
        """Returns key parameters for logging purposes"""
        return {"manifold_thickness": self.manifold_thickness}


class UnstructuredPrior(nn.Module):
    """
    Unstructured prior, with uncorrelated uniform Gaussian densities on z and z', as a drop-in
    replacement for the SCM class.
    """

    def __init__(
        self, base_density=DEFAULT_BASE_DENSITY, dim_z=2, manifold_thickness=0.1, **kwargs
    ):
        super().__init__()
        self.dim_z = dim_z
        self.base_density = base_density
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))
        self.graph = None

    def sample(self, n, **kwargs):
        """Samples a single latent vector, either observed or under an intervention"""

        return self._sample_noise(n)

    def sample_weakly_supervised(self, n, **kwargs):
        """Samples in the weakly supervised setting for a given intervention"""

        z1 = self._sample_noise(n)
        z2 = self._sample_noise(n)

        return z1, z2

    def log_prob_weakly_supervised(self, z1, z2, intervention, **kwargs):
        """
        Given weakly supervised samples and the intervention mask, computes the corresponding noise
        variables and log likelihoods
        """

        # Sanitize inputs
        intervention = self._sanitize_intervention(intervention, z1.shape[0])

        # Observed data point
        logprob_observed = self.base_density.log_prob(z1.reshape((-1, 1))).reshape((-1, self.dim_z))
        logprob_observed = torch.sum(logprob_observed, 1)

        # After intervention: density for intervened-upon variables
        logprob_intervened = self.base_density.log_prob(z2.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_intervened = intervention * logprob_intervened
        logprob_intervened = torch.sum(logprob_intervened, 1)

        # Counterfactual discrepancy
        logprob_nonintervened = self.base_density.log_prob(z2.reshape((-1, 1))).reshape(
            -1, self.dim_z
        )
        logprob_nonintervened = (1.0 - intervention) * logprob_nonintervened
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1)

        # Package outputs
        logprob = logprob_observed + logprob_intervened + logprob_nonintervened
        outputs = dict()

        return logprob, outputs

    def log_prob_noise_weakly_supervised(self, epsilon1, epsilon2, intervention, **kwargs):
        """
        Given weakly supervised as noise encodings epsilon1, epsilon2 and the intervention mask,
        computes the corresponding causal variables and log likelihoods
        """

        return self.log_prob_weakly_supervised(epsilon1, epsilon2, intervention=intervention)

    def noise_to_causal(self, epsilon, **kwargs):
        """Given noise encoding, returns causal encoding"""
        return epsilon

    def causal_to_noise(self, z, **kwargs):
        """Given causal latents, returns noise encoding"""
        return z

    @property
    def manifold_thickness(self):
        """Returns thickness of counterfactual manifold"""
        return self._manifold_thickness

    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        """Sets thickness of counterfactual manifold"""
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))

    def get_scm_parameters(self):
        """Gets key parameters (for logging purposes)"""
        return {"manifold_thickness": self.manifold_thickness}

    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)

        return intervention

    def _sample_noise(self, n, sample_consistency_noise=False):
        """Samples noise"""
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)
