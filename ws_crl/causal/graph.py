# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ws_crl.gumbel import sample_permutation, gumbel_bernouilli
from ws_crl.utils import upper_triangularize, topological_sort


class LearnedGraph(nn.Module):
    """Graph base class."""

    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z

    @property
    def adjacency_matrix(self):
        """
        Read-only property that returns the soft adjacency matrix, with entries in [0, 1] signaling
        the probability
        of an edge existing
        """

        raise NotImplementedError

    @property
    def num_edges(self):
        """Read-only property that returns the sum of edge probabilities"""

        return torch.sum(self.adjacency_matrix)

    @property
    def acyclicity_regularizer(self):
        """Read-only property that returns the sum of edge probabilities"""

        adj = self.adjacency_matrix
        return (torch.trace(torch.matrix_exp(adj)) - adj.shape[0]) ** 2

    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        raise NotImplementedError

    def descendant_masks(self, adjacency_matrix, intervention, eps=1.0e-9):
        """
        Given adjacency matrices (compatible with the assumed causal ordering) and intervention
        mask, this returns a mask that selects the descendants of the intervention targets.

        Arguments:
        ----------
        adjacency_matrix: torch.Tensor of spape (..., self.dim_z, self.dim_z) and dtype torch.float
            Adjacency matrices.
        intervention: torch.Tensor of shape (..., self.dim_z) and dtype torch.bool
            Intervention masks (True for each variable that is intervened upon)

        Returns:
        --------
        descendant_mask: torch.Tensor of shape (..., self.dim_z) and dtype torch.float
            Descendant mask (1 for each variable that is part of the intervention targets or a
            descendant of one of the intervention targets)
        """

        return 1.0 - self.non_descendant_mask(adjacency_matrix, intervention, eps=eps)

    def non_descendant_mask(self, adjacency_matrix, intervention, eps=1.0e-9):
        """
        Given adjacency matrices and intervention masks, returns masks that selects the
        non-descendants of the intervention targets.

        Arguments:
        ----------
        adjacency_matrix: torch.Tensor of spape (..., self.dim_z, self.dim_z) and dtype torch.float
            Adjacency matrices.
        intervention: torch.Tensor of shape (..., self.dim_z) and dtype torch.bool
            Intervention masks (True for each variable that is intervened upon)

        Returns:
        --------
        non_descendant_mask: torch.Tensor of shape (..., self.dim_z) and dtype torch.float
            Non-descendant mask (1 for each variable that is *not* part of the intervention targets
            and *not* a descendant of any of the intervention targets)
        """

        # Check input
        assert adjacency_matrix.shape[-2:] == (self.dim_z, self.dim_z)
        assert intervention.shape[-1] == self.dim_z

        # Non-descendancy matrix
        nd = self._nondescendancy_matrix(adjacency_matrix)

        # Descendance from interventions?
        # The probability of *not* descending from a set of intervention targets is the *product* of
        # the probabilities of descending from each element in the intervention target set
        # We compute this in log space
        intervention_nd = torch.exp(
            torch.sum(intervention.to(torch.float).unsqueeze(-1) * torch.log(nd + eps), dim=-2)
        )

        # Dealing with small errors from eps
        zeros_with_gradients = intervention_nd - intervention_nd.detach()  # STE
        intervention_nd = torch.where(intervention_nd >= eps, intervention_nd, zeros_with_gradients)

        return intervention_nd

    @torch.no_grad()
    def freeze(self):
        """
        Freezes the adjacency matrix such that the edge probabilities do not receive gradients
        anymore
        """

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze(self):
        """
        Unfreezes the adjacency matrix such that the edge probabilities receive gradients again
        """

        for param in self.parameters():
            param.requires_grad = True

    def get_graph_parameters(self):
        """Get graph parameters for logging purposes."""
        raise NotImplementedError

    def _nondescendancy_matrix(self, adjacency_matrix):
        """Computes the non-descendancy matrix"""
        # Let's compute the non-descendancy matrix: the probability of j not being a descendant of i
        # The idea is that this has to be gradient-friendly, soft adjacency-friendly way.
        # non_desc = (1 - identity) * (1 - adj) * (1 - adj^2) * (1 - adj^3) * ... * (1 - adj^(n-1))
        nondescendancy_matrix = torch.ones_like(adjacency_matrix)
        for n in range(0, self.dim_z):
            nondescendancy_matrix *= 1.0 - torch.linalg.matrix_power(adjacency_matrix, n)

        return nondescendancy_matrix


class ENCOLearnedGraph(LearnedGraph):
    """
    Learnable graph with an ENCO-like edge parameterization.
    """

    def __init__(self, dim_z):
        super().__init__(dim_z)

        # Default order (does not necessarily correspond to any topological order here and is really
        # only included for legacy reasons)
        self.order = list(range(self.dim_z))

        # Edge parameterization, consisting of existence logits and orientation logits
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_existence_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self.edge_orientation_logits = torch.nn.Parameter(torch.empty(self.n_edges))

        # Initialize edge weights
        self._initialize_edges()

    @property
    def edge_existence_matrix(self):
        """
        Read-only property that returns the soft edge-existence matrix, with entries in [0, 1]
        signaling the probability of an edge existing (independent of the orientation)
        """

        edge_probs = torch.sigmoid(self.edge_existence_logits)
        edge_matrix = upper_triangularize(edge_probs, self.dim_z)
        edge_matrix = edge_matrix + edge_matrix.T

        return edge_matrix

    @property
    def edge_orientation_matrix(self):
        """
        Read-only property that returns the soft edge-orientation matrix, with entries in [0, 1]
        signaling the probability of an edge being oriented in a certain way (independent of the
        existence)
        """

        orientation_probs = torch.sigmoid(self.edge_orientation_logits)
        orientation_matrix = (
            upper_triangularize(orientation_probs, self.dim_z)
            + upper_triangularize(1.0 - orientation_probs, self.dim_z).T
        )

        return orientation_matrix

    @property
    def adjacency_matrix(self):
        """
        Read-only property that returns the soft adjacency matrix, with entries in [0, 1] signaling
        the probability of an edge existing"""

        return self.edge_existence_matrix * self.edge_orientation_matrix

    @property
    def hard_adjacency_matrix(self):
        """
        Returns an adjacency matrix with binary entries, A_ij exists if the edge existence prob.
        is >= 0.5 and the orientation i -> j is more likely than j -> i
        """

        return ((self.edge_existence_matrix >= 0.5) * (self.edge_orientation_matrix >= 0.5)).to(
            torch.float
        )

    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        """
        Samples adjacency matrices using the hard adjacency matrix, Gumbel-Softmax, or
        Gumbel-Softmax STE
        """

        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"

        # Determinstic sampling
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices = self.sample_adjacency_matrices(n, "soft", temperature)
            return hard_adjacency_matrices.detach() + (
                soft_adjacency_matrices - soft_adjacency_matrices.detach()
            )

        # Sample edge existence
        existence_logits = self.edge_existence_logits.unsqueeze(0).broadcast_to((n, self.n_edges))
        edge_existence, log_prob_existence = gumbel_bernouilli(
            existence_logits, tau=temperature, hard=mode == "hard"
        )

        # Sample edge orientation
        orientation_logits = self.edge_orientation_logits.unsqueeze(0).broadcast_to(
            (n, self.n_edges)
        )
        edge_orientation, log_prob_orientation = gumbel_bernouilli(
            orientation_logits, tau=temperature, hard=mode == "hard"
        )

        # Package as square nice adjacency matrices
        edge_matrix = upper_triangularize(edge_existence, self.dim_z)
        edge_matrix = edge_matrix + torch.transpose(edge_matrix, -2, -1)
        orientation_matrix = upper_triangularize(edge_orientation, self.dim_z) + torch.transpose(
            upper_triangularize(1.0 - edge_orientation, self.dim_z), -2, -1
        )
        adjacency_matrix = edge_matrix * orientation_matrix

        return adjacency_matrix

    @torch.no_grad()
    def get_graph_parameters(self):
        parameters = {}
        for i, val in enumerate(self.edge_existence_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()
        for i, val in enumerate(self.edge_orientation_logits):
            parameters[f"edge_orientation_logit_{i}"] = val.cpu().detach()

        return parameters

    @torch.no_grad()
    def _initialize_edges(self):
        torch.nn.init.normal_(
            self.edge_existence_logits, mean=4, std=0.01
        )  # Initially, all edges are around 98% likely
        torch.nn.init.normal_(
            self.edge_orientation_logits, mean=0, std=0.01
        )  # Edge orientations equally likely


class DDSLearnedGraph(LearnedGraph):
    """
    Learnable graph based on Gumbel-Top-K sampling of a permutation matrix, like in Differentiable
    DAG Sampling (ICLR 2022).
    """

    def __init__(self, dim_z):
        super().__init__(dim_z)

        # Default order (does not necessarily correspond to any topological order here and is really
        # only included for legacy reasons)
        self.order = list(range(self.dim_z))

        # Edge parameterization, consisting of existence logits and orientation logits
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_existence_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self.permutation_logits = torch.nn.Parameter(torch.empty(self.dim_z))

        # Initialize edge weights
        self._initialize_edges()

    @property
    def standard_adjacency_matrix(self):
        """
        Read-only property that returns the soft adjacency matrix in default topological order
        (without permutations)
        """

        edge_probs = torch.sigmoid(self.edge_existence_logits)
        edge_matrix = upper_triangularize(edge_probs, self.dim_z)

        return edge_matrix

    @property
    def permutation_matrix(self):
        """Read-only property that returns the most likely permutation matrix"""

        return sample_permutation(
            self.permutation_logits.unsqueeze(0), mode="deterministic"
        ).squeeze(0)

    @property
    def adjacency_matrix(self):
        """
        Read-only property that returns the soft adjacency matrix, with entries in [0, 1] signaling
        the probability of an edge existing
        """

        permutation = self.permutation_matrix
        adjacency_matrix = permutation.T @ self.standard_adjacency_matrix @ permutation

        return adjacency_matrix

    @property
    def num_edges(self):
        """Read-only property that returns the sum of edge probabilities"""

        return torch.sum(self.standard_adjacency_matrix)

    @property
    def acyclicity_regularizer(self):
        """Read-only property that returns the sum of edge probabilities"""

        return 0.0

    @property
    def hard_adjacency_matrix(self):
        """
        Returns an adjacency matrix with binary entries, A_ij exists if the edge existence prob. is
        >= 0.5 and the orientation i -> j is more likely than j -> i
        """

        standard_adjacency_matrix = (self.standard_adjacency_matrix >= 0.5).to(torch.float)
        permutation = self.permutation_matrix
        adjacency_matrix = permutation.T @ standard_adjacency_matrix @ permutation

        return adjacency_matrix

    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        """
        Samples adjacency matrices using the hard adjacency matrix, Gumbel-Softmax, or
        Gumbel-Softmax STE
        """

        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"

        # Determinstic sampling
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices = self.sample_adjacency_matrices(n, "soft", temperature)
            return hard_adjacency_matrices.detach() + (
                soft_adjacency_matrices - soft_adjacency_matrices.detach()
            )

        # Sample edge existence
        existence_logits = (
            self.edge_existence_logits.unsqueeze(0).broadcast_to((n, self.n_edges)).unsqueeze(2)
        )
        existence_logits = torch.cat((existence_logits, torch.zeros_like(existence_logits)), dim=2)
        edge_existence = F.gumbel_softmax(existence_logits, tau=temperature, hard=mode == "hard")[
            ..., 0
        ]

        # Sample permutation matrix
        # Note that we fix the temperature here and always use hard sampling, following the
        # "Differentiable DAG sampling" paper from ICLR 2022
        scores = self.permutation_logits.unsqueeze(0).expand(n, self.dim_z)
        permutation = sample_permutation(scores, tau=1.0, mode="hard")

        # Package as square nice adjacency matrices
        standard_adjacency_matrix = upper_triangularize(edge_existence, self.dim_z)
        adjacency_matrix = (
            torch.transpose(permutation, 1, 2) @ standard_adjacency_matrix @ permutation
        )

        return adjacency_matrix

    @torch.no_grad()
    def get_graph_parameters(self):
        """Get graph parameters for logging purposes."""
        parameters = {}
        for i, val in enumerate(self.edge_existence_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()
        for i, val in enumerate(self.permutation_logits):
            parameters[f"permutation_score_{i}"] = val.cpu().detach()

        return parameters

    @torch.no_grad()
    def _initialize_edges(self):
        """Edge probability initialization."""
        torch.nn.init.normal_(
            self.edge_existence_logits, mean=4, std=0.01
        )  # All edges around 98% likely
        torch.nn.init.normal_(
            self.permutation_logits, mean=0, std=0.01
        )  # Equally likely permutations


class FixedOrderLearnedGraph(LearnedGraph):
    """
    Learnable acyclic graph.

    Assumes a fixed topological order [1, ..., n] and allows for any edges compatible with this
    order to exist.

    Arguments:
    ----------
    adjacency_matrix: torch.Tensor of shape (self.dim_z, self.dim_z) and dtype torch.bool
        Adjacency matrix. adjacency_matrix[i,j] == 1 means that i -> j.
    """

    def __init__(self, dim_z):
        super().__init__(dim_z)

        # Topological order
        self.order = list(range(self.dim_z))

        # Edge weights
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self._initialize_edges()

    @property
    def adjacency_matrix(self):
        """
        Read-only property that returns the soft adjacency matrix, with entries in [0, 1] signaling
        the probability of an edge existing"""

        edge_probs = torch.sigmoid(self.edge_logits)
        adjacency_matrix = upper_triangularize(edge_probs, self.dim_z)

        return adjacency_matrix

    @property
    def hard_adjacency_matrix(self):
        """
        Returns an adjacency matrix with binary entries, A_ij exists if the edge probability is
        >= 0.5
        """

        return (self.adjacency_matrix >= 0.5).to(torch.float)

    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        """Samples adjacency matrices with Gumbel-Softmax (STE)"""

        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"

        # Determinstic sampling (at zero temperature) for zero temperature
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices, log_prob = self.sample_adjacency_matrices(
                n, "soft", temperature
            )
            det_adjacency_matrices = (
                hard_adjacency_matrices.detach()
                + soft_adjacency_matrices
                - soft_adjacency_matrices.detach()
            )
            return det_adjacency_matrices, log_prob

        # Sample edges through GS
        logits = self.edge_logits.unsqueeze(0).broadcast_to((n, self.n_edges))
        edges, log_prob = gumbel_bernouilli(logits, tau=temperature, hard=mode == "hard")

        # Package as square nice adjacency matrices
        adjacency_matrices = upper_triangularize(edges, self.dim_z)
        log_prob = torch.sum(log_prob, dim=1)

        return adjacency_matrices, log_prob

    @torch.no_grad()
    def get_graph_parameters(self):
        """Get graph parameters for logging purposes."""

        parameters = {}
        for i, val in enumerate(self.edge_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()

        return parameters

    @property
    def acyclicity_regularizer(self):
        """Read-only property that returns the sum of edge probabilities"""

        return 0.0

    @torch.no_grad()
    def _initialize_edges(self):
        """Edge probability initialization."""
        torch.nn.init.normal_(
            self.edge_logits, mean=4, std=0.01
        )  # Initially, all edges are around 98% likely


class FixedGraph(nn.Module):
    """
    Graph representation. Essentially a wrapper around a (discrete) adjacency matrix.

    Arguments:
    ----------
    adjacency_matrix: torch.Tensor of shape (self.dim_z, self.dim_z) and dtype torch.bool
        Adjacency matrix. adjacency_matrix[i,j] == 1 means that i -> j.
    """

    def __init__(self, adjacency_matrix, dim_z=2, epsilon=1.0e-9):
        super().__init__()

        self.dim_z = dim_z

        self.register_buffer("adjacency_matrix", adjacency_matrix.to(torch.bool))
        self.order = topological_sort(self.adjacency_matrix)

        # Check that adjacency matrix has correct shape and represents a DAG (through NOTEARS)
        assert self.adjacency_matrix.shape == (self.dim_z, self.dim_z)
        assert (
            torch.abs(
                torch.trace(torch.matrix_exp(self.adjacency_matrix.to(torch.float))) - self.dim_z
            )
            < epsilon
        ), "Adjacency matrix not acyclical"

    @property
    def hard_adjacency_matrix(self):
        """
        Returns an adjacency matrix with binary entries, A_ij exists if the edge probability is
        >= 0.5
        """

        return self.adjacency_matrix

    @property
    def num_edges(self):
        """Read-only property that returns the sum of edge probabilities"""

        return torch.sum(self.adjacency_matrix)

    @property
    def acyclicity_regularizer(self):
        """Read-only property that returns the sum of edge probabilities"""

        return 0.0

    def sample_adjacency_matrices(self, n, **kwargs):
        """Adjacency matrix sampling. Here only for code compatibility."""

        return self.adjacency_matrix.unsqueeze(0).expand((n, *self.adjacency_matrix.shape)), None

    def descendant_masks(self, intervention, epsilon=1.0e-9, **kwargs):
        """
        Given an intervention mask, returns a mask that selects the descendants of the intervention
        targets (batched).

        Arguments:
        ----------
        intervention: torch.Tensor of shape (..., self.dim_z) and dtype torch.bool
            Intervention mask (True for each variable that is intervened upon)
        epsilon: float, optional
            Small constant for floating-point comparisons. Default value 1.e-9

        Returns:
        --------
        descendant_mask: torch.Tensor of shape (..., self.dim_z,) and dtype torch.bool
            Descendant mask (True for each variable that is part of the intervention targets or a
            descendant of one of the intervention targets)
        """

        # Check input
        assert intervention.shape[-1] == self.dim_z

        descendants = torch.matrix_exp(
            self.adjacency_matrix.to(torch.float)
        )  # descendants[i,j] says whether j descended from i
        descended_from_intervention = torch.einsum(
            "...i,ij->...j", intervention.to(torch.float), descendants
        )  # (batchsize, dim_z)
        descendant_mask = torch.abs(descended_from_intervention) > epsilon

        return descendant_mask

    @torch.no_grad()
    def freeze(self):
        """Freezes graph."""
        pass

    @torch.no_grad()
    def unfreeze(self):
        """Unfreezes graph."""
        pass

    @torch.no_grad()
    def get_graph_parameters(self):
        """Graph parameters for logging purposes."""
        return {}

    def non_descendant_mask(self, intervention, epsilon=1.0e-9, **kwargs):
        """
        Given an intervention mask, returns a mask that selects the non-descendants of the
        intervention targets.

        Arguments:
        ----------
        intervention: torch.Tensor of shape (self.dim_z,) or (batchsize, self.dim_z) and dtype
                      torch.bool
            Intervention mask (True for each variable that is intervened upon)
        epsilon: float, optional
            Small constant for floating-point comparisons. Default value 1.e-9

        Returns:
        --------
        non_descendant_mask: torch.Tensor of shape (self.dim_z,) or (batchsize, self.dim_z)
                             dtype torch.bool
            Non-descendant mask (True for each variable that is *not* part of the intervention
            targets and *not* a descendant of any of the intervention targets)
        """

        return ~self.descendant_masks(intervention, epsilon)

    def parent_masks_from_intervention_masks(self, intervention, **kwargs):
        """
        Returns masks that selects any parent of any intervention target (but not the intervention
        targets themselves) (batched)
        """

        assert intervention.shape[-1] == self.dim_z

        parents = torch.einsum(
            "ij,...j->...i", self.adjacency_matrix.to(torch.float), intervention.to(torch.float)
        )
        parents -= intervention * self.dim_z  # Exclude intervention targets themselves
        parent_masks = parents > 0

        return parent_masks

    def parent_masks(self, indices, **kwargs):
        """
        Returns masks that select the parents of nodes specified by indices (batched)
        """

        return self.adjacency_matrix.T[indices]


def bools_from_seed(seed, n_bools):
    """Given an integer seed, produces a series of Bernouilli (p = 0.5) bools."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bools).astype(bool)


def create_graph(dim_z, mode, edge_existence_seed, permutation=None):
    """Creates a fixed graph for a given dimension"""

    # Permutations are not yet implemented! Annoying to enumerate them in high dims...
    if permutation:
        raise NotImplementedError

    # Adjacency matrix in topological order
    n_edges = dim_z * (dim_z - 1) // 2
    if mode == "full":
        edges = torch.ones((n_edges,), dtype=torch.bool)
        adjacency_matrix = upper_triangularize(edges, dim_z)
    elif mode == "empty":
        adjacency_matrix = torch.zeros((dim_z, dim_z), dtype=torch.bool)
    elif mode == "chain":
        adjacency_matrix = torch.diag_embed(torch.ones((dim_z - 1,), dtype=torch.bool), offset=1)
    elif mode == "random":
        edges = torch.BoolTensor(bools_from_seed(edge_existence_seed, n_edges))
        adjacency_matrix = upper_triangularize(edges, dim_z)
    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    # Graph object
    graph = FixedGraph(adjacency_matrix, dim_z=dim_z)

    return graph
