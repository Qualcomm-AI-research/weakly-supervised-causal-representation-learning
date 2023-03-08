# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

from functools import lru_cache
import torch
from ws_crl.utils import mask


# noinspection PyUnresolvedReferences
def dependance(
    transform,
    inputs,
    context,
    component,
    invert=False,
    measure=torch.nn.functional.mse_loss,
    normalize=True,
    **kwargs,
):
    """
    Computes a measure of functional dependence of a transform on a given component of the context
    """

    # Shuffle the component of the context
    context_shuffled = context.clone()
    batchsize = context.shape[0]
    idx = torch.randperm(batchsize)
    context_shuffled[:, component] = context_shuffled[idx, component]

    # Compute function with and without permutation
    function = transform.inverse if invert else transform
    f, _ = function(inputs, context=context, **kwargs)
    f_shuffled, _ = function(inputs, context=context_shuffled, **kwargs)

    # Normalize so that this becomes comparable
    if normalize:
        mean, std = torch.mean(f), torch.std(f)
        std = torch.clamp(std, 0.1)
        f = (f - mean) / std
        f_shuffled = (f_shuffled - mean) / std

    # Compute difference
    difference = measure(f, f_shuffled)

    return difference


def solution_dependance_on_noise(model, i, j, noise):
    """Tests whether solution s_i depends on noise variable e_j"""

    transform = model.scm.solution_functions[i]
    inputs = noise[:, i].unsqueeze(1)

    mask_ = torch.ones_like(noise)
    mask_[:, i] = 0
    context = mask(noise, mask_)

    # Note that we need to invert here b/c the transform is defined from z to e
    return dependance(transform, inputs, context, j, invert=True)


def find_topological_order(model, noise):
    """
    Extracts the topological order from a noise-centric model by iteratively looking for the
    least-dependant solution function
    """

    @lru_cache()
    def solution_dependance_on_noise(i, j):
        """Tests how strongly solution s_i depends on noise variable e_j"""

        transform = model.scm.solution_functions[i]
        inputs = noise[:, i].unsqueeze(1)

        mask_ = torch.ones_like(noise)
        mask_[:, i] = 0
        context = mask(noise, mask_)

        # Note that we need to invert here b/c the transform is defined from z to e
        return dependance(transform, inputs, context, j, invert=True)

    topological_order = []
    components = set(range(model.dim_z))

    while components:
        least_dependant_solution = None
        least_dependant_score = float("inf")

        # For each variable, check how strongly its solution function depends on the other noise
        # vars
        for i in components:
            others = [j for j in components if j != i]
            score = sum(solution_dependance_on_noise(i, j) for j in others)

            if score < least_dependant_score:
                least_dependant_solution = i
                least_dependant_score = score

        # The "least dependant" variable will the be next in our topological order, then we remove
        # it and consider only the remaining vars
        topological_order.append(least_dependant_solution)
        components.remove(least_dependant_solution)

    return topological_order


class CausalMechanism(torch.nn.Module):
    """Causal mechanismm extracted from a solution function learned by an ILCM"""

    def __init__(self, solution_transform, component, ancestor_mechanisms):
        super().__init__()

        self.component = component
        self.solution_transform = solution_transform
        self.ancestor_mechanisms = ancestor_mechanisms

    def forward(self, inputs, context, noise, computed_noise=None):
        """Transforms noise (and parent causal variables) to causal variable"""

        solution_context = self._compute_context(inputs, context, noise, computed_noise)

        # Note that the solution transform implements z -> e, here we want forward to mean e -> z
        return self.solution_transform.inverse(inputs, context=solution_context)

    def inverse(self, inputs, context, noise, computed_noise=None):
        """Transforms causal variable (and parent causal variables) to noise"""

        solution_context = self._compute_context(inputs, context, noise, computed_noise)

        # Note that the solution transform implements z -> e, here we want forward to mean e -> z
        return self.solution_transform(inputs, context=solution_context)

    def _compute_context(self, inputs, context, noise, computed_noise=None):
        # Random noise for non-ancestors
        noise = self._randomize_noise(noise)

        # Compute noise encodings corresponding to ancestors
        if computed_noise is None:
            computed_noise = dict()

        for a, mech in self.ancestor_mechanisms.items():
            if a not in computed_noise:
                # print(f'{self.component} -> {a}')
                this_noise, _ = mech.inverse(
                    context[:, a].unsqueeze(1), context, noise, computed_noise=computed_noise
                )
                computed_noise[a] = this_noise.squeeze()

            noise[:, a] = computed_noise[a]

        return noise

    def _randomize_noise(self, noise):
        noise = noise.clone()
        for k in range(noise.shape[1]):
            noise[:, k] = noise[torch.randperm(noise.shape[0]), k]

        return noise


def construct_causal_mechanisms(model, topological_order):
    """Extracts causal mechanisms from model given a topological order"""
    causal_mechanisms = {}

    for i in topological_order:
        solution = model.scm.get_masked_solution_function(i)
        causal_mechanisms[i] = CausalMechanism(
            solution,
            component=i,
            ancestor_mechanisms={a: mech for a, mech in causal_mechanisms.items()},
        )

    return causal_mechanisms


def compute_implicit_causal_effects(model, noise):
    """Tests whether a causal mechanism f_i depends on a particular causal variable z_j"""

    model.eval()

    z = model.scm.noise_to_causal(noise)
    causal_effect = torch.zeros((model.dim_z, model.dim_z))
    # causal_effect[j,i] quantifies how strongly z_j influences z_i

    topological_order = find_topological_order(model, noise)
    mechanisms = construct_causal_mechanisms(model, topological_order)

    for pos, i in enumerate(topological_order):
        for j in topological_order[:pos]:
            causal_effect[j, i] = dependance(
                mechanisms[i], noise[:, i : i + 1], z, j, invert=False, noise=noise
            )

    return causal_effect, topological_order
