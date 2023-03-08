# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Utility functions """

import collections
import itertools
import numpy as np
import torch


def get_first_batch(dataloader):
    """Returns the first batch from a PyTorch DataLoader"""

    for data in dataloader:
        return data

    raise RuntimeError("Cannot get first batch from DataLoader")


def logmeanexp(x, dim):
    """Like logsumexp, but using a mean instead of the sum"""
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])


def inverse_softplus(x, beta=1.0):
    """Inverse of the softplus function"""
    return 1 / beta * np.log(np.exp(beta * x) - 1.0)


def update_dict(storage, new_entries):
    """Helper function to update dicts of lists"""
    for key, val in new_entries.items():
        storage[key].append(val)


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a hierarchical dict into a simple, non-nested dict"""

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def upper_triangularize(entries, size):
    """
    Given a vector of n * (n-1) / 2 components, constructs an upper triangular matrix with those
    components as entries (and 0 elsewhere)
    """

    batch_dims = entries.shape[:-1]
    a = torch.zeros(*batch_dims, size, size, device=entries.device, dtype=entries.dtype)
    i, j = torch.triu_indices(size, size, offset=1)
    a[..., i, j] = entries

    return a


def generate_permutation(elements, permutation, inverse=False):
    """Generates a permutation tensor of `elements` elements."""

    idx = list(range(elements))
    permutations = tuple(itertools.permutations(idx))

    permutation_tensor = torch.LongTensor(permutations[permutation])
    if inverse:
        permutation_tensor = torch.LongTensor(np.argsort(permutations[permutation]))

    return permutation_tensor


def topological_sort(adjacency_matrix):
    """Topological sort"""

    # Make a copy of the adjacency matrix
    a = adjacency_matrix.clone().to(torch.int)
    dim = a.shape[0]
    assert a.shape == (dim, dim)

    # Kahn's algorithm
    ordering = []
    to_do = list(range(dim))

    while to_do:
        root_nodes = [i for i in to_do if not torch.sum(torch.abs(a[:, i]))]
        for i in root_nodes:
            ordering.append(i)
            del to_do[to_do.index(i)]
            a[i, :] = 0

    return ordering


def to_tuple(array):
    """Transforms a tensor into a nested tuple"""
    try:
        return tuple(to_tuple(component) for component in array)
    except TypeError:
        return array


def mask(data, mask_, mask_data=None, concat_mask=True):
    """Masking on a tensor, optionally adding the mask to the data"""

    if mask_data is None:
        masked_data = mask_ * data
    else:
        masked_data = mask_ * data + (1 - mask_) * mask_data

    if concat_mask:
        masked_data = torch.cat((masked_data, mask_), dim=1)

    return masked_data


def clean_and_clamp(inputs, min_=-1.0e12, max_=1.0e12):
    """Clamps a tensor and replaces NaNs"""
    return torch.clamp(torch.nan_to_num(inputs), min_, max_)
