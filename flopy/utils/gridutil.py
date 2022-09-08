"""
Grid utilities
"""
from math import floor
from typing import Collection, Iterable, List, Sequence, Tuple, Union

import numpy as np


def get_lni(ncpl, nodes) -> List[Tuple[int, int]]:
    """
    Get layer index and within-layer node index (both 0-based).

     | Node count per layer may be an int or array-like of integers.
    An integer ncpl indicates all layers have the same node count.
    If an integer ncpl is less than any specified node numbers, the
    grid is understood to have at least enough layers to contain them.

     | If ncpl is array-like it is understood to describe node count
    per zero-indexed layer.

    Parameters
    ----------
    ncpl: node count per layer (int or array-like of ints)
    nodes : node numbers (array-like of nodes)

    Returns
    -------
        A list of tuples (layer index, node index)
    """

    if not isinstance(ncpl, (int, list, tuple, np.ndarray)):
        raise ValueError(f"ncpl must be int or array-like")
    if not isinstance(nodes, (list, tuple, np.ndarray)):
        raise ValueError(f"nodes must be array-like")

    if len(nodes) == 0:
        return []

    if isinstance(ncpl, int):
        # infer min number of layers to hold given node numbers
        layers = range(floor(np.max(nodes) / ncpl) if len(nodes) > 0 else 1)
        counts = [ncpl for _ in layers]
    else:
        counts = list(ncpl)

    tuples = []
    for nn in nodes if nodes else range(sum(counts)):
        csum = np.cumsum([0] + counts)
        layer = max(0, np.searchsorted(csum, nn) - 1)
        nidx = nn - sum([counts[l] for l in range(0, layer)])

        # np.searchsorted assigns the first index of each layer
        # to the previous layer in layer 2+, so correct for it
        correct = layer + 1 < len(csum) and nidx == counts[layer]
        tuples.append((layer + 1, 0) if correct else (layer, nidx))

    return tuples
