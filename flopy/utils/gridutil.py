"""
Grid utilities
"""
from typing import Iterable, List, Tuple, Union

import numpy as np


def get_lni(
    ncpl: Union[int, Iterable[int]], *nodes
) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Get the layer index and within-layer node index (both 0-based)
    given node count per layer and node number (grid-scoped index).
    if no nodes are specified, all are returned in ascending order.

    Parameters
    ----------
    ncpl: node count per layer (int or list of ints)
    nodes : node numbers (zero or more ints)

    Returns
    -------
        A tuple (layer index, node index), or a
        list of such if multiple nodes provided
    """

    v = []
    counts = [ncpl] if isinstance(ncpl, int) else list(ncpl)

    for nn in nodes if nodes else range(sum(ncpl)):
        csum = np.cumsum([0] + counts)
        layer = max(0, np.searchsorted(csum, nn) - 1)
        nidx = nn - sum([counts[l] for l in range(0, layer)])

        # np.searchsorted assigns the first index of each layer
        # to the previous layer in layer 2+, so correct for it
        correct = layer + 1 < len(csum) and nidx == counts[layer]
        v.append((layer + 1, 0) if correct else (layer, nidx))

    return v if len(v) > 1 else v[0]
