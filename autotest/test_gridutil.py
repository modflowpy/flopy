import pytest

from flopy.utils.gridutil import get_lni

cases = [
    (10, 0, 0, 0),
    ([10, 10], 0, 0, 0),
    ([10, 10], 10, 1, 0),
    ([10, 10], 9, 0, 9),
    ([10, 10], 15, 1, 5),
    ([10, 20], 29, 1, 19),
]


@pytest.mark.parametrize("ncpl, nn, expected_layer, expected_ni", cases)
def test_get_lni_one_node(ncpl, nn, expected_layer, expected_ni):
    actual_layer, actual_i = get_lni(ncpl, nn)
    assert actual_layer == expected_layer
    assert actual_i == expected_ni


@pytest.mark.parametrize("ncpl, nn, expected_layer, expected_ni", cases)
def test_get_lni_multiple_nodes(ncpl, nn, expected_layer, expected_ni):
    # use previous neighbor if last index
    # in a layer, otherwise next neighbor
    t = 1
    if nn == 9 or nn == 29:
        t = -1

    nodes = [nn, nn + t]
    lni = get_lni(ncpl, *nodes)
    assert isinstance(lni, list)
    i = 0
    for (actual_layer, actual_ni) in lni:
        assert actual_layer == expected_layer
        assert actual_ni == expected_ni + (i * t)
        i += 1


@pytest.mark.parametrize("ncpl", [c[0] for c in cases[1:]])
def test_get_lni_no_nodes(ncpl):
    lni = get_lni(ncpl)
    nnodes = sum(ncpl)
    assert len(lni) == nnodes
    for nn in range(nnodes):
        assert lni[nn] == get_lni(ncpl, nn)
