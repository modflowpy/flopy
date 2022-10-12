import pytest

from flopy.utils.gridutil import get_lni


@pytest.mark.parametrize(
    "ncpl, nn, expected_layer, expected_ni",
    [
        (10, 0, 0, 0),
        ([10, 10], 0, 0, 0),
        ([10, 10], 10, 1, 0),
        ([10, 10], 9, 0, 9),
        ([10, 10], 15, 1, 5),
        ([10, 20], 29, 1, 19),
    ],
)
def test_get_lni(ncpl, nn, expected_layer, expected_ni):
    # pair with next neighbor unless last in layer,
    # in which case pair with previous neighbor
    t = 1
    if nn == 9 or nn == 29:
        t = -1

    nodes = [nn, nn + t]
    lni = get_lni(ncpl, nodes)
    assert isinstance(lni, list)
    i = 0
    for (actual_layer, actual_ni) in lni:
        assert actual_layer == expected_layer
        assert actual_ni == expected_ni + (i * t)
        i += 1


def test_get_lni_no_nodes():
    lni = get_lni(10, [])
    assert isinstance(lni, list)
    assert len(lni) == 0


@pytest.mark.parametrize(
    "ncpl, nodes, expected",
    [
        (5, [14], [(2, 4)]),
        (10, [14], [(1, 4)]),
        (20, [14], [(0, 14)]),
        (20, [14, 24], [(0, 14), (1, 4)]),
    ],
)
def test_get_lni_infers_layer_count_when_int_ncpl(ncpl, nodes, expected):
    lni = get_lni(ncpl, nodes)
    assert isinstance(lni, list)
    for i, ln in enumerate(lni):
        assert ln == expected[i]
