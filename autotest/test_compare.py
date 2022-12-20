import numpy as np
import pytest

from flopy.mf6.utils import MfGrdFile
from flopy.utils.compare import _diffmax, _difftol


def test_diffmax():
    a1 = np.array([1, 2, 3])
    a2 = np.array([4, 5, 7])
    d, indices = _diffmax(a1, a2)
    indices = indices[
        0
    ]  # return value is a tuple of arrays (1 for each dimension)
    assert d == 4
    assert list(indices) == [2]


def test_difftol():
    a1 = np.array([1, 2, 3])
    a2 = np.array([3, 5, 7])
    d, indices = _difftol(a1, a2, 2.5)
    indices = indices[
        0
    ]  # return value is a tuple of arrays (1 for each dimension)
    assert d == 4
    print(d, indices)
    assert list(indices) == [1, 2]


@pytest.mark.skip(reason="todo")
def test_eval_bud_diff(example_data_path):
    # get ia from grdfile
    mfgrd_test_path = example_data_path / "mfgrd_test"
    grb_path = mfgrd_test_path / "nwtp3.dis.grb"
    grb = MfGrdFile(str(grb_path), verbose=True)
    ia = grb._datadict["IA"] - 1

    # TODO: create/run minimal model, then compare budget files


@pytest.mark.skip(reason="todo")
def test_compare_budget():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_swrbudget():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_heads():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_concs():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_stages():
    pass


@pytest.mark.skip(reason="todo")
def test_compare():
    pass
