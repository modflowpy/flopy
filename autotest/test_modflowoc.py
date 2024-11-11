import pytest

from flopy.modflow import Modflow, ModflowDis, ModflowOc


def test_modflowoc_load(example_data_path):
    model = Modflow()
    mpath = example_data_path / "mf2005_test"
    ModflowDis.load(mpath / "fhb.dis", model, check=False)
    ModflowOc.load(mpath / "fhb.oc", model, ext_unit_dict=None)


@pytest.mark.parametrize(
    "nper, nstp, nlay", [(3, None, 0), (3, None, 1), (0, None, 1), (3, 1, 1)]
)
def test_modflowoc_load_fails_when_wrong_nlay_nper_nstp(
    nper, nstp, nlay, example_data_path
):
    model = Modflow()
    mpath = example_data_path / "mf2005_test"
    # noinspection PyTypeChecker
    with pytest.raises((ValueError, OSError)):
        ModflowOc.load(mpath / "fhb.oc", model, nper=nper, nstp=nstp, nlay=nlay)
