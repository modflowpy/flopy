import pytest

from autotest.conftest import get_example_data_path
from flopy.utils.mfreadnam import (
    attribs_from_namfile_header,
    get_entries_from_namefile,
)

_example_data_path = get_example_data_path()


@pytest.mark.parametrize(
    "path",
    [
        _example_data_path / "mf6" / "test001a_Tharmonic" / "mfsim.nam",
        _example_data_path / "mf6" / "test001e_UZF_3lay" / "mfsim.nam",
        _example_data_path / "mf6-freyberg" / "mfsim.nam",
    ],
)
def test_get_entries_from_namefile_mf6(path):
    package = "IMS6"
    entries = get_entries_from_namefile(path, ftype=package)
    assert len(entries) == 1

    entry = entries[0]
    assert path.parent.name in entry[0]
    assert entry[1] == package


@pytest.mark.skip(reason="only supports mf6 namefiles")
@pytest.mark.parametrize(
    "path",
    [
        _example_data_path / "mf6-freyberg" / "freyberg.nam",
    ],
)
def test_get_entries_from_namefile_mf2005(path):
    package = "IC6"
    entries = get_entries_from_namefile(path, ftype=package)
    assert len(entries) == 1

    entry = entries[0]
    assert path.parent.name in entry[0]
    assert entry[1] == package


@pytest.mark.parametrize(
    "path,expected",
    [
        pytest.param(
            None,
            {
                "crs": None,
                "rotation": 0.0,
                "xll": None,
                "xul": None,
                "yll": None,
                "yul": None,
            },
            id="None",
        ),
        pytest.param(
            _example_data_path / "freyberg" / "freyberg.nam",
            {
                "crs": None,
                "rotation": 0.0,
                "xll": None,
                "xul": None,
                "yll": None,
                "yul": None,
            },
            id="freyberg",
        ),
        pytest.param(
            _example_data_path
            / "freyberg_multilayer_transient"
            / "freyberg.nam",
            {
                "crs": "+proj=utm +zone=14 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
                "rotation": 15.0,
                "start_datetime": "1/1/2015",
                "xll": None,
                "xul": 619653.0,
                "yll": None,
                "yul": 3353277.0,
            },
            id="freyberg_multilayer_transient",
        ),
        pytest.param(
            _example_data_path
            / "mt3d_test"
            / "mfnwt_mt3dusgs"
            / "sft_crnkNic"
            / "CrnkNic.nam",
            {
                "crs": "EPSG:26916",
                "rotation": 0.0,
                "start_datetime": "1-1-1970",
                "xll": None,
                "xul": 0.0,
                "yll": None,
                "yul": 15.0,
            },
            id="CrnkNic",
        ),
    ],
)
def test_attribs_from_namfile_header(path, expected):
    assert attribs_from_namfile_header(path) == expected
