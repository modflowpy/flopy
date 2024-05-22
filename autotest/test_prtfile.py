import pytest

from autotest.conftest import get_project_root_path
from flopy.utils.prtfile import PathlineFile

pytestmark = pytest.mark.mf6
proj_root = get_project_root_path()
prt_data_path = (
    proj_root / "examples" / "data" / "mf6" / "prt_data" / "001"
)


@pytest.mark.parametrize(
    "path, header_path",
    [
        (
            prt_data_path / "prt001.trk",
            prt_data_path / "prt001.trk.hdr",
        ),
        (prt_data_path / "prt001.trk.csv", None),
    ],
)
def test_init(path, header_path):
    file = PathlineFile(path, header_filename=header_path)
    assert file.fname == path
    if path.suffix == ".csv":
        assert len(file._data) == len(open(path).readlines()) - 1


@pytest.mark.parametrize(
    "path, header_path",
    [
        (
            prt_data_path / "prt001.trk",
            prt_data_path / "prt001.trk.hdr",
        ),
        (prt_data_path / "prt001.trk.csv", None),
    ],
)
def test_intersect(path, header_path):
    file = PathlineFile(path, header_filename=header_path)
    nodes = [1, 11, 21]
    intersection = file.intersect(nodes)
    assert any(intersection)
    assert all(d.icell in nodes for d in intersection.itertuples())


@pytest.mark.parametrize(
    "path, header_path",
    [
        (
            prt_data_path / "prt001.trk",
            prt_data_path / "prt001.trk.hdr",
        ),
        (prt_data_path / "prt001.trk.csv", None),
    ],
)
def test_validate(path, header_path):
    file = PathlineFile(path, header_filename=header_path)
    file.validate()
