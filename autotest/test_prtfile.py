from pathlib import Path

import pytest

from autotest.conftest import get_project_root_path
from flopy.utils.prtfile import PathlineFile

pytestmark = pytest.mark.mf6
prt_data_path = (
    get_project_root_path() / "examples" / "data" / "mf6" / "prt_data" / "001"
)


@pytest.mark.parametrize(
    "path, header_path",
    [
        (Path(prt_data_path) / "prt001.trk", None),
        (
            Path(prt_data_path) / "prt001.trk",
            Path(prt_data_path) / "prt001.trk.hdr",
        ),
        (Path(prt_data_path) / "prt001.trk.csv", None),
    ],
)
def test_init(path, header_path):
    file = PathlineFile(path, header_filename=header_path)
    assert file.fname == path
    assert file.dtype == PathlineFile.dtypes["full"]
    if path.suffix == ".csv":
        assert len(file._data) == len(open(path).readlines()) - 1


@pytest.mark.parametrize(
    "path",
    [
        Path(prt_data_path) / "prt001.trk",
        Path(prt_data_path) / "prt001.trk.csv",
    ],
)
def test_intersect(path):
    file = PathlineFile(path)
    nodes = [1, 11, 21]
    intersection = file.intersect(nodes)
    assert any(intersection)
    assert all(d.icell in nodes for d in intersection.itertuples())
