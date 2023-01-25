import pytest
from autotest.conftest import get_example_data_path

from flopy.utils.mfreadnam import get_entries_from_namefile

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
