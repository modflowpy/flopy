import pytest
from modflow_devtools.dfn import get_dfns
from modflow_devtools.dfn2toml import convert

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.codegen import make_all

PROJ_ROOT = get_project_root_path()
MF6_PATH = PROJ_ROOT / "flopy" / "mf6"
DFN_PATH = PROJ_ROOT / "autotest" / "temp" / "dfn"
TOML_PATH = DFN_PATH / "toml"
MF6_OWNER = "MODFLOW-USGS"
MF6_REPO = "modflow6"
MF6_REF = "develop"


def pytest_generate_tests(metafunc):
    if not any(DFN_PATH.glob("*.dfn")):
        get_dfns(MF6_OWNER, MF6_REPO, MF6_REF, DFN_PATH, verbose=True)

    convert(DFN_PATH, TOML_PATH)
    dfns = list(DFN_PATH.glob("*.dfn"))
    assert all(
        (TOML_PATH / f"{dfn.stem}.toml").is_file()
        for dfn in dfns
        if "common" not in dfn.stem
    )


@pytest.mark.parametrize("version,dfn_path", [(1, DFN_PATH), (2, TOML_PATH)])
def test_make_all(function_tmpdir, version, dfn_path):
    make_all(dfn_path, function_tmpdir, verbose=True, version=version)
    assert any(function_tmpdir.glob("*.py"))
