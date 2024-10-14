import pytest

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.codegen.context import Context
from flopy.mf6.utils.codegen.dfn import Dfn
from flopy.mf6.utils.codegen.make import make_all, make_targets

PROJ_ROOT = get_project_root_path()
MF6_PATH = PROJ_ROOT / "flopy" / "mf6"
TGT_PATH = MF6_PATH / "modflow"
DFN_PATH = MF6_PATH / "data" / "dfn"
DFN_NAMES = [
    dfn.stem
    for dfn in DFN_PATH.glob("*.dfn")
    if dfn.stem not in ["common", "flopy"]
]


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_dfn_load(dfn_name):
    dfn_path = DFN_PATH / f"{dfn_name}.dfn"

    common_path = DFN_PATH / "common.dfn"
    with open(common_path, "r") as f:
        common, _ = Dfn._load(f)

    with open(dfn_path, "r") as f:
        dfn = Dfn.load(f, name=Dfn.Name(*dfn_name.split("-")), common=common)
        if dfn_name in ["sln-ems", "exg-gwfprt", "exg-gwfgwe", "exg-gwfgwt"]:
            assert not any(dfn)
        else:
            assert any(dfn)


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_make_targets(dfn_name, function_tmpdir):
    common_path = DFN_PATH / "common.dfn"
    with open(common_path, "r") as f:
        common, _ = Dfn._load(f)

    with open(DFN_PATH / f"{dfn_name}.dfn", "r") as f:
        dfn = Dfn.load(f, name=Dfn.Name(*dfn_name.split("-")), common=common)

    make_targets(dfn, function_tmpdir, verbose=True)

    for name in Context.Name.from_dfn(dfn):
        source_path = function_tmpdir / name.target
        assert source_path.is_file()


def test_make_all(function_tmpdir):
    make_all(DFN_PATH, function_tmpdir, verbose=True)
    assert any(function_tmpdir.glob("*.py"))
