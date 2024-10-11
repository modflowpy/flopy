import pytest

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.codegen.context import get_context_names
from flopy.mf6.utils.codegen.dfn import Dfn
from flopy.mf6.utils.codegen.make import (
    DfnName,
    make_all,
    make_targets,
)

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
    with open(dfn_path, "r") as f:
        dfn = Dfn.load(f, name=DfnName(*dfn_name.split("-")))
        if dfn_name in ["sln-ems", "exg-gwfprt", "exg-gwfgwe", "exg-gwfgwt"]:
            assert not any(dfn)
        else:
            assert any(dfn)


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_make_targets(dfn_name, function_tmpdir):
    with open(DFN_PATH / "common.dfn") as f:
        common = Dfn.load(f)

    with open(DFN_PATH / f"{dfn_name}.dfn", "r") as f:
        dfn_name = DfnName(*dfn_name.split("-"))
        dfn = Dfn.load(f, name=dfn_name)

    make_targets(dfn, function_tmpdir, commonvars=common)
    for ctx_name in get_context_names(dfn_name):
        source_path = function_tmpdir / ctx_name.target
        assert source_path.is_file()


def test_make_all(function_tmpdir):
    make_all(DFN_PATH, function_tmpdir, verbose=True)
    assert any(function_tmpdir.glob("*.py"))
