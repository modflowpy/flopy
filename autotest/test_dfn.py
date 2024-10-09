import pytest

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.codegen.dfn import Dfn
from flopy.mf6.utils.codegen.make import DfnName

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
def test_load_dfn(dfn_name):
    dfn_path = DFN_PATH / f"{dfn_name}.dfn"
    with open(dfn_path, "r") as f:
        dfn = Dfn.load(f, name=DfnName(*dfn_name.split("-")))
