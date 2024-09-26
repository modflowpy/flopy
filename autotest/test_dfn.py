import pytest

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.dfn import load_dfn

PROJ_ROOT = get_project_root_path()
DFNS_PATH = PROJ_ROOT / "flopy" / "mf6" / "data" / "dfn"
DFNS = [dfn.name for dfn in DFNS_PATH.glob("*.dfn")]


@pytest.mark.parametrize("dfn", DFNS)
def test_load_dfn(dfn):
    dfn_path = DFNS_PATH / dfn
    with open(dfn_path, "r") as f:
        vars, meta = load_dfn(f)
