import pytest

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.codegen import make_all, make_targets
from flopy.mf6.utils.codegen.context import Context
from flopy.mf6.utils.codegen.dfn import Dfn
from flopy.mf6.utils.codegen.filters import Filters

PROJ_ROOT = get_project_root_path()
MF6_PATH = PROJ_ROOT / "flopy" / "mf6"
DFN_PATH = MF6_PATH / "data" / "dfn"
DFN_NAMES = [
    dfn.stem
    for dfn in DFN_PATH.glob("*.dfn")
    if dfn.stem not in ["common", "flopy"]
]


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_dfn_load(dfn_name):
    with (
        open(DFN_PATH / "common.dfn", "r") as common_file,
        open(DFN_PATH / f"{dfn_name}.dfn", "r") as dfn_file,
    ):
        name = Dfn.Name.parse(dfn_name)
        common, _ = Dfn._load_v1_flat(common_file)
        Dfn.load(dfn_file, name=name, common=common)


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_make_targets(dfn_name, function_tmpdir):
    with (
        open(DFN_PATH / "common.dfn", "r") as common_file,
        open(DFN_PATH / f"{dfn_name}.dfn", "r") as dfn_file,
    ):
        name = Dfn.Name.parse(dfn_name)
        common, _ = Dfn._load_v1_flat(common_file)
        dfn = Dfn.load(dfn_file, name=name, common=common)

    make_targets(dfn, function_tmpdir, verbose=True)
    assert all(
        (function_tmpdir / f"mf{Filters.Cls.title(name)}.py").is_file()
        for name in Context.Name.from_dfn(dfn)
    )


def test_make_all(function_tmpdir):
    make_all(DFN_PATH, function_tmpdir, verbose=True)
    assert any(function_tmpdir.glob("*.py"))
