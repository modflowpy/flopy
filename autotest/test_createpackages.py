import pytest
from modflow_devtools.misc import run_cmd

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.createpackages import (
    DfnName,
    load_dfn,
    make_all,
    make_context,
    make_targets,
)

PROJ_ROOT = get_project_root_path()
DFN_PATH = PROJ_ROOT / "flopy" / "mf6" / "data" / "dfn"
DFN_NAMES = [
    dfn.stem
    for dfn in DFN_PATH.glob("*.dfn")
    if dfn.stem not in ["common", "flopy"]
]


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_load_dfn(dfn_name):
    dfn_path = DFN_PATH / f"{dfn_name}.dfn"
    with open(dfn_path, "r") as f:
        dfn = load_dfn(f, name=DfnName(*dfn_name.split("-")))


@pytest.mark.parametrize(
    "dfn_name, n_flat, n_params", [("gwf-ic", 2, 6), ("prt-prp", 40, 22)]
)
def test_make_context(dfn_name, n_flat, n_params):
    with open(DFN_PATH / "common.dfn") as f:
        common = load_dfn(f)

    with open(DFN_PATH / f"{dfn_name}.dfn") as f:
        dfn_name = DfnName(*dfn_name.split("-"))
        dfn = load_dfn(f, name=dfn_name)

    ctx_name = dfn_name.contexts[0]
    context = make_context(ctx_name, dfn, common=common)
    assert len(dfn_name.contexts) == 1
    assert len(context.variables) == n_params
    assert len(context.metadata) == n_flat + 1  # +1 for metadata


@pytest.mark.parametrize("dfn_name", DFN_NAMES)
def test_make_targets(dfn_name, function_tmpdir):
    with open(DFN_PATH / "common.dfn") as f:
        common = load_dfn(f)

    with open(DFN_PATH / f"{dfn_name}.dfn", "r") as f:
        dfn_name = DfnName(*dfn_name.split("-"))
        dfn = load_dfn(f, name=dfn_name)

    make_targets(dfn, function_tmpdir, common=common)
    for ctx_name in dfn_name.contexts:
        run_cmd("ruff", "format", function_tmpdir, verbose=True)
        run_cmd("ruff", "check", "--fix", function_tmpdir, verbose=True)
        assert (function_tmpdir / ctx_name.target).is_file()


def test_make_all(function_tmpdir):
    make_all(DFN_PATH, function_tmpdir, verbose=True)
    run_cmd("ruff", "format", function_tmpdir, verbose=True)
    run_cmd("ruff", "check", "--fix", function_tmpdir, verbose=True)
