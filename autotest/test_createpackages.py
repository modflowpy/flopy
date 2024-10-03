from ast import AST, expr
from ast import parse as parse_ast
from itertools import zip_longest
from pprint import pformat
from shutil import copy, copytree
from typing import List, Union

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


def compare_ast(
    node1: Union[expr, List[expr]], node2: Union[expr, List[expr]]
) -> bool:
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, AST):
        for k, v in vars(node1).items():
            if k in {
                "lineno",
                "end_lineno",
                "col_offset",
                "end_col_offset",
                "ctx",
            }:
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True

    elif isinstance(node1, list) and isinstance(node2, list):
        return all(compare_ast(n1, n2) for n1, n2 in zip_longest(node1, node2))
    else:
        return node1 == node2


def test_equivalence(function_tmpdir):
    prev_dir = function_tmpdir / "prev"
    test_dir = function_tmpdir / "test"
    test_dir.mkdir()
    copytree(TGT_PATH, prev_dir)
    make_all(DFN_PATH, test_dir, verbose=True)
    prev_files = list(prev_dir.glob("*.py"))
    test_files = list(test_dir.glob("*.py"))
    prev_names = set([p.name for p in prev_files])
    test_names = set([p.name for p in test_files])
    diff = prev_names ^ test_names
    assert not any(diff), (
        f"previous files don't match test files\n"
        f"=> symmetric difference:\n{pformat(diff)}\n"
        f"=> prev - test:\n{pformat(prev_names - test_names)}\n"
        f"=> test - prev:\n{pformat(test_names - prev_names)}\n"
    )
    for prev_file, test_file in zip(prev_files, test_files):
        prev = parse_ast(open(prev_file).read())
        test = parse_ast(open(test_file).read())
