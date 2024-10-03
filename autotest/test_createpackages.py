from ast import AST, Assign, ClassDef, expr
from ast import parse as parse_ast
from itertools import zip_longest
from pprint import pformat
from shutil import copy, copytree
from typing import List, Union
from warnings import warn

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
    t1 = type(node1)
    t2 = type(node2)
    if t1 is not t2:
        print(f"type mismatch: {t1} != {t2}")
        return False

    if t1 is ClassDef:
        assert t2 is ClassDef
        assert node1.name == node2.name
        for base1, base2 in zip(node1.bases, node2.bases):
            def _id(b):
                attrs = ["id", "name", "attr"]
                for attr in attrs:
                    try:
                        return getattr(b, attr)
                    except:
                        pass
                return None
            assert _id(base1) == _id(base2)

        body1, body2 = node1.body, node2.body
        assert len(body1) == len(body2), f"body mismatch in {node1.name}"

        for b1, b2 in zip(body1, body2):
            if isinstance(b1, Assign):
                assert isinstance(b2, Assign)
                b1tgts = set(sorted([t.id for t in b1.targets]))
                b2tgts = set(sorted([t.id for t in b2.targets]))
                diff = b1tgts ^ b2tgts
                if any(diff):
                    warn(
                        f"assignment targets don't match in {node1.name}\n"
                        f"=> symmetric difference:\n{pformat(diff)}\n"
                        f"=> prev - test:\n{pformat(b1tgts - b2tgts)}\n"
                        f"=> test - prev:\n{pformat(b2tgts - b1tgts)}\n"
                    )


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
        prev_classes = [n for n in prev.body if isinstance(n, ClassDef)]
        test_classes = [n for n in test.body if isinstance(n, ClassDef)]
        prev_clsnames = set([c.name for c in prev_classes])
        test_clsnames = set([c.name for c in test_classes])
        diff = prev_clsnames ^ test_clsnames
        assert not any(diff), (
            f"previous classes don't match test classes in {test_file.name}\n"
            f"=> symmetric difference:\n{pformat(diff)}\n"
            f"=> prev - test:\n{pformat(prev_clsnames - test_clsnames)}\n"
            f"=> test - prev:\n{pformat(test_clsnames - prev_clsnames)}\n"
        )
        for prev_cls, test_cls in zip(prev_classes, test_classes):
            compare_ast(prev_cls, test_cls)
