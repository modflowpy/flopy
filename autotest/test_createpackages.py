import pytest
from modflow_devtools.misc import run_cmd

from autotest.conftest import get_project_root_path
from flopy.mf6.createpackages import (
    TEMPLATE_ENV,
    ContextType,
    DefinitionName,
    generate_targets,
    get_template_context,
    load_dfn,
)
from flopy.mf6.mfpackage import MFPackage

PROJ_ROOT = get_project_root_path()
DFN_PATH = PROJ_ROOT / "flopy" / "mf6" / "data" / "dfn"
DFNS = [
    dfn
    for dfn in DFN_PATH.glob("*.dfn")
    if dfn.stem not in ["common", "flopy"]
]


@pytest.mark.parametrize("dfn", DFNS)
def test_load_dfn(dfn):
    dfn_path = DFN_PATH / dfn
    with open(dfn_path, "r") as f:
        definition = load_dfn(f)


# only test packages for which we know the
# expected number of consolidated variables
@pytest.mark.parametrize(
    "dfn, n_flat, n_params", [("gwf-ic", 2, 6), ("prt-prp", 40, 22)]
)
def test_get_template_context(dfn, n_flat, n_params):
    dfn_name = DefinitionName(*dfn.split("-"))

    with open(DFN_PATH / "common.dfn") as f:
        common, _ = load_dfn(f)

    with open(DFN_PATH / f"{dfn}.dfn") as f:
        definition = load_dfn(f)

    context = get_template_context(
        dfn_name,
        MFPackage,
        common,
        definition,
    )
    assert context["name"] == dfn_name
    assert len(context["parameters"]) == n_params
    assert len(context["dfn"]) == n_flat + 1  # +1 for metadata


@pytest.mark.parametrize("dfn", [dfn.stem for dfn in DFNS])
def test_render_template(dfn, function_tmpdir):
    dfn_name = DefinitionName(*dfn.split("-"))
    context_type = ContextType.from_dfn_name(dfn_name)
    template = TEMPLATE_ENV.get_template("context.jinja")

    with open(DFN_PATH / "common.dfn") as f:
        common, _ = load_dfn(f)

    with open(DFN_PATH / f"{dfn}.dfn", "r") as f:
        definition = load_dfn(f)

    context = {
        "context": context_type.value,
        **get_template_context(
            dfn_name,
            context_type.base,
            common,
            definition,
        ),
    }
    source = template.render(**context)
    source_path = function_tmpdir / dfn_name.target
    with open(source_path, "w") as f:
        f.write(source)
        run_cmd("ruff", "format", source_path, verbose=True)


def test_generate_components(function_tmpdir):
    generate_targets(DFN_PATH, function_tmpdir, verbose=True)
