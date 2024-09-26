import pytest
from modflow_devtools.misc import run_cmd

from autotest.conftest import get_project_root_path
from flopy.mf6.utils.createpackages import (
    TEMPLATE_ENV,
    TemplateType,
    generate_components,
    get_src_name,
    get_template_context,
)
from flopy.mf6.utils.dfn import load_dfn

PROJ_ROOT = get_project_root_path()
DFNS_PATH = PROJ_ROOT / "flopy" / "mf6" / "data" / "dfn"
DFNS = [
    dfn
    for dfn in DFNS_PATH.glob("*.dfn")
    if dfn.stem not in ["common", "flopy"]
]


# only test packages for which we know the
# expected consolidated number of variables
@pytest.mark.parametrize(
    "dfn, n_flat, n_nested", [("gwf-ic", 2, 2), ("prt-prp", 40, 18)]
)
def test_get_template_context(dfn, n_flat, n_nested):
    component, subcomponent = dfn.split("-")

    with open(DFNS_PATH / "common.dfn") as f:
        common_vars, _ = load_dfn(f)

    with open(DFNS_PATH / "flopy.dfn") as f:
        flopy_vars, _ = load_dfn(f)

    with open(DFNS_PATH / f"{dfn}.dfn") as f:
        variables, metadata = load_dfn(f)

    context = get_template_context(
        component, subcomponent, common_vars, flopy_vars, variables, metadata
    )
    assert context["component"] == component
    assert context["subcomponent"] == subcomponent
    assert len(context["variables"]) == n_nested
    assert len(context["dfn"]) == n_flat + 1  # +1 for metadata


@pytest.mark.parametrize("dfn", [dfn.stem for dfn in DFNS])
def test_render_template(dfn, function_tmpdir):
    component, subcomponent = dfn.split("-")
    context_name = f"{component}{subcomponent}"
    template_type = TemplateType.from_pair(component, subcomponent).value
    template = TEMPLATE_ENV.get_template(f"{template_type}.jinja")

    with open(DFNS_PATH / "common.dfn") as f:
        common_vars, _ = load_dfn(f)

    with open(DFNS_PATH / "flopy.dfn") as f:
        flopy_vars, _ = load_dfn(f)

    with open(DFNS_PATH / f"{dfn}.dfn", "r") as f:
        variables, metadata = load_dfn(f)

    context = get_template_context(
        component, subcomponent, common_vars, flopy_vars, variables, metadata
    )
    source = template.render(**context)
    source_path = function_tmpdir / get_src_name(component, subcomponent)
    with open(source_path, "w") as f:
        f.write(source)
        run_cmd("ruff", "format", source_path, verbose=True)


def test_generate_components(function_tmpdir):
    generate_components(function_tmpdir, verbose=True)
