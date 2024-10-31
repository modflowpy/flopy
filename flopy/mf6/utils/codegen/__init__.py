from pathlib import Path
from warnings import warn

from flopy.utils import import_optional_dependency

__all__ = ["make_targets", "make_all"]

jinja = import_optional_dependency("jinja2", errors="ignore")
if jinja:
    _TEMPLATES_PATH = "mf6/utils/codegen/templates/"
    _TEMPLATE_LOADER = jinja.PackageLoader("flopy", _TEMPLATES_PATH)
    _TEMPLATE_ENV = jinja.Environment(loader=_TEMPLATE_LOADER)


def make_targets(dfn, outdir: Path, verbose: bool = False):
    """Generate Python source file(s) from the given input definition."""

    from flopy.mf6.utils.codegen.context import Context

    if not jinja:
        raise RuntimeError("Jinja2 not installed, can't make targets")

    for context in Context.from_dfn(dfn):
        name = context.name
        target = outdir / name.target
        template = _TEMPLATE_ENV.get_template(name.template)
        with open(target, "w") as f:
            f.write(template.render(**context.render()))
            if verbose:
                print(f"Wrote {target}")


def make_all(dfndir: Path, outdir: Path, verbose: bool = False):
    """Generate Python source files from the DFN files in the given location."""

    from flopy.mf6.utils.codegen.context import Context
    from flopy.mf6.utils.codegen.dfn import Dfn

    if not jinja:
        raise RuntimeError("Jinja2 not installed, can't make targets")

    # load dfns
    dfns = Dfn.load_all(dfndir)

    # make target files
    for dfn in dfns.values():
        make_targets(dfn, outdir, verbose)

    # make __init__.py file
    init_path = outdir / "__init__.py"
    with open(init_path, "w") as f:
        for dfn in dfns.values():
            for name in Context.Name.from_dfn(dfn):
                prefix = "MF" if name.base == "MFSimulationBase" else "Modflow"
                f.write(
                    f"from .mf{name.title} import {prefix}{name.title.title()}\n"
                )
