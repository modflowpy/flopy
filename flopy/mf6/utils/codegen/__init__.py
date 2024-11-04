from os import PathLike
from pathlib import Path

from flopy.utils import import_optional_dependency

__all__ = ["make_targets", "make_all"]


def _get_template_env():
    from flopy.mf6.utils.codegen.jinja import Filters

    jinja = import_optional_dependency("jinja2")
    loader = jinja.PackageLoader("flopy", "mf6/utils/codegen/templates/")
    env = jinja.Environment(loader=loader)
    env.filters["parent"] = Filters.parent
    env.filters["prefix"] = Filters.prefix
    env.filters["skip"] = Filters.skip
    return env


def make_init(dfns: dict, outdir: PathLike, verbose: bool = False):
    """Generate a Python __init__.py file for the given input definitions."""

    from flopy.mf6.utils.codegen.context import Context

    env = _get_template_env()
    outdir = Path(outdir).expanduser()
    contexts = [
        c
        for cc in [
            [ctx for ctx in Context.from_dfn(dfn)] for dfn in dfns.values()
        ]
        for c in cc
    ]  # ugly, but it's the fastest way to flatten the list
    target_name = "__init__.py"
    target = outdir / target_name
    template = env.get_template(f"{target_name}.jinja")
    with open(target, "w") as f:
        f.write(template.render(contexts=contexts))
        if verbose:
            print(f"Wrote {target}")


def make_targets(dfn, outdir: PathLike, verbose: bool = False):
    """Generate Python source file(s) from the given input definition."""

    from flopy.mf6.utils.codegen.context import Context

    env = _get_template_env()
    outdir = Path(outdir).expanduser()
    for context in Context.from_dfn(dfn):
        name = context.name
        target = outdir / name.target
        template = env.get_template(name.template)
        with open(target, "w") as f:
            f.write(template.render(**context.render()))
            if verbose:
                print(f"Wrote {target}")


def make_all(dfndir: Path, outdir: Path, verbose: bool = False):
    """Generate Python source files from the DFN files in the given location."""

    from flopy.mf6.utils.codegen.dfn import Dfn

    dfns = Dfn.load_all(dfndir)
    make_init(dfns, outdir, verbose)
    for dfn in dfns.values():
        make_targets(dfn, outdir, verbose)
