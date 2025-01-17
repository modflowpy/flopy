from itertools import chain
from os import PathLike
from pathlib import Path

__all__ = ["make_init", "make_targets", "make_all"]


def _get_template_env():
    # import here instead of module so we don't
    # expect optional deps at module init time
    import jinja2

    loader = jinja2.PackageLoader("flopy", "mf6/utils/codegen/templates/")
    env = jinja2.Environment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        line_statement_prefix="_",
        keep_trailing_newline=True,
    )

    from flopy.mf6.utils.codegen.filters import Filters

    env.filters["base"] = Filters.base
    env.filters["title"] = Filters.title
    env.filters["description"] = Filters.description
    env.filters["prefix"] = Filters.prefix
    env.filters["parent"] = Filters.parent
    env.filters["skip_init"] = Filters.skip_init
    env.filters["package_abbr"] = Filters.package_abbr
    env.filters["attrs"] = Filters.attrs
    env.filters["init"] = Filters.init
    env.filters["untag"] = Filters.untag
    env.filters["type"] = Filters.type
    env.filters["children"] = Filters.children
    env.filters["default"] = Filters.default
    env.filters["safe_name"] = Filters.safe_name
    env.filters["escape_trailing_underscore"] = (
        Filters.escape_trailing_underscore
    )
    env.filters["value"] = Filters.value
    env.filters["math"] = Filters.math
    env.filters["clean"] = Filters.clean

    return env


def make_init(dfns: dict, outdir: PathLike, verbose: bool = False):
    """Generate a Python __init__.py file for the given input definitions."""

    env = _get_template_env()
    outdir = Path(outdir).expanduser().absolute()

    from flopy.mf6.utils.codegen.context import Context

    contexts = list(
        chain.from_iterable(Context.from_dfn(dfn) for dfn in dfns.values())
    )
    target_name = "__init__.py"
    target_path = outdir / target_name
    template = env.get_template(f"{target_name}.jinja")
    with open(target_path, "w") as f:
        f.write(template.render(contexts=contexts))
        if verbose:
            print(f"Wrote {target_path}")


def make_targets(dfn, outdir: PathLike, verbose: bool = False):
    """Generate Python source file(s) from the given input definition."""

    env = _get_template_env()
    outdir = Path(outdir).expanduser().absolute()

    from flopy.mf6.utils.codegen.context import Context
    from flopy.mf6.utils.codegen.filters import Filters

    def _get_template_name(ctx_name) -> str:
        """The template file to use."""
        base = Filters.base(ctx_name)
        if base == "MFSimulationBase":
            return "simulation.py.jinja"
        elif base == "MFModel":
            return "model.py.jinja"
        elif base == "MFPackage":
            if ctx_name.l == "exg":
                return "exchange.py.jinja"
            return "package.py.jinja"
        else:
            raise NotImplementedError(f"Unknown base class: {base}")
        
    for context in Context.from_dfn(dfn):
        name = context["name"]
        target_path = outdir / f"mf{Filters.title(name)}.py"
        template = env.get_template(_get_template_name(name))
        with open(target_path, "w") as f:
            f.write(template.render(**context))
            if verbose:
                print(f"Wrote {target_path}")


def make_all(dfndir: Path, outdir: PathLike, verbose: bool = False, version: int = 1):
    """Generate Python source files from the DFN files in the given location."""

    from modflow_devtools.dfn import Dfn

    dfns = Dfn.load_all(dfndir, version=version)
    make_init(dfns, outdir, verbose)
    for dfn in dfns.values():
        make_targets(dfn, outdir, verbose)
