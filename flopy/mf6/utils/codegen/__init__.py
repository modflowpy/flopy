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
    env.filters["variables"] = Filters.variables
    env.filters["attrs"] = Filters.attrs
    env.filters["init"] = Filters.init
    env.filters["untag"] = Filters.untag
    env.filters["type"] = Filters.type
    env.filters["children"] = Filters.children
    env.filters["default_value"] = Filters.default_value
    env.filters["safe_name"] = Filters.safe_name
    env.filters["value"] = Filters.value
    env.filters["math"] = Filters.math
    env.filters["clean"] = Filters.clean

    return env


def make_init(dfns: dict, outdir: PathLike, verbose: bool = False):
    """Generate a Python __init__.py file for the given input definitions."""

    env = _get_template_env()
    outdir = Path(outdir).expanduser().absolute()

    # import here instead of module so we don't
    # expect optional deps at module init time
    from flopy.mf6.utils.codegen.component import ComponentDescriptor

    components = list(
        chain.from_iterable(ComponentDescriptor.from_dfn(dfn) for dfn in dfns.values())
    )
    target_name = "__init__.py"
    target_path = outdir / target_name
    template = env.get_template(f"{target_name}.jinja")
    with open(target_path, "w") as f:
        f.write(template.render(components=components))
        if verbose:
            print(f"Wrote {target_path}")


def make_targets(dfn, outdir: PathLike, verbose: bool = False):
    """Generate Python source file(s) from the given input definition."""

    env = _get_template_env()
    outdir = Path(outdir).expanduser().absolute()

    # import here instead of module so we don't
    # expect optional deps at module init time
    from flopy.mf6.utils.codegen.component import ComponentDescriptor
    from flopy.mf6.utils.codegen.filters import Filters

    def _get_template_name(component_name) -> str:
        base = Filters.base(component_name)
        if base == "MFSimulationBase":
            return "simulation.py.jinja"
        elif base == "MFModel":
            return "model.py.jinja"
        elif base == "MFPackage":
            if component_name[0] == "exg":
                return "exchange.py.jinja"
            return "package.py.jinja"
        else:
            raise NotImplementedError(f"Unknown base class: {base}")
        
    for component in ComponentDescriptor.from_dfn(dfn):
        component_name = component["name"]
        target_path = outdir / f"mf{Filters.title(component_name)}.py"
        template = env.get_template(_get_template_name(component_name))
        with open(target_path, "w") as f:
            f.write(template.render(**component))
            if verbose:
                print(f"Wrote {target_path}")


def make_all(dfndir: Path, outdir: PathLike, verbose: bool = False, version: int = 1):
    """Generate Python source files from the DFN files in the given location."""

    # import here instead of module so we don't
    # expect optional deps at module init time
    from flopy.mf6.utils.dfn import Dfn

    dfns = Dfn.load_all(dfndir, version=version)
    make_init(dfns, outdir, verbose)
    for dfn in dfns.values():
        make_targets(dfn, outdir, verbose)
