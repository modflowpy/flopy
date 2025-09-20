from itertools import chain
from os import PathLike
from pathlib import Path

__all__ = ["make_init", "make_targets", "make_all"]


def _get_template_env(developmode: bool = True):
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

    from flopy.mf6.utils.codegen import filters

    env.filters["base"] = filters.base
    env.filters["title"] = filters.title
    env.filters["description"] = filters.description
    env.filters["prefix"] = filters.prefix
    env.filters["parent"] = filters.parent
    env.filters["skip_init"] = filters.skip_init
    env.filters["package_abbr"] = filters.package_abbr
    env.filters["variables"] = lambda dfn: filters.variables(dfn, developmode=developmode)
    env.filters["attrs"] = lambda dfn, component_name: filters.attrs(dfn, component_name, developmode=developmode)
    env.filters["init"] = lambda dfn, component_name: filters.init(dfn, component_name, developmode=developmode)
    env.filters["untag"] = filters.untag
    env.filters["type"] = filters.type
    env.filters["children"] = filters.children
    env.filters["default_value"] = filters.default_value
    env.filters["safe_name"] = filters.safe_name
    env.filters["value"] = filters.value
    env.filters["math"] = filters.math
    env.filters["clean"] = filters.clean

    env.globals.update({"developmode": developmode})

    return env


def make_init(dfns: dict, outdir: PathLike, verbose: bool = False, developmode: bool = True):
    """Generate a Python __init__.py file for the given input definitions."""

    env = _get_template_env(developmode=developmode)
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


def make_targets(dfn, outdir: PathLike, verbose: bool = False, developmode: bool = True):
    """Generate Python source file(s) from the given input definition."""

    env = _get_template_env(developmode=developmode)
    outdir = Path(outdir).expanduser().resolve().absolute()

    # import here instead of module so we don't
    # expect optional deps at module init time
    from flopy.mf6.utils.codegen import filters
    from flopy.mf6.utils.codegen.component import ComponentDescriptor

    def _get_template_name(component_name) -> str:
        base = filters.base(component_name)
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

    if verbose:
        print(f"Making target for DFN {dfn['name']!r} ...")

    for component in ComponentDescriptor.from_dfn(dfn):
        component_name = component["name"]
        target_path = outdir / f"mf{filters.title(component_name)}.py"
        template = env.get_template(_get_template_name(component_name))
        with open(target_path, "w") as f:
            f.write(template.render(**component))
            if verbose:
                print(f"Wrote {target_path}")


def make_all(
    dfndir: PathLike,
    outdir: PathLike,
    verbose: bool = False,
    version: int = 1,
    legacydir: PathLike | None = None,
    developmode: bool = True,
):
    """Generate Python source files from the DFN files in the given location."""

    # import here instead of module so we don't
    # expect optional deps at module init time
    from modflow_devtools.dfn import Dfn

    dfndir = Path(dfndir).expanduser().resolve().absolute()
    dfns = Dfn.load_all(dfndir, version=version)

    # rename dfn keys with "-nam" for simulations and models.
    # won't be necessary for 4.x.
    def _add_nam_suffix(dfn):
        nam_types = {"sim", "gwf", "gwt", "gwe", "prt", "olf", "chf", "swf"}
        name = dfn["name"]
        new_name = name + "-nam" if name in nam_types else name
        return new_name, {**dfn, "name": new_name}

    dfns = dict(_add_nam_suffix(dfn) for dfn in dfns.values())

    # below is a temporary workaround to attach the legacy DFN
    # representation to generated classes. at the moment it is
    # parsed haphazardly throughout the mf6 module. TODO: when
    # the legacy DFN is no longer needed at runtime, remove.
    if version == 2:
        assert legacydir is not None, (
            "legacydir must be provided for version 2 DFNs"
        )
        legacydir = Path(legacydir).expanduser().resolve().absolute()
        with open(legacydir / "common.dfn") as cf:
            common, _ = Dfn._load_v1_flat(cf)
            for dfn_name, dfn in dfns.items():
                with open(legacydir / f"{dfn_name}.dfn") as df:
                    legacy_dfn, legacy_meta = Dfn._load_v1_flat(df, common=common)
                    dfn["legacy_dfn"] = legacy_dfn
                    dfn["legacy_meta"] = legacy_meta

    make_init(dfns, outdir, verbose)
    for dfn in dfns.values():
        make_targets(dfn, outdir, verbose, developmode=developmode)
