from pathlib import Path

from jinja2 import Environment, PackageLoader

from flopy.mf6.utils.codegen.context import Context
from flopy.mf6.utils.codegen.dfn import Dfn, Dfns
from flopy.mf6.utils.codegen.ref import Ref, Refs

_TEMPLATE_LOADER = PackageLoader("flopy", "mf6/utils/codegen/templates/")
_TEMPLATE_ENV = Environment(loader=_TEMPLATE_LOADER)
_TEMPLATE_NAME = "context.py.jinja"
_TEMPLATE = _TEMPLATE_ENV.get_template(_TEMPLATE_NAME)


def make_targets(dfn: Dfn, outdir: Path, verbose: bool = False):
    """Generate Python source file(s) from the given input definition."""

    for context in Context.from_dfn(dfn):
        target = outdir / context.name.target
        with open(target, "w") as f:
            source = _TEMPLATE.render(**context.render())
            f.write(source)
            if verbose:
                print(f"Wrote {target}")


def make_all(dfndir: Path, outdir: Path, verbose: bool = False):
    """Generate Python source files from the DFN files in the given location."""

    # find definition files
    paths = [
        p for p in dfndir.glob("*.dfn") if p.stem not in ["common", "flopy"]
    ]

    # try to load common variables
    common_path = dfndir / "common.dfn"
    if not common_path.is_file:
        common = None
    else:
        with open(common_path, "r") as f:
            common, _ = Dfn._load(f)

    # load subpackages first so we can pass them as references
    # to load() for the rest of the input contexts
    refs: Refs = {}
    for path in paths:
        name = Dfn.Name(*path.stem.split("-"))
        with open(path) as f:
            dfn = Dfn.load(f, name=name, common=common)
            ref = Ref.from_dfn(dfn)
            if ref:
                refs[ref.key] = ref

    # load all the input definitions before we generate input
    # contexts so we can create foreign key refs between them.
    dfns: Dfns = {}
    for path in paths:
        name = Dfn.Name(*path.stem.split("-"))
        with open(path) as f:
            dfn = Dfn.load(f, name=name, refs=refs, common=common)
            dfns[name] = dfn

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
