from pathlib import Path
from typing import (
    Optional,
)
from warnings import warn

from jinja2 import Environment, FileSystemLoader

# noqa: F401
from flopy.mf6.utils.codegen.context import (
    get_context_names,
    make_context,
    make_contexts,
)
from flopy.mf6.utils.codegen.dfn import Dfn, DfnName, Dfns
from flopy.mf6.utils.codegen.ref import Ref, Refs

_TEMPLATE_LOADER = FileSystemLoader(Path(__file__).parent / "templates")
_TEMPLATE_ENV = Environment(loader=_TEMPLATE_LOADER)
_TEMPLATE_NAME = "context.py.jinja"
_TEMPLATE = _TEMPLATE_ENV.get_template(_TEMPLATE_NAME)


def make_targets(
    definition: Dfn,
    outdir: Path,
    commonvars: Optional[Dfn] = None,
    references: Optional[Refs] = None,
    verbose: bool = False,
):
    """Generate Python source file(s) from the given input definition."""

    for context in make_contexts(
        definition=definition, commonvars=commonvars, references=references
    ):
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
        warn("No common input definition file...")
        common = None
    else:
        with open(common_path, "r") as f:
            common = Dfn.load(f)

    # load all the input definitions before we generate input
    # contexts so we can create foreign key refs between them.
    dfns: Dfns = {}
    refs: Refs = {}
    for p in paths:
        name = DfnName(*p.stem.split("-"))
        with open(p) as f:
            dfn = Dfn.load(f, name=name)
            dfns[name] = dfn
            ref = Ref.from_dfn(dfn)
            if ref:
                # key is the name of the file record
                # that's the reference's foreign key
                refs[ref.key] = ref

    # generate target files
    for dfn in dfns.values():
        with open(p) as f:
            make_targets(
                definition=dfn,
                outdir=outdir,
                references=refs,
                commonvars=common,
                verbose=verbose,
            )

    # generate __init__.py file
    init_path = outdir / "__init__.py"
    with open(init_path, "w") as f:
        for dfn in dfns.values():
            for ctx in get_context_names(dfn.name):
                prefix = "MF" if ctx.base == "MFSimulationBase" else "Modflow"
                f.write(
                    f"from .mf{ctx.title} import {prefix}{ctx.title.title()}\n"
                )
