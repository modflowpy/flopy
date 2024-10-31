import argparse
from pathlib import Path

from flopy.utils import import_optional_dependency

_MF6_PATH = Path(__file__).parents[2]
_DFN_PATH = _MF6_PATH / "data" / "dfn"
_TOML_PATH = _MF6_PATH / "data" / "toml"

__tomlkit = import_optional_dependency("tomlkit")


if __name__ == "__main__":
    """Convert DFN files to TOML."""

    if not __tomlkit:
        raise RuntimeError("tomlkit not installed, can't convert DFNs to TOML")

    from flopy.mf6.utils.codegen.dfn import Dfn

    parser = argparse.ArgumentParser(description="Convert DFN files to TOML.")
    parser.add_argument(
        "--dfndir",
        type=str,
        default=_DFN_PATH,
        help="Directory containing DFN files.",
    )
    parser.add_argument(
        "--outdir",
        default=_TOML_PATH,
        help="Output directory.",
    )

    args = parser.parse_args()
    dfndir = Path(args.dfndir)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    for dfn in Dfn.load_all(dfndir).values():
        with open(Path(outdir) / f"{'-'.join(dfn.name)}.toml", "w") as f:
            __tomlkit.dump(dfn.render(), f)
