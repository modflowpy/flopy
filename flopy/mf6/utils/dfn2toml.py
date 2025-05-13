"""Convert DFNs to TOML."""

import argparse
from os import PathLike
from pathlib import Path

import tomli_w as tomli
from boltons.iterutils import remap

from flopy.mf6.utils.dfn import Dfn

# mypy: ignore-errors


def convert(indir: PathLike, outdir: PathLike):
    indir = Path(indir).expanduser().absolute()
    outdir = Path(outdir).expanduser().absolute()
    outdir.mkdir(exist_ok=True, parents=True)
    for dfn in Dfn.load_all(indir).values():
        with Path.open(outdir / f"{dfn['name']}.toml", "wb") as f:

            def drop_none_or_empty(path, key, value):
                if value is None or value == "" or value == [] or value == {}:
                    return False
                return True

            tomli.dump(remap(dfn, visit=drop_none_or_empty), f)


if __name__ == "__main__":
    """Convert DFN files to TOML."""

    parser = argparse.ArgumentParser(description="Convert DFN files to TOML.")
    parser.add_argument(
        "--indir",
        "-i",
        type=str,
        help="Directory containing DFN files.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="Output directory.",
    )
    args = parser.parse_args()
    convert(args.indir, args.outdir)
