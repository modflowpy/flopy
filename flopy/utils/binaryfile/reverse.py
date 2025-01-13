import argparse
from pathlib import Path

from flopy.utils.binaryfile import CellBudgetFile, HeadFile

if __name__ == "__main__":
    """Reverse head or budget files."""

    parser = argparse.ArgumentParser(description="Reverse head or budget files.")
    parser.add_argument(
        "--infile",
        "-i",
        type=str,
        help="Input file.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        help="Output file.",
    )
    args = parser.parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    suffix = infile.suffix.lower()
    if suffix in [".hds", ".hed"]:
        HeadFile(infile).reverse(outfile)
    elif suffix in [".bud", ".cbc"]:
        CellBudgetFile(infile).reverse(outfile)
    else:
        raise ValueError(f"Unrecognized file suffix: {suffix}")
