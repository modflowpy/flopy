import argparse
from pathlib import Path

if __name__ == "__main__":
    """Reverse a TDIS (ASCII) input file or a head or budget (binary) output file."""

    parser = argparse.ArgumentParser(description="Reverse head, budget, or TDIS files.")
    parser.add_argument(
        "--head",
        type=str,
        help="Head file",
    )
    parser.add_argument(
        "--budget",
        type=str,
        help="Budget file",
    )
    parser.add_argument(
        "--tdis",
        type=str,
        help="TDIS file",
    )
    parser.add_argument(
        "--head-output",
        type=str,
        help="Reversed head file",
    )
    parser.add_argument(
        "--budget-output",
        type=str,
        help="Reversed budget file",
    )
    parser.add_argument(
        "--tdis-output",
        type=str,
        help="Reversed TDIS file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.head:
        from flopy.utils.binaryfile import HeadFile

        head_input = Path(args.head).expanduser().absolute()
        if not head_input.is_file():
            raise FileNotFoundError(f"Head file {head_input} not found.")
        if args.head_output:
            head_output = Path(args.head_output).expanduser().absolute()
        else:
            head_output = head_input.with_suffix(".reversed.hds")
        if args.verbose:
            print(f"Reversing head file {head_input} to {head_output}")
        HeadFile(head_input).reverse(head_output)

    if args.budget:
        from flopy.utils.binaryfile import CellBudgetFile

        budget_input = Path(args.budget).expanduser().absolute()
        if not budget_input.is_file():
            raise FileNotFoundError(f"Budget file {budget_input} not found.")
        if args.budget_output:
            budget_output = Path(args.budget_output).expanduser().absolute()
        else:
            budget_output = budget_input.with_suffix(".reversed.cbc")
        if args.verbose:
            print(f"Reversing budget file {budget_input} to {budget_output}")
        CellBudgetFile(budget_input).reverse(budget_output)

    if args.tdis:
        from flopy.discretization.modeltime import ModelTime
        from flopy.mf6 import MFSimulation, ModflowTdis

        tdis_input = Path(args.tdis).expanduser().absolute()
        if not tdis_input.is_file():
            raise FileNotFoundError(f"TDIS file {tdis_input} not found.")
        if args.tdis_output:
            tdis_output = Path(args.tdis_output).expanduser().absolute()
        else:
            tdis_output = tdis_input.with_suffix(".reversed.tdis")
        if args.verbose:
            print(f"Reversing TDIS file {tdis_input} to {tdis_output}")
        tdis = ModflowTdis(MFSimulation(), filename=tdis_input)
        tdis.load()
        time = ModelTime.from_perioddata(tdis.perioddata.get_data())
        trev = time.reverse()
        tdis.perioddata.set_data(trev.perioddata)
        tdis._filename = tdis_output
        tdis.write()
