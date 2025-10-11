import cmd
import shutil
import tempfile
from pathlib import Path

_PROJ_ROOT_PATH = Path(__file__).parents[3].expanduser().resolve().absolute()
_MF6_MODULE_PATH = _PROJ_ROOT_PATH / "flopy" / "mf6"
_MF6_AUTOGEN_PATH = _MF6_MODULE_PATH / "modflow"
_MF6_REPO_OWNER = "MODFLOW-ORG"
_MF6_REPO_NAME = "modflow6"
_CMD = cmd.Cmd()  # for pretty printing


def generate_classes(
    owner=_MF6_REPO_OWNER,
    repo=_MF6_REPO_NAME,
    ref=None,
    dfnpath=None,
    verbose=False,
    developmode=True,
):
    """
    Generate Python classes for MODFLOW 6 using definition files fetched
    from the MODFLOW 6 repository or available on the local filesystem.

    Parameters
    ----------
    owner : str, default "MODFLOW-ORG"
        Owner of the MODFLOW 6 repository to use to update the definition
        files and generate the MODFLOW 6 classes.
    repo : str, default "modflow6"
        Name of the MODFLOW 6 repository to use to update the definition.
    ref : str, default "master"
        Branch name, tag, or commit hash to use to update the definition.
    dfnpath : str
        Path to a definition file folder that will be used to generate the
        MODFLOW 6 classes.  Default is none, which means that the branch
        will be used instead.  dfnpath will take precedence over branch
        if dfnpath is specified.
    backup : bool, default True
        Keep a backup of the definition files in dfn_backup with a date and
        timestamp from when the definition files were replaced.
    verbose : bool, default False
        If True, print information about the code generation process.
    developmode : bool, default True
        If True, include all variables, including developmode variables.
        If False, omit developmode variables from generated modules.
    """

    if dfnpath is None and ref is None:
        raise ValueError("Need remote 'ref' or local 'dfnpath'")

    # import here instead of module so we don't
    # expect optional deps at module init time
    from modflow_devtools.dfn2toml import convert as dfn2toml
    from modflow_devtools.download import download_and_unzip

    from flopy.mf6.utils.codegen import make_all

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if dfnpath is not None:
            dfnpath = Path(dfnpath).expanduser().resolve().absolute()
            if not dfnpath.is_dir():
                raise FileNotFoundError(
                    f"Specified DFN path '{dfnpath}' does not exist or is not a directory."
                )
        else:
            if verbose:
                print(f"Fetching MODFLOW 6 definitions from: {owner}/{repo}/{ref}")

            url = f"https://github.com/{owner}/{repo}/archive/{ref}.zip"
            dl_path = download_and_unzip(url=url, path=tmpdir, verbose=verbose)
            if (proj_root := next(iter(dl_path.glob("modflow6-*")), None)) is None:
                raise ValueError(f"Could not find MODFLOW 6 project root in: {dl_path}")
            dfnpath = tmpdir / "dfn"
            shutil.copytree(
                proj_root / "doc" / "mf6io" / "mf6ivar" / "dfn", dfnpath
            )

        dfns = list(dfnpath.glob("*.dfn"))
        module_name = ".".join(_MF6_AUTOGEN_PATH.relative_to(_PROJ_ROOT_PATH).parts)
        print(
            f"Generating module {module_name} from {len(dfns)} DFNs in: {dfnpath}"
        )
        print()
        _CMD.columnize([f.name for f in dfns])
        print()

        tomlpath = dfnpath / "toml"
        tomlpath.mkdir(exist_ok=True)
        dfn2toml(dfnpath, tomlpath)

        shutil.rmtree(_MF6_AUTOGEN_PATH)
        _MF6_AUTOGEN_PATH.mkdir(parents=True)
        make_all(tomlpath, _MF6_AUTOGEN_PATH, version=2, legacydir=dfnpath, verbose=verbose, developmode=developmode)

        files = list(_MF6_AUTOGEN_PATH.glob("*.py"))
        print(f"Generated {len(files)} module files in: {_MF6_AUTOGEN_PATH}")
        print()
        _CMD.columnize([f.name for f in files])
        print()


def cli_main():
    """Command-line interface for generate_classes()."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=generate_classes.__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=_MF6_REPO_OWNER,
        help=f"GitHub repository owner; default is '{_MF6_REPO_OWNER}'.",
    )
    parser.add_argument(
        "--repo",
        default=_MF6_REPO_NAME,
        help=f"Name of GitHub repository; default is '{_MF6_REPO_NAME}'.",
    )
    parser.add_argument(
        "--ref",
        default="master",
        help="Branch name, tag, or commit hash to use to update the "
        "definition; default is 'master'.",
    )
    parser.add_argument(
        "--dfnpath",
        help="Path to a definition file folder that will be used to generate "
        "the MODFLOW 6 classes.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print extra information about the code generation process; "
        "default shows verbose output.",
    )
    parser.add_argument(
        "--exclude", help="Exclude DFNs matching a pattern.", action="append"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Set to disable backup. "
        "Default behavior is to keep a backup of the definition files in "
        "dfn_backup with a date and timestamp from when the definition "
        "files were replaced.",
    )
    parser.add_argument(
        "-r",
        "--releasemode",
        required=False,
        action="store_true",
        help="Omit developmode variables from generated modules "
        "(DFN variables with 'developmode true'). Defaults to false.",
    )
    args = vars(parser.parse_args())
    args["developmode"] = not args.pop("releasemode", False)

    # ignore/warn removed options
    exclude = args.pop("exclude", None)
    no_backup = args.pop("no_backup", None)
    if exclude:
        print(
            "The '--exclude' option is no longer supported. "
            "Exclude DFNs and corresponding source files manually."
        )
    if no_backup:
        print(
            "The '--no-backup' option is no longer supported. "
            "Exclude DFNs and corresponding source files manually."
        )

    try:
        generate_classes(**args)
    except (EOFError, KeyboardInterrupt):
        sys.exit(f" cancelling '{sys.argv[0]}'")


if __name__ == "__main__":
    """Run command-line with: python -m flopy.mf6.utils.generate_classes"""
    cli_main()
