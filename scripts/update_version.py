import argparse
import re
import textwrap
from datetime import datetime
from pathlib import Path

import yaml
from filelock import FileLock
from packaging.version import Version

_epilog = """\
Update version information stored in version.txt in the project root,
as well as several other files in the repository. If --version is not
provided, the version number will not be changed. A file lock is held
to synchronize file access. The version tag must comply with standard
'<major>.<minor>.<patch>' format conventions for semantic versioning.
To show the version without changing anything, use --get (short -g).
"""
_project_name = "flopy"
_project_root_path = Path(__file__).parent.parent
_version_txt_path = _project_root_path / "version.txt"
_version_py_path = _project_root_path / "flopy" / "version.py"

# file names and the path to the file relative to the repo root directory
file_paths_list = [
    _project_root_path / "CITATION.cff",
    _project_root_path / "README.md",
    _project_root_path / "docs" / "PyPI_release.md",
    _project_root_path / "flopy" / "version.py",
]
file_paths = {pth.name: pth for pth in file_paths_list}  # keys for each file


def split_nonnumeric(s):
    match = re.compile("[^0-9]").search(s)
    return [s[: match.start()], s[match.start() :]] if match else s


_current_version = Version(_version_txt_path.read_text().strip())


def update_version_txt(version: Version):
    with open(_version_txt_path, "w") as f:
        f.write(str(version))
    print(f"Updated {_version_txt_path} to version {version}")


def update_version_py(timestamp: datetime, version: Version):
    with open(_version_py_path, "w") as f:
        f.write(
            f"# {_project_name} version file automatically created using\n"
            f"# {Path(__file__).name} on {timestamp:%B %d, %Y %H:%M:%S}\n\n"
        )
        f.write(f'__version__ = "{version}"\n')
        f.close()
    print(f"Updated {_version_py_path} to version {version}")


def get_software_citation(timestamp: datetime, version: Version):
    # get data Software/Code citation for FloPy
    citation = yaml.safe_load(file_paths["CITATION.cff"].read_text())

    # format author names
    authors = []
    for author in citation["authors"]:
        tauthor = author["family-names"] + ", "
        gnames = author["given-names"].split()
        if len(gnames) > 1:
            for gname in gnames:
                tauthor += gname[0]
                if len(gname) > 1:
                    tauthor += "."
                tauthor += " "
        else:
            tauthor += author["given-names"]
        authors.append(tauthor.rstrip())

    line = "["
    for ipos, tauthor in enumerate(authors):
        if ipos > 0:
            line += ", "
        if ipos == len(authors) - 1:
            line += "and "
        # add formatted author name to line
        line += tauthor

    # add the rest of the citation
    line += (
        f", {timestamp.year}, FloPy v{version}: "
        f"U.S. Geological Survey Software Release, {timestamp:%d %B %Y}, "
        "https://doi.org/10.5066/F7BK19FH]"
        "(https://doi.org/10.5066/F7BK19FH)"
    )

    return line


def update_readme_markdown(timestamp: datetime, version: Version):
    # read README.md into memory
    fpth = file_paths["README.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite README.md
    with open(fpth, "w") as f:
        for line in lines:
            if "### Version " in line:
                line = f"### Version {version}"
            elif "[flopy continuous integration]" in line:
                line = (
                    "[![flopy continuous integration](https://github.com/"
                    "modflowpy/flopy/actions/workflows/commit.yml/badge.svg?"
                    "branch=develop)](https://github.com/modflowpy/flopy/actions/"
                    "workflows/commit.yml)"
                )
            elif "[Read the Docs]" in line:
                line = (
                    "[![Read the Docs](https://github.com/modflowpy/flopy/"
                    "actions/workflows/rtd.yml/badge.svg?branch=develop)]"
                    "(https://github.com/modflowpy/flopy/actions/"
                    "workflows/rtd.yml)"
                )
            elif "[Coverage Status]" in line:
                line = (
                    "[![Coverage Status](https://coveralls.io/repos/github/"
                    "modflowpy/flopy/badge.svg?branch=develop)]"
                    "(https://coveralls.io/github/modflowpy/"
                    "flopy?branch=develop)"
                )
            elif "doi.org/10.5066/F7BK19FH" in line:
                line = get_software_citation(timestamp, version)

            f.write(f"{line}\n")

    print(f"Updated {fpth} to version {version}")


def update_citation_cff(timestamp: datetime, version: Version):
    # read CITATION.cff to modify
    fpth = file_paths["CITATION.cff"]
    citation = yaml.safe_load(fpth.read_text())

    # update version and date-released
    citation["version"] = str(version)
    citation["date-released"] = timestamp.strftime("%Y-%m-%d")

    # write CITATION.cff
    with open(fpth, "w") as f:
        yaml.safe_dump(
            citation,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"Updated {fpth} to version {version}")


def update_pypi_release(timestamp: datetime, version: Version):
    # read PyPI_release.md into memory
    fpth = file_paths["PyPI_release.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite PyPI_release.md
    f = open(fpth, "w")
    for line in lines:
        if "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(timestamp, version)

        f.write(f"{line}\n")

    f.close()
    print(f"Updated {fpth} to version {version}")


def update_version(
    timestamp: datetime = datetime.now(),
    version: Version = None,
):
    lock_path = Path(_version_txt_path.name + ".lock")
    try:
        lock = FileLock(lock_path)
        previous = Version(_version_txt_path.read_text().strip())
        version = (
            version
            if version
            else Version(previous.major, previous.minor, previous.micro)
        )

        with lock:
            update_version_txt(version)
            update_version_py(timestamp, version)
            update_readme_markdown(timestamp, version)
            update_citation_cff(timestamp, version)
            update_pypi_release(timestamp, version)
    finally:
        try:
            lock_path.unlink()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"Update {_project_name} version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(_epilog),
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        help="Specify the release version",
    )
    parser.add_argument(
        "-g",
        "--get",
        required=False,
        action="store_true",
        help="Just get the current version number, no updates (defaults false)",
    )
    args = parser.parse_args()

    if args.get:
        print(_current_version)
    else:
        update_version(
            timestamp=datetime.now(),
            version=(Version(args.version) if args.version else _current_version),
        )
