import argparse
import json
import re
import subprocess
import textwrap
from datetime import datetime
from enum import Enum
from os import PathLike, environ
from pathlib import Path
from typing import NamedTuple, Optional

import yaml
from filelock import FileLock

_project_name = "flopy"
_project_root_path = Path(__file__).parent.parent
_version_txt_path = _project_root_path / "version.txt"
_version_py_path = _project_root_path / "flopy" / "version.py"

# file names and the path to the file relative to the repo root directory
file_paths_list = [
    _project_root_path / "CITATION.cff",
    _project_root_path / "code.json",
    _project_root_path / "README.md",
    _project_root_path / "docs" / "PyPI_release.md",
    _project_root_path / "flopy" / "version.py",
    _project_root_path / "flopy" / "DISCLAIMER.md",
]
file_paths = {pth.name: pth for pth in file_paths_list}  # keys for each file


approved_disclaimer = """Disclaimer
----------

This software is provided "as is" and "as-available", and makes no 
representations or warranties of any kind concerning the software, whether 
express, implied, statutory, or other. This includes, without limitation, 
warranties of title, merchantability, fitness for a particular purpose, 
non-infringement, absence of latent or other defects, accuracy, or the 
presence or absence of errors, whether or not known or discoverable.
"""

preliminary_disclaimer = """Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is 
being provided to meet the need for timely best science. This software is 
provided "as is" and "as-available", and makes no representations or warranties 
of any kind concerning the software, whether express, implied, statutory, or 
other. This includes, without limitation, warranties of title, 
merchantability, fitness for a particular purpose, non-infringement, absence 
of latent or other defects, accuracy, or the presence or absence of errors, 
whether or not known or discoverable.
"""


def split_nonnumeric(s):
    match = re.compile("[^0-9]").search(s)
    return [s[: match.start()], s[match.start() :]] if match else s


class Version(NamedTuple):
    """Semantic version number, optionally with a short label.
    The label may contain numbers but must not begin with a number."""

    major: int = 0
    minor: int = 0
    patch: int = 0
    label: Optional[str] = None

    def __repr__(self):
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.label is not None and self.label != "":
            s += self.label
        return s

    @classmethod
    def from_string(cls, version: str) -> "Version":
        t = version.split(".")
        assert len(t) > 2
        vmajor = int(t[0])
        vminor = int(t[1])
        tt = split_nonnumeric(t[2])
        vpatch = int(tt[0])
        vlabel = tt[1] if len(tt) > 1 else None
        return cls(major=vmajor, minor=vminor, patch=vpatch, label=vlabel)

    @classmethod
    def from_file(cls, path: PathLike) -> "Version":
        path = Path(path).expanduser().absolute()
        lines = [line.rstrip("\n") for line in open(Path(path), "r")]
        assert len(lines) == 1
        return Version.from_string(lines[0])


_initial_version = Version(0, 0, 1)
_current_version = Version.from_file(_version_txt_path)


def get_disclaimer(approved: bool = False):
    return approved_disclaimer if approved else preliminary_disclaimer


def update_version_txt(version: Version):
    with open(_version_txt_path, "w") as f:
        f.write(str(version))
    print(f"Updated {_version_txt_path} to version {version}")


def update_version_py(
    timestamp: datetime, version: Version
):
    with open(_version_py_path, "w") as f:
        f.write(
            f"# {_project_name} version file automatically created using "
            f"{Path(__file__).name} on {timestamp:%B %d, %Y %H:%M:%S}\n\n"
        )
        f.write("# created on..." + f"{timestamp.strftime('%B %d, %Y %H:%M:%S')}\n")
        f.write("\n")
        f.write(f"major = {version.major}\n")
        f.write(f"minor = {version.minor}\n")
        f.write(f"micro = {version.patch}\n")
        f.write(
            "label = " + (("'" + version.label + "'") if version.label else "''") + "\n"
        )
        f.write("__version__ = '{:d}.{:d}.{:d}'.format(major, minor, micro)\n")
        f.write("if label:\n")
        f.write("\t__version__ += '{}{}'.format(__version__, label)")
        f.close()
    print(f"Updated {_version_py_path} to version {version}")


def get_software_citation(
    timestamp: datetime, version: Version, approved: bool = False
):
    # get data Software/Code citation for FloPy
    citation = yaml.safe_load(file_paths["CITATION.cff"].read_text())

    sb = ""
    if not approved:
        sb = f" (preliminary)"
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
        f", {timestamp.year}, FloPy v{version}{sb}: "
        f"U.S. Geological Survey Software Release, {timestamp:%d %B %Y}, "
        "https://doi.org/10.5066/F7BK19FH]"
        "(https://doi.org/10.5066/F7BK19FH)"
    )

    return line


def update_codejson(
    timestamp: datetime, version: Version, approved: bool = False
):
    # define json filename
    json_fname = file_paths["code.json"]

    # load and modify json file
    data = json.loads(json_fname.read_text())

    # modify the json file data
    data[0]["date"]["metadataLastUpdated"] = timestamp.strftime("%Y-%m-%d")
    data[0]["version"] = str(version)
    data[0]["status"] = "Release" if approved else "Preliminary"

    # rewrite the json file
    with open(json_fname, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"Updated {json_fname} to version {version}")


def update_readme_markdown(
     timestamp: datetime, version: Version, approved: bool = False
):
    # create disclaimer text
    disclaimer = get_disclaimer(approved)

    # read README.md into memory
    fpth = file_paths["README.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite README.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "### Version " in line:
            line = f"### Version {version}"
            if not approved:
                line += f" (preliminary)"
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
        elif "[Binder]" in line:
            # [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)
            line = (
                "[![Binder](https://mybinder.org/badge_logo.svg)]"
                "(https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)"
            )
        elif "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(timestamp, version, approved)
        elif "Disclaimer" in line:
            line = disclaimer
            terminate = True
        f.write(f"{line}\n")
        if terminate:
            break

    f.close()
    print(f"Updated {fpth} to version {version}")

    # write disclaimer markdown file
    file_paths["DISCLAIMER.md"].write_text(disclaimer)
    print(f"Updated {file_paths['DISCLAIMER.md']} to version {version}")


def update_citation_cff(
    timestamp: datetime, version: Version
):
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


def update_PyPI_release(
    timestamp: datetime, version: Version, approved: bool = False
):
    # create disclaimer text
    disclaimer = get_disclaimer(approved)

    # read PyPI_release.md into memory
    fpth = file_paths["PyPI_release.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite PyPI_release.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(timestamp, version, approved)
        elif "Disclaimer" in line:
            line = disclaimer
            terminate = True
        f.write(f"{line}\n")
        if terminate:
            break

    f.close()
    print(f"Updated {fpth} to version {version}")


def update_version(
    timestamp: datetime = datetime.now(),
    version: Version = None,
    approved: bool = False
):
    lock_path = Path(_version_txt_path.name + ".lock")
    try:
        lock = FileLock(lock_path)
        previous = Version.from_file(_version_txt_path)
        version = (
            version
            if version
            else Version(previous.major, previous.minor, previous.patch)
        )

        with lock:
            update_version_txt(version)
            update_version_py(timestamp, version)
            update_readme_markdown(timestamp, version, approved)
            update_citation_cff(timestamp, version)
            update_codejson(timestamp, version, approved)
            update_PyPI_release(timestamp, version, approved)
    finally:
        try:
            lock_path.unlink()
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"Update {_project_name} version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Update version information stored in version.txt in the project root,
            as well as several other files in the repository. If --version is not
            provided, the version number will not be changed. A file lock is held
            to synchronize file access. The version tag must comply with standard
            '<major>.<minor>.<patch>' format conventions for semantic versioning.
            """
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        help="Specify the release version",
    )
    parser.add_argument(
        "-a",
        "--approve",
        required=False,
        action="store_true",
        help="Approve the release (defaults to false for preliminary/development distributions)",
    )
    parser.add_argument(
        "-g",
        "--get",
        required=False,
        action="store_true",
        help="Just get the current version number, don't update anything (defaults to false)",
    )
    args = parser.parse_args()

    if args.get:
        print(Version.from_file(_project_root_path / "version.txt"))
    else:
        update_version(
            timestamp=datetime.now(),
            version=Version.from_string(args.version)
            if args.version
            else _current_version,
            approved=args.approve
        )
