import argparse
import json
import subprocess
import textwrap
from datetime import datetime
from enum import Enum
from os import PathLike, environ
from pathlib import Path
from typing import NamedTuple

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
    _project_root_path / "docs" / "notebook_examples.md",
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


class Version(NamedTuple):
    """Semantic version number"""

    major: int = 0
    minor: int = 0
    patch: int = 0

    def __repr__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version: str) -> "Version":
        t = version.split(".")

        vmajor = int(t[0])
        vminor = int(t[1])
        vpatch = int(t[2])

        return cls(major=vmajor, minor=vminor, patch=vpatch)

    @classmethod
    def from_file(cls, path: PathLike) -> "Version":
        lines = [
            line.rstrip("\n")
            for line in open(Path(path).expanduser().absolute())
        ]
        vmajor = vminor = vpatch = None
        for line in lines:
            line = line.strip()
            if not any(line):
                continue
            t = line.split(".")
            vmajor = int(t[0])
            vminor = int(t[1])
            vpatch = int(t[2])

        assert (
            vmajor is not None and vminor is not None and vpatch is not None
        ), "version string must follow semantic version format: major.minor.patch"
        return cls(major=vmajor, minor=vminor, patch=vpatch)


class ReleaseType(Enum):
    CANDIDATE = "Release Candidate"
    APPROVED = "Production"


_initial_version = Version(0, 0, 1)
_current_version = Version.from_file(_version_txt_path)


def get_disclaimer(release_type: ReleaseType):
    if release_type == ReleaseType.APPROVED:
        return approved_disclaimer
    else:
        return preliminary_disclaimer


def get_branch():
    branch = None
    error = ValueError("Coulnd't detect branch")
    try:
        # determine current branch
        b = subprocess.Popen(
            ("git", "status"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ).communicate()[0]
        if isinstance(b, bytes):
            b = b.decode("utf-8")

        # determine current branch
        for line in b.splitlines():
            if "On branch" in line:
                branch = line.replace("On branch ", "").rstrip()
        if branch is None:
            raise error
    except:
        branch = environ.get("GITHUB_REF_NAME", None)

    if branch is None:
        raise error
    else:
        print(f"Detected branch: {branch}")

    return branch


def update_version_txt(version: Version):
    with open(_version_txt_path, "w") as f:
        f.write(str(version))
    print(f"Updated {_version_txt_path} to version {version}")


def update_version_py(
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    with open(_version_py_path, "w") as f:
        f.write(
            f"# {_project_name} version file automatically created using "
            f"{Path(__file__).name} on {timestamp:%B %d, %Y %H:%M:%S}\n\n"
        )
        f.write(f"major = {version.major}\n")
        f.write(f"minor = {version.minor}\n")
        f.write(f"micro = {version.patch}\n")
        f.write('__version__ = f"{major}.{minor}.{micro}'\n")
    print(f"Updated {_version_py_path} to version {version}")


def get_software_citation(
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    # get data Software/Code citation for FloPy
    citation = yaml.safe_load(file_paths["CITATION.cff"].read_text())

    sb = ""
    if release_type != ReleaseType.APPROVED:
        sb = f" &mdash; {release_type.value.lower()}"
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
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    # define json filename
    json_fname = file_paths["code.json"]

    # load and modify json file
    data = json.loads(json_fname.read_text())

    # modify the json file data
    data[0]["date"]["metadataLastUpdated"] = timestamp.strftime("%Y-%m-%d")
    data[0]["version"] = str(version)
    data[0]["status"] = release_type.value

    # rewrite the json file
    with open(json_fname, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"Updated {json_fname} to version {version}")


def update_readme_markdown(
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    # create disclaimer text
    disclaimer = get_disclaimer(release_type)

    # read README.md into memory
    fpth = file_paths["README.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite README.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "### Version " in line:
            line = f"### Version {version}"
            if release_type != ReleaseType.APPROVED:
                line += f" &mdash; {release_type.value.lower()}"
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
            line = get_software_citation(release_type, timestamp, version)
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
    release_type: ReleaseType, timestamp: datetime, version: Version
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


def update_notebook_examples_markdown(
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    # create disclaimer text
    disclaimer = get_disclaimer(release_type)

    # read notebook_examples.md into memory
    fpth = file_paths["notebook_examples.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite notebook_examples.md
    f = open(fpth, "w")
    for line in lines:
        if "[Binder]" in line:
            # [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)
            line = (
                "[![Binder](https://mybinder.org/badge_logo.svg)]"
                "(https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)"
            )
        f.write(f"{line}\n")
    f.close()
    print(f"Updated {fpth} to version {version}")


def update_PyPI_release(
    release_type: ReleaseType, timestamp: datetime, version: Version
):
    # create disclaimer text
    disclaimer = get_disclaimer(release_type)

    # read PyPI_release.md into memory
    fpth = file_paths["PyPI_release.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite PyPI_release.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(release_type, timestamp, version)
        elif "Disclaimer" in line:
            line = disclaimer
            terminate = True
        f.write(f"{line}\n")
        if terminate:
            break

    f.close()
    print(f"Updated {fpth} to version {version}")


def update_version(
    release_type: ReleaseType,
    timestamp: datetime = datetime.now(),
    version: Version = None,
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
            update_version_py(release_type, timestamp, version)
            update_readme_markdown(release_type, timestamp, version)
            update_citation_cff(release_type, timestamp, version)
            update_notebook_examples_markdown(release_type, timestamp, version)
            update_codejson(release_type, timestamp, version)
            update_PyPI_release(release_type, timestamp, version)
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
        help="Indicate release is approved (defaults to false for preliminary/development distributions)",
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
            release_type=ReleaseType.APPROVED
            if args.approve
            else ReleaseType.CANDIDATE,
            timestamp=datetime.now(),
            version=Version.from_string(args.version)
            if args.version
            else _current_version,
        )
