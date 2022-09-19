#!/usr/bin/python

import datetime
import json
import subprocess
import sys
import yaml
from pathlib import Path
from textwrap import dedent

# file_paths_list has file names and the path to the file relative to
# the repo root directory. The dictionary file_paths has keys for each file.
repo_root = Path(__file__).parent.parent.resolve()
file_paths_list = [
    repo_root / "CITATION.cff",
    repo_root / "code.json",
    repo_root / "README.md",
    repo_root / "docs" / "notebook_examples.md",
    repo_root / "docs" / "PyPI_release.md",
    repo_root / "flopy" / "version.py",
    repo_root / "flopy" / "DISCLAIMER.md",
]
file_paths = {pth.name: pth for pth in file_paths_list}

now = datetime.datetime.now()

pak = "flopy"

approved = """Disclaimer
----------

This software has been approved for release by the U.S. Geological Survey
(USGS). Although the software has been subjected to rigorous review, the USGS
reserves the right to update the software as needed pursuant to further analysis
and review. No warranty, expressed or implied, is made by the USGS or the U.S.
Government as to the functionality of the software and related material nor
shall the fact of release constitute any such warranty. Furthermore, the
software is released on condition that neither the USGS nor the U.S. Government
shall be held liable for any damages resulting from its authorized or
unauthorized use.
"""

preliminary = """Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.
"""


def get_disclaimer():
    # get current branch
    branch = get_branch()

    if branch.lower().startswith("release") or "master" in branch.lower():
        disclaimer = approved
        is_approved = True
    else:
        disclaimer = preliminary
        is_approved = False

    return is_approved, disclaimer


def get_branch():
    branch = None

    # determine if branch defined on command line
    for argv in sys.argv:
        if "master" in argv:
            branch = "master"
        elif "develop" in argv.lower():
            branch = "develop"

    if branch is None:
        try:
            # determine current branch
            b = subprocess.Popen(
                ("git", "status"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).communicate()[0]
            if isinstance(b, bytes):
                b = b.decode("utf-8")

            for line in b.splitlines():
                if "On branch" in line:
                    branch = line.replace("On branch ", "").rstrip()

        except:
            msg = "Could not determine current branch. Is git installed?"
            raise ValueError(msg)

    return branch


def get_tag(*v):
    return ".".join(str(vi) for vi in v)


def get_software_citation(version, is_approved):

    # get data Software/Code citation for FloPy
    citation = yaml.safe_load(file_paths["CITATION.cff"].read_text())

    sb = ""
    if not is_approved:
        sb = " &mdash; release candidate"
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
        f", {now.year}, FloPy v{version}{sb}: "
        f"U.S. Geological Survey Software Release, {now:%d %B %Y}, "
        "https://doi.org/10.5066/F7BK19FH]"
        "(https://doi.org/10.5066/F7BK19FH)"
    )

    return line


def update_version():
    fpth = file_paths["version.py"]
    lines = fpth.read_text().rstrip().split("\n")
    vmajor = 0
    vminor = 0
    vmicro = 0
    for idx, line in enumerate(lines):
        t = line.split()
        if "major =" in line:
            vmajor = int(t[2])
        elif "minor =" in line:
            vminor = int(t[2])
        elif "micro =" in line:
            vmicro = int(t[2])

    content = dedent(
        f"""\
    # {pak} version file automatically created using...{Path(__file__).name}
    # created on...{now:%B %d, %Y %H:%M:%S}

    major = {vmajor}
    minor = {vminor}
    micro = {vmicro}
    __version__ = f"{{major}}.{{minor}}.{{micro}}"
    """
    )
    fpth.write_text(content)

    print("Successfully updated version.py")

    # update README.md with new version information
    update_readme_markdown(vmajor, vminor, vmicro)

    # update CITATION.cff with new version information
    update_citation_cff(vmajor, vminor, vmicro)

    # update notebook_examples.md
    update_notebook_examples_markdown()

    # update code.json
    update_codejson(vmajor, vminor, vmicro)

    # update PyPI_release.md
    update_PyPI_release(vmajor, vminor, vmicro)


def update_codejson(vmajor, vminor, vmicro):
    # define json filename
    json_fname = file_paths["code.json"]

    # get branch
    branch = get_branch()

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # load and modify json file
    data = json.loads(json_fname.read_text())

    # modify the json file data
    data[0]["date"]["metadataLastUpdated"] = now.strftime("%Y-%m-%d")
    if branch.lower().startswith("release") or "master" in branch.lower():
        data[0]["version"] = version
        data[0]["status"] = "Production"
    else:
        data[0]["version"] = version
        data[0]["status"] = "Release Candidate"

    # rewrite the json file
    with open(json_fname, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    return


def update_readme_markdown(vmajor, vminor, vmicro):
    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # define branch
    if is_approved:
        branch = "master"
    else:
        branch = "develop"

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read README.md into memory
    fpth = file_paths["README.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite README.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "### Version " in line:
            line = f"### Version {version}"
            if not is_approved:
                line += " &mdash; release candidate"
        elif "[flopy continuous integration]" in line:
            line = (
                "[![flopy continuous integration](https://github.com/"
                "modflowpy/flopy/actions/workflows/commit.yml/badge.svg?"
                "branch={})](https://github.com/modflowpy/flopy/actions/"
                "workflows/commit.yml)".format(branch)
            )
        elif "[Read the Docs]" in line:
            line = (
                "[![Read the Docs](https://github.com/modflowpy/flopy/"
                "actions/workflows/rtd.yml/badge.svg?branch={})]"
                "(https://github.com/modflowpy/flopy/actions/"
                "workflows/rtd.yml)".format(branch)
            )
        elif "[Coverage Status]" in line:
            line = (
                "[![Coverage Status](https://coveralls.io/repos/github/"
                "modflowpy/flopy/badge.svg?branch={0})]"
                "(https://coveralls.io/github/modflowpy/"
                "flopy?branch={0})".format(branch)
            )
        elif "[Binder]" in line:
            # [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)
            line = (
                "[![Binder](https://mybinder.org/badge_logo.svg)]"
                "(https://mybinder.org/v2/gh/modflowpy/flopy.git/"
                "{})".format(branch)
            )
        elif "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(version, is_approved)
        elif "Disclaimer" in line:
            line = disclaimer
            terminate = True
        f.write(f"{line}\n")
        if terminate:
            break

    f.close()

    # write disclaimer markdown file
    file_paths["DISCLAIMER.md"].write_text(disclaimer)

    return


def update_citation_cff(vmajor, vminor, vmicro):
    # read CITATION.cff to modify
    fpth = file_paths["CITATION.cff"]
    citation = yaml.safe_load(fpth.read_text())

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # update version and date-released
    citation["version"] = version
    citation["date-released"] = now.strftime("%Y-%m-%d")

    # write CITATION.cff
    with open(fpth, "w") as f:
        yaml.safe_dump(
            citation,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    return


def update_notebook_examples_markdown():
    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # define branch
    if is_approved:
        branch = "master"
    else:
        branch = "develop"

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
                "(https://mybinder.org/v2/gh/modflowpy/flopy.git/"
                "{})".format(branch)
            )
        f.write(f"{line}\n")
    f.close()


def update_PyPI_release(vmajor, vminor, vmicro):
    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read PyPI_release.md into memory
    fpth = file_paths["PyPI_release.md"]
    lines = fpth.read_text().rstrip().split("\n")

    # rewrite PyPI_release.md
    terminate = False
    f = open(fpth, "w")
    for line in lines:
        if "doi.org/10.5066/F7BK19FH" in line:
            line = get_software_citation(version, is_approved)
        elif "Disclaimer" in line:
            line = disclaimer
            terminate = True
        f.write(f"{line}\n")
        if terminate:
            break

    f.close()

    return


if __name__ == "__main__":
    update_version()
    get_software_citation("3.1.1", True)
