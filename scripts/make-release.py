#!/usr/bin/python

import datetime
import json
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader

# file_paths dictionary has file names and the path to the file. Enter '.'
# as the path if the file is in the root repository directory
file_paths = {
    "version.py": "../flopy",
    "README.md": "../",
    "PyPI_release.md": "../docs",
    "code.json": "../",
    "DISCLAIMER.md": "../flopy",
    "notebook_examples.md": "../docs",
}

pak = "flopy"

# local import of package variables in flopy/version.py
loader = SourceFileLoader("version", os.path.join("..", "flopy", "version.py"))
version_mod = loader.load_module()

# build authors list for Software/Code citation for FloPy
authors = []
for key in version_mod.author_dict.keys():
    t = key.split()
    author = f"{t[-1]}"
    for str in t[0:-1]:
        author += f" {str}"
    authors.append(author)

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


def get_version_str(v0, v1, v2):
    version_type = (f"{v0}", f"{v1}", f"{v2}")
    version = ".".join(version_type)
    return version


def get_tag(v0, v1, v2):
    tag_type = (f"{v0}", f"{v1}", f"{v2}")
    tag = ".".join(tag_type)
    return tag


def get_software_citation(version, is_approved):
    now = datetime.datetime.now()
    sb = ""
    if not is_approved:
        sb = " &mdash; release candidate"
    # format author names
    line = "["
    for ipos, author in enumerate(authors):
        if ipos > 0:
            line += ", "
        if ipos == len(authors) - 1:
            line += "and "
        sv = author.split()
        tauthor = f"{sv[0]}"
        if len(sv) < 3:
            gname = sv[1]
            if len(gname) > 1:
                tauthor += f", {gname}"
            else:
                tauthor += f", {gname[0]}."
        else:
            tauthor += f", {sv[1][0]}. {sv[2][0]}."
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
    name_pos = None
    try:
        file = "version.py"
        fpth = os.path.join(file_paths[file], file)

        vmajor = 0
        vminor = 0
        vmicro = 0
        lines = [line.rstrip("\n") for line in open(fpth, "r")]
        for idx, line in enumerate(lines):
            t = line.split()
            if "major =" in line:
                vmajor = int(t[2])
            elif "minor =" in line:
                vminor = int(t[2])
            elif "micro =" in line:
                vmicro = int(t[2])
            elif "__version__" in line:
                name_pos = idx + 1

    except:
        raise OSError("There was a problem updating the version file")

    try:
        # write new version file
        f = open(fpth, "w")
        f.write(
            (
                f"# {pak} version file automatically created "
                f"using...{os.path.basename(__file__)}\n"
            )
        )
        f.write(
            f"# created on...{datetime.datetime.now():%B %d, %Y %H:%M:%S}\n"
        )
        f.write("\n")
        f.write(f"major = {vmajor}\n")
        f.write(f"minor = {vminor}\n")
        f.write(f"micro = {vmicro}\n")
        f.write('__version__ = f"{major}.{minor}.{micro}"\n')

        # write the remainder of the version file
        if name_pos is not None:
            for line in lines[name_pos:]:
                f.write(f"{line}\n")
        f.close()
        print("Successfully updated version.py")
    except:
        raise OSError("There was a problem updating the version file")

    # update README.md with new version information
    update_readme_markdown(vmajor, vminor, vmicro)

    # update notebook_examples.md
    update_notebook_examples_markdown()

    # update code.json
    update_codejson(vmajor, vminor, vmicro)

    # update PyPI_release.md
    update_PyPI_release(vmajor, vminor, vmicro)


def update_codejson(vmajor, vminor, vmicro):
    # define json filename
    file = "code.json"
    json_fname = os.path.join(file_paths[file], file)

    # get branch
    branch = get_branch()

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # load and modify json file
    with open(json_fname, "r") as f:
        data = json.load(f)

    # modify the json file data
    now = datetime.datetime.now()
    sdate = now.strftime("%Y-%m-%d")
    data[0]["date"]["metadataLastUpdated"] = sdate
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
    file = "README.md"
    fpth = os.path.join(file_paths[file], file)
    with open(fpth, "r") as file:
        lines = [line.rstrip() for line in file]

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
    file = "DISCLAIMER.md"
    fpth = os.path.join(file_paths[file], file)
    f = open(fpth, "w")
    f.write(disclaimer)
    f.close()

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
    file = "notebook_examples.md"
    fpth = os.path.join(file_paths[file], file)
    with open(fpth, "r") as file:
        lines = [line.rstrip() for line in file]

    # rewrite notebook_examples.md
    terminate = False
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

    # read README.md into memory
    file = "PyPI_release.md"
    fpth = os.path.join(file_paths[file], file)
    with open(fpth, "r") as file:
        lines = [line.rstrip() for line in file]

    # rewrite README.md
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
