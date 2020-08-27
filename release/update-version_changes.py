import os
import sys
from subprocess import Popen, PIPE, STDOUT

PY3 = sys.version_info[0] >= 3


def process_Popen(cmdlist, verbose=False):
    """Generic function to initialize a Popen process.

    Parameters
    ----------
    cmdlist : list
        command list passed to Popen

    Returns
    -------
    proc : Popen
        Popen instance

    """
    process = Popen(cmdlist, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = process.communicate()

    if stdout:
        if PY3:
            stdout = stdout.decode()
        if verbose:
            print(stdout)
    if stderr:
        if PY3:
            stderr = stderr.decode()
        if verbose:
            print(stderr)

    # catch non-zero return code
    if process.returncode != 0:
        errmsg = "{} failed\n".format(
            " ".join(process.args)
        ) + "\tstatus code {}\n".format(process.returncode)
        raise ChildProcessError(errmsg)

    return stdout


def get_version():
    major = 0
    minor = 0
    micro = 0

    # read existing version file
    fpth = os.path.join("..", "flopy", "version.py")
    f = open(fpth, "r")
    lines = [line.rstrip("\n") for line in open(fpth, "r")]
    for idx, line in enumerate(lines):
        t = line.split()
        if "major =" in line:
            major = int(t[2])
        elif "minor =" in line:
            minor = int(t[2])
        elif "micro =" in line:
            micro = int(t[2])

    f.close()

    return "{:d}.{:d}.{:d}".format(major, minor, micro)


def get_branch():
    branch = None

    # determine if branch defined on command line
    for argv in sys.argv:
        if "--master" in argv:
            branch = "master"
        elif "--develop" in argv.lower():
            branch = "develop"

    # use git to determine branch
    if branch is None:
        cmdlist = ("git", "status")
        stdout = process_Popen(cmdlist)
        for line in stdout.splitlines():
            if "On branch" in line:
                branch = line.replace("On branch ", "").rstrip()
                break
    return branch


def get_last_tag_date():
    cmdlist = (
        "git",
        "log",
        "--tags",
        "--no-walk",
        "--date=iso-local",
        "--pretty='%cd %D'",
    )
    stdout = process_Popen(cmdlist)
    line = stdout.splitlines()[0]
    ipos = line.find("tag")
    if ipos > -1:
        tag_date = line[1:20]
        tag = line[ipos + 4 :].split(",")[0].strip()

    return tag, tag_date


def get_hash_dict(branch):
    tag, tag_date = get_last_tag_date()

    # get hash and
    fmt = '--pretty="%H"'
    since = '--since="{}"'.format(tag_date)
    hash_dict = {"fix": {}, "feat": {}}
    cmdlist = ("git", "log", branch, fmt, since)
    stdout = process_Popen(cmdlist)

    fix_dict = {}
    feat_dict = {}

    fix_tags = ["fix", "bug"]
    feat_tags = ["feat"]
    # parse stdout
    for line in stdout.splitlines():
        hash = line.split()[0].replace('"', "")
        url = "https://github.com/modflowpy/flopy/commit/{}".format(hash)
        fmt = '--pretty="%s."'
        cmdlist = ("git", "log", fmt, "-n1", hash)
        subject = process_Popen(cmdlist).strip().replace('"', "")
        fmt = '--pretty="Committed by %aN on %cd."'
        cmdlist = ("git", "log", "--date=short", fmt, "-n1", hash)
        cdate = process_Popen(cmdlist).strip().replace('"', "")
        ipos = subject.find(":")
        key = None
        if ipos > -1:
            type = subject[0:ipos]
            subject = subject.replace(type + ":", "").strip().capitalize()
            for tag in fix_tags:
                if type.lower().startswith(tag):
                    key = "fix"
                    type = key + type[3:]
                    break
            if key is None:
                for tag in feat_tags:
                    if type.lower().startswith(tag):
                        key = "feat"
                        break
        else:
            type = None
            slower = subject.lower()
            for tag in fix_tags:
                if slower.startswith(tag):
                    key = "fix"
                    type = "fix()"
                    break
            if key is None:
                for tag in feat_tags:
                    if slower.startswith(tag):
                        key = "feat"
                        type = "feat()"
                        break

        if key is not None:
            message = "[{}]({}): ".format(type, url)
            message += subject + " " + cdate
            if key == "fix":
                fix_dict[hash] = message
            elif key == "feat":
                feat_dict[hash] = message

    # add dictionaries to the hash dictionary
    hash_dict["fix"] = fix_dict
    hash_dict["feat"] = feat_dict

    return hash_dict


def create_md(hash_dict):
    # get current version information
    version = get_version()
    tag = "### Version"
    version_text = "{} {}".format(tag, version)
    #

    # read the lines in the existing version_changes.md
    fpth = os.path.join("..", "docs", "version_changes.md")
    with open(fpth) as f:
        lines = f.read().splitlines()

    # rewrite the existing version_changes.md
    f = open(fpth, "w")

    write_line = True
    write_update = True
    for line in lines:
        if line.startswith(tag):
            write_line = True
            if version_text in line:
                write_line = False

            # write the changes for the latest comment
            if write_update:
                write_update = False
                f.write("{}\n\n".format(version_text))
                write_version_changes(f, hash_dict)

        if write_line:
            f.write("{}\n".format(line))

    f.close()


def write_version_changes(f, hash_dict):
    # features
    istart = True
    for key, val in hash_dict["feat"].items():
        if istart:
            istart = False
            f.write("* New features:\n\n")
        f.write(4 * " " + "* " + val + "\n")
    if not istart:
        f.write("\n\n")

    # bug fixes
    istart = True
    for key, val in hash_dict["fix"].items():
        if istart:
            istart = False
            f.write("* Bug fixes:\n\n")
        f.write(4 * " " + "* " + val + "\n")
    if not istart:
        f.write("\n")

    return


def main():
    branch = get_branch()
    hash_dict = get_hash_dict(branch)

    create_md(hash_dict)
    return


if __name__ == "__main__":
    main()
