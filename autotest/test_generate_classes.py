import sys
from os import environ
from pathlib import Path
from platform import system
from pprint import pprint
from typing import Iterable
from warnings import warn

import pytest
from modflow_devtools.misc import get_current_branch, run_cmd
from virtualenv import cli_run

branch = get_current_branch()


def nonempty(itr: Iterable):
    for x in itr:
        if x:
            yield x


def pytest_generate_tests(metafunc):
    """
    Test mf6 module code generation on a small, hopefully
    fairly representative set of MODFLOW 6 input & output
    specification versions, including the develop branch,
    the latest official release, and a few older releases
    and commits.

    TODO: May make sense to run the full battery of tests
    against all of the versions of mf6io flopy guarantees
    support for- maybe develop and latest release? Though
    some backwards compatibility seems ideal if possible.
    """

    owner = "MODFLOW-USGS"
    repo = "modflow6"
    ref = [
        f"{owner}/{repo}/develop",
        f"{owner}/{repo}/master",
        f"{owner}/{repo}/6.4.1",
        f"{owner}/{repo}/4458f9f",
        f"{owner}/{repo}/4458f9f7a6244182e6acc2430a6996f9ca2df367",
    ]

    # refs provided as env vars override the defaults
    ref_env = environ.get("TEST_GENERATE_CLASSES_REF")
    if ref_env:
        ref = nonempty(ref_env.strip().split(","))

    # refs given as CLI options override everything
    ref_opt = metafunc.config.getoption("--ref")
    if ref_opt:
        ref = nonempty([o.strip() for o in ref_opt])

    # drop duplicates
    ref = list(dict.fromkeys(ref))

    # drop and warn refs with invalid format
    # i.e. not "owner/repo/branch"
    for r in ref:
        spl = r.split("/")
        if len(spl) != 3 or not all(spl):
            warn(f"Skipping invalid ref: {r}")
            ref.remove(r)

    key = "ref"
    if key in metafunc.fixturenames:
        metafunc.parametrize(key, ref, scope="session")


@pytest.mark.generation
@pytest.mark.mf6
@pytest.mark.slow
def test_generate_classes_from_github_refs(
    request, project_root_path, ref, worker_id, function_tmpdir
):
    # skip if run in parallel with pytest-xdist without --dist loadfile
    argv = (
        request.config.workerinput["mainargv"]
        if hasattr(request.config, "workerinput")
        else []
    )
    if worker_id != "master" and "loadfile" not in argv:
        pytest.skip("can't run in parallel")

    # create virtual environment
    venv = function_tmpdir / "venv"
    win = system() == "Windows"
    bin = "Scripts" if win else "bin"
    python = venv / bin / ("python" + (".exe" if win else ""))
    pip = venv / bin / ("pip" + (".exe" if win else ""))
    cli_run([str(venv)])
    print(f"Using temp venv at {venv} to test class generation from {ref}")

    # install flopy and dependencies
    deps = [str(project_root_path), "modflow-devtools"]
    for dep in deps:
        out, err, ret = run_cmd(str(pip), "install", dep, verbose=True)
        assert not ret, out + err

    # get creation time of files
    flopy_path = (
        (venv / "Lib" / "site-packages" / "flopy")
        if win
        else (
            venv
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "flopy"
        )
    )
    assert flopy_path.is_dir()
    mod_files = list((flopy_path / "mf6" / "modflow").rglob("*")) + list(
        (flopy_path / "mf6" / "data" / "dfn").rglob("*")
    )
    mod_file_times = [Path(mod_file).stat().st_mtime for mod_file in mod_files]
    pprint(mod_files)

    # split ref into owner, repo, ref name
    spl = ref.split("/")
    owner = spl[0]
    repo = spl[1]
    ref = spl[2]

    # generate classes
    out, err, ret = run_cmd(
        str(python),
        "-m",
        "flopy.mf6.utils.generate_classes",
        "--owner",
        owner,
        "--repo",
        repo,
        "--ref",
        ref,
        "--no-backup",
        verbose=True,
    )
    assert not ret, out + err

    def get_mtime(f):
        try:
            return Path(f).stat().st_mtime
        except:
            return 0  # if file not found

    # make sure files were regenerated
    modified_files = [
        mod_files[i]
        for i, (before, after) in enumerate(
            zip(
                mod_file_times,
                [get_mtime(f) for f in mod_files],
            )
        )
        if after > 0 and after > before
    ]
    assert any(modified_files)
    print(f"{len(modified_files)} files were modified:")
    pprint(modified_files)

    # todo checkout mf6 and test with dfnpath? test with backups?
