import sys
from datetime import datetime
from os import environ
from pathlib import Path
from pprint import pprint
from typing import Iterable
from warnings import warn

import pytest


def nonempty(itr: Iterable):
    for x in itr:
        if x:
            yield x


def pytest_generate_tests(metafunc):
    # defaults
    ref = [
        "MODFLOW-USGS/modflow6/develop",
        "MODFLOW-USGS/modflow6/master",
        "MODFLOW-USGS/modflow6/6.4.1",
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
    ref = list(set(ref))

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


@pytest.mark.mf6
@pytest.mark.slow
def test_generate_classes_from_dfn(virtualenv, project_root_path, ref):
    python = virtualenv.python
    venv = Path(python).parent
    print(
        f"Using temp venv at {venv} with python {python} to test class generation from {ref}"
    )

    # install flopy/dependencies
    pprint(virtualenv.run(f"pip install {project_root_path}"))
    for dependency in ["modflow-devtools"]:
        pprint(virtualenv.run(f"pip install {dependency}"))

    # get creation time of files
    flopy_path = (
        venv.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "flopy"
    )
    assert flopy_path.is_dir()
    mod_files = list((flopy_path / "mf6" / "modflow").rglob("*")) + list(
        (flopy_path / "mf6" / "data" / "dfn").rglob("*")
    )
    mod_file_times = [Path(mod_file).stat().st_mtime for mod_file in mod_files]
    pprint(mod_files)

    # generate classes
    spl = ref.split("/")
    owner = spl[0]
    branch = spl[2]
    pprint(
        virtualenv.run(
            "python -c 'from flopy.mf6.utils import generate_classes; generate_classes(owner=\""
            + owner
            + '", branch="'
            + branch
            + "\", backup=False)'"
        )
    )

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
