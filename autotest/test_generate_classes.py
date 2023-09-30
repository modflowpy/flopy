import sys
from pathlib import Path
from pprint import pprint

import pytest
from modflow_devtools.misc import get_current_branch

branch = get_current_branch()


@pytest.mark.mf6
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.skipif(
    branch == "master" or branch.startswith("v"),
    reason="skip on master and release branches",
)
def test_generate_classes_from_github_refs(
    request, virtualenv, project_root_path, ref, worker_id
):
    argv = (
        request.config.workerinput["mainargv"]
        if hasattr(request.config, "workerinput")
        else []
    )
    if worker_id != "master" and "loadfile" not in argv:
        pytest.skip("can't run in parallel")

    python = virtualenv.python
    venv = Path(python).parent
    print(f"Using temp venv at {venv} with python {python}")

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

    # generate classes from develop branch
    owner = "MODFLOW-USGS"
    branch = "develop"
    pprint(
        virtualenv.run(
            "python -c 'from flopy.mf6.utils import generate_classes; generate_classes(owner=\""
            + owner
            + '", branch="'
            + branch
            + "\", backup=False)'"
        )
    )

    # make sure files were regenerated
    modified_files = [
        mod_files[i]
        for i, (before, after) in enumerate(
            zip(
                mod_file_times,
                [Path(mod_file).stat().st_mtime for mod_file in mod_files],
            )
        )
        if after > before
    ]
    assert any(modified_files)
    print(f"{len(modified_files)} files were modified:")
    pprint(modified_files)

    # try with master branch
    branch = "master"
    pprint(
        virtualenv.run(
            "python -c 'from flopy.mf6.utils import generate_classes; generate_classes(owner=\""
            + owner
            + '", branch="'
            + branch
            + "\", backup=False)'"
        )
    )

    # todo checkout mf6 and test with dfnpath? test with backups?
