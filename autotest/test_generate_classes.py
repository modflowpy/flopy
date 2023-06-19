from pathlib import Path
from pprint import pprint

import pytest


@pytest.mark.mf6
def test_generate_classes_from_dfn(virtualenv, project_root_path):
    python = virtualenv.python
    venv = Path(python).parent
    print(f"Using temp venv at {venv} with python {python}")

    # install flopy/dependencies
    pprint(virtualenv.run(f"pip install {project_root_path}"))
    for dependency in ["modflow-devtools"]:
        pprint(virtualenv.run(f"pip install {dependency}"))

    # get creation time of files
    mod_files = list(
        venv.parent.rglob("lib/python3.10/site-packages/flopy/mf6/modflow/*")
    ) + list(
        venv.parent.rglob("lib/python3.10/site-packages/flopy/mf6/data/dfn/*")
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
