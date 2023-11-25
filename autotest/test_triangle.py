import shutil
from os import environ
from platform import system

import pytest
from modflow_devtools.markers import requires_exe
from modflow_devtools.misc import set_env

import flopy
from flopy.utils.triangle import Triangle


@requires_exe("mf6")
def test_output_files_not_found(function_tmpdir):
    tri = Triangle(model_ws=function_tmpdir, maximum_area=1.0, angle=30)

    # if expected output files are not found,
    # Triangle should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        tri._load_results()
