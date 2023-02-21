import os
import platform
from os import getcwd
from os.path import relpath, splitdrive
from pathlib import Path
from shutil import which

import pytest
from modflow_devtools.markers import requires_exe
from modflow_devtools.misc import set_dir

from flopy.utils.flopy_io import line_parse, relpath_safe


def test_line_parse():
    """t027 test line_parse method in MNW2 Package class"""
    # ensure that line_parse is working correctly
    # comment handling
    line = line_parse("Well-A  -1                   ; 2a. WELLID,NNODES")
    assert line == ["Well-A", "-1"]


@requires_exe("mf6")
@pytest.mark.parametrize("scrub", [True, False])
def test_relpath_safe(function_tmpdir, scrub):
    if (
        platform.system() == "Windows"
        and splitdrive(function_tmpdir)[0] != splitdrive(getcwd())[0]
    ):
        assert Path(relpath_safe(function_tmpdir)) == function_tmpdir.absolute()
        assert relpath_safe(which("mf6")) == str(Path(which("mf6")).absolute())
    else:
        assert Path(
            relpath_safe(function_tmpdir, function_tmpdir.parent)
        ) == Path(function_tmpdir.name)
        assert (
            Path(relpath_safe(function_tmpdir, function_tmpdir.parent.parent))
            == Path(function_tmpdir.parent.name) / function_tmpdir.name
        )

        assert relpath_safe(which("mf6")) == relpath(which("mf6"), getcwd())

        # test user login obfuscation
        with set_dir("/"):
            try:
                login = os.getlogin()
                p = relpath_safe(Path.home(), scrub=scrub)
                if login in str(Path.home()) and scrub:
                    assert "***" in p
                    assert login not in p
            except OSError:
                # OSError is possible in CI, e.g. 'No such device or address'
                pass
