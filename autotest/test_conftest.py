import inspect
import os
import platform
from pathlib import Path
from shutil import which

import pytest
from _pytest.config import ExitCode
from autotest.conftest import get_example_data_path, get_project_root_path
from modflow_devtools.markers import (
    excludes_platform,
    requires_exe,
    requires_pkg,
    requires_platform,
)


def test_get_project_root_path():
    root = get_project_root_path()

    assert root.is_dir()

    contents = [p.name for p in root.glob("*")]
    assert "autotest" in contents and "README.md" in contents


def test_get_example_data_path():
    parts = get_example_data_path().parts
    assert parts[-2:] == ("examples", "data")


def test_get_paths():
    example_data = get_example_data_path()
    project_root = get_project_root_path()

    assert example_data.parent.parent == project_root
