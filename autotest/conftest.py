import importlib
import os
import socket
import sys
from os import environ
from os.path import basename, normpath
from pathlib import Path
from platform import system
from shutil import copytree, which
from subprocess import PIPE, Popen
from typing import List, Optional
from urllib import request
from warnings import warn

import matplotlib.pyplot as plt
import pkg_resources
import pytest
from modflow_devtools.misc import is_in_ci

# import modflow-devtools fixtures

pytest_plugins = ["modflow_devtools.fixtures"]


# constants

SHAPEFILE_EXTENSIONS = ["prj", "shx", "dbf"]


# misc utilities


def get_project_root_path() -> Path:
    return Path(__file__).parent.parent


def get_example_data_path() -> Path:
    return get_project_root_path() / "examples" / "data"


def get_flopy_data_path() -> Path:
    return get_project_root_path() / "flopy" / "data"


# path fixtures


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    return get_project_root_path()


@pytest.fixture(scope="session")
def example_data_path() -> Path:
    return get_example_data_path()


@pytest.fixture(scope="session")
def flopy_data_path() -> Path:
    return get_flopy_data_path()


@pytest.fixture(scope="session")
def example_shapefiles(example_data_path) -> List[Path]:
    return [f.resolve() for f in (example_data_path / "prj_test").glob("*")]


# fixture to automatically close any plots (or optionally show them)


@pytest.fixture(autouse=True)
def close_plot(request):
    yield

    # plots only shown if requested via CLI flag,
    # figures are available, and we're not in CI
    show = request.config.getoption("--show-plots")
    if len(plt.get_fignums()) > 0 and not is_in_ci() and show:
        plt.show()
    else:
        plt.close("all")


@pytest.fixture(scope="session", autouse=True)
def patch_macos_ci_matplotlib():
    # use noninteractive matplotlib backend if in Mac OS CI to avoid pytest-xdist node failure
    # e.g. https://github.com/modflowpy/flopy/runs/7748574375?check_suite_focus=true#step:9:57
    if is_in_ci() and system().lower() == "darwin":
        import matplotlib

        matplotlib.use("agg")


# pytest configuration hooks


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # this is necessary so temp dir fixtures can
    # inspect test results and check for failure
    # (see https://doc.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures)

    outcome = yield
    rep = outcome.get_result()

    # report attribute for each phase (setup, call, teardown)
    # we're only interested in result of the function call
    setattr(item, "rep_" + rep.when, rep)


def pytest_addoption(parser):
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Show any figure windows created by test cases. (Useful to display plots for visual inspection, "
        "but automated tests should probably also check patch collections or figure & axis properties.)",
    )


def pytest_report_header(config):
    """Header for pytest to show versions of packages."""

    # if we ever drop support for python 3.7, could use importlib.metadata instead?
    # or importlib_metadata backport: https://importlib-metadata.readthedocs.io/en/latest/
    # pkg_resources discouraged: https://setuptools.pypa.io/en/latest/pkg_resources.html

    processed = set()
    flopy_pkg = pkg_resources.get_distribution("flopy")
    lines = []
    items = []
    for pkg in flopy_pkg.requires():
        name = pkg.name
        processed.add(name)
        try:
            version = pkg_resources.get_distribution(name).version
            items.append(f"{name}-{version}")
        except pkg_resources.DistributionNotFound:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))
    installed = []
    not_found = []
    for pkg in flopy_pkg.requires(["optional"]):
        name = pkg.name
        if name in processed:
            continue
        processed.add(name)
        try:
            version = pkg_resources.get_distribution(name).version
            installed.append(f"{name}-{version}")
        except pkg_resources.DistributionNotFound:
            not_found.append(name)
    if installed:
        lines.append("optional packages: " + ", ".join(installed))
    if not_found:
        lines.append("optional packages not found: " + ", ".join(not_found))
    return "\n".join(lines)
