import re
from importlib import metadata
from io import BytesIO, StringIO
from pathlib import Path
from platform import system
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.misc import is_in_ci

# import modflow-devtools fixtures

pytest_plugins = ["modflow_devtools.fixtures"]


# constants

SHAPEFILE_EXTENSIONS = ["prj", "shx", "dbf"]


# path utilities


def get_project_root_path() -> Path:
    return Path(__file__).parent.parent


def get_example_data_path() -> Path:
    return get_project_root_path() / "examples" / "data"


def get_flopy_data_path() -> Path:
    return get_project_root_path() / "flopy" / "data"


# fixtures


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


@pytest.fixture(autouse=True)
def close_plot(request):
    # fixture to automatically close any plots (or optionally show them)
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

    # for test_generate_classes.py
    parser.addoption(
        "--ref",
        action="append",
        type=str,
        help="Include extra refs to test. Useful for testing branches on a fork, e.g. <your GitHub username>/modflow6/<your branch>.",
    )


def pytest_report_header(config):
    """Header for pytest to show versions of packages."""

    required = []
    extra = {}
    for item in metadata.requires("flopy"):
        pkg_name = re.findall(r"[a-z0-9_\-]+", item, re.IGNORECASE)[0]
        if res := re.findall("extra == ['\"](.+)['\"]", item):
            assert len(res) == 1, item
            pkg_extra = res[0]
            if pkg_extra not in extra:
                extra[pkg_extra] = []
            extra[pkg_extra].append(pkg_name)
        else:
            required.append(pkg_name)

    processed = set()
    lines = []
    items = []
    for name in required:
        processed.add(name)
        try:
            version = metadata.version(name)
            items.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))
    installed = []
    not_found = []
    for name in extra["optional"]:
        if name in processed:
            continue
        processed.add(name)
        try:
            version = metadata.version(name)
            installed.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            not_found.append(name)
    if installed:
        lines.append("optional packages: " + ", ".join(installed))
    if not_found:
        lines.append("optional packages not found: " + ", ".join(not_found))
    return "\n".join(lines)


# snapshot classes and fixtures

from syrupy.extensions.single_file import (
    SingleFileSnapshotExtension,
    WriteMode,
)
from syrupy.types import (
    PropertyFilter,
    PropertyMatcher,
    SerializableData,
    SerializedData,
)


def _serialize_bytes(data):
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


class BinaryArrayExtension(SingleFileSnapshotExtension):
    """
    Binary snapshot of a NumPy array. Can be read back into NumPy with
    .load(), preserving dtype and shape. This is the recommended array
    snapshot approach if human-readability is not a necessity, as disk
    space is minimized.
    """

    _write_mode = WriteMode.BINARY
    _file_extension = "npy"

    def serialize(
        self,
        data,
        *,
        exclude=None,
        include=None,
        matcher=None,
    ):
        return _serialize_bytes(data)


class TextArrayExtension(SingleFileSnapshotExtension):
    """
    Text snapshot of a NumPy array. Flattens the array before writing.
    Can be read back into NumPy with .loadtxt() assuming you know the
    shape of the expected data and subsequently reshape it if needed.
    """

    _write_mode = WriteMode.TEXT
    _file_extension = "txt"

    def serialize(
        self,
        data: "SerializableData",
        *,
        exclude: Optional["PropertyFilter"] = None,
        include: Optional["PropertyFilter"] = None,
        matcher: Optional["PropertyMatcher"] = None,
    ) -> "SerializedData":
        buffer = StringIO()
        np.savetxt(buffer, data.ravel())
        return buffer.getvalue()


class ReadableArrayExtension(SingleFileSnapshotExtension):
    """
    Human-readable snapshot of a NumPy array. Preserves array shape
    at the expense of possible loss of precision (default 8 places)
    and more difficulty loading into NumPy than TextArrayExtension.
    """

    _write_mode = WriteMode.TEXT
    _file_extension = "txt"

    def serialize(
        self,
        data: "SerializableData",
        *,
        exclude: Optional["PropertyFilter"] = None,
        include: Optional["PropertyFilter"] = None,
        matcher: Optional["PropertyMatcher"] = None,
    ) -> "SerializedData":
        return np.array2string(data, threshold=np.inf)


@pytest.fixture
def array_snapshot(snapshot):
    return snapshot.use_extension(BinaryArrayExtension)


@pytest.fixture
def text_array_snapshot(snapshot):
    return snapshot.use_extension(TextArrayExtension)


@pytest.fixture
def readable_array_snapshot(snapshot):
    return snapshot.use_extension(ReadableArrayExtension)
