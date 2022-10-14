import inspect
import os
import platform
from pathlib import Path
from shutil import which

import pytest
from _pytest.config import ExitCode
from autotest.conftest import (
    excludes_platform,
    get_example_data_path,
    get_project_root_path,
    requires_exe,
    requires_pkg,
    requires_platform,
)

# temporary directory fixtures


def test_tmpdirs(tmpdir, module_tmpdir):
    # function-scoped temporary directory
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()
    assert inspect.currentframe().f_code.co_name in tmpdir.stem

    # module-scoped temp dir (accessible to other tests in the script)
    assert module_tmpdir.is_dir()
    assert "autotest" in module_tmpdir.stem


def test_function_scoped_tmpdir(tmpdir):
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()
    assert inspect.currentframe().f_code.co_name in tmpdir.stem


@pytest.mark.parametrize("name", ["noslash", "forward/slash", "back\\slash"])
def test_function_scoped_tmpdir_slash_in_name(tmpdir, name):
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()

    # node name might have slashes if test function is parametrized
    # (e.g., test_function_scoped_tmpdir_slash_in_name[a/slash])
    replaced1 = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    replaced2 = name.replace("/", "_").replace("\\", "__").replace(":", "_")
    assert (
        f"{inspect.currentframe().f_code.co_name}[{replaced1}]" in tmpdir.stem
        or f"{inspect.currentframe().f_code.co_name}[{replaced2}]"
        in tmpdir.stem
    )


class TestClassScopedTmpdir:
    filename = "hello.txt"

    @pytest.fixture(autouse=True)
    def setup(self, class_tmpdir):
        with open(class_tmpdir / self.filename, "w") as file:
            file.write("hello, class-scoped tmpdir")

    def test_class_scoped_tmpdir(self, class_tmpdir):
        assert isinstance(class_tmpdir, Path)
        assert class_tmpdir.is_dir()
        assert self.__class__.__name__ in class_tmpdir.stem
        assert Path(class_tmpdir / self.filename).is_file()


def test_module_scoped_tmpdir(module_tmpdir):
    assert isinstance(module_tmpdir, Path)
    assert module_tmpdir.is_dir()
    assert Path(inspect.getmodulename(__file__)).stem in module_tmpdir.name


def test_session_scoped_tmpdir(session_tmpdir):
    assert isinstance(session_tmpdir, Path)
    assert session_tmpdir.is_dir()


# misc utilities


def test_get_project_root_path():
    root = get_project_root_path()

    assert root.is_dir()

    contents = [p.name for p in root.glob("*")]
    assert (
        "autotest" in contents
        and "README.md" in contents
    )


def test_get_paths():
    example_data = get_example_data_path()
    project_root = get_project_root_path()

    assert example_data.parent.parent == project_root


def test_get_example_data_path():
    parts = get_example_data_path().parts
    assert (
        parts[-3] == "flopy"
        and parts[-2] == "examples"
        and parts[-1] == "data"
    )


# requiring/excluding executables & platforms


@requires_exe("mf6")
def test_mf6():
    assert which("mf6")


exes = ["mfusg", "mfnwt"]


@requires_exe(*exes)
def test_mfusg_and_mfnwt():
    assert all(which(exe) for exe in exes)


@requires_pkg("numpy")
def test_numpy():
    import numpy

    assert numpy is not None


@requires_pkg("numpy", "matplotlib")
def test_numpy_and_matplotlib():
    import matplotlib
    import numpy

    assert numpy is not None and matplotlib is not None


@requires_platform("Windows")
def test_needs_windows():
    assert platform.system() == "Windows"


@excludes_platform("Darwin", ci_only=True)
def test_breaks_osx_ci():
    if "CI" in os.environ:
        assert platform.system() != "Darwin"


# meta-test marker and CLI argument --meta (-M)


@pytest.mark.meta("test_meta")
def test_meta_inner():
    pass


class TestMeta:
    def pytest_terminal_summary(self, terminalreporter):
        stats = terminalreporter.stats
        assert "failed" not in stats

        passed = [test.head_line for test in stats["passed"]]
        assert len(passed) == 1
        assert test_meta_inner.__name__ in passed

        deselected = [fn.name for fn in stats["deselected"]]
        assert len(deselected) > 0


def test_meta():
    args = [
        f"{__file__}",
        "-v",
        "-s",
        "-k",
        test_meta_inner.__name__,
        "-M",
        "test_meta",
    ]
    assert pytest.main(args, plugins=[TestMeta()]) == ExitCode.OK


# CLI arguments --keep (-K) and --keep-failed

FILE_NAME = "hello.txt"


@pytest.mark.meta("test_keep")
def test_keep_function_scoped_tmpdir_inner(tmpdir):
    with open(tmpdir / FILE_NAME, "w") as f:
        f.write("hello, function-scoped tmpdir")


@pytest.mark.meta("test_keep")
class TestKeepClassScopedTmpdirInner:
    def test_keep_class_scoped_tmpdir_inner(self, class_tmpdir):
        with open(class_tmpdir / FILE_NAME, "w") as f:
            f.write("hello, class-scoped tmpdir")


@pytest.mark.meta("test_keep")
def test_keep_module_scoped_tmpdir_inner(module_tmpdir):
    with open(module_tmpdir / FILE_NAME, "w") as f:
        f.write("hello, module-scoped tmpdir")


@pytest.mark.meta("test_keep")
def test_keep_session_scoped_tmpdir_inner(session_tmpdir):
    with open(session_tmpdir / FILE_NAME, "w") as f:
        f.write("hello, session-scoped tmpdir")


@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_function_scoped_tmpdir(tmpdir, arg):
    inner_fn = test_keep_function_scoped_tmpdir_inner.__name__
    args = [
        __file__,
        "-v",
        "-s",
        "-k",
        inner_fn,
        "-M",
        "test_keep",
        "-K",
        tmpdir,
    ]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{inner_fn}0" / FILE_NAME).is_file()


@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_class_scoped_tmpdir(tmpdir, arg):
    args = [
        __file__,
        "-v",
        "-s",
        "-k",
        TestKeepClassScopedTmpdirInner.test_keep_class_scoped_tmpdir_inner.__name__,
        "-M",
        "test_keep",
        "-K",
        tmpdir,
    ]
    assert pytest.main(args) == ExitCode.OK
    assert Path(
        tmpdir / f"{TestKeepClassScopedTmpdirInner.__name__}0" / FILE_NAME
    ).is_file()


@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_module_scoped_tmpdir(tmpdir, arg):
    args = [
        __file__,
        "-v",
        "-s",
        "-k",
        test_keep_module_scoped_tmpdir_inner.__name__,
        "-M",
        "test_keep",
        "-K",
        tmpdir,
    ]
    assert pytest.main(args) == ExitCode.OK
    this_file_path = Path(__file__)
    this_test_dir = (
        tmpdir
        / f"{str(this_file_path.parent.name)}.{str(this_file_path.stem)}0"
    )
    assert FILE_NAME in [f.name for f in this_test_dir.glob("*")]


@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_session_scoped_tmpdir(tmpdir, arg, request):
    args = [
        __file__,
        "-v",
        "-s",
        "-k",
        test_keep_session_scoped_tmpdir_inner.__name__,
        "-M",
        "test_keep",
        "-K",
        tmpdir,
    ]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{request.session.name}0" / FILE_NAME).is_file()


@pytest.mark.meta("test_keep_failed")
def test_keep_failed_function_scoped_tmpdir_inner(tmpdir):
    with open(tmpdir / FILE_NAME, "w") as f:
        f.write("hello, function-scoped tmpdir")

    assert False, "oh no"


@pytest.mark.parametrize("keep", [True, False])
def test_keep_failed_function_scoped_tmpdir(tmpdir, keep):
    inner_fn = test_keep_failed_function_scoped_tmpdir_inner.__name__
    args = [__file__, "-v", "-s", "-k", inner_fn, "-M", "test_keep_failed"]
    if keep:
        args += ["--keep-failed", tmpdir]
    assert pytest.main(args) == ExitCode.TESTS_FAILED

    kept_file = Path(tmpdir / f"{inner_fn}0" / FILE_NAME).is_file()
    assert kept_file if keep else not kept_file
