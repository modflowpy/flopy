import inspect
from pathlib import Path

import pytest
from _pytest.config import ExitCode


# temporary directory fixtures
from conftest import MF6SIMULATION_RELATIVE_PATHS, MODEL_RELATIVE_PATHS
from flopy.mf6 import MFSimulation
from flopy.modflow import Modflow


@pytest.mark.unit
def test_function_scoped_tmpdir(tmpdir):
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()
    assert inspect.currentframe().f_code.co_name in tmpdir.stem


@pytest.mark.unit
@pytest.mark.parametrize("name", ["noslash", "a/slash"])
def test_function_scoped_tmpdir_slash_in_name(tmpdir, name):
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()

    # node name might have slashes if test function is parametrized
    # (e.g., test_function_scoped_tmpdir_slash_in_name[a/slash])
    assert f"{inspect.currentframe().f_code.co_name}[{name.replace('/', '_')}]" in tmpdir.stem


@pytest.mark.unit
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


@pytest.mark.unit
def test_module_scoped_tmpdir(module_tmpdir):
    assert isinstance(module_tmpdir, Path)
    assert module_tmpdir.is_dir()
    assert Path(__file__).stem in module_tmpdir.stem


@pytest.mark.unit
def test_session_scoped_tmpdir(session_tmpdir):
    assert isinstance(session_tmpdir, Path)
    assert session_tmpdir.is_dir()


# example model/simulation fixtures


@pytest.mark.unit
def test_get_example_model(get_example_model):
    name = "mf2005_test"
    namfile = "bcf2ss.nam"
    model = get_example_model(name, namfile=namfile)
    assert isinstance(model, Modflow)
    assert model.name == Path(namfile).stem
    assert model.namefile == namfile


@pytest.mark.unit
@pytest.mark.parametrize("name", MF6SIMULATION_RELATIVE_PATHS)
def test_get_example_mf6simulation(name, get_example_mf6simulation):
    model = get_example_mf6simulation(name)
    assert isinstance(model, MFSimulation)
    assert model.name == "modflowsim"


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


@pytest.mark.unit
def test_meta():
    args = [f"{__file__}", "-v", "-s",
            "-k", test_meta_inner.__name__,
            "-M", "test_meta"]
    assert pytest.main(args, plugins=[TestMeta()]) == ExitCode.OK


# CLI argument --keep (-K)


HELLO_FNAME = 'hello.txt'


@pytest.mark.unit
@pytest.mark.meta("test_keep")
def test_keep_function_scoped_tmpdir_inner(tmpdir):
    with open(tmpdir / HELLO_FNAME, "w") as f:
        f.write("hello, function-scoped tmpdir")


@pytest.mark.unit
@pytest.mark.meta("test_keep")
class TestKeepClassScopedTmpdir:
    def test_keep_class_scoped_tmpdir_inner(self, class_tmpdir):
        with open(class_tmpdir / HELLO_FNAME, "w") as f:
            f.write("hello, class-scoped tmpdir")


@pytest.mark.unit
@pytest.mark.meta("test_keep")
def test_keep_module_scoped_tmpdir_inner(module_tmpdir):
    with open(module_tmpdir / HELLO_FNAME, "w") as f:
        f.write("hello, module-scoped tmpdir")


@pytest.mark.unit
@pytest.mark.meta("test_keep")
def test_keep_session_scoped_tmpdir_inner(session_tmpdir):
    with open(session_tmpdir / HELLO_FNAME, "w") as f:
        f.write("hello, session-scoped tmpdir")


@pytest.mark.unit
@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_function_scoped_tmpdir(tmpdir, arg):
    inner_fn = test_keep_function_scoped_tmpdir_inner.__name__
    args = [__file__, "-v", "-s",
            "-k", inner_fn,
            "-M", "test_keep",
            "-K", tmpdir]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{inner_fn}0" / HELLO_FNAME).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_class_scoped_tmpdir(tmpdir, arg):
    args = [__file__, "-v", "-s",
            "-k", TestKeepClassScopedTmpdir.test_keep_class_scoped_tmpdir_inner.__name__,
            "-M", "test_keep",
            "-K", tmpdir]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{TestKeepClassScopedTmpdir.__name__}0" / HELLO_FNAME).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_module_scoped_tmpdir(tmpdir, arg):
    args = [__file__, "-v", "-s",
            "-k", test_keep_module_scoped_tmpdir_inner.__name__,
            "-M", "test_keep",
            "-K", tmpdir]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{Path(__file__).stem}0" / HELLO_FNAME).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("arg", ["--keep", "-K"])
def test_keep_session_scoped_tmpdir(tmpdir, arg, request):
    args = [__file__, "-v", "-s",
            "-k", test_keep_session_scoped_tmpdir_inner.__name__,
            "-M", "test_keep",
            "-K", tmpdir]
    assert pytest.main(args) == ExitCode.OK
    assert Path(tmpdir / f"{request.session.name}0" / HELLO_FNAME).is_file()
