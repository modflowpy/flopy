"""
Benchmarks for flopy.utils.Util2d and Util3d operations including:
- Array creation
- Binary file I/O
- Array access performance
"""

import numpy as np
import pytest

from flopy.modflow import Modflow
from flopy.utils import Util2d, Util3d

SIZES = {
    "small": {"nlay": 3, "nrow": 10, "ncol": 10},
    "medium": {"nlay": 10, "nrow": 100, "ncol": 100},
    "large": {"nlay": 20, "nrow": 200, "ncol": 200},
}


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util2d_create(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nrow"], dims["ncol"])
    data = np.random.random(shape)
    ml = Modflow(model_ws=function_tmpdir)

    def create_util2d():
        return Util2d(ml, shape, np.float32, data.copy(), "test")

    benchmark(create_util2d)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util3d_create(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nlay"], dims["nrow"], dims["ncol"])
    data = np.random.random(shape)
    ml = Modflow(model_ws=function_tmpdir)

    def create_util3d():
        return Util3d(ml, shape, np.float32, data.copy(), "test")

    benchmark(create_util3d)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util2d_external_write(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nrow"], dims["ncol"])
    data = np.random.random(shape).astype(np.float32)
    fpath = function_tmpdir / "test_array.dat"

    def write_bin():
        Util2d.write_bin(shape, fpath, data, bintype="head")

    benchmark(write_bin)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util3d_external_write(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nlay"], dims["nrow"], dims["ncol"])
    data = np.random.random(shape).astype(np.float32)
    fpath = function_tmpdir / "test_array3d.dat"

    def write_bin():
        for i in range(shape[0]):
            layer_path = function_tmpdir / f"test_array3d_lay{i}.dat"
            Util2d.write_bin((shape[1], shape[2]), layer_path, data[i], bintype="head")

    benchmark(write_bin)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util2d_array_copy(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nrow"], dims["ncol"])
    data = np.random.random(shape)
    ml = Modflow(model_ws=function_tmpdir)
    u2d = Util2d(ml, shape, np.float32, data, "test")
    benchmark(lambda: u2d.array.copy())


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_util3d_array_copy(benchmark, function_tmpdir, size):
    dims = SIZES[size]
    shape = (dims["nlay"], dims["nrow"], dims["ncol"])
    data = np.random.random(shape)
    ml = Modflow(model_ws=function_tmpdir)
    u3d = Util3d(ml, shape, np.float32, data, "test")
    benchmark(lambda: u3d.array.copy())
