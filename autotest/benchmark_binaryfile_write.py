"""Benchmark tests for binaryfile write methods."""

import numpy as np
import pytest
from pathlib import Path

from flopy.utils import HeadFile, CellBudgetFile
from flopy.mf6.utils import MfGrdFile


@pytest.fixture
def freyberg_hds_path(example_data_path):
    """Path to freyberg multilayer transient head file."""
    return example_data_path / "freyberg_multilayer_transient" / "freyberg.hds"


@pytest.fixture
def freyberg_cbc_path(example_data_path):
    """Path to freyberg multilayer transient budget file."""
    return example_data_path / "freyberg_multilayer_transient" / "freyberg.cbc"


@pytest.fixture
def mfgrd_dis_path(example_data_path):
    """Path to DIS grid file."""
    return example_data_path / "mf6-freyberg" / "freyberg.dis.grb"


@pytest.fixture
def mfgrd_disv_path(example_data_path):
    """Path to DISV grid file."""
    return example_data_path / "mf6" / "test006_gwf3_disv" / "expected_output" / "flow.disv.grb"


@pytest.mark.slow
def test_headfile_write_benchmark(benchmark, freyberg_hds_path, tmp_path):
    """Benchmark HeadFile.write() with multi-layer transient data."""
    # Load head file with many time steps
    hds = HeadFile(freyberg_hds_path)

    # Get first 100 timesteps for benchmark (or all if fewer)
    nsteps = min(100, len(hds.kstpkper))
    kstpkper = hds.kstpkper[:nsteps]

    output_file = tmp_path / "benchmark_output.hds"

    def write_head():
        hds.write(output_file, kstpkper=kstpkper)

    # Run benchmark
    benchmark(write_head)

    # Verify output exists
    assert output_file.exists()


@pytest.mark.slow
def test_cellbudgetfile_write_benchmark(benchmark, freyberg_cbc_path, tmp_path):
    """Benchmark CellBudgetFile.write() with multi-term budget data."""
    # Load budget file with many terms and time steps
    cbc = CellBudgetFile(freyberg_cbc_path)

    # Get first 50 timesteps for benchmark (or all if fewer)
    nsteps = min(50, len(cbc.kstpkper))
    kstpkper = cbc.kstpkper[:nsteps]

    output_file = tmp_path / "benchmark_output.cbc"

    def write_budget():
        cbc.write(output_file, kstpkper=kstpkper)

    # Run benchmark
    benchmark(write_budget)

    # Verify output exists
    assert output_file.exists()


@pytest.mark.slow
def test_mfgrdfile_write_benchmark_dis(benchmark, tmp_path):
    """Benchmark MfGrdFile.write() for DIS grid."""
    # Create a moderately-sized DIS grid for benchmarking
    # Use a realistic size: 100 x 100 x 10 cells
    nlay, nrow, ncol = 10, 100, 100
    nodes = nlay * nrow * ncol

    # Calculate nja for structured grid
    # For interior cells: 7 connections (self + 6 neighbors)
    # Simplified: ~7 * nodes
    nja = 7 * nodes - 2 * (nlay * nrow + nlay * ncol + nrow * ncol)

    # Create grid file with synthetic data
    grb_data = {
        'NCELLS': nodes,
        'NLAY': nlay,
        'NROW': nrow,
        'NCOL': ncol,
        'NJA': nja,
        'XORIGIN': 0.0,
        'YORIGIN': 0.0,
        'ANGROT': 0.0,
        'DELR': np.ones(ncol, dtype=np.float64) * 100.0,
        'DELC': np.ones(nrow, dtype=np.float64) * 100.0,
        'TOP': np.ones(nodes, dtype=np.float64) * 100.0,
        'BOTM': np.arange(nodes, dtype=np.float64),
        'IA': np.arange(nodes + 1, dtype=np.int32),
        'JA': np.arange(nja, dtype=np.int32) % nodes,
        'IDOMAIN': np.ones(nodes, dtype=np.int32),
        'ICELLTYPE': np.ones(nodes, dtype=np.int32),
    }

    # Write a temporary grb file to load
    from flopy.utils.utils_def import FlopyBinaryData
    temp_grb = tmp_path / "temp_input.grb"

    writer = FlopyBinaryData()
    writer.precision = "double"

    with open(temp_grb, "wb") as f:
        writer.file = f
        # Write minimal header
        writer.write_text("GRID DIS\n", 50)
        writer.write_text("VERSION 1\n", 50)
        writer.write_text("NTXT 16\n", 50)
        writer.write_text("LENTXT 100\n", 50)

        # Write variable definitions
        var_list = [
            ("NCELLS", "INTEGER", 0, []),
            ("NLAY", "INTEGER", 0, []),
            ("NROW", "INTEGER", 0, []),
            ("NCOL", "INTEGER", 0, []),
            ("NJA", "INTEGER", 0, []),
            ("XORIGIN", "DOUBLE", 0, []),
            ("YORIGIN", "DOUBLE", 0, []),
            ("ANGROT", "DOUBLE", 0, []),
            ("DELR", "DOUBLE", 1, [ncol]),
            ("DELC", "DOUBLE", 1, [nrow]),
            ("TOP", "DOUBLE", 1, [nodes]),
            ("BOTM", "DOUBLE", 1, [nodes]),
            ("IA", "INTEGER", 1, [nodes + 1]),
            ("JA", "INTEGER", 1, [nja]),
            ("IDOMAIN", "INTEGER", 1, [nodes]),
            ("ICELLTYPE", "INTEGER", 1, [nodes]),
        ]

        for name, dtype_str, ndim, dims in var_list:
            if ndim == 0:
                line = f"{name} {dtype_str} NDIM {ndim}\n"
            else:
                dims_str = " ".join(str(d) for d in dims[::-1])
                line = f"{name} {dtype_str} NDIM {ndim} {dims_str}\n"
            writer.write_text(line, 100)

        # Write binary data
        for name, dtype_str, ndim, dims in var_list:
            value = grb_data[name]
            if ndim == 0:
                if dtype_str == "INTEGER":
                    writer.write_integer(int(value))
                else:
                    writer.write_real(float(value))
            else:
                arr = np.asarray(value)
                if dtype_str == "INTEGER":
                    arr = arr.astype(np.int32)
                elif dtype_str == "DOUBLE":
                    arr = arr.astype(np.float64)
                writer.write_record(arr.flatten(order="F"), dtype=arr.dtype)

    # Load the grid file
    grb = MfGrdFile(str(temp_grb), verbose=False)

    output_file = tmp_path / "benchmark_output.grb"

    def write_grb():
        grb.write(output_file, verbose=False)

    # Run benchmark
    benchmark(write_grb)

    # Verify output exists
    assert output_file.exists()


if __name__ == "__main__":
    # For quick testing
    pytest.main([__file__, "-v", "-s"])
