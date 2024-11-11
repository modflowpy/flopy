import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe

import flopy
from autotest.conftest import get_example_data_path
from flopy.mf6 import MFSimulation

MEMORY_UNITS = ("gigabytes", "megabytes", "kilobytes", "bytes")


def base_model(sim_path, memory_print_option=None):
    MEMORY_PRINT_OPTIONS = ("summary", "all")
    if memory_print_option is not None:
        if memory_print_option.lower() not in MEMORY_PRINT_OPTIONS:
            raise ValueError(
                f"invalid memory_print option ({memory_print_option.lower()})"
            )

    load_path = get_example_data_path() / "mf6-freyberg"

    sim = MFSimulation.load(sim_ws=load_path)
    if memory_print_option is not None:
        sim.memory_print_option = memory_print_option
    sim.set_sim_path(sim_path)
    sim.write_simulation()
    sim.run_simulation()

    return sim


@pytest.mark.xfail
def test_mfsimlist_nofile(function_tmpdir):
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "fail.lst")


@requires_exe("mf6")
def test_mfsimlist_normal(function_tmpdir):
    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")
    assert mfsimlst.normal_termination, "model did not terminate normally"


@pytest.mark.xfail
def test_mfsimlist_runtime_fail(function_tmpdir):
    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")
    runtime_sec = mfsimlst.get_runtime(units="abc")


@requires_exe("mf6")
def test_mfsimlist_runtime(function_tmpdir):
    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")
    for sim_timer in ("elapsed", "formulate", "solution"):
        runtime_sec = mfsimlst.get_runtime(simulation_timer=sim_timer)

        if not np.isnan(runtime_sec):
            runtime_min = mfsimlst.get_runtime(
                units="minutes", simulation_timer=sim_timer
            )
            assert runtime_sec / 60.0 == runtime_min, (
                f"model {sim_timer} time conversion from "
                + "sec to minutes does not match"
            )

            runtime_hrs = mfsimlst.get_runtime(
                units="hours", simulation_timer=sim_timer
            )
            assert runtime_min / 60.0 == runtime_hrs, (
                f"model {sim_timer} time conversion from "
                + "minutes to hours does not match"
            )


@requires_exe("mf6")
def test_mfsimlist_iterations(function_tmpdir):
    it_outer_answer = 13
    it_total_answer = 413

    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")

    it_outer = mfsimlst.get_outer_iterations()
    assert it_outer == it_outer_answer, (
        f"outer iterations is not equal to {it_outer_answer} " + f"({it_outer})"
    )

    it_total = mfsimlst.get_total_iterations()
    assert it_total == it_total_answer, (
        f"total iterations is not equal to {it_total_answer} " + f"({it_total})"
    )


@requires_exe("mf6")
def test_mfsimlist_memory(function_tmpdir):
    virtual_answer = 0.0

    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")

    total_memory = mfsimlst.get_memory_usage()
    assert total_memory > 0.0, (
        "total memory is not greater than 0.0 " + f"({total_memory})"
    )

    total_memory_kb = mfsimlst.get_memory_usage(units="kilobytes")
    assert np.allclose(total_memory_kb, total_memory * 1e6), (
        f"total memory in kilobytes ({total_memory_kb}) is not equal to "
        + "the total memory converted to kilobytes "
        + f"({total_memory * 1e6})"
    )

    virtual_memory = mfsimlst.get_memory_usage(virtual=True)
    if not np.isnan(virtual_memory):
        assert virtual_memory == virtual_answer, (
            f"virtual memory is not equal to {virtual_answer} " + f"({virtual_memory})"
        )

        non_virtual_memory = mfsimlst.get_non_virtual_memory_usage()
        assert total_memory == non_virtual_memory, (
            f"total memory ({total_memory}) "
            + f"does not equal non-virtual memory ({non_virtual_memory})"
        )


@requires_exe("mf6")
@pytest.mark.parametrize("mem_option", (None, "summary"))
def test_mfsimlist_memory_summary(mem_option, function_tmpdir):
    KEYS = ("TDIS", "FREYBERG", "SLN_1")
    sim = base_model(function_tmpdir, memory_print_option=mem_option)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")

    if mem_option is None:
        mem_dict = mfsimlst.get_memory_summary()
        assert mem_dict is None, "Expected None to be returned"
    else:
        for units in MEMORY_UNITS:
            mem_dict = mfsimlst.get_memory_summary(units=units)
            for key in KEYS:
                assert key in KEYS, f"memory summary key ({key}) not in KEYS"


@requires_exe("mf6")
@pytest.mark.parametrize("mem_option", (None, "all"))
def test_mfsimlist_memory_all(mem_option, function_tmpdir):
    sim = base_model(function_tmpdir, memory_print_option=mem_option)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")

    if mem_option is None:
        mem_dict = mfsimlst.get_memory_all()
        assert mem_dict is None, "Expected None to be returned"
    else:
        for units in MEMORY_UNITS:
            mem_dict = mfsimlst.get_memory_all(units=units)
            total = 0.0
            for key, value in mem_dict.items():
                total += value["MEMORYSIZE"]
            assert total > 0.0, "memory is not greater than zero"
