import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from modflow_devtools.markers import requires_exe

import flopy
from flopy.mf6 import MFSimulation


def base_model(sim_path):
    load_path = get_example_data_path() / "mf6-freyberg"

    sim = MFSimulation.load(sim_ws=load_path)
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
    assert mfsimlst.is_normal_termination, "model did not terminate normally"


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
        if runtime_sec == np.nan:
            continue
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
        f"outer iterations is not equal to {it_outer_answer} "
        + f"({it_outer})"
    )

    it_total = mfsimlst.get_total_iterations()
    assert it_total == it_total_answer, (
        f"total iterations is not equal to {it_total_answer} "
        + f"({it_total})"
    )


@requires_exe("mf6")
def test_mfsimlist_memory(function_tmpdir):
    total_answer = 0.000547557
    virtual_answer = 0.0

    sim = base_model(function_tmpdir)
    mfsimlst = flopy.mf6.utils.MfSimulationList(function_tmpdir / "mfsim.lst")

    total_memory = mfsimlst.get_memory_usage()
    assert total_memory == total_answer, (
        f"total memory is not equal to {total_answer} " + f"({total_memory})"
    )

    virtual_memory = mfsimlst.get_memory_usage(virtual=True)
    assert virtual_memory == virtual_answer, (
        f"virtual memory is not equal to {virtual_answer} "
        + f"({virtual_memory})"
    )

    non_virtual_memory = mfsimlst.get_non_virtual_memory_usage()
    assert total_memory == non_virtual_memory, (
        f"total memory ({total_memory}) "
        + f"does not equal non-virtual memory ({non_virtual_memory})"
    )
