"""
Test TIMEARRAYSERIES (TAS) package functionality.

Tests for GitHub issue #2702:
- Problem 1: Array entries repeated when accessing TAS data via .array property
- Problem 2: set_all_data_external() not externalizing TAS arrays
"""

from pathlib import Path

import numpy as np
import pytest

import flopy

pytestmark = pytest.mark.mf6


def test_tas_array_property(function_tmpdir):
    sim_ws = Path(function_tmpdir) / "tas_array_test"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="tas_test", version="mf6", sim_ws=str(sim_ws))
    tdis = flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=3,
        perioddata=[(1.0, 1, 1.0), (10.0, 10, 1.0), (10.0, 10, 1.0)],
    )
    model = flopy.mf6.ModflowGwf(sim, modelname="model")
    ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY")
    sim.register_ims_package(ims, [model.name])
    dis = flopy.mf6.ModflowGwfdis(
        model,
        nlay=1,
        nrow=10,
        ncol=10,
        delr=100.0,
        delc=100.0,
        top=0.0,
        botm=-10.0,
    )
    ic = flopy.mf6.ModflowGwfic(model, strt=0.0)
    npf = flopy.mf6.ModflowGwfnpf(model, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(model)
    rch = flopy.mf6.ModflowGwfrcha(model, recharge="TIMEARRAYSERIES rcharray")

    rch_array_0 = 0.001 * np.ones((10, 10))
    rch_array_0[0, 0] = 0.01

    rch_array_6 = 0.002 * np.ones((10, 10))
    rch_array_6[5, 5] = 0.02

    rch_array_12 = 0.003 * np.ones((10, 10))
    rch_array_12[9, 9] = 0.03

    tas = flopy.mf6.ModflowUtltas(
        rch,
        filename="model.rch.tas",
        tas_array={0.0: rch_array_0, 6.0: rch_array_6, 12.0: rch_array_12},
        time_series_namerecord="rcharray",
        interpolation_methodrecord="stepwise",
    )

    sim.write_simulation()
    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    model2 = sim2.get_model("model")
    rch2 = model2.get_package("rcha")
    tas2 = rch2.get_package("tas")

    active_keys = tas2.tas_array.get_active_key_list()
    keys = [key for key, _ in active_keys]
    assert len(active_keys) == 3
    assert keys == [0.0, 6.0, 12.0]

    data_0 = tas2.tas_array.get_data(0.0)
    data_6 = tas2.tas_array.get_data(6.0)
    data_12 = tas2.tas_array.get_data(12.0)

    assert np.allclose(data_0.max(), 0.01)
    assert np.allclose(data_6.max(), 0.02)
    assert np.allclose(data_12.max(), 0.03)

    array_data = tas2.tas_array.array
    assert array_data.shape[0] == 3
    assert np.allclose(array_data[0].max(), 0.01)
    assert np.allclose(array_data[1].max(), 0.02)
    assert np.allclose(array_data[2].max(), 0.03)


def test_tas_set_all_data_external(function_tmpdir):
    sim_ws = Path(function_tmpdir) / "tas_external_test"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="tas_ext", version="mf6", sim_ws=str(sim_ws))
    tdis = flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=2,
        perioddata=[(10.0, 10, 1.0), (10.0, 10, 1.0)],
    )
    model = flopy.mf6.ModflowGwf(sim, modelname="model")
    ims = flopy.mf6.ModflowIms(sim)
    sim.register_ims_package(ims, [model.name])
    dis = flopy.mf6.ModflowGwfdis(
        model,
        nlay=1,
        nrow=5,
        ncol=5,
        delr=100.0,
        delc=100.0,
        top=0.0,
        botm=-10.0,
    )
    ic = flopy.mf6.ModflowGwfic(model, strt=0.0)
    npf = flopy.mf6.ModflowGwfnpf(model, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(model)
    rch = flopy.mf6.ModflowGwfrcha(model, recharge="TIMEARRAYSERIES rcharray")
    tas = flopy.mf6.ModflowUtltas(
        rch,
        filename="model.rch.tas",
        tas_array={0.0: 0.001 * np.ones((5, 5)), 10.0: 0.002 * np.ones((5, 5))},
        time_series_namerecord="rcharray",
        interpolation_methodrecord="stepwise",
    )

    sim.set_all_data_external()
    sim.write_simulation()

    tas_file = sim_ws / "model.rch.tas"
    assert tas_file.exists()

    with open(tas_file, "r") as f:
        content = f.read()
        assert "OPEN/CLOSE" in content
        assert "INTERNAL" not in content

    external_files = list(sim_ws.glob("*.txt"))
    assert len(external_files) >= 2
