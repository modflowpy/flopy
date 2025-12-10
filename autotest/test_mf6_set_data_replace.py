"""
Test set_data() replace parameter (issue #2663). This parameter
toggles whether .set_data() has update or replacement semantics.
"""

from pathlib import Path

import numpy as np
import pytest

import flopy

pytestmark = pytest.mark.mf6


def count_stress_periods(file_path):
    """Count the number of 'BEGIN period' statements in an input file."""
    with open(file_path, "r") as f:
        return sum(1 for line in f if line.strip().upper().startswith("BEGIN PERIOD"))


@pytest.mark.parametrize("replace", [False, True], ids=["replace", "no_replace"])
@pytest.mark.parametrize("use_pandas", [False, True], ids=["use_pandas", "no_pandas"])
def test_set_data_replace_array_based_pkg(function_tmpdir, replace, use_pandas):
    name = "array_based"
    og_ws = Path(function_tmpdir) / "original"
    og_ws.mkdir(exist_ok=True)

    nlay, nrow, ncol = 1, 10, 10
    nper_original = 48
    nper_new = 12

    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        sim_ws=str(og_ws),
        exe_name="mf6",
        use_pandas=use_pandas,
    )
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper_original,
        perioddata=[(1.0, 1, 1.0) for _ in range(nper_original)],
    )
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    rch_data = {kper: 0.001 + kper * 0.0001 for kper in range(nper_original)}
    rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_data)

    sim.write_simulation()

    original_rch_file = og_ws / f"{name}.rcha"
    original_sp_count = count_stress_periods(original_rch_file)
    assert original_sp_count == nper_original

    # Update RCH
    new_rch_data = {kper: 0.002 + kper * 0.0002 for kper in range(nper_new)}
    rcha.recharge.set_data(new_rch_data, replace=replace)

    # Update TDIS
    tdis.nper = nper_new
    tdis.perioddata = [(1.0, 1, 1.0) for _ in range(nper_new)]

    mod_ws = Path(function_tmpdir) / f"modified_replace_{replace}"
    mod_ws.mkdir(exist_ok=True)
    sim.set_sim_path(str(mod_ws))
    sim.write_simulation()

    modified_rch_file = mod_ws / f"{name}.rcha"
    modified_sp_count = count_stress_periods(modified_rch_file)

    if replace:
        # With replace=True, should only have 12 stress periods
        assert modified_sp_count == nper_new, (
            f"Expected {nper_new} stress periods "
            f"with replace=True, got {modified_sp_count}"
        )
    else:
        # With replace=False (backwards compatible), all 48 periods remain
        assert modified_sp_count == nper_original, (
            f"Expected {nper_original} stress periods "
            f"with replace=False, got {modified_sp_count}"
        )

    with open(modified_rch_file, "r") as f:
        content = f.read()
        assert "0.00200000" in content or "2.00000000E-03" in content
        assert "0.00420000" in content or "4.20000000E-03" in content


@pytest.mark.parametrize("replace", [False, True], ids=["replace", "no_replace"])
@pytest.mark.parametrize("use_pandas", [False, True], ids=["use_pandas", "no_pandas"])
def test_set_data_replace_list_based_pkg(function_tmpdir, replace, use_pandas):
    name = "list_based"
    sim_ws = Path(function_tmpdir) / "wel_original"
    sim_ws.mkdir(exist_ok=True)

    nlay, nrow, ncol = 1, 10, 10
    nper_original = 24
    nper_new = 6

    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(sim_ws), exe_name="mf6", use_pandas=use_pandas
    )
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper_original,
        perioddata=[(1.0, 1, 1.0) for _ in range(nper_original)],
    )
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    wel_data = {
        kper: [[(0, 5, 5), -1000.0 - kper * 10.0]] for kper in range(nper_original)
    }
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_data)

    sim.write_simulation()

    original_wel_file = sim_ws / f"{name}.wel"
    original_sp_count = count_stress_periods(original_wel_file)
    assert original_sp_count == nper_original

    # Update WEL
    new_wel_data = {
        kper: [[(0, 5, 5), -2000.0 - kper * 20.0]] for kper in range(nper_new)
    }
    wel.stress_period_data.set_data(new_wel_data, replace=replace)

    # Update TDIS
    tdis.nper = nper_new
    tdis.perioddata = [(1.0, 1, 1.0) for _ in range(nper_new)]

    mod_ws = Path(function_tmpdir) / f"wel_modified_replace_{replace}"
    mod_ws.mkdir(exist_ok=True)
    sim.set_sim_path(str(mod_ws))
    sim.write_simulation()

    modified_wel_file = mod_ws / f"{name}.wel"
    modified_sp_count = count_stress_periods(modified_wel_file)

    if replace:
        # With replace=True, should only have 6 stress periods
        assert modified_sp_count == nper_new, (
            f"Expected {nper_new} stress periods with "
            f"replace=True, got {modified_sp_count}"
        )
    else:
        # With replace=False, all 24 periods remain
        assert modified_sp_count == nper_original, (
            f"Expected {nper_original} stress periods with "
            f"replace=False, got {modified_sp_count}"
        )


def test_set_data_update_array_based_pkg(function_tmpdir):
    name = "update_array_based"
    sim_ws = Path(function_tmpdir) / "compat"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(sim_ws), exe_name="mf6")
    tdis = flopy.mf6.ModflowTdis(
        sim, nper=10, perioddata=[(1.0, 1, 1.0) for _ in range(10)]
    )
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=10, ncol=10)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(gwf)

    initial_data = dict.fromkeys(range(5), 0.001)
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=initial_data)

    additional_data = dict.fromkeys(range(5, 10), 0.002)
    rch.recharge.set_data(additional_data)  # replace defaults to False

    sim.write_simulation()

    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    gwf2 = sim2.get_model(name)
    rch2 = gwf2.get_package("RCHA")

    for kper in range(10):
        data = rch2.recharge.get_data(key=kper)
        assert np.allclose(data, 0.001 if kper < 5 else 0.002)


def test_set_data_update_list_based_pkg(function_tmpdir):
    name = "update_list_based"
    sim_ws = Path(function_tmpdir) / "wel_update"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(sim_ws), exe_name="mf6")
    tdis = flopy.mf6.ModflowTdis(
        sim, nper=10, perioddata=[(1.0, 1, 1.0) for _ in range(10)]
    )
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=10, ncol=10)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    oc = flopy.mf6.ModflowGwfoc(gwf)

    initial_data = {kper: [[(0, 5, 5), -1000.0]] for kper in range(5)}
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=initial_data)

    additional_data = {kper: [[(0, 7, 7), -2000.0]] for kper in range(5, 10)}
    wel.stress_period_data.set_data(additional_data)  # replace defaults to False

    sim.write_simulation()

    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    gwf2 = sim2.get_model(name)
    wel2 = gwf2.get_package("WEL")

    for kper in range(10):
        data = wel2.stress_period_data.get_data(key=kper)
        assert data is not None, f"Period {kper} should have data"
        if kper < 5:
            # Original data should be at (0, 5, 5)
            assert len(data) == 1
            assert data[0]["cellid"] == (0, 5, 5)
            assert data[0]["q"] == -1000.0
        else:
            # Additional data should be at (0, 7, 7)
            assert len(data) == 1
            assert data[0]["cellid"] == (0, 7, 7)
            assert data[0]["q"] == -2000.0
