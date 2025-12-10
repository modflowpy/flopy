"""
Test for set_data() replace parameter (issue #2663).

Tests that the replace parameter correctly removes stress period data
for periods not included in the provided dictionary.
"""

from pathlib import Path

import pytest

import flopy

pytestmark = pytest.mark.mf6


def count_stress_periods_in_file(file_path):
    """Count the number of 'BEGIN period' statements in a file."""
    with open(file_path, "r") as f:
        return sum(1 for line in f if line.strip().upper().startswith("BEGIN PERIOD"))


@pytest.mark.parametrize("replace", [False, True])
def test_set_data_replace_array(function_tmpdir, replace):
    """Test set_data() replace parameter with MFTransientArray (RCH package)."""
    # Create a model with 48 stress periods
    sim_name = "test_model"
    sim_ws = Path(function_tmpdir) / "original"
    sim_ws.mkdir(exist_ok=True)

    nper_original = 48
    nper_new = 12

    # Create simulation with 48 stress periods
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        sim_ws=str(sim_ws),
        exe_name="mf6",
    )

    # Create TDIS with 48 stress periods
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper_original,
        perioddata=[(1.0, 1, 1.0) for _ in range(nper_original)],
    )

    # Create IMS
    flopy.mf6.ModflowIms(sim)

    # Create groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=sim_name)

    # Create DIS
    nlay, nrow, ncol = 1, 10, 10
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )

    # Create IC
    flopy.mf6.ModflowGwfic(gwf, strt=100.0)

    # Create NPF
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)

    # Create OC
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{sim_name}.cbc",
        head_filerecord=f"{sim_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Create RCH package with different recharge for each stress period
    rch_data = {kper: 0.001 + kper * 0.0001 for kper in range(nper_original)}
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_data)

    # Write the original simulation
    sim.write_simulation()

    # Count stress periods in original file
    original_rch_file = sim_ws / f"{sim_name}.rcha"
    original_sp_count = count_stress_periods_in_file(original_rch_file)
    assert original_sp_count == nper_original

    # Load the simulation
    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    gwf2 = sim2.get_model(sim_name)
    rch2 = gwf2.get_package("RCHA")

    # Create new stress period dictionary with only 12 periods
    new_rch_data = {kper: 0.002 + kper * 0.0002 for kper in range(nper_new)}

    # Update TDIS NPER
    tdis2 = sim2.get_package("TDIS")
    tdis2.nper = nper_new
    tdis2.perioddata = [(1.0, 1, 1.0) for _ in range(nper_new)]

    # Use set_data() with the replace parameter
    rch2.recharge.set_data(new_rch_data, replace=replace)

    # Write the modified simulation
    sim2_ws = Path(function_tmpdir) / f"modified_replace_{replace}"
    sim2_ws.mkdir(exist_ok=True)
    sim2.set_sim_path(str(sim2_ws))
    sim2.write_simulation()

    # Count stress periods in modified file
    modified_rch_file = sim2_ws / f"{sim_name}.rcha"
    modified_sp_count = count_stress_periods_in_file(modified_rch_file)

    if replace:
        # With replace=True, should only have 12 stress periods
        # NOTE: Currently fails due to block header persistence issue
        # When fixed, this should pass
        assert modified_sp_count == nper_new, (
            f"Expected {nper_new} stress periods with replace=True, got {modified_sp_count}"
        )
    else:
        # With replace=False (backwards compatible), all 48 periods remain
        # Periods 12-47 will be written as empty periods
        assert modified_sp_count == nper_original, (
            f"Expected {nper_original} stress periods with replace=False, got {modified_sp_count}"
        )

    # Verify data values are correct for the new periods
    with open(modified_rch_file, "r") as f:
        content = f.read()
        # Check that period 1 has the new recharge value
        assert "0.00200000" in content or "2.00000000E-03" in content
        # Check that period 12 has the new recharge value
        assert "0.00420000" in content or "4.20000000E-03" in content


@pytest.mark.parametrize("replace", [False, True])
def test_set_data_replace_list(function_tmpdir, replace):
    """Test set_data() replace parameter with MFTransientList (WEL package)."""
    # Create a model with 24 stress periods
    sim_name = "test_wel_model"
    sim_ws = Path(function_tmpdir) / "wel_original"
    sim_ws.mkdir(exist_ok=True)

    nper_original = 24
    nper_new = 6

    # Create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        sim_ws=str(sim_ws),
        exe_name="mf6",
    )

    # Create TDIS
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper_original,
        perioddata=[(1.0, 1, 1.0) for _ in range(nper_original)],
    )

    # Create IMS
    flopy.mf6.ModflowIms(sim)

    # Create groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=sim_name)

    # Create DIS
    nlay, nrow, ncol = 1, 10, 10
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )

    # Create IC
    flopy.mf6.ModflowGwfic(gwf, strt=100.0)

    # Create NPF
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)

    # Create OC
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{sim_name}.cbc",
        head_filerecord=f"{sim_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Create WEL package with different pumping rates for each stress period
    wel_data = {
        kper: [[(0, 5, 5), -1000.0 - kper * 10.0]] for kper in range(nper_original)
    }
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_data)

    # Write the original simulation
    sim.write_simulation()

    # Count stress periods in original file
    original_wel_file = sim_ws / f"{sim_name}.wel"
    original_sp_count = count_stress_periods_in_file(original_wel_file)
    assert original_sp_count == nper_original

    # Load the simulation
    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    gwf2 = sim2.get_model(sim_name)
    wel2 = gwf2.get_package("WEL")

    # Create new stress period dictionary with only 6 periods
    new_wel_data = {
        kper: [[(0, 5, 5), -2000.0 - kper * 20.0]] for kper in range(nper_new)
    }

    # Update TDIS NPER
    tdis2 = sim2.get_package("TDIS")
    tdis2.nper = nper_new
    tdis2.perioddata = [(1.0, 1, 1.0) for _ in range(nper_new)]

    # Use set_data() with the replace parameter
    wel2.stress_period_data.set_data(new_wel_data, replace=replace)

    # Write the modified simulation
    sim2_ws = Path(function_tmpdir) / f"wel_modified_replace_{replace}"
    sim2_ws.mkdir(exist_ok=True)
    sim2.set_sim_path(str(sim2_ws))
    sim2.write_simulation()

    # Count stress periods in modified file
    modified_wel_file = sim2_ws / f"{sim_name}.wel"
    modified_sp_count = count_stress_periods_in_file(modified_wel_file)

    if replace:
        # With replace=True, should only have 6 stress periods
        # NOTE: Currently fails due to block header persistence issue
        assert modified_sp_count == nper_new, (
            f"Expected {nper_new} stress periods with replace=True, got {modified_sp_count}"
        )
    else:
        # With replace=False, all 24 periods remain
        assert modified_sp_count == nper_original, (
            f"Expected {nper_original} stress periods with replace=False, got {modified_sp_count}"
        )


def test_set_data_without_replace_backwards_compatible(function_tmpdir):
    """Test that set_data() without replace parameter maintains backwards compatibility."""
    # This test ensures that existing code that relies on the "update" behavior
    # continues to work as expected
    sim_name = "test_compat"
    sim_ws = Path(function_tmpdir) / "compat"
    sim_ws.mkdir(exist_ok=True)

    # Create a simple model with 10 stress periods
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=str(sim_ws), exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=10, perioddata=[(1.0, 1, 1.0) for _ in range(10)])
    flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=sim_name)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=10, ncol=10)
    flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    flopy.mf6.ModflowGwfoc(gwf)

    # Create RCH with initial data for periods 0-4
    initial_data = dict.fromkeys(range(5), 0.001)
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=initial_data)

    # Update periods 5-9 using set_data without replace parameter
    # This should ADD to the existing data, not replace it
    additional_data = dict.fromkeys(range(5, 10), 0.002)
    rch.recharge.set_data(additional_data)  # replace defaults to False

    # Write simulation
    sim.write_simulation()

    # Load and verify all 10 periods are present
    sim2 = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws))
    gwf2 = sim2.get_model(sim_name)
    rch2 = gwf2.get_package("RCHA")

    # Check that both sets of periods are present
    for kper in range(10):
        data = rch2.recharge.get_data(key=kper)
        assert data is not None, f"Period {kper} should have data"
