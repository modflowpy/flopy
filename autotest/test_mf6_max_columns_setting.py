"""
Test for max_columns_of_data setting behavior (GitHub issues #2419/#2420).

Tests that max_columns_of_data is respected when the simulation is created
from scratch or loaded from file, and that the user setting is chosen over
auto-detected values.
"""

import numpy as np
import pytest

import flopy


def test_backward_compat_user_set_flag(function_tmpdir):
    sim_ws = function_tmpdir / "test_compat_user_set"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="test_sim", sim_ws=sim_ws)

    assert not sim.simulation_data.max_columns_user_set
    assert not sim.simulation_data.max_columns_auto_set

    sim.simulation_data.max_columns_of_data = 10

    assert sim.simulation_data.max_columns_user_set
    assert not sim.simulation_data.max_columns_auto_set

    sim.simulation_data.max_columns_user_set = False
    assert not sim.simulation_data.max_columns_user_set


def test_backward_compat_auto_set_flag(function_tmpdir):
    sim_ws = function_tmpdir / "test_compat_auto_set"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="test_sim", sim_ws=sim_ws)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    model = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file="model.nam")
    dis = flopy.mf6.ModflowGwfdis(
        model, nlay=1, nrow=10, ncol=25, delr=100.0, delc=100.0, top=100.0, botm=0.0
    )
    ic = flopy.mf6.ModflowGwfic(model, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(model, save_flows=True, icelltype=0, k=1.0)
    ims = flopy.mf6.ModflowIms(sim)
    sim.register_ims_package(ims, [model.name])

    sim.write_simulation()

    loaded_sim = flopy.mf6.MFSimulation.load(sim_name="test_sim", sim_ws=sim_ws)

    assert loaded_sim.simulation_data.max_columns_auto_set
    assert not loaded_sim.simulation_data.max_columns_user_set

    loaded_sim.simulation_data.max_columns_auto_set = False
    assert not loaded_sim.simulation_data.max_columns_auto_set

    loaded_sim.simulation_data.max_columns_of_data = 1
    assert loaded_sim.simulation_data.max_columns_of_data == 1
    assert loaded_sim.simulation_data.max_columns_user_set


def test_backward_compat_flag_transitions(function_tmpdir):
    sim_ws = function_tmpdir / "test_compat_transitions"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="test_sim", sim_ws=sim_ws)

    assert not sim.simulation_data.max_columns_user_set
    assert not sim.simulation_data.max_columns_auto_set

    # manually set auto_set flag (simulate old code path)
    sim.simulation_data.max_columns_auto_set = True
    assert sim.simulation_data.max_columns_auto_set
    assert not sim.simulation_data.max_columns_user_set

    # user sets a value - should transition to user_set
    sim.simulation_data.max_columns_of_data = 5
    assert sim.simulation_data.max_columns_user_set
    assert not sim.simulation_data.max_columns_auto_set

    # clear user_set
    sim.simulation_data.max_columns_user_set = False
    assert not sim.simulation_data.max_columns_user_set
    assert not sim.simulation_data.max_columns_auto_set


def test_max_columns_internal_array(function_tmpdir):
    sim_ws = function_tmpdir / "test_internal"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="test_sim", sim_ws=sim_ws)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file="model.nam")
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=3,
        ncol=3,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=np.arange(9).reshape(1, 3, 3).astype(float))
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=0, k=1.0)
    ims = flopy.mf6.ModflowIms(sim)
    sim.register_ims_package(ims, [gwf.name])
    sim.write_simulation()

    ic_file = sim_ws / "model.ic"
    assert ic_file.exists()

    def check_columns(expected):
        with open(ic_file) as f:
            content = f.read()

        content_upper = content.upper()
        assert "BEGIN GRIDDATA" in content_upper
        assert "STRT" in content_upper

        lines = content.split("\n")
        found_strt = False
        found_internal = False
        strt_lines = []

        for line in lines:
            line_upper = line.upper()

            if "STRT" in line_upper and not found_strt:
                found_strt = True
                continue

            if found_strt and not found_internal and "INTERNAL" in line_upper:
                found_internal = True
                continue

            if found_internal:
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.upper().startswith("END")
                    or stripped.upper().startswith("BEGIN")
                ):
                    break
                if stripped:
                    strt_lines.append(stripped)

        assert len(strt_lines) > 0
        for i, line in enumerate(strt_lines):
            values = line.split()
            assert len(values) == expected

    check_columns(expected=3)

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    sim.simulation_data.max_columns_of_data = 1
    sim.write_simulation()

    check_columns(expected=1)


reason_ext = (
    "set_all_data_external() writes files immediately. Changing output settings"
    "afterward has no effect unless set_all_data_external() is called again. "
    "You must set max_columns_of_data BEFORE calling set_all_data_external(). "
    "This is an architectural limitation requiring a deferred write implementation."
)


@pytest.mark.parametrize(
    ("which_ext", "first_set", "workspace"),
    [
        ("all", "opt", "same"),
        ("all", "opt", "diff"),
        pytest.param("all", "ext", "same", marks=pytest.mark.xfail(reason=reason_ext)),
        pytest.param("all", "ext", "diff", marks=pytest.mark.xfail(reason=reason_ext)),
        ("one", "opt", "same"),
        ("one", "opt", "diff"),
        pytest.param("one", "ext", "same", marks=pytest.mark.xfail(reason=reason_ext)),
        pytest.param("one", "ext", "diff", marks=pytest.mark.xfail(reason=reason_ext)),
    ],
)
def test_max_columns_external_array(function_tmpdir, which_ext, first_set, workspace):
    sim_ws = function_tmpdir / "test_external"
    sim_ws.mkdir(exist_ok=True)

    sim = flopy.mf6.MFSimulation(sim_name="test_sim", sim_ws=sim_ws)
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file="model.nam")
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=5,
        ncol=5,
        delr=100.0,
        delc=100.0,
        top=100.0,
        botm=0.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=np.arange(25).reshape(1, 5, 5).astype(float))
    if which_ext == "one":
        ic.strt.store_as_external_file("model.ic_strt.txt")
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=0, k=1.0)
    ims = flopy.mf6.ModflowIms(sim)
    sim.register_ims_package(ims, [gwf.name])
    if which_ext == "all":
        sim.set_all_data_external()
    sim.write_simulation()

    def check_columns(ws, expected):
        strt_file = ws / "model.ic_strt.txt"
        assert strt_file.exists()
        with open(strt_file) as f:
            lines = f.readlines()

        data_lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

        for i, line in enumerate(data_lines):
            values = line.split()
            assert len(values) == expected, (
                f"Line {i} in external file should have 1 value (max_columns=1), "
                f"but has {len(values)}: {line}"
            )

    check_columns(ws=sim_ws, expected=5)

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    if workspace == "diff":
        sim_ws = function_tmpdir / "test_external_new"
        sim_ws.mkdir()
        sim.set_sim_path(sim_ws)
    if first_set == "opt":
        sim.simulation_data.max_columns_of_data = 1
        # For "same" workspace, need replace_existing=True to rewrite existing files
        replace_existing = workspace == "same"
        sim.set_all_data_external(replace_existing=replace_existing)
    elif first_set == "ext":
        sim.set_all_data_external()
        sim.simulation_data.max_columns_of_data = 1
    sim.write_simulation()

    check_columns(ws=sim_ws, expected=1)
