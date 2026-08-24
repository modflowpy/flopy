"""
Test that optional package variables with default values aren't written to
input files when write_defaults=False.

Reproduces https://github.com/modflowpy/flopy/issues/2710.
"""

from pathlib import Path

import pytest

import flopy

pytestmark = pytest.mark.mf6


def _build_prt_sim(ws, coordinate_check_method="eager"):
    sim = flopy.mf6.MFSimulation(sim_name="prt", sim_ws=str(ws))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    ems = flopy.mf6.ModflowEms(sim)
    prt = flopy.mf6.ModflowPrt(sim, modelname="prt")
    flopy.mf6.ModflowPrtdis(
        prt,
        nlay=1,
        nrow=1,
        ncol=3,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowPrtmip(prt, porosity=0.1)
    flopy.mf6.ModflowPrtprp(
        prt,
        nreleasepts=1,
        packagedata=[(0, (0, 0, 0), 0.5, 0.5, 0.5)],
        perioddata={0: ["FIRST"]},
        coordinate_check_method=coordinate_check_method,
    )
    flopy.mf6.ModflowPrtoc(prt, track_filerecord=[("prt.trk",)])
    sim.register_solution_package(ems, [prt.name])
    return sim


def _prp_text(ws):
    prp_files = list(Path(ws).glob("*.prp"))
    assert len(prp_files) == 1, f"expected one .prp file, found: {prp_files}"
    return prp_files[0].read_text().upper()


def test_coordinate_check_method(function_tmpdir):
    # write_defaults=True, default value
    ws = Path(function_tmpdir) / "write_defaults"
    ws.mkdir()
    sim = _build_prt_sim(ws, coordinate_check_method="eager")
    sim.write_simulation()
    text = _prp_text(ws)
    assert "COORDINATE_CHECK_METHOD" in text
    assert "EAGER" in text

    # write_defaults=False, default value
    ws = Path(function_tmpdir) / "eager"
    ws.mkdir()
    sim = _build_prt_sim(ws, coordinate_check_method="eager")
    sim.simulation_data.write_defaults = False
    sim.write_simulation()
    assert "COORDINATE_CHECK_METHOD" not in _prp_text(ws)

    # write_defaults=False, non-default value
    ws = Path(function_tmpdir) / "none"
    ws.mkdir()
    sim = _build_prt_sim(ws, coordinate_check_method="none")
    sim.simulation_data.write_defaults = False
    sim.write_simulation()
    text = _prp_text(ws)
    assert "COORDINATE_CHECK_METHOD" in text
    assert "NONE" in text

    # write_defaults=False passed directly to write_simulation(), default
    # value, overriding simulation_data.write_defaults for this call only
    ws = Path(function_tmpdir) / "write_simulation_kwarg"
    ws.mkdir()
    sim = _build_prt_sim(ws, coordinate_check_method="eager")
    sim.write_simulation(write_defaults=False)
    assert "COORDINATE_CHECK_METHOD" not in _prp_text(ws)
    assert sim.simulation_data.write_defaults is True
