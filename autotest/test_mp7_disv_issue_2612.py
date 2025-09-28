"""
Test script to reproduce issue #2612: MP7 izone handling with DISV grids

This test demonstrates the problem where MP7 expects a 3D izone array
(nlay, nrow, ncol) even for unstructured DISV grids where a 2D array
(nlay, nnodes) would be more appropriate.

The test compares behavior between MF6 PRT (which correctly handles
2D izone arrays for DISV) and MP7 (which currently forces 3D arrays).
"""


import numpy as np
import pytest

from autotest.test_grid_cases import GridCases
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfdisv,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowIms,
    ModflowTdis,
)
from flopy.modpath import Modpath7, Modpath7Bas, Modpath7Sim, ParticleGroup


def mf6_sim(sim_name, workspace):
    sim = MFSimulation(
        sim_name=sim_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=workspace,
    )
    tdis = ModflowTdis(
        sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(1.0, 1, 1.0)]
    )
    ims = ModflowIms(
        sim,
        complexity="SIMPLE",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6
    )
    gwf = ModflowGwf(
        sim,
        modelname=sim_name,
        save_flows=True
    )
    grid = GridCases.vertex_small()
    disv = ModflowGwfdisv(
        gwf,
        nlay=grid.nlay,
        ncpl=grid.ncpl,
        nvert=grid.nvert,
        vertices=grid._vertices,
        cell2d=grid.cell2d,
        top=grid.top,
        botm=grid.botm
    )
    npf = ModflowGwfnpf(
        gwf,
        k=1.0,
        save_flows=True
    )
    ic = ModflowGwfic(gwf, strt=50.0)
    oc = ModflowGwfoc(
        gwf,
        budget_filerecord=f"{sim_name}.cbc",
        head_filerecord=f"{sim_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")]
    )
    return sim


@pytest.mark.parametrize("shape", ["2d", "3d"])
def test_issue_2612(function_tmpdir, shape):
    sim_name = "test_issue_2612"
    mf6_ws = function_tmpdir / "mf6"
    mf6_ws.mkdir()
    sim = mf6_sim(sim_name, mf6_ws)
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success, buff

    mp7_ws = function_tmpdir / "mp7"
    mp7_ws.mkdir()
    gwf = sim.get_model()
    mp7 = Modpath7(
        modelname="test_mp7",
        flowmodel=gwf,
        model_ws=mp7_ws,
        exe_name="mp7"
    )
    bas = Modpath7Bas(mp7)

    if shape == "2d":
        zones = np.ones((gwf.modelgrid.nlay, gwf.modelgrid.ncpl), dtype=np.int32)
        zones[0, :gwf.modelgrid.ncpl//2] = 2
        mp7sim = Modpath7Sim(
            mp7,
            zonedataoption="on",
            zones=zones,
            particlegroups=[ParticleGroup()]
        )
    else:
        zones = np.ones((gwf.modelgrid.nlay, 1, gwf.modelgrid.ncpl), dtype=np.int32)
        zones[0, 0, :gwf.modelgrid.ncpl//2] = 2
        mp7sim = Modpath7Sim(
            mp7,
            zonedataoption="on",
            zones=zones,
            particlegroups=[ParticleGroup()]
        )

    mp7.write_input()
    success, buff = mp7.run_model()
    assert success, buff
