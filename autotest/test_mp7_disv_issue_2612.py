"""
Tests for issue #2612: MODPATH 7 izone/zones handling on DISV grids.

The issue as reported: passing a 2D (nlay, ncpl) zones array to
Modpath7Sim for a DISV-grid model raised
``ValueError: Util3d: expected 3 dimensions, found shape (nlay, ncpl)``.
The maintainer suspected this had already been fixed by #1415 (which
taught Util3d.build_2d_instances to tile a (nlay, ncpl)-shaped value
across layers the same way it already handled a genuine 3D array), but
the reporter never confirmed either way.

These tests exercise both the 2D (nlay, ncpl) and 3D (nlay, 1, ncpl)
zones array forms against a small DISV grid with real flow, so that a
regression wouldn't just raise on construction but would also be
caught if it silently produced the wrong zone values. They also build
an MF6 PRT model against the same grid and flow field: PRT's izone
(MIP package) is a native 2D (nlay, ncpl) griddata array for DISV --
no 3D workaround is needed -- so it's used here as a second reference
implementation that the MP7 zone semantics can be checked against.

Grid: GridCases.vertex_small() is a 3-layer, 5-cell-per-layer DISV
grid. Cell adjacency within a layer (from shared vertices): 0-1, 0-2,
1-3, 2-3, 2-4. Cell 2 is a cut vertex -- cell 4 is only reachable
through cell 2 -- so marking cell 2 as a stopzone guarantees that a
particle released at cell 1 and flowing toward a sink at cell 4 must
be intercepted at cell 2, regardless of which side of the small
"diamond" (0 or 3) the flow solver routes it through.
"""

import numpy as np
import pandas as pd
import pytest

from autotest.test_grid_cases import GridCases
from flopy.mf6 import (
    MFSimulation,
    ModflowEms,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdisv,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowIms,
    ModflowPrt,
    ModflowPrtdisv,
    ModflowPrtfmi,
    ModflowPrtmip,
    ModflowPrtoc,
    ModflowPrtprp,
    ModflowTdis,
)
from flopy.modpath import Modpath7, Modpath7Bas, Modpath7Sim, ParticleGroup
from flopy.modpath.mp7particledata import ParticleData
from flopy.utils.modpathfile import EndpointFile

SOURCE_CELL = 1
JUNCTION_CELL = 2
SINK_CELL = 4
SOURCE_HEAD = 8.0
SINK_HEAD = 6.0
STOPZONE = 2


def build_gwf_sim(name, ws):
    grid = GridCases.vertex_small()
    sim = MFSimulation(sim_name=name, version="mf6", exe_name="mf6", sim_ws=ws)
    ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)])
    ModflowIms(
        sim,
        complexity="SIMPLE",
        outer_dvclose=1e-6,
        outer_maximum=200,
        inner_dvclose=1e-7,
        inner_maximum=200,
    )
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)
    ModflowGwfdisv(
        gwf,
        nlay=grid.nlay,
        ncpl=grid.ncpl,
        nvert=grid.nvert,
        vertices=grid._vertices,
        cell2d=grid.cell2d,
        top=grid.top,
        botm=grid.botm,
    )
    # k33 near zero decouples the layers vertically, so flow (and the
    # tracked particle) stay in layer 0 where the zones are defined --
    # otherwise vertical leakage could carry the particle into a layer
    # with no stopzone and the test would be checking the wrong thing.
    ModflowGwfnpf(
        gwf,
        k=1.0,
        k33=1e-3,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
    )
    ModflowGwfic(gwf, strt=7.0)
    ModflowGwfchd(
        gwf,
        stress_period_data=[
            [(0, SOURCE_CELL), SOURCE_HEAD],
            [(0, SINK_CELL), SINK_HEAD],
        ],
    )
    ModflowGwfoc(
        gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim, grid


def make_zones(grid, shape):
    """Zones array marking the junction cell (layer 0) with STOPZONE and
    everything else zone 1, in either 2D (nlay, ncpl) or 3D (nlay, 1, ncpl)
    form -- the two shapes issue #2612 says should be, but weren't always,
    accepted interchangeably."""
    zones2d = np.ones((grid.nlay, grid.ncpl), dtype=np.int32)
    zones2d[0, JUNCTION_CELL] = STOPZONE
    if shape == "2d":
        return zones2d
    elif shape == "3d":
        return np.expand_dims(zones2d, axis=1)
    raise ValueError(shape)


def make_particle_data():
    # node 1 == (layer 0, cell2d SOURCE_CELL), 0-based
    return ParticleData(
        partlocs=[SOURCE_CELL],
        structured=False,
        localx=[0.5],
        localy=[0.5],
        localz=[0.5],
        drape=0,
    )


@pytest.mark.parametrize("shape", ["2d", "3d"])
def test_mp7_disv_zones(function_tmpdir, shape):
    """A DISV zones array, given as either 2D (nlay, ncpl) or 3D
    (nlay, 1, ncpl), is accepted by Modpath7Sim and actually honored:
    a particle released upstream of the stopzone cell is intercepted
    there rather than continuing on to the CHD sink."""
    gwf_name = "gwf"
    sim, grid = build_gwf_sim(gwf_name, function_tmpdir / "mf6")
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success, buff

    mp7_ws = function_tmpdir / "mp7"
    gwf = sim.get_model()
    mp7 = Modpath7(modelname="mp7", flowmodel=gwf, model_ws=mp7_ws, exe_name="mp7")
    Modpath7Bas(mp7)
    Modpath7Sim(
        mp7,
        simulationtype="pathline",
        trackingdirection="forward",
        weaksinkoption="stop_at",
        zonedataoption="on",
        stopzone=STOPZONE,
        zones=make_zones(grid, shape),
        particlegroups=[ParticleGroup(particledata=make_particle_data())],
    )
    mp7.write_input()
    success, buff = mp7.run_model()
    assert success, buff

    ep = EndpointFile(mp7_ws / "mp7.mpend").get_data()
    assert len(ep) == 1
    # the particle must stop at the junction cell -- i.e. because it
    # entered the stopzone -- not travel on to the CHD sink cell
    assert ep["k"][0] == 0
    assert ep["node"][0] == JUNCTION_CELL
    assert ep["zone"][0] == STOPZONE


def test_mp7_disv_zones_2d_3d_equivalent(function_tmpdir):
    """The 2D (nlay, ncpl) and 3D (nlay, 1, ncpl) zones arrays are two
    spellings of the same data and Util3d must interpret them
    identically -- this equivalence is the actual substance of #2612."""
    sim, grid = build_gwf_sim("gwf", function_tmpdir / "mf6")
    gwf = sim.get_model()

    def zones_array_for(shape):
        mp7 = Modpath7(
            modelname="mp7",
            flowmodel=gwf,
            model_ws=function_tmpdir / f"mp7_{shape}",
            exe_name="mp7",
        )
        Modpath7Bas(mp7)
        mp7sim = Modpath7Sim(
            mp7,
            zonedataoption="on",
            stopzone=STOPZONE,
            zones=make_zones(grid, shape),
            particlegroups=[ParticleGroup(particledata=make_particle_data())],
        )
        return mp7sim.zones.array

    np.testing.assert_array_equal(zones_array_for("2d"), zones_array_for("3d"))


def test_prt_disv_zones(function_tmpdir):
    """MF6 PRT takes izone as a native 2D (nlay, ncpl) griddata array for
    DISV grids -- no Util3d workaround needed -- and produces the same
    stopzone behavior as MODPATH 7 against the same flow field."""
    gwf_name = "gwf"
    mf6_ws = function_tmpdir / "mf6"
    gwf_sim, grid = build_gwf_sim(gwf_name, mf6_ws)
    gwf_sim.write_simulation()
    success, buff = gwf_sim.run_simulation()
    assert success, buff
    gwf = gwf_sim.get_model()

    prt_name = "prt"
    prt_ws = function_tmpdir / "prt"
    prt_sim = MFSimulation(sim_name=prt_name, version="mf6", exe_name="mf6", sim_ws=prt_ws)
    ModflowTdis(prt_sim, time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)])
    prt = ModflowPrt(prt_sim, modelname=prt_name)
    ModflowPrtdisv(
        prt,
        nlay=grid.nlay,
        ncpl=grid.ncpl,
        nvert=grid.nvert,
        vertices=grid._vertices,
        cell2d=grid.cell2d,
        top=grid.top,
        botm=grid.botm,
    )

    izone = np.ones((grid.nlay, grid.ncpl), dtype=np.int32)
    izone[0, JUNCTION_CELL] = STOPZONE
    ModflowPrtmip(prt, porosity=0.3, izone=izone)

    releasepts = list(make_particle_data().to_prp(gwf.modelgrid))
    ModflowPrtprp(
        prt,
        nreleasepts=len(releasepts),
        packagedata=releasepts,
        perioddata={0: ["FIRST"]},
        istopzone=STOPZONE,
        # the default ("eager") writes COORDINATE_CHECK_METHOD, which is
        # gated behind IDEVELOPMODE in some mf6 release builds
        coordinate_check_method=None,
    )
    ModflowPrtoc(
        prt,
        budget_filerecord=f"{prt_name}.bud",
        track_filerecord=f"{prt_name}.trk",
        trackcsv_filerecord=f"{prt_name}.trk.csv",
        saverecord=[("BUDGET", "ALL")],
    )
    ModflowPrtfmi(
        prt,
        packagedata=[
            ("GWFHEAD", f"../{mf6_ws.name}/{gwf_name}.hds"),
            ("GWFBUDGET", f"../{mf6_ws.name}/{gwf_name}.cbc"),
        ],
    )
    ems = ModflowEms(prt_sim, filename=f"{prt_name}.ems")
    prt_sim.register_solution_package(ems, [prt.name])

    prt_sim.write_simulation()
    success, buff = prt_sim.run_simulation()
    assert success, buff

    trk = pd.read_csv(prt_ws / f"{prt_name}.trk.csv")
    term = trk[trk.ireason == 3]  # termination event
    assert len(term) == 1
    # 1-based layer/cell2d indices in the PRT track file
    assert term.iloc[0]["ilay"] == 1
    assert term.iloc[0]["icell"] == JUNCTION_CELL + 1
    assert term.iloc[0]["izone"] == STOPZONE
