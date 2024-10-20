import os
from pathlib import Path
from platform import system

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe

from autotest.test_mp7_cases import Mp7Cases
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfdis,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfwel,
    ModflowIms,
    ModflowTdis,
)
from flopy.modpath import (
    CellDataType,
    FaceDataType,
    LRCParticleData,
    Modpath7,
    Modpath7Bas,
    Modpath7Sim,
    NodeParticleData,
    ParticleData,
    ParticleGroup,
    ParticleGroupLRCTemplate,
    ParticleGroupNodeTemplate,
)
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile

pytestmark = pytest.mark.mf6

ex01b_mf6_model_name = "ex01b_mf6"


@pytest.fixture
def ex01b_mf6_model(function_tmpdir):
    """
    MODPATH 7 example 1 for MODFLOW 6
    """

    # model data
    nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
    nlay, nrow, ncol = 3, 21, 20
    delr = delc = 500.0
    top = 400.0
    botm = [220.0, 200.0, 0.0]
    laytyp = [1, 0, 0]
    kh = [50.0, 0.01, 200.0]
    kv = [10.0, 0.01, 20.0]
    wel_loc = (2, 10, 9)
    wel_q = -150000.0
    rch = 0.005
    riv_h = 320.0
    riv_z = 317.0
    riv_c = 1.0e5

    # Create the Flopy simulation object
    sim = MFSimulation(
        sim_name=ex01b_mf6_model_name,
        exe_name="mf6",
        version="mf6",
        sim_ws=function_tmpdir,
    )

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{ex01b_mf6_model_name}.nam"
    gwf = ModflowGwf(
        sim,
        modelname=ex01b_mf6_model_name,
        model_nam_file=model_nam_file,
        save_flows=True,
    )

    # Create the Flopy iterative model solver (ims) Package object
    ims = ModflowIms(sim, pname="ims", complexity="SIMPLE")

    # create gwf file
    dis = ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        length_units="FEET",
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    # Create the initial conditions package
    ic = ModflowGwfic(gwf, pname="ic", strt=top)

    # Create the node property flow package
    npf = ModflowGwfnpf(gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv)

    # recharge
    ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    ModflowGwfwel(gwf, maxbound=1, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    ModflowGwfriv(gwf, stress_period_data={0: rd})
    # Create the output control package
    headfile = f"{ex01b_mf6_model_name}.hds"
    head_record = [headfile]
    budgetfile = f"{ex01b_mf6_model_name}.cbb"
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    # Write the datasets
    sim.write_simulation()
    return sim, function_tmpdir


def build_modpath(ws, mpn, particlegroups, grid):
    # particle data
    zone3 = np.ones((grid.nrow, grid.ncol), dtype=np.int32)
    wel_loc = (2, 10, 9)
    zone3[wel_loc[1:]] = 2
    zones = [1, 1, zone3]
    defaultiface6 = {"RCH": 6, "EVT": 6}

    # load the MODFLOW 6 model
    sim = MFSimulation.load("mf6mod", "mf6", "mf6", ws)
    gwf = sim.get_model(ex01b_mf6_model_name)

    # create modpath files
    mp = Modpath7(modelname=mpn, flowmodel=gwf, exe_name="mp7", model_ws=ws)
    Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface6)
    Modpath7Sim(
        mp,
        simulationtype="endpoint",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        referencetime=0.0,
        stoptimeoption="extend",
        zonedataoption="on",
        zones=zones,
        particlegroups=particlegroups,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    success, buff = mp.run_model()
    assert success, f"mp7 model ({mp.name}) did not run"


def endpoint_compare(fpth0, epf):
    # get base endpoint data
    e = EndpointFile(fpth0)
    maxtime0 = e.get_maxtime()
    maxid0 = e.get_maxid()
    maxtravel0 = e.get_maxtraveltime()
    e0 = e.get_alldata()

    names = ["x", "y", "z", "x0", "y0", "z0"]
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("x0", np.float32),
            ("y0", np.float32),
            ("z0", np.float32),
        ]
    )
    t0 = np.rec.fromarrays((e0[name] for name in names), dtype=dtype)

    for fpth1 in epf:
        e = EndpointFile(fpth1)
        maxtime1 = e.get_maxtime()
        maxid1 = e.get_maxid()
        maxtravel1 = e.get_maxtraveltime()
        e1 = e.get_alldata()

        # check maxid
        msg = (
            f"endpoint maxid ({maxid0}) in {os.path.basename(fpth0)} "
            f"are not equal to the endpoint maxid ({maxid1}) "
            f"in {os.path.basename(fpth1)}"
        )
        assert maxid0 == maxid1, msg

        # check maxtravel
        msg = (
            f"endpoint maxtraveltime ({maxtravel0}) "
            f"in {os.path.basename(fpth0)} are not equal to the endpoint "
            f"maxtraveltime ({maxtravel1}) in {os.path.basename(fpth1)}"
        )
        assert maxtravel0 == maxtravel1, msg

        # check maxtimes
        msg = (
            f"endpoint maxtime ({maxtime0}) in {os.path.basename(fpth0)} "
            f"are not equal to the endpoint maxtime ({maxtime1}) "
            f"in {os.path.basename(fpth1)}"
        )
        assert maxtime0 == maxtime1, msg

        # check that endpoint data are approximately the same
        t1 = np.rec.fromarrays((e1[name] for name in names), dtype=dtype)
        for name in names:
            msg = (
                f"endpoints in {os.path.basename(fpth0)} are not equal "
                f"(within 1e-5) to the endpoints in {os.path.basename(fpth1)} "
                f"for column {name}."
            )
            assert np.allclose(t0[name], t1[name]), msg


@requires_exe("mf6", "mp7")
def test_default_modpath(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model

    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_default"
    pg = ParticleGroup(particlegroupname="DEFAULT")
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_faceparticles_is1(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model

    local = np.array(
        [
            [0.1666666667e00, 0.1666666667e00, 1.0],
            [0.5000000000e00, 0.1666666667e00, 1.0],
            [0.8333333333e00, 0.1666666667e00, 1.0],
            [0.1666666667e00, 0.5000000000e00, 1.0],
            [0.5000000000e00, 0.5000000000e00, 1.0],
            [0.8333333333e00, 0.5000000000e00, 1.0],
            [0.1666666667e00, 0.8333333333e00, 1.0],
            [0.5000000000e00, 0.8333333333e00, 1.0],
            [0.8333333333e00, 0.8333333333e00, 1.0],
        ]
    )

    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_face_t1node"
    locs = []
    localx = []
    localy = []
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    for i in range(grid.nrow):
        for j in range(grid.ncol):
            node = i * grid.ncol + j
            for xloc, yloc, zloc in local:
                locs.append(node)
                localx.append(xloc)
                localy.append(yloc)
    p = ParticleData(
        locs, structured=False, drape=0, localx=localx, localy=localy, localz=1
    )
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroup(
        particlegroupname="T1NODEPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )

    # set base file name
    fpth0 = function_tmpdir / "ex01b_mf6_mp_face_t1node.mpend"

    # get list of node endpath files
    epf = [
        function_tmpdir / name
        for name in os.listdir(function_tmpdir)
        if ".mpend" in name and "_face_" in name and "_t2a" not in name
    ]
    epf.remove(fpth0)

    endpoint_compare(fpth0, epf)


@requires_exe("mf6", "mp7")
def test_facenode_is3(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_face_t3node"
    locs = []
    for i in range(grid.nrow):
        for j in range(grid.ncol):
            node = i * grid.ncol + j
            locs.append(node)
    sd = FaceDataType(
        drape=0,
        verticaldivisions1=0,
        horizontaldivisions1=0,
        verticaldivisions2=0,
        horizontaldivisions2=0,
        verticaldivisions3=0,
        horizontaldivisions3=0,
        verticaldivisions4=0,
        horizontaldivisions4=0,
        rowdivisions5=0,
        columndivisions5=0,
        rowdivisions6=3,
        columndivisions6=3,
    )
    p = NodeParticleData(subdivisiondata=sd, nodes=locs)
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupNodeTemplate(
        particlegroupname="T3NODEPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_facenode_is3a(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_face_t3anode"
    locsa = []
    for i in range(11):
        for j in range(grid.ncol):
            node = i * grid.ncol + j
            locsa.append(node)
    locsb = []
    for i in range(11, grid.nrow):
        for j in range(grid.ncol):
            node = i * grid.ncol + j
            locsb.append(node)
    sd = FaceDataType(
        drape=0,
        verticaldivisions1=0,
        horizontaldivisions1=0,
        verticaldivisions2=0,
        horizontaldivisions2=0,
        verticaldivisions3=0,
        horizontaldivisions3=0,
        verticaldivisions4=0,
        horizontaldivisions4=0,
        rowdivisions5=0,
        columndivisions5=0,
        rowdivisions6=3,
        columndivisions6=3,
    )
    p = NodeParticleData(subdivisiondata=[sd, sd], nodes=[locsa, locsb])
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupNodeTemplate(
        particlegroupname="T3ANODEPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_facenode_is2a(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_face_t2anode"
    locsa = [[0, 0, 0, 0, 10, grid.ncol - 1]]
    locsb = [[0, 11, 0, 0, grid.nrow - 1, grid.ncol - 1]]
    sd = FaceDataType(
        drape=0,
        verticaldivisions1=0,
        horizontaldivisions1=0,
        verticaldivisions2=0,
        horizontaldivisions2=0,
        verticaldivisions3=0,
        horizontaldivisions3=0,
        verticaldivisions4=0,
        horizontaldivisions4=0,
        rowdivisions5=0,
        columndivisions5=0,
        rowdivisions6=3,
        columndivisions6=3,
    )
    p = LRCParticleData(subdivisiondata=[sd, sd], lrcregions=[locsa, locsb])
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupNodeTemplate(
        particlegroupname="T2ANODEPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_cellparticles_is1(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_cell_t1node"
    locs = []
    for k in range(grid.nlay):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                node = k * grid.nrow * grid.ncol + i * grid.ncol + j
                locs.append(node)
    p = ParticleData(
        locs, structured=False, drape=0, localx=0.5, localy=0.5, localz=0.5
    )
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroup(
        particlegroupname="T1NODEPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )

    # set base file name
    fpth0 = function_tmpdir / "ex01b_mf6_mp_cell_t1node.mpend"

    # get list of node endpath files
    epf = [
        function_tmpdir / name
        for name in os.listdir(function_tmpdir)
        if ".mpend" in name and "_cell_" in name and "_t2a" not in name
    ]
    epf.remove(fpth0)

    endpoint_compare(fpth0, epf)


@requires_exe("mf6", "mp7")
def test_cellparticleskij_is1(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_cell_t1kij"
    locs = []
    for k in range(grid.nlay):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                locs.append((k, i, j))
    p = ParticleData(
        locs, structured=True, drape=0, localx=0.5, localy=0.5, localz=0.5
    )
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroup(
        particlegroupname="T1KIJPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_cellnode_is3(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_cell_t3node"
    locs = []
    for k in range(grid.nlay):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                node = k * grid.nrow * grid.ncol + i * grid.ncol + j
                locs.append(node)
    sd = CellDataType(
        drape=0,
        columncelldivisions=1,
        rowcelldivisions=1,
        layercelldivisions=1,
    )
    p = NodeParticleData(subdivisiondata=sd, nodes=locs)
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupNodeTemplate(
        particlegroupname="T3CELLPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_cellnode_is3a(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_cell_t3anode"
    locsa = []
    for k in range(1):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                node = k * grid.nrow * grid.ncol + i * grid.ncol + j
                locsa.append(node)
    locsb = []
    for k in range(1, 2):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                node = k * grid.nrow * grid.ncol + i * grid.ncol + j
                locsb.append(node)
    locsc = []
    for k in range(2, grid.nlay):
        for i in range(grid.nrow):
            for j in range(grid.ncol):
                node = k * grid.nrow * grid.ncol + i * grid.ncol + j
                locsc.append(node)
    sd = CellDataType(
        drape=0,
        columncelldivisions=1,
        rowcelldivisions=1,
        layercelldivisions=1,
    )
    p = NodeParticleData(
        subdivisiondata=[sd, sd, sd], nodes=[locsa, locsb, locsc]
    )
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupNodeTemplate(
        particlegroupname="T3ACELLPG", particledata=p, filename=fpth
    )
    build_modpath(
        function_tmpdir,
        mpnam,
        pg,
        sim.get_model(ex01b_mf6_model_name).modelgrid,
    )


@requires_exe("mf6", "mp7")
def test_cellnode_is2a(ex01b_mf6_model):
    sim, function_tmpdir = ex01b_mf6_model
    grid = sim.get_model(ex01b_mf6_model_name).modelgrid

    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    mpnam = f"{ex01b_mf6_model_name}_mp_cell_t2anode"
    locsa = [
        [0, 0, 0, 0, grid.nrow - 1, grid.ncol - 1],
        [1, 0, 0, 1, grid.nrow - 1, grid.ncol - 1],
    ]
    locsb = [[2, 0, 0, 2, grid.nrow - 1, grid.ncol - 1]]
    sd = CellDataType(
        drape=0,
        columncelldivisions=1,
        rowcelldivisions=1,
        layercelldivisions=1,
    )
    p = LRCParticleData(subdivisiondata=[sd, sd], lrcregions=[locsa, locsb])
    fpth = f"{mpnam}.sloc"
    pg = ParticleGroupLRCTemplate(
        particlegroupname="T2ACELLPG", particledata=p, filename=fpth
    )

    build_modpath(function_tmpdir, mpnam, pg, grid)


ex01_mf6_model_name = "ex01_mf6"


@pytest.fixture
def ex01_mf6_model(function_tmpdir):
    """
    MODPATH 7 example 1 for MODFLOW 6
    """

    # Create the Flopy simulation object
    sim = MFSimulation(
        sim_name=ex01_mf6_model_name,
        exe_name="mf6",
        version="mf6",
        sim_ws=function_tmpdir,
    )

    # model data
    nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
    nlay, nrow, ncol = 3, 21, 20
    delr = delc = 500.0
    top = 400.0
    botm = [220.0, 200.0, 0.0]
    laytyp = [1, 0, 0]
    kh = [50.0, 0.01, 200.0]
    kv = [10.0, 0.01, 20.0]
    wel_loc = (2, 10, 9)
    wel_q = -150000.0
    rch = 0.005
    riv_h = 320.0
    riv_z = 317.0
    riv_c = 1.0e5

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{ex01_mf6_model_name}.nam"
    gwf = ModflowGwf(
        sim,
        modelname=ex01_mf6_model_name,
        model_nam_file=model_nam_file,
        save_flows=True,
    )

    # Create the Flopy iterative model solver (ims) Package object
    ims = ModflowIms(sim, pname="ims", complexity="SIMPLE")

    # create gwf file
    dis = ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        length_units="FEET",
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    # Create the initial conditions package
    ic = ModflowGwfic(gwf, pname="ic", strt=top)

    # Create the node property flow package
    npf = ModflowGwfnpf(gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv)

    # recharge
    ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    ModflowGwfwel(gwf, maxbound=1, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    ModflowGwfriv(gwf, stress_period_data={0: rd})
    # Create the output control package
    headfile = f"{ex01_mf6_model_name}.hds"
    head_record = [headfile]
    budgetfile = f"{ex01_mf6_model_name}.cbb"
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    # Write the datasets
    sim.write_simulation()

    return sim, function_tmpdir


@requires_exe("mf6", "mp7")
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.slow
def test_basic_mp7_model(ex01_mf6_model, direction):
    sim, function_tmpdir = ex01_mf6_model
    success, buff = sim.run_simulation()
    assert success, buff

    mpnam = f"{ex01_mf6_model_name}_mp_{direction}"

    # load the MODFLOW 6 model
    sim = MFSimulation.load("mf6mod", "mf6", "mf6", function_tmpdir)
    gwf = sim.get_model(ex01_mf6_model_name)

    mp = Modpath7.create_mp7(
        modelname=mpnam,
        trackdir=direction,
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=function_tmpdir,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    success, buff = mp.run_model()
    assert success, f"mp7 model ({mp.name}) did not run"


@requires_exe("mf2005", "mf6", "mp7")
@pytest.mark.parametrize("case", ["mf2005", "mf6"])
def test_mp7_output(function_tmpdir, case, array_snapshot):
    # build model
    if case == "mf2005":
        model = Mp7Cases.mp7_mf2005(function_tmpdir)
    else:
        model = Mp7Cases.mp7_mf6(function_tmpdir)

    # write and run model
    model.write_input()
    success, buff = model.run_model()
    assert success, buff

    # check pathline output files
    pathline_file = Path(model.model_ws) / f"ex01_{case}_mp.mppth"
    p = PathlineFile(pathline_file)
    assert p.get_maxid() == 22
    pathlines = p.get_alldata()
    assert len(pathlines) == 23
    pathlines = pd.DataFrame(np.concatenate(pathlines))
    assert pathlines.particleid.nunique() == 23
    if system() != "Darwin":
        assert array_snapshot == pathlines.round(3).to_records(index=False)

    # check endpoint output files
    endpoint_file = Path(model.model_ws) / f"ex01_{case}_mp.mpend"
    e = EndpointFile(endpoint_file)
    assert e.get_maxid() == 22
    endpoints = e.get_alldata()
    assert len(endpoints) == 23

    modelgrid = model.flowmodel.modelgrid
    nodes = list(range(modelgrid.nnodes))
    pmv = PlotMapView(modelgrid=modelgrid, layer=0)
    pmv.plot_grid()
    linecol = pmv.plot_pathline(p.get_destination_data(nodes), layer="all")
    linecol2 = pmv.plot_pathline(p.get_alldata(), layer="all")
    if not len(linecol._paths) == len(linecol2._paths):
        raise AssertionError(
            "plot_pathline not properly splitting particles from recarray"
        )
    # plt.show()
    plt.close()


@requires_exe("mf6", "mp7")
def test_mp7sim_replacement(function_tmpdir):
    mf6sim = Mp7Cases.mf6(function_tmpdir)
    mf6sim.write_simulation()
    mf6sim.run_simulation()

    # create mp7 model
    mp = Modpath7(
        modelname=f"{mf6sim.name}_mp",
        flowmodel=mf6sim.get_model(mf6sim.name),
        exe_name="mp7",
        model_ws=mf6sim.sim_path,
    )
    defaultiface6 = {"RCH": 6, "EVT": 6}
    mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface6)
    mpsim = Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        budgetoutputoption="summary",
        budgetcellnumbers=[1049, 1259],
        traceparticledata=[1, 1000],
        referencetime=[0, 0, 0.0],
        stoptimeoption="extend",
        timepointdata=[500, 1000.0],
        zonedataoption="on",
        zones=Mp7Cases.zones,
        particlegroups=Mp7Cases.particlegroups,
    )

    # add a duplicate mp7sim package
    with pytest.warns(UserWarning, match="Two packages of the same type"):
        mpsim = Modpath7Sim(
            mp,
            simulationtype="combined",
            trackingdirection="forward",
            weaksinkoption="pass_through",
            weaksourceoption="pass_through",
            budgetoutputoption="summary",
            budgetcellnumbers=[1049, 1259],
            traceparticledata=[1, 1000],
            referencetime=[0, 0, 0.0],
            stoptimeoption="extend",
            timepointdata=[500, 1000.0],
            zonedataoption="on",
            zones=Mp7Cases.zones,
            particlegroups=Mp7Cases.particlegroups,
        )

    mp.write_input()
    success, buff = mp.run_model()
    assert success, buff


@requires_exe("mf6", "mp7")
@pytest.mark.parametrize(
    "porosity_type", ("constant", "list", "array_1d", "array")
)
@pytest.mark.slow
def test_mp7bas_porosity(ex01_mf6_model, porosity_type):
    sim, function_tmpdir = ex01_mf6_model
    sim.run_simulation()

    mpnam = f"{ex01_mf6_model_name}_mp_forward"
    gwf = sim.get_model(ex01_mf6_model_name)

    if porosity_type == "constant":
        porosity = 0.30
    elif porosity_type == "list":
        porosity = [0.35, 0.30, 0.25]  # constant per layer
    elif porosity_type == "array_1d":
        porosity = np.array([0.35, 0.30, 0.25])  # constant per layer
    elif porosity_type == "array":
        # Initialize porosity array based on dim of dis.botm
        # lay_1=0.35, lay_2=0.30, lay_3=0.25
        porosity = np.array([[[0.35]], [[0.30]], [[0.25]]]) * np.ones(
            (gwf.dis.botm.array.shape)
        )
        # Diversify porosity values, adjust both halves of the model
        porosity[:, :, : porosity.shape[2] // 2] -= 0.05
        porosity[:, :, porosity.shape[2] // 2 :] += 0.05

    mp = Modpath7.create_mp7(
        modelname=mpnam,
        trackdir="forward",
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=function_tmpdir,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
        porosity=porosity,
    )

    # Check mean of assigned porosity
    assert np.isclose(np.mean(mp.get_package("MPBAS").porosity.array), 0.3)

    mp.write_input()
    success, _ = mp.run_model()
    assert success, f"mp7 model ({mp.name}) did not run"


@requires_exe("mf6", "mp7")
def test_flopy_2223(function_tmpdir):
    mf6sim = Mp7Cases.mf6(function_tmpdir)
    mf6sim.get_model().get_package("ic").strt = 0
    mf6sim.write_simulation()
    mf6sim.run_simulation()

    # create mp7 model
    mp = Modpath7(
        modelname=f"{mf6sim.name}_mp",
        flowmodel=mf6sim.get_model(mf6sim.name),
        exe_name="mp7",
        model_ws=mf6sim.sim_path,
    )
    defaultiface6 = {"RCH": 6, "EVT": 6}
    mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface6)
    part0 = ParticleData([(0, 0, 0)], structured=True, particleids=[0])
    pg0 = ParticleGroup(
        particlegroupname="PG1", particledata=part0, filename="ex01a.sloc"
    )
    mpsim = Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        budgetoutputoption="summary",
        budgetcellnumbers=[1049, 1259],
        traceparticledata=[1, 1000],
        referencetime=[0, 0, 0.0],
        stoptimeoption="extend",
        timepointdata=[500, 1000.0],
        zonedataoption="on",
        zones=Mp7Cases.zones,
        particlegroups=Mp7Cases.particlegroups,
    )

    mp.write_input()
    success, buff = mp.run_model()
    assert success, buff

    pathline_file = Path(mp.model_ws) / "ex01_mf6_mp.mppth"
    p = PathlineFile(pathline_file)
    pathlines = p.get_alldata()
    assert len(pathlines) == 2
    assert all(len(pl) > 0 for pl in pathlines)
