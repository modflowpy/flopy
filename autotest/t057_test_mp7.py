import os
import numpy as np
import flopy
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

exe_names = {"mf2005": "mf2005", "mf6": "mf6", "mp7": "mp7"}
run = True
for key in exe_names.keys():
    v = flopy.which(exe_names[key])
    if v is None:
        run = False
        break

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

zone3 = np.ones((nrow, ncol), dtype=np.int32)
zone3[wel_loc[1:]] = 2
zones = [1, 1, zone3]

# create particles
partlocs = []
partids = []
for i in range(nrow):
    partlocs.append((0, i, 2))
    partids.append(i)
part0 = flopy.modpath.ParticleData(
    partlocs, structured=True, particleids=partids
)
pg0 = flopy.modpath.ParticleGroup(
    particlegroupname="PG1", particledata=part0, filename="ex01a.sloc"
)

v = [(0,), (400,)]
pids = [1, 2]  # [1000, 1001]
part1 = flopy.modpath.ParticleData(
    v, structured=False, drape=1, particleids=pids
)
pg1 = flopy.modpath.ParticleGroup(
    particlegroupname="PG2", particledata=part1, filename="ex01a.pg2.sloc"
)

particlegroups = [pg0, pg1]

defaultiface = {"RECHARGE": 6, "ET": 6}
defaultiface6 = {"RCH": 6, "EVT": 6}


def test_pathline_output():
    model_ws = f"{base_dir}_test_pathline_output"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    # build and run the models
    build_mf2005(model_ws)
    build_mf6(model_ws)

    # if models not run then there will be no output
    if not run:
        return

    fpth0 = os.path.join(model_ws, "mf2005", "ex01_mf2005_mp.mppth")
    p = flopy.utils.PathlineFile(fpth0)
    maxtime0 = p.get_maxtime()
    maxid0 = p.get_maxid()
    p0 = p.get_alldata()
    fpth1 = os.path.join(model_ws, "mf6", "ex01_mf6_mp.mppth")
    p = flopy.utils.PathlineFile(fpth1)
    maxtime1 = p.get_maxtime()
    maxid1 = p.get_maxid()
    p1 = p.get_alldata()

    # check maxid
    msg = (
        f"pathline maxid ({maxid0}) in {os.path.basename(fpth0)} are not "
        f"equal to the pathline maxid ({maxid1}) in {os.path.basename(fpth1)}"
    )
    assert maxid0 == maxid1, msg

    return


def test_endpoint_output():
    model_ws = f"{base_dir}_test_endpoint_output"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    # build and run the models
    build_mf2005(model_ws)
    build_mf6(model_ws)

    # if models not run then there will be no output
    if not run:
        return

    fpth0 = os.path.join(model_ws, "mf2005", "ex01_mf2005_mp.mpend")
    e = flopy.utils.EndpointFile(fpth0)
    maxtime0 = e.get_maxtime()
    maxid0 = e.get_maxid()
    maxtravel0 = e.get_maxtraveltime()
    e0 = e.get_alldata()
    fpth1 = os.path.join(model_ws, "mf6", "ex01_mf6_mp.mpend")
    e = flopy.utils.EndpointFile(fpth1)
    maxtime1 = e.get_maxtime()
    maxid1 = e.get_maxid()
    maxtravel1 = e.get_maxtraveltime()
    e1 = e.get_alldata()

    # check maxid
    msg = (
        f"endpoint maxid ({maxid0}) in {os.path.basename(fpth0)} are not "
        f"equal to the endpoint maxid ({maxid1}) in {os.path.basename(fpth1)}"
    )
    assert maxid0 == maxid1, msg

    # check that endpoint data are approximately the same
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
    d = np.rec.fromarrays((e0[name] - e1[name] for name in names), dtype=dtype)
    msg = (
        f"endpoints in {os.path.basename(fpth0)} are not equal (within 1e-5) "
        f"to the endpoints  in {os.path.basename(fpth1)}"
    )
    # assert not np.allclose(t0, t1), msg

    return


def build_mf2005(model_ws):
    """
    MODPATH 7 example 1 for MODFLOW-2005
    """

    ws = os.path.join(model_ws, "mf2005")
    nm = "ex01_mf2005"
    exe_name = exe_names["mf2005"]
    iu_cbc = 130
    m = flopy.modflow.Modflow(nm, model_ws=ws, exe_name=exe_name)
    flopy.modflow.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        itmuni=4,
        lenuni=1,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        steady=True,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.modflow.ModflowLpf(m, ipakcb=iu_cbc, laytyp=laytyp, hk=kh, vka=kv)
    flopy.modflow.ModflowBas(m, ibound=1, strt=top)
    # recharge
    flopy.modflow.ModflowRch(m, ipakcb=iu_cbc, rech=rch, nrchop=1)
    # wel
    wd = [i for i in wel_loc] + [wel_q]
    flopy.modflow.ModflowWel(m, ipakcb=iu_cbc, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([0, i, ncol - 1, riv_h, riv_c, riv_z])
    flopy.modflow.ModflowRiv(m, ipakcb=iu_cbc, stress_period_data={0: rd})
    # output control
    flopy.modflow.ModflowOc(
        m,
        stress_period_data={
            (0, 0): ["save head", "save budget", "print head"]
        },
    )
    flopy.modflow.ModflowPcg(m, hclose=1e-6, rclose=1e-3, iter1=100, mxiter=50)

    m.write_input()

    if run:
        success, buff = m.run_model()
        assert success, "mf2005 model did not run"

    # create modpath files
    exe_name = exe_names["mp7"]
    mp = flopy.modpath.Modpath7(
        modelname=f"{nm}_mp", flowmodel=m, exe_name=exe_name, model_ws=ws
    )
    mpbas = flopy.modpath.Modpath7Bas(
        mp, porosity=0.1, defaultiface=defaultiface
    )
    mpsim = flopy.modpath.Modpath7Sim(
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
        zones=zones,
        particlegroups=particlegroups,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    if run:
        success, buff = mp.run_model()
        assert success, f"mp7 model ({mp.name}) did not run"

    return


def build_mf6(model_ws):
    """
    MODPATH 7 example 1 for MODFLOW 6
    """

    ws = os.path.join(model_ws, "mf6")
    nm = "ex01_mf6"
    exe_name = exe_names["mf6"]

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=nm, exe_name="mf6", version="mf6", sim_ws=ws
    )

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{nm}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=nm, model_nam_file=model_nam_file, save_flows=True
    )

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.modflow.mfims.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        inner_hclose=1e-6,
        rcloserecord=1e-3,
        outer_hclose=1e-6,
        outer_maximum=50,
        inner_maximum=100,
    )

    # create gwf file
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
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
    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=top)

    # Create the node property flow package
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv
    )

    # recharge
    flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(
        gwf, maxbound=1, stress_period_data={0: wd}
    )
    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(gwf, stress_period_data={0: rd})
    # Create the output control package
    headfile = f"{nm}.hds"
    head_record = [headfile]
    budgetfile = f"{nm}.cbb"
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    # Write the datasets
    sim.write_simulation()

    # Run the simulation
    if run:
        success, buff = sim.run_simulation()
        assert success, "mf6 model did not run"

    # create modpath files
    exe_name = exe_names["mp7"]
    mp = flopy.modpath.Modpath7(
        modelname=f"{nm}_mp", flowmodel=gwf, exe_name=exe_name, model_ws=ws
    )
    mpbas = flopy.modpath.Modpath7Bas(
        mp, porosity=0.1, defaultiface=defaultiface6
    )
    mpsim = flopy.modpath.Modpath7Sim(
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
        zones=zones,
        particlegroups=particlegroups,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    if run:
        success, buff = mp.run_model()
        assert success, f"mp7 model ({mp.name}) did not run"

    return


if __name__ == "__main__":
    test_pathline_output()
    test_endpoint_output()
