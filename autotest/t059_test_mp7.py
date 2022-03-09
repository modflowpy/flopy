from ci_framework import FlopyTestSetup, base_test_dir

import flopy

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

exe_names = {"mf6": "mf6", "mp7": "mp7"}
run = True
for key in exe_names.keys():
    v = flopy.which(exe_names[key])
    if v is None:
        run = False
        break

nm = "ex01_mf6"

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


def test_forward():
    model_ws = f"{base_dir}_test_forward"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)
    build_mf6(model_ws)

    mpnam = f"{nm}_mp_forward"
    exe_name = exe_names["mp7"]

    # load the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation.load("mf6mod", "mf6", "mf6", model_ws)
    gwf = sim.get_model(nm)

    mp = flopy.modpath.Modpath7.create_mp7(
        modelname=mpnam,
        trackdir="forward",
        flowmodel=gwf,
        exe_name=exe_name,
        model_ws=model_ws,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
    )

    # build and run the MODPATH 7 models
    build_modpath(mp)
    return


def test_backward():
    model_ws = f"{base_dir}_test_backward"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)
    build_mf6(model_ws)

    mpnam = f"{nm}_mp_backward"
    exe_name = exe_names["mp7"]

    # load the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation.load("mf6mod", "mf6", "mf6", model_ws)
    gwf = sim.get_model(nm)

    mp = flopy.modpath.Modpath7.create_mp7(
        modelname=mpnam,
        trackdir="backward",
        flowmodel=gwf,
        exe_name=exe_name,
        model_ws=model_ws,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
    )

    # build and run the MODPATH 7 models
    build_modpath(mp)
    return


def build_mf6(model_ws):
    """
    MODPATH 7 example 1 for MODFLOW 6
    """

    exe_name = exe_names["mf6"]

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=nm,
        exe_name="mf6",
        version="mf6",
        sim_ws=model_ws,
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
        sim, pname="ims", complexity="SIMPLE"
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


def build_modpath(mp):
    # write modpath datasets
    mp.write_input()

    # run modpath
    if run:
        success, buff = mp.run_model()
        assert success, f"mp7 model ({mp.name}) did not run"

    return


if __name__ == "__main__":
    # build and run modflow 6
    test_mf6()

    # build forward tracking model
    test_forward()

    # build forward tracking model
    test_backward()
