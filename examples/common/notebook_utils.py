import os
import sys

import numpy as np

try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", "..", ".."))
    sys.path.append(fpth)
    import flopy

from pathlib import Path


def get_project_root_path(path=None):
    """
    Infers the path to the project root given the path to the current working directory.
    The current working location must be somewhere in the project, i.e. below the root.

    Parameters
    ----------
    path : the path to the current working directory

    Returns
    -------
        The path to the project root
    """

    cwd = Path(path) if path is not None else Path(os.getcwd())
    if cwd.name == "autotest":
        # we're in top-level autotest folder
        return cwd.parent
    elif "autotest" in cwd.parts and cwd.parts.index(
        "autotest"
    ) > cwd.parts.index("flopy"):
        # we're somewhere inside autotests
        parts = cwd.parts[0 : cwd.parts.index("autotest")]
        return Path(*parts)
    elif "examples" in cwd.parts and cwd.parts.index(
        "examples"
    ) > cwd.parts.index("flopy"):
        # we're somewhere inside examples folder
        parts = cwd.parts[0 : cwd.parts.index("examples")]
        return Path(*parts)
    elif cwd.parts.count("flopy") >= 2:
        # we're somewhere inside the project or flopy module
        tries = [1]
        for t in tries:
            parts = cwd.parts[0 : cwd.parts.index("flopy") + (t)]
            pth = Path(*parts)
            if (
                next(iter([p for p in pth.glob("setup.cfg")]), None)
                is not None
            ):
                return pth
        raise Exception(
            f"Can't infer location of project root from {cwd}"
            f"(run from project root, flopy module, examples, or autotest)"
        )
    elif cwd.parts.count("flopy") == 1 and cwd.name == "flopy":
        # we're in project root
        return cwd
    else:
        raise Exception(
            f"Can't infer location of project root from {cwd}"
            f"(run from project root, flopy module, examples, or autotest)"
        )


def run(ws):
    ## load and run vertex grid example
    # run installed version of flopy or add local path
    if not os.path.exists(ws):
        os.mkdir(ws)

    from flopy.utils.gridgen import Gridgen

    Lx = 10000.0
    Ly = 10500.0
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, 0]

    ms = flopy.modflow.Modflow()
    dis5 = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    model_name = "mp7p2"
    model_ws = os.path.join(ws, "mp7_ex2", "mf6")
    gridgen_ws = os.path.join(model_ws, "gridgen")
    g = Gridgen(dis5, model_ws=gridgen_ws)

    rf0shp = os.path.join(gridgen_ws, "rf0")
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

    rf1shp = os.path.join(gridgen_ws, "rf1")
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

    rf2shp = os.path.join(gridgen_ws, "rf2")
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))

    g.build(verbose=False)

    gridprops = g.get_gridprops_disv()
    ncpl = gridprops["ncpl"]
    top = gridprops["top"]
    botm = gridprops["botm"]
    nvert = gridprops["nvert"]
    vertices = gridprops["vertices"]
    cell2d = gridprops["cell2d"]
    # cellxy = gridprops['cellxy']

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=model_name, version="mf6", exe_name="mf6", sim_ws=model_ws
    )

    # create tdis package
    tdis_rc = [(1000.0, 1, 1.0)]
    tdis = flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", perioddata=tdis_rc
    )

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )
    gwf.name_file.save_flows = True

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_hclose=1.0e-5,
        outer_maximum=100,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=1.0e-6,
        rcloserecord=0.1,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
    )
    sim.register_ims_package(ims, [gwf.name])

    # disv
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=nlay,
        ncpl=ncpl,
        top=top,
        botm=botm,
        nvert=nvert,
        vertices=vertices,
        cell2d=cell2d,
    )

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=320.0)

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        xt3doptions=[("xt3d")],
        save_specific_discharge=True,
        icelltype=[1, 0, 0],
        k=[50.0, 0.01, 200.0],
        k33=[10.0, 0.01, 20.0],
    )

    # wel
    wellpoints = [(4750.0, 5250.0)]
    welcells = g.intersect(wellpoints, "point", 0)
    # welspd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, maxbound=1, aux_vars=['iface'])
    welspd = [[(2, icpl), -150000, 0] for icpl in welcells["nodenumber"]]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=True,
        auxiliary=[("iface",)],
        stress_period_data=welspd,
    )

    # rch
    aux = [np.ones(ncpl, dtype=int) * 6]
    rch = flopy.mf6.ModflowGwfrcha(
        gwf, recharge=0.005, auxiliary=[("iface",)], aux={0: [6]}
    )
    # riv
    riverline = [[(Lx - 1.0, Ly), (Lx - 1.0, 0.0)]]
    rivcells = g.intersect(riverline, "line", 0)
    rivspd = [
        [(0, icpl), 320.0, 100000.0, 318] for icpl in rivcells["nodenumber"]
    ]
    riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=rivspd)

    # output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{model_name}.cbb",
        head_filerecord=f"{model_name}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sim.write_simulation()
    sim.run_simulation()

    mp_namea = f"{model_name}a_mp"
    mp_nameb = f"{model_name}b_mp"

    pcoord = np.array(
        [
            [0.000, 0.125, 0.500],
            [0.000, 0.375, 0.500],
            [0.000, 0.625, 0.500],
            [0.000, 0.875, 0.500],
            [1.000, 0.125, 0.500],
            [1.000, 0.375, 0.500],
            [1.000, 0.625, 0.500],
            [1.000, 0.875, 0.500],
            [0.125, 0.000, 0.500],
            [0.375, 0.000, 0.500],
            [0.625, 0.000, 0.500],
            [0.875, 0.000, 0.500],
            [0.125, 1.000, 0.500],
            [0.375, 1.000, 0.500],
            [0.625, 1.000, 0.500],
            [0.875, 1.000, 0.500],
        ]
    )
    nodew = gwf.disv.ncpl.array * 2 + welcells["nodenumber"][0]
    plocs = [nodew for i in range(pcoord.shape[0])]

    # create particle data
    pa = flopy.modpath.ParticleData(
        plocs,
        structured=False,
        localx=pcoord[:, 0],
        localy=pcoord[:, 1],
        localz=pcoord[:, 2],
        drape=0,
    )

    # create backward particle group
    fpth = f"{mp_namea}.sloc"
    pga = flopy.modpath.ParticleGroup(
        particlegroupname="BACKWARD1", particledata=pa, filename=fpth
    )

    facedata = flopy.modpath.FaceDataType(
        drape=0,
        verticaldivisions1=10,
        horizontaldivisions1=10,
        verticaldivisions2=10,
        horizontaldivisions2=10,
        verticaldivisions3=10,
        horizontaldivisions3=10,
        verticaldivisions4=10,
        horizontaldivisions4=10,
        rowdivisions5=0,
        columndivisions5=0,
        rowdivisions6=4,
        columndivisions6=4,
    )
    pb = flopy.modpath.NodeParticleData(subdivisiondata=facedata, nodes=nodew)
    # create forward particle group
    fpth = f"{mp_nameb}.sloc"
    pgb = flopy.modpath.ParticleGroupNodeTemplate(
        particlegroupname="BACKWARD2", particledata=pb, filename=fpth
    )

    # create modpath files
    mp = flopy.modpath.Modpath7(
        modelname=mp_namea, flowmodel=gwf, exe_name="mp7", model_ws=model_ws
    )
    flopy.modpath.Modpath7Bas(mp, porosity=0.1)
    flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="backward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        referencetime=0.0,
        stoptimeoption="extend",
        timepointdata=[500, 1000.0],
        particlegroups=pga,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()

    # create modpath files
    mp = flopy.modpath.Modpath7(
        modelname=mp_nameb, flowmodel=gwf, exe_name="mp7", model_ws=model_ws
    )
    flopy.modpath.Modpath7Bas(mp, porosity=0.1)
    flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="endpoint",
        trackingdirection="backward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        referencetime=0.0,
        stoptimeoption="extend",
        particlegroups=pgb,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()
    return


example_name = "ex-gwt-keating"

# Model units

length_units = "m"
time_units = "days"

# Table of model parameters

nlay = 80  # Number of layers
nrow = 1  # Number of rows
ncol = 400  # Number of columns
delr = 25.0  # Column width ($m$)
delc = 1.0  # Row width ($m$)
delz = 25.0  # Layer thickness ($m$)
top = 2000.0  # Top of model domain ($m$)
bottom = 0.0  # Bottom of model domain ($m$)
hka = 1.0e-12  # Permeability of aquifer ($m^2$)
hkc = 1.0e-18  # Permeability of aquitard ($m^2$)
h1 = 800.0  # Head on left side ($m$)
h2 = 100.0  # Head on right side ($m$)
recharge = 0.5  # Recharge ($kg/s$)
recharge_conc = 1.0  # Normalized recharge concentration (unitless)
alpha_l = 1.0  # Longitudinal dispersivity ($m$)
alpha_th = 1.0  # Transverse horizontal dispersivity ($m$)
alpha_tv = 1.0  # Transverse vertical dispersivity ($m$)
period1 = 730  # Length of first simulation period ($d$)
period2 = 29270.0  # Length of second simulation period ($d$)
porosity = 0.1  # Porosity of mobile domain (unitless)
obs1 = (49, 1, 119)  # Layer, row, and column for observation 1
obs2 = (77, 1, 359)  # Layer, row, and column for observation 2

obs1 = tuple([i - 1 for i in obs1])
obs2 = tuple([i - 1 for i in obs2])
seconds_to_days = 24.0 * 60.0 * 60.0
permeability_to_conductivity = 1000.0 * 9.81 / 1.0e-3 * seconds_to_days
hka = hka * permeability_to_conductivity
hkc = hkc * permeability_to_conductivity
botm = [top - (k + 1) * delz for k in range(nlay)]
x = np.arange(0, 10000.0, delr) + delr / 2.0
plotaspect = 1.0

# Fill hydraulic conductivity array
hydraulic_conductivity = np.ones((nlay, nrow, ncol), dtype=float) * hka
for k in range(nlay):
    if 1000.0 <= botm[k] < 1100.0:
        for j in range(ncol):
            if 3000.0 <= x[j] <= 6000.0:
                hydraulic_conductivity[k, 0, j] = hkc

# Calculate recharge by converting from kg/s to m/d
rcol = []
for jcol in range(ncol):
    if 4200.0 <= x[jcol] <= 4800.0:
        rcol.append(jcol)
number_recharge_cells = len(rcol)
rrate = recharge * seconds_to_days / 1000.0
cell_area = delr * delc
rrate = rrate / (float(number_recharge_cells) * cell_area)
rchspd = {}
rchspd[0] = [[(0, 0, j), rrate, recharge_conc] for j in rcol]
rchspd[1] = [[(0, 0, j), rrate, 0.0] for j in rcol]


def build_mf6gwf(sim_folder):
    ws = os.path.join(sim_folder, "mf6-gwt-keating")
    name = "flow"
    sim_ws = os.path.join(ws, "mf6gwf")
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_ws, exe_name="mf6")
    tdis_ds = ((period1, 1, 1.0), (period2, 1, 1.0))
    flopy.mf6.ModflowTdis(
        sim, nper=len(tdis_ds), perioddata=tdis_ds, time_units=time_units
    )
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        no_ptcrecord="all",
        outer_dvclose=1.0e-4,
        outer_maximum=2000,
        under_relaxation="dbd",
        linear_acceleration="BICGSTAB",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.08,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=2.0,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=5.0e-4,
        inner_dvclose=1.0e-5,
        rcloserecord=[0.0001, "relative_rclose"],
        inner_maximum=100,
        relaxation_factor=0.0,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions=["newton"]
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=1,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=600.0)
    chdspd = [[(k, 0, 0), h1] for k in range(nlay) if botm[k] < h1]
    chdspd += [[(k, 0, ncol - 1), h2] for k in range(nlay) if botm[k] < h2]
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chdspd,
        print_input=True,
        print_flows=True,
        save_flows=False,
        pname="CHD-1",
    )
    flopy.mf6.ModflowGwfrch(
        gwf,
        stress_period_data=rchspd,
        auxiliary=["concentration"],
        pname="RCH-1",
    )

    head_filerecord = f"{name}.hds"
    budget_filerecord = f"{name}.bud"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def build_mf6gwt(sim_folder):
    ws = os.path.join(sim_folder, "mf6-gwt-keating")
    name = "trans"
    sim_ws = os.path.join(ws, "mf6gwt")
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        sim_ws=sim_ws,
        exe_name="mf6",
        continue_=True,
    )
    tdis_ds = ((period1, 73, 1.0), (period2, 2927, 1.0))
    flopy.mf6.ModflowTdis(
        sim, nper=len(tdis_ds), perioddata=tdis_ds, time_units=time_units
    )
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        outer_dvclose=1.0e-4,
        outer_maximum=100,
        under_relaxation="none",
        linear_acceleration="BICGSTAB",
        rcloserecord=[1000.0, "strict"],
        inner_maximum=20,
        inner_dvclose=1.0e-4,
        relaxation_factor=0.0,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
    )
    gwt = flopy.mf6.ModflowGwt(sim, modelname=name, save_flows=True)
    flopy.mf6.ModflowGwtdis(
        gwt,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwtic(gwt, strt=0)
    flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    flopy.mf6.ModflowGwtadv(gwt, scheme="upstream")
    flopy.mf6.ModflowGwtdsp(
        gwt, xt3d_off=True, alh=alpha_l, ath1=alpha_th, atv=alpha_tv
    )
    pd = [
        ("GWFHEAD", "../mf6gwf/flow.hds"),
        ("GWFBUDGET", "../mf6gwf/flow.bud"),
    ]
    flopy.mf6.ModflowGwtfmi(
        gwt, flow_imbalance_correction=True, packagedata=pd
    )
    sourcerecarray = [
        ("RCH-1", "AUX", "CONCENTRATION"),
    ]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)
    saverecord = {
        0: [
            ("CONCENTRATION", "STEPS", 10),
            ("CONCENTRATION", "LAST"),
            ("CONCENTRATION", "FREQUENCY", 10),
        ],
        1: [
            ("CONCENTRATION", "STEPS", 27, 227),
            ("CONCENTRATION", "LAST"),
            ("CONCENTRATION", "FREQUENCY", 10),
        ],
    }
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{name}.cbc",
        concentration_filerecord=f"{name}.ucn",
        concentrationprintrecord=[
            ("COLUMNS", ncol, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=saverecord,
        printrecord=[
            ("CONCENTRATION", "LAST"),
            (
                "BUDGET",
                "ALL",
            ),
        ],
    )
    obs_data = {
        f"{name}.obs.csv": [
            ("obs1", "CONCENTRATION", obs1),
            ("obs2", "CONCENTRATION", obs2),
        ],
    }
    flopy.mf6.ModflowUtlobs(
        gwt, digits=10, print_input=True, continuous=obs_data
    )
    return sim


def build_model(ws):
    sim_mf6gwf = build_mf6gwf(ws)
    sim_mf6gwt = build_mf6gwt(ws)
    sim_mf2005 = None  # build_mf2005(sim_name)
    sim_mt3dms = None  # build_mt3dms(sim_name, sim_mf2005)
    sims = (sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms)
    return sims


def write_model(sims, silent=True):
    sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms = sims
    sim_mf6gwf.write_simulation(silent=silent)
    sim_mf6gwt.write_simulation(silent=silent)


def run_keating_model(ws=example_name, silent=True):
    sim = build_model(ws)
    write_model(sim, silent=silent)
    sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms = sim

    print("Running mf6gwf model...")
    success, buff = sim_mf6gwf.run_simulation(silent=silent)
    if not success:
        print(buff)

    print("Running mf6gwt model...")
    success, buff = sim_mf6gwt.run_simulation(silent=silent)
    if not success:
        print(buff)

    return success


if __name__ == "__main__":
    run()
    run_keating_model()
