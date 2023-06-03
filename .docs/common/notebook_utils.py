import os
import sys
from pathlib import Path

import numpy as np

try:
    import flopy
except ImportError:
    fpth = os.path.abspath(os.path.join("..", "..", ".."))
    sys.path.append(fpth)
    import flopy


def get_project_root_path() -> Path:
    return Path.cwd().parent.parent

geometries = {
    "boundary": """1.868012422360248456e+05 4.695652173913043953e+04
    1.790372670807453396e+05 5.204968944099379587e+04
    1.729813664596273447e+05 5.590062111801243009e+04
    1.672360248447204940e+05 5.987577639751553215e+04
    1.631987577639751253e+05 6.335403726708075556e+04
    1.563664596273291972e+05 6.819875776397516893e+04
    1.509316770186335489e+05 7.229813664596274612e+04
    1.453416149068323139e+05 7.527950310559007630e+04
    1.395962732919254631e+05 7.627329192546584818e+04
    1.357142857142857101e+05 7.664596273291927355e+04
    1.329192546583850926e+05 7.751552795031057030e+04
    1.268633540372670832e+05 8.062111801242237561e+04
    1.218944099378881947e+05 8.285714285714286962e+04
    1.145962732919254486e+05 8.571428571428572468e+04
    1.069875776397515583e+05 8.869565217391305487e+04
    1.023291925465838431e+05 8.931677018633540138e+04
    9.456521739130433707e+04 9.068322981366459862e+04
    8.804347826086955320e+04 9.080745341614908830e+04
    7.950310559006211406e+04 9.267080745341615693e+04
    7.562111801242236106e+04 9.391304347826087906e+04
    6.692546583850930620e+04 9.602484472049689793e+04
    5.667701863354037778e+04 9.763975155279504543e+04
    4.906832298136646568e+04 9.689440993788820924e+04
    3.897515527950309479e+04 9.540372670807455142e+04
    3.167701863354036323e+04 9.304347826086958230e+04
    2.375776397515527788e+04 8.757763975155279331e+04
    1.847826086956521613e+04 8.161490683229814749e+04
    1.164596273291925172e+04 7.739130434782608063e+04
    6.211180124223596977e+03 7.055900621118013805e+04
    4.347826086956512881e+03 6.422360248447205959e+04
    1.863354037267072272e+03 6.037267080745341809e+04
    2.639751552795024509e+03 5.602484472049689793e+04
    1.552795031055893560e+03 5.279503105590062478e+04
    7.763975155279410956e+02 4.186335403726709046e+04
    2.018633540372667312e+03 3.813664596273292409e+04
    6.055900621118013078e+03 3.341614906832297856e+04
    1.335403726708074100e+04 2.782608695652173992e+04
    2.577639751552794405e+04 2.086956521739130767e+04
    3.416149068322980747e+04 1.763975155279503815e+04
    4.642857142857142753e+04 1.440993788819875044e+04
    5.636645962732918997e+04 1.130434782608694877e+04
    6.459627329192546313e+04 9.813664596273290954e+03
    8.555900621118012350e+04 6.832298136645956220e+03
    9.829192546583850344e+04 5.093167701863346338e+03
    1.085403726708074391e+05 4.347826086956525614e+03
    1.200310559006211115e+05 4.223602484472040487e+03
    1.296583850931677007e+05 4.347826086956525614e+03
    1.354037267080745369e+05 5.590062111801232277e+03
    1.467391304347825935e+05 1.267080745341615875e+04
    1.563664596273291972e+05 1.937888198757762802e+04
    1.630434782608695677e+05 2.198757763975155467e+04
    1.694099378881987650e+05 2.434782608695652743e+04
    1.782608695652173774e+05 2.981366459627329095e+04
    1.833850931677018234e+05 3.180124223602484562e+04
    1.868012422360248456e+05 3.577639751552795497e+04""",
    "streamseg1": """1.868012422360248456e+05 4.086956521739130403e+04
    1.824534161490683327e+05 4.086956521739130403e+04
    1.770186335403726553e+05 4.124223602484472940e+04
    1.737577639751552779e+05 4.186335403726709046e+04
    1.703416149068323139e+05 4.310559006211180531e+04
    1.670807453416148783e+05 4.397515527950310934e+04
    1.636645962732919143e+05 4.484472049689441337e+04
    1.590062111801242281e+05 4.559006211180124228e+04
    1.555900621118012350e+05 4.559006211180124228e+04
    1.510869565217391064e+05 4.546583850931677443e+04
    1.479813664596273156e+05 4.534161490683229931e+04
    1.453416149068323139e+05 4.496894409937888850e+04
    1.377329192546583654e+05 4.447204968944099528e+04
    1.326086956521739194e+05 4.447204968944099528e+04
    1.285714285714285652e+05 4.434782608695652743e+04
    1.245341614906832110e+05 4.472049689440993825e+04
    1.215838509316770069e+05 4.509316770186335634e+04
    1.161490683229813585e+05 4.509316770186335634e+04
    1.125776397515527933e+05 4.459627329192547040e+04
    1.074534161490683036e+05 4.385093167701864149e+04
    1.018633540372670686e+05 4.347826086956522340e+04
    9.798136645962731563e+04 4.360248447204969125e+04
    9.223602484472049400e+04 4.310559006211180531e+04
    8.602484472049689793e+04 4.198757763975155831e+04
    7.981366459627327276e+04 4.173913043478261534e+04
    7.468944099378881219e+04 4.248447204968944425e+04
    7.034161490683228476e+04 4.385093167701864149e+04
    6.785714285714285506e+04 4.621118012422360334e+04
    6.583850931677018525e+04 4.919254658385094081e+04
    6.319875776397513982e+04 5.192546583850932075e+04
    6.009316770186335634e+04 5.677018633540373412e+04
    5.605590062111800216e+04 5.950310559006211406e+04
    5.279503105590060295e+04 6.124223602484472940e+04
    4.751552795031056303e+04 6.211180124223603343e+04
    3.990683229813664366e+04 6.335403726708075556e+04
    3.276397515527949508e+04 6.409937888198757719e+04
    2.934782608695651652e+04 6.509316770186336362e+04
    2.546583850931676716e+04 6.832298136645962950e+04""",
    "streamseg2": """7.025161490683228476e+04 4.375093167701864149e+04
    6.816770186335404287e+04 4.273291925465839449e+04
    6.490683229813665093e+04 4.211180124223603343e+04
    6.164596273291925900e+04 4.173913043478262261e+04
    5.776397515527951327e+04 4.124223602484472940e+04
    5.450310559006211406e+04 4.049689440993789322e+04
    4.984472049689442065e+04 3.937888198757764621e+04
    4.534161490683231386e+04 3.801242236024845624e+04
    4.114906832298137306e+04 3.664596273291926627e+04
    3.913043478260868869e+04 3.565217391304348712e+04
    3.649068322981366509e+04 3.416149068322981475e+04
    3.322981366459628043e+04 3.242236024844721760e+04
    3.012422360248447148e+04 3.105590062111801672e+04
    2.608695652173913550e+04 2.957521739130435890e+04""",
    "streamseg3": """1.059006211180124228e+05 4.335403726708074828e+04
    1.029503105590062187e+05 4.223602484472050128e+04
    1.004658385093167890e+05 4.024844720496894297e+04
    9.937888198757765349e+04 3.788819875776398112e+04
    9.627329192546584818e+04 3.490683229813664366e+04
    9.285714285714286962e+04 3.316770186335403559e+04
    8.897515527950311662e+04 3.093167701863354159e+04
    8.338509316770188161e+04 2.795031055900621504e+04
    7.872670807453416637e+04 2.670807453416148928e+04
    7.329192546583851799e+04 2.385093167701863058e+04
    6.863354037267081731e+04 2.111801242236025064e+04
    6.304347826086958230e+04 1.863354037267081003e+04""",
    "streamseg4": """1.371118012422360480e+05 4.472049689440994553e+04
    1.321428571428571595e+05 4.720496894409938250e+04
    1.285714285714285652e+05 4.981366459627330187e+04
    1.243788819875776535e+05 5.341614906832298584e+04
    1.189440993788819906e+05 5.540372670807454415e+04
    1.125776397515527933e+05 5.627329192546584818e+04
    1.065217391304347839e+05 5.726708074534162733e+04
    1.020186335403726698e+05 5.913043478260870324e+04
    9.409937888198759174e+04 6.273291925465840177e+04
    9.192546583850932075e+04 6.633540372670808574e+04
    8.881987577639751544e+04 7.242236024844722124e+04
    8.586956521739131131e+04 7.552795031055902655e+04
    8.369565217391305487e+04 7.962732919254660374e+04"""
}

def string2geom(geostring, conversion=None):
    if conversion is None:
        multiplier = 1.0
    else:
        multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        line = line.strip()
        line = line.split(" ")
        x = float(line[0]) * multiplier
        y = float(line[1]) * multiplier
        res.append((x, y))
    return res


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
    g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)

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
    success, buff = sim.run_simulation(silent=True, report=True)
    if success:
        for line in buff:
            print(line)
    else:
        raise ValueError("Failed to run.")

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
    success, buff = mp.run_model(silent=True, report=True)
    if success:
        for line in buff:
            print(line)
    else:
        raise ValueError("Failed to run.")

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
    success, buff = mp.run_model(silent=True, report=True)
    if success:
        for line in buff:
            print(line)
    else:
        raise ValueError("Failed to run.")


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
