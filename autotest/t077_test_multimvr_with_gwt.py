import os
import numpy as np
import shutil
import math


try:
    import flopy
    from flopy.utils.lgrutil import Lgr
except:
    msg = "Error. FloPy package is not available.\n"
    msg += "Try installing using the following command:\n"
    msg += " pip install flopy"
    raise Exception(msg)

mf6exe = "mf6"

name = "lgr"
# mvr_scens = ["mltmvr", "mltmvr5050", "mltmvr7525"]
ws = os.path.join("temp", name)
exdirs = [ws]
sim_workspaces = []
gwf_names = []

# ----------------
# Universal input
# ----------------
numdays = 1
perlen = [1] * numdays
nper = len(perlen)
nstp = [1] * numdays
tsmult = [1.0] * numdays

icelltype = [1, 0, 0]

# Aquifer properties
hk = 1
k33 = 1

# Transport related
mixelm = -1
al = 10
ath1 = al
atv = al
prsity = 0.2
rhob = 1.5
Kd = 0.176

# Solver settings
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 0.97

# ------------------------------------------
# Static input associated with parent model
# ------------------------------------------
nlayp = 3
nrowp = 15
ncolp = 15
delrp = 1544.1 / ncolp
delcp = 1029.4 / nrowp
x = [round(x, 3) for x in np.linspace(50.0, 45.0, ncolp)]
topp = np.repeat(x, nrowp).reshape((15, 15)).T
z = [round(z, 3) for z in np.linspace(50.0, 0.0, nlayp + 1)]
botmp = [topp - z[len(z) - 2], topp - z[len(z) - 3], topp - z[0]]
idomainp = np.ones((nlayp, nrowp, ncolp), dtype=np.int)
# Zero out where the child grid will reside
idomainp[0:2, 6:11, 2:8] = 0


# ------------------------------------------
# Common SFR data for all parent models
# ------------------------------------------

# Package_data information
sfrcells = [
    (0, 0, 1),
    (0, 1, 1),
    (0, 2, 1),
    (0, 2, 2),
    (0, 3, 2),
    (0, 4, 2),
    (0, 4, 3),
    (0, 5, 3),
    (0, 8, 8),
    (0, 8, 9),
    (0, 8, 10),
    (0, 8, 11),
    (0, 7, 11),
    (0, 7, 12),
    (0, 6, 12),
    (0, 6, 13),
    (0, 6, 14),
    (0, 5, 14),
]
rlen = [
    65.613029,
    72.488609,
    81.424789,
    35.850410,
    75.027390,
    90.887520,
    77.565651,
    74.860397,
    120.44695,
    112.31332,
    109.00368,
    91.234566,
    67.486000,
    24.603355,
    97.547943,
    104.97595,
    8.9454498,
    92.638367,
]
rwid = 5
rgrd1 = 0.12869035e-02
rgrd2 = 0.12780087e-02
rbtp = [
    49.409676,
    49.320812,
    49.221775,
    49.146317,
    49.074970,
    48.968212,
    48.859821,
    48.761742,
    45.550678,
    45.401943,
    45.260521,
    45.132568,
    45.031143,
    44.972298,
    44.894241,
    44.764832,
    44.692032,
    44.627121,
]
rbth = 1.5
rbhk = 0.1
man = 0.04
ustrf = 1.0
ndv = 0

# -----------------------------------------------
# Child model SFR data (common to all scenarios)
# -----------------------------------------------
connsc = []
for i in np.arange(89):
    if i == 0:
        connsc.append((i, -1 * (i + 1)))
    elif i == 88:
        connsc.append((i, i - 1))
    else:
        connsc.append((i, i - 1, -1 * (i + 1)))

# Package_data information
sfrcellsc = [
    (0, 0, 3),
    (0, 1, 3),
    (0, 1, 2),
    (0, 2, 2),
    (0, 2, 1),
    (0, 3, 1),
    (0, 4, 1),
    (0, 5, 1),
    (0, 6, 1),
    (0, 7, 1),
    (0, 7, 2),
    (0, 7, 3),
    (0, 7, 4),
    (0, 6, 4),
    (0, 5, 4),
    (0, 4, 4),
    (0, 3, 4),
    (0, 3, 5),
    (0, 3, 6),
    (0, 4, 6),
    (0, 4, 7),
    (0, 5, 7),
    (0, 5, 8),
    (0, 6, 8),
    (0, 7, 8),
    (0, 7, 7),
    (0, 8, 7),
    (0, 8, 6),
    (0, 8, 5),
    (0, 8, 4),
    (0, 9, 4),
    (0, 9, 3),
    (0, 10, 3),
    (0, 11, 3),
    (0, 12, 3),
    (0, 13, 3),
    (0, 13, 4),
    (0, 14, 4),
    (0, 14, 5),
    (0, 14, 6),
    (0, 13, 6),
    (0, 13, 7),
    (0, 12, 7),
    (0, 11, 7),
    (0, 11, 8),
    (0, 10, 8),
    (0, 9, 8),
    (0, 8, 8),
    (0, 7, 8),
    (0, 7, 9),
    (0, 6, 9),
    (0, 5, 9),
    (0, 4, 9),
    (0, 3, 9),
    (0, 2, 9),
    (0, 2, 10),
    (0, 1, 10),
    (0, 0, 10),
    (0, 0, 11),
    (0, 0, 12),
    (0, 0, 13),
    (0, 1, 13),
    (0, 2, 13),
    (0, 3, 13),
    (0, 4, 13),
    (0, 5, 13),
    (0, 6, 13),
    (0, 6, 12),
    (0, 7, 12),
    (0, 8, 12),
    (0, 9, 12),
    (0, 10, 12),
    (0, 11, 12),
    (0, 12, 12),
    (0, 12, 13),
    (0, 13, 13),
    (0, 13, 14),
    (0, 13, 15),
    (0, 12, 15),
    (0, 11, 15),
    (0, 10, 15),
    (0, 10, 16),
    (0, 9, 16),
    (0, 9, 15),
    (0, 8, 15),
    (0, 7, 15),
    (0, 6, 15),
    (0, 6, 16),
    (0, 6, 17),
]

rlenc = [
    24.637711,
    31.966246,
    26.376442,
    11.773884,
    22.921772,
    24.949730,
    23.878050,
    23.190311,
    24.762365,
    24.908625,
    34.366299,
    37.834534,
    6.7398176,
    25.150850,
    22.888292,
    24.630053,
    24.104542,
    35.873375,
    20.101446,
    35.636936,
    39.273537,
    7.8477302,
    15.480835,
    22.883194,
    6.6126003,
    31.995899,
    9.4387379,
    35.385513,
    35.470993,
    23.500074,
    18.414469,
    12.016913,
    24.691732,
    23.105467,
    23.700483,
    19.596104,
    5.7555680,
    34.423119,
    36.131992,
    7.4424477,
    35.565659,
    1.6159637,
    32.316132,
    20.131876,
    6.5242062,
    25.575630,
    25.575630,
    24.303566,
    1.9158504,
    21.931326,
    23.847176,
    23.432203,
    23.248718,
    23.455051,
    15.171843,
    11.196334,
    34.931976,
    4.4492774,
    36.034172,
    38.365566,
    0.8766859,
    30.059759,
    25.351671,
    23.554117,
    24.691738,
    26.074226,
    13.542957,
    13.303432,
    28.145079,
    24.373089,
    23.213642,
    23.298107,
    24.627758,
    27.715137,
    1.7645065,
    39.549232,
    37.144009,
    14.943290,
    24.851254,
    23.737432,
    15.967736,
    10.632832,
    11.425938,
    20.009295,
    24.641207,
    27.960585,
    4.6452723,
    36.717735,
    34.469074,
]
rwidc = 5
rgrdc = 0.14448310e-02
rbtpc = [
    48.622822,
    48.581932,
    48.539783,
    48.512222,
    48.487160,
    48.452576,
    48.417301,
    48.383297,
    48.348656,
    48.312775,
    48.269951,
    48.217793,
    48.185593,
    48.162552,
    48.127850,
    48.093521,
    48.058315,
    48.014984,
    47.974548,
    47.934284,
    47.880165,
    47.846127,
    47.829273,
    47.801556,
    47.780251,
    47.752357,
    47.722424,
    47.690044,
    47.638855,
    47.596252,
    47.565975,
    47.543991,
    47.517471,
    47.482941,
    47.449127,
    47.417850,
    47.399536,
    47.370510,
    47.319538,
    47.288059,
    47.256992,
    47.230129,
    47.205616,
    47.167728,
    47.148472,
    47.125282,
    47.088329,
    47.052296,
    47.033356,
    47.016129,
    46.983055,
    46.948902,
    46.915176,
    46.881439,
    46.853535,
    46.834484,
    46.801159,
    46.772713,
    46.743465,
    46.689716,
    46.661369,
    46.639019,
    46.598988,
    46.563660,
    46.528805,
    46.492130,
    46.463512,
    46.444118,
    46.414173,
    46.376232,
    46.341858,
    46.308254,
    46.273632,
    46.235821,
    46.214523,
    46.184677,
    46.129272,
    46.091644,
    46.062897,
    46.027794,
    45.999111,
    45.979897,
    45.963959,
    45.941250,
    45.908993,
    45.870995,
    45.847439,
    45.817558,
    45.766132,
]
rbthc = 1.5
rbhkc = 0.1
manc = 0.04
ustrfc = 1.0
ndvc = 0


# ---------------------------------------------------
# Parent model mover conns
# ---------------------------------------------------
connsp_mvr = [
    (0, -1),
    (1, 0, -2),
    (2, 1, -3),
    (3, 2, -4),
    (4, 3, -5),
    (5, 4, -6),
    (6, 5, -7),
    (7, 6),
    (8, -9),
    (9, 8, -10),
    (10, 9, -11),
    (11, 10, -12),
    (12, 11, -13),
    (13, 12, -14),
    (14, 13, -15),
    (15, 14),
    (16, -17),
    (17, 16),
]

# ---------------------------------------------------
# Scenario specific MVR connection data
# (for simulation- and gwf-level MVRs)
# ---------------------------------------------------
# parent model gwf mvr
# static data
mvrpack = [["WEL-1"], ["SFR-parent"]]
maxpackages = len(mvrpack)
maxmvr = 10

# scenario specific data
parent_mvr_frac = [None, 0.50, 0.75]


def get_parent_mvr_info():
    # return the appropriate mvr info for the current scenario
    mvrperioddata = [
        ("WEL-1", 0, "SFR-parent", 10, "FACTOR", 1.0),
        ("SFR-parent", 15, "SFR-parent", 16, "FACTOR", 0.5),
    ]

    mvrspd = {0: mvrperioddata}

    return mvrspd


# child model gwf mvr_scen
mvrpackc = [["WEL-2"], ["SFR-child"]]
maxpackagesc = len(mvrpackc)
mvrperioddatac = [("WEL-2", 0, "SFR-child", 53, "FACTOR", 1.0)]
mvrspdc = {0: mvrperioddatac}


# simulation mvr
def generate_parentmod_sfr_input(conns):
    pkdat = []
    for i in np.arange(len(rlen)):
        if i < 8:
            rgrd = rgrd1
        else:
            rgrd = rgrd2

        cln_list = len(
            [itm for itm in conns[i] if itm is not None and itm is not np.nan]
        )
        ncon = cln_list - 1
        pkdat.append(
            (
                i,
                sfrcells[i],
                rlen[i],
                rwid,
                rgrd,
                rbtp[i],
                rbth,
                rbhk,
                man,
                ncon,
                ustrf,
                ndv,
            )
        )

    return pkdat


def generate_childmod_sfr_input():
    pkdatc = []
    for i in np.arange(len(rlenc)):
        cln_list = len(
            [itm for itm in connsc[i] if itm is not None and itm is not np.nan]
        )
        nconc = cln_list - 1
        pkdatc.append(
            (
                i,
                sfrcellsc[i],
                rlenc[i],
                rwidc,
                rgrdc,
                rbtpc[i],
                rbthc,
                rbhkc,
                manc,
                nconc,
                ustrfc,
                ndvc,
            )
        )

    return pkdatc


def instantiate_base_models(scen_nam, gwfname, gwfnamec):
    # All pckgseen 3 test models the same except for parent model SFR input
    # static model data
    scen_ws = ws
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        version="mf6",
        exe_name=mf6exe,
        sim_ws=scen_ws,
        continue_=False,
    )

    # Instantiate time discretization package
    tdis_rc = []
    for i in range(len(perlen)):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=tdis_rc
    )

    # Instantiate the gwf model (parent model)
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        newtonoptions=True,
        model_nam_file="{}.nam".format(gwfname),
    )

    # Create iterative model solution and register the gwf model with it
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="{}.ims".format(gwfname),
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    # Instantiate the discretization package
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlayp,
        nrow=nrowp,
        ncol=ncolp,
        delr=delrp,
        delc=delcp,
        top=topp,
        botm=botmp,
        idomain=idomainp,
        filename="{}.dis".format(gwfname),
    )

    # Instantiate initial conditions package
    strt = [topp - 0.25, topp - 0.25, topp - 0.25]
    ic = flopy.mf6.ModflowGwfic(
        gwf, strt=strt, filename="{}.ic".format(gwfname)
    )

    # Instantiate node property flow package
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=False,
        alternative_cell_averaging="AMT-LMK",
        icelltype=icelltype,
        k=hk,
        k33=k33,
        save_specific_discharge=False,
        filename="{}.npf".format(gwfname),
    )

    # output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="{}.bud".format(gwfname),
        head_filerecord="{}.hds".format(gwfname),
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    # Instantiate constant head package
    rowList = np.arange(0, nrowp).tolist()
    layList = np.arange(0, nlayp).tolist()
    chdspd_left = []
    chdspd_right = []

    # Loop through rows, the left & right sides will appear in separate,
    # dedicated packages
    hd_left = 49.75
    hd_right = 44.75
    for l in layList:
        for r in rowList:
            # first, do left side of model
            chdspd_left.append([(l, r, 0), hd_left])
            # finally, do right side of model
            chdspd_right.append([(l, r, ncolp - 1), hd_right])

    chdspd = {0: chdspd_left}
    chd1 = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        pname="CHD-1",
        filename="{}.chd1.chd".format(gwfname),
    )
    chdspd = {0: chdspd_right}
    chd2 = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=False,
        pname="CHD-2",
        filename="{}.chd2.chd".format(gwfname),
    )

    welspd_mf6 = []
    #                 [(layer,   row, column), flow, conc]
    welspd_mf6.append([(3 - 1, 8 - 1, 10 - 1), -5.0, 0.0])
    wel_mf6_spd = {0: welspd_mf6}
    maxbound = len(welspd_mf6)
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=False,
        print_flows=True,
        maxbound=maxbound,
        mover=True,
        auto_flow_reduce=0.1,
        stress_period_data=wel_mf6_spd,  # wel_spd established in the MVR setup
        boundnames=False,
        save_flows=True,
        pname="WEL-1",
        auxiliary="CONCENTRATION",
        filename="{}.wel".format(gwfname),
    )

    # ---------------------------
    # Now work on the child grid
    # ---------------------------
    ncpp = 3
    ncppl = [3, 3, 0]

    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=0.0,
        yllp=0.0,
    )

    # Get child grid info:
    delrc, delcc = lgr.get_delr_delc()
    idomainc = lgr.get_idomain()  # child idomain
    topc, botmc = lgr.get_top_botm()  # top/bottom of child grid

    # Instantiate the gwf model (child model)
    gwfc = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfnamec,
        save_flows=True,
        newtonoptions=True,
        model_nam_file="{}.nam".format(gwfnamec),
    )

    # Instantiate the discretization package
    child_dis_shp = lgr.get_shape()
    nlayc = child_dis_shp[0]
    nrowc = child_dis_shp[1]
    ncolc = child_dis_shp[2]
    disc = flopy.mf6.ModflowGwfdis(
        gwfc,
        nlay=nlayc,
        nrow=nrowc,
        ncol=ncolc,
        delr=delrc,
        delc=delcc,
        top=topc,
        botm=botmc,
        idomain=idomainc,
        filename="{}.dis".format(gwfnamec),
    )

    # Instantiate initial conditions package
    strtc = [
        topc - 0.25,
        topc - 0.25,
        topc - 0.25,
        topc - 0.25,
        topc - 0.25,
        topc - 0.25,
    ]
    icc = flopy.mf6.ModflowGwfic(
        gwfc, strt=strtc, filename="{}.ic".format(gwfnamec)
    )

    # Instantiate node property flow package
    icelltypec = [1, 0, 0, 0, 0, 0]
    npfc = flopy.mf6.ModflowGwfnpf(
        gwfc,
        save_flows=False,
        alternative_cell_averaging="AMT-LMK",
        icelltype=icelltypec,
        k=hk,
        k33=k33,
        save_specific_discharge=False,
        filename="{}.npf".format(gwfnamec),
    )

    # output control
    occ = flopy.mf6.ModflowGwfoc(
        gwfc,
        budget_filerecord="{}.bud".format(gwfnamec),
        head_filerecord="{}.hds".format(gwfnamec),
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    welspd_mf6c = []
    #                 [(layer,   row, column),  flow, conc]
    welspd_mf6c.append([(6 - 1, 4 - 1, 9 - 1), -10.0, 0.0])
    wel_mf6_spdc = {0: welspd_mf6c}
    maxboundc = len(welspd_mf6c)
    welc = flopy.mf6.ModflowGwfwel(
        gwfc,
        print_input=False,
        print_flows=True,
        maxbound=maxboundc,
        mover=True,
        auto_flow_reduce=0.1,
        stress_period_data=wel_mf6_spdc,  # wel_spd established in the MVR setup
        boundnames=False,
        save_flows=True,
        pname="WEL-2",
        auxiliary="CONCENTRATION",
        filename="{}.wel".format(gwfnamec),
    )

    return sim, gwf, gwfc, lgr


def add_parent_sfr(gwf, gwfname, conns):
    # Instatiate a scenario-specific sfr package
    pkdat = generate_parentmod_sfr_input(conns)
    sfrspd = {0: [[0, "INFLOW", 40.0]]}
    sfr = flopy.mf6.ModflowGwfsfr(
        gwf,
        print_stage=False,
        print_flows=True,
        budget_filerecord=gwfname + ".sfr.bud",
        save_flows=True,
        mover=True,
        pname="SFR-parent",
        unit_conversion=86400.00,
        boundnames=False,
        nreaches=len(conns),
        packagedata=pkdat,
        connectiondata=conns,
        perioddata=sfrspd,
        filename="{}.sfr".format(gwfname),
    )


def add_child_sfr(gwfc, gwfnamec):
    # Instantiate child model sfr package (same for all scenarios)
    pkdatc = generate_childmod_sfr_input()
    sfrspd = {0: [[0, "INFLOW", 0.0]]}
    sfrc = flopy.mf6.ModflowGwfsfr(
        gwfc,
        print_stage=False,
        print_flows=True,
        budget_filerecord=gwfnamec + ".sfr.bud",
        save_flows=True,
        mover=True,
        pname="SFR-child",
        unit_conversion=86400.00,
        boundnames=False,
        nreaches=len(connsc),
        packagedata=pkdatc,
        connectiondata=connsc,
        perioddata=sfrspd,
        filename="{}.sfr".format(gwfnamec),
    )


def add_parent_gwf_mvr(gwf, gwfname):
    # get scenario specific mvr data
    mvrspd = get_parent_mvr_info()
    mvr = flopy.mf6.ModflowGwfmvr(
        gwf,
        maxmvr=maxmvr,
        print_flows=True,
        maxpackages=maxpackages,
        packages=mvrpack,
        perioddata=mvrspd,
        budget_filerecord=gwfname + ".mvr.bud",
        filename="{}.mvr".format(gwfname),
    )


def add_child_gwf_mvr(gwfc, gwfnamec):
    mvrc = flopy.mf6.ModflowGwfmvr(
        gwfc,
        maxmvr=maxmvr,
        print_flows=True,
        maxpackages=maxpackagesc,
        packages=mvrpackc,
        perioddata=mvrspdc,
        budget_filerecord=gwfnamec + ".mvr.bud",
        filename="{}.mvr".format(gwfnamec),
    )


def add_sim_mvr(sim, gwfname, gwfnamec):
    # simulation-level mvr data
    mvrpack_sim = [[gwfname, "SFR-parent"], [gwfnamec, "SFR-child"]]
    maxpackages_sim = len(mvrpack_sim)

    # Set up static SFR-to-SFR connections that remain fixed for entire simulation
    sim_mvr_perioddata = [  # don't forget to use 0-based values
        [
            mvrpack_sim[0][0],
            mvrpack_sim[0][1],
            7,
            mvrpack_sim[1][0],
            mvrpack_sim[1][1],
            0,
            "FACTOR",
            1.00,
        ],
        [
            mvrpack_sim[1][0],
            mvrpack_sim[1][1],
            88,
            mvrpack_sim[0][0],
            mvrpack_sim[0][1],
            8,
            "FACTOR",
            1.00,
        ],
        [
            mvrpack_sim[0][0],
            mvrpack_sim[0][1],
            15,
            mvrpack_sim[0][0],
            mvrpack_sim[0][1],
            16,
            "FACTOR",
            0.5,
        ],
    ]

    mvrspd = {0: sim_mvr_perioddata}
    maxmvr = 3
    mvr = flopy.mf6.ModflowMvr(
        sim,
        modelnames=True,
        maxmvr=maxmvr,
        print_flows=True,
        maxpackages=maxpackages,
        packages=mvrpack_sim,
        perioddata=mvrspd,
        filename="{}.mvr".format(name),
    )


def add_parent_transport(sim, gwtname, imsname):
    # Instantiating MODFLOW 6 groundwater transport package
    gwt = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtname,
        model_nam_file="{}.nam".format(gwtname),
    )
    gwt.name_file.save_flows = True

    # create iterative model solution and register the gwt model with it
    # imsgwt = flopy.mf6.ModflowIms(
    #    sim,
    #    print_option="SUMMARY",
    #    outer_dvclose=hclose,
    #    outer_maximum=nouter,
    #    under_relaxation="NONE",
    #    inner_maximum=ninner,
    #    inner_dvclose=hclose,
    #    rcloserecord=rclose,
    #    linear_acceleration="BICGSTAB",
    #    scaling_method="NONE",
    #    reordering_method="NONE",
    #    relaxation_factor=relax,
    #    filename="{}.ims".format(gwtname),
    # )
    # sim.register_ims_package(imsgwt, [imsname])

    # Instantiating MODFLOW 6 transport discretization package
    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlayp,
        nrow=nrowp,
        ncol=ncolp,
        delr=delrp,
        delc=delcp,
        top=topp,
        botm=botmp,
        idomain=idomainp,
        filename="{}.dis".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport initial concentrations
    flopy.mf6.ModflowGwtic(gwt, strt=100.0, filename="{}.ic".format(gwtname))

    # Instantiating MODFLOW 6 transport advection package
    if mixelm == 0:
        scheme = "UPSTREAM"
    elif mixelm == -1:
        scheme = "TVD"
    else:
        raise Exception()

    flopy.mf6.ModflowGwtadv(
        gwt, scheme=scheme, filename="{}.adv".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport dispersion package
    if al != 0:
        flopy.mf6.ModflowGwtdsp(
            gwt, alh=al, ath1=ath1, atv=atv, filename="{}.dsp".format(gwtname)
        )

    # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
    flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=prsity,
        first_order_decay=False,
        decay=None,
        decay_sorbed=None,
        sorption="linear",
        bulk_density=rhob,
        distcoef=Kd,
        filename="{}.mst".format(gwtname),
    )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]

    flopy.mf6.ModflowGwtssm(
        gwt, sources=sourcerecarray, filename="{}.ssm".format(gwtname)
    )

    # Instantiating MODFLOW 6 transport output control package
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord="{}.cbc".format(gwtname),
        concentration_filerecord="{}.ucn".format(gwtname),
        concentrationprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # Start of Advanced Transport Package Instantiations
    # Instantiating MODFLOW 6 streamflow transport (SFT) package
    sftpkdat = []
    for irno in range(len(sfrcells)):
        t = (irno, 50.0)
        sftpkdat.append(t)

    # Period data
    #             irno,  setting, strtc
    sftspd = {0: [[0, "INFLOW", 45.0]]}

    sft = flopy.mf6.modflow.ModflowGwtsft(
        gwt,
        boundnames=False,
        flow_package_name="SFR-parent",
        print_concentration=True,
        save_flows=True,
        concentration_filerecord=gwtname + ".sft.bin",
        budget_filerecord=gwtname + ".sft.bud",
        packagedata=sftpkdat,
        reachperioddata=sftspd,
        pname="SFT-parent",
        filename="{}.sft".format(gwtname),
    )

    # Instantiating MODFLOW 6 mover transport (MVT) package
    flopy.mf6.modflow.ModflowGwtmvt(
        gwt,
        save_flows=True,
        budget_filerecord=gwtname + ".mvt.bud",
        filename="{}.mvt".format(gwtname),
    )

    # Instantiating MODFLOW 6 flow-model interface (FMI) package
    flopy.mf6.modflow.ModflowGwtfmi(
        gwt, flow_imbalance_correction=True, filename="{}.fmi".format(gwtname)
    )

    return gwt


def add_child_transport(sim, lgr, gwtnamec, imsname):
    # Instantiating MODFLOW 6 groundwater transport package
    gwtc = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtnamec,
        model_nam_file="{}.nam".format(gwtnamec),
    )
    gwtc.name_file.save_flows = True

    # create iterative model solution and register the gwt model with it
    # imsgwtc = flopy.mf6.ModflowIms(
    #    sim,
    #    print_option="SUMMARY",
    #    outer_dvclose=hclose,
    #    outer_maximum=nouter,
    #    under_relaxation="NONE",
    #    inner_maximum=ninner,
    #    inner_dvclose=hclose,
    #    rcloserecord=rclose,
    #    linear_acceleration="BICGSTAB",
    #    scaling_method="NONE",
    #    reordering_method="NONE",
    #    relaxation_factor=relax,
    #    filename="{}.ims".format(gwtnamec),
    # )
    # sim.register_ims_package(imsgwtc, [imsname])

    # Instantiating MODFLOW 6 transport discretization package
    child_dis_shp = lgr.get_shape()
    nlayc = child_dis_shp[0]
    nrowc = child_dis_shp[1]
    ncolc = child_dis_shp[2]
    # Get child grid info:
    delrc, delcc = lgr.get_delr_delc()
    idomainc = lgr.get_idomain()  # child idomain
    topc, botmc = lgr.get_top_botm()  # top/bottom of child grid
    flopy.mf6.ModflowGwtdis(
        gwtc,
        nlay=nlayc,
        nrow=nrowc,
        ncol=ncolc,
        delr=delrc,
        delc=delcc,
        top=topc,
        botm=botmc,
        idomain=idomainc,
        filename="{}.dis".format(gwtnamec),
    )

    # Instantiating MODFLOW 6 transport initial concentrations
    flopy.mf6.ModflowGwtic(gwtc, strt=100.0, filename="{}.ic".format(gwtnamec))

    # Instantiating MODFLOW 6 transport advection package
    if mixelm == 0:
        scheme = "UPSTREAM"
    elif mixelm == -1:
        scheme = "TVD"
    else:
        raise Exception()

    flopy.mf6.ModflowGwtadv(
        gwtc, scheme=scheme, filename="{}.adv".format(gwtnamec)
    )

    # Instantiating MODFLOW 6 transport dispersion package
    if al != 0:
        flopy.mf6.ModflowGwtdsp(
            gwtc,
            alh=al,
            ath1=ath1,
            atv=atv,
            filename="{}.dsp".format(gwtnamec),
        )

    # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
    flopy.mf6.ModflowGwtmst(
        gwtc,
        porosity=prsity,
        first_order_decay=False,
        decay=None,
        decay_sorbed=None,
        sorption="linear",
        bulk_density=rhob,
        distcoef=Kd,
        filename="{}.mst".format(gwtnamec),
    )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [("WEL-2", "AUX", "CONCENTRATION")]

    flopy.mf6.ModflowGwtssm(
        gwtc, sources=sourcerecarray, filename="{}.ssm".format(gwtnamec)
    )

    # Instantiating MODFLOW 6 transport output control package
    flopy.mf6.ModflowGwtoc(
        gwtc,
        budget_filerecord="{}.cbc".format(gwtnamec),
        concentration_filerecord="{}.ucn".format(gwtnamec),
        concentrationprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    # Start of Advanced Transport Package Instantiations
    # Instantiating MODFLOW 6 streamflow transport (SFT) package
    sftpkdatc = []
    for irno in range(len(sfrcellsc)):
        t = (irno, 50.0)
        sftpkdatc.append(t)

    # Period data
    #             irno,  setting, strtc
    sftspdc = {0: [[0, "INFLOW", 0.0]]}

    sftc = flopy.mf6.modflow.ModflowGwtsft(
        gwtc,
        boundnames=False,
        flow_package_name="SFR-child",
        print_concentration=True,
        save_flows=True,
        concentration_filerecord=gwtnamec + ".sft.bin",
        budget_filerecord=gwtnamec + ".sft.bud",
        packagedata=sftpkdatc,
        reachperioddata=sftspdc,
        pname="SFT-child",
        filename="{}.sft".format(gwtnamec),
    )

    # Instantiating MODFLOW 6 mover transport (MVT) package
    flopy.mf6.modflow.ModflowGwtmvt(
        gwtc,
        save_flows=True,
        budget_filerecord=gwtnamec + ".mvt.bud",
        filename="{}.mvt".format(gwtnamec),
    )

    # Instantiating MODFLOW 6 flow-model interface (FMI) package
    flopy.mf6.modflow.ModflowGwtfmi(
        gwtc,
        flow_imbalance_correction=True,
        filename="{}.fmi".format(gwtnamec),
    )

    return gwtc


def add_Gwfgwf_exchange(sim, lgr, gwfname, gwfnamec):
    # exchange data
    exchange_data = lgr.get_exchange_data(angldegx=True, cdist=True)

    # Establish GWF-GWF exchange
    gwfgwf = flopy.mf6.ModflowGwfgwf(
        sim,
        exgtype="GWF6-GWF6",
        print_flows=True,
        print_input=True,
        exgmnamea=gwfname,
        exgmnameb=gwfnamec,
        nexg=len(exchange_data),
        exchangedata=exchange_data,
        mvr_filerecord="{}.mvr".format(name),
        pname="EXG-1",
        auxiliary=["ANGLDEGX", "CDIST"],
        filename="{}.exg".format(name),
    )


def setup_gwfgwt_exchng(sim, gwfname, gwtname):
    # Instantiating MODFLOW 6 flow-transport exchange mechanism
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwtname,
        filename="{}.gwfgwt".format(gwfname),
    )


def instantiate_simulations():
    scen_nm_parent = name + "-mltmvr" + "_p_q"
    scen_nm_child = name + "-mltmvr" + "_c_q"
    scen_nm_parent_transport = name + "-mltmvr" + "_p_qw"
    scen_nm_child_transport = name + "-mltmvr" + "_c_qw"

    # Take care of flow model set up
    sim, gwf, gwfc, lgr = instantiate_base_models(
        name, scen_nm_parent, scen_nm_child
    )
    # add the sfr packages
    add_parent_sfr(gwf, scen_nm_parent, connsp_mvr)
    add_child_sfr(gwfc, scen_nm_child)
    # add the mover packages (simulation level and gwf level)
    add_parent_gwf_mvr(gwf, scen_nm_parent)
    add_child_gwf_mvr(gwfc, scen_nm_child)

    # Now setup up transport model
    gwt = add_parent_transport(sim, scen_nm_parent_transport, gwf.name)
    gwtc = add_child_transport(sim, lgr, scen_nm_child_transport, gwf.name)
    # Add the Gwf-Gwt exchange for the parent model
    setup_gwfgwt_exchng(sim, scen_nm_parent, scen_nm_parent_transport)
    setup_gwfgwt_exchng(sim, scen_nm_child, scen_nm_child_transport)

    add_Gwfgwf_exchange(sim, lgr, scen_nm_parent, scen_nm_child)
    add_sim_mvr(sim, scen_nm_parent, scen_nm_child)

    sim.write_simulation()

    # Run the simulation
    success, buff = sim.run_simulation(silent=False)
    if not success:
        print(buff)

    # Check scenario output (perhaps just check that the .lst files exist)
    # Check that 5 different .lst file are present
    files = os.listdir(ws)
    num_lst = 0
    for file in files:
        if ".lst" in file:
            num_lst += 1

    assert (
        num_lst == 5
    ), "Simulation doesn't appear to have run all of the models within the simulation"

    return success


# - No need to change any code below
def test_mf6model():
    # build the models
    instantiate_simulations()

    return


def main():
    # build the models
    instantiate_simulations()

    return


if __name__ == "__main__":
    # print message
    print("standalone run of {}".format(os.path.basename(__file__)))

    # run main routine
    main()
