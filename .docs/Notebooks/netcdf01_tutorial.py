import sys
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")

DNODATA = 3.0e30


def create_sim(ws):
    name = "uzf01"
    perlen = [500.0]
    nper = len(perlen)
    nstp = [10]
    tsmult = nper * [1.0]
    crs = "EPSG:26916"
    nlay, nrow, ncol = 100, 1, 1
    delr = 1.0
    delc = 1.0
    delv = 1.0
    top = 100.0
    botm = [top - (k + 1) * delv for k in range(nlay)]
    strt = 0.5
    hk = 1.0
    laytyp = 1
    ss = 0.0
    sy = 0.1

    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=ws
    )

    # create tdis package
    tdis = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

    # create iterative model solution and register the gwf model with it
    nouter, ninner = 100, 10
    hclose, rclose, relax = 1.5e-6, 1e-6, 0.97
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="DBD",
        under_relaxation_theta=0.7,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
    )

    # create gwf model
    newtonoptions = "NEWTON UNDER_RELAXATION"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        newtonoptions=newtonoptions,
        save_flows=True,
    )

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        crs=crs,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=np.ones((nlay, nrow, ncol), dtype=int),
    )

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=False, icelltype=laytyp, k=hk)
    # storage
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=False,
        iconvert=laytyp,
        ss=ss,
        sy=sy,
        steady_state={0: False},
        transient={0: True},
    )

    # ghbg
    ghb_obs = {f"{name}.ghb.obs.csv": [("100_1_1", "GHB", (99, 0, 0))]}
    bhead = np.full(nlay * nrow * ncol, DNODATA, dtype=float)
    cond = np.full(nlay * nrow * ncol, DNODATA, dtype=float)
    bhead[nlay - 1] = 1.5
    cond[nlay - 1] = 1.0
    ghb = flopy.mf6.ModflowGwfghbg(
        gwf,
        print_input=True,
        print_flows=True,
        bhead=bhead,
        cond=cond,
        save_flows=False,
    )

    ghb.obs.initialize(
        filename=f"{name}.ghb.obs",
        digits=20,
        print_input=True,
        continuous=ghb_obs,
    )

    # note: for specifying lake number, use fortran indexing!
    uzf_obs = {
        f"{name}.uzf.obs.csv": [
            ("wc 02", "water-content", 2, 0.5),
            ("wc 50", "water-content", 50, 0.5),
            ("wcbn 02", "water-content", "uzf 002", 0.5),
            ("wcbn 50", "water-content", "UZF 050", 0.5),
            ("rch 02", "uzf-gwrch", "uzf 002"),
            ("rch 50", "uzf-gwrch", "uzf 050"),
        ]
    }

    sd = 0.1
    vks = hk
    thtr = 0.05
    thti = thtr
    thts = sy
    eps = 4
    uzf_pkdat = [[0, (0, 0, 0), 1, 1, sd, vks, thtr, thts, thti, eps, "uzf 001"]] + [
        [k, (k, 0, 0), 0, k + 1, sd, vks, thtr, thts, thti, eps, f"uzf {k + 1:03d}"]
        for k in range(1, nlay - 1)
    ]
    uzf_pkdat[-1][3] = -1
    infiltration = 2.01
    uzf_spd = {0: [[0, infiltration, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}
    uzf = flopy.mf6.ModflowGwfuzf(
        gwf,
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        ntrailwaves=15,
        nwavesets=40,
        nuzfcells=len(uzf_pkdat),
        packagedata=uzf_pkdat,
        perioddata=uzf_spd,
        budget_filerecord=f"{name}.uzf.bud",
        budgetcsv_filerecord=f"{name}.uzf.bud.csv",
        observations=uzf_obs,
        filename=f"{name}.uzf",
    )

    # output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{name}.bud",
        head_filerecord=f"{name}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "ALL")],
    )

    obs_lst = []
    obs_lst.append(["obs1", "head", (0, 0, 0)])
    obs_lst.append(["obs2", "head", (1, 0, 0)])
    obs_dict = {f"{name}.obs.csv": obs_lst}
    obs = flopy.mf6.ModflowUtlobs(gwf, pname="head_obs", digits=20, continuous=obs_dict)

    return sim


def add_netcdf_vars(dataset, nc_info, dimmap):
    def _data_shape(shape):
        dims_l = []
        for d in shape:
            dims_l.append(dimmap[d])

        return dims_l

    for v in nc_info:
        varname = nc_info[v]["varname"]
        data = np.full(
            _data_shape(nc_info[v]["nc_shape"]),
            nc_info[v]["attrs"]["_FillValue"],
            dtype=nc_info[v]["nc_type"],
        )
        var_d = {varname: (nc_info[v]["nc_shape"], data)}
        dataset = dataset.assign(var_d)
        for a in nc_info[v]["attrs"]:
            dataset[varname].attrs[a] = nc_info[v]["attrs"][a]

    return dataset


temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# run the non-netcdf simulation
sim = create_sim(ws=workspace)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf")
gwf = sim.get_model("uzf01")
gwf.name_file.nc_filerecord = "uzf01.structured.nc"
sim.write_simulation()

# create the netcdf dataset
ds = xr.Dataset()

# get model netcdf info
nc_info = gwf.netcdf_info()

# update dataset with required attributes
for a in nc_info["attrs"]:
    ds.attrs[a] = nc_info["attrs"][a]

# set dimensional info
dis = gwf.modelgrid
xoff = dis.xoffset
yoff = dis.yoffset
x = xoff + dis.xycenters[0]
y = yoff + dis.xycenters[1]
z = [float(x) for x in range(1, dis.nlay + 1)]
nstp = sum(gwf.modeltime.nstp)
time = gwf.modeltime.tslen
nlay = dis.nlay
nrow = dis.nrow
ncol = dis.ncol
dimmap = {"time": nstp, "z": nlay, "y": nrow, "x": ncol}

# create coordinate vars
var_d = {"time": (["time"], time), "z": (["z"], z), "y": (["y"], y), "x": (["x"], x)}
ds = ds.assign(var_d)

# dis
dis = gwf.get_package("dis")
nc_info = dis.netcdf_info()

# create dis dataset variables
ds = add_netcdf_vars(ds, nc_info, dimmap)

# update data
ds["dis_delr"].values = dis.delr.get_data()
ds["dis_delc"].values = dis.delc.get_data()
ds["dis_top"].values = dis.top.get_data()
ds["dis_botm"].values = dis.botm.get_data()
ds["dis_idomain"].values = dis.idomain.get_data()

# update dis to read from netcdf
with open(workspace / "netcdf" / "uzf01.dis", "w") as f:
    f.write("BEGIN options\n")
    f.write("  crs  EPSG:26916\n")
    f.write("END options\n\n")
    f.write("BEGIN dimensions\n")
    f.write("  NLAY  100\n")
    f.write("  NROW  1\n")
    f.write("  NCOL  1\n")
    f.write("END dimensions\n\n")
    f.write("BEGIN griddata\n")
    f.write("  delr NETCDF\n")
    f.write("  delc NETCDF\n")
    f.write("  top NETCDF\n")
    f.write("  botm NETCDF\n")
    f.write("  idomain NETCDF\n")
    f.write("END griddata\n")

# npf
npf = gwf.get_package("npf")
nc_info = npf.netcdf_info()

# create npf dataset variables
ds = add_netcdf_vars(ds, nc_info, dimmap)

# update data
ds["npf_icelltype"].values = npf.icelltype.get_data()
ds["npf_k"].values = npf.k.get_data()

# update npf to read from netcdf
with open(workspace / "netcdf" / "uzf01.npf", "w") as f:
    f.write("BEGIN options\n")
    f.write("END options\n\n")
    f.write("BEGIN griddata\n")
    f.write("  icelltype NETCDF\n")
    f.write("  k NETCDF\n")
    f.write("END griddata\n")

# get ghbg package netcdf info
ghbg = gwf.get_package("ghbg_0")
nc_info = ghbg.netcdf_info()

# create ghbg dataset variables
ds = add_netcdf_vars(ds, nc_info, dimmap)

# update bhead netcdf array from flopy perioddata
for p in ghbg.bhead.get_data():
    istp = sum(gwf.modeltime.nstp[0:p])
    ds["ghbg_0_bhead"].values[istp] = ghbg.bhead.get_data()[p]

# update cond netcdf array from flopy perioddata
for p in ghbg.cond.get_data():
    istp = sum(gwf.modeltime.nstp[0:p])
    ds["ghbg_0_cond"].values[istp] = ghbg.cond.get_data()[p]

# write the netcdf
ds.to_netcdf(
    workspace / "netcdf/uzf01.structured.nc", format="NETCDF4", engine="netcdf4"
)

# update ghbg to read from netcdf
with open(workspace / "netcdf/uzf01.ghbg", "w") as f:
    f.write("BEGIN options\n")
    f.write("  READARRAYGRID\n")
    f.write("  PRINT_INPUT\n")
    f.write("  PRINT_FLOWS\n")
    f.write("  OBS6  FILEIN  uzf01.ghb.obs\n")
    f.write("END options\n\n")
    f.write("BEGIN period 1\n")
    f.write("  bhead NETCDF\n")
    f.write("  cond NETCDF\n")
    f.write("END period 1\n")

# TODO need extended modflow 6 to run this simulation
# run the netcdf sim
# success, buff = sim.run_simulation(silent=True, report=True)
# assert success, pformat(buff)
