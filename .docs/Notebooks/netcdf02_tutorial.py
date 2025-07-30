import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")
# -

DNODATA = 3.0e30


def get_flow_sim(ws):
    name = "flow"
    gwfname = name
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name="mf6")
    tdis_rc = [(100.0, 1, 1.0), (100.0, 1, 1.0)]
    nper = len(tdis_rc)
    tdis = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)

    # ims
    hclose = 1.0e-6
    rclose = 1.0e-6
    nouter = 1000
    ninner = 100
    relax = 0.99
    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="ALL",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{gwfname}.ims",
    )

    nlay = 1
    nrow = 10
    ncol = 10
    delr = 10.0
    delc = 10.0
    top = 100.0
    botm = 0.0

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        xt3doptions=False,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=[1],
        k=10.0,
    )

    sto_on = False
    if sto_on:
        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=True,
            iconvert=[1],
            ss=1.0e-5,
            sy=0.3,
            steady_state={0: True},
            transient={0: False},
        )

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwfname}.bud",
        head_filerecord=f"{gwfname}.hds",
        headprintrecord=[("COLUMNS", ncol, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    rch_on = False
    if rch_on:
        rch = flopy.mf6.ModflowGwfrcha(gwf, recharge={0: 4.79e-3}, pname="RCH-1")

    # wel
    q = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
    welconc = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
    for i in range(nrow):
        q[0, i, 0] = 100.0
        welconc[0, i, 0] = 100.0
    wel = flopy.mf6.ModflowGwfwelg(
        gwf,
        auxiliary=["concentration"],
        pname="WEL-1",
        q=q,
        aux=welconc,
    )

    # ghb
    rows = [0, 1, 2, 3]
    for ipak, i in enumerate(rows):
        fname = f"flow.{ipak + 1}.ghb"
        pname = f"GHB-{ipak + 1}"
        bhead = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        cond = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        conc = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        bhead[0, i, ncol - 1] = 50.0
        cond[0, i, ncol - 1] = 1000.0
        conc[0, i, ncol - 1] = 100.0
        flopy.mf6.ModflowGwfghbg(
            gwf,
            auxiliary=["concentration"],
            filename=f"{fname}g",
            pname=pname,
            bhead=bhead,
            cond=cond,
            aux=conc,
        )

    # riv
    rows = [4, 5, 6]
    for ipak, i in enumerate(rows):
        fname = f"flow.{ipak + 1}.riv"
        pname = f"RIV-{ipak + 1}"
        stage = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        cond = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        rbot = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        conc = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        stage[0, i, ncol - 1] = 50.0
        cond[0, i, ncol - 1] = 1000.0
        rbot[0, i, ncol - 1] = 0.0
        conc[0, i, ncol - 1] = 100.0
        riv = flopy.mf6.ModflowGwfrivg(
            gwf,
            auxiliary=["concentration"],
            filename=f"{fname}g",
            pname=pname,
            stage=stage,
            cond=cond,
            rbot=rbot,
            aux=[conc],
        )

    # drn
    rows = [7, 8, 9]
    for ipak, i in enumerate(rows):
        fname = f"flow.{ipak + 1}.drn"
        pname = f"DRN-{ipak + 1}"
        elev = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        cond = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        conc = np.full((nlay, nrow, ncol), DNODATA, dtype=float)
        elev[0, i, ncol - 1] = 50.0
        cond[0, i, ncol - 1] = 1000.0
        conc[0, i, ncol - 1] = 100.0
        drn = flopy.mf6.modflow.ModflowGwfdrng(
            gwf,
            auxiliary=["concentration"],
            filename=f"{fname}g",
            pname=pname,
            elev=elev,
            cond=cond,
            aux=[conc],
        )

    return sim


temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# run the non-netcdf simulation
sim = get_flow_sim(workspace)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf")
gwf = sim.get_model("flow")
gwf.name_file.nc_filerecord = "flow.structured.nc"
sim.write_simulation()

# create the netcdf dataset
ds = xr.Dataset()

# get model netcdf info
nc_info = gwf.netcdf_info()

# update dataset with required attributes
for a in nc_info["attrs"]:
    ds.attrs[a] = nc_info["attrs"][a]

# get dim info from modelgrid
dis = gwf.modelgrid
xoff = dis.xoffset
yoff = dis.yoffset
x = xoff + dis.xycenters[0]
y = yoff + dis.xycenters[1]
z = [float(x) for x in range(1, dis.nlay + 1)]

# set nstp and time
nstp = sum(gwf.modeltime.nstp)
time = gwf.modeltime.tslen

# create coordinate vars
var_d = {"time": (["time"], time), "z": (["z"], z), "y": (["y"], y), "x": (["x"], x)}
ds = ds.assign(var_d)

# shape list for data arrays
shape = ["time", "z", "y", "x"]

# update for welg
welg = gwf.get_package("wel-1")
nc_info = welg.netcdf_info()

for v in nc_info:
    varname = nc_info[v]["varname"]
    data = np.full((nstp, dis.nlay, dis.nrow, dis.ncol), DNODATA, dtype=float)
    var_d = {varname: (shape, data)}
    ds = ds.assign(var_d)
    # add required modflow 6 param attributes
    for a in nc_info[v]["attrs"]:
        ds[varname].attrs[a] = nc_info[v]["attrs"][a]
    ds[varname].attrs["_FillValue"] = DNODATA

# update q netcdf array from flopy perioddata
for p in welg.q.get_data():
    if welg.q.get_data()[p] is not None:
        ds["wel-1_q"].values[p] = welg.q.get_data()[p]

# update conc netcdf array from flopy perioddata
for p in welg.aux.get_data():
    if welg.aux.get_data()[p] is not None:
        ds["wel-1_concentration"].values[p] = welg.aux.get_data()[p][0]

# update welg input to read from netcdf
with open(workspace / "netcdf" / "flow.welg", "w") as f:
    f.write("BEGIN options\n")
    f.write("  READARRAYGRID\n")
    f.write("  auxiliary  CONCENTRATION\n")
    f.write("END options\n\n")
    f.write("BEGIN period  1\n")
    f.write("  q NETCDF\n")
    f.write("  concentration NETCDF\n")
    f.write("END period  1\n\n")

# update for ghbg
for n in range(4):
    ip = n + 1

    # get ghbg package netcdf info
    ghbg = gwf.get_package(f"ghb-{ip}")
    nc_info = ghbg.netcdf_info()

    for v in nc_info:
        varname = nc_info[v]["varname"]
        data = np.full((nstp, dis.nlay, dis.nrow, dis.ncol), DNODATA, dtype=float)
        var_d = {varname: (shape, data)}
        ds = ds.assign(var_d)
        # add required modflow 6 param attributes
        for a in nc_info[v]["attrs"]:
            ds[varname].attrs[a] = nc_info[v]["attrs"][a]
        ds[varname].attrs["_FillValue"] = DNODATA

    # update bhead netcdf array from flopy perioddata
    for p in ghbg.bhead.get_data():
        if ghbg.bhead.get_data()[p] is not None:
            ds[f"ghb-{ip}_bhead"].values[p] = ghbg.bhead.get_data()[p]

    # update cond netcdf array from flopy perioddata
    for p in ghbg.cond.get_data():
        if ghbg.cond.get_data()[p] is not None:
            ds[f"ghb-{ip}_cond"].values[p] = ghbg.cond.get_data()[p]

    # update conc netcdf array from flopy perioddata
    for p in ghbg.aux.get_data():
        if ghbg.aux.get_data()[p] is not None:
            ds[f"ghb-{ip}_concentration"].values[p] = ghbg.aux.get_data()[p][0]

    # update ghbg input to read from netcdf
    with open(workspace / "netcdf" / f"flow.{ip}.ghbg", "w") as f:
        f.write("BEGIN options\n")
        f.write("  READARRAYGRID\n")
        f.write("  auxiliary  CONCENTRATION\n")
        f.write("END options\n\n")
        f.write("BEGIN period  1\n")
        f.write("  bhead NETCDF\n")
        f.write("  cond NETCDF\n")
        f.write("  concentration NETCDF\n")
        f.write("END period  1\n\n")


# update for rivg
for n in range(3):
    ip = n + 1

    # get rivg package netcdf info
    rivg = gwf.get_package(f"riv-{ip}")
    nc_info = rivg.netcdf_info()

    for v in nc_info:
        varname = nc_info[v]["varname"]
        data = np.full((nstp, dis.nlay, dis.nrow, dis.ncol), DNODATA, dtype=float)
        var_d = {varname: (shape, data)}
        ds = ds.assign(var_d)
        # add required modflow 6 param attributes
        for a in nc_info[v]["attrs"]:
            ds[varname].attrs[a] = nc_info[v]["attrs"][a]
        ds[varname].attrs["_FillValue"] = DNODATA

    # update stage netcdf array from flopy perioddata
    for p in rivg.stage.get_data():
        if rivg.stage.get_data()[p] is not None:
            ds[f"riv-{ip}_stage"].values[p] = rivg.stage.get_data()[p]

    # update cond netcdf array from flopy perioddata
    for p in rivg.cond.get_data():
        if rivg.cond.get_data()[p] is not None:
            ds[f"riv-{ip}_cond"].values[p] = rivg.cond.get_data()[p]

    # update rbot netcdf array from flopy perioddata
    for p in rivg.rbot.get_data():
        if rivg.rbot.get_data()[p] is not None:
            ds[f"riv-{ip}_rbot"].values[p] = rivg.rbot.get_data()[p]

    # update conc netcdf array from flopy perioddata
    for p in rivg.aux.get_data():
        if rivg.aux.get_data()[p] is not None:
            ds[f"riv-{ip}_concentration"].values[p] = rivg.aux.get_data()[p][0]

    # update rivg input to read from netcdf
    with open(workspace / "netcdf" / f"flow.{ip}.rivg", "w") as f:
        f.write("BEGIN options\n")
        f.write("  READARRAYGRID\n")
        f.write("  auxiliary  CONCENTRATION\n")
        f.write("END options\n\n")
        f.write("BEGIN period  1\n")
        f.write("  stage NETCDF\n")
        f.write("  cond NETCDF\n")
        f.write("  rbot NETCDF\n")
        f.write("  concentration NETCDF\n")
        f.write("END period  1\n\n")


# update for drng
for n in range(3):
    ip = n + 1

    # get drng package netcdf info
    drng = gwf.get_package(f"drn-{ip}")
    nc_info = drng.netcdf_info()

    for v in nc_info:
        varname = nc_info[v]["varname"]
        data = np.full((nstp, dis.nlay, dis.nrow, dis.ncol), DNODATA, dtype=float)
        var_d = {varname: (shape, data)}
        ds = ds.assign(var_d)
        # add required modflow 6 param attributes
        for a in nc_info[v]["attrs"]:
            ds[varname].attrs[a] = nc_info[v]["attrs"][a]
        ds[varname].attrs["_FillValue"] = DNODATA

    # update elev netcdf array from flopy perioddata
    for p in drng.elev.get_data():
        if drng.elev.get_data()[p] is not None:
            ds[f"drn-{ip}_elev"].values[p] = drng.elev.get_data()[p]

    # update cond netcdf array from flopy perioddata
    for p in drng.cond.get_data():
        if drng.cond.get_data()[p] is not None:
            ds[f"drn-{ip}_cond"].values[p] = drng.cond.get_data()[p]

    # update conc netcdf array from flopy perioddata
    for p in drng.aux.get_data():
        if drng.aux.get_data()[p] is not None:
            ds[f"drn-{ip}_concentration"].values[p] = drng.aux.get_data()[p][0]

    # update drng input to read from netcdf
    with open(workspace / "netcdf" / f"flow.{ip}.drng", "w") as f:
        f.write("BEGIN options\n")
        f.write("  READARRAYGRID\n")
        f.write("  auxiliary  CONCENTRATION\n")
        f.write("END options\n\n")
        f.write("BEGIN period  1\n")
        f.write("  elev NETCDF\n")
        f.write("  cond NETCDF\n")
        f.write("  concentration NETCDF\n")
        f.write("END period  1\n\n")

# write the netcdf
ds.to_netcdf(
    workspace / "netcdf/flow.structured.nc", format="NETCDF4", engine="netcdf4"
)

# run the netcdf sim
success, buff = sim.run_simulation(silent=False, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")
