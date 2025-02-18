import math
import os
import shutil
from traceback import format_exc
from warnings import warn

import numpy as np
import pytest
import xarray as xr
from modflow_devtools.markers import requires_exe, requires_pkg
from pyproj import CRS

import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.datautil import DatumUtil
from flopy.utils.gridutil import get_disv_kwargs
from flopy.utils.model_netcdf import create_dataset


def compare_netcdf(base, gen, projection=False, update=None):
    """Check for functional equivalence"""
    xrb = xr.open_dataset(base, engine="netcdf4")
    xrg = xr.open_dataset(gen, engine="netcdf4")

    # global attributes
    for a in xrb.attrs:
        assert a in xrg.attrs
        if a == "title" or a == "history" or a == "source":
            continue
        assert xrb.attrs[a] == xrg.attrs[a]

    # coordinates
    for coordname, da in xrb.coords.items():
        compare_netcdf_var(
            coordname, xrb.coords, xrg.data_vars, xrg.coords, projection, update
        )

    # variables
    for varname, da in xrb.data_vars.items():
        if varname == "projection":
            if projection:
                assert varname in xrg.data_vars
                assert "wkt" in da.attrs or "crs_wkt" in da.attrs
                if "wkt" in da.attrs:
                    attr = "wkt"
                else:
                    attr = "crs_wkt"
                assert attr in xrg.data_vars[varname].attrs

                # TODO
                # crs_b = CRS.from_wkt(da.attrs[attr])
                # epsg_b = crs_b.to_epsg(min_confidence=90)
                # crs_g = CRS.from_wkt(xrg.data_vars[varname].attrs[attr])
                # epsg_g = crs_g.to_epsg(min_confidence=90)
                # assert epsg_b == epsg_g
            continue

        compare_netcdf_var(
            varname, xrb.data_vars, xrg.data_vars, xrg.coords, projection, update
        )


def compare_netcdf_data(base, gen):
    """Data comparison check"""
    xrb = xr.open_dataset(base, engine="netcdf4")
    xrg = xr.open_dataset(gen, engine="netcdf4")

    # coordinates
    for coordname, da in xrb.coords.items():
        compare_netcdf_var(coordname, xrb.coords, xrg.data_vars, xrg.coords)

    # variables
    for varname, da in xrb.data_vars.items():
        if varname == "projection":
            continue

        compare_netcdf_var(varname, xrb.data_vars, xrg.data_vars, xrg.coords)


def compare_netcdf_var(varname, base_d, gen_d, coord_d, projection=False, update=None):
    # check variable name
    assert varname in gen_d or varname in coord_d

    if varname in gen_d:
        var_d = gen_d
    else:
        var_d = coord_d

    # encodings
    for e in base_d[varname].encoding:
        assert e in var_d[varname].encoding
        if e.lower() == "source":
            continue
        if e == "_FillValue":
            if np.isnan(base_d[varname].encoding[e]):
                assert np.isnan(var_d[varname].encoding[e])
            else:
                assert np.allclose(
                    base_d[varname].encoding[e], var_d[varname].encoding[e]
                )
        else:
            assert base_d[varname].encoding[e] == var_d[varname].encoding[e]

    # check variable attributes
    for a in base_d[varname].attrs:
        if a == "grid_mapping" and not projection:
            continue
        assert a in var_d[varname].attrs
        assert base_d[varname].attrs[a] == var_d[varname].attrs[a]

    # check variable data
    print(f"NetCDF file check data equivalence for variable: {varname}")
    if update and varname in update:
        assert np.allclose(update[varname], var_d[varname].data)
    else:
        assert np.allclose(base_d[varname].data, var_d[varname].data)


@pytest.mark.regression
def test_load_gwfsto01(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "mesh2d",
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "structured",
        },
    }
    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)
        # gwf = sim.get_model("gwf_sto01")

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf=test["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                # "gwf_sto01.dis.ncf":   # TODO wkt string missing on write?
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1.lower() == line2.lower()
            else:
                compare_netcdf(base, gen)


@pytest.mark.regression
def test_update_gwfsto01(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "mesh2d",
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "structured",
        },
    }

    nlay, nrow, ncol = 3, 10, 10

    dis_top = np.full((nrow, ncol), 50)

    # ic
    strt1 = np.full((nrow, ncol), 0.15)
    strt2 = np.full((nrow, ncol), 0.21)
    strt3 = np.full((nrow, ncol), 1.21)
    ic_strt = np.array([strt1, strt2, strt3])

    update = {
        "dis_top": dis_top.flatten()[0],
        "ic_strt": ic_strt,
        "ic_strt_l1": ic_strt[0].flatten(),
        "ic_strt_l2": ic_strt[1].flatten(),
        "ic_strt_l3": ic_strt[2].flatten(),
    }

    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

        # get model instance
        gwf = sim.get_model("gwf_sto01")

        # update dis top
        gwf.dis.top = dis_top

        # update ic strt
        gwf.ic.strt.set_data(ic_strt)

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf=test["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                # "gwf_sto01.dis.ncf":   # TODO wkt string missing on write?
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1.lower() == line2.lower()
            else:
                compare_netcdf(base, gen, update=update)


@pytest.mark.regression
def test_create_gwfsto01(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "mesh2d",
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "structured",
        },
    }
    name = "gwf_sto01"

    # static model data
    # temporal discretization
    nper = 31
    perlen = [1.0] + [365.2500000 for _ in range(nper - 1)]
    nstp = [1] + [6 for _ in range(nper - 1)]
    tsmult = [1.0] + [1.3 for _ in range(nper - 1)]
    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    # spatial discretization data
    nlay, nrow, ncol = 3, 10, 10
    delr, delc = 1000.0, 2000.0
    botm = [-100, -150.0, -350.0]
    strt = 0.0

    # calculate hk
    hk1fact = 1.0 / 50.0
    hk1 = np.ones((nrow, ncol), dtype=float) * 0.5 * hk1fact
    hk1[0, :] = 1000.0 * hk1fact
    hk1[-1, :] = 1000.0 * hk1fact
    hk1[:, 0] = 1000.0 * hk1fact
    hk1[:, -1] = 1000.0 * hk1fact
    hk = [20.0, hk1, 5.0]

    # calculate vka
    vka = [1e6, 7.5e-5, 1e6]

    # all cells are active and layer 1 is convertible
    ib = 1
    laytyp = [1, 0, 0]

    # solver options
    nouter, ninner = 500, 300
    hclose, rclose, relax = 1e-9, 1e-6, 1.0
    newtonoptions = "NEWTON"
    imsla = "BICGSTAB"

    # chd data
    c = []
    c6 = []
    ccol = [3, 4, 5, 6]
    for j in ccol:
        c.append([0, nrow - 1, j, strt, strt])
        c6.append([(0, nrow - 1, j), strt])
    cd = {0: c}
    cd6 = {0: c6}
    maxchd = len(cd[0])

    # pumping well data
    wr = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    wc = [0, 1, 8, 9, 0, 9, 0, 9, 0, 0]
    wrp = [2, 2, 3, 3]
    wcp = [5, 6, 5, 6]
    wq = [-14000.0, -8000.0, -5000.0, -3000.0]
    d = []
    d6 = []
    for r, c, q in zip(wrp, wcp, wq):
        d.append([2, r, c, q])
        d6.append([(2, r, c), q])
    wd = {1: d}
    wd6 = {1: d6}
    maxwel = len(wd[1])

    # recharge data
    q = 3000.0 / (delr * delc)
    v = np.zeros((nrow, ncol), dtype=float)
    for r, c in zip(wr, wc):
        v[r, c] = q
    rech = {0: v}

    # storage and compaction data
    ske = [6e-4, 3e-4, 6e-4]

    # build
    ws = function_tmpdir / "ws"
    for idx, (dirname, test) in enumerate(tests.items()):
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # build MODFLOW 6 files
        sim = flopy.mf6.MFSimulation(
            sim_name=name, version="mf6", exe_name="mf6", sim_ws=ws
        )

        # create tdis package
        tdis = flopy.mf6.ModflowTdis(
            sim,
            time_units="DAYS",
            start_date_time="2041-01-01t00:00:00-05:00",
            nper=nper,
            perioddata=tdis_rc,
        )

        # create gwf model
        top = 0.0
        zthick = [top - botm[0], botm[0] - botm[1], botm[1] - botm[2]]
        elevs = [top] + botm

        # create model
        kwargs = {}
        kwargs["crs"] = "EPSG:26918"
        gwf = flopy.mf6.ModflowGwf(
            sim, modelname=name, newtonoptions=newtonoptions, save_flows=True, **kwargs
        )

        # create iterative model solution and register the gwf model with it
        ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="NONE",
            inner_maximum=ninner,
            inner_dvclose=hclose,
            rcloserecord=rclose,
            linear_acceleration=imsla,
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
        )
        sim.register_ims_package(ims, [gwf.name])

        # dis
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            filename=f"{name}.dis",
        )

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt, filename=f"{name}.ic")

        # node property flow
        npf = flopy.mf6.ModflowGwfnpf(
            gwf,
            save_flows=False,
            icelltype=laytyp,
            k=hk,
            k33=vka,
        )

        # storage
        sto = flopy.mf6.ModflowGwfsto(
            gwf,
            save_flows=False,
            iconvert=laytyp,
            ss=ske,
            sy=0,
            storagecoefficient=None,
            steady_state={0: True},
            transient={1: True},
        )

        # recharge
        rch = flopy.mf6.ModflowGwfrcha(gwf, readasarrays=True, recharge=rech)

        # wel file
        wel = flopy.mf6.ModflowGwfwel(
            gwf,
            print_input=True,
            print_flows=True,
            maxbound=maxwel,
            stress_period_data=wd6,
            save_flows=False,
        )

        # chd files
        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
            gwf, maxbound=maxchd, stress_period_data=cd6, save_flows=False
        )

        # output control
        oc = flopy.mf6.ModflowGwfoc(
            gwf,
            budget_filerecord=f"{name}.cbc",
            head_filerecord=f"{name}.hds",
            headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
            printrecord=[("HEAD", "LAST"), ("BUDGET", "ALL")],
        )

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf=test["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1 == line2
            else:
                compare_netcdf(base, gen)


@pytest.mark.regression
def test_gwfsto01(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "mesh2d",
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
            "netcdf_type": "structured",
        },
    }

    # spatial discretization data
    nlay, nrow, ncol = 3, 10, 10
    delr = [1000.0]
    delc = [2000.0]
    top = np.full((nrow, ncol), 0.0)
    botm = []
    botm.append(np.full((nrow, ncol), -100.0))
    botm.append(np.full((nrow, ncol), -150.0))
    botm.append(np.full((nrow, ncol), -350.0))
    botm = np.array(botm)

    # ic
    strt = np.full((nlay, nrow, ncol), 0.0)

    # npf
    # icelltype
    ic1 = np.full((nrow, ncol), np.int32(1))
    ic2 = np.full((nrow, ncol), np.int32(0))
    ic3 = np.full((nrow, ncol), np.int32(0))
    icelltype = np.array([ic1, ic2, ic3])

    # k
    hk2fact = 1.0 / 50.0
    hk2 = np.ones((nrow, ncol), dtype=float) * 0.5 * hk2fact
    hk2[0, :] = 1000.0 * hk2fact
    hk2[-1, :] = 1000.0 * hk2fact
    hk2[:, 0] = 1000.0 * hk2fact
    hk2[:, -1] = 1000.0 * hk2fact
    k1 = np.full((nrow, ncol), 20.0)
    k3 = np.full((nrow, ncol), 5.0)
    k = np.array([k1, hk2, k3])

    # k33
    k33_1 = np.full((nrow, ncol), 1e6)
    k33_2 = np.full((nrow, ncol), 7.5e-5)
    k33_3 = np.full((nrow, ncol), 1e6)
    k33 = np.array([k33_1, k33_2, k33_3])

    # sto
    iconvert = icelltype

    # storage and compaction data
    ss1 = np.full((nrow, ncol), 6e-4)
    ss2 = np.full((nrow, ncol), 3e-4)
    ss3 = np.full((nrow, ncol), 6e-4)
    ss = np.array([ss1, ss2, ss3])
    sy = np.full((nlay, nrow, ncol), 0.0)

    # define longnames
    delr_longname = "spacing along a row"
    delc_longname = "spacing along a column"
    top_longname = "cell top elevation"
    botm_longname = "cell bottom elevation"
    icelltype_longname = "confined or convertible indicator"
    k_longname = "hydraulic conductivity (L/T)"
    k33_longname = "hydraulic conductivity of third ellipsoid axis (L/T)"
    iconvert_longname = "convertible indicator"
    ss_longname = "specific storage"
    sy_longname = "specific yield"
    strt_longname = "starting head"

    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)
        os.mkdir(test_path)

        # create discretization
        dis = flopy.discretization.StructuredGrid(
            delc=np.array(delc * nrow),
            delr=np.array(delr * ncol),
            top=top,
            botm=botm,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            crs="EPSG:26918",
        )

        # create the dataset
        ds = create_dataset(
            "gwf6",
            "gwf_sto01",
            test["netcdf_type"],
            test["netcdf_output_file"],
            dis,
        )

        # add dis arrays
        ds.create_array("dis", "delc", dis.delc, ["nrow"], delc_longname)
        ds.create_array("dis", "delr", dis.delr, ["ncol"], delr_longname)
        ds.create_array("dis", "top", dis.top, ["nrow", "ncol"], top_longname)
        ds.create_array(
            "dis", "botm", dis.botm, ["nlay", "nrow", "ncol"], botm_longname
        )

        # add ic array
        ds.create_array("ic", "strt", strt, ["nlay", "nrow", "ncol"], strt_longname)

        # add npf arrays
        ds.create_array(
            "npf", "icelltype", icelltype, ["nlay", "nrow", "ncol"], icelltype_longname
        )
        ds.create_array("npf", "k", k, ["nlay", "nrow", "ncol"], k_longname)
        ds.create_array("npf", "k33", k33, ["nlay", "nrow", "ncol"], k33_longname)

        # add sto arrays
        ds.create_array(
            "sto", "iconvert", iconvert, ["nlay", "nrow", "ncol"], iconvert_longname
        )
        ds.create_array("sto", "ss", ss, ["nlay", "nrow", "ncol"], ss_longname)
        ds.create_array("sto", "sy", sy, ["nlay", "nrow", "ncol"], sy_longname)

        # write to netcdf
        ds.write(test_path)

        # compare
        compare_netcdf(
            os.path.join(base_path, test["netcdf_output_file"]),
            os.path.join(test_path, test["netcdf_output_file"]),
            projection=True,
        )


@pytest.mark.regression
def test_load_disv01b(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_disv01b": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
        },
    }
    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf="mesh2d")

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1.lower() == line2.lower()
            else:
                compare_netcdf(base, gen)


@pytest.mark.regression
def test_update_disv01b(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_disv01b": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
        },
    }

    nlay, nrow, ncol = 3, 3, 3
    ncpl = nrow * ncol
    strt = np.full((nlay, ncpl), 0.999)

    idomain = np.array(
        [
            [1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
        ]
    )

    botm = [
        [-15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0],
        [-25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0, -25.0],
        [-35.0, -35.0, -35.0, -35.0, -35.0, -35.0, -35.0, -35.0, -35.0],
    ]

    update = {
        "disv_idomain_l1": idomain[0],
        "disv_idomain_l2": idomain[1],
        "disv_idomain_l3": idomain[2],
        "disv_botm_l1": botm[0],
        "disv_botm_l2": botm[1],
        "disv_botm_l3": botm[2],
        "ic_strt_l1": strt[0],
        "ic_strt_l2": strt[1],
        "ic_strt_l3": strt[2],
    }

    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

        # get model instance
        gwf = sim.get_model("disv01b")

        # update disv idomain and botm
        gwf.disv.idomain = idomain
        gwf.disv.botm.set_data(botm)

        # update ic strt
        gwf.ic.strt.set_data(strt)

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf="mesh2d")

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1.lower() == line2.lower()
            else:
                compare_netcdf(base, gen, update=update)


@pytest.mark.regression
def test_create_disv01b(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_disv01b": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
            "netcdf_type": "mesh2d",
        },
    }

    name = "disv01b"
    nlay = 3
    nrow = 3
    ncol = 3
    delr = 10.0
    delc = 10.0
    top = 0
    botm = [-10, -20, -30]
    xoff = 100000000.0
    yoff = 100000000.0
    disvkwargs = get_disv_kwargs(nlay, nrow, ncol, delr, delc, top, botm, xoff, yoff)
    idomain = np.ones((nlay, nrow * ncol), dtype=int)
    idomain[0, 1] = 0
    disvkwargs["idomain"] = idomain

    # build
    ws = function_tmpdir / "ws"
    for idx, (dirname, test) in enumerate(tests.items()):
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # create simulation
        sim = flopy.mf6.MFSimulation(
            sim_name=name,
            version="mf6",
            exe_name="mf6",
            sim_ws=ws,
        )
        tdis = flopy.mf6.ModflowTdis(sim, start_date_time="2041-01-01t00:00:00-05:00")
        kwargs = {}
        kwargs["crs"] = "EPSG:26918"
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, **kwargs)
        ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY")
        disv = flopy.mf6.ModflowGwfdisv(gwf, **disvkwargs)
        ic = flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        npf = flopy.mf6.ModflowGwfnpf(gwf)
        spd = {0: [[(0, 0), 1.0], [(0, nrow * ncol - 1), 0.0]]}
        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, stress_period_data=spd)
        oc = flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{name}.hds",
            saverecord=[("HEAD", "ALL")],
        )

        # set path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf=test["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1 == line2
            else:
                compare_netcdf(base, gen)


@pytest.mark.regression
def test_disv01b(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_disv01b": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
            "netcdf_type": "mesh2d",
        },
    }

    nlay, nrow, ncol = 3, 3, 3
    ncpl = nrow * ncol
    # delr = 10.0
    # delc = 10.0
    # xoff = 100000000.0
    # yoff = 100000000.0

    vertices = [
        (0, 1.0000000e08, 1.0000003e08),
        (1, 1.0000001e08, 1.0000003e08),
        (2, 1.0000002e08, 1.0000003e08),
        (3, 1.0000003e08, 1.0000003e08),
        (4, 1.0000000e08, 1.0000002e08),
        (5, 1.0000001e08, 1.0000002e08),
        (6, 1.0000002e08, 1.0000002e08),
        (7, 1.0000003e08, 1.0000002e08),
        (8, 1.0000000e08, 1.0000001e08),
        (9, 1.0000001e08, 1.0000001e08),
        (10, 1.0000002e08, 1.0000001e08),
        (11, 1.0000003e08, 1.0000001e08),
        (12, 1.0000000e08, 1.0000000e08),
        (13, 1.0000001e08, 1.0000000e08),
        (14, 1.0000002e08, 1.0000000e08),
        (15, 1.0000003e08, 1.0000000e08),
    ]

    cell2d = [
        (0, 1.00000005e08, 1.00000025e08, 4, 0, 1, 5, 4),
        (1, 1.00000015e08, 1.00000025e08, 4, 1, 2, 6, 5),
        (2, 1.00000025e08, 1.00000025e08, 4, 2, 3, 7, 6),
        (3, 1.00000005e08, 1.00000015e08, 4, 4, 5, 9, 8),
        (4, 1.00000015e08, 1.00000015e08, 4, 5, 6, 10, 9),
        (5, 1.00000025e08, 1.00000015e08, 4, 6, 7, 11, 10),
        (6, 1.00000005e08, 1.00000005e08, 4, 8, 9, 13, 12),
        (7, 1.00000015e08, 1.00000005e08, 4, 9, 10, 14, 13),
        (8, 1.00000025e08, 1.00000005e08, 4, 10, 11, 15, 14),
    ]

    top = np.array(np.full((ncpl), 0.0))

    idomain = np.array(
        [
            [1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )

    botm = []
    botm.append(np.full((ncpl), -10.0))
    botm.append(np.full((ncpl), -20.0))
    botm.append(np.full((ncpl), -30.0))
    botm = np.array(botm)

    # npf
    icelltype = np.full((nlay, ncpl), np.int32(0))
    k = np.full((nlay, ncpl), 1.0)

    # ic
    strt = np.full((nlay, ncpl), 0.0)

    # define longnames
    top_longname = "model top elevation"
    botm_longname = "model bottom elevation"
    idomain_longname = "idomain existence array"
    icelltype_longname = "confined or convertible indicator"
    k_longname = "hydraulic conductivity (L/T)"
    strt_longname = "starting head"

    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)
        os.mkdir(test_path)

        # create discretization
        disv = VertexGrid(
            vertices=vertices,
            cell2d=cell2d,
            top=top,
            idomain=idomain,
            botm=botm,
            nlay=nlay,
            ncpl=ncpl,
            crs="EPSG:26918",
        )

        # create dataset
        ds = create_dataset(
            "gwf6",
            "disv01b",
            test["netcdf_type"],
            test["netcdf_output_file"],
            disv,
        )

        # add dis arrays
        ds.create_array("disv", "top", disv.top, ["ncpl"], top_longname)
        ds.create_array("disv", "botm", disv.botm, ["nlay", "ncpl"], botm_longname)
        ds.create_array(
            "disv", "idomain", disv.idomain, ["nlay", "ncpl"], idomain_longname
        )

        # add npf arrays
        ds.create_array(
            "npf", "icelltype", icelltype, ["nlay", "ncpl"], icelltype_longname
        )
        ds.create_array("npf", "k", k, ["nlay", "ncpl"], k_longname)

        # add ic arrays
        ds.create_array("ic", "strt", strt, ["nlay", "ncpl"], strt_longname)

        # write to netcdf
        ds.write(test_path)

        # compare
        compare_netcdf(
            os.path.join(base_path, test["netcdf_output_file"]),
            os.path.join(test_path, test["netcdf_output_file"]),
            projection=True,
        )


@pytest.mark.regression
def test_dis_transform(function_tmpdir, example_data_path):
    transform_ws = function_tmpdir
    cmp_pth = transform_ws / "compare"
    nc_types = ["mesh2d", "structured"]
    data_path_base = example_data_path / "mf6" / "netcdf" / "test_dis_transform"
    shutil.copytree(data_path_base, cmp_pth)

    # define transform / projection info
    kwargs = {}
    kwargs["crs"] = "EPSG:31370"
    kwargs["xll"] = 199000
    kwargs["yll"] = 215500
    kwargs["rotation"] = 30

    # create simulation
    nlay, nrow, ncol = 3, 10, 15
    sim = flopy.mf6.MFSimulation(sim_ws=transform_ws, sim_name="transform")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, complexity="simple")

    gwf = flopy.mf6.ModflowGwf(sim, modelname="transform", print_input=True, **kwargs)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=1.0,
        delc=1.0,
        top=10.0,
        botm=[0.0, -10.0, -30.0],
        xorigin=199000,
        yorigin=215500,
        angrot=30,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=[1, 1, 1],
    )
    strt = np.array([np.linspace(-0.999, 0.999, nlay * nrow * ncol)])
    flopy.mf6.ModflowGwfic(
        gwf,
        # strt=strt,
        strt=10.0,
    )
    data = {0: [[(0, 0, 0), 1.0000000], [(1, 0, 14), 0.0000000]]}
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=data,
    )

    for t in nc_types:
        ws = transform_ws / t
        sim.set_sim_path(ws)
        sim.write_simulation(netcdf=t)
        compare_netcdf(
            cmp_pth / f"transform.{t}.nc", ws / "transform.in.nc", projection=True
        )


@requires_exe("triangle")
@pytest.mark.regression
def test_disv_transform(function_tmpdir, example_data_path):
    # create triangular grid
    from flopy.utils.triangle import Triangle

    transform_ws = function_tmpdir
    cmp_pth = transform_ws / "compare"
    data_path_base = example_data_path / "mf6" / "netcdf" / "test_disv_transform"
    shutil.copytree(data_path_base, cmp_pth)

    nc_type = "mesh2d"

    triangle_ws = transform_ws / "triangle"
    triangle_ws.mkdir()

    active_area = [(0, 0), (0, 1000), (1000, 1000), (1000, 0)]
    tri = Triangle(model_ws=triangle_ws, angle=30)
    tri.add_polygon(active_area)
    tri.add_region((1, 1), maximum_area=50**2)

    tri.build()

    # strt array
    strt = np.array([np.linspace(-0.999, 0.999, len(tri.get_cell2d()))])

    ###
    # vertex discretization based run
    vertex_ws = triangle_ws / "vertex"
    os.mkdir(vertex_ws)

    # build vertex grid object
    vgrid = flopy.discretization.VertexGrid(
        vertices=tri.get_vertices(),
        cell2d=tri.get_cell2d(),
        xoff=199000,
        yoff=215500,
        crs=31370,
        angrot=30,
    )

    ds = create_dataset(
        "example",  # model type
        "trimodel",  # model name
        nc_type,  # netcdf file type
        "tri.nc",  # netcdf file name
        vgrid,
    )

    ds.create_array("start_conditions", "head", strt, ["nlay", "ncpl"], None)

    # write to netcdf
    ds.write(vertex_ws)

    ###
    # MOFLOW 6 sim based run
    mf6_ws = triangle_ws / "mf6"
    sim = flopy.mf6.MFSimulation(
        sim_name="tri_disv",
        version="mf6",
        exe_name="mf6",
        sim_ws=mf6_ws,
    )
    tdis = flopy.mf6.ModflowTdis(sim, start_date_time="2041-01-01t00:00:00-05:00")

    # set projection and transform info
    kwargs = {}
    kwargs["crs"] = "EPSG:31370"
    kwargs["xll"] = 199000
    kwargs["yll"] = 215500
    kwargs["rotation"] = 30

    gwf = flopy.mf6.ModflowGwf(sim, modelname="tri", **kwargs)
    ims = flopy.mf6.ModflowIms(sim, print_option="SUMMARY")
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=1,
        ncpl=tri.ncpl,
        nvert=tri.nvert,
        vertices=tri.get_vertices(),
        cell2d=tri.get_cell2d(),
        top=0,
        botm=-150,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
    npf = flopy.mf6.ModflowGwfnpf(gwf)
    data = {0: [[(0, 0), 1.0000000], [(0, 14), 0.0000000]]}
    chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, stress_period_data=data)

    sim.write_simulation(netcdf=nc_type)

    compare_netcdf(cmp_pth / f"tri.{nc_type}.nc", mf6_ws / "tri.in.nc", projection=True)


@pytest.mark.regression
def test_utlncf_load(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_utlncf_load": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
        },
    }
    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf="mesh2d")

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_path)
            if os.path.isfile(os.path.join(test_path, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, f))
        ]

        assert len(gen_files) == len(base_files)
        for f in base_files:
            base = os.path.join(base_path, f)
            gen = os.path.join(test_path, f)
            if f != test["netcdf_output_file"]:
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        if line1.lower().startswith("  wkt"):
                            break
                        assert line1.lower() == line2.lower()
            else:
                compare_netcdf_data(base, gen)


@pytest.mark.regression
def test_utlncf_create(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_utlncf_create": {
            "base_sim_dir": "disv01b",
            "netcdf_output_file": "disv01b.in.nc",
        },
    }
    ws = function_tmpdir / "ws"
    for dirname, test in tests.items():
        data_path = os.path.join(data_path_base, dirname, test["base_sim_dir"])

        # copy example data into working directory
        base_path = os.path.join(ws, f"{dirname}_base")
        test_path = os.path.join(ws, f"{dirname}_test")
        shutil.copytree(data_path, base_path)

        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

        # set simulation path and write simulation
        sim.set_sim_path(test_path)
        sim.write_simulation(netcdf="mesh2d")

        # compare generated files
        compare_path = os.path.join(base_path, "compare")
        base = os.path.join(compare_path, "disv01b.in.nc")
        gen = os.path.join(test_path, test["netcdf_output_file"])
        compare_netcdf_data(base, gen)
