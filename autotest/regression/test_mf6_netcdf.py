import math
import os
import shutil
from traceback import format_exc
from warnings import warn

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.datautil import DatumUtil
from flopy.utils.gridutil import get_disv_kwargs
from flopy.utils.model_netcdf import create_dataset


def compare_netcdf(base, gen, update=None):
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
        assert coordname in xrg.coords
        # TODO
        assert np.allclose(xrb.coords[coordname].data, xrg.coords[coordname].data)
        for a in da.attrs:
            assert a in xrg.coords[coordname].attrs
            assert da.attrs[a] == xrg.coords[coordname].attrs[a]

    # variables
    for varname, da in xrb.data_vars.items():
        print(varname)
        if varname == "projection":
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

        # check variable name
        assert varname in xrg.data_vars

        # check variable attributes
        for a in da.attrs:
            # TODO long_name
            # if a == "long_name":
            #    continue
            print(a)
            assert da.attrs[a] == xrg.data_vars[varname].attrs[a]

        # check variable data
        print(f"NetCDF file check data equivalence for variable: {varname}")
        if update and varname in update:
            assert np.allclose(update[varname], xrg.data_vars[varname].data)
        else:
            assert np.allclose(da.data, xrg.data_vars[varname].data)


@pytest.mark.regression
def test_load_gwfsto01(function_tmpdir, example_data_path):
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
        },
    }
    ws = function_tmpdir / "ws"
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_model_folder)
        # change simulation path
        sim.set_sim_path(test_model_folder)
        # write example simulation to reset path
        sim.write_simulation()

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
        },
        "test_gwf_sto01_structured": {
            "base_sim_dir": "gwf_sto01",
            "netcdf_output_file": "gwf_sto01.in.nc",
        },
    }

    nlay, nrow, ncol = 3, 10, 10

    dis_delr = np.array([1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010])
    dis_delc = [2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090]

    # ic
    strt1 = np.full((nrow, ncol), 0.15)
    strt2 = np.full((nrow, ncol), 0.21)
    strt3 = np.full((nrow, ncol), 1.21)
    ic_strt = np.array([strt1, strt2, strt3])

    update = {
        "dis_delr": dis_delr,
        "dis_delc": dis_delc,
        "ic_strt": ic_strt,
        "ic_strt_l1": ic_strt[0].flatten(),
        "ic_strt_l2": ic_strt[1].flatten(),
        "ic_strt_l3": ic_strt[2].flatten(),
    }

    ws = function_tmpdir / "ws"
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_model_folder)
        # get model instance
        gwf = sim.get_model("gwf_sto01")
        # update dis delr and delc
        gwf.dis.delr = dis_delr
        gwf.dis.delc = dis_delc
        # update ic strt
        gwf.ic.strt.set_data(ic_strt)
        # change simulation path
        sim.set_sim_path(test_model_folder)
        # write example simulation to reset path
        sim.write_simulation()

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
    for idx, (base_folder, test_info) in enumerate(tests.items()):
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)

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

        sim.set_sim_path(test_model_folder)
        sim.write_simulation(netcdf=test_info["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
    ic1 = np.full((nrow, ncol), 1)
    ic2 = np.full((nrow, ncol), 0)
    ic3 = np.full((nrow, ncol), 0)
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

    sy = np.full((nlay, nrow, ncol), 0)

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
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        os.mkdir(test_model_folder)

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

        ds = create_dataset(
            "gwf6",
            "gwf_sto01",
            test_info["netcdf_type"],
            test_info["netcdf_output_file"],
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
        ds.write(test_model_folder)

        # compare
        compare_netcdf(
            os.path.join(base_model_folder, test_info["netcdf_output_file"]),
            os.path.join(test_model_folder, test_info["netcdf_output_file"]),
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
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_model_folder)
        # change simulation path
        sim.set_sim_path(test_model_folder)
        # write example simulation to reset path
        sim.write_simulation()

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        # load example
        sim = flopy.mf6.MFSimulation.load(sim_ws=base_model_folder)
        # get model instance
        gwf = sim.get_model("disv01b")
        # update disv idomain and botm
        gwf.disv.idomain = idomain
        gwf.disv.botm.set_data(botm)
        # update ic strt
        gwf.ic.strt.set_data(strt)
        # change simulation path
        sim.set_sim_path(test_model_folder)
        # write example simulation to reset path
        sim.write_simulation()

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
    for idx, (base_folder, test_info) in enumerate(tests.items()):
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)

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

        sim.set_sim_path(test_model_folder)
        sim.write_simulation(netcdf=test_info["netcdf_type"])

        # compare generated files
        gen_files = [
            f
            for f in os.listdir(test_model_folder)
            if os.path.isfile(os.path.join(test_model_folder, f))
        ]
        base_files = [
            f
            for f in os.listdir(base_model_folder)
            if os.path.isfile(os.path.join(base_model_folder, f))
        ]
        assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info["netcdf_output_file"]:
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
        ]
    )

    botm = []
    botm.append(np.full((ncpl), -10.0))
    botm.append(np.full((ncpl), -20.0))
    botm.append(np.full((ncpl), -30.0))
    botm = np.array(botm)

    # npf
    icelltype = np.full((nlay, ncpl), 0)
    k = np.full((nlay, ncpl), 1)

    # ic
    strt = np.full((nlay, ncpl), 0.0)

    top_longname = "model top elevation"
    botm_longname = "model bottom elevation"
    idomain_longname = "idomain existence array"
    icelltype_longname = "confined or convertible indicator"
    k_longname = "hydraulic conductivity (L/T)"
    strt_longname = "starting head"

    ws = function_tmpdir / "ws"
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(data_path_base, base_folder, test_info["base_sim_dir"])
        # copy example data into working directory
        base_model_folder = os.path.join(ws, f"{base_folder}_base")
        test_model_folder = os.path.join(ws, f"{base_folder}_test")
        shutil.copytree(data_path, base_model_folder)
        os.mkdir(test_model_folder)

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
            test_info["netcdf_type"],
            test_info["netcdf_output_file"],
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
        ds.write(test_model_folder)

        # compare
        compare_netcdf(
            os.path.join(base_model_folder, test_info["netcdf_output_file"]),
            os.path.join(test_model_folder, test_info["netcdf_output_file"]),
        )
