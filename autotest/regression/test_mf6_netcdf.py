import math
import os
import shutil
from traceback import format_exc
from warnings import warn

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.utils import import_optional_dependency
from flopy.utils.datautil import DatumUtil


class TestInfo:
    def __init__(
        self,
        original_simulation_folder,
        netcdf_simulation_folder,
        netcdf_output_file,
    ):
        self.original_simulation_folder = original_simulation_folder
        self.netcdf_simulation_folder = netcdf_simulation_folder
        self.netcdf_output_file = netcdf_output_file


@requires_pkg("xarray")
@requires_exe("mf6")
@pytest.mark.regression
def test_load_netcdf_gwfsto01(function_tmpdir, example_data_path):
    xr = import_optional_dependency("xarray")
    data_path_base = example_data_path / "mf6" / "netcdf"
    tests = {
        "test_gwf_sto01_mesh": TestInfo(
            "gwf_sto01",
            "gwf_sto01_write",
            "gwf_sto01.ugrid.nc",
        ),
        "test_gwf_sto01_structured": TestInfo(
            "gwf_sto01",
            "gwf_sto01_write",
            "gwf_sto01.structured.nc",
        ),
    }
    ws = function_tmpdir / "ws"
    for base_folder, test_info in tests.items():
        print(f"RUNNING TEST: {base_folder}")
        data_path = os.path.join(
            data_path_base, base_folder, test_info.original_simulation_folder
        )
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
        # assert len(gen_files) == len(base_files)
        for f in base_files:
            print(f"cmp => {f}")
            base = os.path.join(base_model_folder, f)
            gen = os.path.join(test_model_folder, f)
            if f != test_info.netcdf_output_file:
                # "gwf_sto01.dis.ncf":   # TODO wkt string missing on write?
                with open(base, "r") as file1, open(gen, "r") as file2:
                    # Skip first line
                    next(file1)
                    next(file2)

                    for line1, line2 in zip(file1, file2):
                        assert line1 == line2
            else:
                # TODO compare nc files
                assert os.path.exists(gen)
        continue


@requires_pkg("xarray")
@requires_exe("mf6")
@pytest.mark.regression
def test_create_netcdf_gwfsto01(function_tmpdir, example_data_path):
    xr = import_optional_dependency("xarray")

    cases = ["structured", "mesh2d"]

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
    for name in cases:
        # name = cases[0]
        ws = function_tmpdir / name

        # build MODFLOW 6 files
        sim = flopy.mf6.MFSimulation(
            sim_name=name, version="mf6", exe_name="mf6", sim_ws=ws
        )
        # create tdis package
        tdis = flopy.mf6.ModflowTdis(
            sim, time_units="DAYS", nper=nper, perioddata=tdis_rc
        )

        # create gwf model
        top = 0.0
        zthick = [top - botm[0], botm[0] - botm[1], botm[1] - botm[2]]
        elevs = [top] + botm

        gwf = flopy.mf6.ModflowGwf(
            sim, modelname=name, newtonoptions=newtonoptions, save_flows=True
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

        sim.write_simulation(netcdf=name)

        try:
            success, buff = flopy.run_model(
                "mf6",
                ws / "mfsim.nam",
                model_ws=ws,
                report=True,
            )
        except Exception:
            warn(
                "MODFLOW 6 serial test",
                name,
                f"failed with error:\n{format_exc()}",
            )
            success = False

        assert success
