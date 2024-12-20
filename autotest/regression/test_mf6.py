import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.mf6 import (
    ExtFileAction,
    MFModel,
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisv,
    ModflowGwfdrn,
    ModflowGwfevt,
    ModflowGwfevta,
    ModflowGwfghb,
    ModflowGwfgnc,
    ModflowGwfgwf,
    ModflowGwfgwt,
    ModflowGwfhfb,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrch,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfsfr,
    ModflowGwfsto,
    ModflowGwfwel,
    ModflowGwtadv,
    ModflowGwtdis,
    ModflowGwtic,
    ModflowGwtmst,
    ModflowGwtoc,
    ModflowGwtssm,
    ModflowIms,
    ModflowTdis,
    ModflowUtlhpc,
    ModflowUtltas,
)
from flopy.mf6.data.mfdatastorage import DataStorageType
from flopy.mf6.mfbase import FlopyException, MFDataException
from flopy.mf6.utils import testutils
from flopy.utils import CellBudgetFile
from flopy.utils.compare import compare_heads
from flopy.utils.datautil import PyListUtil

pytestmark = pytest.mark.mf6


@requires_exe("mf6")
@pytest.mark.regression
def test_ts(function_tmpdir, example_data_path):
    ws = function_tmpdir / "ws"
    name = "test_ts"

    # create the flopy simulation and tdis objects
    sim = flopy.mf6.MFSimulation(
        sim_name=name, exe_name="mf6", version="mf6", sim_ws=ws
    )
    tdis_rc = [(1.0, 1, 1.0), (10.0, 5, 1.0), (10.0, 5, 1.0), (10.0, 1, 1.0)]
    tdis_package = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim, time_units="DAYS", nper=4, perioddata=tdis_rc
    )
    # create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)
    # create the flopy iterative model solver (ims) package object
    ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
    # create the discretization package
    bot = np.linspace(-3.0, -50.0 / 3.0, 3)
    delrow = delcol = 4.0
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nogrb=True,
        nlay=3,
        nrow=101,
        ncol=101,
        delr=delrow,
        delc=delcol,
        top=0.0,
        botm=bot,
    )
    # create the initial condition (ic) and node property flow (npf) packages
    ic_package = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, strt=50.0)
    npf_package = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        save_flows=True,
        icelltype=[1, 0, 0],
        k=[5.0, 0.1, 4.0],
        k33=[0.5, 0.005, 0.1],
    )
    oc = ModflowGwfoc(
        gwf,
        budget_filerecord=[(f"{name}.cbc",)],
        head_filerecord=[(f"{name}.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("HEAD", "ALL")],
    )

    # build ghb stress period data
    ghb_spd_ts = {}
    ghb_period = []
    for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
        for row in range(0, 15):
            ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
    ghb_spd_ts[0] = ghb_period

    # build ts data
    ts_data = []
    for n in range(0, 365):
        time = float(n / 11.73)
        val = float(n / 60.0)
        ts_data.append((time, val))
    ts_dict = {
        "filename": "tides.ts",
        "time_series_namerecord": "tide",
        "timeseries": ts_data,
        "interpolation_methodrecord": "linearend",
        "sfacrecord": 1.1,
    }

    # build ghb package
    ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
        gwf,
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        timeseries=ts_dict,
        pname="ghb",
        maxbound=30,
        stress_period_data=ghb_spd_ts,
    )

    # set required time series attributes
    ghb.ts.time_series_namerecord = "tides"

    # clean up for next example
    gwf.remove_package("ghb")

    # build ghb stress period data
    ghb_spd_ts = {}
    ghb_period = []
    for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
        for row in range(0, 15):
            if row < 10:
                ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
            else:
                ghb_period.append(((layer, row, 9), "wl", cond, "Estuary-L2"))
    ghb_spd_ts[0] = ghb_period

    # build ts data
    ts_data = []
    for n in range(0, 365):
        time = float(n / 11.73)
        val = float(n / 60.0)
        ts_data.append((time, val))
    ts_data2 = []
    for n in range(0, 365):
        time = float(n / 11.73)
        val = float(n / 30.0)
        ts_data2.append((time, val))
    ts_data3 = []
    for n in range(0, 365):
        time = float(n / 11.73)
        val = float(n / 20.0)
        ts_data3.append((time, val))

    # build ghb package
    ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
        gwf,
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        pname="ghb",
        maxbound=30,
        stress_period_data=ghb_spd_ts,
    )

    # initialize first time series
    ghb.ts.initialize(
        filename="tides.ts",
        timeseries=ts_data,
        time_series_namerecord="tides",
        interpolation_methodrecord="linearend",
        sfacrecord=1.1,
    )

    # append additional time series
    ghb.ts.append_package(
        filename="wls.ts",
        timeseries=ts_data2,
        time_series_namerecord="wl",
        interpolation_methodrecord="stepwise",
        sfacrecord=1.2,
    )
    # append additional time series
    ghb.ts.append_package(
        filename="wls2.ts",
        timeseries=ts_data3,
        time_series_namerecord="wl2",
        interpolation_methodrecord="stepwise",
        sfacrecord=1.3,
    )

    sim.write_simulation()
    ret = sim.run_simulation()
    assert ret
    sim2 = flopy.mf6.MFSimulation.load("mfsim.nam", sim_ws=ws, exe_name="mf6")
    sim2_ws = os.path.join(ws, "2")
    sim2.set_sim_path(sim2_ws)
    sim2.write_simulation()
    ret = sim2.run_simulation()
    assert ret

    # compare datasets
    model2 = sim2.get_model()
    ghb_m2 = model2.get_package("ghb")
    wls_m2 = ghb_m2.ts[1]
    wls_m1 = ghb.ts[1]

    ts_m1 = wls_m1.timeseries.get_data()
    ts_m2 = wls_m2.timeseries.get_data()

    assert ts_m1[0][1] == 0.0
    assert ts_m1[30][1] == 1.0
    for m1_line, m2_line in zip(ts_m1, ts_m2):
        assert abs(m1_line[1] - m2_line[1]) < 0.000001

    # compare output to expected results
    head_1 = os.path.join(ws, f"{name}.hds")
    head_2 = os.path.join(sim2_ws, f"{name}.hds")
    outfile = os.path.join(ws, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[head_1],
        files2=[head_2],
        outfile=outfile,
    )


@requires_exe("mf6")
@pytest.mark.regression
def test_np001(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "np001"
    model_name = "np001_mod"
    data_path = example_data_path / "mf6" / "create_tests" / test_ex_name
    ws = function_tmpdir / "ws"
    # copy example data into working directory
    shutil.copytree(data_path, ws)

    expected_output_folder = data_path / "expected_output"
    expected_head_file = expected_output_folder / "np001_mod.hds"
    expected_cbc_file = expected_output_folder / "np001_mod.cbc"

    # model tests
    test_sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=ws,
        continue_=True,
        memory_print_option="summary",
    )
    name = test_sim.name_file
    assert name.continue_.get_data()
    assert name.nocheck.get_data() is None
    assert name.memory_print_option.get_data() == "summary"

    kwargs = {}
    kwargs["bad_kwarg"] = 20
    try:
        ex = False
        bad_model = ModflowGwf(
            test_sim,
            modelname=model_name,
            model_nam_file=f"{model_name}.nam",
            **kwargs,
        )
    except FlopyException:
        ex = True
    assert ex is True

    kwargs = {}
    kwargs["xul"] = 20.5
    good_model = ModflowGwf(
        test_sim,
        modelname=model_name,
        model_nam_file=f"{model_name}.nam",
        model_rel_path="model_folder",
        **kwargs,
    )

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=data_path,
        write_headers=False,
    )
    sim.set_sim_path(ws)
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(2.0, 1, 1.0)]
    )
    # specifying the tdis package twice should remove the old tdis package
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=2, perioddata=tdis_rc)
    # first ims file to be replaced
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename="old_name.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=10,
        under_relaxation="NONE",
        inner_maximum=10,
        inner_dvclose=0.001,
        linear_acceleration="CG",
        preconditioner_levels=2,
        preconditioner_drop_tolerance=0.00001,
        number_orthogonalizations=5,
    )
    # replace with real ims file
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename=f"{test_ex_name}.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )

    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    # test case insensitive lookup
    assert sim.get_model(model_name.upper()) is not None

    # test getting model using attribute
    model = sim.np001_mod
    assert model is not None and model.name == "np001_mod"
    tdis = sim.tdis
    assert tdis is not None and tdis.package_type == "tdis"

    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=1,
        delr=100.0,
        delc=100.0,
        top=60.0,
        botm=50.0,
        filename=f"{model_name}.dis",
        pname="mydispkg",
    )
    # specifying dis package twice with the same name should automatically
    # remove the old dis package
    top = {"filename": "top.bin", "data": 100.0, "binary": True}
    botm = {"filename": "botm.bin", "data": 50.0, "binary": True}
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=top,
        botm=botm,
        filename=f"{model_name}.dis",
        pname="mydispkg",
    )
    top_data = dis_package.top.get_data()
    assert top_data[0, 0] == 100.0
    ic_package = ModflowGwfic(
        model, strt="initial_heads.txt", filename=f"{model_name}.ic"
    )
    npf_package = ModflowGwfnpf(
        model,
        pname="npf_1",
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=5.0,
    )

    # remove package test using .remove_package(name)
    assert model.get_package(npf_package.package_name) is not None
    model.remove_package(npf_package.package_name)
    assert model.get_package(npf_package.package_name) is None
    # remove package test using .remove()
    npf_package = ModflowGwfnpf(
        model,
        pname="npf_1",
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=5.0,
    )
    npf_package.remove()
    assert model.get_package(npf_package.package_name) is None

    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=5.0,
    )

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("np001_mod 1.cbc",)],
        head_filerecord=[("np001_mod 1.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("HEAD", "ALL")],
    )
    empty_sp_text = oc_package.saverecord.get_file_entry(1)
    assert empty_sp_text == ""
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    oc_package.saverecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)

    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )

    # verify templates with and without cellid expanded
    pkd = ModflowGwfsfr.packagedata.empty(model)
    assert pkd.dtype.names[1] == "cellid"
    pkd_ex = ModflowGwfsfr.packagedata.empty(model, cellid_expanded=True)
    assert pkd_ex.dtype.names[1] == "layer"
    assert pkd_ex.dtype.names[2] == "row"
    assert pkd_ex.dtype.names[3] == "column"

    pkd_dtype = ModflowGwfsfr.packagedata.dtype(model)
    assert pkd_dtype[1][0] == "cellid"
    pkd_dtype_ex = ModflowGwfsfr.packagedata.dtype(model, cellid_expanded=True)
    assert pkd_dtype_ex[1][0] == "layer"
    assert pkd_dtype_ex[2][0] == "row"
    assert pkd_dtype_ex[3][0] == "column"

    # test saving a binary file with list data
    well_spd = {
        0: {
            "filename": "wel 0.bin",
            "binary": True,
            "data": [(0, 0, 4, -2000.0), (0, 0, 7, -2.0)],
        },
        1: None,
    }
    wel_package = ModflowGwfwel(
        model,
        filename=f"well_folder\\{model_name}.wel",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=well_spd,
    )
    wel_package.stress_period_data.add_transient_key(1)
    wel_package.stress_period_data.set_data({1: {"filename": "wel.txt", "factor": 1.0}})

    # test getting data from a binary file
    well_data = wel_package.stress_period_data.get_data(0)
    assert well_data[0][0] == (0, 0, 4)
    assert well_data[0][1] == -2000.0

    drn_package = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        timeseries=[(0.0, 60.0), (100000.0, 60.0)],
        stress_period_data=[((0, 0, 0), 80, "drn_1")],
    )
    drn_package.ts.time_series_namerecord = "drn_1"
    drn_package.ts.interpolation_methodrecord = "linearend"

    riv_spd = {
        0: {
            "filename": os.path.join("riv_folder", "riv.txt"),
            "data": [((0, 0, 9), 110, 90.0, 100.0, 1.0, 2.0, 3.0)],
        }
    }
    riv_package = ModflowGwfriv(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        auxiliary=["var1", "var2", "var3"],
        stress_period_data=riv_spd,
    )
    riv_data = riv_package.stress_period_data.get_data(0)
    assert riv_data[0][0] == (0, 0, 9)
    assert riv_data[0][1] == 110
    assert riv_data[0][2] == 90.0
    assert riv_data[0][3] == 100.0
    assert riv_data[0][4] == 1.0
    assert riv_data[0][5] == 2.0
    assert riv_data[0][6] == 3.0

    # verify package look-up
    pkgs = model.get_package()
    assert len(pkgs) == 9
    pkg = model.get_package("oc")
    assert isinstance(pkg, ModflowGwfoc)
    pkg = sim.get_package("tdis")
    assert isinstance(pkg, ModflowTdis)
    pkg = model.get_package("mydispkg")
    assert isinstance(pkg, ModflowGwfdis) and pkg.package_name == "mydispkg"
    pkg = model.mydispkg
    assert isinstance(pkg, ModflowGwfdis) and pkg.package_name == "mydispkg"

    # verify external file contents
    array_util = PyListUtil()
    ic_data = ic_package.strt
    ic_array = ic_data.get_data()
    assert array_util.array_comp(
        ic_array,
        [[[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]]],
    )

    # make folder to save simulation
    sim.set_sim_path(ws)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()
    assert sim.simulation_data.max_columns_of_data == dis_package.ncol.get_data()
    # test package file with relative path to simulation path
    wel_path = os.path.join(ws, "well_folder", f"{model_name}.wel")
    assert os.path.exists(wel_path)
    # test data file with relative path to simulation path
    riv_path = os.path.join(ws, "riv_folder", "riv.txt")
    assert os.path.exists(riv_path)

    # run simulation
    sim.run_simulation()

    # inspect cells
    cell_list = [(0, 0, 0), (0, 0, 4), (0, 0, 9)]
    out_file = function_tmpdir / "inspect_test_np001.csv"
    model.inspect_cells(cell_list, output_file_path=out_file, stress_period=0)

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file, precision="double")
    budget_frf_valid = np.array(budget_obj.get_data(text="RIV", full3D=False))

    # compare output to expected results
    head_new = os.path.join(ws, "np001_mod 1.hds")
    outfile = os.path.join(ws, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )
    # budget_frf = sim.simulation_data.mfdata[(model_name, "CBC", "RIV")]
    budget_frf = model.output.budget().get_data(text="RIV", full3D=False)
    assert array_util.riv_array_comp(budget_frf_valid, budget_frf)

    # clean up
    sim.delete_output_files()

    # test path changes, model file path relative to the simulation folder
    md_folder = "model_folder"
    model.set_model_relative_path(md_folder)
    run_folder_new = os.path.join(ws, md_folder)
    # set all data external
    sim.set_all_data_external(external_data_folder=function_tmpdir / "data")
    sim.write_simulation()

    # test file with relative path to model relative path
    wel_path = os.path.join(ws, md_folder, "well_folder", f"{model_name}.wel")
    assert os.path.exists(wel_path)
    # test data file was recreated by set_all_data_external
    riv_path = function_tmpdir / "data" / "np001_mod.riv_stress_period_data_1.txt"
    assert os.path.exists(riv_path)

    assert sim.simulation_data.max_columns_of_data == dis_package.ncol.get_data()
    # run simulation from new path with external files
    sim.run_simulation()

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file, precision="double")
    budget_frf_valid = np.array(budget_obj.get_data(text="RIV", full3D=False))

    # compare output to expected results
    head_new = os.path.join(run_folder_new, "np001_mod 1.hds")
    outfile = os.path.join(run_folder_new, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    budget_frf = model.output.budget().get_data(text="RIV", full3D=False)
    assert array_util.riv_array_comp(budget_frf_valid, budget_frf)

    # clean up
    sim.delete_output_files()

    # test rename all packages
    rename_folder = os.path.join(ws, "rename")
    sim.rename_all_packages("file_rename")
    sim.set_sim_path(rename_folder)
    sim.write_simulation()

    sim.run_simulation()
    sim.delete_output_files()

    # test error checking
    sim.simulation_data.verify_data = False
    drn_package = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        timeseries=[(0.0, 60.0), (100000.0, 60.0)],
        stress_period_data=[
            ((100, 0, 0), np.nan, "drn_1"),
            ((0, 0, 0), 10.0, "drn_2"),
        ],
    )
    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=100001.0,
        k33=1e-12,
    )
    sim.simulation_data.verify_data = True
    chk = sim.check()
    summary = ".".join(chk[0].summary_array.desc)
    assert "drn_1 package: invalid BC index" in summary
    assert (
        "npf package: vertical hydraulic conductivity values below "
        "checker threshold of 1e-11" in summary
    )
    assert (
        "npf package: horizontal hydraulic conductivity values above "
        "checker threshold of 100000.0" in summary
    )
    data_invalid = False
    try:
        drn_package = ModflowGwfdrn(
            model,
            print_input=True,
            print_flows=True,
            save_flows=True,
            maxbound=1,
            timeseries=[(0.0, 60.0), (100000.0, 60.0)],
            stress_period_data=[((0, 0, 0), 10.0)],
        )
    except MFDataException:
        data_invalid = True
    assert data_invalid

    # test -1, -1, -1 cellid
    well_spd = {0: [(-1, -1, -1, -2000.0), (0, 0, 7, -2.0)]}
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=well_spd,
    )
    wel_package.write()
    mpath = sim.simulation_data.mfpath.get_model_path(model.name)
    spath = sim.simulation_data.mfpath.get_sim_path()
    found_cellid = False
    with open(os.path.join(mpath, "np001_mod.wel")) as fd:
        for line in fd:
            line_lst = line.strip().split()
            if (
                len(line) > 2
                and line_lst[0] == "0"
                and line_lst[1] == "0"
                and line_lst[2] == "0"
            ):
                found_cellid = True
    assert found_cellid

    # test empty stress period and remove output
    well_spd = {0: [(-1, -1, -1, -2000.0), (0, 0, 7, -2.0)], 1: []}
    wel_package = ModflowGwfwel(
        model,
        pname="wel_2",
        filename="file_rename.wel",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=well_spd,
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("np001_mod 1.cbc",)],
        head_filerecord=[("np001_mod 1.hds",)],
        saverecord={0: []},
        printrecord={0: []},
    )
    sim.write_simulation()
    found_begin = False
    found_end = False
    text_between_begin_and_end = False
    with open(os.path.join(mpath, "file_rename.wel")) as fd:
        for line in fd:
            if line.strip().lower() == "begin period  2":
                found_begin = True
            elif found_begin and not found_end:
                if line.strip().lower() == "end period  2":
                    found_end = True
                else:
                    if len(line.strip()) > 0:
                        text_between_begin_and_end = True
    assert found_begin and found_end and not text_between_begin_and_end

    # test loading and re-writing empty stress period
    test_sim = MFSimulation.load(
        test_ex_name,
        "mf6",
        "mf6",
        spath,
        write_headers=False,
    )
    # test to make sure oc empty record dictionary is set
    oc = test_sim.get_model().get_package("oc")
    assert oc.saverecord.empty_keys[0] is True
    # test wel package
    wel = test_sim.get_model().get_package("wel_2")
    wel._filename = "np001_spd_test.wel"
    wel.write()
    found_begin = False
    found_end = False
    text_between_begin_and_end = False
    with open(os.path.join(mpath, "np001_spd_test.wel")) as fd:
        for line in fd:
            if line.strip().lower() == "begin period  2":
                found_begin = True
            elif found_begin and not found_end:
                if line.strip().lower() == "end period  2":
                    found_end = True
                else:
                    if len(line.strip()) > 0:
                        text_between_begin_and_end = True
    assert found_begin and found_end and not text_between_begin_and_end

    # test adding package with invalid data
    try:
        error_occurred = False
        well_spd = {
            0: {
                "filename": "wel0.bin",
                "binary": True,
                "data": [((0, 0, 4), -2000.0), ((0, 0, 7), -2.0)],
            }
        }
        wel_package = ModflowGwfwel(
            model,
            boundnames=True,
            print_input=True,
            print_flows=True,
            save_flows=True,
            maxbound=2,
            stress_period_data=well_spd,
        )
    except MFDataException:
        error_occurred = True
    assert error_occurred


@requires_exe("mf6")
@pytest.mark.regression
def test_np002(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "np002"
    model_name = "np002_mod"
    data_folder = example_data_path / "mf6" / "create_tests" / test_ex_name
    ws = function_tmpdir / "ws"
    # copy example data into working directory
    shutil.copytree(data_folder, ws)
    expected_output_folder = data_folder / "expected_output"
    expected_head_file = expected_output_folder / "np002_mod.hds"
    expected_cbc_file = expected_output_folder / "np002_mod.cbc"

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=ws,
        nocheck=True,
    )
    sim.set_sim_path(ws)
    sim.simulation_data.max_columns_of_data = 22

    name = sim.name_file
    assert name.continue_.get_data() is None
    assert name.nocheck.get_data() is True
    assert name.memory_print_option.get_data() is None

    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=2, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    sim.register_ims_package(ims_package, [model.name])

    # get rid of top_data.txt so that a later test does not automatically pass
    top_data_file = os.path.join(ws, "top data.txt")
    if os.path.isfile(top_data_file):
        os.remove(top_data_file)
    # test loading data to be stored in a file and loading data from a file
    # using the "dictionary" input format
    top = {
        "filename": "top data.txt",
        "factor": 1.0,
        "data": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }
    botm = {"filename": "botm.txt", "factor": 1.0}
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=top,
        botm=botm,
        idomain=2,
        filename=f"{model_name}.dis",
    )
    assert sim.simulation_data.max_columns_of_data == 22
    sim.simulation_data.max_columns_of_data = dis_package.ncol.get_data()

    ic_vals = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ic_package = ModflowGwfic(model, strt=ic_vals, filename=f"{model_name}.ic")
    ic_package.strt.store_as_external_file("initial_heads.txt")
    npf_package = ModflowGwfnpf(model, save_flows=True, icelltype=1, k=100.0)
    npf_package.k.store_as_external_file("k.bin", binary=True)
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("np002_mod.cbc",)],
        head_filerecord=[("np002_mod.hds",)],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    oc_package.saverecord.add_transient_key(1)
    oc_package.saverecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)

    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=0, ss=0.000001, sy=None, pname="sto_t"
    )
    sto_package.check()

    model.remove_package("sto_t")
    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )
    hfb_package = ModflowGwfhfb(
        model,
        print_input=True,
        maxhfb=1,
        stress_period_data=[((0, 0, 3), (0, 0, 4), 0.00001)],
    )
    chd_package = ModflowGwfchd(
        model,
        print_input=True,
        print_flows=True,
        maxbound=1,
        stress_period_data=[((0, 0, 0), 65.0)],
    )
    ghb_package = ModflowGwfghb(
        model,
        print_input=True,
        print_flows=True,
        maxbound=1,
        stress_period_data=[((0, 0, 9), 125.0, 60.0)],
    )

    rch_package = ModflowGwfrcha(
        model,
        print_input=True,
        print_flows=True,
        recharge="TIMEARRAYSERIES rcharray",
    )

    rch_array = np.zeros((1, 10))
    rch_array[0, 3] = 0.02
    rch_array[0, 6] = 0.1

    tas = ModflowUtltas(
        rch_package,
        filename="np002_mod.rch.tas",
        tas_array={0.0: rch_array, 6.0: rch_array, 12.0: rch_array},
        time_series_namerecord="rcharray",
        interpolation_methodrecord="linear",
    )

    # write simulation to new location
    sim.write_simulation()
    assert os.path.isfile(top_data_file)

    # run simulation
    sim.run_simulation()

    cell_list = [(0, 0, 0), (0, 0, 3), (0, 0, 4), (0, 0, 9)]
    out_file = function_tmpdir / "inspect_test_np002.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    sim2 = MFSimulation.load(sim_ws=ws)
    model_ = sim2.get_model(model_name)
    npf_package = model_.get_package("npf")
    k = npf_package.k.array

    # compare output to expected results
    head_new = os.path.join(ws, "np002_mod.hds")
    outfile = os.path.join(ws, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # verify external text file was written correctly
    ext_file_path = os.path.join(ws, "initial_heads.txt")
    fd = open(ext_file_path, "r")
    line = fd.readline()
    line_array = line.split()
    assert len(ic_vals) == len(line_array)
    for index in range(0, len(ic_vals)):
        assert ic_vals[index] == float(line_array[index])
    fd.close()

    # clean up
    sim.delete_output_files()

    # test error checking
    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.00000001, sy=0.6
    )
    chd_package = ModflowGwfchd(
        model,
        pname="chd_2",
        print_input=True,
        print_flows=True,
        maxbound=1,
        stress_period_data=[((0, 0, 0), np.nan)],
    )
    chk = sim.check()
    summary = ".".join(chk[0].summary_array.desc)
    assert (
        "sto package: specific storage values below "
        "checker threshold of 1e-06" in summary
    )
    assert (
        "sto package: specific yield values above "
        "checker threshold of 0.5" in summary
    )
    assert "Not a number" in summary
    model.remove_package("chd_2")
    # check case where aux variable defined and stress period data has empty
    # stress period
    model.remove_package("ghb")
    ghb_package = ModflowGwfghb(
        model,
        print_input=True,
        print_flows=True,
        maxbound=1,
        stress_period_data={0: [((0, 0, 9), 125.0, 60.0, 0.0)], 1: [()]},
        auxiliary=["CONCENTRATION"],
    )
    # add adaptive time stepping
    period_data = [
        (0, 3.0, 1.0e-5, 6.0, 2.0, 5.0),
        (1, 3.0, 1.0e-5, 6.0, 2.0, 5.0),
    ]
    ats = tdis_package.ats.initialize(maxats=2, perioddata=period_data)

    sim.write_simulation()
    sim.run_simulation()
    sim2 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=ws,
    )
    md2 = sim2.get_model()
    ghb2 = md2.get_package("ghb")
    spd2 = ghb2.stress_period_data.get_data(1)
    assert len(spd2) == 0

    # test paths
    sim_path_test = Path(ws) / "sim_path"
    sim.set_sim_path(sim_path_test)
    model.set_model_relative_path("model")
    # make external data folder path relative to simulation folder
    sim_data = sim_path_test / "data"
    sim.set_all_data_external(external_data_folder=sim_data)
    sim.write_simulation()
    # test
    assert Path(sim_data, "np002_mod.dis_botm.txt").exists()

    # run
    sim.run_simulation()


@requires_exe("mf6")
@pytest.mark.regression
def test021_twri(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test021_twri"
    model_name = "twri"
    data_folder = example_data_path / "mf6" / "create_tests" / test_ex_name
    ws = function_tmpdir / "ws"

    # copy example data into working directory
    shutil.copytree(data_folder, ws)

    expected_output_folder = os.path.join(data_folder, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "twri.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=data_folder,
    )
    sim.set_sim_path(function_tmpdir)
    tdis_rc = [(86400.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="SECONDS", nper=1, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.0001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=0.0001,
        rcloserecord=0.001,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    # build top binary data
    text = "TOP"
    fname = "top.bin"
    nrow = 15
    ncol = 15
    data_folder = os.path.join(sim.simulation_data.mfpath.get_sim_path(), fname)
    f = open(data_folder, "wb")
    header = flopy.utils.BinaryHeader.create(
        bintype="HEAD",
        precision="double",
        text=text,
        nrow=nrow,
        ncol=ncol,
        ilay=1,
        pertim=1.0,
        totim=1.0,
        kstp=1,
        kper=1,
    )
    flopy.utils.Util2d.write_bin(
        (nrow, ncol),
        f,
        np.full((nrow, ncol), 200.0, dtype=np.float64),
        header_data=header,
    )
    f.close()
    top = {"factor": 1.0, "filename": fname, "data": None, "binary": True, "iprn": 1}

    dis_package = ModflowGwfdis(
        model,
        nlay=3,
        nrow=15,
        ncol=15,
        delr=5000.0,
        delc=5000.0,
        top=top,
        botm=[-200, -300, -450],
        filename=f"{model_name}.dis",
    )
    strt = [
        {"filename": "strt.txt", "factor": 1.0, "data": 0.0},
        {"filename": "strt2.bin", "factor": 1.0, "data": 1.0, "binary": "True"},
        2.0,
    ]
    ic_package = ModflowGwfic(model, strt=strt, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        perched=True,
        cvoptions="dewatered",
        icelltype=[1, 0, 0],
        k=[0.001, 0.0001, 0.0002],
        k33=0.00000002,
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="twri.cbc",
        head_filerecord="twri.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL")],
    )

    # build stress_period_data for chd package
    stress_period_data = []
    for layer in range(0, 2):
        for row in range(0, 15):
            stress_period_data.append(((layer, row, 0), 0.0))
    chd_package = ModflowGwfchd(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=100,
        stress_period_data=stress_period_data,
    )

    # build stress_period_data for drn package
    conc = np.ones((15, 15), dtype=float) * 35.0
    auxdata = {0: [6, conc]}

    stress_period_data = []
    drn_heads = [0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0]
    for col, head in zip(range(1, 10), drn_heads):
        stress_period_data.append(((0, 7, col), head, 1.0, f"name_{col}"))
    drn_package = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=9,
        boundnames=True,
        stress_period_data=stress_period_data,
    )
    rch_package = ModflowGwfrcha(
        model,
        readasarrays=True,
        fixed_cell=True,
        recharge="TIMEARRAYSERIES rcharray",
        auxiliary=[("iface", "conc")],
        aux=auxdata,
    )
    rch_package.tas.initialize(
        filename="twri.rch.tas",
        tas_array={
            0.0: 0.00000003 * np.ones((15, 15)),
            86400.0: 0.00000003 * np.ones((15, 15)),
        },
        time_series_namerecord="rcharray",
        interpolation_methodrecord="linear",
    )

    aux = rch_package.aux.get_data()

    stress_period_data = []
    layers = [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rows = [4, 3, 5, 8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12]
    cols = [10, 5, 11, 7, 9, 11, 13, 7, 9, 11, 13, 7, 9, 11, 13]
    for layer, row, col in zip(layers, rows, cols):
        stress_period_data.append(((layer, row, col), -5.0))
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=15,
        stress_period_data=stress_period_data,
    )

    # change folder to save simulation
    sim.set_sim_path(ws)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    sim2 = MFSimulation.load(sim_ws=ws)
    model2 = sim2.get_model()
    ic2 = model2.get_package("ic")
    strt2 = ic2.strt.get_data()
    drn2 = model2.get_package("drn")
    drn_spd = drn2.stress_period_data.get_data()
    assert strt2[0, 0, 0] == 0.0
    assert strt2[1, 0, 0] == 1.0
    assert strt2[2, 0, 0] == 2.0
    assert drn_spd[0][1][3] == "name_2"

    # compare output to expected results
    head_new = os.path.join(ws, "twri.hds")
    outfile = os.path.join(ws, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test005_create_tests_advgw_tidal(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test005_advgw_tidal"
    model_name = "AdvGW_tidal"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file = expected_output_folder / "AdvGW_tidal.hds"

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=pth)
    # test tdis package deletion
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(2.0, 2, 1.0)]
    )
    sim.remove_package(tdis_package.package_type)

    tdis_rc = [
        (1.0, 1, 1.0),
        (10.0, 120, 1.0),
        (10.0, 120, 1.0),
        (10.0, 120, 1.0),
    ]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=4, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=0.0001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=0.0001,
        rcloserecord=0.001,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    bot_data = [-100 for x in range(150)]
    dis_package = ModflowGwfdis(
        model,
        nlay=3,
        nrow=15,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=50.0,
        botm=[5.0, -10.0, {"factor": 1.0, "data": bot_data}],
        filename=f"{model_name}.dis",
    )
    ic_package = ModflowGwfic(model, strt=50.0, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        icelltype=[1, 0, 0],
        k=[5.0, 0.1, 4.0],
        k33=[0.5, 0.005, 0.1],
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="AdvGW_tidal.cbc",
        head_filerecord="AdvGW_tidal.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "FIRST"), ("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    # test empty
    sy_template = ModflowGwfsto.sy.empty(model, True)
    for layer in range(0, 3):
        sy_template[layer]["data"] = 0.2
    layer_storage_types = [
        DataStorageType.internal_array,
        DataStorageType.internal_constant,
        DataStorageType.internal_array,
    ]
    ss_template = ModflowGwfsto.ss.empty(model, True, layer_storage_types, 0.000001)
    sto_package = ModflowGwfsto(
        model,
        save_flows=True,
        iconvert=1,
        ss=ss_template,
        sy=sy_template,
        steady_state={0: True},
        transient={1: True},
    )

    # wel, evt, ghb, obs, riv, rch, ts
    # well package
    # test empty with aux vars, bound names, and time series
    period_two = ModflowGwfwel.stress_period_data.empty(
        model,
        maxbound=3,
        aux_vars=["var1", "var2", "var3"],
        boundnames=True,
        timeseries=True,
    )
    period_two[0][0] = ((0, 11, 2), -50.0, -1, -2, -3, None)
    period_two[0][1] = ((2, 4, 7), "well_1_rate", 1, 2, 3, "well_1")
    period_two[0][2] = ((2, 3, 2), "well_2_rate", 4, 5, 6, "well_2")
    period_three = ModflowGwfwel.stress_period_data.empty(
        model,
        maxbound=2,
        aux_vars=["var1", "var2", "var3"],
        boundnames=True,
        timeseries=True,
    )
    period_three[0][0] = ((2, 3, 2), "well_2_rate", 1, 2, 3, "well_2")
    period_three[0][1] = ((2, 4, 7), "well_1_rate", 4, 5, 6, "well_1")
    period_four = ModflowGwfwel.stress_period_data.empty(
        model,
        maxbound=5,
        aux_vars=["var1", "var2", "var3"],
        boundnames=True,
        timeseries=True,
    )
    period_four[0][0] = ((2, 4, 7), "well_1_rate", 1, 2, 3, "well_1")
    period_four[0][1] = ((2, 3, 2), "well_2_rate", 4, 5, 6, "well_2")
    period_four[0][2] = ((0, 11, 2), -10.0, 7, 8, 9, None)
    period_four[0][3] = ((0, 2, 4), -20.0, 17, 18, 19, None)
    period_four[0][4] = ((0, 13, 5), -40.0, 27, 28, 29, None)
    stress_period_data = {}
    stress_period_data[1] = period_two[0]
    stress_period_data[2] = period_three[0]
    stress_period_data[3] = period_four[0]
    # well ts package
    timeseries = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, -200.0, 0.0, -100.0),
        (11.0, -1800.0, -500.0, -200.0),
        (21.0, -200.0, -400.0, -300.0),
        (31.0, 0.0, -600.0, -400.0),
    ]
    ts_dict = {
        "filename": os.path.join("well-rates", "well-rates.ts"),
        "timeseries": timeseries,
        "time_series_namerecord": [("well_1_rate", "well_2_rate", "well_3_rate")],
        "interpolation_methodrecord": [("stepwise", "stepwise", "stepwise")],
    }
    # test removing package with child packages
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        auxiliary=[("var1", "var2", "var3")],
        maxbound=5,
        stress_period_data=stress_period_data,
        boundnames=True,
        save_flows=True,
        timeseries=ts_dict,
    )
    wel_package.remove()
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        auxiliary=[("var1", "var2", "var3")],
        maxbound=5,
        stress_period_data=stress_period_data,
        boundnames=True,
        save_flows=True,
        timeseries=ts_dict,
    )

    # test nseg = 1
    evt_period = ModflowGwfevt.stress_period_data.empty(model, 150, nseg=1)
    for col in range(0, 10):
        for row in range(0, 15):
            evt_period[0][col * 15 + row] = (
                (0, row, col),
                50.0,
                0.0004,
                10.0,
                None,
            )
    evt_package_test = ModflowGwfevt(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=150,
        nseg=1,
        stress_period_data=evt_period,
    )
    evt_package_test.remove()

    # test empty
    evt_period = ModflowGwfevt.stress_period_data.empty(model, 150, nseg=3)
    for col in range(0, 10):
        for row in range(0, 15):
            evt_period[0][col * 15 + row] = (
                (0, row, col),
                50.0,
                0.0004,
                10.0,
                0.2,
                0.5,
                0.3,
                0.1,
                None,
            )
    evt_package = ModflowGwfevt(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=150,
        nseg=3,
        stress_period_data=evt_period,
    )

    # build ghb
    ghb_period = {}
    ghb_period_array = []
    for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
        for row in range(0, 15):
            ghb_period_array.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
    ghb_period[0] = ghb_period_array

    # build ts ghb
    ts_recarray = []
    fd = open(os.path.join(pth, "tides.txt"), "r")
    for line in fd:
        line_list = line.strip().split(",")
        ts_recarray.append((float(line_list[0]), float(line_list[1])))
    ts_package_dict = {
        "filename": "tides.ts",
        "timeseries": ts_recarray,
        "time_series_namerecord": "tides",
        "interpolation_methodrecord": "linear",
    }

    obs_dict = {
        ("ghb_obs.csv", "binary"): [
            ("ghb- 2-6-10", "GHB", (1, 5, 9)),
            ("ghb-3-6-10", "GHB", (2, 5, 9)),
        ],
        "ghb_flows.csv": [
            ("Estuary2", "GHB", "Estuary-L2"),
            ("Estuary3", "GHB", "Estuary-L3"),
        ],
        "filename": "AdvGW_tidal.ghb.obs",
        "digits": 10,
        "print_input": True,
    }

    ghb_package = ModflowGwfghb(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        timeseries=ts_package_dict,
        observations=obs_dict,
        maxbound=30,
        stress_period_data=ghb_period,
    )

    riv_period = {}
    riv_period_array = [
        ((0, 2, 0), "river_stage_1", 1001.0, 35.9, None),
        ((0, 3, 1), "river_stage_1", 1002.0, 35.8, None),
        ((0, 4, 2), "river_stage_1", 1003.0, 35.7, None),
        ((0, 4, 3), "river_stage_1", 1004.0, 35.6, None),
        ((0, 5, 4), "river_stage_1", 1005.0, 35.5, None),
        ((0, 5, 5), "river_stage_1", 1006.0, 35.4, "riv1_c6"),
        ((0, 5, 6), "river_stage_1", 1007.0, 35.3, "riv1_c7"),
        ((0, 4, 7), "river_stage_1", 1008.0, 35.2, None),
        ((0, 4, 8), "river_stage_1", 1009.0, 35.1, None),
        ((0, 4, 9), "river_stage_1", 1010.0, 35.0, None),
        ((0, 9, 0), "river_stage_2", 1001.0, 36.9, "riv2_upper"),
        ((0, 8, 1), "river_stage_2", 1002.0, 36.8, "riv2_upper"),
        ((0, 7, 2), "river_stage_2", 1003.0, 36.7, "riv2_upper"),
        ((0, 6, 3), "river_stage_2", 1004.0, 36.6, None),
        ((0, 6, 4), "river_stage_2", 1005.0, 36.5, None),
        ((0, 5, 5), "river_stage_2", 1006.0, 36.4, "riv2_c6"),
        ((0, 5, 6), "river_stage_2", 1007.0, 36.3, "riv2_c7"),
        ((0, 6, 7), "river_stage_2", 1008.0, 36.2, None),
        ((0, 6, 8), "river_stage_2", 1009.0, 36.1),
        ((0, 6, 9), "river_stage_2", 1010.0, 36.0),
    ]

    riv_period[0] = riv_period_array
    # riv time series
    ts_data = [
        (0.0, 40.0, 41.0),
        (1.0, 41.0, 41.5),
        (2.0, 43.0, 42.0),
        (3.0, 45.0, 42.8),
        (4.0, 44.0, 43.0),
        (6.0, 43.0, 43.1),
        (9.0, 42.0, 42.4),
        (11.0, 41.0, 41.5),
        (31.0, 40.0, 41.0),
    ]
    ts_dict = {
        "filename": "river_stages.ts",
        "timeseries": ts_data,
        "time_series_namerecord": [("river_stage_1", "river_stage_2")],
        "interpolation_methodrecord": [("linear", "stepwise")],
    }
    # riv obs
    obs_dict = {
        "riv_obs.csv": [
            ("rv1-3-1", "RIV", (0, 2, 0)),
            ("rv1-4-2", "RIV", (0, 3, 1)),
            ("rv1-5-3", "RIV", (0, 4, 2)),
            ("rv1-5-4", "RIV", (0, 4, 3)),
            ("rv1-6-5", "RIV", (0, 5, 4)),
            ("rv1-c6", "RIV", "riv1_c6"),
            ("rv1-c7", "RIV", "riv1_c7"),
            ("rv2-upper", "RIV", "riv2_upper"),
            ("rv-2-7-4", "RIV", (0, 6, 3)),
            ("rv2-8-5", "RIV", (0, 6, 4)),
            ("rv-2-9-6", "RIV", (0, 5, 5)),
        ],
        "riv_flowsA.csv": [
            ("riv1-3-1", "RIV", (0, 2, 0)),
            ("riv1-4-2", "RIV", (0, 3, 1)),
            ("riv1-5-3", "RIV", (0, 4, 2)),
        ],
        "riv_flowsB.csv": [
            ("riv2-10-1", "RIV", (0, 9, 0)),
            ("riv-2-9-2", "RIV", (0, 8, 1)),
            ("riv2-8-3", "RIV", (0, 7, 2)),
        ],
        "filename": "AdvGW_tidal.riv.obs",
        "digits": 10,
        "print_input": True,
    }

    riv_package = ModflowGwfriv(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        timeseries=ts_dict,
        maxbound=20,
        stress_period_data=riv_period,
        observations=obs_dict,
    )

    rch1_period = {}
    rch1_period_array = []
    col_range = {0: 3, 1: 4, 2: 5}
    for row in range(0, 15):
        if row in col_range:
            col_max = col_range[row]
        else:
            col_max = 6
        for col in range(0, col_max):
            if (
                (row == 3 and col == 5)
                or (row == 2 and col == 4)
                or (row == 1 and col == 3)
                or (row == 0 and col == 2)
            ):
                mult = 0.5
            else:
                mult = 1.0
            if row == 0 and col == 0:
                bnd = "rch-1-1"
            elif row == 0 and col == 1:
                bnd = "rch-1-2"
            elif row == 1 and col == 2:
                bnd = "rch-2-3"
            else:
                bnd = None
            rch1_period_array.append(((0, row, col), "rch_1", mult, bnd))
    rch1_period[0] = rch1_period_array
    rch1_package = ModflowGwfrch(
        model,
        filename="AdvGW_tidal_1.rch",
        pname="rch_1",
        fixed_cell=True,
        auxiliary="MULTIPLIER",
        auxmultname="MULTIPLIER",
        print_input=True,
        print_flows=True,
        save_flows=True,
        boundnames=True,
        maxbound=84,
        stress_period_data=rch1_period,
    )
    ts_data = [
        (0.0, 0.0015),
        (1.0, 0.0010),
        (11.0, 0.0015),
        (21.0, 0.0025),
        (31.0, 0.0015),
    ]
    rch1_package.ts.initialize(
        timeseries=ts_data,
        filename="recharge_rates_1.ts",
        time_series_namerecord="rch_1",
        interpolation_methodrecord="stepwise",
    )

    rch2_period = {}
    rch2_period_array = [
        ((0, 0, 2), "rch_2", 0.5),
        ((0, 0, 3), "rch_2", 1.0),
        ((0, 0, 4), "rch_2", 1.0),
        ((0, 0, 5), "rch_2", 1.0),
        ((0, 0, 6), "rch_2", 1.0),
        ((0, 0, 7), "rch_2", 1.0),
        ((0, 0, 8), "rch_2", 1.0),
        ((0, 0, 9), "rch_2", 0.5),
        ((0, 1, 3), "rch_2", 0.5),
        ((0, 1, 4), "rch_2", 1.0),
        ((0, 1, 5), "rch_2", 1.0),
        ((0, 1, 6), "rch_2", 1.0),
        ((0, 1, 7), "rch_2", 1.0),
        ((0, 1, 8), "rch_2", 0.5),
        ((0, 2, 4), "rch_2", 0.5),
        ((0, 2, 5), "rch_2", 1.0),
        ((0, 2, 6), "rch_2", 1.0),
        ((0, 2, 7), "rch_2", 0.5),
        ((0, 3, 5), "rch_2", 0.5),
        ((0, 3, 6), "rch_2", 0.5),
    ]
    rch2_period[0] = rch2_period_array
    rch2_package = ModflowGwfrch(
        model,
        filename="AdvGW_tidal_2.rch",
        pname="rch_2",
        fixed_cell=True,
        auxiliary="MULTIPLIER",
        auxmultname="MULTIPLIER",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=20,
        stress_period_data=rch2_period,
    )
    ts_data = [
        (0.0, 0.0016),
        (1.0, 0.0018),
        (11.0, 0.0019),
        (21.0, 0.0016),
        (31.0, 0.0018),
    ]
    rch2_package.ts.initialize(
        timeseries=ts_data,
        filename="recharge_rates_2.ts",
        time_series_namerecord="rch_2",
        interpolation_methodrecord="linear",
    )

    rch3_period = {}
    rch3_period_array = []
    col_range = {0: 9, 1: 8, 2: 7}
    for row in range(0, 15):
        if row in col_range:
            col_min = col_range[row]
        else:
            col_min = 6
        for col in range(col_min, 10):
            if (
                (row == 0 and col == 9)
                or (row == 1 and col == 8)
                or (row == 2 and col == 7)
                or (row == 3 and col == 6)
            ):
                mult = 0.5
            else:
                mult = 1.0
            rch3_period_array.append(((0, row, col), "rch_3", mult))
    rch3_period[0] = rch3_period_array
    rch3_package = ModflowGwfrch(
        model,
        filename="AdvGW_tidal_3.rch",
        pname="rch_3",
        fixed_cell=True,
        auxiliary="MULTIPLIER",
        auxmultname="MULTIPLIER",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=54,
        stress_period_data=rch3_period,
    )
    ts_data = [
        (0.0, 0.0017),
        (1.0, 0.0020),
        (11.0, 0.0017),
        (21.0, 0.0018),
        (31.0, 0.0020),
    ]
    rch3_package.ts.initialize(
        timeseries=ts_data,
        filename="recharge_rates_3.ts",
        time_series_namerecord="rch_3",
        interpolation_methodrecord="linear",
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # test time series data file with relative path to simulation path
    ts_path = function_tmpdir / "well-rates" / "well-rates.ts"
    assert os.path.exists(ts_path)

    # run simulation
    sim.run_simulation()

    # inspect cells
    cell_list = [(2, 3, 2), (0, 4, 2), (0, 2, 4), (0, 5, 5), (0, 9, 9)]
    out_file = function_tmpdir / "inspect_AdvGW_tidal.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = function_tmpdir / "AdvGW_tidal.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # test rename all
    model.rename_all_packages("new_name")
    assert model.name_file.filename == "new_name.nam"
    package_type_dict = {}
    for package in model.packagelist:
        if package.package_type not in package_type_dict:
            filename = os.path.split(package.filename)[1]
            assert filename == f"new_name.{package.package_type}"
            package_type_dict[package.package_type] = 1
    sim.write_simulation()
    name_file = function_tmpdir / "new_name.nam"
    assert os.path.exists(name_file)
    dis_file = function_tmpdir / "new_name.dis"
    assert os.path.exists(dis_file)
    # test time series data file with relative path to simulation path
    ts_path = function_tmpdir / "well-rates" / "new_name.ts"
    assert os.path.exists(ts_path)

    sim.rename_all_packages("all_files_same_name")
    package_type_dict = {}
    for package in model.packagelist:
        if package.package_type not in package_type_dict:
            filename = os.path.split(package.filename)[1]
            assert filename == f"all_files_same_name.{package.package_type}"
            package_type_dict[package.package_type] = 1
    assert sim._tdis_file.filename == "all_files_same_name.tdis"
    for ims_file in sim._solution_files.values():
        assert ims_file.filename == "all_files_same_name.ims"
    sim.write_simulation()
    name_file = function_tmpdir / "all_files_same_name.nam"
    assert os.path.exists(name_file)
    dis_file = function_tmpdir / "all_files_same_name.dis"
    assert os.path.exists(dis_file)
    tdis_file = function_tmpdir / "all_files_same_name.tdis"
    assert os.path.exists(tdis_file)
    # test time series data file with relative path to simulation path
    ts_path = function_tmpdir / "well-rates" / "all_files_same_name.ts"
    assert os.path.exists(ts_path)

    # load simulation
    sim_load = MFSimulation.load(
        sim.name,
        "mf6",
        "mf6",
        sim.simulation_data.mfpath.get_sim_path(),
        verbosity_level=0,
    )
    model = sim_load.get_model()
    # confirm ghb obs data has two blocks with correct file names
    ghb = model.get_package("ghb")
    obs = ghb.obs
    obs_data = obs.continuous.get_data()
    found_flows = False
    found_obs = False
    for key, value in obs_data.items():
        if key.lower() == "ghb_flows.csv":
            # there should be only one
            assert not found_flows
            found_flows = True
        if key.lower() == "ghb_obs.csv":
            # there should be only one
            assert not found_obs
            found_obs = True
            assert value[0][0] == "ghb- 2-6-10"
    assert found_flows and found_obs

    # check model.time steady state and transient
    sto_package = model.get_package("sto")
    sto_package.steady_state.set_data({0: True, 1: False, 2: False, 3: False})
    sto_package.transient.set_data({0: False, 1: True, 2: True, 3: True})
    flopy.mf6.ModflowGwfdrn(model, pname="storm")
    ss = model.modeltime.steady_state
    assert ss[0]
    assert not ss[1]
    assert not ss[2]
    assert not ss[3]

    # clean up
    sim.delete_output_files()

    # check packages
    chk = sim.check()
    summary = ".".join(chk[0].summary_array.desc)
    assert summary == ""


@requires_exe("mf6")
@pytest.mark.regression
def test004_create_tests_bcfss(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test004_bcfss"
    model_name = "bcf2ss"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "bcf2ss.hds")

    # create simulation
    sim = MFSimulation(sim_name=model_name, version="mf6", exe_name="mf6", sim_ws=pth)
    tdis_rc = [(1.0, 1, 1.0), (1.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=2, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="ALL",
        csv_output_filerecord="bcf2ss.ims.csv",
        complexity="SIMPLE",
        outer_dvclose=0.000001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=0.000001,
        rcloserecord=0.001,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    dis_package = ModflowGwfdis(
        model,
        nlay=2,
        nrow=10,
        ncol=15,
        delr=500.0,
        delc=500.0,
        top=150.0,
        botm=[50.0, -50.0],
        filename=f"{model_name}.dis",
    )
    ic_package = ModflowGwfic(model, strt=0.0, filename=f"{model_name}.ic")
    wetdry_data = []
    for row in range(0, 10):
        if row == 2 or row == 7:
            wetdry_data += [2.0, 2.0, 2.0, -2.0, 2.0, 2.0, 2.0, 2.0]
        else:
            wetdry_data += [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        wetdry_data += [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
    for row in range(0, 10):
        for col in range(0, 15):
            wetdry_data.append(0.0)
    npf_package = ModflowGwfnpf(
        model,
        rewet_record=[("WETFCT", 1.0, "IWETIT", 1, "IHDWET", 0)],
        save_flows=True,
        icelltype=[1, 0],
        wetdry=wetdry_data,
        k=[10.0, 5.0],
        k33=0.1,
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="bcf2ss.cbb",
        head_filerecord="bcf2ss.hds",
        headprintrecord=[("COLUMNS", 15, "WIDTH", 12, "DIGITS", 2, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    # include a bad aux value
    aux = {0: [[50.0], [1.3]], 1: [[200.0], np.nan]}
    # aux = {0: [[100.0], [2.3]]}
    rch_package = ModflowGwfrcha(
        model,
        readasarrays=True,
        save_flows=True,
        auxiliary=[("var1", "var2")],
        recharge={0: 0.004, 1: []},
        aux=aux,
    )  # *** test if aux works ***
    chk = rch_package.check()
    summary = ".".join(chk.summary_array.desc)
    assert summary == "One or more nan values were found in auxiliary data."

    # fix aux values
    aux = {0: [[50.0], [1.3]], 1: [[200.0], [1.5]]}
    rch_package.aux = aux
    # aux tests
    aux_out = rch_package.aux.get_data()
    assert aux_out[0][0][0, 0] == 50.0
    assert aux_out[0][1][0, 0] == 1.3
    assert aux_out[1][0][0, 0] == 200.0
    assert aux_out[1][1][0, 0] == 1.5
    # write test
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    # update recharge
    recharge = {0: 0.004, 1: 0.004}

    riv_period = {}
    riv_period_array = []
    aux_vals = [1.0, 5.0, 4.0, 8.0, 3.0, "bad value", 5.5, 6.3, 8.1, 18.3]
    for row in range(0, 10):
        riv_period_array.append(((1, row, 14), 0.0, 10000.0, -5.0, aux_vals[row], 10.0))
    riv_period[0] = riv_period_array
    riv_package = ModflowGwfriv(
        model,
        auxiliary=[("var1", "var2")],
        save_flows="bcf2ss.cbb",
        maxbound=10,
        stress_period_data=riv_period,
    )
    chk = riv_package.check()
    summary = ".".join(chk.summary_array.desc)
    assert summary == "Invalid non-numeric value 'bad value' in auxiliary data."
    # test with boundnames
    riv_package.boundnames = True
    riv_period_array = []
    for row in range(0, 10):
        riv_period_array.append(((1, row, 14), 0.0, 10000.0, -5.0, aux_vals[row], 10.0))
    riv_period[0] = riv_period_array
    riv_package.stress_period_data = riv_period
    chk = riv_package.check()
    summary = ".".join(chk.summary_array.desc)
    assert summary == "Invalid non-numeric value 'bad value' in auxiliary data."

    # fix aux variable
    riv_package.boundnames = False
    riv_period = {}
    riv_period_array = []
    aux_vals = [1.0, 5.0, 4.0, 8.0, 3.0, 5.0, 5.5, 6.3, 8.1, 18.3]
    for row in range(0, 10):
        riv_period_array.append(((1, row, 14), 0.0, 10000.0, -5.0, aux_vals[row], 10.0))
    riv_period[0] = riv_period_array
    riv_package.stress_period_data = riv_period
    # check again
    chk = riv_package.check()
    assert len(chk.summary_array) == 0

    wel_period = {}
    stress_period_data = [
        ((1, 2, 3), -35000.0, 1, 2, 3),
        ((1, 7, 3), -35000.0, 4, 5, 6),
    ]
    wel_period[1] = stress_period_data
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary=[("var1", "var2", "var3")],
        maxbound=2,
        stress_period_data=wel_period,
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_new = function_tmpdir / "bcf2ss.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.regression
def test035_create_tests_fhb(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test035_fhb"
    model_name = "fhb2015"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "fhb2015_fhb.hds")

    # create simulation
    sim = MFSimulation(sim_name=model_name, version="mf6", exe_name="mf6", sim_ws=pth)
    tdis_rc = [(400.0, 10, 1.0), (200.0, 4, 1.0), (400.0, 6, 1.1)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=3, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=0.001,
        outer_maximum=120,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=0.0001,
        rcloserecord=0.1,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.001,
        number_orthogonalizations=2,
    )
    sim.register_ims_package(ims_package, [model.name])
    dis_package = ModflowGwfdis(
        model,
        length_units="UNDEFINED",
        nlay=1,
        nrow=3,
        ncol=10,
        delr=1000.0,
        delc=1000.0,
        top=50.0,
        botm=-200.0,
        filename=f"{model_name}.dis",
    )
    ic_package = ModflowGwfic(model, strt=0.0, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(model, perched=True, icelltype=0, k=20.0, k33=1.0)
    oc_package = ModflowGwfoc(
        model,
        head_filerecord="fhb2015_fhb.hds",
        headprintrecord=[("COLUMNS", 20, "WIDTH", 5, "DIGITS", 2, "FIXED")],
        saverecord={0: [("HEAD", "ALL")], 2: [("HEAD", "ALL")]},
        printrecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            2: [("HEAD", "ALL"), ("BUDGET", "ALL")],
        },
    )
    sto_package = ModflowGwfsto(
        model, storagecoefficient=True, iconvert=0, ss=0.01, sy=0.0
    )
    time = model.modeltime
    assert not (time.steady_state[0] or time.steady_state[1] or time.steady_state[2])
    wel_period = {0: [((0, 1, 0), "flow")]}
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        stress_period_data=wel_period,
    )
    well_ts = [
        (0.0, 2000.0),
        (307.0, 6000.0),
        (791.0, 5000.0),
        (1000.0, 9000.0),
    ]
    wel_package.ts.initialize(
        filename="fhb_flow.ts",
        timeseries=well_ts,
        time_series_namerecord="flow",
        interpolation_methodrecord="linear",
    )

    chd_period = {0: [((0, 0, 9), "head"), ((0, 1, 9), "head"), ((0, 2, 9), "head")]}
    chd_package = ModflowGwfchd(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=3,
        stress_period_data=chd_period,
    )
    chd_ts = [(0.0, 0.0), (307.0, 1.0), (791.0, 5.0), (1000.0, 2.0)]
    chd_package.ts.initialize(
        filename="fhb_head.ts",
        timeseries=chd_ts,
        time_series_namerecord="head",
        interpolation_methodrecord="linearend",
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_new = function_tmpdir / "fhb2015_fhb.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@requires_pkg("pyshp", name_map={"pyshp": "shapefile"})
@pytest.mark.regression
def test006_create_tests_gwf3_disv(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test006_gwf3_disv"
    model_name = "flow"
    data_path = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = data_path / "expected_output"
    expected_head_file = expected_output_folder / "flow.hds"

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=data_path,
    )
    sim.set_sim_path(function_tmpdir)
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.00000001,
        outer_maximum=1000,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_dvclose=0.00000001,
        rcloserecord=0.01,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    vertices = testutils.read_vertices(os.path.join(data_path, "vertices.txt"))
    c2drecarray = testutils.read_cell2d(os.path.join(data_path, "cell2d.txt"))
    disv_package = ModflowGwfdisv(
        model,
        ncpl=121,
        nlay=1,
        nvert=148,
        top=0.0,
        botm=-100.0,
        idomain=1,
        vertices=vertices,
        cell2d=c2drecarray,
        filename=f"{model_name}.disv",
    )
    strt_list = [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    ic_package = ModflowGwfic(model, strt=strt_list, filename=f"{model_name}.ic")
    k = {"filename": "k.bin", "factor": 1.0, "data": 1.0, "binary": "True"}
    npf_package = ModflowGwfnpf(model, save_flows=True, icelltype=0, k=k, k33=1.0)
    k_data = npf_package.k.get_data()
    assert k_data[0, 0] == 1.0

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="flow.cbc",
        head_filerecord="flow.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # build stress_period_data for chd package
    set_1 = [0, 7, 14, 18, 22, 26, 33]
    set_2 = [6, 13, 17, 21, 25, 32, 39]
    stress_period_data = []
    for value in set_1:
        stress_period_data.append(((0, value), 1.0))
    for value in set_2:
        stress_period_data.append(((0, value), 0.0))
    chd_package = ModflowGwfchd(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=14,
        stress_period_data=stress_period_data,
    )

    period_rch = {}
    rch_array = []
    for val in range(0, 10):
        rch_array.append(((0, val), 0.0))
    period_rch[0] = rch_array
    rch_package = ModflowGwfrch(
        model, fixed_cell=True, maxbound=10, stress_period_data=period_rch
    )

    gncrecarray = [
        (0, 9, 0, 40, 0, 8, 0.333333333333),
        (0, 9, 0, 42, 0, 10, 0.333333333333),
        (0, 10, 0, 43, 0, 9, 0.333333333333),
        (0, 10, 0, 45, 0, 11, 0.333333333333),
        (0, 11, 0, 46, 0, 10, 0.333333333333),
        (0, 11, 0, 48, 0, 12, 0.333333333333),
        (0, 15, 0, 40, 0, 8, 0.333333333333),
        (0, 15, 0, 58, 0, 19, 0.333333333333),
        (0, 16, 0, 48, 0, 12, 0.333333333333),
        (0, 16, 0, 66, 0, 20, 0.333333333333),
        (0, 19, 0, 67, 0, 15, 0.333333333333),
        (0, 19, 0, 85, 0, 23, 0.333333333333),
        (0, 20, 0, 75, 0, 16, 0.333333333333),
        (0, 20, 0, 93, 0, 24, 0.333333333333),
        (0, 23, 0, 94, 0, 19, 0.333333333333),
        (0, 23, 0, 112, 0, 27, 0.333333333333),
        (0, 24, 0, 102, 0, 20, 0.333333333333),
        (0, 24, 0, 120, 0, 31, 0.333333333333),
        (0, 28, 0, 112, 0, 27, 0.333333333333),
        (0, 28, 0, 114, 0, 29, 0.333333333333),
        (0, 29, 0, 115, 0, 28, 0.333333333333),
        (0, 29, 0, 117, 0, 30, 0.333333333333),
        (0, 30, 0, 118, 0, 29, 0.333333333333),
        (0, 30, 0, 120, 0, 31, 0.333333333333),
    ]
    gnc_package = ModflowGwfgnc(
        model,
        print_input=True,
        print_flows=True,
        numgnc=24,
        numalphaj=1,
        gncdata=gncrecarray,
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # inspect cells
    cell_list = [(0, 0), (0, 7), (0, 17)]
    out_file = function_tmpdir / "inspect_test_gwf3_disv.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = function_tmpdir / "flow.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # export to netcdf - temporarily disabled
    # model.export(os.path.join(run_folder, "test006_gwf3.nc"))
    # export to shape file

    model.export(function_tmpdir / "test006_gwf3.shp")

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.regression
def test006_create_tests_2models_gnc(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test006_2models_gnc"
    model_name_1 = "model1"
    model_name_2 = "model2"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_1 = os.path.join(expected_output_folder, "model1.hds")
    expected_head_file_2 = os.path.join(expected_output_folder, "model2.hds")

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=pth)
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=tdis_rc)
    model_1 = ModflowGwf(
        sim,
        modelname=model_name_1,
        model_nam_file=f"{model_name_1}.nam",
    )
    model_2 = ModflowGwf(
        sim,
        modelname=model_name_2,
        model_nam_file=f"{model_name_2}.nam",
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.00000001,
        outer_maximum=1000,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_dvclose=0.00000001,
        rcloserecord=0.01,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model_1.name, model_2.name])
    idom = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    dis_package_1 = ModflowGwfdis(
        model_1,
        length_units="METERS",
        nlay=1,
        nrow=7,
        ncol=7,
        idomain=idom,
        delr=100.0,
        delc=100.0,
        top=0.0,
        botm=-100.0,
        filename=f"{model_name_1}.dis",
    )
    dis_package_2 = ModflowGwfdis(
        model_2,
        length_units="METERS",
        nlay=1,
        nrow=9,
        ncol=9,
        delr=33.33,
        delc=33.33,
        top=0.0,
        botm=-100.0,
        filename=f"{model_name_2}.dis",
    )

    strt_list = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    ic_package_1 = ModflowGwfic(model_1, strt=strt_list, filename=f"{model_name_1}.ic")
    ic_package_2 = ModflowGwfic(model_2, strt=1.0, filename=f"{model_name_2}.ic")
    npf_package_1 = ModflowGwfnpf(
        model_1, save_flows=True, perched=True, icelltype=0, k=1.0, k33=1.0
    )
    npf_package_2 = ModflowGwfnpf(
        model_2, save_flows=True, perched=True, icelltype=0, k=1.0, k33=1.0
    )
    oc_package_1 = ModflowGwfoc(
        model_1,
        budget_filerecord="model1.cbc",
        head_filerecord="model1.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    oc_package_2 = ModflowGwfoc(
        model_2,
        budget_filerecord="model2.cbc",
        head_filerecord="model2.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # build periodrecarray for chd package
    set_1 = [0, 7, 14, 18, 22, 26, 33]
    set_2 = [6, 13, 17, 21, 25, 32, 39]
    stress_period_data = []
    for value in range(0, 7):
        stress_period_data.append(((0, value, 0), 1.0))
    for value in range(0, 7):
        stress_period_data.append(((0, value, 6), 0.0))
    chd_package = ModflowGwfchd(
        model_1,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=30,
        stress_period_data=stress_period_data,
    )

    exgrecarray = testutils.read_exchangedata(os.path.join(pth, "exg.txt"))

    # build obs dictionary
    gwf_obs = {
        ("gwfgwf_obs.csv"): [
            ("gwf-1-3-2_1-1-1", "flow-ja-face", (0, 2, 1), (0, 0, 0)),
            ("gwf-1-3-2_1-2-1", "flow-ja-face", (0, 2, 1), (0, 1, 0)),
        ]
    }

    # test exg delete
    newexgrecarray = exgrecarray[10:]
    gnc_path = os.path.join("gnc", "test006_2models_gnc.gnc")
    exg_package = ModflowGwfgwf(
        sim,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary="testaux",
        nexg=26,
        exchangedata=newexgrecarray,
        exgtype="gwf6-gwf6",
        exgmnamea=model_name_1,
        exgmnameb=model_name_2,
    )
    sim.remove_package(exg_package.package_type)

    exg_data = {"filename": "exg_data.txt", "data": exgrecarray, "binary": True}
    exg_package = ModflowGwfgwf(
        sim,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary="testaux",
        nexg=36,
        exchangedata=exg_data,
        exgtype="gwf6-gwf6",
        exgmnamea=model_name_1,
        exgmnameb=model_name_2,
        observations=gwf_obs,
    )

    gncrecarray = testutils.read_gncrecarray(os.path.join(pth, "gnc.txt"))
    # test gnc delete
    new_gncrecarray = gncrecarray[10:]
    gnc_package = exg_package.gnc.initialize(
        filename=gnc_path,
        print_input=True,
        print_flows=True,
        numgnc=36,
        numalphaj=1,
        gncdata=gncrecarray,
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)
    exg_package.exchangedata.set_record(exg_data)

    # write simulation to new location
    sim.write_simulation()

    # test gnc file was created in correct location
    gnc_full_path = function_tmpdir / gnc_path
    assert os.path.exists(gnc_full_path)

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_new = function_tmpdir / "model1.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_1],
        files2=[head_new],
        outfile=outfile,
    )

    # compare output to expected results
    head_new = function_tmpdir / "model2.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_2],
        files2=[head_new],
        outfile=outfile,
    )

    # test external file paths
    sim_path = function_tmpdir / "path_test"
    sim.set_sim_path(sim_path)
    model_1.set_model_relative_path("model1")
    model_2.set_model_relative_path("model2")
    sim.set_all_data_external(external_data_folder=function_tmpdir / "data")
    sim.write_simulation()
    ext_file_path_1 = function_tmpdir / "data" / "model1.dis_botm.txt"
    assert os.path.exists(ext_file_path_1)
    ext_file_path_2 = function_tmpdir / "data" / "model2.dis_botm.txt"
    assert os.path.exists(ext_file_path_2)
    # test gnc file was created in correct location
    gnc_full_path = os.path.join(sim_path, gnc_path)
    assert os.path.exists(gnc_full_path)

    sim.run_simulation()
    sim.delete_output_files()

    # test rename all packages
    rename_folder = function_tmpdir / "rename"
    sim.rename_all_packages("file_rename")
    sim.set_sim_path(rename_folder)
    sim.write_simulation()
    # test gnc file was created in correct location
    gnc_full_path = os.path.join(rename_folder, "gnc", "file_rename.gnc")
    assert os.path.exists(gnc_full_path)

    sim.run_simulation()
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test050_create_tests_circle_island(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test050_circle_island"
    model_name = "ci"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file = expected_output_folder / "ci.output.hds"

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=pth)
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=tdis_rc)
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.000001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_dvclose=0.000001,
        rcloserecord=0.000001,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.0,
    )
    sim.register_ims_package(ims_package, [model.name])
    vertices = testutils.read_vertices(os.path.join(pth, "vertices.txt"))
    c2drecarray = testutils.read_cell2d(os.path.join(pth, "cell2d.txt"))
    disv_package = ModflowGwfdisv(
        model,
        ncpl=5240,
        nlay=2,
        nvert=2778,
        top=0.0,
        botm=[-20.0, -40.0],
        idomain=1,
        vertices=vertices,
        cell2d=c2drecarray,
        filename=f"{model_name}.disv",
    )
    ic_package = ModflowGwfic(model, strt=0.0, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(model, save_flows=True, icelltype=0, k=10.0, k33=0.2)
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="ci.output.cbc",
        head_filerecord="ci.output.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    stress_period_data = testutils.read_ghbrecarray(os.path.join(pth, "ghb.txt"), 2)
    ghb_package = ModflowGwfghb(
        model, maxbound=3173, stress_period_data=stress_period_data
    )

    rch_data = ["OPEN/CLOSE", "rech.dat", "FACTOR", 1.0, "IPRN", 0]
    rch_package = ModflowGwfrcha(
        model, readasarrays=True, save_flows=True, recharge=rch_data
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_new = function_tmpdir / "ci.output.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.xfail(
    reason="possible python3.7/windows incompatibilities in testutils.read_std_array "
    "https://github.com/modflowpy/flopy/runs/7581629193?check_suite_focus=true#step:11:1753"
)
@pytest.mark.regression
def test028_create_tests_sfr(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test028_sfr"
    model_name = "test1tr"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file = expected_output_folder / "test1tr.hds"

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=pth)
    sim.name_file.continue_.set_data(True)
    tdis_rc = [(1577889000, 50, 1.1), (1577889000, 50, 1.1)]
    tdis_package = ModflowTdis(
        sim,
        time_units="SECONDS",
        nper=2,
        perioddata=tdis_rc,
        filename="simulation.tdis",
    )
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    model.name_file.save_flows.set_data(True)
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.00001,
        outer_maximum=100,
        under_relaxation="DBD",
        under_relaxation_theta=0.85,
        under_relaxation_kappa=0.0001,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.1,
        backtracking_number=0,
        backtracking_tolerance=1.1,
        backtracking_reduction_factor=0.7,
        backtracking_residual_limit=1.0,
        inner_dvclose=0.00001,
        rcloserecord=0.1,
        inner_maximum=100,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
        filename="model.ims",
    )
    sim.register_ims_package(ims_package, [model.name])
    top = testutils.read_std_array(os.path.join(pth, "top.txt"), "float")
    botm = testutils.read_std_array(os.path.join(pth, "botm.txt"), "float")
    idomain = testutils.read_std_array(os.path.join(pth, "idomain.txt"), "int")
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=15,
        ncol=10,
        delr=5000.0,
        delc=5000.0,
        top=top,
        botm=botm,
        idomain=idomain,
        filename=f"{model_name}.dis",
    )
    strt = testutils.read_std_array(os.path.join(pth, "strt.txt"), "float")
    strt_int = ["internal", "factor", 1.0, "iprn", 0, strt]
    ic_package = ModflowGwfic(model, strt=strt_int, filename=f"{model_name}.ic")

    k_vals = testutils.read_std_array(os.path.join(pth, "k.txt"), "float")
    k = ["internal", "factor", 3.000e-03, "iprn", 0, k_vals]
    npf_package = ModflowGwfnpf(model, icelltype=1, k=k, k33=1.0)
    npf_package.k.factor = 2.000e-04

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="test1tr.cbc",
        head_filerecord="test1tr.hds",
        saverecord={0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]},
        printrecord={0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]},
    )

    sy_vals = testutils.read_std_array(os.path.join(pth, "sy.txt"), "float")
    sy = {"factor": 0.2, "iprn": 0, "data": sy_vals}
    sto_package = ModflowGwfsto(model, iconvert=1, ss=1.0e-6, sy=sy)

    surf = testutils.read_std_array(os.path.join(pth, "surface.txt"), "float")
    surf_data = ["internal", "factor", 1.0, "iprn", -1, surf]

    # build time array series
    tas = {
        0.0: 9.5e-08,
        6.0e09: 9.5e-08,
        "filename": "test028_sfr.evt.tas",
        "time_series_namerecord": "evtarray_1",
        "interpolation_methodrecord": "LINEAR",
    }

    evt_package = ModflowGwfevta(
        model,
        readasarrays=True,
        timearrayseries=tas,
        surface=surf_data,
        depth=15.0,
        rate="TIMEARRAYSERIES evtarray_1",
        filename="test1tr.evt",
    )
    # attach obs package to evt
    obs_dict = {
        "test028_sfr.evt.csv": [
            ("obs-1", "EVT", (0, 1, 5)),
            ("obs-2", "EVT", (0, 2, 3)),
        ]
    }
    evt_package.obs.initialize(
        filename="test028_sfr.evt.obs", print_input=True, continuous=obs_dict
    )

    stress_period_data = {0: [((0, 12, 0), 988.0, 0.038), ((0, 13, 8), 1045.0, 0.038)]}
    ghb_package = ModflowGwfghb(
        model, maxbound=2, stress_period_data=stress_period_data
    )

    rch = testutils.read_std_array(os.path.join(pth, "recharge.txt"), "float")
    # test empty
    rch_data = ModflowGwfrcha.recharge.empty(model)
    rch_data[0]["data"] = rch
    rch_data[0]["factor"] = 5.000e-10
    rch_data[0]["iprn"] = -1
    rch_package = ModflowGwfrcha(
        model, readasarrays=True, recharge=rch_data, filename="test1tr.rch"
    )

    sfr_rec = testutils.read_sfr_rec(os.path.join(pth, "sfr_rec.txt"), 3)
    reach_con_rec = testutils.read_reach_con_rec(
        os.path.join(pth, "sfr_reach_con_rec.txt")
    )
    reach_div_rec = testutils.read_reach_div_rec(
        os.path.join(pth, "sfr_reach_div_rec.txt")
    )
    reach_per_rec = testutils.read_reach_per_rec(
        os.path.join(pth, "sfr_reach_per_rec.txt")
    )

    # test trying to get empty perioddata
    # should fail because data is jagged
    try:
        pd = ModflowGwfsfr.perioddata.empty(model)
        success = True
    except MFDataException:
        success = False
    assert not success

    # test zero based indexes
    reach_con_rec[0] = (0, -0.0)
    sfr_package = ModflowGwfsfr(
        model,
        unit_conversion=1.486,
        stage_filerecord="test1tr.sfr.stage.bin",
        budget_filerecord="test1tr.sfr.cbc",
        nreaches=36,
        packagedata=sfr_rec,
        connectiondata=reach_con_rec,
        diversions=reach_div_rec,
        perioddata={0: reach_per_rec},
    )
    assert sfr_package.connectiondata.get_data()[0][1] == -0.0
    assert sfr_package.connectiondata.get_data()[1][1] == 0.0
    assert sfr_package.connectiondata.get_data()[2][1] == 1.0
    assert sfr_package.packagedata.get_data()[1][1].lower() == "none"

    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    model = sim.get_model(model_name)
    sfr_package = model.get_package("sfr")
    # sfr_package.set_all_data_external()
    assert sfr_package.connectiondata.get_data()[0][1] == -0.0
    assert sfr_package.connectiondata.get_data()[1][1] == 0.0
    assert sfr_package.connectiondata.get_data()[2][1] == 1.0
    pdata = sfr_package.packagedata.get_data()
    assert sfr_package.packagedata.get_data()[1][1].lower() == "none"

    # undo zero based test and move on
    model.remove_package(sfr_package.package_type)
    reach_con_rec = testutils.read_reach_con_rec(
        os.path.join(pth, "sfr_reach_con_rec.txt")
    )

    # set sfr settings back to expected package data
    rec_line = (sfr_rec[1][0], (0, 1, 1)) + sfr_rec[1][2:]
    sfr_rec[1] = rec_line

    sfr_package = ModflowGwfsfr(
        model,
        unit_conversion=1.486,
        stage_filerecord="test1tr.sfr.stage.bin",
        budget_filerecord="test1tr.sfr.cbc",
        nreaches=36,
        packagedata=sfr_rec,
        connectiondata=reach_con_rec,
        diversions=reach_div_rec,
        perioddata={0: reach_per_rec},
    )

    obs_data_1 = testutils.read_obs(os.path.join(pth, "sfr_obs_1.txt"))
    obs_data_2 = testutils.read_obs(os.path.join(pth, "sfr_obs_2.txt"))
    obs_data_3 = testutils.read_obs(os.path.join(pth, "sfr_obs_3.txt"))
    obs_data = {
        "test1tr.sfr.csv": obs_data_1,
        "test1tr.sfr.qaq.csv": obs_data_2,
        "test1tr.sfr.flow.csv": obs_data_3,
    }
    sfr_package.obs.initialize(
        filename="test1tr.sfr.obs",
        digits=10,
        print_input=True,
        continuous=obs_data,
    )

    wells = testutils.read_wells(os.path.join(pth, "well.txt"))
    wel_package = ModflowGwfwel(
        model,
        boundnames=True,
        maxbound=10,
        stress_period_data={0: wells, 1: [()]},
    )

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # inspect cells
    cell_list = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
    out_file = function_tmpdir / "inspect_test028_sfr.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = function_tmpdir / "test1tr.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
        htol=10.0,
    )

    # test hpc package
    part = [("model1", 1), ("model2", 2)]
    hpc = ModflowUtlhpc(sim, dev_log_mpi=True, partitions=part, filename="test.hpc")

    assert sim.hpc.dev_log_mpi.get_data()
    assert hpc.filename == "test.hpc"
    part = hpc.partitions.get_data()
    assert part[0][0] == "model1"
    assert part[0][1] == 1
    assert part[1][0] == "model2"
    assert part[1][1] == 2

    sim.write_simulation()
    sim2 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    hpc_a = sim2.get_package("hpc")
    assert hpc_a.filename == "test.hpc"
    fr = sim2.name_file._hpc_filerecord.get_data()
    assert fr[0][0] == "test.hpc"
    assert hpc_a.dev_log_mpi.get_data()
    part_a = hpc_a.partitions.get_data()
    assert part_a[0][0] == "model1"
    assert part_a[0][1] == 1
    assert part_a[1][0] == "model2"
    assert part_a[1][1] == 2

    sim2.remove_package(hpc_a)
    sim2.set_sim_path(os.path.join(function_tmpdir, "temp"))
    sim2.write_simulation()
    sim3 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=os.path.join(function_tmpdir, "temp"),
    )
    hpc_n = sim3.get_package("hpc")
    assert hpc_n is None
    fr_2 = sim3.name_file._hpc_filerecord.get_data()
    assert fr_2 is None
    sim3.set_sim_path(function_tmpdir)

    hpc_data = {
        "filename": "hpc_data_file.hpc",
        "dev_log_mpi": True,
        "partitions": part,
    }
    sim4 = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=pth,
        hpc_data=hpc_data,
    )
    fr_4 = sim4.name_file._hpc_filerecord.get_data()
    assert fr_4[0][0] == "hpc_data_file.hpc"
    assert sim4.hpc.filename == "hpc_data_file.hpc"
    assert sim4.hpc.dev_log_mpi.get_data()
    part = sim4.hpc.partitions.get_data()
    assert part[0][0] == "model1"
    assert part[0][1] == 1
    assert part[1][0] == "model2"
    assert part[1][1] == 2

    # clean up
    sim3.delete_output_files()


@requires_exe("mf6")
@pytest.mark.regression
def test_create_tests_transport(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test_transport"
    name = "mst03"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file = expected_output_folder / "gwf_mst03.hds"
    expected_conc_file = expected_output_folder / "gwt_mst03.ucn"

    laytyp = [1]
    ss = [1.0e-10]
    sy = [0.1]
    nlay, nrow, ncol = 1, 1, 1

    nper = 2
    perlen = [2.0, 2.0]
    nstp = [14, 14]
    tsmult = [1.0, 1.0]
    delr = 10.0
    delc = 10.0
    top = 10.0
    botm = [0.0]
    strt = top
    hk = 1.0

    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 0.97

    tdis_rc = []
    for idx in range(nper):
        tdis_rc.append((perlen[idx], nstp[idx], tsmult[idx]))
    idx = 0

    # build MODFLOW 6 files
    sim = MFSimulation(
        sim_name=name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    # create tdis package
    tdis = ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

    # create gwf model
    gwfname = f"gwf_{name}"
    newtonoptions = ["NEWTON", "UNDER_RELAXATION"]
    gwf = ModflowGwf(
        sim,
        modelname=gwfname,
        newtonoptions=newtonoptions,
    )

    # create iterative model solution and register the gwf model with it
    imsgwf = ModflowIms(
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
        filename=f"{gwfname}.ims",
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    dis = ModflowGwfdis(
        gwf,
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
    ic = ModflowGwfic(gwf, strt=strt)

    # node property flow
    npf = ModflowGwfnpf(gwf, save_flows=False, icelltype=laytyp[idx], k=hk, k33=hk)
    # storage
    sto = ModflowGwfsto(
        gwf,
        save_flows=False,
        iconvert=laytyp[idx],
        ss=ss[idx],
        sy=sy[idx],
        steady_state={0: False},
        transient={0: True},
    )

    # wel files
    welspdict = {0: [[(0, 0, 0), -25.0, 0.0]], 1: [[(0, 0, 0), 25.0, 0.0]]}
    wel = ModflowGwfwel(
        gwf,
        print_input=True,
        print_flows=True,
        stress_period_data=welspdict,
        save_flows=False,
        auxiliary="CONCENTRATION",
        pname="WEL-1",
    )

    # output control
    oc = ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwfname}.cbc",
        head_filerecord=f"{gwfname}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # create gwt model
    gwtname = f"gwt_{name}"
    gwt = MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtname,
        model_nam_file=f"{gwtname}.nam",
    )

    # create iterative model solution and register the gwt model with it
    imsgwt = ModflowIms(
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
        filename=f"{gwtname}.ims",
    )
    sim.register_ims_package(imsgwt, [gwt.name])

    dis = ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=1,
        filename=f"{gwtname}.dis",
    )

    # initial conditions
    ic = ModflowGwtic(gwt, strt=100.0)

    # advection
    adv = ModflowGwtadv(gwt, scheme="UPSTREAM", filename=f"{gwtname}.adv")

    # mass storage and transfer
    mst = ModflowGwtmst(gwt, porosity=sy[idx], filename=f"{gwtname}.mst")

    # sources
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
    ssm = ModflowGwtssm(gwt, sources=sourcerecarray, filename=f"{gwtname}.ssm")

    # output control
    oc = ModflowGwtoc(
        gwt,
        budget_filerecord=f"{gwtname}.cbc",
        concentration_filerecord=f"{gwtname}.ucn",
        concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("CONCENTRATION", "ALL")],
        printrecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )

    # GWF GWT exchange
    gwfgwt = ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwtname,
        filename=f"{name}.gwfgwt",
    )

    # write MODFLOW 6 files
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # inspect cells
    cell_list = [(0, 0, 0)]
    out_file = function_tmpdir / "inspect_transport_gwf.csv"
    gwf.inspect_cells(cell_list, output_file_path=out_file)
    out_file = function_tmpdir / "inspect_transport_gwt.csv"
    gwt.inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = function_tmpdir / "gwf_mst03.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file],
        files2=[head_new],
        outfile=outfile,
    )
    conc_new = function_tmpdir / "gwt_mst03.ucn"
    assert compare_heads(
        None,
        None,
        files1=expected_conc_file,
        files2=conc_new,
        outfile=outfile,
        text="concentration",
    )

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@requires_pkg("shapely")
@pytest.mark.slow
@pytest.mark.regression
def test001a_tharmonic(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test001a_Tharmonic"
    model_name = "flow15"

    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(expected_output_folder, "flow15_flow_unch.hds")
    expected_head_file_b = os.path.join(expected_output_folder, "flow15_flow_adj.hds")
    expected_cbc_file_a = os.path.join(expected_output_folder, "flow15_flow_unch.cbc")
    expected_cbc_file_b = os.path.join(expected_output_folder, "flow15_flow_adj.cbc")

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(
        model_name,
        "mf6",
        "mf6",
        pth,
        verbosity_level=0,
        verify_data=True,
        write_headers=False,
    )
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external(external_data_folder="data")
    sim.write_simulation(silent=True)
    # verify external data written to correct location
    data_folder = function_tmpdir / "data" / "flow15.dis_botm.txt"
    assert os.path.exists(data_folder)
    # model export test
    model = sim.get_model(model_name)
    model.export(f"{model.model_ws}/tharmonic.nc")
    model.export(f"{model.model_ws}/tharmonic.shp")
    model.dis.botm.export(f"{model.model_ws}/botm.shp")

    mg = model.modelgrid

    # run simulation
    success, buff = sim.run_simulation()
    print(sim.name)
    assert success, f"simulation {sim.name} did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_a, precision="auto")
    budget_frf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )

    # compare output to expected results
    head_new = function_tmpdir / "flow15_flow.hds"
    assert compare_heads(None, None, files1=[expected_head_file_a], files2=[head_new])

    budget_frf = sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    assert array_util.array_comp(budget_frf_valid, budget_frf)

    # change some settings
    hk_data = sim.simulation_data.mfdata[(model_name, "npf", "griddata", "k")]
    hk_array = hk_data.get_data()
    hk_array[0, 0, 1] = 20.0
    hk_data.set_data(hk_array)

    model = sim.get_model(model_name)
    ic = model.get_package("ic")
    ic_data = ic.strt
    ic_array = ic_data.get_data()
    ic_array[0, 0, 0] = 1.0
    ic_array[0, 0, 9] = 1.0
    ic_data.set_data(ic_array)

    get_test = hk_data[0, 0, 0]
    assert get_test == 10.0
    get_test = hk_data.array
    assert array_util.array_comp(
        get_test,
        [[10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
    )
    get_test = hk_data[:]
    assert array_util.array_comp(
        get_test,
        [[[10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]],
    )

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_b, precision="auto")
    budget_frf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )

    # compare output to expected results
    head_new = os.path.join(save_folder, "flow15_flow.hds")
    assert compare_heads(None, None, files1=[expected_head_file_b], files2=[head_new])

    budget_frf = sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    assert array_util.array_comp(budget_frf_valid, budget_frf)


@requires_exe("mf6")
@pytest.mark.regression
def test003_gwfs_disv(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test003_gwfs_disv"
    model_name = "gwf_1"
    data_folder = example_data_path / "mf6" / test_ex_name
    expected_output_folder = data_folder / "expected_output"
    expected_head_file_a = expected_output_folder / "model_unch.hds"
    expected_head_file_b = expected_output_folder / "model_adj.hds"
    expected_cbc_file_a = expected_output_folder / "model_unch.cbc"
    expected_cbc_file_b = expected_output_folder / "model_adj.cbc"

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", data_folder, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.simulation_data.max_columns_of_data = 10
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_a, precision="auto")
    budget_fjf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )

    head_new = os.path.join(function_tmpdir, "model.hds")
    assert compare_heads(None, None, files1=[expected_head_file_a], files2=[head_new])

    budget_frf = sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    assert array_util.array_comp(budget_fjf_valid, budget_frf)

    model = sim.get_model(model_name)
    model.export(function_tmpdir / f"{test_ex_name}.shp")

    # change some settings
    chd_head_left = model.get_package("CHD_LEFT")
    chd_left_period = chd_head_left.stress_period_data.get_data(0)
    chd_left_period[4][1] = 15.0
    chd_head_left.stress_period_data.set_data(chd_left_period, 0)

    chd_head_right = model.get_package("CHD_RIGHT")
    chd_right_period = chd_head_right.stress_period_data
    chd_right_data = chd_right_period.get_data(0)
    chd_right_data_slice = chd_right_data[3:10]
    chd_right_period.set_data(chd_right_data_slice, 0)

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_b, precision="double")
    budget_fjf_valid = np.array(budget_obj.get_data(text="FLOW JA FACE", full3D=True))

    # compare output to expected results
    head_new = os.path.join(save_folder, "model.hds")
    assert compare_heads(None, None, files1=[expected_head_file_b], files2=[head_new])

    budget_frf = sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    assert array_util.array_comp(budget_fjf_valid, budget_frf)


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test005_advgw_tidal(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test005_advgw_tidal"
    model_name = "gwf_1"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(expected_output_folder, "AdvGW_tidal_unch.hds")
    expected_head_file_b = os.path.join(expected_output_folder, "AdvGW_tidal_adj.hds")

    # load simulation
    sim = MFSimulation.load(
        model_name, "mf6", "mf6", pth, verbosity_level=2, verify_data=True
    )

    # test obs/ts package interface
    model = sim.get_model(model_name)
    time = model.modeltime
    assert (
        time.steady_state[0]
        and not time.steady_state[1]
        and not time.steady_state[2]
        and not time.steady_state[3]
    )
    ghb = model.get_package("ghb")
    obs = ghb.obs
    digits = obs.digits.get_data()
    assert digits == 10
    names = ghb.ts.time_series_namerecord.get_data()
    assert names[0][0] == "tides"

    # test obs blocks
    obs_pkg = model.get_package("obs-1")
    cont_mfl = obs_pkg.continuous
    cont_data = cont_mfl.get_data()
    assert len(cont_data) == 2
    assert "head_hydrographs.csv" in cont_data
    assert "gwf-advtidal.obs.flow.csv" in cont_data
    flow = cont_data["gwf-advtidal.obs.flow.csv"]
    assert flow[0][0] == "icf1"
    assert flow[0][1] == "flow-ja-face"
    assert flow[0][2] == (2, 4, 6)
    assert flow[0][3] == (2, 4, 7)

    # add a stress period beyond nper
    spd = ghb.stress_period_data.get_data()
    spd[20] = copy.deepcopy(spd[0])
    ghb.stress_period_data.set_data(spd)

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = os.path.join(function_tmpdir, "advgw_tidal.hds")
    outfile = os.path.join(function_tmpdir, "head_compare.dat")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
        outfile=outfile,
    )


@requires_exe("mf6")
@pytest.mark.regression
def test006_2models_different_dis(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test006_2models_diff_dis"
    model_name_1 = "model1"
    model_name_2 = "model2"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_1 = os.path.join(expected_output_folder, "model1.hds")
    expected_head_file_2 = os.path.join(expected_output_folder, "model2.hds")

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=pth)
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=tdis_rc)
    model_1 = ModflowGwf(
        sim,
        modelname=model_name_1,
        model_nam_file=f"{model_name_1}.nam",
    )
    model_2 = ModflowGwf(
        sim,
        modelname=model_name_2,
        model_nam_file=f"{model_name_2}.nam",
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.00000001,
        outer_maximum=1000,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_dvclose=0.00000001,
        rcloserecord=0.01,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model_1.name, model_2.name])
    dis_package = ModflowGwfdis(
        model_1,
        length_units="METERS",
        nlay=1,
        nrow=7,
        ncol=7,
        idomain=1,
        delr=100.0,
        delc=100.0,
        top=0.0,
        botm=-100.0,
        filename=f"{model_name_1}.dis",
    )

    vertices = testutils.read_vertices(os.path.join(pth, "vertices.txt"))
    c2drecarray = testutils.read_cell2d(os.path.join(pth, "cell2d.txt"))
    disv_package = ModflowGwfdisv(
        model_2,
        ncpl=121,
        nlay=1,
        nvert=148,
        top=0.0,
        botm=-40.0,
        idomain=1,
        vertices=vertices,
        cell2d=c2drecarray,
        filename=f"{model_name_2}.disv",
    )
    ic_package_1 = ModflowGwfic(model_1, strt=1.0, filename=f"{model_name_1}.ic")
    ic_package_2 = ModflowGwfic(model_2, strt=1.0, filename=f"{model_name_2}.ic")
    npf_package_1 = ModflowGwfnpf(
        model_1, save_flows=True, perched=True, icelltype=0, k=1.0, k33=1.0
    )
    npf_package_2 = ModflowGwfnpf(
        model_2, save_flows=True, perched=True, icelltype=0, k=1.0, k33=1.0
    )
    oc_package_1 = ModflowGwfoc(
        model_1,
        budget_filerecord="model1.cbc",
        head_filerecord="model1.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    oc_package_2 = ModflowGwfoc(
        model_2,
        budget_filerecord="model2.cbc",
        head_filerecord="model2.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # build periodrecarray for chd package
    set_1 = [0, 7, 14, 18, 22, 26, 33]
    set_2 = [6, 13, 17, 21, 25, 32, 39]
    stress_period_data = []
    for value in range(0, 7):
        stress_period_data.append(((0, value, 0), 1.0))
    for value in range(0, 7):
        stress_period_data.append(((0, value, 6), 0.0))
    chd_package = ModflowGwfchd(
        model_1,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=30,
        stress_period_data=stress_period_data,
    )
    exgrecarray = testutils.read_exchangedata(os.path.join(pth, "exg.txt"), 3, 2)
    exg_data = {
        "filename": "exg_data.bin",
        "data": exgrecarray,
        "binary": True,
    }

    # build obs dictionary
    gwf_obs = {
        ("gwfgwf_obs.csv"): [
            ("gwf-1-3-2_1-1-1", "flow-ja-face", (0, 2, 1), (0, 0, 0)),
            ("gwf-1-3-2_1-2-1", "flow-ja-face", (0, 2, 1), (0, 1, 0)),
        ]
    }

    exg_package = ModflowGwfgwf(
        sim,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary="testaux",
        nexg=9,
        exchangedata=exg_data,
        exgtype="gwf6-gwf6",
        exgmnamea=model_name_1,
        exgmnameb=model_name_2,
        observations=gwf_obs,
    )

    gnc_path = os.path.join("gnc", "test006_2models_gnc.gnc")
    gncrecarray = testutils.read_gncrecarray(os.path.join(pth, "gnc.txt"), 3, 2)
    gnc_package = exg_package.gnc.initialize(
        filename=gnc_path,
        print_input=True,
        print_flows=True,
        numgnc=9,
        numalphaj=1,
        gncdata=gncrecarray,
    )

    # change folder to save simulation
    sim.set_sim_path(function_tmpdir)
    exg_package.exchangedata.set_record(exg_data)

    # write simulation to new location
    sim.write_simulation()
    # run simulation
    success, buff = sim.run_simulation()
    assert success

    sim2 = MFSimulation.load(sim_ws=sim.sim_path)
    exh = sim2.get_package("gwfgwf")
    exh_data = exh.exchangedata.get_data()
    assert exh_data[0][0] == (0, 2, 1)
    assert exh_data[0][1] == (0, 0)
    assert exh_data[3][0] == (0, 3, 1)
    assert exh_data[3][1] == (0, 3)
    gnc = sim2.get_package("gnc")
    gnc_data = gnc.gncdata.get_data()
    assert gnc_data[0][0] == (0, 2, 1)
    assert gnc_data[0][1] == (0, 0)
    assert gnc_data[0][2] == (0, 1, 1)

    # test remove_model
    sim2.remove_model(model_name_2)
    sim2.write_simulation()
    success, buff = sim2.run_simulation()
    assert success
    sim3 = MFSimulation.load(sim_ws=sim.sim_path)
    assert sim3.get_model(model_name_1) is not None
    assert sim3.get_model(model_name_2) is None
    assert len(sim3.name_file.models.get_data()) == 1
    assert sim3.name_file.exchanges.get_data() is None

    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.regression
def test006_gwf3(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test006_gwf3"
    model_name = "gwf_1"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file_a = expected_output_folder / "flow_unch.hds"
    expected_head_file_b = expected_output_folder / "flow_adj.hds"
    expected_cbc_file_a = expected_output_folder / "flow_unch.cbc"
    expected_cbc_file_b = expected_output_folder / "flow_adj.cbc"

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, verify_data=True)
    sim.set_sim_path(function_tmpdir)
    model = sim.get_model()
    disu = model.get_package("disu")
    # test switching disu array to internal array
    disu.ja = disu.ja.array
    # test writing hwva and cl12 arrays out to different locations
    disu.hwva = {
        "filename": "flow.disu.hwva_new.dat",
        "factor": 1.0,
        "data": disu.hwva.array,
    }
    disu.cl12 = {
        "filename": "flow.disu.cl12_new.dat",
        "factor": 1.0,
        "data": disu.cl12.array,
    }

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)
    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # inspect cells
    cell_list = [(0,), (7,), (14,)]
    out_file = function_tmpdir / "inspect_test006_gwf3.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    budget_obj = CellBudgetFile(expected_cbc_file_a, precision="double")
    budget_fjf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )
    jaentries = budget_fjf_valid.shape[-1]
    budget_fjf_valid.shape = (-1, jaentries)

    # compare output to expected results
    head_new = function_tmpdir / "flow.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
    )

    budget_fjf = np.array(
        sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    )
    assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

    # change some settings
    model = sim.get_model(model_name)
    hk = model.get_package("npf").k
    hk_data = hk.get_data()
    hk_data[2] = 3.5
    hk.set_data(hk_data)
    ex_happened = False
    try:
        hk.make_layered()
    except:
        ex_happened = True
    assert ex_happened

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun(2) did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_b, precision="auto")
    budget_fjf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )
    jaentries = budget_fjf_valid.shape[-1]
    budget_fjf_valid.shape = (-1, jaentries)

    # compare output to expected results
    head_new = os.path.join(save_folder, "flow.hds")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
    )

    budget_fjf = np.array(
        sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    )
    assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

    # confirm that files did move
    save_folder = function_tmpdir / "save02"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)

    # write with "copy_external_files" turned off so external files
    # do not get copied to new location
    sim.write_simulation(ext_file_action=ExtFileAction.copy_none)

    # store strt in an external binary file
    model = sim.get_model()
    ic = model.get_package("ic")
    ic.strt.store_as_external_file("initial_heads.bin", binary=True)

    strt_data = ic.strt.array
    # update packages
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun(3) did not run"

    # get expected results
    budget_obj = CellBudgetFile(expected_cbc_file_b, precision="double")
    budget_fjf_valid = np.array(
        budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
    )
    jaentries = budget_fjf_valid.shape[-1]
    budget_fjf_valid.shape = (-1, jaentries)

    # compare output to expected results
    head_new = os.path.join(save_folder, "flow.hds")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
    )

    budget_fjf = np.array(
        sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
    )
    assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

    # confirm that files did not move
    assert not os.path.isfile(os.path.join(save_folder, "flow.disu.ja.dat"))
    assert not os.path.isfile(os.path.join(save_folder, "flow.disu.iac.dat"))
    assert not os.path.isfile(os.path.join(save_folder, "flow.disu.cl12.dat"))
    assert not os.path.isfile(os.path.join(save_folder, "flow.disu.area.dat"))
    assert not os.path.isfile(os.path.join(save_folder, "flow.disu.hwva.dat"))
    # confirm external binary file was created
    assert os.path.isfile(os.path.join(save_folder, "initial_heads.bin"))

    # clean up
    sim.delete_output_files()


@requires_exe("mf6")
@pytest.mark.regression
def test045_lake1ss_table(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test045_lake1ss_table"
    model_name = "lakeex1b"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(expected_output_folder, "lakeex1b_unch.hds")
    expected_head_file_b = os.path.join(expected_output_folder, "lakeex1b_adj.hds")

    # load simulation
    sim = MFSimulation.load(
        sim_name=model_name,
        exe_name="mf6",
        sim_ws=pth,
        verify_data=True,
    )

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = function_tmpdir / "lakeex1b.hds"
    outfile = function_tmpdir / "headcompare_a.txt"
    success = compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
        outfile=outfile,
    )
    assert success

    # change some settings
    model = sim.get_model(model_name)
    lak = model.get_package("lak")
    laktbl = lak.get_package("laktab")
    laktbl = model.get_package("laktab").table
    laktbl_data = laktbl.get_data()
    laktbl_data[-1][0] = 700.0
    laktbl.set_data(laktbl_data)
    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.set_all_data_external(
        external_data_folder="test_folder",
        base_name="ext_file",
        binary=True,
    )
    sim.write_simulation()
    # verify external files were written
    ext_folder = os.path.join(save_folder, "test_folder")
    files_to_check = [
        "ext_file_lakeex1b.dis_botm_layer1.bin",
        "ext_file_lakeex1b.dis_botm_layer2.bin",
        "ext_file_lakeex1b.dis_botm_layer3.bin",
        "ext_file_lakeex1b.dis_botm_layer4.bin",
        "ext_file_lakeex1b.dis_botm_layer5.bin",
        "ext_file_lakeex1b.npf_k_layer1.bin",
        "ext_file_lakeex1b.npf_k_layer5.bin",
        "ext_file_lakeex1b.chd_stress_period_data_1.bin",
        "ext_file_lakeex1b.lak_connectiondata.txt",
        "ext_file_lakeex1b.lak_packagedata.txt",
        "ext_file_lakeex1b.lak_perioddata_1.txt",
        "ext_file_lakeex1b_table.ref_table.txt",
        "ext_file_lakeex1b.evt_depth_1.bin",
        "ext_file_lakeex1b.evt_rate_1.bin",
        "ext_file_lakeex1b.evt_surface_1.bin",
    ]
    for file in files_to_check:
        data_file_path = os.path.join(ext_folder, file)
        assert os.path.exists(data_file_path)

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # compare output to expected results
    head_new = save_folder / "lakeex1b.hds"
    outfile = function_tmpdir / "headcompare_b.txt"
    success = compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
        outfile=outfile,
    )
    assert success


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test006_2models_mvr(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test006_2models_mvr"
    sim_name = "test006_2models_mvr"
    model_names = ["parent", "child"]
    data_folder = example_data_path / "mf6" / test_ex_name
    # copy example data into working directory
    ws = function_tmpdir / "ws"
    shutil.copytree(data_folder, ws)

    expected_output_folder = ws / "expected_output"
    expected_head_file_a = expected_output_folder / "model1_unch.hds"
    expected_head_file_aa = expected_output_folder / "model2_unch.hds"
    expected_cbc_file_a = expected_output_folder / "model1_unch.cbc"
    expected_head_file_b = expected_output_folder / "model1_adj.hds"
    expected_head_file_bb = expected_output_folder / "model2_adj.hds"

    # load simulation
    sim = MFSimulation.load(sim_name, "mf6", "mf6", data_folder, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(ws)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = ws / "model1.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
    )

    head_new = ws / "model2.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_aa],
        files2=[head_new],
    )

    budget_obj = CellBudgetFile(
        expected_cbc_file_a,
        precision="double",
    )

    # test getting models
    model_dict = sim.model_dict
    assert len(model_dict) == 2
    for model in model_dict.values():
        assert model.name in model_names
    names = sim.model_names
    assert len(names) == 2
    for name in names:
        assert name in model_names
        model = sim.get_model(name)
        assert model.model_type == "gwf6"
    models = sim.gwf
    assert len(models) == 2
    for model in models:
        assert model.name in model_names
        assert model.model_type == "gwf6"

    # change some settings
    parent_model = sim.get_model(model_names[0])
    maw_pkg = parent_model.get_package("maw")
    period_data = maw_pkg.perioddata.get_data()
    period_data[0][0][2] = -1.0
    maw_pkg.perioddata.set_data(period_data[0], 0)
    well_rec_data = maw_pkg.packagedata.get_data()
    assert well_rec_data[0][0] == 0

    exg_pkg = sim.get_exchange_file("simulation.exg")
    exg_data = exg_pkg.exchangedata.get_data()
    for index in range(0, len(exg_data)):
        exg_data[index][6] = 500.0
    exg_pkg.exchangedata.set_data(exg_data)

    # test getting packages
    pkg_list = parent_model.get_package()
    assert len(pkg_list) == 6
    # confirm that this is a copy of the original dictionary with references
    # to the packages
    del pkg_list[0]
    assert len(pkg_list) == 5
    pkg_list = parent_model.get_package()
    assert len(pkg_list) == 6

    dis_pkg = parent_model.get_package("dis")
    old_val = dis_pkg.nlay.get_data()
    dis_pkg.nlay = 22
    pkg_list = parent_model.get_package()
    assert dis_pkg.nlay.get_data() == 22
    dis_pkg.nlay = old_val

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    cell_list = [(0, 3, 1)]
    out_file = ws / "inspect_test006_2models_mvr.csv"
    models[0].inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = os.path.join(save_folder, "model1.hds")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
    )

    head_new = os.path.join(save_folder, "model2.hds")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_bb],
        files2=[head_new],
    )

    # test load_only
    model_package_check = ["ic", "maw", "npf", "oc"]
    load_only_lists = [
        ["ic6", "npf6", "oc", "gwf6-gwf6", "ims"],
        ["ic", "maw", "npf", "gwf-gwf", "ims"],
        ["ic", "maw6", "npf"],
    ]
    for load_only in load_only_lists:
        sim = MFSimulation.load(
            sim_name, "mf6", "mf6", data_folder, load_only=load_only
        )
        for model_name in model_names:
            model = sim.get_model(model_name)
            for package in model_package_check:
                assert (
                    model.get_package(package, type_only=True) is not None
                    or sim.get_package(package, type_only=True) is not None
                ) == (package in load_only or f"{package}6" in load_only)
        assert (len(sim._exchange_files) > 0) == (
            "gwf6-gwf6" in load_only or "gwf-gwf" in load_only
        )
        assert (len(sim._solution_files) > 0) == (
            "ims6" in load_only or "ims" in load_only
        )

    # load package by name
    load_only_list = ["ic6", "maw", "npf_p1", "oc_p2", "ims"]
    sim = MFSimulation.load(
        sim_name, "mf6", "mf6", data_folder, load_only=load_only_list
    )
    model_parent = sim.get_model("parent")
    model_child = sim.get_model("child")
    assert model_parent.get_package("oc") is None
    assert model_child.get_package("oc") is not None
    assert model_parent.get_package("npf") is not None
    assert model_child.get_package("npf") is None

    # test running a runnable load_only case
    sim = MFSimulation.load(
        sim_name, "mf6", "mf6", data_folder, load_only=load_only_lists[0]
    )
    sim.set_sim_path(ws)
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test001e_uzf_3lay(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test001e_UZF_3lay"
    model_name = "gwf_1"
    pth = example_data_path / "mf6" / test_ex_name

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.write_simulation()

    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # change some settings
    model = sim.get_model(model_name)
    uzf = model.get_package("uzf")
    uzf_data = uzf.packagedata
    uzf_array = uzf_data.get_data()
    # increase initial water content
    for index in range(0, len(uzf_array)):
        uzf_array[index][7] = 0.3
    uzf_data.set_data(uzf_array)

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # inspect cells
    cell_list = [(0, 0, 1), (0, 0, 2), (2, 0, 8)]
    out_file = function_tmpdir / "inspect_test001e_uzf_3lay.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    # test load_only
    model_package_check = ["chd", "ic", "npf", "oc", "sto", "uzf"]
    load_only_lists = [
        ["chd6", "ic6", "ims", "npf6", "obs", "oc", "sto"],
        ["chd6", "ims", "npf6", "obs", "oc", "sto", "uzf6"],
        ["chd", "ic", "npf", "obs", "sto"],
        ["ic6", "ims", "obs6", "oc6"],
    ]
    for load_only in load_only_lists:
        sim = MFSimulation.load(model_name, "mf6", "mf6", pth, load_only=load_only)
        sim.set_sim_path(function_tmpdir)
        model = sim.get_model()
        for package in model_package_check:
            assert (model.get_package(package, type_only=True) is not None) == (
                package in load_only or f"{package}6" in load_only
            )
    # test running a runnable load_only case
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, load_only=load_only_lists[0])
    sim.set_sim_path(function_tmpdir)
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} from load did not run"

    cbc = CellBudgetFile(
        function_tmpdir / "test001e_UZF_3lay.uzf.cbc", precision="auto"
    )
    data = cbc.get_data(text="GWF", full3D=False)
    assert data[2].node[0] == 1, "Budget precision error for imeth 6"

    sim = MFSimulation.load("mfsim", sim_ws=function_tmpdir, exe_name="mf6")

    ims = sim.ims
    sim.remove_package(ims)

    ims = ModflowIms(sim, print_option="SUMMARY", complexity="COMPLEX")
    sim.register_ims_package(ims, ["GwF_1"])
    sim.write_simulation()

    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test045_lake2tr(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test045_lake2tr"
    model_name = "lakeex2a"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = pth / "expected_output"
    expected_head_file_a = expected_output_folder / "lakeex2a_unch.hds"
    expected_head_file_b = expected_output_folder / "lakeex2a_adj.hds"

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, verify_data=True)

    # write simulation to new location
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = function_tmpdir / "lakeex2a.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
        htol=10.0,
    )

    # change some settings
    model = sim.get_model(model_name)
    evt = model.get_package("evt")
    evt.rate.set_data([0.05], key=0)

    lak = model.get_package("lak")
    lak_period = lak.perioddata
    lak_period_data = lak_period.get_data()
    lak_period_data[0][2][2] = "0.05"
    lak_period.set_data(lak_period_data[0], 0)

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # inspect cells
    cell_list = [(0, 6, 5), (0, 8, 5), (1, 18, 6)]
    out_file = function_tmpdir / "inspect_test045_lake2tr.csv"
    model.inspect_cells(cell_list, output_file_path=out_file)

    # compare output to expected results
    head_new = save_folder / "lakeex2a.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
        htol=10.0,
    )


@requires_exe("mf6")
@pytest.mark.regression
def test036_twrihfb(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test036_twrihfb"
    model_name = "twrihfb2015"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "twrihfb2015_output_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "twrihfb2015_output_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = function_tmpdir / "twrihfb2015_output.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
    )

    # change some settings
    hydchr = sim.simulation_data.mfdata[
        (model_name, "hfb", "period", "stress_period_data")
    ]
    hydchr_data = hydchr.get_data()
    hydchr_data[0][2][2] = 0.000002
    hydchr_data[0][3][2] = 0.000003
    hydchr_data[0][4][2] = 0.0000004
    hydchr.set_data(hydchr_data[0], 0)
    cond = sim.simulation_data.mfdata[
        (model_name, "drn", "period", "stress_period_data")
    ]
    cond_data = cond.get_data()
    for index in range(0, len(cond_data[0])):
        cond_data[0][index][2] = 2.1
    cond.set_data(cond_data[0], 0)

    rch = sim.simulation_data.mfdata[(model_name, "rcha", "period", "recharge")]
    rch_data = rch.get_data()
    assert rch_data[0][5, 1] == 0.00000003

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # compare output to expected results
    head_new = save_folder / "twrihfb2015_output.hds"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
    )


@requires_exe("mf6")
@pytest.mark.slow
@pytest.mark.regression
def test027_timeseriestest(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test027_TimeseriesTest"
    model_name = "gwf_1"
    pth = example_data_path / "mf6" / test_ex_name
    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "timeseriestest_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "timeseriestest_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", "mf6", pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(function_tmpdir)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # reload sim
    sim = MFSimulation.load(model_name, "mf6", "mf6", function_tmpdir, verify_data=True)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # compare output to expected results
    head_new = function_tmpdir / "timeseriestest.hds"
    outfile = function_tmpdir / "head_compare.dat"
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_a],
        files2=[head_new],
        outfile=outfile,
        htol=10.0,
    )

    model = sim.get_model(model_name)
    rch = model.get_package("rcha")
    tas_rch = rch.get_package("tas")
    tas_array_data = tas_rch.tas_array.get_data(12.0)
    assert tas_array_data == 0.0003
    tas_array_data = 0.02
    tas_rch.tas_array.set_data(tas_array_data, key=12.0)

    # write simulation again
    save_folder = function_tmpdir / "save"
    save_folder.mkdir()
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    # run simulation
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} rerun did not run"

    # compare output to expected results
    head_new = os.path.join(save_folder, "timeseriestest.hds")
    assert compare_heads(
        None,
        None,
        files1=[expected_head_file_b],
        files2=[head_new],
        htol=10.0,
    )


@pytest.mark.regression
def test099_create_tests_int_ext(function_tmpdir, example_data_path):
    # init paths
    test_ex_name = "test099_int_ext"
    model_name = "test099_int_ext"
    pth = example_data_path / "mf6" / "create_tests" / test_ex_name

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    sim.name_file.continue_.set_data(True)
    tdis_rc = [(1577889000, 50, 1.1), (1577889000, 50, 1.1)]
    tdis_package = ModflowTdis(
        sim,
        time_units="SECONDS",
        nper=2,
        perioddata=tdis_rc,
        filename="simulation.tdis",
    )
    model = ModflowGwf(sim, modelname=model_name, model_nam_file=f"{model_name}.nam")
    model.name_file.save_flows.set_data(True)
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=0.00001,
        outer_maximum=100,
        under_relaxation="DBD",
        under_relaxation_theta=0.85,
        under_relaxation_kappa=0.0001,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.1,
        backtracking_number=0,
        backtracking_tolerance=1.1,
        backtracking_reduction_factor=0.7,
        backtracking_residual_limit=1.0,
        inner_dvclose=0.00001,
        rcloserecord=0.1,
        inner_maximum=100,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
        filename="model.ims",
    )
    sim.register_ims_package(ims_package, [model.name])
    top = 100.0
    botm = np.zeros((15, 10), float)
    idomain = 1
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=15,
        ncol=10,
        delr=5000.0,
        delc=5000.0,
        top=top,
        botm=botm,
        idomain=idomain,
        filename=f"{model_name}.dis",
    )
    strt = np.ones((15, 10), float) * 50.0
    strt_int = {"filename": "strt.txt", "factor": 0.8, "iprn": 0, "data": strt}
    ic_package = ModflowGwfic(model, strt=strt_int, filename=f"{model_name}.ic")

    k_vals = np.ones((15, 10), float) * 10.0
    assert k_vals[0, 0] == 10.0
    k = {"filename": "k.txt", "factor": 3.000e-03, "iprn": 0, "data": k_vals}
    npf_package = ModflowGwfnpf(model, icelltype=1, k=k, k33=1.0)
    npf_package.k.factor = 2.000e-04

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="test1tr.cbc",
        head_filerecord="test1tr.hds",
        saverecord={0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]},
        printrecord={0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]},
    )

    sy_vals = np.ones((15, 10), float) * 0.1
    sy = {"factor": 0.2, "iprn": 0, "data": sy_vals}
    sto_package = ModflowGwfsto(model, iconvert=1, ss=1.0e-6, sy=sy)

    sim.write_simulation()
    sim_2 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    sim_2.set_sim_path(os.path.join(function_tmpdir, "sim_2"))
    model = sim_2.get_model(model_name)
    npf_package = model.get_package("npf")
    k_record = npf_package.k.get_record()
    assert k_record["factor"] == 2.000e-04
    assert k_record["data"][0, 0, 0] == 10.0

    ic_package = model.get_package("ic")
    strt_record = ic_package.strt.get_record()
    assert strt_record["factor"] == 0.8
    assert strt_record["data"][0, 0, 0] == 50.0

    sim_3 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    model = sim_3.get_model(model_name)
    npf_package = model.get_package("npf")
    k_record = npf_package.k.get_record()
    assert k_record["factor"] == 2.000e-04
    assert k_record["data"][0, 0, 0] == 10.0

    ic_package = model.get_package("ic")
    strt_record = ic_package.strt.get_record()
    assert strt_record["factor"] == 0.8
    assert strt_record["data"][0, 0, 0] == 50.0

    sim_3.set_all_data_external()
    sim_3.write_simulation()

    sim_4 = MFSimulation.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    model = sim_4.get_model(model_name)
    npf_package = model.get_package("npf")
    k_record = npf_package.k.get_record()
    assert "filename" in k_record
    assert k_record["factor"] == 2.000e-04
    assert k_record["data"][0, 0, 0] == 10.0

    ic_package = model.get_package("ic")
    strt_record = ic_package.strt.get_record()
    assert "filename" in strt_record
    assert strt_record["factor"] == 0.8
    assert strt_record["data"][0, 0, 0] == 50.0

    k_record["factor"] = 4.000e-04
    npf_package.k.set_record(k_record)
    k_record = npf_package.k.get_record()
    assert k_record["factor"] == 4.000e-04
    assert k_record["data"][0, 0, 0] == 10.0

    k_vals = np.ones((15, 10), float) * 50.0
    k_record["data"] = k_vals
    npf_package.k.set_record(k_record)
    k_record = npf_package.k.get_record()
    assert k_record["factor"] == 4.000e-04
    assert k_record["data"][0, 0, 0] == 50.0
