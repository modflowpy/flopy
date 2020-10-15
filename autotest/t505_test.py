import os

import numpy as np

import flopy
import flopy.utils.binaryfile as bf
from flopy.mf6.data.mfdatastorage import DataStorageType
from flopy.utils.datautil import PyListUtil
from flopy.mf6.mfbase import FlopyException
from flopy.mf6.modflow.mfgwf import ModflowGwf
from flopy.mf6.modflow.mfgwfchd import ModflowGwfchd
from flopy.mf6.modflow.mfgwfdis import ModflowGwfdis
from flopy.mf6.modflow.mfgwfdisv import ModflowGwfdisv
from flopy.mf6.modflow.mfgwfdrn import ModflowGwfdrn
from flopy.mf6.modflow.mfgwfevt import ModflowGwfevt
from flopy.mf6.modflow.mfgwfevta import ModflowGwfevta
from flopy.mf6.modflow.mfgwfghb import ModflowGwfghb
from flopy.mf6.modflow.mfgwfgnc import ModflowGwfgnc
from flopy.mf6.modflow.mfgwfgwf import ModflowGwfgwf
from flopy.mf6.modflow.mfgwfhfb import ModflowGwfhfb
from flopy.mf6.modflow.mfgwfic import ModflowGwfic
from flopy.mf6.modflow.mfgwfnpf import ModflowGwfnpf
from flopy.mf6.modflow.mfgwfoc import ModflowGwfoc
from flopy.mf6.modflow.mfgwfrch import ModflowGwfrch
from flopy.mf6.modflow.mfgwfrcha import ModflowGwfrcha
from flopy.mf6.modflow.mfgwfriv import ModflowGwfriv
from flopy.mf6.modflow.mfgwfsfr import ModflowGwfsfr
from flopy.mf6.modflow.mfgwfsto import ModflowGwfsto
from flopy.mf6.modflow.mfgwfwel import ModflowGwfwel
from flopy.mf6.modflow.mfims import ModflowIms
from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.mf6.modflow.mftdis import ModflowTdis
from flopy.mf6.modflow.mfutlobs import ModflowUtlobs
from flopy.mf6.modflow.mfutlts import ModflowUtlts
from flopy.mf6.utils import testutils
from flopy.mf6.mfbase import MFDataException

try:
    import shapefile
    if int(shapefile.__version__.split('.')[0]) < 2:
        shapefile = None
except ImportError:
    shapefile = None

try:
    import pymake
except:
    print("could not import pymake")

exe_name = "mf6"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False

cpth = os.path.join("temp", "t505")
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)


def np001():
    # init paths
    test_ex_name = "np001"
    model_name = "np001_mod"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "np001_mod.hds")
    expected_cbc_file = os.path.join(expected_output_folder, "np001_mod.cbc")

    # model tests
    test_sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=run_folder,
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
            model_nam_file="{}.nam".format(model_name),
            **kwargs
        )
    except FlopyException:
        ex = True
    assert ex == True

    kwargs = {}
    kwargs["xul"] = 20.5
    good_model = ModflowGwf(
        test_sim,
        modelname=model_name,
        model_nam_file="{}.nam".format(model_name),
        model_rel_path="model_folder",
        **kwargs
    )

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=pth,
        write_headers=False,
    )
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=[(2.0, 1, 1.0)]
    )
    # specifying the tdis package twice should remove the old tdis package
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )
    # first ims file to be replaced
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename="old_name.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_hclose=0.00001,
        outer_maximum=10,
        under_relaxation="NONE",
        inner_maximum=10,
        inner_hclose=0.001,
        linear_acceleration="CG",
        preconditioner_levels=2,
        preconditioner_drop_tolerance=0.00001,
        number_orthogonalizations=5,
    )
    # replace with real ims file
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename="{}.ims".format(test_ex_name),
        print_option="ALL",
        complexity="SIMPLE",
        outer_hclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_hclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )

    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    # test case insensitive lookup
    assert sim.get_model(model_name.upper()) is not None

    # test getting model using attribute
    model = sim.np001_mod
    assert model is not None and model.name == "np001_mod"
    tdis = sim.tdis
    assert tdis is not None and tdis.package_type == "tdis"

    dis_package = flopy.mf6.ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=1,
        delr=100.0,
        delc=100.0,
        top=60.0,
        botm=50.0,
        filename="{}.dis".format(model_name),
        pname="mydispkg",
    )
    # specifying dis package twice with the same name should automatically
    # remove the old dis package
    top = {"filename": "top.bin", "data": 100.0, "binary": True}
    botm = {"filename": "botm.bin", "data": 50.0, "binary": True}
    dis_package = flopy.mf6.ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=top,
        botm=botm,
        filename="{}.dis".format(model_name),
        pname="mydispkg",
    )
    top_data = dis_package.top.get_data()
    assert top_data[0, 0] == 100.0
    ic_package = flopy.mf6.ModflowGwfic(
        model, strt="initial_heads.txt", filename="{}.ic".format(model_name)
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
        budget_filerecord=[("np001_mod.cbc",)],
        head_filerecord=[("np001_mod.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [("HEAD", "ALL"), ("BUDGET", "ALL")],
        },
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)

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
            "filename": "wel0.bin",
            "binary": True,
            "data": [(0, 0, 4, -2000.0), (0, 0, 7, -2.0)],
        },
        1: None,
    }
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=well_spd,
    )
    wel_package.stress_period_data.add_transient_key(1)
    wel_package.stress_period_data.set_data(
        {1: {"filename": "wel.txt", "factor": 1.0}}
    )

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
            "filename": "riv.txt",
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
    assert (
        isinstance(pkg, flopy.mf6.ModflowGwfdis)
        and pkg.package_name == "mydispkg"
    )
    pkg = model.mydispkg
    assert (
        isinstance(pkg, flopy.mf6.ModflowGwfdis)
        and pkg.package_name == "mydispkg"
    )

    # verify external file contents
    array_util = PyListUtil()
    ic_data = ic_package.strt
    ic_array = ic_data.get_data()
    assert array_util.array_comp(
        ic_array,
        [
            [
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ]
        ],
    )

    # make folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file)
        budget_obj = bf.CellBudgetFile(budget_file, precision="double")
        budget_frf_valid = np.array(
            budget_obj.get_data(text="FLOW-JA-FACE", full3D=True)
        )

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "np001_mod.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

        # clean up
        sim.delete_output_files()

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

    # test error checking
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
    found_cellid = False
    with open(os.path.join(mpath, "np001_mod.wel"), "r") as fd:
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

    return


def np002():
    # init paths
    test_ex_name = "np002"
    model_name = "np002_mod"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    pth_for_mf = os.path.join("..", "..", "..", pth)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "np002_mod.hds")
    expected_cbc_file = os.path.join(expected_output_folder, "np002_mod.cbc")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=run_folder,
        nocheck=True,
    )
    name = sim.name_file
    assert name.continue_.get_data() == None
    assert name.nocheck.get_data() == True
    assert name.memory_print_option.get_data() == None

    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="ALL",
        complexity="SIMPLE",
        outer_hclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_hclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    sim.register_ims_package(ims_package, [model.name])

    # get rid of top_data.txt so that a later test does not automatically pass
    top_data_file = os.path.join(run_folder, "top_data.txt")
    if os.path.isfile(top_data_file):
        os.remove(top_data_file)
    # test loading data to be stored in a file and loading data from a file
    # using the "dictionary" input format
    top = {
        "filename": "top_data.txt",
        "factor": 1.0,
        "data": [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    }
    botm_file = os.path.join(pth_for_mf, "botm.txt")
    botm = {"filename": botm_file, "factor": 1.0}
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
        filename="{}.dis".format(model_name),
    )
    ic_vals = [
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
    ]
    ic_package = ModflowGwfic(
        model, strt=ic_vals, filename="{}.ic".format(model_name)
    )
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
    rch_package = ModflowGwfrch(
        model,
        print_input=True,
        print_flows=True,
        maxbound=2,
        stress_period_data=[((0, 0, 3), 0.02), ((0, 0, 6), 0.1)],
    )

    # write simulation to new location
    sim.write_simulation()

    assert os.path.isfile(top_data_file)

    if run:
        # run simulation
        sim.run_simulation()

        sim2 = MFSimulation.load(sim_ws=run_folder)
        model_ = sim2.get_model(model_name)
        npf_package = model_.get_package("npf")
        k = npf_package.k.array

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file)
        budget_obj = bf.CellBudgetFile(budget_file, precision="double")
        budget_frf_valid = np.array(
            budget_obj.get_data(text="FLOW JA FACE    ", full3D=True)
        )

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "np002_mod.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        array_util = PyListUtil()
        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

        # verify external text file was written correctly
        ext_file_path = os.path.join(run_folder, "initial_heads.txt")
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

    return


def test021_twri():
    # init paths
    test_ex_name = "test021_twri"
    model_name = "twri"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "twri.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(86400.0, 1, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="SECONDS", nper=1, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_hclose=0.0001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=0.0001,
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
    pth = os.path.join(sim.simulation_data.mfpath.get_sim_path(), fname)
    f = open(pth, "wb")
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
    top = {
        "factor": 1.0,
        "filename": fname,
        "data": None,
        "binary": True,
        "iprn": 1,
    }

    dis_package = flopy.mf6.ModflowGwfdis(
        model,
        nlay=3,
        nrow=15,
        ncol=15,
        delr=5000.0,
        delc=5000.0,
        top=top,
        botm=[-200, -300, -450],
        filename="{}.dis".format(model_name),
    )
    strt = [
        {"filename": "strt.txt", "factor": 1.0, "data": 0.0},
        {
            "filename": "strt2.bin",
            "factor": 1.0,
            "data": 1.0,
            "binary": "True",
        },
        2.0,
    ]
    ic_package = ModflowGwfic(
        model, strt=strt, filename="{}.ic".format(model_name)
    )
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
    conc = np.ones((15, 15), dtype=np.float) * 35.0
    auxdata = {0: [6, conc]}

    stress_period_data = []
    drn_heads = [0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0]
    for col, head in zip(range(1, 10), drn_heads):
        stress_period_data.append(
            ((0, 7, col), head, 1.0, "name_{}".format(col))
        )
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
        recharge={0: 0.00000003},
        auxiliary=[("iface", "conc")],
        aux=auxdata,
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
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    sim2 = MFSimulation.load(sim_ws=run_folder)
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
    head_file = os.path.join(os.getcwd(), expected_head_file)
    head_new = os.path.join(run_folder, "twri.hds")
    outfile = os.path.join(run_folder, "head_compare.dat")
    assert pymake.compare_heads(
        None, None, files1=head_file, files2=head_new, outfile=outfile
    )

    # clean up
    sim.delete_output_files()

    return


def test005_advgw_tidal():
    # init paths
    test_ex_name = "test005_advgw_tidal"
    model_name = "AdvGW_tidal"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(
        expected_output_folder, "AdvGW_tidal.hds"
    )

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
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
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=4, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_hclose=0.0001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=0.0001,
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
        filename="{}.dis".format(model_name),
    )
    ic_package = ModflowGwfic(
        model, strt=50.0, filename="{}.ic".format(model_name)
    )
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
    ss_template = ModflowGwfsto.ss.empty(
        model, True, layer_storage_types, 0.000001
    )
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
        "filename": "well-rates.ts",
        "timeseries": timeseries,
        "time_series_namerecord": [
            ("well_1_rate", "well_2_rate", "well_3_rate")
        ],
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
            ghb_period_array.append(
                ((layer, row, 9), "tides", cond, "Estuary-L2")
            )
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
            ("ghb-2-6-10", "GHB", (1, 5, 9)),
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
            ("rv-2-9-6", "RIV", (0, 5, 5,)),
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
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_file = os.path.join(os.getcwd(), expected_head_file)
    head_new = os.path.join(run_folder, "AdvGW_tidal.hds")
    outfile = os.path.join(run_folder, "head_compare.dat")
    assert pymake.compare_heads(
        None, None, files1=head_file, files2=head_new, outfile=outfile
    )

    # test rename all
    model.rename_all_packages("new_name")
    assert model.name_file.filename == "new_name.nam"
    package_type_dict = {}
    for package in model.packagelist:
        if not package.package_type in package_type_dict:
            assert package.filename == "new_name.{}".format(
                package.package_type
            )
            package_type_dict[package.package_type] = 1
    sim.write_simulation()
    name_file = os.path.join(run_folder, "new_name.nam")
    assert os.path.exists(name_file)
    dis_file = os.path.join(run_folder, "new_name.dis")
    assert os.path.exists(dis_file)

    sim.rename_all_packages("all_files_same_name")
    package_type_dict = {}
    for package in model.packagelist:
        if not package.package_type in package_type_dict:
            assert package.filename == "all_files_same_name.{}".format(
                package.package_type
            )
            package_type_dict[package.package_type] = 1
    assert sim._tdis_file.filename == "all_files_same_name.tdis"
    for ims_file in sim._ims_files.values():
        assert ims_file.filename == "all_files_same_name.ims"
    sim.write_simulation()
    name_file = os.path.join(run_folder, "all_files_same_name.nam")
    assert os.path.exists(name_file)
    dis_file = os.path.join(run_folder, "all_files_same_name.dis")
    assert os.path.exists(dis_file)
    tdis_file = os.path.join(run_folder, "all_files_same_name.tdis")
    assert os.path.exists(tdis_file)

    # load simulation
    sim_load = MFSimulation.load(
        sim.name,
        "mf6",
        exe_name,
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
    assert found_flows and found_obs

    # clean up
    sim.delete_output_files()

    # check packages
    chk = sim.check()
    summary = ".".join(chk[0].summary_array.desc)
    assert summary == ""

    return


def test004_bcfss():
    # init paths
    test_ex_name = "test004_bcfss"
    model_name = "bcf2ss"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "bcf2ss.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=model_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(1.0, 1, 1.0), (1.0, 1, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="ALL",
        csv_output_filerecord="bcf2ss.ims.csv",
        complexity="SIMPLE",
        outer_hclose=0.000001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=0.000001,
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
        filename="{}.dis".format(model_name),
    )
    ic_package = ModflowGwfic(
        model, strt=0.0, filename="{}.ic".format(model_name)
    )
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
    aux = {0: [[50.0], [1.3]], 1: [[200.0], [1.5]]}
    # aux = {0: [[100.0], [2.3]]}
    rch_package = ModflowGwfrcha(
        model,
        readasarrays=True,
        save_flows=True,
        auxiliary=[("var1", "var2")],
        recharge={0: 0.004},
        aux=aux,
    )  # *** test if aux works ***

    # aux tests
    aux_out = rch_package.aux.get_data()
    assert aux_out[0][0][0, 0] == 50.0
    assert aux_out[0][1][0, 0] == 1.3
    assert aux_out[1][0][0, 0] == 200.0
    assert aux_out[1][1][0, 0] == 1.5

    riv_period = {}
    riv_period_array = []
    for row in range(0, 10):
        riv_period_array.append(((1, row, 14), 0.0, 10000.0, -5.0))
    riv_period[0] = riv_period_array
    riv_package = ModflowGwfriv(
        model,
        save_flows="bcf2ss.cbb",
        maxbound=10,
        stress_period_data=riv_period,
    )

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
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "bcf2ss.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # clean up
        sim.delete_output_files()

    return


def test035_fhb():
    # init paths
    test_ex_name = "test035_fhb"
    model_name = "fhb2015"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(
        expected_output_folder, "fhb2015_fhb.hds"
    )

    # create simulation
    sim = MFSimulation(
        sim_name=model_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(400.0, 10, 1.0), (200.0, 4, 1.0), (400.0, 6, 1.1)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=3, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_hclose=0.001,
        outer_maximum=120,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=0.0001,
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
        filename="{}.dis".format(model_name),
    )
    ic_package = ModflowGwfic(
        model, strt=0.0, filename="{}.ic".format(model_name)
    )
    npf_package = ModflowGwfnpf(
        model, perched=True, icelltype=0, k=20.0, k33=1.0
    )
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
    assert (
        time.steady_state[0] == False
        and time.steady_state[1] == False
        and time.steady_state[2] == False
    )
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

    chd_period = {
        0: [((0, 0, 9), "head"), ((0, 1, 9), "head"), ((0, 2, 9), "head")]
    }
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
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "fhb2015_fhb.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # clean up
        sim.delete_output_files()

    return


def test006_gwf3_disv():
    # init paths
    test_ex_name = "test006_gwf3_disv"
    model_name = "flow"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "flow.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_hclose=0.00000001,
        outer_maximum=1000,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_hclose=0.00000001,
        rcloserecord=0.01,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_package, [model.name])
    vertices = testutils.read_vertices(os.path.join(pth, "vertices.txt"))
    c2drecarray = testutils.read_cell2d(os.path.join(pth, "cell2d.txt"))
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
        filename="{}.disv".format(model_name),
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
    ic_package = ModflowGwfic(
        model, strt=strt_list, filename="{}.ic".format(model_name)
    )
    k = {"filename": "k.bin", "factor": 1.0, "data": 1.0, "binary": "True"}
    npf_package = ModflowGwfnpf(
        model, save_flows=True, icelltype=0, k=k, k33=1.0
    )
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
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "flow.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # export to netcdf - temporarily disabled
        # model.export(os.path.join(run_folder, "test006_gwf3.nc"))
        if shapefile:
            # export to shape file
            model.export(os.path.join(run_folder, "test006_gwf3.shp"))

        # clean up
        sim.delete_output_files()

    return


def test006_2models_gnc():
    # init paths
    test_ex_name = "test006_2models_gnc"
    model_name_1 = "model1"
    model_name_2 = "model2"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_1 = os.path.join(expected_output_folder, "model1.hds")
    expected_head_file_2 = os.path.join(expected_output_folder, "model2.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=tdis_rc
    )
    model_1 = ModflowGwf(
        sim,
        modelname=model_name_1,
        model_nam_file="{}.nam".format(model_name_1),
    )
    model_2 = ModflowGwf(
        sim,
        modelname=model_name_2,
        model_nam_file="{}.nam".format(model_name_2),
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_hclose=0.00000001,
        outer_maximum=1000,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_hclose=0.00000001,
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
        filename="{}.dis".format(model_name_1),
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
        filename="{}.dis".format(model_name_2),
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
    ic_package_1 = ModflowGwfic(
        model_1, strt=strt_list, filename="{}.ic".format(model_name_1)
    )
    ic_package_2 = ModflowGwfic(
        model_2, strt=1.0, filename="{}.ic".format(model_name_2)
    )
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

    gncrecarray = testutils.read_gncrecarray(os.path.join(pth, "gnc.txt"))
    # test gnc delete
    new_gncrecarray = gncrecarray[10:]
    gnc_package = ModflowGwfgnc(
        sim,
        print_input=True,
        print_flows=True,
        numgnc=26,
        numalphaj=1,
        gncdata=new_gncrecarray,
    )
    sim.remove_package(gnc_package.package_type)

    gnc_package = ModflowGwfgnc(
        sim,
        print_input=True,
        print_flows=True,
        numgnc=36,
        numalphaj=1,
        gncdata=gncrecarray,
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
    exg_package = ModflowGwfgwf(
        sim,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary="testaux",
        gnc_filerecord="test006_2models_gnc.gnc",
        nexg=26,
        exchangedata=newexgrecarray,
        exgtype="gwf6-gwf6",
        exgmnamea=model_name_1,
        exgmnameb=model_name_2,
    )
    sim.remove_package(exg_package.package_type)

    exg_package = ModflowGwfgwf(
        sim,
        print_input=True,
        print_flows=True,
        save_flows=True,
        auxiliary="testaux",
        gnc_filerecord="test006_2models_gnc.gnc",
        nexg=36,
        exchangedata=exgrecarray,
        exgtype="gwf6-gwf6",
        exgmnamea=model_name_1,
        exgmnameb=model_name_2,
        observations=gwf_obs,
    )

    # change folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_1)
        head_new = os.path.join(run_folder, "model1.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_2)
        head_new = os.path.join(run_folder, "model2.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # clean up
        sim.delete_output_files()

    return


def test050_circle_island():
    # init paths
    test_ex_name = "test050_circle_island"
    model_name = "ci"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "ci.output.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
    )
    tdis_rc = [(1.0, 1, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=1, perioddata=tdis_rc
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_hclose=0.000001,
        outer_maximum=500,
        under_relaxation="NONE",
        inner_maximum=1000,
        inner_hclose=0.000001,
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
        filename="{}.disv".format(model_name),
    )
    ic_package = ModflowGwfic(
        model, strt=0.0, filename="{}.ic".format(model_name)
    )
    npf_package = ModflowGwfnpf(
        model, save_flows=True, icelltype=0, k=10.0, k33=0.2
    )
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="ci.output.cbc",
        head_filerecord="ci.output.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    stress_period_data = testutils.read_ghbrecarray(
        os.path.join(pth, "ghb.txt"), 2
    )
    ghb_package = ModflowGwfghb(
        model, maxbound=3173, stress_period_data=stress_period_data
    )

    rch_data = ["OPEN/CLOSE", "rech.dat", "FACTOR", 1.0, "IPRN", 0]
    rch_package = ModflowGwfrcha(
        model, readasarrays=True, save_flows=True, recharge=rch_data
    )

    # change folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # run simulation
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "ci.output.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # clean up
        sim.delete_output_files()

    return


def test028_sfr():
    # init paths
    test_ex_name = "test028_sfr"
    model_name = "test1tr"

    pth = os.path.join(
        "..", "examples", "data", "mf6", "create_tests", test_ex_name
    )
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file = os.path.join(expected_output_folder, "test1tr.hds")

    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name, version="mf6", exe_name=exe_name, sim_ws=pth
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
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )
    model.name_file.save_flows.set_data(True)
    ims_package = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_hclose=0.00001,
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
        inner_hclose=0.00001,
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
        filename="{}.dis".format(model_name),
    )
    strt = testutils.read_std_array(os.path.join(pth, "strt.txt"), "float")
    strt_int = ["internal", "factor", 1.0, "iprn", 0, strt]
    ic_package = ModflowGwfic(
        model, strt=strt_int, filename="{}.ic".format(model_name)
    )

    k_vals = testutils.read_std_array(os.path.join(pth, "k.txt"), "float")
    k = ["internal", "factor", 3.000e-03, "iprn", 0, k_vals]
    npf_package = ModflowGwfnpf(model, icelltype=1, k=k, k33=1.0)
    npf_package.k.factor = 2.000e-04

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord="test1tr.cbc",
        head_filerecord="test1tr.hds",
        saverecord={0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]},
        printrecord={
            0: [("HEAD", "FREQUENCY", 5), ("BUDGET", "FREQUENCY", 5)]
        },
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

    stress_period_data = {
        0: [((0, 12, 0), 988.0, 0.038), ((0, 13, 8), 1045.0, 0.038)]
    }
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

    sim.simulation_data.mfpath.set_sim_path(run_folder)
    sim.write_simulation()
    sim.load(
        sim_name=test_ex_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=run_folder,
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
    if run:
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, "test1tr.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None, None, files1=head_file, files2=head_new, outfile=outfile
        )

        # clean up
        sim.delete_output_files()

    return


if __name__ == "__main__":
    np001()
    np002()
    test004_bcfss()
    test005_advgw_tidal()
    test006_2models_gnc()
    test006_gwf3_disv()
    test021_twri()
    test028_sfr()
    test035_fhb()
    test050_circle_island()
