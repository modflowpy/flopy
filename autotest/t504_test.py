import os, copy

import numpy as np

import flopy
import flopy.utils.binaryfile as bf
from flopy.utils.datautil import PyListUtil
from flopy.mf6.modflow.mfsimulation import MFSimulation
from ci_framework import baseTestDir, flopyTest

try:
    import shapefile

    if int(shapefile.__version__.split(".")[0]) < 2:
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

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def test001a_tharmonic():

    # init paths
    test_ex_name = "test001a_Tharmonic"
    model_name = "flow15"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "flow15_flow_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "flow15_flow_adj.hds"
    )
    expected_cbc_file_a = os.path.join(
        expected_output_folder, "flow15_flow_unch.cbc"
    )
    expected_cbc_file_b = os.path.join(
        expected_output_folder, "flow15_flow_adj.cbc"
    )

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(
        model_name,
        "mf6",
        exe_name,
        pth,
        verbosity_level=0,
        verify_data=True,
        write_headers=False,
    )
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external(external_data_folder="data")
    sim.write_simulation(silent=True)
    # verify external data written to correct location
    data_path = os.path.join(run_folder, "data", "flow15.dis_botm.txt")
    assert os.path.exists(data_path)
    # model export test
    model = sim.get_model(model_name)
    model.export(f"{model.model_ws}/tharmonic.nc")
    model.export(f"{model.model_ws}/tharmonic.shp")
    model.dis.botm.export(f"{model.model_ws}/botm.shp")

    mg = model.modelgrid

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        print(sim.name)
        assert success, f"simulation {sim.name} did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(expected_cbc_file_a, precision="auto")
        budget_obj.list_records()
        budget_frf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )

        # compare output to expected results
        head_new = os.path.join(run_folder, "flow15_flow.hds")
        assert pymake.compare_heads(
            None, None, files1=expected_head_file_a, files2=head_new
        )

        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(expected_cbc_file_b, precision="auto")
        budget_frf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )

        # compare output to expected results
        head_new = os.path.join(save_folder, "flow15_flow.hds")
        assert pymake.compare_heads(
            None, None, files1=expected_head_file_b, files2=head_new
        )

        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

    return


def test003_gwfs_disv():
    # init paths
    test_ex_name = "test003_gwfs_disv"
    model_name = "gwf_1"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "model_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "model_adj.hds"
    )
    expected_cbc_file_a = os.path.join(
        expected_output_folder, "model_unch.cbc"
    )
    expected_cbc_file_b = os.path.join(expected_output_folder, "model_adj.cbc")

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.simulation_data.max_columns_of_data = 10
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(expected_cbc_file_a, precision="auto")
        budget_fjf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )

        head_new = os.path.join(run_folder, "model.hds")
        assert pymake.compare_heads(
            None, None, files1=expected_head_file_a, files2=head_new
        )

        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
        assert array_util.array_comp(budget_fjf_valid, budget_frf)

    model = sim.get_model(model_name)
    if shapefile:
        model.export(f"{pth}/{test_ex_name}.shp")

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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(expected_cbc_file_b, precision="double")
        budget_fjf_valid = np.array(
            budget_obj.get_data(text="FLOW JA FACE", full3D=True)
        )

        # compare output to expected results
        head_new = os.path.join(save_folder, "model.hds")
        assert pymake.compare_heads(
            None, None, files1=expected_head_file_b, files2=head_new
        )

        budget_frf = sim.simulation_data.mfdata[
            (model_name, "CBC", "FLOW-JA-FACE")
        ]
        assert array_util.array_comp(budget_fjf_valid, budget_frf)

    return


def test005_advgw_tidal():
    # init paths
    test_ex_name = "test005_advgw_tidal"
    model_name = "gwf_1"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "AdvGW_tidal_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "AdvGW_tidal_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(
        model_name, "mf6", exe_name, pth, verbosity_level=2, verify_data=True
    )

    # test obs/ts package interface
    model = sim.get_model(model_name)
    time = model.modeltime
    assert (
        time.steady_state[0] == True
        and time.steady_state[1] == False
        and time.steady_state[2] == False
        and time.steady_state[3] == False
    )
    ghb = model.get_package("ghb")
    obs = ghb.obs
    digits = obs.digits.get_data()
    assert digits == 10
    names = ghb.ts.time_series_namerecord.get_data()
    assert names[0][0] == "tides"

    # add a stress period beyond nper
    spd = ghb.stress_period_data.get_data()
    spd[20] = copy.deepcopy(spd[0])
    ghb.stress_period_data.set_data(spd)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # compare output to expected results
        head_new = os.path.join(run_folder, "advgw_tidal.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
            outfile=outfile,
        )


def test006_gwf3():
    # init paths
    test_ex_name = "test006_gwf3"
    model_name = "gwf_1"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "flow_unch.hds"
    )
    expected_head_file_b = os.path.join(expected_output_folder, "flow_adj.hds")
    expected_cbc_file_a = os.path.join(expected_output_folder, "flow_unch.cbc")
    expected_cbc_file_b = os.path.join(expected_output_folder, "flow_adj.cbc")

    array_util = PyListUtil()

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

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
    sim.set_sim_path(run_folder)
    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        budget_obj = bf.CellBudgetFile(expected_cbc_file_a, precision="double")
        budget_fjf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_new = os.path.join(run_folder, "flow.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
        )

        budget_fjf = np.array(
            sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
        )
        assert array_util.array_comp(
            np.array(budget_fjf_valid), np.array(budget_fjf)
        )

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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun(2) did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(expected_cbc_file_b, precision="auto")
        budget_fjf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_new = os.path.join(save_folder, "flow.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
        )

        budget_fjf = np.array(
            sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
        )
        assert array_util.array_comp(
            np.array(budget_fjf_valid), np.array(budget_fjf)
        )

    # confirm that files did move
    save_folder = f"{run_folder}_save02"
    testFramework.addTestDir(save_folder, create=True)

    sim.set_sim_path(save_folder)

    # write with "copy_external_files" turned off so external files do not get copied to new location
    sim.write_simulation(
        ext_file_action=flopy.mf6.mfbase.ExtFileAction.copy_none
    )

    # store strt in an external binary file
    model = sim.get_model()
    ic = model.get_package("ic")
    ic.strt.store_as_external_file("initial_heads.bin", binary=True)

    strt_data = ic.strt.array
    # update packages
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun(3) did not run"

        # get expected results
        budget_obj = bf.CellBudgetFile(
            expected_cbc_file_b,
            precision="double",
        )
        budget_fjf_valid = np.array(
            budget_obj.get_data(text="    FLOW JA FACE", full3D=True)
        )
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_new = os.path.join(save_folder, "flow.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
        )

        budget_fjf = np.array(
            sim.simulation_data.mfdata[(model_name, "CBC", "FLOW-JA-FACE")]
        )
        assert array_util.array_comp(
            np.array(budget_fjf_valid), np.array(budget_fjf)
        )

        # confirm that files did not move
        assert not os.path.isfile(
            os.path.join(save_folder, "flow.disu.ja.dat")
        )
        assert not os.path.isfile(
            os.path.join(save_folder, "flow.disu.iac.dat")
        )
        assert not os.path.isfile(
            os.path.join(save_folder, "flow.disu.cl12.dat")
        )
        assert not os.path.isfile(
            os.path.join(save_folder, "flow.disu.area.dat")
        )
        assert not os.path.isfile(
            os.path.join(save_folder, "flow.disu.hwva.dat")
        )
        # confirm external binary file was created
        assert os.path.isfile(os.path.join(save_folder, "initial_heads.bin"))

        # clean up
        sim.delete_output_files()

    return


# def test045_lake1ss_table():
#     # init paths
#     test_ex_name = "test045_lake1ss_table"
#     model_name = "lakeex1b"
#
#     pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
#     run_folder = f"{baseDir}_{test_ex_name}"
#     save_folder = f"{run_folder}_save"
#     testFramework = flopyTest(verbose=True)
#     testFramework.addTestDir([run_folder, save_folder], create=True)
#
#     expected_output_folder = os.path.join(pth, "expected_output")
#     expected_head_file_a = os.path.join(
#         expected_output_folder, "lakeex1b_unch.hds"
#     )
#     expected_head_file_b = os.path.join(
#         expected_output_folder, "lakeex1b_adj.hds"
#     )
#
#     # load simulation
#     sim = MFSimulation.load(
#         sim_name=model_name,
#         exe_name=exe_name,
#         sim_ws=pth,
#         verify_data=True,
#     )
#
#     # make temp folder to save simulation
#     sim.set_sim_path(run_folder)
#
#     # write simulation to new location
#     sim.write_simulation()
#
#     if run:
#         # run simulation
#         success, buff = sim.run_simulation()
#         assert success, f"simulation {sim.name} did not run"
#
#         # compare output to expected results
#         head_new = os.path.join(run_folder, "lakeex1b.hds")
#         outfile = os.path.join(run_folder, "headcompare_a.txt")
#         success = pymake.compare_heads(
#             None,
#             None,
#             files1=expected_head_file_a,
#             files2=head_new,
#             outfile=outfile,
#         )
#         assert success
#
#     # change some settings
#     model = sim.get_model(model_name)
#     laktbl = model.get_package("tab").table
#     laktbl_data = laktbl.get_data()
#     laktbl_data[-1][0] = 700.0
#     laktbl.set_data(laktbl_data)
#
#     # write simulation again
#     sim.set_sim_path(save_folder)
#     sim.write_simulation()
#
#     if run:
#         # run simulation
#         success, buff = sim.run_simulation()
#         assert success, f"simulation {sim.name} rerun did not run"
#
#         # compare output to expected results
#         head_new = os.path.join(save_folder, "lakeex1b.hds")
#         outfile = os.path.join(run_folder, "headcompare_b.txt")
#         success = pymake.compare_heads(
#             None,
#             None,
#             files1=expected_head_file_b,
#             files2=head_new,
#             outfile=outfile,
#         )
#         assert success
#
#
#
#     return


def test006_2models_mvr():
    # init paths
    test_ex_name = "test006_2models_mvr"
    sim_name = "test006_2models_mvr"
    model_names = ["parent", "child"]

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "model1_unch.hds"
    )
    expected_head_file_aa = os.path.join(
        expected_output_folder, "model2_unch.hds"
    )
    expected_cbc_file_a = os.path.join(
        expected_output_folder, "model1_unch.cbc"
    )

    expected_head_file_b = os.path.join(
        expected_output_folder, "model1_adj.hds"
    )
    expected_head_file_bb = os.path.join(
        expected_output_folder, "model2_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(sim_name, "mf6", exe_name, pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # compare output to expected results
        head_new = os.path.join(run_folder, "model1.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
        )

        head_new = os.path.join(run_folder, "model2.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_aa,
            files2=head_new,
        )

        budget_obj = bf.CellBudgetFile(
            expected_cbc_file_a,
            precision="double",
        )
        budget_obj.list_records()

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
        assert model.model_type == "gwf"
    models = sim.gwf
    assert len(models) == 2
    for model in models:
        assert model.name in model_names
        assert model.model_type == "gwf"

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
    pkg_dict = parent_model.package_dict
    assert len(pkg_dict) == 6
    pkg_names = parent_model.package_names
    assert len(pkg_names) == 6
    # confirm that this is a copy of the original dictionary with references
    # to the packages
    del pkg_dict[pkg_names[0]]
    assert len(pkg_dict) == 5
    pkg_dict = parent_model.package_dict
    assert len(pkg_dict) == 6

    old_val = pkg_dict["dis"].nlay.get_data()
    pkg_dict["dis"].nlay = 22
    pkg_dict = parent_model.package_dict
    assert pkg_dict["dis"].nlay.get_data() == 22
    pkg_dict["dis"].nlay = old_val

    # write simulation again
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # compare output to expected results
        head_new = os.path.join(save_folder, "model1.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
        )

        head_new = os.path.join(save_folder, "model2.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_bb,
            files2=head_new,
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
            sim_name, "mf6", exe_name, pth, load_only=load_only
        )
        for model_name in model_names:
            model = sim.get_model(model_name)
            for package in model_package_check:
                assert (
                    package in model.package_type_dict
                    or package in sim.package_type_dict
                ) == (package in load_only or f"{package}6" in load_only)
        assert (len(sim._exchange_files) > 0) == (
            "gwf6-gwf6" in load_only or "gwf-gwf" in load_only
        )
        assert (len(sim._ims_files) > 0) == (
            "ims6" in load_only or "ims" in load_only
        )

    # load package by name
    load_only_list = ["ic6", "maw", "npf_p1", "oc_p2", "ims"]
    sim = MFSimulation.load(
        sim_name, "mf6", exe_name, pth, load_only=load_only_list
    )
    model_parent = sim.get_model("parent")
    model_child = sim.get_model("child")
    assert "oc" not in model_parent.package_type_dict
    assert "oc" in model_child.package_type_dict
    assert "npf" in model_parent.package_type_dict
    assert "npf" not in model_child.package_type_dict

    if run:
        # test running a runnable load_only case
        sim = MFSimulation.load(
            sim_name, "mf6", exe_name, pth, load_only=load_only_lists[0]
        )
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

    return


def test001e_uzf_3lay():
    # init paths
    test_ex_name = "test001e_UZF_3lay"
    model_name = "gwf_1"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

    # test load_only
    model_package_check = ["chd", "ic", "npf", "oc", "sto", "uzf"]
    load_only_lists = [
        ["chd6", "ic6", "ims", "npf6", "obs", "oc", "sto"],
        ["chd6", "ims", "npf6", "obs", "oc", "sto", "uzf6"],
        ["chd", "ic", "npf", "obs", "sto"],
        ["ic6", "ims", "obs6", "oc6"],
    ]
    for load_only in load_only_lists:
        sim = MFSimulation.load(
            model_name, "mf6", exe_name, pth, load_only=load_only
        )
        model = sim.get_model()
        for package in model_package_check:
            assert (package in model.package_type_dict) == (
                package in load_only or f"{package}6" in load_only
            )
    if run:
        # test running a runnable load_only case
        sim = MFSimulation.load(
            model_name, "mf6", exe_name, pth, load_only=load_only_lists[0]
        )
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} from load did not run"

    if run:
        eval_cbc_precision()
        eval_replace_ims_package()


def test045_lake2tr():
    # init paths
    test_ex_name = "test045_lake2tr"
    model_name = "lakeex2a"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "lakeex2a_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "lakeex2a_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

    # write simulation to new location
    sim.set_sim_path(run_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # compare output to expected results
        head_new = os.path.join(run_folder, "lakeex2a.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # compare output to expected results
        head_new = os.path.join(save_folder, "lakeex2a.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
            htol=10.0,
        )


def test036_twrihfb():
    # init paths
    test_ex_name = "test036_twrihfb"
    model_name = "twrihfb2015"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "twrihfb2015_output_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "twrihfb2015_output_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # compare output to expected results
        head_new = os.path.join(run_folder, "twrihfb2015_output.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
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

    rch = sim.simulation_data.mfdata[
        (model_name, "rcha", "period", "recharge")
    ]
    rch_data = rch.get_data()
    assert rch_data[0][5, 1] == 0.00000003

    # write simulation again
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # compare output to expected results
        head_new = os.path.join(save_folder, "twrihfb2015_output.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
        )


def test027_timeseriestest():
    # init paths
    test_ex_name = "test027_TimeseriesTest"
    model_name = "gwf_1"

    pth = os.path.join("..", "examples", "data", "mf6", test_ex_name)
    run_folder = f"{baseDir}_{test_ex_name}"
    save_folder = f"{run_folder}_save"
    testFramework = flopyTest(verbose=True)
    testFramework.addTestDir([run_folder, save_folder], create=True)

    expected_output_folder = os.path.join(pth, "expected_output")
    expected_head_file_a = os.path.join(
        expected_output_folder, "timeseriestest_unch.hds"
    )
    expected_head_file_b = os.path.join(
        expected_output_folder, "timeseriestest_adj.hds"
    )

    # load simulation
    sim = MFSimulation.load(model_name, "mf6", exe_name, pth, verify_data=True)

    # make temp folder to save simulation
    sim.set_sim_path(run_folder)

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # reload sim
    sim = MFSimulation.load(
        model_name, "mf6", exe_name, run_folder, verify_data=True
    )
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        # compare output to expected results
        head_new = os.path.join(run_folder, "timeseriestest.hds")
        outfile = os.path.join(run_folder, "head_compare.dat")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_a,
            files2=head_new,
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
    sim.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} rerun did not run"

        # compare output to expected results
        head_new = os.path.join(save_folder, "timeseriestest.hds")
        assert pymake.compare_heads(
            None,
            None,
            files1=expected_head_file_b,
            files2=head_new,
            htol=10.0,
        )


def eval_cbc_precision():
    pth = os.path.join(
        f"{baseDir}_test001e_UZF_3lay", "test001e_UZF_3lay.uzf.cbc"
    )
    cbc = flopy.utils.CellBudgetFile(pth, precision="auto")
    data = cbc.get_data(text="GWF", full3D=False)
    if data[2].node[0] != 1:
        raise AssertionError("Budget precision error for imeth 6")


def eval_replace_ims_package():
    pth = f"{baseDir}_test001e_UZF_3lay"
    sim = flopy.mf6.MFSimulation.load("mfsim", sim_ws=pth, exe_name=exe_name)

    ims = sim.ims
    sim.remove_package(ims)

    ims = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", complexity="COMPLEX"
    )
    sim.register_ims_package(
        ims,
        [
            "GwF_1",
        ],
    )
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"


def test_mf6_output():
    ex_name = "test001e_UZF_3lay"
    sim_ws = os.path.join("..", "examples", "data", "mf6", ex_name)
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name=exe_name)

    ws = f"{baseDir}_{ex_name}_mf6_output"
    testFramework = flopyTest(verbose=True, testDirs=ws, create=True)

    sim.set_sim_path(ws)
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    ml = sim.get_model("gwf_1")

    bud = ml.oc.output.budget()
    budcsv = ml.oc.output.budgetcsv()
    hds = ml.oc.output.head()
    lst = ml.oc.output.list()

    idomain = np.ones(ml.modelgrid.shape, dtype=int)
    zonbud = ml.oc.output.zonebudget(idomain)

    if not isinstance(bud, flopy.utils.CellBudgetFile):
        raise TypeError()

    if not isinstance(budcsv, flopy.utils.observationfile.CsvFile):
        raise TypeError()

    if not isinstance(hds, flopy.utils.HeadFile):
        raise TypeError()

    if not isinstance(zonbud, flopy.utils.ZoneBudget6):
        raise AssertionError()

    if not isinstance(lst, flopy.utils.Mf6ListBudget):
        raise AssertionError()

    bud = ml.output.budget()
    budcsv = ml.output.budgetcsv()
    hds = ml.output.head()
    zonbud = ml.output.zonebudget(idomain)
    lst = ml.output.list()

    if not isinstance(bud, flopy.utils.CellBudgetFile):
        raise TypeError()

    if not isinstance(budcsv, flopy.utils.observationfile.CsvFile):
        raise TypeError()

    if not isinstance(hds, flopy.utils.HeadFile):
        raise TypeError()

    if not isinstance(zonbud, flopy.utils.ZoneBudget6):
        raise TypeError()

    if not isinstance(lst, flopy.utils.Mf6ListBudget):
        raise TypeError()

    uzf = ml.uzf
    uzf_bud = uzf.output.budget()
    uzf_budcsv = uzf.output.budgetcsv()
    conv = uzf.output.package_convergence()
    uzf_obs = uzf.output.obs()
    uzf_zonbud = uzf.output.zonebudget(idomain)

    if not isinstance(uzf_bud, flopy.utils.CellBudgetFile):
        raise TypeError()

    if not isinstance(uzf_budcsv, flopy.utils.observationfile.CsvFile):
        raise TypeError()

    if conv is not None:
        if not isinstance(conv, flopy.utils.observationfile.CsvFile):
            raise TypeError()

    if not isinstance(uzf_obs, flopy.utils.Mf6Obs):
        raise TypeError()

    if not isinstance(uzf_zonbud, flopy.utils.ZoneBudget6):
        raise TypeError()

    if ml.dis.output.methods() is not None:
        raise AssertionError()


def test_mf6_output_add_observation():
    model_name = "lakeex2a"
    sim_ws = os.path.join("..", "examples", "data", "mf6", "test045_lake2tr")
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name=exe_name)
    gwf = sim.get_model(model_name)

    # remove sfr_obs and add a new sfr obs
    sfr = gwf.sfr

    obs_file = f"{model_name}.sfr.obs"
    csv_file = f"{obs_file}.csv"
    obs_dict = {
        csv_file: [
            ("l08_stage", "stage", (8,)),
            ("l09_stage", "stage", (9,)),
            ("l14_stage", "stage", (14,)),
            ("l15_stage", "stage", (15,)),
        ]
    }
    gwf.sfr.obs.initialize(
        filename=obs_file, digits=10, print_input=True, continuous=obs_dict
    )

    ws = f"{baseDir}_test045_lake2tr_obs"
    testFramework = flopyTest(verbose=True, testDirs=ws, create=True)

    sim.set_sim_path(ws)
    sim.write_simulation()

    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # check that .output finds the newly added OBS package
    sfr_obs = gwf.sfr.output.obs()

    if not isinstance(sfr_obs, flopy.utils.Mf6Obs):
        raise TypeError("remove and add observation test (Mf6Output) failed")


if __name__ == "__main__":
    test001a_tharmonic()
    test001e_uzf_3lay()
    test003_gwfs_disv()
    test005_advgw_tidal()
    test006_2models_mvr()
    test006_gwf3()
    test027_timeseriestest()
    test036_twrihfb()
    test045_lake2tr()
    test_mf6_output()
    test_mf6_output_add_observation()
