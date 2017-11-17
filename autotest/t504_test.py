import os
import flopy
import platform
from flopy.mf6.mfsimulation import MFSimulation
from flopy.mf6.data.mfdatautil import ArrayUtil
import flopy.utils.binaryfile as bf
import numpy as np
try:
    import pymake
except:
    print('could not import pymake')

#exe_name = 'mf6'
exe_name = 'C:\\WrdApp\\mf6.0.1\\bin\\mf6'
if platform.system() == 'Windows':
    exe_name += '.exe'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False

cpth = os.path.join('temp', 't504')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)


def test001a_tharmonic():
    # init paths
    test_ex_name = 'test001a_Tharmonic'
    model_name = 'flow15'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'flow15_flow_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'flow15_flow_adj.hds')
    expected_cbc_file_a = os.path.join(expected_output_folder, 'flow15_flow_unch.cbc')
    expected_cbc_file_b = os.path.join(expected_output_folder, 'flow15_flow_adj.cbc')

    array_util = ArrayUtil()

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_a)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_obj.list_records()
        budget_frf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'flow15_flow.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

    # change some settings
    hk_data = sim.simulation_data.mfdata[(model_name, 'npf', 'griddata', 'k')]
    hk_array = hk_data.get_data()
    hk_array[0,0,1] = 20.0
    hk_data.set_data(hk_array)

    model = sim.get_model(model_name)
    ic = model.get_package('ic')
    ic_data = ic.strt
    ic_array = ic_data.get_data()
    ic_array[0,0,0] = 1.0
    ic_array[0,0,9] = 1.0
    ic_data.set_data(ic_array)

    get_test = hk_data[0,0,0]
    assert(get_test == 10.0)
    get_test = hk_data.array
    assert(array_util.array_comp(get_test, [[10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]))
    get_test = hk_data[:]
    assert(array_util.array_comp(get_test, [[[10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]]))

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_b)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_frf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'flow15_flow.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

        # clean up
        sim.delete_output_files()

    return


def test003_gwfs_disv():
    # init paths
    test_ex_name = 'test003_gwfs_disv'
    model_name = 'gwf_1'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'model_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'model_adj.hds')
    expected_cbc_file_a = os.path.join(expected_output_folder, 'model_unch.cbc')
    expected_cbc_file_b = os.path.join(expected_output_folder, 'model_adj.cbc')

    array_util = ArrayUtil()

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.simulation_data.max_columns_of_data = 10
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_a)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_fjf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))

        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'model.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
        assert array_util.array_comp(budget_fjf_valid, budget_frf)

    # change some settings
    model = sim.get_model(model_name)
    chd_head_left = model.get_package('CHD_LEFT')
    chd_left_period = chd_head_left.periodrecarray.array
    chd_left_period[4][1] = 15.0

    chd_head_right = model.get_package('CHD_RIGHT')
    chd_right_period = chd_head_right.periodrecarray
    chd_right_data = chd_right_period.get_data(0)
    chd_right_data_slice = chd_right_data[3:10]
    chd_right_period.set_data(chd_right_data_slice, 0)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_b)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_fjf_valid = np.array(budget_obj.get_data(text='FLOW JA FACE', full3D=True))

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'model.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
        assert array_util.array_comp(budget_fjf_valid, budget_frf)

        # clean up
        sim.delete_output_files()

    return

def test005_advgw_tidal():
    # init paths
    test_ex_name = 'test005_advgw_tidal'
    model_name = 'gwf_1'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'AdvGW_tidal_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'AdvGW_tidal_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'AdvGW_tidal.hds')
        outfile = os.path.join(run_folder, 'head_compare.dat')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)

    # change some settings
    """
    hydchr = sim.simulation_data.mfdata[(model_name, 'HFB8', 'PERIOD', 'hydchr')]
    hydchr[2] = 0.000002
    hydchr[3] = 0.000003
    hydchr[4] = 0.0000004
    cond = sim.simulation_data.mfdata[(model_name, 'DRN8_1', 'PERIOD', 'cond')]
    for index in range(0, len(cond)):
        cond[index] = 2.1

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_obj = bf.HeadFile(head_file, precision='double')
        head_valid = np.array(head_obj.get_alldata())

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'AdvGW_tidal.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)
    """

def test006_gwf3():
    from flopy.mf6.utils.binaryfile_utils import _reshape_binary_data

    # init paths
    test_ex_name = 'test006_gwf3'
    model_name = 'gwf_1'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'flow_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'flow_adj.hds')
    expected_cbc_file_a = os.path.join(expected_output_folder, 'flow_unch.cbc')
    expected_cbc_file_b = os.path.join(expected_output_folder, 'flow_adj.cbc')

    array_util = ArrayUtil()

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)
    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        budget_file = os.path.join(os.getcwd(), expected_cbc_file_a)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_fjf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'flow.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_fjf = np.array(sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')])
        assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

    # change some settings
    model = sim.get_model(model_name)
    hk = model.get_package('npf').k
    hk_data = hk.get_data()
    hk_data[2] = 3.5
    hk.set_data(hk_data)
    ex_happened = False
    try:
        hk.make_layered()
    except:
        ex_happened = True
    assert(ex_happened)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_b)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_fjf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'flow.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_fjf = np.array(sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')])
        assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

    # confirm that files did move
    save_folder = os.path.join(run_folder, 'temp_two')
    sim.simulation_data.mfpath.set_sim_path(save_folder)

    # write with "copy_external_files" turned off so external files do not get copied to new location
    sim.write_simulation(ext_file_action=flopy.mf6.mfbase.ExtFileAction.copy_none)

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file_b)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_fjf_valid = np.array(budget_obj.get_data(text='    FLOW JA FACE', full3D=True))
        jaentries = budget_fjf_valid.shape[-1]
        budget_fjf_valid.shape = (-1, jaentries)

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'flow.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_fjf = np.array(sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')])
        assert array_util.array_comp(np.array(budget_fjf_valid), np.array(budget_fjf))

        # confirm that files did not move
        assert not os.path.isfile(os.path.join(save_folder, 'flow.disu.ja.dat'))
        assert not os.path.isfile(os.path.join(save_folder, 'flow.disu.iac.dat'))
        assert not os.path.isfile(os.path.join(save_folder, 'flow.disu.cl12.dat'))
        assert not os.path.isfile(os.path.join(save_folder, 'flow.disu.area.dat'))
        assert not os.path.isfile(os.path.join(save_folder, 'flow.disu.hwva.dat'))

        # clean up
        sim.delete_output_files()

    return


def test045_lake1ss_table():
    # init paths
    test_ex_name = 'test045_lake1ss_table'
    model_name = 'lakeex1b'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'lakeex1b_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'lakeex1b_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'lakeex1b.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

    # change some settings
    model = sim.get_model(model_name)
    laktbl = model.get_package('tab').laktabrecarray
    laktbl_data = laktbl.get_data()
    laktbl_data[-1][0] = 700.0
    laktbl.set_data(laktbl_data)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'lakeex1b.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        # clean up
        sim.delete_output_files()

    return


def test006_2models_mvr():
    # init paths
    test_ex_name = 'test006_2models_mvr'
    sim_name = 'test006_2models_mvr'
    model_names = ['parent', 'child']

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'model1_unch.hds')
    expected_head_file_aa = os.path.join(expected_output_folder, 'model2_unch.hds')
    expected_cbc_file_a = os.path.join(expected_output_folder, 'model1_unch.cbc')

    expected_head_file_b = os.path.join(expected_output_folder, 'model1_adj.hds')
    expected_head_file_bb = os.path.join(expected_output_folder, 'model2_adj.hds')

    # load simulation
    sim = MFSimulation.load(sim_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'model1.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        head_file = os.path.join(os.getcwd(), expected_head_file_aa)
        head_new = os.path.join(run_folder, 'model2.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        budget_file = os.path.join(os.getcwd(), expected_cbc_file_a)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_obj.list_records()

    # change some settings
    parent_model = sim.get_model(model_names[0])
    maw_pkg = parent_model.get_package('maw')
    period_data = maw_pkg.wellperiodrecarray.get_data()
    period_data[0][2] = -1.0
    maw_pkg.wellperiodrecarray.set_data(period_data, 0)

    exg_pkg = sim.get_exchange_file('simulation.exg')
    exg_data = exg_pkg.gwfgwfrecarray.get_data()
    for index in range(0, len(exg_data)):
        exg_data[index][6] = 500.0
    exg_pkg.gwfgwfrecarray.set_data(exg_data)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'model1.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        head_file = os.path.join(os.getcwd(), expected_head_file_bb)
        head_new = os.path.join(save_folder, 'model2.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

        # clean up
        sim.delete_output_files()

    return


def test001e_uzf_3lay():
    # init paths
    test_ex_name = 'test001e_UZF_3lay'
    model_name = 'gwf_1'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'test001e_UZF_3lay_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'test001e_UZF_3lay_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'test001e_UZF_3lay.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

    # change some settings
    model = sim.get_model(model_name)
    uzf = model.get_package('uzf')
    uzf_data = uzf.uzfrecarray
    uzf_array = uzf_data.get_data()
    # increase initial water content
    for index in range(0, len(uzf_array)):
        uzf_array[index][7] = 0.3
    uzf_data.set_data(uzf_array)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'test001e_UZF_3lay.hds')
        outfile = os.path.join(save_folder, 'head_compare.dat')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)


def test045_lake2tr():
    # init paths
    test_ex_name = 'test045_lake2tr'
    model_name = 'lakeex2a'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'lakeex2a_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'lakeex2a_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # write simulation to new location
    sim.simulation_data.mfpath.set_sim_path(run_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'lakeex2a.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

    # change some settings
    model = sim.get_model(model_name)
    evt = model.get_package('evt')
    evt.rate.set_data([0.05], key=0)

    lak = model.get_package('lak')
    lak_period = lak.lakeperiodrecarray
    lak_period_data = lak_period.get_data()
    lak_period_data[2][2] = '0.05'
    lak_period.set_data(lak_period_data, 0)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'lakeex2a.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)


def test036_twrihfb():
    # init paths
    test_ex_name = 'test036_twrihfb'
    model_name = 'twrihfb2015'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'twrihfb2015_output_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'twrihfb2015_output_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'twrihfb2015_output.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)

    # change some settings
    hydchr = sim.simulation_data.mfdata[(model_name, 'hfb', 'period', 'hfbrecarray')]
    hydchr_data = hydchr.get_data()
    hydchr_data[2][2] = 0.000002
    hydchr_data[3][2] = 0.000003
    hydchr_data[4][2] = 0.0000004
    hydchr.set_data(hydchr_data, 0)
    cond = sim.simulation_data.mfdata[(model_name, 'drn', 'period', 'periodrecarray')]
    cond_data = cond.get_data()
    for index in range(0, len(cond_data)):
        cond_data[index][2] = 2.1
    cond.set_data(cond_data, 0)

    rch = sim.simulation_data.mfdata[(model_name, 'rcha', 'period', 'recharge')]
    rch_data = rch.get_data()
    assert(rch_data[5][1] == 0.00000003)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'twrihfb2015_output.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)


def test027_timeseriestest():
    # init paths
    test_ex_name = 'test027_TimeseriesTest'
    model_name = 'gwf_1'

    pth = os.path.join('..', 'examples', 'data', 'mf6', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    save_folder = os.path.join(run_folder, 'temp')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file_a = os.path.join(expected_output_folder, 'timeseriestest_unch.hds')
    expected_head_file_b = os.path.join(expected_output_folder, 'timeseriestest_adj.hds')

    # load simulation
    sim = MFSimulation.load(model_name, 'mf6', exe_name, pth)

    # make temp folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_a)
        head_new = os.path.join(run_folder, 'timeseriestest.hds')
        outfile = os.path.join(run_folder, 'head_compare.dat')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)

    model = sim.get_model(model_name)
    rch = model.get_package('rcha')
    tas_rch = rch.get_package('tas')
    tas_array_data = tas_rch.tas_array.get_data(12.0)
    assert tas_array_data == 0.0003
    tas_array_data = 0.02
    tas_rch.tas_array.set_data(tas_array_data, key=12.0)

    # write simulation again
    sim.simulation_data.mfpath.set_sim_path(save_folder)
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file_b)
        head_new = os.path.join(save_folder, 'timeseriestest.hds')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new)


if __name__ == '__main__':
    test027_timeseriestest()
    test036_twrihfb()
    test045_lake2tr()
    test001e_uzf_3lay()
    test006_2models_mvr()
    test045_lake1ss_table()
    test001a_tharmonic()
    test003_gwfs_disv()
    test005_advgw_tidal()
    test006_gwf3()