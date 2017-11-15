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

exe_name = 'mf6'
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


def test001a_Tharmonic():
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
    sim = MFSimulation.load(model_name, 'mf6', 'mf6.exe', pth)
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
    sim = MFSimulation.load(model_name, 'mf6', 'mf6.exe', pth)

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

if __name__ == '__main__':
    test001a_Tharmonic()
    test003_gwfs_disv()