import os
import flopy
import platform
import numpy as np
import os
from flopy.mf6.mfsimulation import MFSimulation
from flopy.mf6.mfmodel import MFModel
from flopy.mf6.modflow.mfims import ModflowIms
from flopy.mf6.modflow.mftdis import ModflowTdis
from flopy.mf6.modflow.mfgwfic import ModflowGwfic
from flopy.mf6.modflow.mfgwfnpf import ModflowGwfnpf
from flopy.mf6.modflow.mfgwfdis import ModflowGwfdis
from flopy.mf6.modflow.mfgwfriv import ModflowGwfriv
from flopy.mf6.modflow.mfgwfsto import ModflowGwfsto
from flopy.mf6.modflow.mfgwfoc import ModflowGwfoc
from flopy.mf6.modflow.mfgwfwel import ModflowGwfwel
from flopy.mf6.modflow.mfgwfdrn import ModflowGwfdrn
from flopy.mf6.modflow.mfgwfhfb import ModflowGwfhfb
from flopy.mf6.modflow.mfgwfchd import ModflowGwfchd
from flopy.mf6.modflow.mfgwfghb import ModflowGwfghb
from flopy.mf6.modflow.mfgwfrch import ModflowGwfrch
from flopy.mf6.modflow.mfgwfrcha import ModflowGwfrcha
from flopy.mf6.data.mfdatautil import ArrayUtil
import flopy.utils.binaryfile as bf

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

cpth = os.path.join('temp', 't505')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)

def np001():
    # init paths
    test_ex_name = 'np001'
    model_name = 'np001_mod'

    pth = os.path.join('..', 'examples', 'data', 'mf6', 'create_tests', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file = os.path.join(expected_output_folder, 'np001_mod.hds')
    expected_cbc_file = os.path.join(expected_output_folder, 'np001_mod.cbc')

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version='mf6', exe_name=exe_name, sim_ws=pth,
                       sim_tdis_file='{}.tdis'.format(test_ex_name))
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(sim, time_units='DAYS', nper=2, tdisrecarray=tdis_rc)
    ims_package = ModflowIms(sim, print_option='ALL', complexity='SIMPLE',outer_hclose=0.00001,
                             outer_maximum=50, under_relaxation='NONE', inner_maximum=30,
                             inner_hclose=0.00001, linear_acceleration='CG',
                             preconditioner_levels=7, preconditioner_drop_tolerance=0.01,
                             number_orthogonalizations=2)
    model = MFModel(sim, model_type='gwf6', model_name=model_name,
                    model_nam_file='{}.nam'.format(model_name),
                    ims_file_name='{}.ims'.format(model_name))

    dis_package = ModflowGwfdis(model, length_units='FEET', nlay=1, nrow=1, ncol=10, delr=500.0, delc=500.0,
                                top=100.0, botm=50.0, fname='{}.dis'.format(model_name), pname='mydispkg')
    ic_package = ModflowGwfic(model, strt='initial_heads.txt',
                              fname='{}.ic'.format(model_name))
    npf_package = ModflowGwfnpf(model, save_flows=True, alternative_cell_averaging='logarithmic',
                                icelltype=1, k=5.0)

    oc_package = ModflowGwfoc(model, budget_filerecord=[('np001_mod.cbc',)],
                              head_filerecord=[('np001_mod.hds',)],
                              saverecord={0:[('HEAD', 'ALL'), ('BUDGET', 'ALL')],1:[('HEAD', 'ALL'), ('BUDGET', 'ALL')]},
                              printrecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([('HEAD', 'ALL'), ('BUDGET', 'ALL')], 1)

    sto_package = ModflowGwfsto(model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15)

    wel_package = ModflowGwfwel(model, print_input=True, print_flows=True, save_flows=True, maxbound=2,
                                periodrecarray=[((0,0,4), -2000.0), ((0,0,7), -2.0)])
    wel_package.periodrecarray.add_transient_key(1)
    wel_package.periodrecarray.set_data({1:{'filename':'wel.txt', 'factor':1.0}})

    drn_package = ModflowGwfdrn(model, print_input=True, print_flows=True, save_flows=True, maxbound=1,
                                periodrecarray=[((0,0,0), 80, 60.0)])

    riv_package = ModflowGwfriv(model, print_input=True, print_flows=True, save_flows=True, maxbound=1,
                                periodrecarray=[((0,0,9), 110, 90.0, 100.0)])

    # verify package look-up
    pkgs = model.get_package()
    assert(len(pkgs) == 8)
    pkg = model.get_package('oc')
    assert isinstance(pkg, ModflowGwfoc)
    pkg = sim.get_package('tdis')
    assert isinstance(pkg, ModflowTdis)
    pkg = model.get_package('mydispkg')
    assert isinstance(pkg, ModflowGwfdis) and pkg.package_name == 'mydispkg'

    # verify external file contents
    array_util = ArrayUtil()
    ic_data = ic_package.strt
    ic_array = ic_data.get_data()
    assert array_util.array_comp(ic_array, [[[100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]]])

    # make folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # get expected results
    budget_file = os.path.join(os.getcwd(), expected_cbc_file)
    budget_obj = bf.CellBudgetFile(budget_file, precision='double')
    budget_frf_valid = np.array(budget_obj.get_data(text='FLOW-JA-FACE', full3D=True))

    # compare output to expected results
    head_file = os.path.join(os.getcwd(), expected_head_file)
    head_new = os.path.join(run_folder, 'np001_mod.hds')
    outfile = os.path.join(run_folder, 'head_compare.dat')
    assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)

    budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
    assert array_util.array_comp(budget_frf_valid, budget_frf)

    # clean up
    sim.delete_output_files()

    return


def np002():
    # init paths
    test_ex_name = 'np002'
    model_name = 'np002_mod'

    pth = os.path.join('..', 'examples', 'data', 'mf6', 'create_tests', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file = os.path.join(expected_output_folder, 'np002_mod.hds')
    expected_cbc_file = os.path.join(expected_output_folder, 'np002_mod.cbc')

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version='mf6', exe_name=exe_name, sim_ws=pth,
                       sim_tdis_file='{}.tdis'.format(test_ex_name))
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(sim, time_units='DAYS', nper=2, tdisrecarray=tdis_rc)
    model = MFModel(sim, model_type='gwf6', model_name=model_name,
                    model_nam_file='{}.nam'.format(model_name),
                    ims_file_name='{}.ims'.format(model_name))
    ims_package = ModflowIms(sim, print_option='ALL', complexity='SIMPLE',outer_hclose=0.00001,
                             outer_maximum=50, under_relaxation='NONE', inner_maximum=30,
                             inner_hclose=0.00001, linear_acceleration='CG',
                             preconditioner_levels=7, preconditioner_drop_tolerance=0.01,
                             number_orthogonalizations=2)
    sim.register_ims_package(ims_package, [model.name])

    dis_package = ModflowGwfdis(model, length_units='FEET', nlay=1, nrow=1, ncol=10, delr=500.0, delc=500.0,
                                top=100.0, botm=50.0, fname='{}.dis'.format(model_name))
    ic_vals = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ic_package = ModflowGwfic(model, strt=ic_vals,
                              fname='{}.ic'.format(model_name))
    ic_package.strt.store_as_external_file('initial_heads.txt')
    npf_package = ModflowGwfnpf(model, save_flows=True, icelltype=1, k=100.0)
    oc_package = ModflowGwfoc(model, budget_filerecord=[('np002_mod.cbc',)],
                              head_filerecord=[('np002_mod.hds',)],
                              saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                              printrecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    oc_package.saverecord.add_transient_key(1)
    oc_package.saverecord.set_data([('HEAD', 'ALL'), ('BUDGET', 'ALL')], 1)
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([('HEAD', 'ALL'), ('BUDGET', 'ALL')], 1)

    sto_package = ModflowGwfsto(model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15)

    hfb_package = ModflowGwfhfb(model, print_input=True, maxhfb=1, hfbrecarray=[((0,0,3), (0,0,4), 0.00001)])
    chd_package = ModflowGwfchd(model, print_input=True, print_flows=True, maxbound=1, periodrecarray=[((0,0,0), 65.0)])
    ghb_package = ModflowGwfghb(model, print_input=True, print_flows=True, maxbound=1, periodrecarray=[((0,0,9), 125.0, 60.0)])
    rch_package = ModflowGwfrch(model, print_input=True, print_flows=True, maxbound=2, periodrecarray=[((0,0,3), 0.02),((0,0,6), 0.1)])

    # make folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    if run:
        # run simulation
        sim.run_simulation()

        # get expected results
        budget_file = os.path.join(os.getcwd(), expected_cbc_file)
        budget_obj = bf.CellBudgetFile(budget_file, precision='double')
        budget_frf_valid = np.array(budget_obj.get_data(text='FLOW JA FACE    ', full3D=True))

        # compare output to expected results
        head_file = os.path.join(os.getcwd(), expected_head_file)
        head_new = os.path.join(run_folder, 'np002_mod.hds')
        outfile = os.path.join(run_folder, 'head_compare.dat')
        assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)

        array_util = ArrayUtil()
        budget_frf = sim.simulation_data.mfdata[(model_name, 'CBC', 'FLOW-JA-FACE')]
        assert array_util.array_comp(budget_frf_valid, budget_frf)

        # verify external file was written correctly
        ext_file_path = os.path.join(run_folder, 'initial_heads.txt')
        fd = open(ext_file_path, 'r')
        line = fd.readline()
        line_array = line.split()
        assert len(ic_vals) == len(line_array)
        for index in range(0, len(ic_vals)):
            assert ic_vals[index] == float(line_array[index])
        fd.close()

        # clean up
        sim.delete_output_files()

    return


def test021_twri():
    # init paths
    test_ex_name = 'test021_twri'
    model_name = 'twri'

    pth = os.path.join('..', 'examples', 'data', 'mf6', 'create_tests', test_ex_name)
    run_folder = os.path.join(cpth, test_ex_name)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)

    expected_output_folder = os.path.join(pth, 'expected_output')
    expected_head_file = os.path.join(expected_output_folder, 'twri.hds')

    # create simulation
    sim = MFSimulation(sim_name=test_ex_name, version='mf6', exe_name=exe_name, sim_ws=pth,
                       sim_tdis_file='{}.tdis'.format(test_ex_name))
    tdis_rc = [(86400.0, 1, 1.0)]
    tdis_package = ModflowTdis(sim, time_units='SECONDS', nper=1, tdisrecarray=tdis_rc)
    model = MFModel(sim, model_type='gwf6', model_name=model_name,
                    model_nam_file='{}.nam'.format(model_name),
                    ims_file_name='{}.ims'.format(model_name))
    ims_package = ModflowIms(sim, print_option='SUMMARY', outer_hclose=0.0001,
                             outer_maximum=500, under_relaxation='NONE', inner_maximum=100,
                             inner_hclose=0.0001, rcloserecord=0.001, linear_acceleration='CG',
                             scaling_method='NONE', reordering_method='NONE', relaxation_factor=0.97)
    sim.register_ims_package(ims_package, [model.name])
    dis_package = ModflowGwfdis(model, nlay=3, nrow=15, ncol=15, delr=5000.0, delc=5000.0,
                                top=200.0, botm=[-200, -300, -450], fname='{}.dis'.format(model_name))
    ic_package = ModflowGwfic(model, strt=0.0,
                              fname='{}.ic'.format(model_name))
    npf_package = ModflowGwfnpf(model, save_flows=True, perched=True, cvoptions='dewatered',
                                icelltype=[1,0,0], k=[0.001, 0.0001, 0.0002], k33=0.00000002)
    oc_package = ModflowGwfoc(model, budget_filerecord='twri.cbc',
                              head_filerecord='twri.hds',
                              saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                              printrecord=[('HEAD', 'ALL')])

    # build periodrecarray for chd package
    periodrecarray = []
    for layer in range(0, 2):
        for row in range(0, 15):
            periodrecarray.append(((layer, row, 0), 0.0))
    chd_package = ModflowGwfchd(model, print_input=True, print_flows=True, save_flows=True, maxbound=100,
                                periodrecarray=periodrecarray)

    # build periodrecarray for drn package
    periodrecarray = []
    drn_heads = [0.0, 0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0]
    for col, head in zip(range(1,10), drn_heads):
        periodrecarray.append(((0, 7, col), head, 1.0))
    drn_package = ModflowGwfdrn(model, print_input=True, print_flows=True, save_flows=True, maxbound=9,
                                periodrecarray=periodrecarray)
    rch_package = ModflowGwfrcha(model, readasarrays=True, fixed_cell=True, recharge={0:0.00000003})

    periodrecarray = []
    layers = [2,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
    rows = [4,3,5,8,8,8,8,10,10,10,10,12,12,12,12]
    cols = [10,5,11,7,9,11,13,7,9,11,13,7,9,11,13]
    for layer, row, col in zip(layers, rows, cols):
        periodrecarray.append(((layer, row, col), -5.0))
    wel_package = ModflowGwfwel(model, print_input=True, print_flows=True, save_flows=True, maxbound=15,
                                periodrecarray=periodrecarray)

    # make folder to save simulation
    sim.simulation_data.mfpath.set_sim_path(run_folder)

    # write simulation to new location
    sim.write_simulation()

    # run simulation
    sim.run_simulation()

    # compare output to expected results
    head_file = os.path.join(os.getcwd(), expected_head_file)
    head_new = os.path.join(run_folder, 'twri.hds')
    outfile = os.path.join(run_folder, 'head_compare.dat')
    assert pymake.compare_heads(None, None, files1=head_file, files2=head_new, outfile=outfile)

    # clean up
    sim.delete_output_files()

    return


if __name__ == '__main__':
    np001()
    np002()
    test021_twri()