import os
import shutil
import numpy as np
import flopy

model_ws = os.path.join('temp', 't059')
# delete the directory if it exists
if os.path.isdir(model_ws):
    shutil.rmtree(model_ws)

exe_names = {'mf6': 'mf6',
             'mp7': 'mp7'}
run = True
for key in exe_names.keys():
    v = flopy.which(exe_names[key])
    if v is None:
        run = False
        break

ws = model_ws
nm = 'ex01_mf6'

# model data
nper, nstp, perlen, tsmult = 1, 1, 1., 1.
nlay, nrow, ncol = 3, 21, 20
delr = delc = 500.
top = 400.
botm = [220., 200., 0.]
laytyp = [1, 0, 0]
kh = [50., 0.01, 200.]
kv = [10., 0.01, 20.]
wel_loc = (2, 10, 9)
wel_q = -150000.
rch = 0.005
riv_h = 320.
riv_z = 317.
riv_c = 1.e5


def test_mf6():
    # build and run MODPATH 7 with MODFLOW 6
    build_mf6()


def test_forward():
    mpnam = nm + '_mp_forward'
    exe_name = exe_names['mp7']

    # load the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation.load('mf6mod', 'mf6', 'mf6', ws)
    gwf = sim.get_model(nm)

    mp = flopy.modpath.Modpath7.create_mp7(modelname=mpnam,
                                           trackdir='forward',
                                           flowmodel=gwf,
                                           exe_name=exe_name,
                                           model_ws=model_ws)

    # build and run the MODPATH 7 models
    build_modpath(mp)
    return


def test_backward():
    mpnam = nm + '_mp_backward'
    exe_name = exe_names['mp7']

    # load the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation.load('mf6mod', 'mf6', 'mf6', ws)
    gwf = sim.get_model(nm)

    mp = flopy.modpath.Modpath7.create_mp7(modelname=mpnam,
                                           trackdir='backward',
                                           flowmodel=gwf,
                                           exe_name=exe_name,
                                           model_ws=model_ws)

    # build and run the MODPATH 7 models
    build_modpath(mp)
    return


def build_mf6():
    '''
    MODPATH 7 example 1 for MODFLOW 6
    '''

    exe_name = exe_names['mf6']

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(sim_name=nm, exe_name='mf6',
                                 version='mf6', sim_ws=ws)

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, pname='tdis',
                                                time_units='DAYS', nper=nper,
                                                perioddata=[pd])

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = '{}.nam'.format(nm)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=nm,
                               model_nam_file=model_nam_file, save_flows=True)

    # Create the Flopy iterative model solver (ims) Package object
    ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname='ims',
                                             complexity='SIMPLE')

    # create gwf file
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(gwf, pname='dis', nlay=nlay,
                                                   nrow=nrow, ncol=ncol,
                                                   length_units='FEET',
                                                   delr=delr, delc=delc,
                                                   top=top,
                                                   botm=botm)
    # Create the initial conditions package
    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname='ic', strt=top)

    # Create the node property flow package
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, pname='npf',
                                                   icelltype=laytyp, k=kh,
                                                   k33=kv)

    # recharge
    flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, maxbound=1,
                                             stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(gwf, stress_period_data={0: rd})
    # Create the output control package
    headfile = '{}.hds'.format(nm)
    head_record = [headfile]
    budgetfile = '{}.cbb'.format(nm)
    budget_record = [budgetfile]
    saverecord = [('HEAD', 'ALL'),
                  ('BUDGET', 'ALL')]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc',
                                                saverecord=saverecord,
                                                head_filerecord=head_record,
                                                budget_filerecord=budget_record)

    # Write the datasets
    sim.write_simulation()

    # Run the simulation
    if run:
        success, buff = sim.run_simulation()
        assert success, 'mf6 model did not run'


def build_modpath(mp):
    # write modpath datasets
    mp.write_input()

    # run modpath
    if run:
        success, buff = mp.run_model()
        assert success, 'mp7 model ({}) did not run'.format(mp.name)

    return


if __name__ == '__main__':
    # build and run modflow 6
    test_mf6()

    # build forward tracking model
    test_forward()

    # build forward tracking model
    test_backward()
