import os
import shutil
import numpy as np
import flopy

model_ws = os.path.join('temp', 't057')
# delete the directory if it exists
if os.path.isdir(model_ws):
    shutil.rmtree(model_ws)

exe_names = {'mf2005': 'mf2005',
             'mf6': 'mf6',
             'mp7': 'mp7'}
run = True
for key in exe_names.keys():
    v = flopy.which(exe_names[key])
    if v is None:
        run = False
        break

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

zone3 = np.ones((nrow, ncol), dtype=np.int32)
zone3[wel_loc[1:]] = 2
zones = [1, 1, zone3]

# create particles
part0 = flopy.modpath.LRCParticles.get_empty(ncells=21, particleid=True)
part0['k'] = 0
part0['j'] = 3
part0['localx'] = 0.5
part0['localy'] = 0.5
part0['localz'] = 1.
part0['timeoffset'] = 0.
part0['drape'] = 0
for idx in range(part0.shape[0]):
    part0['id'][idx] = idx
    part0['i'][idx] = idx
pg1 = flopy.modpath.LRCParticles(particlegroupname='PG1', particledata=part0,
                                 filename='ex01a.sloc')
particlegroups = [pg1]

defaultiface = {'RECHARGE': 6, 'ET': 6}
defaultiface6 = {'RCH': 6, 'EVT': 6}

def test_mf2005():
    # build and run MODPATH 7 with MODFLOW-2005
    build_mf2005()

def test_mf6():
    # build and run MODPATH 7 with MODFLOW 6
    build_mf6()

def test_output():
    fpth0 = os.path.join(model_ws, 'mf2005', 'ex01_mf2005_mp.mppth')
    fpth1 = os.path.join(model_ws, 'mf6', 'ex01_mf6_mp.mppth')
    return

def build_mf2005():
    '''
    MODPATH 7 example 1 for MODFLOW-2005
    '''

    ws = os.path.join(model_ws, 'mf2005')
    nm = 'ex01_mf2005'
    exe_name = exe_names['mf2005']
    iu_cbc = 130
    m = flopy.modflow.Modflow(nm, model_ws=ws,
                              exe_name=exe_name)
    flopy.modflow.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol,
                             nper=nper, perlen=perlen, nstp=nstp,
                             tsmult=tsmult, steady=True,
                             delr=delr, delc=delc,
                             top=top, botm=botm)
    flopy.modflow.ModflowLpf(m, ipakcb=iu_cbc, laytyp=laytyp, hk=kh, vka=kv)
    flopy.modflow.ModflowBas(m, ibound=1, strt=top)
    # recharge
    flopy.modflow.ModflowRch(m, ipakcb=iu_cbc, rech=rch)
    # wel
    wd = [i for i in wel_loc] + [wel_q]
    flopy.modflow.ModflowWel(m, ipakcb=iu_cbc, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([0, i, ncol-1, riv_h, riv_c, riv_z])
    flopy.modflow.ModflowRiv(m, ipakcb=iu_cbc, stress_period_data={0: rd})
    # output control
    flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ['save head',
                                                            'save budget']})
    flopy.modflow.ModflowPcg(m, hclose=0.01, rclose=1.0)

    m.write_input()
    success, buff = m.run_model()
    assert success, 'mf2005 model did not run'

    # create modpath files
    exe_name = exe_names['mp7']
    mp = flopy.modpath.Modpath7(modelname=nm + '_mp', flowmodel=m,
                                exe_name=exe_name, model_ws=ws)
    mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.1,
                                      defaultiface=defaultiface)
    mpsim = flopy.modpath.Modpath7Sim(mp, simulationtype='combined',
                                      trackingdirection='forward',
                                      weaksinkoption='pass_through',
                                      weaksourceoption='pass_through',
                                      budgetoutputoption='summary',
                                      budgetcellnumbers=[1049, 1259],
                                      referencetime=[0, 0, 0.],
                                      stoptimeoption='extend',
                                      timepointdata=[500, 1000.],
                                      zonedataoption='on', zones=zones,
                                      particlegroups=particlegroups)

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()

    return


def build_mf6():
    '''
    MODPATH 7 example 1 for MODFLOW 6
    '''

    ws = os.path.join(model_ws, 'mf6')
    nm = 'ex01_mf6'
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
    success, buff = sim.run_simulation()
    assert success, 'mf6 model did not run'

    # create modpath files
    exe_name = exe_names['mp7']
    mp = flopy.modpath.Modpath7(modelname=nm + '_mp', flowmodel=gwf,
                                exe_name=exe_name, model_ws=ws)
    mpbas = flopy.modpath.Modpath7Bas(mp, porosity=0.1,
                                      defaultiface=defaultiface6)
    mpsim = flopy.modpath.Modpath7Sim(mp, simulationtype='combined',
                                      trackingdirection='forward',
                                      weaksinkoption='pass_through',
                                      weaksourceoption='pass_through',
                                      budgetoutputoption='summary',
                                      budgetcellnumbers=[1049, 1259],
                                      referencetime=[0, 0, 0.],
                                      stoptimeoption='extend',
                                      timepointdata=[500, 1000.],
                                      zonedataoption='on', zones=zones,
                                      particlegroups=particlegroups)

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()

    return


if __name__ == '__main__':
    test_mf2005()
    test_mf6()
    test_output()
