# Test postprocessing and plotting functions related to specific discharge:
# - get_extended_budget()
# - get_specific_discharge()
# - PlotMapView.plot_vector()
# - PlotCrossSection.plot_vector()

# More precisely:
# - two models are created: one for mf005 and one for mf6
# - the two models are virtually identical; in fact, the options are such that
#   the calculated heads are indeed exactly the same (which is, by the way,
#   quite remarkable!)
# - the model is a very small synthetic test case that just contains enough
#   things to allow for the functions to be thoroughly tested

import flopy
import os
import numpy as np
import flopy.utils.binaryfile as bf

# model names, file names and locations
modelname_mf2005 = 't070_mf2005'
modelname_mf6 = 't070_mf6'
postproc_test_ws = os.path.join('.', 'temp', 't070')
modelws_mf2005 = os.path.join(postproc_test_ws, modelname_mf2005)
modelws_mf6 = os.path.join(postproc_test_ws, modelname_mf6)
cbcfile_mf2005 = os.path.join(modelws_mf2005, modelname_mf2005 + '.cbc')
cbcfile_mf6 = os.path.join(modelws_mf6, modelname_mf6 + '.cbc')
hdsfile_mf2005 = os.path.join(modelws_mf2005, modelname_mf2005 + '.hds')
hdsfile_mf6 = os.path.join(modelws_mf6, modelname_mf6 + '.hds')
namfile_mf2005 = os.path.join(modelws_mf2005, modelname_mf2005 + '.nam')
namfile_mf6 = os.path.join(modelws_mf6, modelname_mf6 + '.nam')

# model domain, grid definition and properties
Lx = 100.
Ly = 100.
ztop = 0.
zbot = -100.
nlay = 4
nrow = 4
ncol = 4
delr = Lx/ncol
delc = Ly/nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk=1.
rchrate = 0.1
lay_to_plot = 1

# variables for the BAS (mf2005) or DIS (mf6) package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[1, 0, 1] = 0 # set a no-flow cell
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)

# add inflow through west boundary using WEL package
Q = 100.
wel_list = []
wel_list_iface = []
for k in range(nlay):
    for i in range(nrow):
        wel_list.append([k, i, 0, Q])
        wel_list_iface.append(wel_list[-1] + [1])

# allow flow through north, south, and bottom boundaries using GHB package
ghb_head = -30. # low enough to have dry cells in first layer
ghb_cond = hk * delr * delv / (0.5 * delc)
ghb_list = []
ghb_list_iface = []
for k in range(1, nlay):
    for j in range(ncol):
        if not (k==1 and j==1): # skip no-flow cell
            ghb_list.append([k, 0, j, ghb_head, ghb_cond])
            ghb_list_iface.append(ghb_list[-1] + [4])
        ghb_list.append([k, nrow-1, j, ghb_head, ghb_cond])
        ghb_list_iface.append(ghb_list[-1] + [3])
for i in range(nrow):
    for j in range(ncol):
        ghb_list.append([nlay-1, i, j, ghb_head, ghb_cond])
        ghb_list_iface.append(ghb_list[-1] + [5])

# river in the eastern part
riv_stage = -30.
riv_cond = hk * delr * delc / (0.5 * delv)
riv_rbot = riv_stage - 5.
riv_list = []
for i in range(nrow):
    riv_list.append([1, i, ncol-1, riv_stage, riv_cond, riv_rbot])

# drain in the south part
drn_stage = -30.
drn_cond = hk * delc * delv / (0.5 * delr)
drn_list = []
for j in range(ncol):
    drn_list.append([1, i, nrow-1, drn_stage, drn_cond])

boundary_ifaces = {'WELLS': wel_list_iface,
                   'HEAD DEP BOUNDS': ghb_list_iface,
                   'RIVER LEAKAGE': 2,
                   'DRAIN': 3,
                   'RECHARGE': 6}

def build_model_mf2005():

    # create folders
    if not os.path.isdir(modelws_mf2005):
        os.makedirs(modelws_mf2005)

    # create modflow model
    mf = flopy.modflow.Modflow(modelname_mf2005, model_ws=modelws_mf2005,
                               exe_name='mf2005')

    # cell by cell flow file unit number
    cbc_unit_nb = 53

    # create DIS package
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:])

    # create BAS package
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # create LPF package
    laytyp = np.zeros(nlay)
    laytyp[0] = 1
    laywet = np.zeros(nlay)
    laywet[0] = 1
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, ipakcb=cbc_unit_nb,
                                   laytyp=laytyp, laywet=laywet, wetdry=-0.01)

    # create WEL package
    wel_dict = {0: wel_list}
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_dict,
                                   ipakcb=cbc_unit_nb)

    # create GHB package
    ghb_dict = {0: ghb_list}
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_dict,
                                   ipakcb=cbc_unit_nb)

    # create RIV package
    riv_dict = {0: riv_list}
    riv = flopy.modflow.ModflowRiv(mf, stress_period_data=riv_dict,
                                   ipakcb=cbc_unit_nb)

    # create DRN package
    drn_dict = {0: drn_list}
    drn = flopy.modflow.ModflowDrn(mf, stress_period_data=drn_dict,
                                   ipakcb=cbc_unit_nb)

    # create RCH package
    rch = flopy.modflow.ModflowRch(mf, rech=rchrate, ipakcb=cbc_unit_nb)

    # create OC package
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    # create PCG package
    pcg = flopy.modflow.ModflowPcg(mf)

    # write the MODFLOW model input files
    mf.write_input()

    # run the MODFLOW model
    success, buff = mf.run_model()
    return

def build_model_mf6():

    if not os.path.isdir(modelws_mf6):
        os.makedirs(modelws_mf6)

    # create simulation
    simname = modelname_mf6
    sim = flopy.mf6.MFSimulation(sim_name=simname, version='mf6',
                                 exe_name='mf6', sim_ws=modelws_mf6)

    # create tdis package
    tdis_rc = [(1.0, 1, 1.0)]
    tdis = flopy.mf6.ModflowTdis(sim, pname='tdis', time_units='DAYS',
                                 perioddata=tdis_rc)

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=modelname_mf6,
                               model_nam_file='{}.nam'.format(modelname_mf6))
    gwf.name_file.save_flows = True

    # create iterative model solution and register the gwf model with it
    rcloserecord = [1e-5, 'STRICT']
    ims = flopy.mf6.ModflowIms(sim, pname='ims', print_option='SUMMARY',
                               complexity='SIMPLE', outer_hclose=1.e-5,
                               outer_maximum=50, under_relaxation='NONE',
                               inner_maximum=30, inner_hclose=1.e-5,
                               rcloserecord=rcloserecord,
                               linear_acceleration='CG',
                               scaling_method='NONE', reordering_method='NONE',
                               relaxation_factor=0.99)
    sim.register_ims_package(ims, [gwf.name])

    # create dis package
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                  delr=delr, delc=delc,
                                  top=ztop, botm=botm[1:], idomain=ibound)

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, pname='ic', strt=strt)

    # create node property flow package
    rewet_record = [('WETFCT', 0.1, 'IWETIT', 1, 'IHDWET', 0)]
    icelltype = np.zeros(ibound.shape)
    icelltype[0, :, :] = 1
    wetdry = np.zeros(ibound.shape)
    wetdry[0, :, :] = -0.01
    npf = flopy.mf6.ModflowGwfnpf(gwf,
                                  icelltype=icelltype,
                                  k=hk,
                                  rewet_record=rewet_record,
                                  wetdry=wetdry,
                                  cvoptions=[()],
                                  save_specific_discharge=True)

    # create wel package
    welspd = [[(wel_i[0], wel_i[1], wel_i[2]), wel_i[3]] for wel_i in wel_list]
    wel = flopy.mf6.ModflowGwfwel(gwf, print_input=True,
                                  stress_period_data=welspd)

    # create ghb package
    ghbspd = [[(ghb_i[0], ghb_i[1], ghb_i[2]), ghb_i[3], ghb_i[4]]
              for ghb_i in ghb_list]
    ghb = flopy.mf6.ModflowGwfghb(gwf, print_input=True,
                                  stress_period_data=ghbspd)

    # create riv package
    rivspd = [[(riv_i[0], riv_i[1], riv_i[2]), riv_i[3], riv_i[4], riv_i[5]]
              for riv_i in riv_list]
    riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=rivspd)

    # create drn package
    drnspd = [[(drn_i[0], drn_i[1], drn_i[2]), drn_i[3], drn_i[4]]
              for drn_i in drn_list]
    drn = flopy.mf6.ModflowGwfdrn(gwf, print_input=True,
                                  stress_period_data=drnspd)

    # create rch package
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=rchrate)

    # output control
    oc = flopy.mf6.ModflowGwfoc(gwf, pname='oc', budget_filerecord=
                                '{}.cbc'.format(modelname_mf6),
                                head_filerecord='{}.hds'.format(modelname_mf6),
                                headprintrecord=[('COLUMNS', 10, 'WIDTH', 15,
                                                  'DIGITS', 6, 'GENERAL')],
                                saverecord=[('HEAD', 'ALL'),
                                            ('BUDGET', 'ALL')],
                                printrecord=[('HEAD', 'ALL'),
                                             ('BUDGET', 'ALL')])

    # write input files
    sim.write_simulation()

    # run simulation
    sim.run_simulation()
    return

def basic_check(Qx_ext, Qy_ext, Qz_ext):
    # check shape
    assert Qx_ext.shape == (nlay, nrow, ncol+1)
    assert Qy_ext.shape == (nlay, nrow+1, ncol)
    assert Qz_ext.shape == (nlay+1, nrow, ncol)

    # check sign
    assert Qx_ext[2, 1, 1] > 0
    assert Qy_ext[2, 1, 1] > 0
    assert Qz_ext[2, 1, 1] < 0
    return

def local_balance_check(Qx_ext, Qy_ext, Qz_ext, hdsfile=None, model=None):
    # calculate water blance at every cell
    local_balance = Qx_ext[:, :, :-1] - Qx_ext[:, :, 1:] + \
                    Qy_ext[:, 1:, :] - Qy_ext[:, :-1, :] + \
                    Qz_ext[1:, :, :] - Qz_ext[:-1, :, :]

    # calculate total flow through every cell
    local_total = np.abs(Qx_ext[:, :, :-1]) + np.abs(Qx_ext[:, :, 1:]) + \
                  np.abs(Qy_ext[:, 1:, :]) + np.abs(Qy_ext[:, :-1, :]) + \
                  np.abs(Qz_ext[1:, :, :]) + np.abs(Qz_ext[:-1, :, :])

    # we should disregard no-flow and dry cells
    if hdsfile is not None and model is not None:
        hds = bf.HeadFile(hdsfile, precision='single')
        head = hds.get_data()
        noflo_or_dry = np.logical_or(head==model.hnoflo, head==model.hdry)
        local_balance[noflo_or_dry] = np.nan

    # check water balance = 0 at every cell
    rel_err = local_balance / local_total
    max_rel_err = np.nanmax(rel_err)
    assert np.allclose(max_rel_err + 1., 1.)

def test_extended_budget_default():
    # build and run MODFLOW 2005 model
    build_model_mf2005()

    # load and postprocess
    Qx_ext, Qy_ext, Qz_ext = \
        flopy.utils.postprocessing.get_extended_budget(cbcfile_mf2005)

    # basic check
    basic_check(Qx_ext, Qy_ext, Qz_ext)

    # overall check
    overall = np.sum(Qx_ext) + np.sum(Qy_ext) + np.sum(Qz_ext)
    assert np.allclose(overall, -1122.4931640625)
    return

def test_extended_budget_comprehensive():
    # load and postprocess
    mf = flopy.modflow.Modflow.load(namfile_mf2005, check=False)
    Qx_ext, Qy_ext, Qz_ext = \
        flopy.utils.postprocessing.get_extended_budget(cbcfile_mf2005,
                    boundary_ifaces=boundary_ifaces,
                    hdsfile=hdsfile_mf2005, model=mf)

    # basic check
    basic_check(Qx_ext, Qy_ext, Qz_ext)

    # local balance check
    local_balance_check(Qx_ext, Qy_ext, Qz_ext, hdsfile_mf2005, mf)

    # overall check
    overall = np.sum(Qx_ext) + np.sum(Qy_ext) + np.sum(Qz_ext)
    assert np.allclose(overall, -1110.646240234375)
    return

def test_specific_discharge_default():
    # load and postprocess
    mf = flopy.modflow.Modflow.load(namfile_mf2005, check=False)
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(mf,
                                            cbcfile_mf2005)

    # overall check
    overall = np.sum(qx) + np.sum(qy) + np.sum(qz)
    assert np.allclose(overall, -1.7959892749786377)
    return

def test_specific_discharge_comprehensive():
    # load and postprocess
    mf = flopy.modflow.Modflow.load(namfile_mf2005, check=False)
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(mf,
                             cbcfile_mf2005,
                             boundary_ifaces=boundary_ifaces,
                             hdsfile=hdsfile_mf2005)

    # check nan values
    assert np.isnan(qx[0, 0, 2])
    assert np.isnan(qx[1, 0, 1])

    # overall check
    overall = np.nansum(qx) + np.nansum(qy) + np.nansum(qz)
    assert np.allclose(overall, -0.8086609840393066)
    return

def test_specific_discharge_mf6():
    from flopy.mf6.modflow.mfsimulation import MFSimulation

    # build and run MODFLOW 6 model
    build_model_mf6()

    # load and postprocess
    sim = MFSimulation.load(sim_name=modelname_mf6, sim_ws=modelws_mf6,
                            verbosity_level=0)
    gwf = sim.get_model(modelname_mf6)
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(gwf,
                       cbcfile_mf6, precision='double',
                       hdsfile=hdsfile_mf6)

    # check nan values
    assert np.isnan(qx[0, 0, 2])
    assert np.isnan(qx[1, 0, 1])

    # overall check
    overall = np.nansum(qx) + np.nansum(qy) + np.nansum(qz)
    assert np.allclose(overall, -2.5768726154495947)
    return

if __name__ == '__main__':
    test_extended_budget_default()
    test_extended_budget_comprehensive()
    test_specific_discharge_default()
    test_specific_discharge_comprehensive()
    test_specific_discharge_mf6()
