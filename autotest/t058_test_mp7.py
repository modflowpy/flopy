import os
import shutil
import numpy as np
import flopy

model_ws = os.path.join('temp', 't058')
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
nm = 'ex01b_mf6'

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

# particle data
zone3 = np.ones((nrow, ncol), dtype=np.int32)
zone3[wel_loc[1:]] = 2
zones = [1, 1, zone3]

defaultiface6 = {'RCH': 6, 'EVT': 6}

local = np.array([[0.1666666667E+00, 0.1666666667E+00, 1.],
                  [0.5000000000E+00, 0.1666666667E+00, 1.],
                  [0.8333333333E+00, 0.1666666667E+00, 1.],
                  [0.1666666667E+00, 0.5000000000E+00, 1.],
                  [0.5000000000E+00, 0.5000000000E+00, 1.],
                  [0.8333333333E+00, 0.5000000000E+00, 1.],
                  [0.1666666667E+00, 0.8333333333E+00, 1.],
                  [0.5000000000E+00, 0.8333333333E+00, 1.],
                  [0.8333333333E+00, 0.8333333333E+00, 1.]])


def test_mf6():
    # build and run MODPATH 7 with MODFLOW 6
    build_mf6()


def test_default_modpath():
    mpnam = nm + '_mp_default'
    pg = flopy.modpath.ParticleGroup(particlegroupname='DEFAULT')
    build_modpath(mpnam, pg)
    return


def test_faceparticles_is1():
    mpnam = nm + '_mp_face_t1node'
    locs = []
    localx = []
    localy = []
    for i in range(nrow):
        for j in range(ncol):
            node = i * ncol + j
            for xloc, yloc, zloc in local:
                locs.append(node)
                localx.append(xloc)
                localy.append(yloc)
    p = flopy.modpath.ParticleData(locs, structured=False, drape=0,
                                   localx=localx, localy=localy, localz=1)
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroup(particlegroupname='T1NODEPG',
                                     particledata=p,
                                     filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_facenode_is3():
    mpnam = nm + '_mp_face_t3node'
    locs = []
    for i in range(nrow):
        for j in range(ncol):
            node = i * ncol + j
            locs.append(node)
    sd = flopy.modpath.FaceDataType(drape=0,
                                    verticaldivisions1=0,
                                    horizontaldivisions1=0,
                                    verticaldivisions2=0,
                                    horizontaldivisions2=0,
                                    verticaldivisions3=0,
                                    horizontaldivisions3=0,
                                    verticaldivisions4=0,
                                    horizontaldivisions4=0,
                                    rowdivisions5=0,
                                    columndivisions5=0,
                                    rowdivisions6=3,
                                    columndivisions6=3)
    p = flopy.modpath.NodeParticleData(subdivisiondata=sd, nodes=locs)
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='T3NODEPG',
                                                 particledata=p,
                                                 filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_facenode_is3a():
    mpnam = nm + '_mp_face_t3anode'
    locsa = []
    for i in range(11):
        for j in range(ncol):
            node = i * ncol + j
            locsa.append(node)
    locsb = []
    for i in range(11, nrow):
        for j in range(ncol):
            node = i * ncol + j
            locsb.append(node)
    sd = flopy.modpath.FaceDataType(drape=0,
                                    verticaldivisions1=0,
                                    horizontaldivisions1=0,
                                    verticaldivisions2=0,
                                    horizontaldivisions2=0,
                                    verticaldivisions3=0,
                                    horizontaldivisions3=0,
                                    verticaldivisions4=0,
                                    horizontaldivisions4=0,
                                    rowdivisions5=0,
                                    columndivisions5=0,
                                    rowdivisions6=3,
                                    columndivisions6=3)
    p = flopy.modpath.NodeParticleData(subdivisiondata=[sd, sd],
                                       nodes=[locsa, locsb])
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='T3ANODEPG',
                                                 particledata=p,
                                                 filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_facenode_is2a():
    mpnam = nm + '_mp_face_t2anode'
    locsa = [[0, 0, 0, 0, 10, ncol - 1]]
    locsb = [[0, 11, 0, 0, nrow-1, ncol - 1]]
    sd = flopy.modpath.FaceDataType(drape=0,
                                    verticaldivisions1=0,
                                    horizontaldivisions1=0,
                                    verticaldivisions2=0,
                                    horizontaldivisions2=0,
                                    verticaldivisions3=0,
                                    horizontaldivisions3=0,
                                    verticaldivisions4=0,
                                    horizontaldivisions4=0,
                                    rowdivisions5=0,
                                    columndivisions5=0,
                                    rowdivisions6=3,
                                    columndivisions6=3)
    p = flopy.modpath.LRCParticleData(subdivisiondata=[sd, sd],
                                      lrcregions=[locsa, locsb])
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='T2ANODEPG',
                                                 particledata=p,
                                                 filename=fpth)
    build_modpath(mpnam, pg)
    return

def test_cellparticles_is1():
    mpnam = nm + '_mp_cell_t1node'
    locs = []
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j
                locs.append(node)
    p = flopy.modpath.ParticleData(locs, structured=False, drape=0,
                                   localx=0.5, localy=0.5, localz=0.5)
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroup(particlegroupname='T1NODEPG',
                                     particledata=p,
                                     filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_cellparticleskij_is1():
    mpnam = nm + '_mp_cell_t1kij'
    locs = []
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                locs.append((k, i, j))
    p = flopy.modpath.ParticleData(locs, structured=True, drape=0,
                                   localx=0.5, localy=0.5, localz=0.5)
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroup(particlegroupname='T1KIJPG',
                                     particledata=p,
                                     filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_cellnode_is3():
    mpnam = nm + '_mp_cell_t3node'
    locs = []
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j
                locs.append(node)
    sd = flopy.modpath.CellDataType(drape=0, columncelldivisions=1,
                                    rowcelldivisions=1, layercelldivisions=1)
    p = flopy.modpath.NodeParticleData(subdivisiondata=sd, nodes=locs)
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='T3CELLPG',
                                                 particledata=p,
                                                 filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_cellnode_is3a():
    mpnam = nm + '_mp_cell_t3anode'
    locsa = []
    for k in range(1):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j
                locsa.append(node)
    locsb = []
    for k in range(1, 2):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j
                locsb.append(node)
    locsc = []
    for k in range(2, nlay):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j
                locsc.append(node)
    sd = flopy.modpath.CellDataType(drape=0, columncelldivisions=1,
                                    rowcelldivisions=1, layercelldivisions=1)
    p = flopy.modpath.NodeParticleData(subdivisiondata=[sd, sd, sd],
                                       nodes=[locsa, locsb, locsc])
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='T3ACELLPG',
                                                 particledata=p,
                                                 filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_cellnode_is2a():
    mpnam = nm + '_mp_cell_t2anode'
    locsa = [[0, 0, 0, 0, nrow - 1, ncol - 1],
             [1, 0, 0, 1, nrow - 1, ncol - 1]]
    locsb = [[2, 0, 0, 2, nrow - 1, ncol - 1]]
    sd = flopy.modpath.CellDataType(drape=0, columncelldivisions=1,
                                    rowcelldivisions=1, layercelldivisions=1)
    p = flopy.modpath.LRCParticleData(subdivisiondata=[sd, sd],
                                      lrcregions=[locsa, locsb])
    fpth = mpnam + '.sloc'
    pg = flopy.modpath.ParticleGroupLRCTemplate(particlegroupname='T2ACELLPG',
                                                particledata=p,
                                                filename=fpth)
    build_modpath(mpnam, pg)
    return


def test_face_endpoint_output():
    # set base file name
    fpth0 = os.path.join(model_ws, 'ex01b_mf6_mp_face_t1node.mpend')

    # get list of node endpath files
    epf = [os.path.join(model_ws, name) for name in os.listdir(model_ws)
           if '.mpend' in name and '_face_' in name and '_t2a' not in name]
    epf.remove(fpth0)

    endpoint_compare(fpth0, epf)
    return


def test_cell_endpoint_output():
    # set base file name
    fpth0 = os.path.join(model_ws, 'ex01b_mf6_mp_cell_t1node.mpend')

    # get list of node endpath files
    epf = [os.path.join(model_ws, name) for name in os.listdir(model_ws)
           if '.mpend' in name and '_cell_' in name and '_t2a' not in name]
    epf.remove(fpth0)

    endpoint_compare(fpth0, epf)
    return


def endpoint_compare(fpth0, epf):
    # get base endpoint data
    e = flopy.utils.EndpointFile(fpth0)
    maxtime0 = e.get_maxtime()
    maxid0 = e.get_maxid()
    maxtravel0 = e.get_maxtraveltime()
    e0 = e.get_alldata()

    names = ['x', 'y', 'z', 'x0', 'y0', 'z0']
    dtype = np.dtype([('x', np.float32), ('y', np.float32),
                      ('z', np.float32), ('x0', np.float32),
                      ('y0', np.float32), ('z0', np.float32)])
    t0 = np.rec.fromarrays((e0[name] for name in names), dtype=dtype)

    for fpth1 in epf:
        e = flopy.utils.EndpointFile(fpth1)
        maxtime1 = e.get_maxtime()
        maxid1 = e.get_maxid()
        maxtravel1 = e.get_maxtraveltime()
        e1 = e.get_alldata()

        # check maxid
        msg = 'endpoint maxid ({}) '.format(maxid0) + \
              'in {} '.format(os.path.basename(fpth0)) + \
              'are not equal to the ' + \
              'endpoint maxid ({}) '.format(maxid1) + \
              'in {}'.format(os.path.basename(fpth1))
        assert maxid0 == maxid1, msg

        # check maxtravel
        msg = 'endpoint maxtraveltime ({}) '.format(maxtravel0) + \
              'in {} '.format(os.path.basename(fpth0)) + \
              'are not equal to the ' + \
              'endpoint maxtraveltime ({}) '.format(maxtravel1) + \
              'in {}'.format(os.path.basename(fpth1))
        assert maxtravel0 == maxtravel1, msg

        # check maxtimes
        msg = 'endpoint maxtime ({}) '.format(maxtime0) + \
              'in {} '.format(os.path.basename(fpth0)) + \
              'are not equal to the ' + \
              'endpoint maxtime ({}) '.format(maxtime1) + \
              'in {}'.format(os.path.basename(fpth1))
        assert maxtime0 == maxtime1, msg

        # check that endpoint data are approximately the same
        t1 = np.rec.fromarrays((e1[name] for name in names), dtype=dtype)
        for name in names:
            msg = 'endpoints in {} '.format(os.path.basename(fpth0)) + \
                  'are not equal (within 1e-5) to the ' + \
                  'endpoints  in {} '.format(os.path.basename(fpth1)) + \
                  'for column {}.'.format(name)
            assert np.allclose(t0[name], t1[name]), msg

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


def build_modpath(mpn, particlegroups):
    # load the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation.load('mf6mod', 'mf6', 'mf6', ws)
    gwf = sim.get_model(nm)

    # create modpath files
    exe_name = exe_names['mp7']
    mp = flopy.modpath.Modpath7(modelname=mpn, flowmodel=gwf,
                                exe_name=exe_name, model_ws=ws)
    flopy.modpath.Modpath7Bas(mp, porosity=0.1,
                              defaultiface=defaultiface6)
    flopy.modpath.Modpath7Sim(mp, simulationtype='endpoint',
                              trackingdirection='forward',
                              weaksinkoption='pass_through',
                              weaksourceoption='pass_through',
                              referencetime=0.,
                              stoptimeoption='extend',
                              zonedataoption='on', zones=zones,
                              particlegroups=particlegroups)

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

    # test default modpath
    test_default_modpath()

    # build face particles
    test_faceparticles_is1()
    test_facenode_is3()
    test_facenode_is3a()

    # build cell particles
    test_cellparticles_is1()
    test_cellparticleskij_is1()
    test_cellnode_is2a()
    test_cellnode_is3()
    test_cellnode_is3a()

    # compare endpoint results
    test_face_endpoint_output()
    test_cell_endpoint_output()
