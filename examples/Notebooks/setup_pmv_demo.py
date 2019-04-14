import matplotlib as mpl
import numpy as np
import os
import platform
import sys


def run():
    # run installed version of flopy or add local path
    try:
        import flopy
    except:
        fpth = os.path.abspath(os.path.join('..', '..'))
        sys.path.append(fpth)
        import flopy

    # Set name of MODFLOW exe
    #  assumes executable is in users path statement
    version = 'mf2005'
    exe_name = 'mf2005'
    exe_mp = 'mp6'
    if platform.system() == 'Windows':
        exe_name += '.exe'
        exe_mp += '.exe'
    mfexe = exe_name

    # Set the paths
    loadpth = os.path.join('..', 'data', 'freyberg')
    modelpth = os.path.join('data')

    # make sure modelpth directory exists
    if not os.path.exists(modelpth):
        os.makedirs(modelpth)

    ml = flopy.modflow.Modflow.load('freyberg.nam', model_ws=loadpth,
                                    exe_name=exe_name, version=version)
    ml.change_model_ws(new_pth=modelpth)
    ml.write_input()
    success, buff = ml.run_model()
    if not success:
        print('Something bad happened.')
    files = ['freyberg.hds', 'freyberg.cbc']
    for f in files:
        if os.path.isfile(os.path.join(modelpth, f)):
            msg = 'Output file located: {}'.format(f)
            print(msg)
        else:
            errmsg = 'Error. Output file cannot be found: {}'.format(f)
            print(errmsg)

    mp = flopy.modpath.Modpath('freybergmp', exe_name=exe_mp, modflowmodel=ml, model_ws=modelpth)
    mpbas = flopy.modpath.ModpathBas(mp, hnoflo=ml.bas6.hnoflo, hdry=ml.lpf.hdry,
                                     ibound=ml.bas6.ibound.array, prsity=0.2, prsityCB=0.2)
    sim = mp.create_mpsim(trackdir='forward', simtype='endpoint', packages='RCH')
    mp.write_input()
    mp.run_model()

    mpp = flopy.modpath.Modpath('freybergmpp', exe_name=exe_mp, modflowmodel=ml, model_ws=modelpth)
    mpbas = flopy.modpath.ModpathBas(mpp, hnoflo=ml.bas6.hnoflo, hdry=ml.lpf.hdry,
                                     ibound=ml.bas6.ibound.array, prsity=0.2, prsityCB=0.2)
    sim = mpp.create_mpsim(trackdir='backward', simtype='pathline', packages='WEL')
    mpp.write_input()
    mpp.run_model()


    ## load and run second example
    # run installed version of flopy or add local path
    try:
        import flopy
    except:
        fpth = os.path.abspath(os.path.join('..', '..'))
        sys.path.append(fpth)
        import flopy

        print(sys.version)
    print('numpy version: {}'.format(np.__version__))
    print('matplotlib version: {}'.format(mpl.__version__))
    print('flopy version: {}'.format(flopy.__version__))

    if not os.path.exists("data"):
        os.mkdir("data")

    from flopy.utils.gridgen import Gridgen
    Lx = 10000.
    Ly = 10500.
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, 0]

    ms = flopy.modflow.Modflow()
    dis5 = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr,
                                    delc=delc, top=top, botm=botm)

    model_name = 'mp7p2'
    model_ws = os.path.join('data', 'mp7_ex2', 'mf6')
    gridgen_ws = os.path.join(model_ws, 'gridgen')
    g = Gridgen(dis5, model_ws=gridgen_ws)

    rf0shp = os.path.join(gridgen_ws, 'rf0')
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 1, range(nlay))

    rf1shp = os.path.join(gridgen_ws, 'rf1')
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 2, range(nlay))

    rf2shp = os.path.join(gridgen_ws, 'rf2')
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 3, range(nlay))

    g.build(verbose=False)

    gridprops = g.get_gridprops_disv()
    ncpl = gridprops['ncpl']
    top = gridprops['top']
    botm = gridprops['botm']
    nvert = gridprops['nvert']
    vertices = gridprops['vertices']
    cell2d = gridprops['cell2d']
    # cellxy = gridprops['cellxy']

    # create simulation
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', exe_name='mf6',
                                 sim_ws=model_ws)

    # create tdis package
    tdis_rc = [(1000.0, 1, 1.0)]
    tdis = flopy.mf6.ModflowTdis(sim, pname='tdis', time_units='DAYS',
                                 perioddata=tdis_rc)

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name,
                               model_nam_file='{}.nam'.format(model_name))
    gwf.name_file.save_flows = True

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(sim, pname='ims', print_option='SUMMARY',
                               complexity='SIMPLE', outer_hclose=1.e-5,
                               outer_maximum=100, under_relaxation='NONE',
                               inner_maximum=100, inner_hclose=1.e-6,
                               rcloserecord=0.1, linear_acceleration='BICGSTAB',
                               scaling_method='NONE', reordering_method='NONE',
                               relaxation_factor=0.99)
    sim.register_ims_package(ims, [gwf.name])

    # disv
    disv = flopy.mf6.ModflowGwfdisv(gwf, nlay=nlay, ncpl=ncpl,
                                    top=top, botm=botm,
                                    nvert=nvert, vertices=vertices,
                                    cell2d=cell2d)

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, pname='ic', strt=320.)

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=[('xt3d')],
                                  icelltype=[1, 0, 0],
                                  k=[50.0, 0.01, 200.0],
                                  k33=[10., 0.01, 20.])

    # wel
    wellpoints = [(4750., 5250.)]
    welcells = g.intersect(wellpoints, 'point', 0)
    # welspd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, maxbound=1, aux_vars=['iface'])
    welspd = [[(2, icpl), -150000, 0] for icpl in welcells['nodenumber']]
    wel = flopy.mf6.ModflowGwfwel(gwf, print_input=True,
                                  auxiliary=[('iface',)],
                                  stress_period_data=welspd)

    # rch
    aux = [np.ones(ncpl, dtype=np.int) * 6]
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=0.005,
                                   auxiliary=[('iface',)],
                                   aux={0: [6]})
    # riv
    riverline = [[[(Lx - 1., Ly), (Lx - 1., 0.)]]]
    rivcells = g.intersect(riverline, 'line', 0)
    rivspd = [[(0, icpl), 320., 100000., 318] for icpl in rivcells['nodenumber']]
    riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=rivspd)

    # output control
    oc = flopy.mf6.ModflowGwfoc(gwf, pname='oc', budget_filerecord='{}.cbb'.format(model_name),
                                head_filerecord='{}.hds'.format(model_name),
                                headprintrecord=[('COLUMNS', 10, 'WIDTH', 15,
                                                  'DIGITS', 6, 'GENERAL')],
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                                printrecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])

    sim.write_simulation()
    sim.run_simulation()

    mp_namea = model_name + 'a_mp'
    mp_nameb = model_name + 'b_mp'

    pcoord = np.array([[0.000, 0.125, 0.500],
                       [0.000, 0.375, 0.500],
                       [0.000, 0.625, 0.500],
                       [0.000, 0.875, 0.500],
                       [1.000, 0.125, 0.500],
                       [1.000, 0.375, 0.500],
                       [1.000, 0.625, 0.500],
                       [1.000, 0.875, 0.500],
                       [0.125, 0.000, 0.500],
                       [0.375, 0.000, 0.500],
                       [0.625, 0.000, 0.500],
                       [0.875, 0.000, 0.500],
                       [0.125, 1.000, 0.500],
                       [0.375, 1.000, 0.500],
                       [0.625, 1.000, 0.500],
                       [0.875, 1.000, 0.500]])
    nodew = gwf.disv.ncpl.array * 2 + welcells['nodenumber'][0]
    plocs = [nodew for i in range(pcoord.shape[0])]

    # create particle data
    pa = flopy.modpath.ParticleData(plocs, structured=False,
                                    localx=pcoord[:, 0],
                                    localy=pcoord[:, 1],
                                    localz=pcoord[:, 2],
                                    drape=0)

    # create backward particle group
    fpth = mp_namea + '.sloc'
    pga = flopy.modpath.ParticleGroup(particlegroupname='BACKWARD1', particledata=pa,
                                      filename=fpth)

    facedata = flopy.modpath.FaceDataType(drape=0,
                                          verticaldivisions1=10, horizontaldivisions1=10,
                                          verticaldivisions2=10, horizontaldivisions2=10,
                                          verticaldivisions3=10, horizontaldivisions3=10,
                                          verticaldivisions4=10, horizontaldivisions4=10,
                                          rowdivisions5=0, columndivisions5=0,
                                          rowdivisions6=4, columndivisions6=4)
    pb = flopy.modpath.NodeParticleData(subdivisiondata=facedata, nodes=nodew)
    # create forward particle group
    fpth = mp_nameb + '.sloc'
    pgb = flopy.modpath.ParticleGroupNodeTemplate(particlegroupname='BACKWARD2',
                                                  particledata=pb,
                                                  filename=fpth)

    # create modpath files
    mp = flopy.modpath.Modpath7(modelname=mp_namea, flowmodel=gwf,
                                exe_name='mp7', model_ws=model_ws)
    flopy.modpath.Modpath7Bas(mp, porosity=0.1)
    flopy.modpath.Modpath7Sim(mp, simulationtype='combined',
                              trackingdirection='backward',
                              weaksinkoption='pass_through',
                              weaksourceoption='pass_through',
                              referencetime=0.,
                              stoptimeoption='extend',
                              timepointdata=[500, 1000.],
                              particlegroups=pga)

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()

    # create modpath files
    mp = flopy.modpath.Modpath7(modelname=mp_nameb, flowmodel=gwf,
                                exe_name='mp7', model_ws=model_ws)
    flopy.modpath.Modpath7Bas(mp, porosity=0.1)
    flopy.modpath.Modpath7Sim(mp, simulationtype='endpoint',
                              trackingdirection='backward',
                              weaksinkoption='pass_through',
                              weaksourceoption='pass_through',
                              referencetime=0.,
                              stoptimeoption='extend',
                              particlegroups=pgb)

    # write modpath datasets
    mp.write_input()

    # run modpath
    mp.run_model()
    return

if __name__ == "__main__":
    run()