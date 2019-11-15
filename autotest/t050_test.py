import shutil
import os
import numpy as np
import flopy
from flopy.export import vtk

# create output directory
cpth = os.path.join('temp', 't050')
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
os.makedirs(cpth)

# binary output directory
binot = os.path.join(cpth, 'bin')
if os.path.isdir(binot):
    shutil.rmtree(binot)
os.makedirs(binot)


def test_vtk_export_array2d():
    """Export 2d array"""
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False)
    m.dis.top.export(os.path.join(cpth, 'array_2d_test'), fmt='vtk')
    # with smoothing
    m.dis.top.export(os.path.join(cpth, 'array_2d_test'), fmt='vtk',
                     name='top_smooth', smooth=True)


def test_vtk_export_array3d():
    """Vtk export 3d array"""
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False)
    m.upw.hk.export(os.path.join(cpth, 'array_3d_test'), fmt='vtk')
    # with point scalars
    m.upw.hk.export(os.path.join(cpth, 'array_3d_test'), fmt='vtk',
                    name='hk_points', point_scalars=True)
    # binary test
    m.upw.hk.export(os.path.join(binot, 'array_3d_test'), fmt='vtk',
                    name='hk_points', point_scalars=True, binary=True)

    return


def test_vtk_transient_array_2d():
    """VTK export transient 2d array"""
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False)
    m.rch.rech.export(os.path.join(cpth, 'transient_2d_test'), fmt='vtk')

    # binary test
    m.rch.rech.export(os.path.join(binot, 'transient_2d_test'), fmt='vtk',
                      binary=True)

    return


def test_vtk_export_packages():
    """testing vtk package export"""
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False)
    # test dis export
    m.dis.export(os.path.join(cpth, 'DIS'), fmt='vtk')
    # upw with point scalar output
    m.upw.export(os.path.join(cpth, 'UPW'), fmt='vtk', point_scalars=True)
    # bas with smoothing on
    m.bas6.export(os.path.join(cpth, 'BAS'), fmt='vtk', smooth=True)
    # transient package drain
    m.drn.export(os.path.join(cpth, 'DRN'), fmt='vtk')
    # binary test
    m.dis.export(os.path.join(binot, 'DIS'), fmt='vtk', binary=True)
    # upw with point scalar output
    m.upw.export(os.path.join(binot, 'UPW'), fmt='vtk', point_scalars=True,
                 binary=True)

    return


# add mf2005 model exports
def test_export_mf2005_vtk():
    """test vtk model export mf2005"""
    pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    namfiles = [namfile for namfile in os.listdir(pth) if
                namfile.endswith('.nam')]
    skip = ['bcf2ss.nam']
    for namfile in namfiles:
        if namfile in skip:
            continue
        print('testing namefile', namfile)
        m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
        m.export(os.path.join(cpth, m.name), fmt='vtk')

        # binary test
        m.export(os.path.join(binot, m.name), fmt='vtk', binary=True)

    return


def test_vtk_mf6():
    mf6expth = os.path.join('..', 'examples', 'data', 'mf6')
    # test vtk mf6 export
    mf6sims = ['test045_lake1ss_table',
               'test036_twrihfb', 'test045_lake2tr', 'test006_2models_mvr']
    # mf6sims = ['test005_advgw_tidal']
    # mf6sims = ['test036_twrihfb']

    for simnm in mf6sims:
        print(simnm)
        simpth = os.path.join(mf6expth, simnm)
        loaded_sim = flopy.mf6.MFSimulation.load(simnm, 'mf6', 'mf6',
                                                 simpth)
        sim_models = loaded_sim.model_names
        print(sim_models)
        for mname in sim_models:
            print(mname)
            m = loaded_sim.get_model(mname)
            m.export(os.path.join(cpth, m.name), fmt='vtk')

    return


def test_vtk_binary_head_export():

    """test vet export of heads"""

    freyberg_pth = os.path.join('..', 'examples', 'data',
                                'freyberg_multilayer_transient')

    hdsfile = os.path.join(freyberg_pth, 'freyberg.hds')

    m = flopy.modflow.Modflow.load('freyberg.nam', model_ws=freyberg_pth,
                                   verbose=False)
    otfolder = os.path.join(cpth, 'heads_test')

    vtk.export_heads(m, hdsfile, otfolder, nanval=-999.99, kstpkper=[(0, 0),
                                                                     (0, 199),
                                                                     (0, 354),
                                                                     (0, 454),
                                                                     (0,
                                                                      1089)])
    # test with points
    otfolder = os.path.join(cpth, 'heads_test_1')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     point_scalars=True, nanval=-999.99)

    # test vtk export heads with smoothing and no point scalars
    otfolder = os.path.join(cpth, 'heads_test_2')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     point_scalars=False, smooth=True, nanval=-999.99)

    # test binary output
    otfolder = os.path.join(cpth, 'heads_test_3')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     point_scalars=False, smooth=True, binary=True,
                     nanval=-999.99)

    otfolder = os.path.join(cpth, 'heads_test_4')
    vtk.export_heads(m, hdsfile, otfolder, kstpkper=(0, 0),
                     point_scalars=False,
                     smooth=True, binary=True, nanval=-999.99)

    return


def test_vtk_cbc():
    # test mf 2005 freyberg
    freyberg_cbc = os.path.join('..', 'examples', 'data',
                                'freyberg_multilayer_transient',
                                'freyberg.cbc')

    freyberg_mpth = os.path.join('..', 'examples', 'data',
                                 'freyberg_multilayer_transient')

    m = flopy.modflow.Modflow.load('freyberg.nam', model_ws=freyberg_mpth,
                                   verbose=False)

    vtk.export_cbc(m, freyberg_cbc, os.path.join(cpth, 'freyberg_CBCTEST'),
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True)

    vtk.export_cbc(m, freyberg_cbc, os.path.join(cpth, 'freyberg_CBCTEST_bin'),
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True,
                   binary=True)

    vtk.export_cbc(m, freyberg_cbc, os.path.join(cpth,
                                                 'freyberg_CBCTEST_bin2'),
                   kstpkper=(0, 0), text='CONSTANT HEAD',
                   point_scalars=True,  binary=True)

    return


if __name__ == '__main__':
    test_vtk_export_array2d()
    test_vtk_export_array3d()
    test_vtk_transient_array_2d()
    test_vtk_export_packages()
    test_export_mf2005_vtk()
    test_vtk_mf6()
    test_vtk_binary_head_export()
    test_vtk_cbc()
