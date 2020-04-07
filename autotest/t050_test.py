import shutil
import numpy as np
import os
import flopy
from flopy.export import vtk

# Test vtk export
# Note: initially thought about asserting that exported file size in bytes is
# unchanged, but this seems to be sensitive to the running environment.
# Thus, only asserting that the number of lines is unchanged.
# Still keeping the file size check commented for development purposes.

# create output directory
cpth = os.path.join('temp', 't050')
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
os.makedirs(cpth)

def count_lines_in_file(filepath, binary=False):
    if binary:
        f = open(filepath, 'rb')
    else:
        f = open(filepath, 'r')
    # note this does not mean much for a binary file but still allows for check
    n = len(f.readlines())
    f.close()
    return n

def test_vtk_export_array2d():
    # test mf 2005 freyberg
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6'])
    output_dir = os.path.join(cpth, 'array_2d_test')

    # export and check
    m.dis.top.export(output_dir, name='top', fmt='vtk')
    filetocheck = os.path.join(output_dir, 'top.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==351846)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==2846)

    # with smoothing
    m.dis.top.export(output_dir, fmt='vtk', name='top_smooth', smooth=True)
    filetocheck = os.path.join(output_dir, 'top_smooth.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==351715)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==2846)

    return

def test_vtk_export_array3d():
    # test mf 2005 freyberg
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6', 'upw'])
    output_dir = os.path.join(cpth, 'array_3d_test')

    # export and check
    m.upw.hk.export(output_dir, fmt='vtk', name='hk')
    filetocheck = os.path.join(output_dir, 'hk.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==992036)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==8486)

    # with point scalars
    m.upw.hk.export(output_dir, fmt='vtk', name='hk_points',
                    point_scalars=True)
    filetocheck = os.path.join(output_dir, 'hk_points.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==1320666)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==10605)

    # with point scalars and binary
    m.upw.hk.export(output_dir, fmt='vtk', name='hk_points_bin',
                    point_scalars=True, binary=True)
    filetocheck = os.path.join(output_dir, 'hk_points_bin.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==629401)
    # nlines2 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines2==2105)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_transient_array_2d():
    # test mf 2005 freyberg
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6', 'rch'])
    output_dir = os.path.join(cpth, 'transient_2d_test')
    output_dir_bin = os.path.join(cpth, 'transient_2d_test_bin')
    kpers = [0, 1, 1096]

    # export and check
    m.rch.rech.export(output_dir, fmt='vtk', kpers=kpers)
    filetocheck = os.path.join(output_dir, 'rech_01.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==355144)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==2851)
    filetocheck = os.path.join(output_dir, 'rech_01097.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==354442)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==2851)

    # with binary
    m.rch.rech.export(output_dir_bin, fmt='vtk', binary=True, kpers=kpers)
    filetocheck = os.path.join(output_dir_bin, 'rech_01.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==168339)
    # nlines2 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines2==846)
    assert(os.path.exists(filetocheck))
    filetocheck = os.path.join(output_dir_bin, 'rech_01097.vtu')
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==168339)
    # nlines3 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines3==846)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_export_packages():
    # test mf 2005 freyberg
    mpath = os.path.join('..', 'examples', 'data',
                         'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6', 'upw', 'DRN'])

    # dis export and check
    output_dir = os.path.join(cpth, 'DIS')
    m.dis.export(output_dir, fmt='vtk')
    filetocheck = os.path.join(output_dir, 'DIS.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==1019857)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==8496)

    # upw with point scalar output
    output_dir = os.path.join(cpth, 'UPW')
    m.upw.export(output_dir, fmt='vtk', point_scalars=True)
    filetocheck = os.path.join(output_dir, 'UPW.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==2559173)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==21215)

    # bas with smoothing on
    output_dir = os.path.join(cpth, 'BAS')
    m.bas6.export(output_dir, fmt='vtk', smooth=True)
    filetocheck = os.path.join(output_dir, 'BAS6.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==1001580)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==8491)

    # transient package drain
    kpers = [0, 1, 1096]
    output_dir = os.path.join(cpth, 'DRN')
    m.drn.export(output_dir, fmt='vtk', kpers=kpers)
    filetocheck = os.path.join(output_dir, 'DRN_01.vtu')
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==20670)
    nlines3 = count_lines_in_file(filetocheck)
    assert(nlines3==191)
    filetocheck = os.path.join(output_dir, 'DRN_01097.vtu')
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==20670)
    nlines4 = count_lines_in_file(filetocheck)
    assert(nlines4==191)

    # dis with binary
    output_dir = os.path.join(cpth, 'DIS_bin')
    m.dis.export(output_dir, fmt='vtk', binary=True)
    filetocheck = os.path.join(output_dir, 'DIS.vtu')
    # totalbytes5 = os.path.getsize(filetocheck)
    # assert(totalbytes5==519516)
    # nlines5 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines5==1797)
    assert(os.path.exists(filetocheck))

    # upw with point scalars and binary
    output_dir = os.path.join(cpth, 'UPW_bin')
    m.upw.export(output_dir, fmt='vtk', point_scalars=True, binary=True)
    filetocheck = os.path.join(output_dir, 'UPW.vtu')
    # totalbytes6 = os.path.getsize(filetocheck)
    # assert(totalbytes6==1349801)
    # nlines6 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines6==4240)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_mf6():
    # test mf6
    mf6expth = os.path.join('..', 'examples', 'data', 'mf6')
    mf6sims = ['test045_lake1ss_table', 'test036_twrihfb', 'test045_lake2tr',
               'test006_2models_mvr']

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

    # check one
    filetocheck = os.path.join(cpth, 'twrihfb2015', 'npf.vtr')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==21609)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==76)

    return


def test_vtk_binary_head_export():
    # test mf 2005 freyberg
    mpth = os.path.join('..', 'examples', 'data',
                        'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    hdsfile = os.path.join(mpth, 'freyberg.hds')
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpth, verbose=False,
                                   load_only=['dis', 'bas6'])
    filenametocheck = 'freyberg_Heads_KPER455_KSTP1.vtu'

    # export and check
    otfolder = os.path.join(cpth, 'heads_test')
    vtk.export_heads(m, hdsfile, otfolder, nanval=-999.99, kstpkper=[(0, 0),
                                                                     (0, 199),
                                                                     (0, 354),
                                                                     (0, 454),
                                                                     (0,
                                                                      1089)])
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==993215)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==8486)

    # with point scalars
    otfolder = os.path.join(cpth, 'heads_test_1')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     point_scalars=True, nanval=-999.99)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==1331858)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==10605)

    # with smoothing
    otfolder = os.path.join(cpth, 'heads_test_2')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     smooth=True, nanval=-999.99)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==993077)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==8486)

    # with smoothing and binary
    otfolder = os.path.join(cpth, 'heads_test_3')
    vtk.export_heads(m, hdsfile, otfolder,
                     kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0,
                                                                      1089)],
                     smooth=True, binary=True, nanval=-999.99)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==493853)
    # nlines3 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines3==1781)
    assert(os.path.exists(filetocheck))

    # with smoothing and binary, single time
    otfolder = os.path.join(cpth, 'heads_test_4')
    vtk.export_heads(m, hdsfile, otfolder, kstpkper=(0, 0),
                     point_scalars=False, smooth=True, binary=True,
                     nanval=-999.99)
    filetocheck = os.path.join(otfolder, 'freyberg_Heads_KPER1_KSTP1.vtu')
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==493853)
    # nlines4 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines4==1787)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_cbc():
    # test mf 2005 freyberg
    mpth = os.path.join('..', 'examples', 'data',
                        'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    cbcfile = os.path.join(mpth, 'freyberg.cbc')
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpth, verbose=False,
                                   load_only=['dis', 'bas6'])
    filenametocheck = 'freyberg_CBC_KPER1_KSTP1.vtu'

    # export and check with point scalar
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==2630875)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==19093)

    # with point scalars and binary
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST_bin')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True,
                   binary=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==1205818)
    # nlines1 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines1==3088)
    assert(os.path.exists(filetocheck))

    # with point scalars and binary, only one budget component
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST_bin2')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=(0, 0), text='CONSTANT HEAD',
                   point_scalars=True,  binary=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==10142)
    # nlines2 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines2==66)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_vector():
    from flopy.utils import postprocessing as pp
    # test mf 2005 freyberg
    mpth = os.path.join('..', 'examples', 'data',
                        'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    cbcfile = os.path.join(mpth, 'freyberg.cbc')
    hdsfile = os.path.join(mpth, 'freyberg.hds')
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpth, verbose=False,
                                   load_only=['dis', 'bas6', 'upw'])
    q = pp.get_specific_discharge(m, cbcfile=cbcfile)
    output_dir = os.path.join(cpth, 'freyberg_vector')
    filenametocheck = 'discharge.vtu'

    # export and check with point scalar
    vtk.export_vector(m, q, output_dir, 'discharge', point_scalars=True)
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==2247857)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==10605)

    # with point scalars and binary
    vtk.export_vector(m, q, output_dir + '_bin', 'discharge',
                      point_scalars=True, binary=True)
    filetocheck = os.path.join(output_dir + '_bin', filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==942413)
    # nlines1 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines1==3824)
    assert(os.path.exists(filetocheck))

    # with values directly given at vertices
    q = pp.get_specific_discharge(m, cbcfile=cbcfile, hdsfile=hdsfile,
                                  position='vertices')
    nancount = np.count_nonzero(np.isnan(q[0]))
    assert(nancount==308)
    overall = np.nansum(q[0]) + np.nansum(q[1]) + np.nansum(q[2])
    assert np.allclose(overall, -15.467904755216372)
    output_dir = os.path.join(cpth, 'freyberg_vector')
    filenametocheck = 'discharge_verts.vtu'
    vtk.export_vector(m, q, output_dir, 'discharge_verts')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==1990047)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==10598)

    # with values directly given at vertices and binary
    vtk.export_vector(m, q, output_dir + '_bin', 'discharge_verts',
                      binary=True)
    filetocheck = os.path.join(output_dir + '_bin', filenametocheck)
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==891486)
    # nlines3 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines3==3012)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_vti():
    # create model with regular and equal grid spacing in x, y and z directions
    name = 'test_vti'
    m = flopy.modflow.Modflow(name)
    nlay, nrow, ncol = 2, 3, 4
    delr = np.ones(ncol)
    delc = np.ones(nrow)
    top = 2. * np.ones((nrow, ncol))
    botm1 = np.ones((1, nrow, ncol))
    botm2 = np.zeros((1, nrow, ncol))
    botm = np.concatenate((botm1, botm2))
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=top, botm=botm)
    output_dir = os.path.join(cpth, m.name)
    filenametocheck = 'DIS.vti'

    # export and check
    dis.export(output_dir, fmt='vtk')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==1075)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==21)

    # with point scalar
    dis.export(output_dir + '_points', fmt='vtk', point_scalars=True)
    filetocheck = os.path.join(output_dir + '_points', filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==2474)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==38)

    # with binary
    dis.export(output_dir + '_bin', fmt='vtk', binary=True)
    filetocheck = os.path.join(output_dir + '_bin', filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==1144)
    # nlines2 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines2==18)
    assert(os.path.exists(filetocheck))

    # force .vtr
    filenametocheck = 'DIS.vtr'
    dis.export(output_dir, fmt='vtk', vtk_grid_type='RectilinearGrid')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==1606)
    nlines3 = count_lines_in_file(filetocheck)
    assert(nlines3==41)

    # force .vtu
    filenametocheck = 'DIS.vtu'
    dis.export(output_dir, fmt='vtk', vtk_grid_type='UnstructuredGrid')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==5723)
    nlines4 = count_lines_in_file(filetocheck)
    assert(nlines4==129)

    # vector
    filenametocheck = 'vect.vti'
    ones_array = np.ones(m.modelgrid.shape)
    v = (ones_array, 2.*ones_array, 3.*ones_array)
    vtk.export_vector(m, v, output_dir, 'vect', point_scalars=True)
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes5 = os.path.getsize(filetocheck)
    # assert(totalbytes5==1578)
    nlines5 = count_lines_in_file(filetocheck)
    assert(nlines5==20)

    # vector with point scalars and binary
    vtk.export_vector(m, v, output_dir + '_bin', 'vect', point_scalars=True,
                      binary=True)
    filetocheck = os.path.join(output_dir + '_bin', filenametocheck)
    # totalbytes6 = os.path.getsize(filetocheck)
    # assert(totalbytes6==2666)
    # nlines6 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines6==18)
    assert(os.path.exists(filetocheck))

    return

def test_vtk_vtr():
    # test mf 2005 l1a2k
    mpth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    namfile = 'l1a2k.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpth, verbose=False)
    output_dir = os.path.join(cpth, m.name)
    filenametocheck = 'EVT_01.vtr'

    # export and check
    m.export(output_dir, fmt='vtk')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==79953)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==87)

    # with point scalar
    m.export(output_dir + '_points', fmt='vtk', point_scalars=True)
    filetocheck = os.path.join(output_dir + '_points', filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==182168)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==121)

    # with binary
    m.export(output_dir + '_bin', fmt='vtk', binary=True)
    filetocheck = os.path.join(output_dir + '_bin', filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==47874)
    # nlines2 = count_lines_in_file(filetocheck, binary=True)
    # assert(nlines2==28)
    assert(os.path.exists(filetocheck))

    # force .vtu
    filenametocheck = 'EVT_01.vtu'
    m.export(output_dir, fmt='vtk', vtk_grid_type='UnstructuredGrid')
    filetocheck = os.path.join(output_dir, filenametocheck)
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==78762)
    nlines3 = count_lines_in_file(filetocheck)
    assert(nlines3==1105)

    return

def test_vtk_export_true2d_regular():
    mpath = os.path.join('..', 'examples', 'data', 'mf2005_test')
    output_dir = os.path.join(cpth, 'true2d_regular')

    # test mf 2005 test1ss, which has one layer with non-constant elevations
    namfile = 'test1ss.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6'])

    # export and check (.vti, with point scalars)
    m.dis.botm.export(output_dir, name='test1ss_botm', fmt='vtk',
                      point_scalars=True, true2d=True)
    filetocheck = os.path.join(output_dir, 'test1ss_botm.vti')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==3371)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==32)

    # vector (.vti, with point scalars)
    vect = (m.dis.botm.array, m.dis.botm.array)
    vtk.export_vector(m, vect, output_dir, 'test1ss_botm_vect',
                      point_scalars=True, true2d=True)
    filetocheck = os.path.join(output_dir, 'test1ss_botm_vect.vti')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==6022)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==32)

    # vector directly at vertices (.vti)
    vect = [m.dis.botm.array, m.dis.botm.array]
    for i, vcomp in enumerate(vect):
        vect[i] = m.modelgrid.array_at_verts(vcomp)
    vtk.export_vector(m, vect, output_dir, 'test1ss_botm_vectv', true2d=True)
    filetocheck = os.path.join(output_dir, 'test1ss_botm_vectv.vti')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==3496)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==27)

    # export and check (force .vtu, with point scalars)
    m.dis.botm.export(output_dir, name='test1ss_botm', fmt='vtk',
                      point_scalars=True, vtk_grid_type='UnstructuredGrid',
                      true2d=True)
    filetocheck = os.path.join(output_dir, 'test1ss_botm.vtu')
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==23827)
    nlines3 = count_lines_in_file(filetocheck)
    assert(nlines3==608)

    # test mf 2005 swiex3, which has one row
    namfile = 'swiex3.nam'
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpath, verbose=False,
                                   load_only=['dis', 'bas6'])

    # export and check (.vtr)
    m.dis.botm.export(output_dir, name='swiex3_botm', fmt='vtk', true2d=True)
    filetocheck = os.path.join(output_dir, 'swiex3_botm.vtr')
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==8022)
    nlines4 = count_lines_in_file(filetocheck)
    assert(nlines4==229)

    # export and check (force .vtu)
    m.dis.botm.export(output_dir, name='swiex3_botm', fmt='vtk',
                      vtk_grid_type='UnstructuredGrid', true2d=True)
    filetocheck = os.path.join(output_dir, 'swiex3_botm.vtu')
    # totalbytes5 = os.path.getsize(filetocheck)
    # assert(totalbytes5==85446)
    nlines5 = count_lines_in_file(filetocheck)
    assert(nlines5==2426)

    return

def test_vtk_export_true2d_nonregxy():
    import flopy.utils.binaryfile as bf
    from flopy.utils import postprocessing as pp
    output_dir = os.path.join(cpth, 'true2d_nonregxy')
    cbc_unit_nb = 53

    # model with one layer, non-regular grid in x and y
    name = 'nonregxy'
    m = flopy.modflow.Modflow(name, model_ws=output_dir, exe_name='mf2005')
    nlay, nrow, ncol = 1, 10, 10
    delr = np.concatenate((np.ones((5,)), 2.*np.ones((5,))))
    delc = delr
    top = 50.
    botm = 0.
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=top, botm=botm)
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.linspace(1., 0., ncol).reshape(1, 1, ncol)
    strt = strt * np.ones((nlay, nrow, ncol))
    bas = flopy.modflow.ModflowBas(m, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(m, hk=1., vka=1., ipakcb=cbc_unit_nb)
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, compact=True)
    pcg = flopy.modflow.ModflowPcg(m)
    m.write_input()
    m.run_model(silent=True)

    # export and check head with point scalar
    hdsfile = os.path.join(output_dir, name + '.hds')
    hds = bf.HeadFile(hdsfile)
    head = hds.get_data()
    vtk.export_array(m, head, output_dir, name + '_head', point_scalars=True,
                     true2d=True)
    filetocheck = os.path.join(output_dir, name + '_head.vtr')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==4997)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==59)

    # export and check specific discharge given at vertices
    cbcfile = os.path.join(output_dir, name + '.cbc')
    q = pp.get_specific_discharge(m, cbcfile, position='vertices')
    vtk.export_vector(m, q, output_dir, name + '_q', point_scalars=True,
                      true2d=True)
    filetocheck = os.path.join(output_dir, name + '_q.vtr')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==5772)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==54)

    return

def test_vtk_export_true2d_nonregxz():
    import flopy.utils.binaryfile as bf
    from flopy.utils import postprocessing as pp
    output_dir = os.path.join(cpth, 'true2d_nonregxz')
    cbc_unit_nb = 53

    # model with one row, non-regular grid in x and stepwise z
    name = 'nonregxz'
    m = flopy.modflow.Modflow(name, model_ws=output_dir, exe_name='mf2005')
    nlay, nrow, ncol = 2, 1, 10
    delr = np.concatenate((np.ones((5,)), 2.*np.ones((5,))))
    delc = 1.
    top = np.linspace(2., 3., ncol).reshape((1, 1, ncol))
    botm1 = np.linspace(1., 2.5, ncol).reshape((1, 1, ncol))
    botm2 = np.linspace(0., 0.5, ncol).reshape((1, 1, ncol))
    botm = np.concatenate((botm1, botm2))
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=top, botm=botm)
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.linspace(0., 1., ncol).reshape(1, 1, ncol)
    strt = strt * np.ones((nlay, nrow, ncol))
    bas = flopy.modflow.ModflowBas(m, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(m, hk=1., vka=1., ipakcb=cbc_unit_nb)
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, compact=True)
    pcg = flopy.modflow.ModflowPcg(m)
    m.write_input()
    m.run_model(silent=True)

    # export and check head
    hdsfile = os.path.join(output_dir, name + '.hds')
    hds = bf.HeadFile(hdsfile)
    head = hds.get_data()
    vtk.export_array(m, head, output_dir, name + '_head', true2d=True)
    filetocheck = os.path.join(output_dir, name + '_head.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==4217)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==105)

    # export and check head with point scalar
    hdsfile = os.path.join(output_dir, name + '.hds')
    hds = bf.HeadFile(hdsfile)
    head = hds.get_data()
    vtk.export_array(m, head, output_dir, name + '_head_points',
                     point_scalars=True, true2d=True)
    filetocheck = os.path.join(output_dir, name + '_head_points.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==6155)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==129)

    # export and check specific discharge given at vertices
    cbcfile = os.path.join(output_dir, name + '.cbc')
    q = pp.get_specific_discharge(m, cbcfile, position='vertices')
    vtk.export_vector(m, q, output_dir, name + '_q', point_scalars=True,
                      true2d=True)
    filetocheck = os.path.join(output_dir, name + '_q.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==7036)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==123)

    return

def test_vtk_export_true2d_nonregyz():
    import flopy.utils.binaryfile as bf
    from flopy.utils import postprocessing as pp
    output_dir = os.path.join(cpth, 'true2d_nonregyz')
    cbc_unit_nb = 53

    # model with one col, non-regular grid in y and stepwise z
    name = 'nonregyz'
    m = flopy.modflow.Modflow(name, model_ws=output_dir, exe_name='mf2005')
    nlay, nrow, ncol = 2, 10, 1
    delr = 1.
    delc = np.concatenate((2.*np.ones((5,)), np.ones((5,))))
    top = np.linspace(3., 2., nrow).reshape((1, nrow, 1))
    botm1 = np.linspace(2.5, 1., nrow).reshape((1, nrow, 1))
    botm2 = np.linspace(0.5, 0., nrow).reshape((1, nrow, 1))
    botm = np.concatenate((botm1, botm2))
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=top, botm=botm)
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, 0, :] = -1
    ibound[:, -1, :] = -1
    strt = np.linspace(1., 0., nrow).reshape(1, nrow, 1)
    strt = strt * np.ones((nlay, nrow, ncol))
    bas = flopy.modflow.ModflowBas(m, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(m, hk=1., vka=1., ipakcb=cbc_unit_nb)
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, compact=True)
    pcg = flopy.modflow.ModflowPcg(m)
    m.write_input()
    m.run_model(silent=True)

    # export and check head
    hdsfile = os.path.join(output_dir, name + '.hds')
    hds = bf.HeadFile(hdsfile)
    head = hds.get_data()
    vtk.export_array(m, head, output_dir, name + '_head', true2d=True)
    filetocheck = os.path.join(output_dir, name + '_head.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==4217)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==105)

    # export and check head with point scalar
    hdsfile = os.path.join(output_dir, name + '.hds')
    hds = bf.HeadFile(hdsfile)
    head = hds.get_data()
    vtk.export_array(m, head, output_dir, name + '_head_points',
                     point_scalars=True, true2d=True)
    filetocheck = os.path.join(output_dir, name + '_head_points.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==6155)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==129)

    # export and check specific discharge given at vertices
    cbcfile = os.path.join(output_dir, name + '.cbc')
    q = pp.get_specific_discharge(m, cbcfile, position='vertices')
    vtk.export_vector(m, q, output_dir, name + '_q', point_scalars=True,
                      true2d=True)
    filetocheck = os.path.join(output_dir, name + '_q.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==7032)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==123)

    return

if __name__ == '__main__':
    test_vtk_export_array2d()
    test_vtk_export_array3d()
    test_vtk_transient_array_2d()
    test_vtk_export_packages()
    test_vtk_mf6()
    test_vtk_binary_head_export()
    test_vtk_cbc()
    test_vtk_vector()
    test_vtk_vti()
    test_vtk_vtr()
    test_vtk_export_true2d_regular()
    test_vtk_export_true2d_nonregxy()
    test_vtk_export_true2d_nonregxz()
    test_vtk_export_true2d_nonregyz()
