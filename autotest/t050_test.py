import shutil
import os
import flopy
from flopy.export import vtk

# Test vtk export
# Note: initially thought about asserting that exported file size in bytes is
# unchanged, but this seems to be sensitive to the running environment.
# Thus, only asserting that the number of lines is unchanged.
# Still keeping the file size check commented for now.

# create output directory
cpth = os.path.join('temp', 't050')
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
os.makedirs(cpth)

def count_lines_in_file(filepath):
    f = open(filepath, 'r')
    n = len(f.readlines())
    return n

def count_lines_in_file_bin(filepath):
    f = open(filepath, 'rb')
    n = len(f.readlines())
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
    # assert(totalbytes==352026)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==2846)

    # with smoothing
    m.dis.top.export(output_dir, fmt='vtk', name='top_smooth', smooth=True)
    filetocheck = os.path.join(output_dir, 'top_smooth.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==351829)
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
    # assert(totalbytes==992576)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==8486)

    # with point scalars
    m.upw.hk.export(output_dir, fmt='vtk', name='hk_points',
                    point_scalars=True)
    filetocheck = os.path.join(output_dir, 'hk_points.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==1321292)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==10605)

    # with point scalars and binary
    m.upw.hk.export(output_dir, fmt='vtk', name='hk_points_bin',
                    point_scalars=True, binary=True)
    filetocheck = os.path.join(output_dir, 'hk_points_bin.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==637861)
    nlines2 = count_lines_in_file_bin(filetocheck)
    assert(nlines2==1848)

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

    # export and check
    r = m.rch.rech
    m.rch.rech.export(output_dir, fmt='vtk')
    filetocheck = os.path.join(output_dir, 'rech_01.vtu')
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==355324)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==2851)
    filetocheck = os.path.join(output_dir, 'rech_01097.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==354622)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==2851)

    # with binary
    m.rch.rech.export(output_dir_bin, fmt='vtk', binary=True)
    filetocheck = os.path.join(output_dir_bin, 'rech_01.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==168339)
    nlines2 = count_lines_in_file_bin(filetocheck)
    assert(nlines2==762)
    filetocheck = os.path.join(output_dir_bin, 'rech_01097.vtu')
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==168339)
    nlines3 = count_lines_in_file_bin(filetocheck)
    assert(nlines3==762)

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
    # assert(totalbytes==1010527)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==8496)

    # upw with point scalar output
    output_dir = os.path.join(cpth, 'UPW')
    m.upw.export(output_dir, fmt='vtk', point_scalars=True)
    filetocheck = os.path.join(output_dir, 'UPW.vtu')
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==2485295)
    nlines1 = count_lines_in_file(filetocheck)
    assert(nlines1==21215)

    # bas with smoothing on
    output_dir = os.path.join(cpth, 'BAS')
    m.bas6.export(output_dir, fmt='vtk', smooth=True)
    filetocheck = os.path.join(output_dir, 'BAS6.vtu')
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==1002054)
    nlines2 = count_lines_in_file(filetocheck)
    assert(nlines2==8491)

    # transient package drain
    output_dir = os.path.join(cpth, 'DRN')
    m.drn.export(output_dir, fmt='vtk')
    filetocheck = os.path.join(output_dir, 'DRN_01.vtu')
    # totalbytes3 = os.path.getsize(filetocheck)
    # assert(totalbytes3==20702)
    nlines3 = count_lines_in_file(filetocheck)
    assert(nlines3==191)
    filetocheck = os.path.join(output_dir, 'DRN_01097.vtu')
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==20702)
    nlines4 = count_lines_in_file(filetocheck)
    assert(nlines4==191)

    # dis with binary
    output_dir = os.path.join(cpth, 'DIS_bin')
    m.dis.export(output_dir, fmt='vtk', binary=True)
    filetocheck = os.path.join(output_dir, 'DIS.vtu')
    # totalbytes5 = os.path.getsize(filetocheck)
    # assert(totalbytes5==536436)
    nlines5 = count_lines_in_file_bin(filetocheck)
    assert(nlines5==1545)

    # upw with point scalars and binary
    output_dir = os.path.join(cpth, 'UPW_bin')
    m.upw.export(output_dir, fmt='vtk', point_scalars=True, binary=True)
    filetocheck = os.path.join(output_dir, 'UPW.vtu')
    # totalbytes6 = os.path.getsize(filetocheck)
    # assert(totalbytes6==1400561)
    nlines6 = count_lines_in_file_bin(filetocheck)
    assert(nlines6==1868)

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
    # assert(totalbytes==993755)
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
    # assert(totalbytes1==1332346)
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
    # assert(totalbytes2==993551)
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
    # assert(totalbytes3==502313)
    nlines3 = count_lines_in_file_bin(filetocheck)
    assert(nlines3==1538)

    # with smoothing and binary, single time
    otfolder = os.path.join(cpth, 'heads_test_4')
    vtk.export_heads(m, hdsfile, otfolder, kstpkper=(0, 0),
                     point_scalars=False, smooth=True, binary=True,
                     nanval=-999.99)
    filetocheck = os.path.join(otfolder, 'freyberg_Heads_KPER1_KSTP1.vtu')
    # totalbytes4 = os.path.getsize(filetocheck)
    # assert(totalbytes4==502313)
    nlines4 = count_lines_in_file_bin(filetocheck)
    assert(nlines4==1528)

    return

def test_vtk_cbc():
    # test mf 2005 freyberg
    mpth = os.path.join('..', 'examples', 'data',
                        'freyberg_multilayer_transient')
    namfile = 'freyberg.nam'
    cbcfile = os.path.join(mpth, 'freyberg.cbc')
    m = flopy.modflow.Modflow.load(namfile, model_ws=mpth, verbose=False)
    filenametocheck = 'freyberg_CBC_KPER1_KSTP1.vtu'

    # export and check with point scalar
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==2496132)
    nlines = count_lines_in_file(filetocheck)
    assert(nlines==19093)

    # with point scalars and binary
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST_bin')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=[(0, 0), (0, 1), (0, 2)], point_scalars=True,
                   binary=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes1 = os.path.getsize(filetocheck)
    # assert(totalbytes1==1248118)
    nlines1 = count_lines_in_file_bin(filetocheck)
    assert(nlines1==2609)

    # with point scalars and binary, only one budget component
    otfolder = os.path.join(cpth, 'freyberg_CBCTEST_bin2')
    vtk.export_cbc(m, cbcfile, otfolder,
                   kstpkper=(0, 0), text='CONSTANT HEAD',
                   point_scalars=True,  binary=True)
    filetocheck = os.path.join(otfolder, filenametocheck)
    # totalbytes2 = os.path.getsize(filetocheck)
    # assert(totalbytes2==10262)
    nlines2 = count_lines_in_file_bin(filetocheck)
    assert(nlines2==61)

    return

if __name__ == '__main__':
    test_vtk_export_array2d()
    test_vtk_export_array3d()
    test_vtk_transient_array_2d()
    test_vtk_export_packages()
    test_vtk_mf6()
    test_vtk_binary_head_export()
    test_vtk_cbc()
