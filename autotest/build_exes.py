# Build the executables that are used in the flopy autotests
import os
import sys
import shutil
import platform
import subprocess
import flopy

try:
    import pymake
except:
    print('pymake is not installed...will not build executables')
    pymake = None

fc = 'gfortran'
cc = 'gcc'
dbleprec = False
# bindir should be in the user path to run flopy tests with appropriate
# executables
#
# by default bindir will be in user directory
# On windows will be C:\\Users\\username\\.local\\bin
# On linux and osx will be /Users/username/.local/bin
bindir = os.path.join(os.path.expanduser('~'), '.local', 'bin')
bindir = os.path.abspath(bindir)
# pass --bindir path/to/directory to define a different bin dir
for ipos, arg in enumerate(sys.argv):
    if arg.lower() == '--bindir':
        bindir = sys.argv[ipos + 1]
    elif arg.lower() == '--dryrun':
        print('will perform dryrun and not build executables')
        pymake = None
print(bindir)
if not os.path.exists(bindir):
    os.makedirs(bindir, exist_ok=True)


def update_mt3dfiles(srcdir):
    # Replace the getcl command with getarg
    f1 = open(os.path.join(srcdir, 'mt3dms5.for'), 'r')
    f2 = open(os.path.join(srcdir, 'mt3dms5.for.tmp'), 'w')
    for line in f1:
        f2.write(line.replace('CALL GETCL(FLNAME)', 'CALL GETARG(1,FLNAME)'))
    f1.close()
    f2.close()
    os.remove(os.path.join(srcdir, 'mt3dms5.for'))
    shutil.move(os.path.join(srcdir, 'mt3dms5.for.tmp'),
                os.path.join(srcdir, 'mt3dms5.for'))

    # Replace filespec with standard fortran
    l = '''
          CHARACTER*20 ACCESS,FORM,ACTION(2)
          DATA ACCESS/'STREAM'/
          DATA FORM/'UNFORMATTED'/
          DATA (ACTION(I),I=1,2)/'READ','READWRITE'/
    '''
    fn = os.path.join(srcdir, 'FILESPEC.INC')
    f = open(fn, 'w')
    f.write(l)
    f.close()

    return


def update_seawatfiles(srcdir):
    # rename all source files to lower case so compilation doesn't
    # bomb on case-sensitive operating systems
    srcfiles = os.listdir(srcdir)
    for filename in srcfiles:
        src = os.path.join(srcdir, filename)
        dst = os.path.join(srcdir, filename.lower())
        os.rename(src, dst)
    return


def update_mf2000files(srcdir):
    # Remove six src folders
    dlist = ['beale2k', 'hydprgm', 'mf96to2k', 'mfpto2k', 'resan2k', 'ycint2k']
    for d in dlist:
        dname = os.path.join(srcdir, d)
        if os.path.isdir(dname):
            print('Removing ', dname)
            shutil.rmtree(os.path.join(srcdir, d))

    # Move src files and serial src file to src directory
    tpth = os.path.join(srcdir, 'mf2k')
    files = [f for f in os.listdir(tpth) if
             os.path.isfile(os.path.join(tpth, f))]
    for f in files:
        shutil.move(os.path.join(tpth, f), srcdir)
    tpth = os.path.join(srcdir, 'mf2k', 'serial')
    files = [f for f in os.listdir(tpth) if
             os.path.isfile(os.path.join(tpth, f))]
    for f in files:
        shutil.move(os.path.join(tpth, f), srcdir)

    # Remove mf2k directory in source directory
    tpth = os.path.join(srcdir, 'mf2k')
    shutil.rmtree(tpth)


def update_mp6files(srcdir):
    fname1 = os.path.join(srcdir, 'MP6Flowdata.for')
    f = open(fname1, 'r')

    fname2 = os.path.join(srcdir, 'MP6Flowdata_mod.for')
    f2 = open(fname2, 'w')
    for line in f:
        line = line.replace('CD.QX2', 'CD%QX2')
        f2.write(line)
    f.close()
    f2.close()
    os.remove(fname1)

    fname1 = os.path.join(srcdir, 'MP6MPBAS1.for')
    f = open(fname1, 'r')

    fname2 = os.path.join(srcdir, 'MP6MPBAS1_mod.for')
    f2 = open(fname2, 'w')
    for line in f:
        line = line.replace('MPBASDAT(IGRID)%NCPPL=NCPPL',
                            'MPBASDAT(IGRID)%NCPPL=>NCPPL')
        f2.write(line)
    f.close()
    f2.close()
    os.remove(fname1)


def update_mp7files(srcdir):
    fpth = os.path.join(srcdir, 'StartingLocationReader.f90')
    with open(fpth) as f:
        lines = f.readlines()
    f = open(fpth, 'w')
    for line in lines:
        if 'pGroup%Particles(n)%InitialFace = 0' in line:
            continue
        f.write(line)
    f.close()


def test_build_modflow():
    if pymake is None:
        return
    starget = 'MODFLOW-2005'
    exe_name = 'mf2005'
    dirname = 'MF2005.1_12u'
    url = "https://water.usgs.gov/ogw/modflow/MODFLOW-2005_v1.12.00/MF2005.1_12u.zip"

    build_target(starget, exe_name, url, dirname)

    return


def test_build_mfnwt():
    if pymake is None:
        return
    starget = 'MODFLOW-NWT'
    exe_name = 'mfnwt'
    dirname = 'MODFLOW-NWT_1.1.4'
    url = "http://water.usgs.gov/ogw/modflow-nwt/{0}.zip".format(dirname)

    build_target(starget, exe_name, url, dirname)

    return


def run_cmdlist(cmdlist, cwd='.'):
    proc = subprocess.Popen(cmdlist, shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            cwd=cwd)
    stdout_data, stderr_data = proc.communicate()
    if proc.returncode != 0:
        if isinstance(stdout_data, bytes):
            stdout_data = stdout_data.decode('utf-8')
        if isinstance(stderr_data, bytes):
            stderr_data = stderr_data.decode('utf-8')
        msg = '{} failed\n'.format(cmdlist) + \
              'status code:\n{}\n'.format(proc.returncode) + \
              'stdout:\n{}\n'.format(stdout_data) + \
              'stderr:\n{}\n'.format(stderr_data)
        assert False, msg
    else:
        if isinstance(stdout_data, bytes):
            stdout_data = stdout_data.decode('utf-8')
        print(stdout_data)

    return


def test_build_usg():
    if pymake is None:
        return
    starget = 'MODFLOW-USG'
    exe_name = 'mfusg'
    dirname = 'mfusg1_5'
    url = 'https://water.usgs.gov/water-resources/software/MODFLOW-USG/{0}.zip'.format(dirname)

    build_target(starget, exe_name, url, dirname)
    return


def test_build_mf6():
    if pymake is None:
        return
    starget = 'MODFLOW6'
    exe_name = 'mf6'
    dirname = 'mf6.0.3'
    url = 'https://water.usgs.gov/ogw/modflow/{0}.zip'.format(dirname)

    build_target(starget, exe_name, url, dirname, include_subdirs=True)
    return


def test_build_lgr():
    if pymake is None:
        return
    starget = 'MODFLOW-LGR'
    exe_name = 'mflgr'
    dirname = 'mflgr.2_0'
    url = "https://water.usgs.gov/ogw/modflow-lgr/modflow-lgr-v2.0.0/mflgrv2_0_00.zip"

    build_target(starget, exe_name, url, dirname)
    return


def test_build_mf2000():
    if pymake is None:
        return
    starget = 'MODFLOW-2000'
    exe_name = 'mf2000'
    dirname = 'mf2k.1_19'
    url = "https://water.usgs.gov/nrp/gwsoftware/modflow2000/mf2k1_19_01.tar.gz"

    build_target(starget, exe_name, url, dirname,
                 replace_function=update_mf2000files)
    return


def test_build_mt3dusgs():
    if pymake is None:
        return
    starget = 'MT3D-USGS'
    exe_name = 'mt3dusgs'
    dirname = 'mt3d-usgs_Distribution'
    url = "https://water.usgs.gov/ogw/mt3d-usgs/mt3d-usgs_1.0.zip"

    build_target(starget, exe_name, url, dirname)
    return


def test_build_mt3dms():
    if pymake is None:
        return
    starget = 'MT3DMS'
    exe_name = 'mt3dms'
    dirname = '.'
    url = "http://hydro.geo.ua.edu/mt3d/mt3dms_530.exe"

    build_target(starget, exe_name, url, dirname,
                 srcname=os.path.join('src', 'standard'),
                 verify=False,
                 replace_function=update_mt3dfiles)
    return


def test_build_seawat():
    if pymake is None:
        return
    starget = 'SEAWAT'
    exe_name = 'swt_v4'
    dirname = 'swt_v4_00_05'
    url = "https://water.usgs.gov/ogw/seawat/{0}.zip".format(dirname)

    build_target(starget, exe_name, url, dirname,
                 srcname='source',
                 replace_function=update_seawatfiles,
                 dble=True, keep=True)
    return


def test_build_modpath6():
    if pymake is None:
        return
    starget = 'MODPATH 6'
    exe_name = 'mp6'
    dirname = 'modpath.6_0'
    url = "https://water.usgs.gov/ogw/modpath/archive/modpath_v6.0.01/modpath.6_0_01.zip"

    build_target(starget, exe_name, url, dirname,
                 replace_function=update_mp6files,
                 keep=True)
    return


def test_build_modpath7():
    if pymake is None:
        return
    starget = 'MODPATH 7'
    exe_name = 'mp7'
    dirname = 'modpath_7_2_001'
    url = "https://water.usgs.gov/ogw/modpath/modpath_7_2_001.zip"

    build_target(starget, exe_name, url, dirname, srcname='source',
                 replace_function=update_mp7files,
                 keep=True)
    return


def test_build_gridgen(keep=True):
    if pymake is None:
        return
    starget = 'GRIDGEN'
    exe_name = 'gridgen'
    dirname = 'gridgen.1.0.02'
    url = "https://water.usgs.gov/ogw/gridgen/{}.zip".format(dirname)

    print('Determining if {} needs to be built'.format(starget))
    if platform.system().lower() == 'windows':
        exe_name += '.exe'

    exe_exists = flopy.which(exe_name)
    if exe_exists is not None and keep:
        print('No need to build {}'.format(starget) +
              ' since it exists in the current path')
        return

    # get current directory
    cpth = os.getcwd()

    # create temporary path
    dstpth = os.path.join('tempbin')
    print('create...{}'.format(dstpth))
    if not os.path.exists(dstpth):
        os.makedirs(dstpth)
    os.chdir(dstpth)

    pymake.download_and_unzip(url)

    # clean
    print('Cleaning...{}'.format(exe_name))
    apth = os.path.join(dirname, 'src')
    cmdlist = ['make', 'clean']
    run_cmdlist(cmdlist, apth)

    # build with make
    print('Building...{}'.format(exe_name))
    apth = os.path.join(dirname, 'src')
    cmdlist = ['make', exe_name]
    run_cmdlist(cmdlist, apth)

    # move the file
    src = os.path.join(apth, exe_name)
    dst = os.path.join(bindir, exe_name)
    try:
        shutil.move(src, dst)
    except:
        print('could not move {}'.format(exe_name))

    # change back to original path
    os.chdir(cpth)

    # Clean up downloaded directory
    print('delete...{}'.format(dstpth))
    if os.path.isdir(dstpth):
        shutil.rmtree(dstpth)

    # make sure the gridgen was built
    msg = '{} does not exist.'.format(os.path.relpath(dst))
    assert os.path.isfile(dst), msg

    return


def test_build_triangle(keep=True):
    if pymake is None:
        return
    starget = 'TRIANGLE'
    exe_name = 'triangle'
    dirname = 'triangle'
    url = "http://www.netlib.org/voronoi/{}.zip".format(dirname)

    print('Determining if {} needs to be built'.format(starget))
    if platform.system().lower() == 'windows':
        exe_name += '.exe'

    exe_exists = flopy.which(exe_name)
    if exe_exists is not None and keep:
        print('No need to build {}'.format(starget) +
              ' since it exists in the current path')
        return

    # get current directory
    cpth = os.getcwd()

    # create temporary path
    dstpth = os.path.join('tempbin', 'triangle')
    print('create...{}'.format(dstpth))
    if not os.path.exists(dstpth):
        os.makedirs(dstpth)
    os.chdir(dstpth)

    pymake.download_and_unzip(url)

    srcdir = 'src'
    os.mkdir(srcdir)
    shutil.move('triangle.c', 'src/triangle.c')
    shutil.move('triangle.h', 'src/triangle.h')

    fct, cct = set_compiler(starget)
    pymake.main(srcdir, 'triangle', fct, cct)

    # move the file
    src = os.path.join('.', exe_name)
    dst = os.path.join(bindir, exe_name)
    try:
        shutil.move(src, dst)
    except:
        print('could not move {}'.format(exe_name))

    # change back to original path
    os.chdir(cpth)

    # Clean up downloaded directory
    print('delete...{}'.format(dstpth))
    if os.path.isdir(dstpth):
        shutil.rmtree(dstpth)

    # make sure the gridgen was built
    msg = '{} does not exist.'.format(os.path.relpath(dst))
    assert os.path.isfile(dst), msg

    return


def set_compiler(starget):
    fct = fc
    cct = cc
    # parse command line arguments to see if user specified options
    # relative to building the target
    msg = ''
    for idx, arg in enumerate(sys.argv):
        if arg.lower() == '--ifort':
            if len(msg) > 0:
                msg += '\n'
            msg += '{} - '.format(arg.lower()) + \
                   '{} will be built with ifort.'.format(starget)
            fct = 'ifort'
        elif arg.lower() == '--cl':
            if len(msg) > 0:
                msg += '\n'
            msg += '{} - '.format(arg.lower()) + \
                   '{} will be built with cl.'.format(starget)
            cct = 'cl'
        elif arg.lower() == '--clang':
            if len(msg) > 0:
                msg += '\n'
            msg += '{} - '.format(arg.lower()) + \
                   '{} will be built with clang.'.format(starget)
            cct = 'clang'
    if len(msg) > 0:
        print(msg)

    return fct, cct


def build_target(starget, exe_name, url, dirname, srcname='src',
                 replace_function=None, verify=True, keep=True,
                 dble=dbleprec, include_subdirs=False):
    print('Determining if {} needs to be built'.format(starget))
    if platform.system().lower() == 'windows':
        exe_name += '.exe'

    exe_exists = flopy.which(exe_name)
    if exe_exists is not None and keep:
        print('No need to build {}'.format(starget) +
              ' since it exists in the current path')
        return

    fct, cct = set_compiler(starget)

    # set up target
    target = os.path.abspath(os.path.join(bindir, exe_name))

    # get current directory
    cpth = os.getcwd()

    # create temporary path
    dstpth = os.path.join('tempbin')
    print('create...{}'.format(dstpth))
    if not os.path.exists(dstpth):
        os.makedirs(dstpth)
    os.chdir(dstpth)

    # Download the distribution
    pymake.download_and_unzip(url, verify=verify)

    # Set srcdir name
    srcdir = os.path.join(dirname, srcname)

    if replace_function is not None:
        replace_function(srcdir)

    # compile code
    print('compiling...{}'.format(os.path.relpath(target)))
    pymake.main(srcdir, target, fct, cct, makeclean=True,
                expedite=False, dryrun=False, double=dble, debug=False,
                include_subdirs=include_subdirs)

    # change back to original path
    os.chdir(cpth)

    msg = '{} does not exist.'.format(os.path.relpath(target))
    assert os.path.isfile(target), msg

    # Clean up downloaded directory
    print('delete...{}'.format(dstpth))
    if os.path.isdir(dstpth):
        shutil.rmtree(dstpth)

    return


if __name__ == '__main__':
    # test_build_mf6()
    # test_build_modflow()
    # test_build_mfnwt()
    # test_build_usg()
    # test_build_mt3dms()
    # test_build_seawat()
    # test_build_gridgen()
    # test_build_triangle()
    test_build_modpath7()
