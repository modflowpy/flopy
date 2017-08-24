# Remove the temp directory and then create a fresh one
import os
import sys
import shutil
import platform
import flopy
import pymake

fc = 'gfortran'
cc = 'gcc'
double = False
bindir = os.path.join(os.path.expanduser('~'), '.local', 'bin')
bindir = os.path.abspath(bindir)
print(bindir)


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


def test_setup():
    tempdir = os.path.join('.', 'temp')
    if os.path.isdir(tempdir):
        shutil.rmtree(tempdir)
    os.mkdir(tempdir)
    return


def test_build_modflow():
    starget = 'MODFLOW-2005'
    exe_name = 'mf2005'
    dirname = 'MF2005.1_12u'
    url = "https://water.usgs.gov/ogw/modflow/MODFLOW-2005_v1.12.00/MF2005.1_12u.zip"

    build_target(starget, exe_name, url, dirname)

    return


def test_build_mfnwt():
    starget = 'MODFLOW-NWT'
    exe_name = 'mfnwt'
    dirname = 'MODFLOW-NWT_1.1.3'
    url = "http://water.usgs.gov/ogw/modflow-nwt/{0}.zip".format(dirname)

    build_target(starget, exe_name, url, dirname)

    return


def test_build_usg():
    starget = 'MODFLOW-USG'
    exe_name = 'mfusg'
    dirname = 'mfusg.1_3'
    url = 'https://water.usgs.gov/ogw/mfusg/{0}.zip'.format(dirname)

    build_target(starget, exe_name, url, dirname)
    return


def test_build_lgr():
    starget = 'MODFLOW-LGR'
    exe_name = 'mflgr'
    dirname = 'mflgr.2_0'
    url = "https://water.usgs.gov/ogw/modflow-lgr/modflow-lgr-v2.0.0/mflgrv2_0_00.zip"

    build_target(starget, exe_name, url, dirname)
    return


def test_build_mt3dusgs():
    starget = 'MT3D-USGS'
    exe_name = 'mt3dusgs'
    dirname = 'mt3d-usgs_Distribution'
    url = "https://water.usgs.gov/ogw/mt3d-usgs/mt3d-usgs_1.0.zip"

    build_target(starget, exe_name, url, dirname)
    return


def test_build_mt3dms():
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
    starget = 'SEAWAT'
    exe_name = 'swt_v4'
    dirname = 'swt_v4_00_05'
    url = "https://water.usgs.gov/ogw/seawat/{0}.zip".format(dirname)

    build_target(starget, exe_name, url, dirname,
                 srcname='source',
                 replace_function=update_seawatfiles)
    return


def set_compiler():
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
                 replace_function=None, verify=True, replace=False):
    print('Determining if {} needs to be built'.format(starget))
    if platform.system().lower() == 'windows':
        exe_name += '.exe'

    is_travis = 'TRAVIS' in os.environ

    exe_exists = flopy.which(exe_name)
    if exe_exists is not None and not is_travis and not replace:
        print('No need to build {} since it exists in the current path')
        return

    fct, cct = set_compiler()

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
                expedite=False, dryrun=False, double=double, debug=False)

    msg = '{} does not exist.'.format(os.path.relpath(target))
    assert os.path.isfile(target), msg

    # change back to original path
    os.chdir(cpth)

    # Clean up downloaded directory
    print('delete...{}'.format(dstpth))
    if os.path.isdir(dstpth):
        shutil.rmtree(dstpth)

    return


if __name__ == '__main__':
    test_setup()
    # test_build_modflow()
    # test_build_mfnwt()
    # test_build_usg()
    # test_build_mt3dms()
    test_build_seawat()
