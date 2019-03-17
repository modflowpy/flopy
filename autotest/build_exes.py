# Build the executables that are used in the flopy autotests
import os
import sys
import json
import shutil

try:
    import pymake
except:
    print('pymake is not installed...will not build executables')
    pymake = None

download_version = '1.0'

# path where downloaded executables will be extracted
exe_pth = 'exe_download'
# make the directory if it does not exist
if not os.path.isdir(exe_pth):
    os.makedirs(exe_pth)

# determine if running on Travis
is_travis = 'TRAVIS' in os.environ

bindir = '.'
dotlocal = False
if is_travis:
    dotlocal = True

if not dotlocal:
    for idx, arg in enumerate(sys.argv):
        if '--travis' in arg.lower():
            dotlocal = True
            break
if dotlocal:
    bindir = os.path.join(os.path.expanduser('~'), '.local', 'bin')
    bindir = os.path.abspath(bindir)

# write where the executables will be downloaded
print('modflow executables will be downloaded to:\n    "{}"'.format(bindir))


def get_targets():
    targets = pymake.usgs_program_data().get_keys(current=True)
    targets.sort()
    targets.remove('vs2dt')
    return targets


def build_target(target):
    if pymake is not None:
        pymake.build_apps(targets=target)

    return


def which(program):
    """
    Test to make sure that the program is executable

    """
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def get_modflow_exes(pth='.', version='', platform=None):
    """
    Get the latest MODFLOW binary executables from a github site
    (https://github.com/MODFLOW-USGS/executables) for the specified
    operating system and put them in the specified path.

    Parameters
    ----------
    pth : str
        Location to put the executables (default is current working directory)

    version : str
        Version of the MODFLOW-USGS/executables release to use.

    platform : str
        Platform that will run the executables.  Valid values include mac,
        linux, win32 and win64.  If platform is None, then routine will
        download the latest asset from the github reposity.

    """

    # Determine the platform in order to construct the zip file name
    if platform is None:
        if sys.platform.lower() == 'darwin':
            platform = 'mac'
        elif sys.platform.lower().startswith('linux'):
            platform = 'linux'
        elif 'win' in sys.platform.lower():
            is_64bits = sys.maxsize > 2 ** 32
            if is_64bits:
                platform = 'win64'
            else:
                platform = 'win32'
        else:
            errmsg = ('Could not determine platform'
                      '.  sys.platform is {}'.format(sys.platform))
            raise Exception(errmsg)
    else:
        assert platform in ['mac', 'linux', 'win32', 'win64']
    zipname = '{}.zip'.format(platform)

    # Determine path for file download and then download and unzip
    url = ('https://github.com/MODFLOW-USGS/executables/'
           'releases/download/{}/'.format(version))
    assets = {p: url + p for p in ['mac.zip', 'linux.zip',
                                   'win32.zip', 'win64.zip']}
    download_url = assets[zipname]
    pymake.download_and_unzip(download_url, pth)

    return


def is_executable(f):
    # write message
    msg = 'testing if {} is executable.'.format(f)
    print(msg)

    # test if file is executable
    fname = os.path.join(exe_pth, f)
    errmsg = '{} not executable'.format(fname)
    assert which(fname) is not None, errmsg
    return


def get_code_json():
    jpth = 'code.json'
    json_dict = None
    if jpth in os.listdir(exe_pth):
        fpth = os.path.join(exe_pth, jpth)
        json_dict = pymake.usgs_program_data.load_json(fpth)

    return json_dict


def evaluate_versions(target, src):
    # get code.json dictionary
    json_dict = get_code_json()

    # get current modflow program dictionary
    prog_dict = pymake.usgs_program_data().get_program_dict()

    if json_dict is not None:
        # extract the json keys
        json_keys = list(json_dict.keys())
        # evaluate if the target is in the json keys
        if target in json_keys:
            source_version = prog_dict[target].version
            git_version = json_dict[target].version

            # write a message
            msg = 'Source code version of {} '.format(target) + \
                  'is "{}"'.format(source_version)
            print(4 * ' ' + msg)
            msg = 'Download code version of {} '.format(target) + \
                  'is "{}"\n'.format(git_version)
            print(4 * ' ' + msg)

            prog_version = source_version.split('.')
            json_version = git_version.split('.')

            # evaluate major, minor, etc. version numbers
            for sp, sj in zip(prog_version, json_version):
                if int(sp) > int(sj):
                    src = None
                    break

    return src


def copy_target(src):
    srcpth = os.path.join(exe_pth, src)
    dstpth = os.path.join(bindir, src)

    # write message showing copy src and dst
    msg = 'copying {} -> {}'.format(srcpth, dstpth)
    print(msg)

    # copy the target
    shutil.copy(srcpth, dstpth)

    return


def cleanup():
    if os.path.isdir(exe_pth):
        shutil.rmtree(exe_pth)

    return


def main():
    get_modflow_exes(exe_pth, download_version)

    etargets = os.listdir(exe_pth)
    for f in etargets:
        is_executable(f)

    # build each target
    targets = get_targets()
    for target in targets:
        src = None
        for etarget in etargets:
            if target in etarget:
                src = etarget
                break

        # evaluate if the usgs source files and newer versions than
        # downloaded executables...if so build the target from source code
        if src is not None:
            src = evaluate_versions(target, src)

        # copy the downloaded executable
        if src is not None:
            pass
            # copy_target(src)
        # build the target from source code
        else:
            msg = 'building {}'.format(target)
            print(msg)
            # build_target(target)

        # build all targets (until github gfortran-8 exes are available)
        build_target(target)

    # clean up the download directory
    cleanup()


def test_download_and_unzip():
    get_modflow_exes(exe_pth, download_version)
    for f in os.listdir(exe_pth):
        yield is_executable, f
    return


def test_build_all_apps():
    # get list of downloaded targets
    etargets = os.listdir(exe_pth)

    # build each target
    targets = get_targets()
    for target in targets:
        src = None
        for etarget in etargets:
            if target in etarget:
                src = etarget
                break

        # evaluate if the usgs source files and newer versions than
        # downloaded executables...if so build the target from source code
        if src is not None:
            src = evaluate_versions(target, src)

        # # copy the downloaded executable
        # if src is not None:
        #     yield copy_target, src
        # # build the target
        # else:
        #     msg = 'building {}'.format(target)
        #     print(msg)
        #     yield build_target, target

        # build all targets (until github gfortran-8 exes are available)
        yield build_target, target

    return


def test_cleanup():
    cleanup()


if __name__ == '__main__':
    main()
