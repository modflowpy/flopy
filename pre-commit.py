#!/usr/bin/python

from __future__ import print_function
import subprocess
import os
import sys
import datetime

files = ['version.py', 'README.md']

pak = 'flopy'


def get_version_str(v0, v1, v2, v3):
    version_type = ('{}'.format(v0),
                    '{}'.format(v1),
                    '{}'.format(v2),
                    '{}'.format(v3))
    version = '.'.join(version_type)
    return version


def get_tag(v0, v1, v2):
    tag_type = ('{}'.format(v0),
                '{}'.format(v1),
                '{}'.format(v2))
    tag = '.'.join(tag_type)
    return tag


def update_version():
    try:
        pth = os.path.join(pak, files[0])

        vmajor = 0
        vminor = 0
        vmicro = 0
        vbuild = 0
        lines = [line.rstrip('\n') for line in open(pth, 'r')]
        for line in lines:
            t = line.split()
            if 'major =' in line:
                vmajor = int(t[2])
            elif 'minor =' in line:
                vminor = int(t[2])
            elif 'micro =' in line:
                vmicro = int(t[2])
            elif 'build =' in line:
                vbuild = int(t[2])

        v0 = get_version_str(vmajor, vminor, vmicro, vbuild)

        # get latest build number
        tag = get_tag(vmajor, vminor, vmicro)
        print('determining version build from {}'.format(tag))
        try:
            b = subprocess.Popen(("git", "describe", "--match", tag),
                                 stdout=subprocess.PIPE).communicate()[0]
            vbuild = int(b.decode().strip().split('-')[1]) + 1
        # assume if tag does not exist that it has not been added
        except:
            vbuild = 0

        v1 = get_version_str(vmajor, vminor, vmicro, vbuild)

        # get current build number
        b = subprocess.Popen(("git", "describe", "--match", "build"),
                             stdout=subprocess.PIPE).communicate()[0]
        vcommit = int(b.decode().strip().split('-')[1]) + 2

        print('Updating version:')
        print('  ', v0, '->', v1)

        # write new version file
        f = open(pth, 'w')
        f.write('# {} version file automatically '.format(pak) +
                'created using...{0}\n'.format(os.path.basename(__file__)))
        f.write('# created on...' +
                '{0}\n'.format(
                    datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")))
        f.write('\n')
        f.write('major = {}\n'.format(vmajor))
        f.write('minor = {}\n'.format(vminor))
        f.write('micro = {}\n'.format(vmicro))
        f.write('build = {}\n'.format(vbuild))
        f.write('commit = {}\n\n'.format(vcommit))
        f.write("__version__ = '{:d}.{:d}.{:d}'.format(major, minor, micro)\n")
        f.write(
            "__build__ = '{:d}.{:d}.{:d}.{:d}'.format(major, minor, micro, build)\n")
        f.write("__git_commit__ = '{:d}'.format(commit)\n")
        f.close()
        print('Succesfully updated version.py')
    except:
        print('There was a problem updating the version file')
        sys.exit(1)

    # update README.md with new version information
    update_readme_markdown(vmajor, vminor, vmicro, vbuild)

    # update docs/USGS_release.md with new version information
    update_USGSmarkdown(vmajor, vminor, vmicro, vbuild)


def add_updated_files():
    try:
        # add modified version file
        print('Adding updated files to repo')
        b = subprocess.Popen(("git", "add", "-u"),
                             stdout=subprocess.PIPE).communicate()[0]
    except:
        print('Could not add updated files')
        sys.exit(1)


def update_readme_markdown(vmajor, vminor, vmicro, vbuild):
    try:
        # determine current buildstat branch
        b = subprocess.Popen(("git", "status"),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT).communicate()[0]
        if isinstance(b, bytes):
            b = b.decode('utf-8')

        # determine current buildstat branch
        for line in b.splitlines():
            if 'On branch' in line:
                branch = line.replace('On branch ', '').rstrip()
    except:
        print('Cannot update README.md - could not determine current branch')
        return

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read README.md into memory
    with open(files[1], 'r') as file:
        lines = [line.rstrip() for line in file]

    # rewrite README.md
    f = open(files[1], 'w')
    for line in lines:
        if '### Version ' in line:
            line = '### Version {}'.format(version)
            if vbuild > 0:
                line += ' {} &mdash; build {}'.format(branch, vbuild)
        elif '[Build Status]' in line:
            line = '[![Build Status](https://travis-ci.org/modflowpy/' + \
                   'flopy.svg?branch={})]'.format(branch) + \
                   '(https://travis-ci.org/modflowpy/flopy)'
        elif '[Coverage Status]' in line:
            line = '[![Coverage Status](https://coveralls.io/repos/github/' + \
                   'modflowpy/flopy/badge.svg?branch={})]'.format(branch) + \
                   '(https://coveralls.io/github/modflowpy/' + \
                   'flopy?branch={})'.format(branch)
        elif 'http://dx.doi.org/10.5066/F7BK19FH' in line:
            now = datetime.datetime.now()
            sb = ''
            if vbuild > 0:
                sb = ' &mdash; {}'.format(branch)
            line = '[Bakker, M., Post, V., Langevin, C.D., Hughes, J.D., ' + \
                   'White, J.T., Starn, J.J., and Fienen, M.N., ' + \
                   '{}, '.format(now.year) + \
                   'FloPy v{}{}: '.format(version, sb) + \
                   'U.S. Geological Survey Software Release, ' + \
                   '{}, '.format(now.strftime('%d %B %Y')) + \
                   'http://dx.doi.org/10.5066/F7BK19FH]' + \
                   '(http://dx.doi.org/10.5066/F7BK19FH)\n'
        f.write('{}\n'.format(line))
    f.close()

    return


def update_USGSmarkdown(vmajor, vminor, vmicro, vbuild):
    try:
        # determine current buildstat branch
        b = subprocess.Popen(("git", "status"),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT).communicate()[0]
        if isinstance(b, bytes):
            b = b.decode('utf-8')

        # determine current buildstat branch
        for line in b.splitlines():
            if 'On branch' in line:
                branch = line.replace('On branch ', '').rstrip()
    except:
        print('Cannot update README.md - could not determine current branch')
        return

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read README.md into memory
    with open(files[1], 'r') as file:
        lines = [line.rstrip() for line in file]

    # write USGS_release.md
    fpth = os.path.join('docs', 'USGS_release.md')
    f = open(fpth, 'w')

    # write USGS_release.md
    fpth = os.path.join('docs', 'PyPi_release.md')
    f2 = open(fpth, 'w')

    # date and branch information
    now = datetime.datetime.now()
    sdate = now.strftime("%m/%d/%Y")
    sb = ''
    if vbuild > 0:
        sb = ' &mdash; {}'.format(branch)

    # write header information
    f.write('---\n')
    f.write('title: FloPy Release Notes\n')
    f.write('author:\n')
    f.write('    - Mark Bakker\n')
    f.write('    - Vincent Post\n')
    f.write('    - Christian D. Langevin\n')
    f.write('    - Joseph D. Hughes\n')
    f.write('    - Jeremy T. White\n')
    f.write('    - Andrew T. Leaf\n')
    f.write('    - Scott R. Paulinski\n')
    f.write('    - Jeffrey Starn\n')
    f.write('    - Michael N. Fienen\n')
    f.write('header-includes:\n')
    f.write('    - \\usepackage{fancyhdr}\n')
    f.write('    - \\usepackage{lastpage}\n')
    f.write('    - \\pagestyle{fancy}\n')
    f.write('    - \\fancyhf{{}}\n')
    f.write('    - \\fancyhead[LE, LO, RE, RO]{}\n')
    f.write('    - \\fancyhead[CE, CO]{FloPy Release Notes}\n')
    f.write('    - \\fancyfoot[LE, RO]{{FloPy version {}{}}}\n'.format(version, sb))
    f.write('    - \\fancyfoot[CO, CE]{\\thepage\\ of \\pageref{LastPage}}\n')
    f.write('    - \\fancyfoot[RE, LO]{{{}}}\n'.format(sdate))
    f.write('geometry: margin=0.75in\n')
    f.write('---\n\n')

    # write select information from README.md
    writeline = False
    for line in lines:
        if line == 'Introduction':
            writeline = True
        elif line == 'Examples':
            writeline = False
        elif line == 'Installation':
            writeline = False
        elif '***Development version of FloPy:***' in line:
            writeline = False
        elif 'Click [here](docs/mf6.md) for more information.' in line:
            line = line.replace('Click [here](docs/mf6.md) for more information.', '')
        if writeline:
            f.write('{}\n'.format(line))
            line = line.replace('***', '*')
            line = line.replace('##### ', '')
            f2.write('{}\n'.format(line))

    # write installation information
    cweb = 'https://water.usgs.gov/ogw/flopy/flopy-{}.zip'.format(version)
    line = ''
    line += 'Installation\n'
    line += '-----------------------------------------------\n'
    line += 'To install FloPy version {}{} '.format(version, sb)
    line += 'from the USGS FloPy website:\n'
    line += '```\n'
    line += 'pip install {}\n'.format(cweb)
    line += '```\n\n'
    line += 'To update to FloPy version {}{} '.format(version, sb)
    line += 'from the USGS FloPy website:\n'
    line += '```\n'
    line += 'pip install {} --upgrade\n'.format(cweb)
    line += '```\n'
    f.write(line)

    # close the USGS_release.md file
    f.close()

    line = line.replace(cweb, 'flopy')
    line = line.replace(' from the USGS FloPy website', '')

    f2.write(line)


    # close the PyPi_release.md file
    f2.close()

    return


if __name__ == "__main__":
    update_version()
    add_updated_files()
