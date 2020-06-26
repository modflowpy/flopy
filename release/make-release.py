#!/usr/bin/python

import subprocess
import os
import sys
import datetime
import json
from collections import OrderedDict

# update files and paths so that there are the same number of
# path and file entries in the paths and files list. Enter '.'
# as the path if the file is in the root repository directory
paths = ['../flopy', '../',
         '../docs', '../docs',
         '../', '../', '../docs']
files = ['version.py', 'README.md',
         'USGS_release.md', 'PyPi_release.md',
         'code.json', 'DISCLAIMER.md', 'notebook_examples.md']

# check that there are the same number of entries in files and paths
if len(paths) != len(files):
    msg = 'The number of entries in paths ' + \
          '({}) must equal '.format(len(paths)) + \
          'the number of entries in files ({})'.format(len(files))
    assert False, msg

pak = 'flopy'

# local import of package variables in flopy/version.py
# imports author_dict
exec(open(os.path.join("..", "flopy", "version.py")).read())

# build authors list for Software/Code citation for FloPy
authors = []
for key in author_dict.keys():
    t = key.split()
    author = '{}'.format(t[-1])
    for str in t[0:-1]:
        author += ' {}'.format(str)
    authors.append(author)

approved = '''Disclaimer
----------

This software has been approved for release by the U.S. Geological Survey
(USGS). Although the software has been subjected to rigorous review, the USGS
reserves the right to update the software as needed pursuant to further analysis
and review. No warranty, expressed or implied, is made by the USGS or the U.S.
Government as to the functionality of the software and related material nor
shall the fact of release constitute any such warranty. Furthermore, the
software is released on condition that neither the USGS nor the U.S. Government
shall be held liable for any damages resulting from its authorized or
unauthorized use.
'''

preliminary = '''Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.
'''


def get_disclaimer():
    # get current branch
    branch = get_branch()

    if branch.lower().startswith('release') or 'master' in branch.lower():
        disclaimer = approved
        is_approved = True
    else:
        disclaimer = preliminary
        is_approved = False

    return is_approved, disclaimer


def get_branch():
    branch = None

    # determine if branch defined on command line
    for argv in sys.argv:
        if 'master' in argv:
            branch = 'master'
        elif 'develop' in argv.lower():
            branch = 'develop'

    if branch is None:
        try:
            # determine current branch
            b = subprocess.Popen(("git", "status"),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT).communicate()[0]
            if isinstance(b, bytes):
                b = b.decode('utf-8')

            for line in b.splitlines():
                if 'On branch' in line:
                    branch = line.replace('On branch ', '').rstrip()

        except:
            msg = 'Could not determine current branch. Is git installed?'
            raise ValueError(msg)

    return branch


def get_version_str(v0, v1, v2):
    version_type = ('{}'.format(v0),
                    '{}'.format(v1),
                    '{}'.format(v2))
    version = '.'.join(version_type)
    return version


def get_tag(v0, v1, v2):
    tag_type = ('{}'.format(v0),
                '{}'.format(v1),
                '{}'.format(v2))
    tag = '.'.join(tag_type)
    return tag


def get_software_citation(version, is_approved):
    now = datetime.datetime.now()
    sb = ''
    if not is_approved:
        sb = ' &mdash; release candidate'
    # format author names
    line = '['
    for ipos, author in enumerate(authors):
        if ipos > 0:
            line += ', '
        if ipos == len(authors) - 1:
            line += 'and '
        sv = author.split()
        tauthor = '{}'.format(sv[0])
        if len(sv) < 3:
            gname = sv[1]
            if len(gname) > 1:
                tauthor += ', {}'.format(gname)
            else:
                tauthor += ', {}.'.format(gname[0])
        else:
            tauthor += ', {}. {}.'.format(sv[1][0], sv[2][0])
        # add formatted author name to line
        line += tauthor

    # add the rest of the citation
    line += ', {}, '.format(now.year) + \
            'FloPy v{}{}: '.format(version, sb) + \
            'U.S. Geological Survey Software Release, ' + \
            '{}, '.format(now.strftime('%d %B %Y')) + \
            'http://dx.doi.org/10.5066/F7BK19FH]' + \
            '(http://dx.doi.org/10.5066/F7BK19FH)'

    return line


def update_version():
    name_pos = None
    try:
        fpth = os.path.join(paths[0], files[0])

        vmajor = 0
        vminor = 0
        vmicro = 0
        lines = [line.rstrip('\n') for line in open(fpth, 'r')]
        for idx, line in enumerate(lines):
            t = line.split()
            if 'major =' in line:
                vmajor = int(t[2])
            elif 'minor =' in line:
                vminor = int(t[2])
            elif 'micro =' in line:
                vmicro = int(t[2])
            elif '__version__' in line:
                name_pos = idx + 1

    except:
        msg = 'There was a problem updating the version file'
        raise IOError(msg)

    try:
        # write new version file
        f = open(fpth, 'w')
        f.write('# {} version file automatically '.format(pak) +
                'created using...{0}\n'.format(os.path.basename(__file__)))
        f.write('# created on...' +
                '{0}\n'.format(
                    datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")))
        f.write('\n')
        f.write('major = {}\n'.format(vmajor))
        f.write('minor = {}\n'.format(vminor))
        f.write('micro = {}\n'.format(vmicro))
        f.write("__version__ = '{:d}.{:d}.{:d}'.format(major, minor, micro)\n")

        # write the remainder of the version file
        if name_pos is not None:
            for line in lines[name_pos:]:
                f.write('{}\n'.format(line))
        f.close()
        print('Successfully updated version.py')
    except:
        msg = 'There was a problem updating the version file'
        raise IOError(msg)

    # update README.md with new version information
    update_readme_markdown(vmajor, vminor, vmicro)

    # update notebook_examples.md
    update_notebook_examples_markdown()

    # update code.json
    update_codejson(vmajor, vminor, vmicro)

    # update docs/USGS_release.md with new version information
    update_USGSmarkdown(vmajor, vminor, vmicro)


def update_codejson(vmajor, vminor, vmicro):
    # define json filename
    json_fname = os.path.join(paths[4], files[4])

    # get branch
    branch = get_branch()

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # load and modify json file
    with open(json_fname, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    # modify the json file data
    now = datetime.datetime.now()
    sdate = now.strftime('%Y-%m-%d')
    data[0]['date']['metadataLastUpdated'] = sdate
    if 'release' in branch.lower() or 'master' in branch.lower():
        data[0]['version'] = version
        data[0]['status'] = 'Production'
    else:
        data[0]['version'] = version
        data[0]['status'] = 'Release Candidate'

    # rewrite the json file
    with open(json_fname, 'w') as f:
        json.dump(data, f, indent=4)
        f.write('\n')

    return


def update_readme_markdown(vmajor, vminor, vmicro):
    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # define branch
    if is_approved:
        branch = 'master'
    else:
        branch = 'develop'

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read README.md into memory
    fpth = os.path.join(paths[1], files[1])
    with open(fpth, 'r') as file:
        lines = [line.rstrip() for line in file]

    # rewrite README.md
    terminate = False
    f = open(fpth, 'w')
    for line in lines:
        if '### Version ' in line:
            line = '### Version {}'.format(version)
            if not is_approved:
                line += ' &mdash; release candidate'
        elif '[Build Status]' in line:
            line = '[![Build Status](https://travis-ci.org/modflowpy/' + \
                   'flopy.svg?branch={})]'.format(branch) + \
                   '(https://travis-ci.org/modflowpy/flopy)'
        elif '[Coverage Status]' in line:
            line = '[![Coverage Status](https://coveralls.io/repos/github/' + \
                   'modflowpy/flopy/badge.svg?branch={})]'.format(branch) + \
                   '(https://coveralls.io/github/modflowpy/' + \
                   'flopy?branch={})'.format(branch)
        elif '[Binder]' in line:
            # [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)
            line = '[![Binder](https://mybinder.org/badge_logo.svg)]' + \
                   '(https://mybinder.org/v2/gh/modflowpy/flopy.git/' + \
                   '{}'.format(branch) + ')'
        elif 'http://dx.doi.org/10.5066/F7BK19FH' in line:
            line = get_software_citation(version, is_approved)
        elif 'Disclaimer' in line:
            line = disclaimer
            terminate = True
        f.write('{}\n'.format(line))
        if terminate:
            break

    f.close()

    # write disclaimer markdown file
    fpth = os.path.join(paths[0], 'DISCLAIMER.md')
    f = open(fpth, 'w')
    f.write(disclaimer)
    f.close()

    return


def update_notebook_examples_markdown():
    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # define branch
    if is_approved:
        branch = 'master'
    else:
        branch = 'develop'

    # read notebook_examples.md into memory
    fpth = os.path.join(paths[6], files[6])
    with open(fpth, 'r') as file:
        lines = [line.rstrip() for line in file]

    # rewrite notebook_examples.md
    terminate = False
    f = open(fpth, 'w')
    for line in lines:
        if '[Binder]' in line:
            # [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)
            line = '[![Binder](https://mybinder.org/badge_logo.svg)]' + \
                   '(https://mybinder.org/v2/gh/modflowpy/flopy.git/' + \
                   '{}'.format(branch) + ')'
        f.write('{}\n'.format(line))
    f.close()


def update_USGSmarkdown(vmajor, vminor, vmicro):
    # get branch
    branch = get_branch()

    # create disclaimer text
    is_approved, disclaimer = get_disclaimer()

    # create version
    version = get_tag(vmajor, vminor, vmicro)

    # read README.md into memory
    fpth = os.path.join(paths[1], files[1])
    with open(fpth, 'r') as file:
        lines = [line.rstrip() for line in file]

    # write USGS_release.md
    fpth = os.path.join(paths[2], files[2])
    f = open(fpth, 'w')

    # write PyPi_release.md
    fpth = os.path.join(paths[3], files[3])
    f2 = open(fpth, 'w')

    # date and branch information
    now = datetime.datetime.now()
    sdate = now.strftime("%m/%d/%Y")

    # write header information
    f.write('---\n')
    f.write('title: FloPy Release Notes\n')
    f.write('author:\n')
    for author in authors:
        sv = author.split()
        tauthor = '{}'.format(sv[1])
        if len(sv) > 2:
            tauthor += ' {}.'.format(sv[2][0])
        tauthor += ' {}'.format(sv[0])
        f.write('    - {}\n'.format(tauthor))
    f.write('header-includes:\n')
    f.write('    - \\usepackage{fancyhdr}\n')
    f.write('    - \\usepackage{lastpage}\n')
    f.write('    - \\pagestyle{fancy}\n')
    f.write('    - \\fancyhf{{}}\n')
    f.write('    - \\fancyhead[LE, LO, RE, RO]{}\n')
    f.write('    - \\fancyhead[CE, CO]{FloPy Release Notes}\n')
    f.write('    - \\fancyfoot[LE, RO]{{FloPy version {}}}\n'.format(version))
    f.write('    - \\fancyfoot[CO, CE]{\\thepage\\ of \\pageref{LastPage}}\n')
    f.write('    - \\fancyfoot[RE, LO]{{{}}}\n'.format(sdate))
    f.write('geometry: margin=0.75in\n')
    f.write('---\n\n')

    # write select information from README.md
    writeline = False
    for line in lines:
        if line == 'Introduction':
            writeline = True
        elif line == 'Installation':
            writeline = False
        elif line == 'Documentation':
            writeline = True
        elif line == 'Getting Started':
            writeline = False
        elif line == 'Contributing':
            writeline = True
        elif line == 'How to Cite':
            writeline = True
        elif 'http://dx.doi.org/10.5066/F7BK19FH' in line:
            writeline = True
            line = get_software_citation(version, is_approved)
        elif line == 'MODFLOW Resources':
            writeline = False
        elif 'Installing the latest FloPy release candidate' in line:
            writeline = False
        elif line == 'Disclaimer':
            writeline = True
        elif '[MODFLOW 6](docs/mf6.md)' in line:
            line = line.replace('[MODFLOW 6](docs/mf6.md)', 'MODFLOW 6')
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
    line += 'To install FloPy version {} '.format(version)
    line += 'from the USGS FloPy website:\n'
    line += '```\n'
    line += 'pip install {}\n'.format(cweb)
    line += '```\n\n'
    line += 'To update to FloPy version {} '.format(version)
    line += 'from the USGS FloPy website:\n'
    line += '```\n'
    line += 'pip install {} --upgrade\n'.format(cweb)
    line += '```\n'

    #
    f.write(line)

    # close the USGS_release.md file
    f.close()

    # close the PyPi_release.md file
    f2.close()

    return


if __name__ == "__main__":
    update_version()
    get_software_citation('3.1.1', True)