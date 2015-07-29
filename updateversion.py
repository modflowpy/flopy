from __future__ import print_function
import subprocess
import os
import sys
import datetime

def update_version(utag=False, major=False, minor=False):
    
    from flopy import __version__
    vmajor = __version__.strip().split('.')[-3]
    vminor = __version__.strip().split('.')[-2]
    vfeature = __version__.strip().split('.')[-1]
    
    if major:
        vmajor = int(vmajor) + 1
        vminor = 1
        vfeature = 1
    elif minor:
        vmajor = int(vmajor)
        vminor = int(vminor) + 1
        vfeature = 1
    else:
        vmajor = int(vmajor)
        vminor = int(vminor)
        vfeature = int(vfeature) + 1
        
    version_type = ('{}'.format(vmajor), '{}'.format(vminor), '{}'.format(vfeature)) 
    version = '.'.join(version_type)

    b = subprocess.Popen(("git", "describe", "--match", "build"), stdout = subprocess.PIPE).communicate()[0]
    build = b.strip().split('-')[1]
    
    print('Updating version:')
    print('  ', __version__, '->', version)
    
    #--write new version file
    f = open(os.path.normpath('flopy/version.py'), 'w')
    f.write('#flopy version file automatically created using...{0}\n'.format(os.path.basename(__file__)))
    f.write('#            created on......{0}\n'.format(datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")))
    f.write("__version__='{0}'\n".format(version))
    f.write("__build__='{0}.{1}'\n".format(version, build))
    f.close()

    # update version tag
    if utag:
        cmdtag = 'git commit ./flopy/version.py -m "version number update"'
        os.system(cmdtag)
        cmdtag = 'git push'
        os.system(cmdtag)
        cmdtag = 'git tag -a {0} -m "Version {0}"'.format(version)
        os.system(cmdtag)
        cmdtag = 'git push --tags'
        os.system(cmdtag)
 
 
if __name__ == "__main__":
    utag = False
    major = False
    minor = False
    for arg in sys.argv:
        if arg.lower() == '--tag' or arg.lower() == '-t':
            utag = True
            sys.stdout.write('  Will update git tag with version number.\n')    
        elif arg.lower() == '--major' or arg.lower() == '-ma':
            major = True
            sys.stdout.write('  Will update major version number by one.\n')    
        elif arg.lower() == '--minor' or arg.lower() == '-mi':
            minor = True
            sys.stdout.write('  Will update minor version number by one.\n')    
    
    update_version(utag, major, minor)