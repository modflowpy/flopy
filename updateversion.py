import subprocess
import os
import sys
import datetime

def update_version():
    
    from flopy import __version__
    v = __version__.strip().split('.')[-1]
    version_type = ('3', '0', '{0}'.format(int(v)+1)) 
    version = '.'.join(version_type)

    b = subprocess.Popen(("git", "describe", "--match", "build"), stdout = subprocess.PIPE).communicate()[0]
    build = b.strip().split('-')[1]
    
    print 'Updating version:'
    print '  ', __version__, '->', version
    f = open(os.path.normpath('flopy/version.py'), 'w')
    f.write('#flopy version file automatically created using...{0}\n'.format(os.path.basename(__file__)))
    f.write('#            created on......{0}\n'.format(datetime.date.today().strftime("%B %d, %Y %H:%M:%S")))
    f.write("__version__='{0}'\n".format(version))
    f.write("__build__='{0}.{1}'\n".format(version, build))
    f.close()

    cmdtag = 'git commit ./flopy/version.py -m "version number update"'
    os.system(cmdtag)
    cmdtag = 'git push'
    os.system(cmdtag)
    cmdtag = 'git tag -a {0} -m "Version {0}"'.format(version)
    os.system(cmdtag)
    cmdtag = 'git push --tags'
    os.system(cmdtag)
 
 
if __name__ == "__main__":
    update_version()