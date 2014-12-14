import subprocess
import os
import sys
import datetime

def update_version():
    unknown = 'unknown'
    
    try:
        v = subprocess.Popen(("git", "describe", "--match", "build"), stdout = subprocess.PIPE).communicate()[0]
        #v = os.system("git describe")
        print v
        err = None
    except Exception, e:
        v = unknown
        err = e
    if v is not unknown:
        v = v.strip().split('-')[1]
        print v
    version_type = ('3', '0', '{0}'.format(int(v)+1)) 
    version = '.'.join(version_type)
    try:
        from flopy import __version__
    except:
        __version__ = unknown
    
    if __version__ != version:
        print 'Need to update version:'
        print '  ', __version__, '->', version
        f = open(os.path.normpath('flopy/version.py'), 'w')
        f.write('#flopy version file automatically created using...{0}\n'.format(os.path.basename(__file__)))
        f.write('#            created on......{0}\n'.format(datetime.date.today().strftime("%B %d, %Y")))
        f.write("__version__='{0}'".format(version))
        f.close()
    else:
        print 'version file with __version__={0} is current.'.format(__version__)
        
    cmdtag = "git tag -a {0} -m 'Version {0}'".format(version)
    os.system(cmdtag)
    cmdtag = "git commit -m 'version number update' ./flopy/version.py"
    os.system(cmdtag)
    cmdtag = "git push --tags"
    os.system(cmdtag)
 
 
if __name__ == "__main__":
    update_version()