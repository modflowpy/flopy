import subprocess
import os
import sys
import datetime

def update_version():
    unknown = 'unknown'
    
    try:
        v = subprocess.Popen(("svnversion", "-q"), stdout = subprocess.PIPE).communicate()[0]
        err = None
    except Exception, e:
        v = unknown
        err = e
    if ':' in v:
        v = v.strip().split(':')[-1]
    if 'M' in v:
        v = int(v.replace('M', '')) + 1
    elif 'S' in v:
        v = v.replace('S', '')
    version_type = ('2', '2', '{0}'.format(v)) 
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
 
 
if __name__ == "__main__":
    update_version()