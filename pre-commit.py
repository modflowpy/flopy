#!/usr/bin/python

from __future__ import print_function
import subprocess
import os
import sys
import datetime

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
        pth = os.path.join(pak, 'version.py')
        
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
                '{0}\n'.format(datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")))
        f.write('\n')
        f.write('major = {}\n'.format(vmajor))
        f.write('minor = {}\n'.format(vminor))
        f.write('micro = {}\n'.format(vmicro))
        f.write('build = {}\n'.format(vbuild))
        f.write('commit = {}\n\n'.format(vcommit))
        f.write("__version__ = '{:d}.{:d}.{:d}'.format(major, minor, micro)\n")
        f.write("__build__ = '{:d}.{:d}.{:d}.{:d}'.format(major, minor, micro, build)\n")
        f.write("__git_commit__ = '{:d}'.format(commit)\n")
        f.close()
        print('Succesfully updated version.py')
    except:
        print('There was a problem updating the version file')
        sys.exit(1)

def add_updated_version():
    try:
        # add modified version file
        print('Adding updated version file to repo')
        b = subprocess.Popen(("git", "add", "{}/version.py".format(pak)),
                             stdout=subprocess.PIPE).communicate()[0]
    except:
        print('Could not add updated version file')
        sys.exit(1) 

if __name__ == "__main__":
    update_version()
    add_updated_version()
    
