import os
import sys
import platform
import subprocess

from updateversion import update_version

def build_distribution(upd=False, utag=False, install=False, 
                       reg=False, winbuild=False, 
                       major=False, minor=False):
    # determine if the version needs to be updated
    if upd:
        update_version(utag, major, minor)
    
    # install the source in site_packages
    if install:
        subprocess.call(['python', 'setup.py', 'install'])
    
    # create the source distribution
    subprocess.call(['python', 'setup.py', 'sdist', '--format=zip'])
    
    # create 32-bit and 64-bit windows installers
    if winbuild:
        if 'windows' in platform.system().lower():
            # 32-bit
            subprocess.call(['python', 'setup.py', 'build', '--plat-name=win', 'bdist_wininst'])    
            # 64-bit
            subprocess.call(['python', 'setup.py', 'build', '--plat-name=win-amd64', 'bdist_wininst'])
    
    # now register the package with PyPI
    if reg:
        subprocess.call(['python', 'setup.py', 'register'])

if __name__ == "__main__":    
    uver = False
    install = False
    utag = False
    register = False
    winbuild = False
    major = False
    minor = False
    for arg in sys.argv:
        if arg.lower() == '--update' or arg.lower() == '-u':
            uver = True
            sys.stdout.write('  Will update version number.\n')    
        elif arg.lower() == '--install' or arg.lower() == '-i':
            install = True
            sys.stdout.write('  Will install flopy in python site_packages directory.\n')    
        elif arg.lower() == '--register' or arg.lower() == '-r':
            register = True
            sys.stdout.write('  Will register version with PyPI.\n')    
        elif arg.lower() == '--tag' or arg.lower() == '-t':
            utag = True
            sys.stdout.write('  Will update git tag with version number.\n')    
        elif arg.lower() == '--win' or arg.lower() == '-w':
            winbuild = True
            sys.stdout.write('  Will build windows installers if possible.\n')    
        elif arg.lower() == '--major' or arg.lower() == '-ma':
            major = True
            sys.stdout.write('  Will update major version number by one.\n')    
        elif arg.lower() == '--minor' or arg.lower() == '-mi':
            minor = True
            sys.stdout.write('  Will update minor version number by one.\n')    

    build_distribution(uver, utag, install, 
                       register, winbuild, major, minor)
   
