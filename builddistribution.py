import os
import sys
import platform
import subprocess

from updateversion import update_version

#--determine if the version needs to be updated
update_version()

#--install the source
subprocess.call(['python', 'setup.py', 'install'])

#--create the source distribution
subprocess.call(['python', 'setup.py', 'sdist', '--format=zip'])

#--create 32-bit and 64-bit windows installers
if 'windows' in platform.system().lower():
    #--32-bit
    subprocess.call(['python', 'setup.py', 'build', '--plat-name=win', 'bdist_wininst'])    
    #--64-bit
    subprocess.call(['python', 'setup.py', 'build', '--plat-name=win-amd64', 'bdist_wininst'])

#--now register the package with PyPI
subprocess.call(['python', 'setup.py', 'register'])
   
