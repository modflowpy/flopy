import sys
import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import flopy

#Set name of MODFLOW exe
#  assumes executable is in users path statement
version = 'mf2005'
exe_name = 'mf2005'
if platform.system() == 'Windows':
    exe_name = 'mf2005.exe'
mfexe = exe_name

#Set the paths
loadpth = os.path.join('..', 'Notebooks', 'data')
modelpth = os.path.join('data')
modelname = 'swiex4_s1'

#make sure modelpth directory exists
if not os.path.exists(modelpth):
    os.makedirs(modelpth)

ml = flopy.modflow.Modflow.load(modelname, model_ws=loadpth, exe_name=exe_name, version=version)
ml.change_model_ws(new_pth=modelpth)
ml.write_input()
#success, buff = ml.run_model()
success = True

if not success:
    print 'Something bad happened.'
files = [modelname+'.hds', modelname+'.zta']
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = 'Output file located: {}'.format(f)
        print (msg)
    else:
        errmsg = 'Error. Output file cannot be found: {}'.format(f)
        print (errmsg)


fname = os.path.join(modelpth, modelname+'.hds')
hdobj = flopy.utils.HeadFile(fname)
kstpkper = hdobj.get_kstpkper()
head = [] 
for kk in kstpkper:
    head.append(hdobj.get_data(kstpkper=kk))

fname = os.path.join(modelpth, modelname+'.zta')
zobj = flopy.utils.CellBudgetFile(fname)
kstpkper = zobj.get_kstpkper()
zeta = []
for kk in kstpkper:
    zeta.append(zobj.get_data(kstpkper=kk, text='ZETASRF  1')[0])


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(ml=ml, line={'Row': 20})
fb = modelxsect.plot_fill_between(head[4], colors=['brown', 'cyan'])
#patches = modelxsect.csplot_ibound(head=head)
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 20')
plt.show()

print '...end'
