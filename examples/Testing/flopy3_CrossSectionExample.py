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
loadpth = os.path.join('..', 'data', 'freyberg')
modelpth = os.path.join('data')

#make sure modelpth directory exists
if not os.path.exists(modelpth):
    os.makedirs(modelpth)



ml = flopy.modflow.Modflow.load('freyberg.nam', model_ws=loadpth, exe_name=exe_name, version=version)
ml.change_model_ws(new_pth=modelpth)
ml.write_input()
success, buff = ml.run_model()
if not success:
    print('Something bad happened.')
files = ['freyberg.hds', 'freyberg.cbc']
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = 'Output file located: {}'.format(f)
        print (msg)
    else:
        errmsg = 'Error. Output file cannot be found: {}'.format(f)
        print (errmsg)


fname = os.path.join(modelpth, 'freyberg.hds')
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Row': 20})
fb = modelxsect.plot_fill_between(head, colors=['brown', 'cyan'], masked_values=[999.00])
#patches = modelxsect.csplot_ibound(head=head)
patches = modelxsect.plot_bc('RIV', head=head)
patches = modelxsect.plot_bc('WEL', color='navy', head=head)
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 20')
plt.show()


# First step is to set up the plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(ml=ml, line={'Row': 20})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
cmap = plt.get_cmap('jet')
cr = np.linspace(10., 25., num=cmap.N)
norm = matplotlib.colors.BoundaryNorm(cr, cmap.N)
hv = modelxsect.plot_array(head, head=head, masked_values=[999.00, -1.00000E+30]) #, norm=norm)
patches = modelxsect.plot_ibound(head=head)
patches = modelxsect.plot_bc('RIV', head=head)
patches = modelxsect.plot_bc('WEL', color='navy', head=head)
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 20')
fig.colorbar(hv, orientation='horizontal', format='%3.1f')
plt.show()



# First step is to set up the plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(ml=ml, line={'Column': 10})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
hv = modelxsect.plot_array(head, masked_values=[999.00, -1.00000E+30], norm=norm)
ct = modelxsect.contour_array(head, masked_values=[999.00, -1.00000E+30],
                                colors='black', linewidths=0.5,
                                levels=[10, 15, 20, 25, 30])
plt.clabel(ct, fontsize=8, fmt='%3.1f')
patches = modelxsect.plot_ibound()
linecollection = modelxsect.plot_grid()
t = ax.set_title('Column 10')
fig.colorbar(hv, orientation='horizontal', format='%3.1f')
plt.show()

