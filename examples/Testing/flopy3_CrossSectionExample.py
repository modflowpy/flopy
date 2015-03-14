import sys
import os
import platform
import numpy as np
import matplotlib.pyplot as plt

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
    print 'Something bad happened.'
files = ['freyberg.hds', 'freyberg.cbc']
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = 'Output file located: {}'.format(f)
        print (msg)
    else:
        errmsg = 'Error. Output file cannot be found: {}'.format(f)
        print (errmsg)



# First step is to set up the plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(ml=ml, line={'Row': 20})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 20')
plt.show()

# First step is to set up the plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the ModelMap class
modelxsect = flopy.plot.ModelCrossSection(ml=ml, line={'Column': 10})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelxsect.plot_grid()
t = ax.set_title('Column 10')
plt.show()

# # The ModelMap instance can be created using several different keyword arguments to position the model grid in space.  The three keywords are: xul, yul, and rotation.  The values represent the x-coordinate of the upper left corner, the y-coordinate of the upper-left coordinate, and the rotation angle (in degrees) of the upper left coordinate.  If these values are not specified, then they default to model coordinates.
# #
# # Here we demonstrate the effects of these values.  In the first two plots, the grid origin (upper left corner) remains fixed at (0, 10000).  The y-value of 10000 is the sum of the model delc array (a sum of all the row heights).
#
# # In[36]:
#
# fig = plt.figure(figsize=(15, 5))
#
# ax = fig.add_subplot(1, 3, 1, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml)
# patchcollection = modelxsect.plot_grid()
# t = ax.set_title('???')
# plt.show()


# ax = fig.add_subplot(1, 3, 2, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=-20)
# linecollection = modelxsect.plot_grid()
# t = ax.set_title('rotation=-20 degrees')
#
# ax = fig.add_subplot(1, 3, 3, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, xul=500000, yul=2934000, rotation=45)
# linecollection = modelxsect.plot_grid()
# t = ax.set_title('xul, yul, and rotation')
#
#
# # ###Ploting Ibound
# # The plot_ibound() method can be used to plot the boundary conditions contained in the ibound arrray, which is part of the MODFLOW Basic Package.  The plot_ibound() method returns a matplotlib QuadMesh object (matplotlib.collections.QuadMesh).  If you are familiar with the matplotlib collections, then this may be important to you, but if not, then don't worry about the return objects of these plotting function.
#
# # In[45]:
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound()
# linecollection = modelxsect.plot_grid()
#
#
# # In[46]:
#
# # Or we could change the colors!
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound(color_noflow='red', color_ch='orange')
# linecollection = modelxsect.plot_grid(colors='yellow')
#
#
# # ###Plotting Boundary Conditions
# # The plot_bc() method can be used to plot boundary conditions.  It is setup to use the following dictionary to assign colors, however, these colors can be changed in the method call.
# #
# #     bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
# #                  'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}
# #
# # Here, we plot the location of river cells and the location of well cells.
#
# # In[47]:
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound()
# quadmesh = modelxsect.plot_bc('RIV')
# quadmesh = modelxsect.plot_bc('WEL')
# linecollection = modelxsect.plot_grid()
#
#
# # In[48]:
#
# # Or we could change the colors!
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound(color_noflow='red', color_ch='orange')
# quadmesh = modelxsect.plot_bc('RIV', color='purple')
# quadmesh = modelxsect.plot_bc('WEL', color='navy')
# linecollection = modelxsect.plot_grid(colors='yellow')
#
#
# # ###Plotting an Array
# # ModelMap has a plot_array() method.  The plot_array() method will accept either a 2D or 3D array.  If a 3D array is passed, then the layer for the ModelMap object will be used (note that the ModelMap object can be created with a 'layer=' argument).
#
# # In[62]:
#
# # Create a random array and plot it
# a = np.random.random((ml.dis.nrow, ml.dis.ncol))
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_title('Random Array')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_array(a)
# linecollection = modelxsect.plot_grid()
# cb = plt.colorbar(quadmesh, shrink=0.5)
#
#
# # In[61]:
#
# # Plot the model bottom array
# a = ml.dis.botm.array
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_title('Model Bottom Elevations')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_array(a)
# linecollection = modelxsect.plot_grid()
# cb = plt.colorbar(quadmesh, shrink=0.5)
#
#
# # ###Contouring an Array
# # ModelMap also has a contour_array() method.  It also takes a 2D or 3D array and will contour the layer slice if 3D.
#
# # In[64]:
#
# # Contour the model bottom array
# a = ml.dis.botm.array
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_title('Model Bottom Elevations')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# contour_set = modelxsect.contour_array(a)
# linecollection = modelxsect.plot_grid()
#
#
# # In[68]:
#
# # The contour_array() method will take any keywords
# # that can be used by the matplotlib.pyplot.contour
# # function. So we can be in levels, for example.
# a = ml.dis.botm.array
# levels = np.arange(0, 20, 0.5)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_title('Model Bottom Elevations')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# contour_set = modelxsect.contour_array(a, levels=levels)
# linecollection = modelxsect.plot_grid()
#
#
# # ###Plotting Heads
# # So this means that we can easily plot results from the simulation by extracting heads using flopy.utils.  Here we plot the simulated heads.
#
# # In[85]:
#
# fname = os.path.join(modelpth, 'freyberg.hds')
# hdobj = flopy.utils.HeadFile(fname)
# head = hdobj.get_data()
# levels = np.arange(10, 30, .5)
#
# fig = plt.figure(figsize=(10, 10))
#
# ax = fig.add_subplot(1, 2, 1, aspect='equal')
# ax.set_title('plot_array()')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound()
# quadmesh = modelxsect.plot_array(head, masked_values=[999.], alpha=0.5)
# linecollection = modelxsect.plot_grid()
#
# ax = fig.add_subplot(1, 2, 2, aspect='equal')
# ax.set_title('contour_array()')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound()
# contour_set = modelxsect.contour_array(head, masked_values=[999.], levels=levels)
# linecollection = modelxsect.plot_grid()
#
#
# # ###Plotting Discharge Vectors
# # ModelMap has a plot_discharge() method, which takes the 'FLOW RIGHT FACE' and 'FLOW FRONT FACE' arrays, which can be written by MODFLOW to the cell by cell flow file.  These array can be extracted from the cell by cell flow file using the flopy.utils.CellBudgetFile object as shown below.  Once they are extracted, they can be passed to the plot_discharge() method.  Note that plot_discharge() also takes the head array as an argument.  The head array is used by plot_discharge() to convert the volumetric discharge in dimensions of $L^3/T$ to specific discharge in dimensions of $L/T$.
#
# # In[86]:
#
# fname = os.path.join(modelpth, 'freyberg.cbc')
# cbb = flopy.utils.CellBudgetFile(fname)
# frf = cbb.get_data(text='FLOW RIGHT FACE')[0]
# fff = cbb.get_data(text='FLOW FRONT FACE')[0]
#
# fig = plt.figure(figsize=(10, 10))
#
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_title('plot_array()')
# modelxsect = flopy.plot.ModelMap(ml=ml, rotation=14)
# quadmesh = modelxsect.plot_ibound()
# quadmesh = modelxsect.plot_array(head, masked_values=[999.], alpha=0.5)
# quiver = modelxsect.plot_discharge(frf, fff, head=head)
# linecollection = modelxsect.plot_grid()
#
#
# # ##Summary
# #
# # This notebook demonstrates some of the plotting functionality available with flopy.  Although not described here, the plotting functionality tries to be general by passing keyword arguments passed to the ModelMap method down into the matplot.pyplot routines that do the actual plotting.  For those looking to customize these plots, it may be necessary to search for the available keywords by understanding the types of objects that are created by the ModelMap methods.  The ModelMap methods return these matplotlib.collections objects so that they could be fine-tuned later in the script before plotting.
# #
# # Hope this gets you started!
#
# # In[ ]:



