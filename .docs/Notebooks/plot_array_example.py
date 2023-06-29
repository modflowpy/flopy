# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   metadata:
#     section: viz
#     authors:
#       - Joseph Hughes
# ---

# # Plotting Model Arrays and Results
#
# This notebook demonstrates the simple array and results plotting capabilities of flopy.  It demonstrates these capabilities by loading and running an existing model, and then showing how the `.plot()` method can be used to make simple plots of the model data and model results.

import os
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# +
from IPython.display import Image

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

print(sys.version)
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(mpl.__version__))
print("flopy version: {}".format(flopy.__version__))

# +
# Set name of MODFLOW exe
#  assumes executable is in users path statement
version = "mf2005"
exe_name = "mf2005"

# Set the paths
loadpth = os.path.join("..", "..", "examples", "data", "secp")

# temporary directory
temp_dir = TemporaryDirectory()
modelpth = temp_dir.name

# make sure modelpth directory exists
if not os.path.isdir(modelpth):
    os.makedirs(modelpth, exist_ok=True)

files = ["secp.hds"]
# -

# ### Load and Run an Existing Model
# A model called the "Southeast Coastal Plain Model" is located in the loadpth folder.  In the following code block, we load that model, then change into a new workspace (modelpth) where we recreate and run the model.  For this to work properly, the MODFLOW-2005 executable (mf2005) must be in the path.  We verify that it worked correctly by checking for the presence of secp.hds.

# +
ml = flopy.modflow.Modflow.load(
    "secp.nam", model_ws=loadpth, exe_name=exe_name, version=version
)
ml.change_model_ws(new_pth=modelpth)
ml.write_input()

success, buff = ml.run_model(silent=True)
if not success:
    print("Something bad happened.")

# confirm that the model files have been created
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = "Output file located: {}".format(f)
        print(msg)
    else:
        errmsg = "Error. Output file cannot be found: {}".format(f)
        print(errmsg)
# -

# ### Plotting Model Data
# Once a model object is created MODFLOW package data can be plotted using the `.plot()` method.
#
# Two-dimensional data (for example the model top) can be plotted by calling the `.plot()` method for each data array.

ml.modelgrid.plot()

# As you can see, the `.plot()` methods returns a `matplotlib.pyplot` axis object, which can be used to add additional data to the figure. Below we will add black contours to the axis returned in the first line.

ax = ml.dis.top.plot()
ml.dis.top.plot(axes=ax, contour=True, pcolor=False)

# You will notice that we passed several keywords in the second line. There are a number of keywords that can be passed to the `.plot()` method to control plotting. Available keywords are:
#
# 1. `axes` - if you already have plot axes you can pass them to the method
# 2. `pcolor` - turns pcolor on if `pcolor=True` or off if `pcolor=False`, default is `pcolor=True`
# 3. `colorbar` - turns on colorbar if `colorbar=True` or off if `colorbar=False`, default is `colorbar=False` and is only used if `pcolor=True`
# 4. `inactive` - turns on a black inactive cell overlay if `inactive=True` or turns off the inactive cell overlay if `inactive=False`, default is `inactive=True`
# 5. `contour` - turns on contours if `contour=True` or off if `contour=False`, default is `contour=False`
# 6. `clabel` - turns on contour labels if `clabel=True` or off if `clabel=False`, default is `clabel=False` and is only used if `contour=True`
# 7. `grid` - turns on model grid if `grid=True` or off if `grid=False`, default is `grid=False`
# 8. `masked_values` - list with unique values to be excluded from the plot (for example, HNOFLO)
# 9. `mflay` - for three-dimensional data (for example layer bottoms or simulated heads) `mflay` can be used to plot data for a single layer - note `mflay` is zero-based
# 10. `kper` - for transient two-dimensional data (for example recharge package data) `kper` can be used to plot data for a single stress period - note `kper` is zero-based
# 11. `filename_base` - a base file name that will be used to automatically generate file names for two-dimensional, three-dimensional, and transient two-dimensional data, default is `filename_base=None`
# 12. `file_extension` - valid `matplotlib` file extension, default is `png` and is only used if `filename_base` is specified
# 13. `matplotlib.pyplot` keywords are also accepted
#
# The previous code block is recreated in a single line using keywords in the code block below.

ml.dis.top.plot(contour=True)

# We can save the same image to a file.

fname = os.path.join(modelpth, "secp")
ml.dis.top.plot(contour=True, filename_base=fname)

# The image file that was just created is shown below

fname = os.path.join(modelpth, "secp_model_top.png")
Image(filename=fname)

# ### Plotting three-dimensional data
#
# Three-dimensional data can be plotted using the `.plot()` method. User's do not actually need to know that the data are two- or three-dimensional. The `.plot()` method is attached to the two- and three-dimensional data objects so it knows how to process the model data. Examples of three-dimensional data are horizontal hydraulic conductivity (`hk`), layer bottoms (`botm`), specific yield (`sy`), *etc.*
#
# Here we plot the horizontal hydraulic conductivity for each layer. We are also masking areas where the horizontal hydraulic conductivity is zero and adding a color bar.

ml.lpf.hk.plot(masked_values=[0.0], colorbar=True)

# In addition to the plots of horizontal hydraulic conductivity you can see that the `.plot()` method returned a list containing 16 axis objects (one for each layer).
#
# #### Plotting data for a single layer
#
# If the `mflay=` keyword is provided to the `plot.()` method then data for an individual layer is plotted. Remember `mflay` is zero-based.
#
# Here we plot the horizontal hydraulic conductivity for layer 11 (`mflay=10`).

ml.lpf.hk.plot(mflay=10, masked_values=[0.0], colorbar=True)

# ### Plotting transient two-dimensional data
#
# Transient two-dimensional data can be plotted using the `.plot()` method. User's do not actually need to know that the data are two- or three-dimensional. The `.plot()` method is attached to the two- and three-dimensional, and transient two-dimensional data objects so it knows how to process the model data. Examples of transient two-dimensional data are recharge rates (`rch.rech`) and evapotranspiration rates (`evt.evtr`).
#
# Here we plot recharge rates for all six stress periods in the model. We are also masking areas where the recharge rate is zero and adding a color bar.

ml.rch.rech.plot(kper="all", masked_values=[0.0], colorbar=True)

# In addition to the plots of recharge rates you can see that the `.plot()` method returned a list containing 6 axis objects (one for each stress period).
#
# #### Plotting data for a single stress period
#
# If the `kper=` keyword is provided to the `plot.()` method then data for an individual stress period is plotted. Remember `kper` is zero-based.
#
# Here we plot the recharge rate for stress period 6 (`kper=5`).

ml.rch.rech.plot(kper=5, masked_values=[0.0], colorbar=True)

# We can also save the image to a file by provided the `filename_base` keyword with an appropriate base file name.

fr = os.path.join(modelpth, "secp")
ml.rch.rech.plot(kper=5, masked_values=[0.0], colorbar=True, filename_base=fr)

# If the `kper` keyword is not provided images are saved for each stress period in the model.
#
# The image file that was just created of recharge rates for stress period 6 is is shown below.

fname = os.path.join(modelpth, "secp_RECH_00006.png")
Image(filename=fname)

# ### Plotting simulated model results
#
# Simulated model results can be plotted using the `.plot()` method.
#
# First we create an instance of the `HeadFile` class with the simulated head file (secp.hds) and extract the simulation times available in the binary head file using the `.get_times()` method. Here we plot last simulated heads in the binary heads file (`totim=times[-1]`). We are also masking cells having the `HDRY` (-9999.0)  value and adding a color bar.

print(files)
fname = os.path.join(modelpth, files[0])
hdobj = flopy.utils.HeadFile(fname, model=ml)
times = hdobj.get_times()
head = hdobj.plot(totim=times[-1], masked_values=[-9999.0], colorbar=True)

# #### Plotting results for a single layer
#
# If the `mflay=` keyword is provided to the `plot.()` method then results for an individual layer can be plotted. Remember `mflay` is zero-based.
#
# Here we plot the results for layer 2 (`mflay=1`) for stress period 2 (`totim=times[1]`). We also add black contour lines.

head = hdobj.plot(
    totim=times[1],
    mflay=1,
    masked_values=[-9999.0],
    colorbar=True,
    contour=True,
    colors="black",
)

# We can also save the plots of the head results for every layer (or a single layer) to a file by provided the `filename_base` keyword with an appropriate base file name.

fh = os.path.join(modelpth, "secp_head")
head = hdobj.plot(
    totim=times[-1],
    masked_values=[-9999.0],
    colorbar=True,
    contour=True,
    colors="black",
    filename_base=fh,
)

# The image file that was just created of the simulated head for layer 2 for stress period 6 is is shown below.

fname = os.path.join(modelpth, "secp_head_Layer2.png")
Image(filename=fname)

# ### Passing other `matplotlib.pyplot` keywords to `.plot()` methods
#
# We can also pass `matplotlib.pyplot` keywords to `.plot()` methods attached to the model input data arrays. For example you can pass a `matplotlib` colormap (`cmap=`) keyword to the `.plot()` method to plot contours of simulated heads over a color flood of hk. We can also use the `norm=LogNorm()` keyword to use a log color scale when plotting hydraulic conductivity.
#
# Available `matplotlib` colormaps can be found at https://matplotlib.org/stable/tutorials/colors/colormaps.html

# +
from matplotlib.colors import LogNorm

ax = ml.lpf.hk.plot(mflay=2, cmap="GnBu", norm=LogNorm(), colorbar=True)
t = hdobj.plot(
    axes=ax,
    mflay=2,
    masked_values=[-9999.0],
    pcolor=False,
    contour=True,
    colors="black",
)
# -

# ### Plotting data for a package or a model
#
# The input data for a model or an individual package can also be plotted using the `.plot()` method. The `.plot()` methods attached to a model or an individual package are meant to provide a method to quickly evaluate model or package input. As a result, there is limited ability to customize the plots. Example of using the `.plot()` method with a model or and individual packages is demonstrated below.
#
# ### Plot all data for a package
#
# All input data for a package can be plotted using the `.plot()` method. Below all of the data for the lpf package is plotted.

ml.dis.plot()

# ### Plot package input data for a specified layer
#
# Package input data for a specified layer can be plotted by passing the `mflay` keyword to the package `.plot()` method. Below lpf package input data for layer 12 (`mflay=11`) is plotted.

ml.lpf.plot(mflay=11)

# ### Plot all input data for a model
#
# All of the input data for a model can be plotted using the `.plot()` method.

ap = ml.plot()

# ### Plot model input data for a specified layer
#
# Model input data for a specified layer can be plotted by passing the `mflay` keyword to the package `.plot()` method.
# Below model input data for layer 12 (`mflay=11`) is plotted.

ap = ml.plot(mflay=11)

# ## Summary
#
# This notebook demonstrates some of the simple plotting functionality available with flopy.  Although not described here, the plotting functionality tries to be general by passing keyword arguments passed to the `plot()` and `plot_data()` methods down into the `matplotlib.pyplot` routines that do the actual plotting.  For those looking to customize these plots, it may be necessary to search for the available keywords.  The `plot()` method return the `matplotlib.pyplot` axis objects that are created (or passed). These axes objects can be used to plot additional data (except when plots are saved as image files).
#
# Hope this gets you started!

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
