Examples
-----------------------------------------------

### jupyter Notebook Examples

The following jupyter Notebooks contain examples for using FloPy pre- and post-processing capabilities with a variety of MODFLOW-based models.

#### MODFLOW-2000, MODFLOW-2005, MODFLOW-NWT, MODFLOW-USG, MODPATH, MT3DMS, MT3D-USGS, and SEAWAT

##### ***Basic examples***

+ An overview of loading existing MODFLOW models, creating models, and common post-processing capabilities using FloPy is presented in the [flopy3_working_stack_demo](../examples/Notebooks/flopy3_working_stack_demo.ipynb) Notebook.

+ An overview of the options to enter *layer, row, column, data* values for packages such as the wel and ghb packages is given in the [flopy3_modflow_boundaries](../examples/Notebooks/flopy3_modflow_boundaries.ipynb) Notebook

+ An overview of how to control the format of numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_array_outputformat_options](../examples/Notebooks/flopy3_array_outputformat_options.ipynb) Notebook.

+ An overview of how FloPy handles external files for numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_external_file_handling](../examples/Notebooks/flopy3_external_file_handling.ipynb) Notebook.

+ An overview of FloPy functionality in the ```SpatialReference``` class for locating the model in a "real world" coordinate reference system is given in the [flopy3_SpatialReference_demo](../examples/Notebooks/flopy3_SpatialReference_demo.ipynb) Notebook.

+ An overview of FloPy capabilities to load a SFR2 file and evaluate data contained in the file is given in the [flopy3_SFR2_load](../examples/Notebooks/flopy3_SFR2_load.ipynb) Notebook.

+ An overview of FloPy capabilities to create a SFR2 file and evaluate data contained in the file is given in the [flopy3_sfrpackage_example](../examples/Notebooks/flopy3_sfrpackage_example.ipynb) Notebook. This example also shows how to read SFR water balance output into a pandas dataframe for additional postprocessing.

+ An overview of FloPy capabilities to create a MNW2 file and evaluate data contained in the file is given in the [flopy3_mnw2package_example](../examples/Notebooks/flopy3_mnw2package_example.ipynb) Notebook.

+ An overview of FloPy capabilities to create a UZF file and evaluate data contained in the file and UZF output files is given in the [flopy3_uzf_example](../examples/Notebooks/flopy3_uzf_example) Notebook.

+ An overview of FloPy capabilities for exporting two-dimensional array data as a binary file is given in the [flopy3_save_binary_data_file](../examples/Notebooks/flopy3_save_binary_data_file.ipynb) Notebook.

+ An overview of FloPy capabilities to create MODPATH models and plot MODPATH results is given in the [flopy3_Modpath_example](../examples/Notebooks/flopy3_Modpath_example.ipynb) Notebook.

+ An overview of using FloPy and GRIDGEN to creating layered quadtree grids for MODFLOW-USG is given in the [flopy3_gridgen](../examples/Notebooks/flopy3_gridgen.ipynb) Notebook.

+ The [lake example](../examples/Notebooks/flopy3_lake_example.ipynb), a very simple FloPy example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 

+ A variant of the [water-table example](../examples/Notebooks/flopy3_WatertableRecharge_example.ipynb), a very simple example of one-dimensional groundwater flow in an unconfined aquifer with recharge, from the MODFLOW-NWT documentation (http://pubs.usgs.gov/tm/tm6a37/). This IPython Notebook build files for MODFLOW-NWT.

+ The [Zaidel discontinuous water-table example](../examples/Notebooks/flopy3_Zaidel_example.ipynb), which simulates a discontinuous water table over a stairway impervious base, from http://onlinelibrary.wiley.com/doi/10.1111/gwat.12019/abstract. This IPython Notebook build files for MODFLOW-USG. (http://pubs.usgs.gov/tm/06/a45/). 

+ An overview of the options for creating a Source/Sink Mixing (SSM) Package for MT3DMS and SEAWAT is given in the [flopy3ssm](../examples/Notebooks/flopy3_multi-component_SSM.ipynb) Notebook.

+ The ['Crank-Nicolson' example distributed with MT3D-USGS](../examples/Notebooks/flopy3_MT3D-USGS_example.ipynb), a simple MT3D-USGS model that uses the SFT Package.

+ The [Henry Problem](../examples/Notebooks/flopy3_SEAWAT_henry_problem.ipynb), a simple saltwater intrusion model developed with FloPy and run using SEAWAT.

##### ***SWI2 examples***

+ [Example 1](../examples/Notebooks/flopy3_swi2package_ex1.ipynb) of the SWI2 manual, simulating a rotating interface.

+ [Example 4](../examples/Notebooks/flopy3_swi2package_ex4.ipynb) of the SWI2 manual, upconing below a pumping well below a two-aquifer island system.

##### ***Model analysis and error checking examples***

+ An overview of the FloPy [model input data `check()` method capabilities](../examples/Notebooks/flopy3_ModelCheckerExample.ipynb).

+ An overview of the FloPy [zone budget `ZoneBudget()` method capabilities](../examples/Notebooks/flopy3_ZoneBudget_example) Notebook. The `ZoneBudget()` method is a python implementation of USGS ZONEBUDGET executable for MODFLOW (Harbaugh, 1990).

+ An overview of the FloPy [`get_transmissivities()` method for computing open interval transmissivities (for weighted averages of heads or fluxes)](../examples/Notebooks/flopy3_get_transmissivities_example.ipynb) Notebook. This method can be used to:
	* compute vertically-averaged head target values representative of observation wells of varying open intervals (including variability in saturated thickness due to the position of the water table). This may be especially important for reducing error in observations used for parameter estimation, in areas with appreciable vertical head gradients (due to aquitards, pumping, discharge to surface water, etc.)
	* apportion boundary fluxes (e.g. from an analytic element model) among model layers based on transmissivity.
	* any other analysis where a distribution of transmissivity is needed for a specified vertical interval of the model.

+ An overview of utilties for [post-processing head results from MODFLOW](../examples/Notebooks/flopy3_Modflow_postprocessing_example.ipynb).

#### ***Export examples***

+ An overview of the FloPy [netCDF and shapefile export capabilities](../examples/Notebooks/flopy3_export.ipynb).

+ [Exporting 2-D arrays as rasters or contour shapefiles](../examples/Notebooks/flopy3_Modflow_postprocessing_example.ipynb).

#### ***Parameter Estimation examples***

+ An overview of the FloPy [parameter estimation capabilities](../examples/Notebooks/flopy3_PEST.ipynb).

#### MODFLOW 6

##### ***Basic examples***

+ A simple MODFLOW 6 example is given in the [flopy3_mf6_A_simple-model](../examples/Notebooks/flopy3_mf6_A_simple-model.ipynb) Notebook.

+ A more complicated MODFLOW 6 example is given in the [flopy3_mf6_B_complex-model](../examples/Notebooks/flopy3_mf6_B_complex-model.ipynb) Notebook.

#### Plotting examples ***(limited functionality for MODFLOW 6 models)***

+ An overview of the FloPy [map and cross-section plotting capabilities](../examples/Notebooks/flopy3_MapExample.ipynb).

+ An overview of the FloPy  [model input and output data `plot()` method capabilities](../examples/Notebooks/flopy3_PlotArrayExample.ipynb)

+ An overview of SWR1 Process Output Processing and Plotting is given in the [flopy3_LoadSWRBinaryData](../examples/Notebooks/flopy3_LoadSWRBinaryData.ipynb) Notebook.

+ The [flopy3_shapefile_features](../examples/Notebooks/flopy3_shapefile_features.ipynb) Notebook illustrates some functionality in flopy for exchanging MODFLOW-related information with shapefiles, including convenience functions for working with shapefile data in numpy recarrays, some simple container objects for storing geographic information, and a demonstration of automatic writing of projection (.prj) files using EPSG codes.

+ An overview of [plotting MODFLOW-USG unstructured grid data and model results](../examples/Notebooks/flopy3_UnstructuredGridPlotting.ipynb).

+ An overview of [how to plot MODFLOW 6 results for a single GWF model](../examples/Notebooks/flopy3_BasicMODFLOW6Plot.ipynb).

#### Additional MODFLOW examples

+ Example problems from the 2015 2nd edition of [Applied Groundwater Modeling](https://github.com/Applied-Groundwater-Modeling-2nd-Ed) by Mary P. Anderson, William W. Woessner, and Randall J. Hunt (https://github.com/Applied-Groundwater-Modeling-2nd-Ed)

