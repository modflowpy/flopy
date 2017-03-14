Examples
-----------------------------------------------

### IPython Notebook Examples

The following IPython Notebooks contain example FloPy scripts for a variety of models and FloPy features

#### Basic examples

+ An overview of the options to enter *layer, row, column, data* values for packages such as the wel and ghb packages is given in the [flopy3_modflow_boundaries](../examples/Notebooks/flopy3_modflow_boundaries.ipynb) Notebook

+ An overview of how to control the format of numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_array_outputformat_options](../examples/Notebooks/flopy3_array_outputformat_options.ipynb) Notebook.

+ An overview of how FloPy3 handles external files for numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_external_file_handling](../examples/Notebooks/flopy3_external_file_handling.ipynb) Notebook.

+ An overview of FloPy3 capabilities to load a SFR2 file and evaluate data contained in the file is given in the [flopy3_SFR2_load](../examples/Notebooks/flopy3_SFR2_load.ipynb) Notebook.

+ An overview of FloPy3 capabilities to create a SFR2 file and evaluate data contained in the file is given in the [flopy3_sfrpackage_example](../examples/Notebooks/flopy3_sfrpackage_example.ipynb) Notebook.

+ An overview of FloPy3 capabilities to create a MNW2 file and evaluate data contained in the file is given in the [flopy3_mnw2package_example](../examples/Notebooks/flopy3_mnw2package_example.ipynb) Notebook.

+ An overview of FloPy3 capabilities to create MODPATH models and plot MODPATH results is given in the [flopy3_Modpath_example](../examples/Notebooks/flopy3_Modpath_example.ipynb) Notebook.

+ The [lake example](../examples/Notebooks/flopy3_lake_example.ipynb), a very simple FloPy example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 

+ A variant of the [water-table example](../examples/Notebooks/flopy3_WatertableRecharge_example.ipynb), a very simple example of one-dimensional groundwater flow in an unconfined aquifer with recharge, from the MODFLOW-NWT documentation (http://pubs.usgs.gov/tm/tm6a37/). This IPython Notebook build files for MODFLOW-NWT.

+ The [Zaidel discontinuous water-table example](../examples/Notebooks/flopy3_Zaidel_example.ipynb), which simulates a discontinuous water table over a stairway impervious base, from http://onlinelibrary.wiley.com/doi/10.1111/gwat.12019/abstract. This IPython Notebook build files for MODFLOW-USG. (http://pubs.usgs.gov/tm/06/a45/). 

+ An overview of the options for creating a Source/Sink Mixing (SSM) Package for MT3DMS and SEAWAT is given in the [flopy3ssm](../examples/Notebooks/flopy3_multi-component_SSM.ipynb) Notebook.

+ The [Henry Problem](../examples/Notebooks/flopy3_SEAWAT_henry_problem.ipynb), a simple saltwater intrusion model developed with Flopy and run using SEAWAT.

#### SWI2 examples

+ [Example 1](../examples/Notebooks/flopy3_swi2package_ex1.ipynb) of the SWI2 manual, simulating a rotating interface.

+ [Example 4](../examples/Notebooks/flopy3_swi2package_ex4.ipynb) of the SWI2 manual, upconing below a pumping well below a two-aquifer island system.

#### Model analysis and error checking examples

+ An overview of the FloPy [model input data `check()` method capabilities](../examples/Notebooks/flopy3_ModelCheckerExample.ipynb)

+ An overview of the FloPy [zone budget `ZoneBudget()` method capabilities](../examples/Notebooks/flopy3_ZoneBudget_example) Notebook. The `ZoneBudget()` method is a python implementation of USGS ZONEBUDGET executable for MODFLOW (Harbaugh, 1990).

+ An overview of the Flopy [`get_transmissivities()` method for computing open interval transmissivities (for weighted averages of heads or fluxes)](../examples/Notebooks/flopy3_get_transmissivities_example.ipynb) Notebook.

#### Plotting examples

+ An overview of the FloPy [map and cross-section plotting capabilities](../examples/Notebooks/flopy3_MapExample.ipynb).

+ An overview of the FloPy  [model input and output data `plot()` method capabilities](../examples/Notebooks/flopy3_PlotArrayExample.ipynb)

+ An overview of SWR1 Process Output Processing and Plotting is given in the [flopy3_LoadSWRBinaryData](../examples/Notebooks/flopy3_LoadSWRBinaryData.ipynb) Notebook.

#### Export examples

+ An overview of the FloPy [netCDF and shapefile export capabilities](../examples/Notebooks/flopy3_export.ipynb).

#### Parameter Estimation examples

+ An overview of the FloPy [parameter estimation capabilities](../examples/Notebooks/flopy3_PEST.ipynb).

#### Additional MODFLOW examples

+ Example problems from the 2015 2nd edition of [Applied Groundwater Modeling](https://github.com/Applied-Groundwater-Modeling-2nd-Ed) by Mary P. Anderson, William W. Woessner, and Randall J. Hunt (https://github.com/Applied-Groundwater-Modeling-2nd-Ed)

