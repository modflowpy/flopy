
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.2.3
[![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=master)](https://travis-ci.org/modflowpy/flopy)
[![PyPI Version](https://img.shields.io/pypi/v/flopy.png)](https://pypi.python.org/pypi/flopy)
[![PyPI Downloads](https://img.shields.io/pypi/dm/flopy.png)](https://pypi.python.org/pypi/flopy)


## Introduction

*FloPy<sub>3</sub>* includes support for MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT. Other supported MODFLOW-based models include MODPATH (version 6), MT3D and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users  Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.

If you think you have found a bug in *FloPy<sub>3</sub>*, or if you would like to suggest an improvement or enhancement, please submit a new Issue through the Github Issue tracker toward the upper-right corner of this page. Pull requests will only be accepted on the develop branch of the repository.


Documentation
-----------------------------------------------

Documentation for *FloPy<sub>3</sub>* is a work in progress. *FloPy<sub>3</sub>* code documentation is available at:

+ [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)

## Examples

### IPython Notebook Examples

The following IPython Notebooks contain example FloPy scripts for a variety of models and FloPy features

#### Basic examples

+ An overview of the options to enter *layer, row, column, data* values for packages such as the wel and ghb packages is given in the [flopy3boundaries](examples/Notebooks/flopy3boundaries.ipynb) Notebook

+ An overview of how to control the format of numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_array_outputformat_options](examples/Notebooks/flopy3_array_outputformat_options.ipynb) Notebook.

+ An overview of how FloPy3 handles external files for numeric arrays written by FloPy to MODFLOW-based input files is given in the [flopy3_external_file_handling](examples/Notebooks/flopy3_external_file_handling.ipynb) Notebook.

+ An overview of FloPy3 capabilities to load a SFR2 file and evaluate data contained in the file is given in the [flopy3_SFR2_load](examples/Notebooks/flopy3_SFR2_load.ipynb) Notebook.

+ The [lake example](examples/Notebooks/lake_example.ipynb), a very simple *FloPy<sub>3</sub>* example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 

+ A variant of the [water-table example](examples/Notebooks/flopy3_WatertableRecharge_example.ipynb), a very simple example of one-dimensional groundwater flow in an unconfined aquifer with recharge, from the MODFLOW-NWT documentation (http://pubs.usgs.gov/tm/tm6a37/). This IPython Notebook build files for MODFLOW-NWT.

+ The [Zaidel discontinuous water-table example](examples/Notebooks/flopy3_Zaidel_example.ipynb), which simulates a discontinuous water table over a stairway impervious base, from http://onlinelibrary.wiley.com/doi/10.1111/gwat.12019/abstract. This IPython Notebook build files for MODFLOW-USG. (http://pubs.usgs.gov/tm/06/a45/). 

+ An overview of the options for creating a Source/Sink Mixing (SSM) Package for MT3DMS and SEAWAT is given in the [flopy3ssm](examples/Notebooks/flopy3_multi-component_SSM.ipynb) Notebook.

+ The [Henry Problem](examples/Notebooks/henry.ipynb), a simple saltwater intrusion model developed with Flopy and run using SEAWAT.

#### SWI2 examples

+ [Example 1](examples/Notebooks/swiex1.ipynb) of the SWI2 manual, simulating a rotating interface.

+ [Example 4](examples/Notebooks/swiex4.ipynb) of the SWI2 manual, upconing below a pumping well below a two-aquifer island system.

#### Plotting examples

+ An overview of the *FloPy<sub>3</sub>* [map and cross-section plotting capabilities](examples/Notebooks/flopy3_MapExample.ipynb).

+ An overview of the *FloPy<sub>3</sub>*  [model input and output data `plot()` method capabilities](examples/Notebooks/flopy3_PlotArrayExample.ipynb)

#### Export examples

+ An overview of the *FloPy<sub>3</sub>* [netCDF export capabilities](examples/Notebooks/flopy3_netCDF.ipynb).

#### Parameter Estimation examples

+ An overview of the *FloPy<sub>3</sub>* [parameter estimation capabilities](examples/Notebooks/flopy3_PEST.ipynb).

#### Additional MODFLOW examples

+ Example problems from the 2015 2nd edition of [Applied Groundwater Modeling](https://github.com/Applied-Groundwater-Modeling-2nd-Ed) by Mary P. Anderson, William W. Woessner, and Randall J. Hunt (https://github.com/Applied-Groundwater-Modeling-2nd-Ed)

### Python Script Examples

*FloPy<sub>3</sub>* scripts for running and post-processing the lake example and SWI2 Examples (examples 1 to 5) that are described in [Bakker et al. (2013)](http://pubs.usgs.gov/tm/6a46/) are available:

+ [Lake Example](examples/scripts/lake_example.py)

+ [SWI2 Example 1](examples/scripts/flopy_swi2_ex1.py)

+ [SWI2 Example 2](examples/scripts/flopy_swi2_ex2.py)

+ [SWI2 Example 3](examples/scripts/flopy_swi2_ex3.py)

+ [SWI2 Example 4](examples/scripts/flopy_swi2_ex4.py)

+ [SWI2 Example 5](examples/scripts/flopy_swi2_ex5.py)

Note that examples 2 and 5 also include *FloPy<sub>3</sub>* code for running and post-processing SEAWAT models.


### Tutorials

A few simple *FloPy<sub>3</sub>* tutorials are available at:

+ [http://modflowpy.github.io/flopydoc/tutorials.html](http://modflowpy.github.io/flopydoc/tutorials.html)


## Installation

**Python versions:**

*FloPy<sub>3</sub>* requires **Python** 2.7 or **Python** 3.3 (or higher)


**Dependencies:**

*FloPy<sub>3</sub>* requires **NumPy** 1.9 (or higher) and **matplotlib** 1.4 (or higher). The mapping and cross-section capabilities in the `flopy.plot` submodule and shapefile export capabilities (`to_shapefile()`) require **Pyshp** 1.2 (or higher). The NetCDF export capabilities in the `flopy.export` submodule require **python-dateutil** 2.4 (or higher), **netcdf4** 1.1 (or higher), and **pyproj** 1.9 (or higher). Other NetCDF dependencies are detailed on the [UniData](http://unidata.github.io/netcdf4-python/) website. The `get_dataframes` method in the `ListBudget` class in the `flopy.utils` submodule require **pandas** 0.15 (or higher).


**For base Python distributions:**

To install *FloPy<sub>3</sub>* type:

    pip install flopy

To update *FloPy<sub>3</sub>* type:

    pip install flopy --upgrade

To uninstall *FloPy<sub>3</sub>* type:

    pip uninstall flopy

**Installing from the git repository:**

***Current Version of FloPy<sub>3</sub>:***

To install the current version of *FloPy<sub>3</sub>* from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master
    
To update your version of *FloPy<sub>3</sub>* with the current version from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master --upgrade

***Development version of FloPy<sub>3</sub>:***

To install the bleeding edge version of *FloPy<sub>3</sub>* from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop
    
To update your version of *FloPy<sub>3</sub>* with the bleeding edge code from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop --upgrade


## FloPy<sub>3</sub> Supported Packages

A list of supported packages in *FloPy<sub>3</sub>* is available in following [markdown document](docs/supported_packages.md) on the github repo.


--------------------------------

## FloPy<sub>3</sub> Changes

A summary of changes in each *FloPy<sub>3</sub>* version is available in the following [markdown document](doc/version_changes.md) on the github repo.


--------------------------------

### MODFLOW Resources

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)
