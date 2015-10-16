
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.2.3

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

+ An overview of the options to enter *layer, row, column, data* values for packages such as the wel and ghb packages is given in the [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3boundaries.ipynb) Notebook

+ The [lake example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/lake_example.ipynb), a very simple *FloPy<sub>3</sub>* example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 

+ A variant of the [water-table example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_WatertableRecharge_example.ipynb), a very simple example of one-dimensional groundwater flow in an unconfined aquifer with recharge, from the MODFLOW-NWT documentation (http://pubs.usgs.gov/tm/tm6a37/). This IPython Notebook build files for MODFLOW-NWT.

+ The [Zaidel discontinuous water-table example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_Zaidel_example.ipynb), which simulates a discontinuous water table over a stairway impervious base, from http://onlinelibrary.wiley.com/doi/10.1111/gwat.12019/abstract. This IPython Notebook build files for MODFLOW-USG. (http://pubs.usgs.gov/tm/06/a45/). 

+ An overview of the options for creating a Source/Sink Mixing (SSM) Package for MT3DMS and SEAWAT is given in the [flopy3ssm](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_multi-component_SSM.ipynb) Notebook.

+ The [Henry Problem](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/henry.ipynb), a simple saltwater intrusion model developed with Flopy and run using SEAWAT.

#### SWI2 examples

+ [Example 1](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/swiex1.ipynb) of the SWI2 manual, simulating a rotating interface.

+ [Example 4](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/swiex4.ipynb) of the SWI2 manual, upconing below a pumping well below a two-aquifer island system.

#### Plotting examples

+ An overview of the *FloPy<sub>3</sub>* [map and cross-section plotting capabilities](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_MapExample.ipynb).
+ An overview of the *FloPy<sub>3</sub>*  [model input and output data `plot()` method capabilities](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_PlotArrayExample.ipynb)

#### Parameter Estimation examples

+ An overview of the *FloPy<sub>3</sub>* [parameter estimation capabilities](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_PEST.ipynb).

#### Additional MODFLOW examples

+ Example problems from the 2015 2nd edition of [Applied Groundwater Modeling](https://github.com/Applied-Groundwater-Modeling-2nd-Ed) by Mary P. Anderson, William W. Woessner, and Randall J. Hunt (https://github.com/Applied-Groundwater-Modeling-2nd-Ed)

### SWI2 Test Problems for *FloPy<sub>3</sub>*

*FloPy<sub>3</sub>* scripts for running and post-processing the SWI2 Examples (examples 1 to 5) that are described in [Bakker et al. (2013)](http://pubs.usgs.gov/tm/6a46/) are available:

+ [SWI2 Example 1](https://github.com/modflowpy/flopy/blob/master/examples/scripts/flopy_swi2_ex1.py)

+ [SWI2 Example 2](https://github.com/modflowpy/flopy/blob/master/examples/scripts/flopy_swi2_ex2.py)

+ [SWI2 Example 3](https://github.com/modflowpy/flopy/blob/master/examples/scripts/flopy_swi2_ex3.py)

+ [SWI2 Example 4](https://github.com/modflowpy/flopy/blob/master/examples/scripts/flopy_swi2_ex4.py)

+ [SWI2 Example 5](https://github.com/modflowpy/flopy/blob/master/examples/scripts/flopy_swi2_ex5.py)

Note that examples 2 and 5 also include *FloPy<sub>3</sub>* code for running and post-processing SEAWAT models.


### Tutorials

A few simple *FloPy<sub>3</sub>* tutorials are available at:

+ [http://modflowpy.github.io/flopydoc/tutorials.html](http://modflowpy.github.io/flopydoc/tutorials.html)


## Installation

**Python versions:**

*FloPy<sub>3</sub>* requires **Python** 2.7 or **Python** 3.3 (or higher)


**Dependencies:**

*FloPy<sub>3</sub>* requires **NumPy** 1.9 (or higher) and **matplotlib** 1.4 (or higher). The mapping and cross-section capabilities in the flopy.plot submodule require **Pyshp** 1.2 (or higher).


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
### MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT
| Package                                        | Creation and Write   | Load Available       | Template Creation    |
| ---------------------------------------------- | -------------------- | -------------------- | -------------------- |
| Basic (BAS6)                                   | Supported            | Supported            | Not supported        |
| Block Centered Flow (BCF)                      | Supported            | Supported            | Not supported        |
| Direct Solver (DE4)                            | Supported            | Supported            | Not supported        |
| Discretization (DIS)                           | Supported            | Supported            | Not supported        |
| Drain (DRN)                                    | Supported            | Supported            | Not supported        |
| Drain Observation (DROB)                       | Not supported        | Not supported        | Not supported        |
| Drain Return (DRT)                             | Not supported        | Not supported        | Not supported        |
| Evapotranspiration (EVT)                       | Supported            | Supported            | Not supported        |
| Evapotranspiration Segments (ETS)              | Not supported        | Not supported        | Not supported        |
| Flow and Head Boundary (FHB)                   | Not supported        | Not supported        | Not supported        |
| Gage (GAGE)                                    | Not supported        | Not supported        | Not supported        |
| General Head Boundary (GHB)                    | Supported            | Supported            | Not supported        |
| General Head Boundary Observation (GBOB)       | Not supported        | Not supported        | Not supported        |
| Geometric Multi-Grid (GMG)                     | Supported            | Supported            | Not supported        |
| Head Observation (HOB)                         | Not supported        | Not supported        | Not supported        |
| Horizontal Flow Barrier (HFB)                  | Supported            | Supported            | Not supported        |
| HYDMOD (HYD)                                   | Not supported        | Not supported        | Not supported        |
| Hydrogeologic-Unit Flow (HUF)                  | Not supported        | Not supported        | Not supported        |
| Flow and Head Boundary (FHB)                   | Not supported        | Not supported        | Not supported        |
| Interbed-Storage (IBS)                         | Not supported        | Not supported        | Not supported        |
| Layer Property Flow (LPF)                      | Supported            | Supported            | Supported            |
| Link-AMG (LMG)                                 | Not supported        | Not supported        | Not supported        |
| MODFLOW Link-MT3DMS (LMT)                      | Supported            | Not supported        | Not supported        |
| Multipler (MULT)                               | Not supported        | Supported            | Not supported        |
| Multi-Node Well 1 (MNW1)                       | Limited support      | Not supported        | Not supported        |
| Multi-Node Well 2 (MNW1)                       | Limited support      | Not supported        | Not supported        |
| Multi-Node Well Information (MNWI)             | Limited support      | Not supported        | Not supported        |
| Newton (NWT)                                   | Supported            | Not supported        | Not supported        |
| Output Control (OC)                            | Supported            | Supported            | Not supported        |
| Periodic Boundary Condition (PBC)              | Supported            | Not supported        | Not supported        |
| Preconditioned Conjugate <br />Gradient (PCG)              | Supported            | Supported            | Not supported        |
| Precondonditioned Conjugate <br />Gradient Nonlinear (PCGN)   | Supported            | Not supported        | Not supported        |
| Parameter Value (PVAL)                         | Not supported        | Supported            | Not supported        |
| Recharge (RCH)                                 | Supported            | Supported            | Not supported        |
| River (RIV)                                    | Supported            | Supported            | Not supported        |
| River Observation (RVOB)                       | Not supported        | Not supported        | Not supported        |
| Streamflow Routing (SFR2)                      | Supported            | Supported            | Not supported        |
| Strongly Implicit Procedure (SIP)              | Supported            | Not supported        | Not supported        |
| Slice-successive Overrelaxation (SOR)          | Supported            | Not supported        | Not supported        |
| Stream (STR)                                   | Supported            | Supported            | Not supported        |
| Stream Observation (STOB)                      | Not supported        | Not supported        | Not supported        |
| Seawater Intrusion (SWI)                       | Supported            | Not supported        | Not supported        |
| Seawater Intrusion (SWI2)                      | Supported            | Supported            | Not supported        |
| Surface-Water Routing (SWR)                    | Not supported        | Not supported        | Not supported        |
| Subsidence (SUB)                               | Not supported        | Not supported        | Not supported        |
| Subsidence and Aquifer-System <br />Compaction (SWT) | Not supported        | Not supported        | Not supported        |
| Specified-Head Flow Observation (CHOB)         | Not supported        | Not supported        | Not supported        |
| Upstream Weighted (UPW)                        | Supported            | Supported            | Not supported        |
| Unzaturated Zone Flow (UZF)                    | Supported            | Not supported        | Not supported        |
| Well (WEL)                                     | Supported            | Supported            | Not supported        |
| Zone (ZONE)                                    | Not supported        | Supported            | Not supported        |

### MODFLOW-USG
| Package                                      | Creation and Write   | Load Available       | Template Creation    |
| -------------------------------------------- | -------------------- | -------------------- | -------------------- |
| Sparse Matrix Solver (SMS)                   | Supported            | Not supported        | Not supported        |

### MODPATH
| Package                                  | Creation and Write   | Load Available       | Template Creation    |
| ---------------------------------------- | -------------------- | -------------------- | -------------------- |
| MODPATH Basic (MPBAS)                    | Supported            | Not supported        | Not supported        |
| MODPATH Simulation (MPSIM)               | Supported            | Not supported        | Not supported        |

### MT3DMS
| Package                                  | Creation and Write   | Load Available       | Template Creation    |
| ---------------------------------------- | -------------------- | -------------------- | -------------------- |
| Advection (ADV)                          | Supported            | Not supported        | Not supported        |
| Basic Transport (BTN)                    | Supported            | Not supported        | Not supported        |
| Dispersion (DSP)                         | Supported            | Not supported        | Not supported        |
| Generalized Conjugate Gradient (GCG)     | Supported            | Not supported        | Not supported        |
| PHT3D-PHREEQC Interface (PHC)            | Supported            | Not supported        | Not supported        |
| Reaction (RCT)                           | Supported            | Not supported        | Not supported        |
| Sink and Source Mixing (SSM)             | Supported            | Not supported        | Not supported        |
| Transport Observation (TOB)              | Supported            | Not supported        | Not supported        |

### SEAWAT
| Package                                  | Creation and Write   | Load Available       | Template Creation    |
| ---------------------------------------- | -------------------- | -------------------- | -------------------- |
| Variable Density Flow (VDF)              | Supported            | Not supported        | Not supported        |
| Viscosity (VSC)                          | Supported            | Not supported        | Not supported        |


--------------------------------

## FloPy<sub>3</sub> Changes

### Version 3.2.3
* `flopy.utils.CellBudgetFile` returns a numpy recarray for list based budget data. Previously a dictionary with the `node` number and `q` were returned. The recarray will return the `node` number, `q`, and the `aux` variables for list based budget data.
* Added support for the SEAWAT viscosity (VSC) package.

### Version 3.2.2
* *FloPy<sub>3</sub>* now supports some simple plotting capabilities for two- and three-dimensional model input data array classes  and transient two-dimensional stress period input data using the `.plot()` methods associated with the data array classes (`util_2d`, `util_3d`, and `transient_2d`). The model results reader classes (`HeadFile`, `UcnFile`, and `CellBudgetFile`) have also been extended to include a `.plot()` method that can be used to create simple plots of model output data. See the notebook [flopy3_PlotArrayExample](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_PlotArrayExample.ipynb).

* Added `.to_shapefile()` method to two- and three-dimensional model input data array classes (`util_2d` and `util_3d`), transient two-dimensional stress period input data classes (`transient_2d`), and model output data classes (`HeadFile`, `UcnFile`, and `CellBudgetFile`) that allows model data to be exported as polygon shapefiles with separate attribute columns for each model layer.

* Added support for ASCII model results files.

* Added support for reading MODPATH version 6 pathline and endpoint output files and plotting MODPATH results using mapping capabilities in `flopy.plot` submodule.

* Added `load()` method for MODFLOW GMG solver.

* Bug fixes:
  1. Multipler in array control record was not being applied to arrays
  2. vani parameter was not supported

### Version 3.2.1
* *FloPy<sub>3</sub>* can now be used with **Python 3.x**

* Revised setters for package class variables stored using the `util_2d` or `util_3d` classes.

* Added option to load a subset of MODFLOW packages in a MODFLOW model name file using `load_only=` keyword.

### Version 3.1
* *FloPy<sub>3</sub>* now supports some simple mapping and cross-section capabilities through the `flopy.plot` submodule. See the notebook [flopy3_MapExample](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_MapExample.ipynb).

* Full support for all Output Control (OC) options including DDREFERENCE, SAVE IBOUND, and layer lists. All Output Control Input is specified using words. Output Control Input using numeric codes is still available in the ModflowOc88 class. The ModflowOc88 class is currently deprecated and no longer actively maintained.

* Added support for standard MULT package FUNCTION and EXPRESSION functionality are supported. MODFLOW parameters are not supported in `write()` methods. 

### Version 3.0

*FloPy<sub>3</sub>* is significantly different from *FloPy<sub>2</sub>* (previously hosted on [googlecode](https://code.google.com/p/flopy/)). The main changes are:

* *FloPy<sub>3</sub>* is fully zero-based. This means that layers, rows and columns start counting at *zero*. The reason for this is consistency. Arrays are zero-based by default in Python, so it was confusing to have a mix.

* Input for packages that take *layer, row, column, data* input (like the wel or ghb package) has changed and is much more flexible now. See the notebook [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3boundaries.ipynb)

* Input for the MT3DMS Source/Sink Mixing (SSM) Package has been modified to be consistent with the new MODFLOW boundary package input and is more flexible than previous versions of *FloPy*. See the notebook [flopy3ssm](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_multi-component_SSM.ipynb)

* Support for use of EXTERNAL and OPEN/CLOSE array specifiers has been improved.

* *load()* methods have been developed for all of the standard MODFLOW packages and a few less used packages (*e.g.* SWI2).

* MODFLOW parameter support has been added to the `load()` methods. MULT, PVAL, and ZONE packages are now supported and parameter data are converted to arrays in the `load()` methods. MODFLOW parameters are not supported in `write()` methods.  

--------------------------------

### MODFLOW Resources

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)
