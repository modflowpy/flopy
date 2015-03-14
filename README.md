
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

## Introduction

*FloPy<sub>3</sub>* includes support for MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT. Other supported MODFLOW-based models include MODPATH (version 6), MT3D and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users  Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed at the bottom of this page in the MODFLOW Resources section.

If you think you have found a bug in *FloPy<sub>3</sub>*, or if you would like to suggest an improvement or enhancement, please submit a new Issue through the Github Issue tracker toward the upper-right corner of this page.

## FloPy<sub>3</sub> Changes

*FloPy<sub>3</sub>* is significantly different from *FloPy<sub>2</sub>* (hosted on [googlecode](https://code.google.com/p/flopy/)). The main changes are:

* *FloPy<sub>3</sub>* is fully zero-based. This means that layers, rows and columns start counting at *zero*. The reason for this is consistency. Arrays are zero-based by default in Python, so it was confusing to have a mix.
* Input for packages that take *layer, row, column, data* input (like the wel or ghb package) has changed and is much more flexible now. See the notebook [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3boundaries.ipynb)
* Support for use of EXTERNAL and OPEN/CLOSE array specifiers has been improved.
* *load()* methods have been developed for all of the standard MODFLOW packages and a few less used packages (*e.g.* SWI2).
* MODFLOW parameter support has been added to the *load()* methods. MULT, PVAL, and ZONE packages are now supported and parameter data are converted to arrays in the *load()* methods. MODFLOW parameters are not supported in *write()* methods. MULT package FUNCTION and EXPRESSION functionality are not currently supported. 
* Full support for all Output Control (OC) options including DDREFERENCE, SAVE IBOUND, and layer lists. All Output Control Input is specified using words. Output Control Input using numeric codes is still available in the ModflowOc88 class. The ModflowOc88 class is currently deprecated and no longer actively maintained.
* *FloPy<sub>3</sub>* now supports some simple mapping capabilities through the flopy.plot submodule. See the notebook [flopy3_MapExample](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_MapExample.ipynb).

## Installation

**Python versions:**

*FloPy<sub>3</sub>* requires **Python** 2.7


**Dependencies:**

*FloPy<sub>3</sub>* requires **NumPy** 1.9 (or higher) and **matplotlib** 1.4 (or higher)


**For base Python distributions:**

To install *FloPy<sub>3</sub>* type:

    pip install flopy

To update *FloPy<sub>3</sub>* type:

    pip install flopy --upgrade

To uninstall *FloPy<sub>3</sub>* type:

    pip uninstall flopy


**Development versions of FloPy<sub>3</sub>:**

To install the bleeding edge version of *FloPy<sub>3</sub>* from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master
    
To update your version of *FloPy<sub>3</sub>* with the bleeding edge code from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master --upgrade


Documentation
-----------------------------------------------

Documentation for *FloPy<sub>3</sub>* is a work in progress. *FloPy<sub>3</sub>* code documentation is available at:

+ [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)

## Examples

### IPython Notebook Examples

The following IPython Notebooks contain example FloPy scripts for a variety of models and FloPy features

#### Basic examples

+ An overview of the options to enter *layer,row,column,data* values for packages such as the wel and ghb packages is given in the [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3boundaries.ipynb) Notebook
+ The [lake example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/lake_example.ipynb), a very simple *FloPy<sub>3</sub>* example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 
+ A variant of the [water-table example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_WatertableRecharge_example.ipynb), a very simple example of one-dimensional groundwater flow in an unconfined aquifer with recharge, from the MODFLOW-NWT documentation (http://pubs.usgs.gov/tm/tm6a37/). This IPython Notebook build files for MODFLOW-NWT.
+ The [Zaidel discontinuous water-table example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/flopy3_Zaidel_example.ipynb), which simulates a discontinuous water table over a stairway impervious base, from http://onlinelibrary.wiley.com/doi/10.1111/gwat.12019/abstract. This IPython Notebook build files for MODFLOW-USG. (http://pubs.usgs.gov/tm/06/a45/). 
+ The [Henry Problem](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/henry.ipynb), a simple saltwater intrusion model developed with Flopy and run using SEAWAT.

#### SWI2 examples

+ [Example 1](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/swiex1.ipynb) of the SWI2 manual, simulating a rotating interface.

+ [Example 4](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/Notebooks/swiex4.ipynb) of the SWI2 manual, upconing below a pumping well below a two-aquifer island system.

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


### MODFLOW Resources

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)
