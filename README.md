
![flopy3](https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png)

## Introduction

*FloPy* includes support for MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT. Other supported MODFLOW-based models include MODPATH (version 6), MT3D and SEAWAT.

## FloPy3 Changes

FloPy3 is significantly different from FloPy2 (hosted on [googlecode](https://code.google.com/p/flopy/)). The main changes are:

* FloPy3 is fully zero-based. This means that layers, rows and columns start counting at *zero*. The reason for this is consistency. Arrays are zero-based by default in Python, so it was confusing to have a mix.
* Input for packages that take *layer,row,column,data* input (like the wel or ghb package) has changed and is much more flexible now. See the notebook [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/basic/flopy3boundaries.ipynb)

## Installation

To install *FloPy* type:

    pip install flopy

To update *FloPy* type:

    pip install flopy --update

To uninstall *FloPy* type:

    pip uninstall flopy


Documentation
-----------------------------------------------

Documentation for *FloPy* is a work in progress. *FloPy* code documentation is available at:

+ [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)

## Examples

### IPython Notebook Examples

The following IPython Notebooks contain example FloPy scripts for a variety of models and FloPy features

#### Basic examples

+ The [lake example](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/basic/lake_example.ipynb), a very simple FloPy example of steady flow in a square model with a fixed head cell in the middle (representing a lake) in a 10-layer model. 
+ An overview of the options to enter *layer,row,column,data* values for packages such as the wel and ghb packages is given in the [flopy3boundaries](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/basic/flopy3boundaries.ipynb) Notebook

#### SWI2 examples

+ [Example 1](http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/swi_examples/swiex1.ipynb) of the SWI2 manual, simulating a rotating interface.

### SWI2 Test Problems for FloPy2

A zip file containing *FloPy* scripts for running and post-processing the SWI2 Examples (examples 1 to 5) that are described in [Bakker et al. (2013)](http://pubs.usgs.gov/tm/6a46/) is available at:

+ [http://flopy.googlecode.com/svn/examples/SWI2ExampleProblems_flopy.zip](http://flopy.googlecode.com/svn/examples/SWI2ExampleProblems_flopy.zip)

Note that examples 2 and 5 also include *FloPy* scripts for running and post-processing SEAWAT models.


### Tutorials

A few simple *FloPy* tutorials are available at:

+ [http://modflowpy.github.io/flopydoc/tutorials.html](http://modflowpy.github.io/flopydoc/tutorials.html)


### MODFLOW Resources

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)
