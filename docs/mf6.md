Introduction
-----------------------------------------------
[MODFLOW 6](https://water.usgs.gov/ogw/modflow/MODFLOW.html) is the latest core release of the U.S. Geological Survey's MODFLOW model.  MODFLOW 6 combines many of the capabilities of previous MODFLOW versions into a single code.  A major change in MODFLOW 6 is the redesign of the input file structure to use blocks and keywords.  Because of this change, existing Flopy classes can no longer be used.

This Flopy release contains beta support for an entirely new set of classes to support MODFLOW 6 models.  These classes were designed from scratch and are based on "definition files", which describe the block and keywords for each input file.  These new MODFLOW 6 capabilities in Flopy are considered beta, because they need additional testing, and because it is likely that some of the underlying code will change to meet user needs. 

There are many important differences between the Flopy classes for MODFLOW 6 and the Flopy classes for previous MODFLOW versions.  Unfortunately, this will mean that existing Flopy users will need to rewrite some parts of their scripts in order to generate MODFLOW 6 models.  Because the new classes are updated programmatically from definition files, it will be easier to keep them up-to-date with new MODFLOW 6 capabilities as they are released. 


Models and Packages
-----------------------------------------------
The following is a list of the classes available to work with MODFLOW 6 models.  These classes should support all of the options available in the current version of MODFLOW 6.

* MFSimulation
* MFModel
* ModflowNam
* ModflowTdis
* ModflowGwfgwf
* ModflowIms
* ModflowMvr
* ModflowGnc
* ModflowUtlobs
* ModflowUtlts
* ModflowUtltab
* ModflowUtltas
* ModflowGwfnam
* ModflowGwfdis
* ModflowGwfdisv
* ModflowGwfdisu
* ModflowGwfic
* ModflowGwfnpf
* ModflowGwfsto
* ModflowGwfhfb
* ModflowGwfchd
* ModflowGwfwel
* ModflowGwfdrn
* ModflowGwfriv
* ModflowGwfghb
* ModflowGwfrch
* ModflowGwfrcha
* ModflowGwfevt
* ModflowGwfevta
* ModflowGwfmaw
* ModflowGwfsfr
* ModflowGwflak
* ModflowGwfuzf
* ModflowGwfmvr
* ModflowGwfgnc
* ModflowGwfoc


Getting Help
-----------------------------------------------
Help for these classes is limited at the moment.  The best way to understand these new capabilities is to look at the Jupyter Notebooks.  There are also several scripts available in the flopy/autotest folder that may provide additional information on creating, writing, and loading MODFLOW 6 models with Flopy.


Notebooks
-----------------------------------------------
Instructions for using the new MODFLOW 6 capabilities in Flopy are described in the following Jupyter Notebooks.

1. Building the simple lake model ([flopy3_mf6_A_simple-model](../examples/Notebooks/flopy3_mf6_A_simple-model.ipynb) Notebook)
2. Building a complex model ([flopy3_mf6_B_complex-model](../examples/Notebooks/flopy3_mf6_B_complex-model.ipynb) Notebook)
3. Building an LGR model (coming soon)
4. Understanding Flopy data objects for MODFLOW 6 (coming soon)
5. Loading models, making changes, and rerunning (coming soon)


Future Plans
-----------------------------------------------
- Improved documentation
- New plotting capabilities and refactoring of the current spatial reference class
- Support for exporting to netcdf, vtk, and other formats

