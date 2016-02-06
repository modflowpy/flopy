!![FloPy](https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png)

### Version 3.2.4
[![Build Status](https://travis-ci.org/modflowpy/flopy.png?branch=develop)](https://travis-ci.org/modflowpy/flopy)


Introduction
-----------------------------------------------

*FloPy<sub>3</sub>* includes support for MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT. Other supported MODFLOW-based models include MODPATH (version 6), MT3D and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users  Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.

If you think you have found a bug in *FloPy<sub>3</sub>*, or if you would like to suggest an improvement or enhancement, please submit a new Issue through the Github Issue tracker toward the upper-right corner of this page. Pull requests will only be accepted on the develop branch of the repository.


Documentation
-----------------------------------------------

*FloPy<sub>3</sub>* code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


Installation
-----------------------------------------------

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


