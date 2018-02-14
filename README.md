
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.2.8 develop &mdash; build 73
[![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=develop)](https://travis-ci.org/modflowpy/flopy)
[![PyPI Version](https://img.shields.io/pypi/v/flopy.png)](https://pypi.python.org/pypi/flopy)
[![Coverage Status](https://coveralls.io/repos/github/modflowpy/flopy/badge.svg?branch=develop)](https://coveralls.io/github/modflowpy/flopy?branch=develop)


Introduction
-----------------------------------------------

FloPy includes support for MODFLOW-2000, MODFLOW-2005, MODFLOW-NWT, and MODFLOW-USG. Other supported MODFLOW-based models include MODPATH (version 6), MT3DMS, MT3D-USGS,  and SEAWAT.

FloPy now includes beta support for MODFLOW 6.  Click [here](docs/mf6.md) for more information.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.

If you think you have found a bug in FloPy, or if you would like to suggest an improvement or enhancement, please submit a new issue through the [Github Issue tracker](https://github.com/modflowpy/flopy/issues). Pull requests will only be accepted on the develop branch of the repository.


Documentation
-----------------------------------------------

FloPy code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


How to Cite
-----------------------------------------------

##### ***Citation for FloPy:***

[Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.](http://dx.doi.org/10.1111/gwat.12413)

##### ***Software/Code citation for FloPy:***

[Bakker, M., Post, V., Langevin, C.D., Hughes, J.D., White, J.T., Starn, J.J., and Fienen, M.N., 2018, FloPy v3.2.8 &mdash; develop: U.S. Geological Survey Software Release, 14 February 2018, http://dx.doi.org/10.5066/F7BK19FH](http://dx.doi.org/10.5066/F7BK19FH)


Examples
-----------------------------------------------

### [jupyter Notebook Examples](docs/notebook_examples.md)

### [Python Script Examples](docs/script_examples.md)

### [Tutorials](http://modflowpy.github.io/flopydoc/tutorials.html)


Installation
-----------------------------------------------

**Python versions:**

FloPy requires **Python** 2.7 or **Python** 3.3 (or higher)


**Dependencies:**

FloPy requires **NumPy** 1.9 (or higher) and **enum34** for **Python** 2.7 or **Python** 3.3.


***Optional Method Dependencies:***

Additional dependencies to use optional FloPy helper methods are listed below.

| Method                                                                               | Python Package                                     |
| ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `.plot()`                                                                            | **matplotlib** >= 1.4                              |
| `.plot_shapefile()`                                                                  | **matplotlib** >= 1.4 and **Pyshp** >= 1.2         |
| `.to_shapefile()`                                                                    | **Pyshp** >= 1.2                                   |
| `.export(*.shp)`                                                                     | **Pyshp** >= 1.2                                   |
| `.export(*.nc)`                                                                      | **netcdf4** >= 1.1 and **python-dateutil** >= 2.4  |
| `.export(*.tif)`                                                                     | **rasterio**                                       |
| `.export(*.asc)` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.ndimage**                                  |
| `.interpolate()` in `flopy.utils.reference` `SpatialReference` class                 | **scipy.interpolate**                              |
| `.interpolate()` in `flopy.mf6.utils.reference` `StructuredSpatialReference` class   | **scipy.interpolate**                              |
| `.get_dataframes()` in `flopy.utils.mflistfile` `ListBudget` class                   | **pandas** >= 0.15                                 |
| `.get_dataframes()` in `flopy.utils.observationfile` `ObsFiles` class                | **pandas** >= 0.15                                 |
| `.get_dataframes()` in `flopy.utils.sfroutputfile` `ModflowSfr2` class               | **pandas** >= 0.15                                 |
| `.get_dataframes()` in `flopy.utils.util_list` `MfList` class                        | **pandas** >= 0.15                                 |
| `.get_dataframes()` in `flopy.utils.zonebud` `ZoneBudget` class                      | **pandas** >= 0.15                                 |
| `.pivot_keyarray()` in `flopy.mf6.utils.arrayutils` `AdvancedPackageUtil` class      | **pandas** >= 0.15                                 |
| `._get_vertices()` in `flopy.mf6.utils.binaryfile_utils` `MFOutputRequester` class   | **pandas** >= 0.15                                 |
| `.get_dataframe()` in `flopy.mf6.utils.mfobservation` `Observations` class           | **pandas** >= 0.15                                 |
| `.df()` in `flopy.modflow.mfsfr2` `SfrFile` class                                    | **pandas** >= 0.15                                 |
| `.time_coverage()` in `flopy.export.metadata` `acc` class - ***used if available***  | **pandas** >= 0.15                                 |
| `.loadtxt()` in `flopy.utils.flopyio` - ***used if available***                      | **pandas** >= 0.15                                 |


**For base Python distributions:**

To install FloPy type:

    pip install flopy

To update FloPy type:

    pip install flopy --upgrade

To uninstall FloPy type:

    pip uninstall flopy

**Installing from the git repository:**

***Current Version of FloPy:***

To install the current version of FloPy from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master

To update your version of FloPy with the current version from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master --upgrade

***Development version of FloPy:***

To install the bleeding edge version of FloPy from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop

To update your version of FloPy with the bleeding edge code from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop --upgrade


FloPy Supported Packages
-----------------------------------------------

A list of supported packages in FloPy is available in [docs/supported_packages.md](docs/supported_packages.md) on the github repo.


FloPy Model Checks
-----------------------------------------------

A table of the supported and proposed model checks implemented in  FloPy is available in [docs/model_checks.md](docs/model_checks.md) on the github repo.


FloPy Changes
-----------------------------------------------

A summary of changes in each FloPy version is available in [docs/version_changes.md](docs/version_changes.md) on the github repo.


MODFLOW Resources
-----------------------------------------------

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)
