
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.2.12
[![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=master)](https://travis-ci.org/modflowpy/flopy)
[![PyPI Version](https://img.shields.io/pypi/v/flopy.png)](https://pypi.python.org/pypi/flopy)
[![Coverage Status](https://coveralls.io/repos/github/modflowpy/flopy/badge.svg?branch=master)](https://coveralls.io/github/modflowpy/flopy?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/b23a5edd021b4aa19e947545ab49e577)](https://www.codacy.com/app/jdhughes-usgs/flopy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=modflowpy/flopy&amp;utm_campaign=Badge_Grade)

Introduction
-----------------------------------------------

FloPy includes support for [MODFLOW 6](docs/mf6.md), MODFLOW-2005, MODFLOW-NWT, MODFLOW-USG, and MODFLOW-2000. Other supported MODFLOW-based models include MODPATH (version 6 and 7), MT3DMS, MT3D-USGS, and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.


Contributing
------------------------------------------------

Bug reports, code contributions, or improvements to the documentation are welcome from the community. Prior to contributing, please read up on our guidelines for [contributing](CONTRIBUTING.md) and then check out one of our issues in the [hotlist: community-help](https://github.com/modflowpy/flopy/labels/hotlist%3A%20community%20help).


Documentation
-----------------------------------------------

FloPy code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


Getting Started
-----------------------------------------------

### MODFLOW 6 Quick Start
```python
import os
import flopy
ws = './mymodel'
name = 'mymodel'
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6')
tdis = flopy.mf6.ModflowTdis(sim)
ims = flopy.mf6.ModflowIms(sim)
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
ic = flopy.mf6.ModflowGwfic(gwf)
npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                       [(0, 9, 9), 0.]])
budget_file = name + '.bud'
head_file = name + '.hds'
oc = flopy.mf6.ModflowGwfoc(gwf,
                            budget_filerecord=budget_file,
                            head_filerecord=head_file,
                            saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
sim.write_simulation()
sim.run_simulation()
head = flopy.utils.HeadFile(os.path.join(ws, head_file)).get_data()
bud = flopy.utils.CellBudgetFile(os.path.join(ws, budget_file),
                                 precision='double')
spdis = bud.get_data(text='DATA-SPDIS')[0]
pmv = flopy.plot.PlotMapView(gwf)
pmv.plot_array(head)
pmv.plot_grid(colors='white')
pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.)
pmv.plot_specific_discharge(spdis, color='white')
```
<img src="examples/images/quickstart.png" alt="plot" style="width:30;height:30">

### [Frequently asked questions](docs/flopyFAQ.md)

### [Tutorials](http://modflowpy.github.io/flopydoc/tutorials.html)

### [Additional jupyter Notebook Examples](docs/notebook_examples.md)

### [Python Script Examples](docs/script_examples.md)


If You Get Stuck
-----------------------------------------------

FloPy usage has been growing rapidly, and as the number of users has increased, so has the number of questions about how to use FloPy.  We ask our users to carefully consider the nature of their problem and seek help in the appropriate manner.

### Questions

For questions related to how to do something with FloPy, we ask our users to submit the question to [Stack Overflow](https://stackoverflow.com) and assign the [flopy](https://stackoverflow.com/questions/tagged/flopy) tag.  Many of our recent questions have been related to MODFLOW or Python, and the Flopy developers cannot always respond to these inquiries.

### Bugs

If you think you have discovered a bug in FloPy in which you feel that the program does not work as intended, then we ask you to submit a [Github issue](https://github.com/modflowpy/flopy/labels/bug).


FloPy Supported Packages
-----------------------------------------------

A list of supported packages in FloPy is available in [docs/supported_packages.md](docs/supported_packages.md) on the github repo.


FloPy Model Checks
-----------------------------------------------

A table of the supported and proposed model checks implemented in  FloPy is available in [docs/model_checks.md](docs/model_checks.md) on the github repo.


FloPy Changes
-----------------------------------------------

A summary of changes in each FloPy version is available in [docs/version_changes.md](docs/version_changes.md) on the github repo.


Installation
-----------------------------------------------

**Python versions:**

FloPy requires **Python** 2.7 or **Python** 3.3 (or higher)


**Dependencies:**

FloPy requires **NumPy** 1.9 (or higher) and **enum34** for **Python** 2.7 or **Python** 3.3.


**For base and Anaconda Python distributions:**

To install FloPy type:

    pip install flopy

or

	conda install -c conda-forge flopy

To update FloPy type:

    pip install flopy --upgrade

or

	conda update -c conda-forge flopy

To uninstall FloPy type:

    pip uninstall flopy

or

	conda uninstall flopy


**Installing from the git repository:**

***Current Version of FloPy:***

To install the current version of FloPy from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master

To update your version of FloPy with the current version from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/master --upgrade

***Development version of FloPy:***

To install the latest development version of FloPy from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop

To update your version of FloPy with the latest development version from the git repository type:

    pip install https://github.com/modflowpy/flopy/zipball/develop --upgrade



***Optional Method Dependencies:***

Additional dependencies to use optional FloPy helper methods are listed below.

| Method                                                                               | Python Package                                     |
| ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `.PlotMapView()` in `flopy.plot`                                                     | **matplotlib** >= 1.4                              |
| `.PlotCrossSection()` in `flopy.plot`                                                | **matplotlib** >= 1.4                              |
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
| `.generate_classes()` in `flopy.mf6.utils`                                           | [**pymake**](https://github.com/modflowpy/pymake)  |
| `.intersect()` in `flopy.discretization.VertexGrid`                                  | **matplotlib** >= 1.4                              |


How to Cite
-----------------------------------------------

##### ***Citation for FloPy:***

[Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.](http://dx.doi.org/10.1111/gwat.12413)

##### ***Software/Code citation for FloPy:***

[Bakker, M., Post, V., Langevin, C.D., Hughes, J.D., White, J.T., Starn, J.J., and Fienen, M.N., 2019, FloPy v3.2.12: U.S. Geological Survey Software Release, 31 May 2019, http://dx.doi.org/10.5066/F7BK19FH](http://dx.doi.org/10.5066/F7BK19FH)


MODFLOW Resources
-----------------------------------------------

+ [MODFLOW and Related Programs](http://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html)
+ [Online guide for MODFLOW-2005](http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)


Disclaimer
----------

This software has been approved for release by the U.S. Geological Survey
(USGS). Although the software has been subjected to rigorous review, the USGS
reserves the right to update the software as needed pursuant to further analysis
and review. No warranty, expressed or implied, is made by the USGS or the U.S.
Government as to the functionality of the software and related material nor
shall the fact of release constitute any such warranty. Furthermore, the
software is released on condition that neither the USGS nor the U.S. Government
shall be held liable for any damages resulting from its authorized or
unauthorized use.

