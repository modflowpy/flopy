
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.5.0.dev0 (preliminary)
[![flopy continuous integration](https://github.com/modflowpy/flopy/actions/workflows/commit.yml/badge.svg?branch=develop)](https://github.com/modflowpy/flopy/actions/workflows/commit.yml)
[![Read the Docs](https://github.com/modflowpy/flopy/actions/workflows/rtd.yml/badge.svg?branch=develop)](https://github.com/modflowpy/flopy/actions/workflows/rtd.yml)

[![codecov](https://codecov.io/gh/modflowpy/flopy/branch/develop/graph/badge.svg)](https://codecov.io/gh/modflowpy/flopy)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3f44f457aa474a8f83ad60c1842f7be2)](https://www.codacy.com/gh/modflowpy/flopy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=modflowpy/flopy&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/flopy/badge/?version=latest)](https://flopy.readthedocs.io/en/latest/?badge=latest)

[![Anaconda Version](https://anaconda.org/conda-forge/flopy/badges/version.svg)](https://anaconda.org/conda-forge/flopy)
[![Anaconda Updated](https://anaconda.org/conda-forge/flopy/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/flopy)
[![Anaconda Platforms](https://anaconda.org/conda-forge/flopy/badges/platforms.svg)](https://anaconda.org/conda-forge/flopy)

[![PyPI Version](https://img.shields.io/pypi/v/flopy.png)](https://pypi.python.org/pypi/flopy)
[![PyPI Status](https://img.shields.io/pypi/status/flopy.png)](https://pypi.python.org/pypi/flopy)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/flopy.png)](https://pypi.python.org/pypi/flopy)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/modflowpy/flopy.git/develop)

Introduction
-----------------------------------------------

FloPy includes support for [MODFLOW 6](docs/mf6.md), MODFLOW-2005, MODFLOW-NWT, MODFLOW-USG, and MODFLOW-2000. Other supported MODFLOW-based models include MODPATH (version 6 and 7), MT3DMS, MT3D-USGS, and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.

Documentation
-----------------------------------------------
* [Latest release](https://flopy.readthedocs.io)
* [Current release candidate](https://flopy.readthedocs.io/en/latest/)

Installation
-----------------------------------------------

FloPy requires **Python** 3.8 (or higher), **NumPy** 1.15.0 (or higher), and **matplotlib** 1.4.0 (or higher).  Dependencies for optional FloPy methods are summarized [here](docs/flopy_method_dependencies.md).

To install FloPy type:

    conda install -c conda-forge flopy

or

    pip install flopy


The release candidate version can also be installed from the git repository using the instructions provided [below](#relcand).

After FloPy is installed, MODFLOW and related programs can be installed using the command:

    get-modflow :flopy

See documentation [get_modflow.md](https://github.com/modflowpy/flopy/blob/develop/docs/get_modflow.md) for more information.


Getting Started
-----------------------------------------------

### MODFLOW 6 Quick Start

```python
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

head = gwf.output.head().get_data()
bud = gwf.output.budget()

spdis = bud.get_data(text='DATA-SPDIS')[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
pmv = flopy.plot.PlotMapView(gwf)
pmv.plot_array(head)
pmv.plot_grid(colors='white')
pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.)
pmv.plot_vector(qx, qy, normalize=True, color="white")
```
<img src="examples/images/quickstart.png" alt="plot" style="width:30;height:30">


Additional FloPy Resources
------------------------------------------------

- [FloPy tutorials](https://flopy.readthedocs.io/en/stable/tutorials.html) and scripts demonstrating the use of FloPy to run and post-process MODFLOW-based models.

- [FloPy example notebooks](https://flopy.readthedocs.io/en/stable/notebooks.html) demonstrating the use of FloPy pre- and post-processing capabilities with a variety of MODFLOW-based models.

- [MODFLOW 6 example problems](https://modflow6-examples.readthedocs.io/en/latest/) demonstrating FloPy use to create, run, and post-process MODFLOW 6 models.

- A list of supported packages in FloPy is available in [docs/supported_packages.md](docs/supported_packages.md) on the github repo.

- A table of the supported and proposed model checks implemented in  FloPy is available in [docs/model_checks.md](docs/model_checks.md) on the github repo.

- A summary of changes in each FloPy version is available in [docs/version_changes.md](docs/version_changes.md) on the github repo.

Questions
------------------------------------------------
FloPy usage has been growing rapidly, and as the number of users has increased, so has the number of questions about how to use FloPy.  We ask our users to carefully consider the nature of their problem and seek help in the appropriate manner.

Do not open issues for general support questions.  We want to keep GitHub issues for bug reports and feature requests. General support questions are better answered in the [Discussions](https://github.com/modflowpy/flopy/discussions) on the FloPy GitHub repository. [Stack Overflow](https://stackoverflow.com/questions/tagged/flopy) and the [MODFLOW google group](https://groups.google.com/forum/#!forum/modflow) are other options but currently neither of these are as active as Discussions on the FloPy GitHub repository. If using Stack Overflow, questions should be tagged with tag `flopy`.

To save your and our time, **we will systematically close all issues that are requests for general support and redirect people to Stack Overflow or the MODFLOW google group**.


Contributing
------------------------------------------------

Bug reports, code contributions, or improvements to the documentation are welcome from the community. See the [developer docs](DEVELOPER.md) to begin. Please also read up on our guidelines for [contributing](CONTRIBUTING.md) and check out our issues in the [hotlist: community-help](https://github.com/modflowpy/flopy/labels/hotlist%3A%20community%20help).


<a name="relcand"></a>Installing the latest FloPy release candidate
------------------------------------------------

To install the latest release candidate type:

    pip install https://github.com/modflowpy/flopy/zipball/develop


How to Cite
-----------------------------------------------

##### ***Citation for FloPy:***

[Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.](https://doi.org/10.1111/gwat.12413)

##### ***Software/Code citation for FloPy:***

[Bakker, Mark, Post, Vincent, Hughes, J. D., Langevin, C. D., White, J. T., Leaf, A. T., Paulinski, S. R., Bellino, J. C., Morway, E. D., Toews, M. W., Larsen, J. D., Fienen, M. N., Starn, J. J., Brakenhoff, D. A., and Bonelli, W. P., 2023, FloPy v3.5.0.dev0 (preliminary): U.S. Geological Survey Software Release, 13 July 2023, https://doi.org/10.5066/F7BK19FH](https://doi.org/10.5066/F7BK19FH)


Additional FloPy Related Publications
-----------------------------------------------

[Leaf A.T, and Fienen M. N., 2022, Flopy&mdash;The Python Interface for MODFLOW: Groundwater, v. 60, no. 6, p. 710-712. doi:10.1111/gwat.13259.](https://doi.org/10.1111/gwat.13259)

[Leaf, A.T. and M.N. Fienen, 2022, Modflow-setup&mdash;Robust automation of groundwater model construction: Frontiers in Earth Science, v. 10, 903965, doi:10.3389/feart.2022.903965.](https://doi.org/10.3389/feart.2022.903965)

[Leaf, A.T., Fienen, M.N. and Reeves, H.W., 2021, SFRmaker and Linesink-Maker&mdash;Rapid Construction of Streamflow Routing Networks from Hydrography Data: Groundwater, v. 59, no. 5, p. 761-771. doi:10.1111/gwat.13095.](https://doi.org/10.1111/gwat.13095)


MODFLOW Resources
-----------------------------------------------

+ [MODFLOW and Related Programs](https://water.usgs.gov/ogw/modflow/)
+ [Online guide for MODFLOW-2000](https://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/)
+ [Online guide for MODFLOW-2005](https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/)
+ [Online guide for MODFLOW-NWT](https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/)


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is 
being provided to meet the need for timely best science. This software is 
provided "as is" and "as-available", and makes no representations or warranties 
of any kind concerning the software, whether express, implied, statutory, or 
other. This includes, without limitation, warranties of title, 
merchantability, fitness for a particular purpose, non-infringement, absence 
of latent or other defects, accuracy, or the presence or absence of errors, 
whether or not known or discoverable.

