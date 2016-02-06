---
title: FloPy Release Notes
author: 
    - Mark Bakker
    - Vincent Post
    - Christian D. Langevin
    - Joseph D. Hughes
    - Jeremy T. White
    - Jeffery Starn
    - Michael N. Fienen
header-includes:
    - \usepackage{fancyhdr}
    - \usepackage{lastpage}
    - \pagestyle{fancy}
    - \fancyhf{}
    - \fancyhead[LE,LO,RE,RO]{}
    - \fancyhead[CE,CO]{Flopy Release Notes}
    - \fancyfoot[LE,RO]{FloPy version 3.2.4}
    - \fancyfoot[CO,CE]{\thepage\ of \pageref{LastPage}}
    - \fancyfoot[RE,LO]{2/6/2015}
geometry: margin=0.75in
---

Introduction
-----------------------------------------------

FloPy is a Python package for developing, running, and post-processing models that are part of the MODFLOW family of codes. FloPy includes support for MODFLOW-2000, MODFLOW-2005, MODFLOW-NWT, and MODFLOW-USG. Other supported MODFLOW-based models include MODPATH (version 6), MT3DMS and SEAWAT.

If you think you have found a bug in FloPy, or if you would like to suggest an improvement or enhancement, please contact one of the points of contact identified on [http://water.usgs.gov/ogw/modflow/flopy.html](http://water.usgs.gov/ogw/modflow/flopy.html). Alternatively submit a new Issue through the [Github Issue tracker](https://github.com/modflowpy/flopy/tree/develop). Pull requests will only be accepted on the develop branch of the repository.


Documentation
-----------------------------------------------

FloPy code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


Installation
-----------------------------------------------

**Python versions:**

FloPy requires **Python** 2.7 or **Python** 3.3 (or higher)


**Dependencies:**

FloPy requires **NumPy** 1.9 (or higher) and **matplotlib** 1.4 (or higher). The mapping and cross-section capabilities in the `flopy.plot` submodule and shapefile export capabilities (`to_shapefile()`) require **Pyshp** 1.2 (or higher). The NetCDF export capabilities in the `flopy.export` submodule require **python-dateutil** 2.4 (or higher), **netcdf4** 1.1 (or higher), and **pyproj** 1.9 (or higher). Other NetCDF dependencies are detailed on the [UniData](http://unidata.github.io/netcdf4-python/) website. The `get_dataframes` method in the `ListBudget` class in the `flopy.utils` submodule require **pandas** 0.15 (or higher).


**Installation:**

To install FloPy version 3.2.4 from the USGS flopy website:

    pip install http://water.usgs.gov/ogw/modflow/FloPy_v3.2.4/flopy-3.2.4.zip
    
To update FloPy version 3.2.4 from the USGS flopy website:

    pip install http://water.usgs.gov/ogw/modflow/FloPy_v3.2.4/flopy-3.2.4.zip --upgrade


