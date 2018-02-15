---
title: FloPy Release Notes
author:
    - Mark Bakker
    - Vincent Post
    - Christian D. Langevin
    - Joseph D. Hughes
    - Jeremy T. White
    - Andrew T. Leaf
    - Scott R. Paulinski
    - Jeffrey Starn
    - Michael N. Fienen
header-includes:
    - \usepackage{fancyhdr}
    - \usepackage{lastpage}
    - \pagestyle{fancy}
    - \fancyhf{{}}
    - \fancyhead[LE, LO, RE, RO]{}
    - \fancyhead[CE, CO]{FloPy Release Notes}
    - \fancyfoot[LE, RO]{FloPy version 3.2.8 &mdash; develop}
    - \fancyfoot[CO, CE]{\thepage\ of \pageref{LastPage}}
    - \fancyfoot[RE, LO]{02/15/2018}
geometry: margin=0.75in
---

Introduction
-----------------------------------------------

FloPy includes support for MODFLOW-2000, MODFLOW-2005, MODFLOW-NWT, and MODFLOW-USG. Other supported MODFLOW-based models include MODPATH (version 6), MT3DMS, MT3D-USGS,  and SEAWAT.

FloPy now includes beta support for MODFLOW 6.  

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.

If you think you have found a bug in FloPy, or if you would like to suggest an improvement or enhancement, please submit a new issue through the [Github Issue tracker](https://github.com/modflowpy/flopy/issues).


Documentation
-----------------------------------------------

FloPy code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


How to Cite
-----------------------------------------------

##### ***Citation for FloPy:***

[Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.](http://dx.doi.org/10.1111/gwat.12413)

##### ***Software/Code citation for FloPy:***

[Bakker, M., Post, V., Langevin, C.D., Hughes, J.D., White, J.T., Starn, J.J., and Fienen, M.N., 2018, FloPy v3.2.8 &mdash; develop: U.S. Geological Survey Software Release, 15 February 2018, http://dx.doi.org/10.5066/F7BK19FH](http://dx.doi.org/10.5066/F7BK19FH)


Installation
-----------------------------------------------
To install FloPy version 3.2.8 &mdash; develop from the USGS FloPy website:
```
pip install https://water.usgs.gov/ogw/flopy/flopy-3.2.8.zip
```

To update to FloPy version 3.2.8 &mdash; develop from the USGS FloPy website:
```
pip install https://water.usgs.gov/ogw/flopy/flopy-3.2.8.zip --upgrade
```
