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
    - \fancyfoot[LE, RO]{FloPy version 3.2.9 &mdash; develop}
    - \fancyfoot[CO, CE]{\thepage\ of \pageref{LastPage}}
    - \fancyfoot[RE, LO]{10/08/2018}
geometry: margin=0.75in
---

Introduction
-----------------------------------------------

FloPy includes support for MODFLOW 6, MODFLOW-2005, MODFLOW-NWT, MODFLOW-USG, and MODFLOW-2000. Other supported MODFLOW-based models include MODPATH (version 6 and ***7 (beta)***), MT3DMS, MT3D-USGS, and SEAWAT.

For general modeling issues, please consult a modeling forum, such as the [MODFLOW Users Group](https://groups.google.com/forum/#!forum/modflow).  Other MODFLOW resources are listed in the [MODFLOW Resources](https://github.com/modflowpy/flopy#modflow-resources) section.


Contributing
------------------------------------------------

Contributions are welcome from the community. Questions can be asked on the [issues page](https://github.com/modflowpy/flopy/issues). Before creating a new issue, please take a  moment to search and make sure a similar issue does not already exist. If one does exist, you can comment (most simply even with just a `:+1:`) to show your support for that issue.

If you have direct contributions you would like considered for incorporation into the project you can [fork this repository](https://help.github.com/articles/fork-a-repo/) and [submit a pull request](https://help.github.com/articles/about-pull-requests/) for review.


Documentation
-----------------------------------------------

FloPy code documentation is available at [http://modflowpy.github.io/flopydoc/](http://modflowpy.github.io/flopydoc/)


How to Cite
-----------------------------------------------

##### ***Citation for FloPy:***

[Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.](http://dx.doi.org/10.1111/gwat.12413)

##### ***Software/Code citation for FloPy:***

[Bakker, M., Post, V., Langevin, C.D., Hughes, J.D., White, J.T., Starn, J.J., and Fienen, M.N., 2018, FloPy v3.2.9 &mdash; develop: U.S. Geological Survey Software Release, 08 October 2018, http://dx.doi.org/10.5066/F7BK19FH](http://dx.doi.org/10.5066/F7BK19FH)


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.

Installation
-----------------------------------------------
To install FloPy version 3.2.9 &mdash; develop from the USGS FloPy website:
```
pip install https://water.usgs.gov/ogw/flopy/flopy-3.2.9.zip
```

To update to FloPy version 3.2.9 &mdash; develop from the USGS FloPy website:
```
pip install https://water.usgs.gov/ogw/flopy/flopy-3.2.9.zip --upgrade
```
