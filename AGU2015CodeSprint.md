
<img src="https://raw.githubusercontent.com/modflowpy/flopy/master/examples/images/flopy3.png" alt="flopy3" style="width:50;height:20">

### Version 3.2.3
[![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=develop)](https://travis-ci.org/modflowpy/flopy)

##AGU 2015 coding sprint focus
###Let's list items here and we can sort later
- refactor namespaces so that `flopy.modflow.ModflowWel` becomes `flopy.ModflowWel` etc
- develop a standard SpatialReference file that `flopy` can write and read
- unstructured grid support
    - Create new mfusg model?  Or work it into modflow?
    - Write disu flopy package (done; improve with better u3d support)
    - Allow u3d to have u2d instances of different sizes
    - In get_file_entry() pass in iac so that array output looks nicer
    - Have unstructured spatial reference constructed from grid spec file.  Make sure it has same methods as structured spatial reference so all plotting still works.
    - Binary and ASCII readers will need work.
- refactor `util_2d` to inherit from `numpy.ndarray`
- refactor all of the non-PEP8 compliant util class names (e.g. `util_2d` to `Util2d`). Even Jeremy's preaching PEP8!
- move shapefile support into `export`
- standardize an observation module, revisit the current template file writer, and move toward a m.calibrate() method that writes the files needed for PEST
- introduce the concept of a hydraulic feature (river, well, etc.) that has spatial attributes and time-series data.  Build an infrastructure to support construction of packages from these features.  Support plotting of features also.
- remove package from mbase and put into its own module
- Change array output style to be managed at array level instead of at model level.
- Modify the way unit numbers for head, ddn, and cbc are specified. Develop a method for creating namefile entries that can accept package budget information going to separate files.
- Modify run_model to return a tuple.  Use psutil to calculate the memory usage, and return that as well (if requested), or perhaps make it more general to get at anything that psutil can get at.
- Within the flopy code base, we should always use relative imports and NOT use 'import flopy.utils' for example.  Importing from flopy can cause problems with autotesting and other cases where there may be multple flopy installs.
- utility to convert mf88 and mf96 (really, really old) models into mf2005 model - what can be converted...
