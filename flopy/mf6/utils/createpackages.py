"""
createpackages.py is a utility script that reads in the file definition
metadata in the .dfn files and creates the package classes in the modflow
folder. Run this script any time changes are made to the .dfn files.

To create a new package that is part of an existing model, first create a new
dfn file for the package in the mf6/data/dfn folder.
1) Follow the file naming convention <model abbr>-<package abbr>.dfn.
2) Run this script (createpackages.py), and check in your new dfn file, and
   the package class and updated __init__.py that createpackages.py created.

A subpackage is a package referenced by another package (vs being referenced
in the name file).  The tas, ts, and obs packages are examples of subpackages.
There are a few additional steps required when creating a subpackage
definition file.  First, verify that the parent package's dfn file has a file
record for the subpackage to the option block.   For example, for the time
series package the file record definition starts with:

    block options
    name ts_filerecord
    type record ts6 filein ts6_filename

Verify that the same naming convention is followed as the example above,
specifically:

    name <subpackage-abbr>_filerecord
    record <subpackage-abbr>6 filein <subpackage-abbr>6_filename

Next, create the child package definition file in the mf6/data/dfn folder
following the naming convention above.

When your child package is ready for release follow the same procedure as
other packages along with these a few additional steps required for
subpackages.

At the top of the child dfn file add two lines describing how the parent and
child packages are related. The first line determines how the subpackage is
linked to the package:

# flopy subpackage <parent record> <abbreviation> <child data>
<data name>

* Parent record is the MF6 record name of the filerecord in parent package
  that references the child packages file name
* Abbreviation is the short abbreviation of the new subclass
* Child data is the name of the child class data that can be passed in as
  parameter to the parent class. Passing in this parameter to the parent class
  automatically creates the child class with the data provided.
* Data name is the parent class parameter name that automatically creates the
  child class with the data provided.

The example below is the first line from the ts subpackage dfn:

# flopy subpackage ts_filerecord ts timeseries timeseries

The second line determines the variable name of the subpackage's parent and
the type of parent (the parent package's object oriented parent):

# flopy parent_name_type <parent package variable name>
<parent package type>

An example below is the second line in the ts subpackage dfn:

# flopy parent_name_type parent_package MFPackage

There are three possible types (or combination of them) that can be used for
"parent package type", MFPackage, MFModel, and MFSimulation. If a package
supports multiple types of parents (for example, it can be either in the model
namefile or in a package, like the obs package), include all the types
supported, separating each type with a / (MFPackage/MFModel).

To create a new type of model choose a unique three letter model abbreviation
("gwf", "gwt", ...). Create a name file dfn with the naming convention
<model abbr>-nam.dfn. The name file must have only an options and packages
block (see gwf-nam.dfn as an example). Create a new dfn file for each of the
packages in your new model, following the naming convention described above.

When your model is ready for release make sure all the dfn files are in the
flopy/mf6/data/dfn folder, run createpackages.py, and check in your new dfn
files, the package classes, and updated init.py that createpackages.py created.

"""

from pathlib import Path

_MF6_PATH = Path(__file__).parents[1]


if __name__ == "__main__":
    from flopy.mf6.utils.codegen import make_all
    make_all(dfndir=_MF6_PATH / "data" / "dfn", outdir=_MF6_PATH / "modflow")
