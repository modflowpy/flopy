API Reference
=============

MODFLOW 6
---------

FloPy for MODFLOW 6 allows for the construction of multi-model simulations.
In order to construct a MODFLOW 6 simulation using FloPy, first construct a
simulation (MFSimulation) object.  Then construct the MODFLOW 6 models
(Modflowgwf and Modflowgwt) and the packages, like TDIS, that are associated
with the simulation.  Finally, construct the packages that are associated with
each of your models.


MODFLOW 6 Base Classes
^^^^^^^^^^^^^^^^^^^^^^^

FloPy for MODFLOW 6 is object oriented code that uses inheritance.  The FloPy classes
used to define different types models and packages share common code that is defined
in these base classes.


Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.mf*


MODFLOW 6 Simulation
^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 allows you to create simulations that can contain multiple models and
packages.  The FloPy for MODFLOW 6 simulation classes define functionality that applies
to the entire MODFLOW 6 simulation.  When using FloPy for MODFLOW 6 the first object
you will most likely create is a simulation (MFSimulation) object.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfsimulation
   :exclude-members: register_model

MODFLOW 6 Simulation Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 simulation packages are the packages that are not necessarily tied to a
specific model and can apply to the entire simulation or a group of models in the
simulation.

Contents:

.. toctree::
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfgnc
   ./source/flopy.mf6.modflow.mfims
   ./source/flopy.mf6.modflow.mfmvr
   ./source/flopy.mf6.modflow.mfnam
   ./source/flopy.mf6.modflow.mftdis


MODFLOW 6 Models
^^^^^^^^^^^^^^^^

MODFLOW 6 supports both groundwater flow (mfgwf.ModflowGwf) and groundwater
transport (mfgwt.ModflowGwt) models.  FloPy for MODFLOW 6 model objects can be
constructed after a FloPy simulation (MFSimulation) object has been constructed.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfgwf
   ./source/flopy.mf6.modflow.mfgwt


MODFLOW 6 Groundwater Flow Model Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 groundwater flow models support a number of required and optional packages.
Once a MODFLOW 6 groundwater flow model object (mfgwf.ModflowGwf) has been constructed
various packages associated with the groundwater flow model can be constructed.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfgwfa*
   ./source/flopy.mf6.modflow.mfgwfb*
   ./source/flopy.mf6.modflow.mfgwfc*
   ./source/flopy.mf6.modflow.mfgwfd*
   ./source/flopy.mf6.modflow.mfgwfe*
   ./source/flopy.mf6.modflow.mfgwff*
   ./source/flopy.mf6.modflow.mfgwfg*
   ./source/flopy.mf6.modflow.mfgwfh*
   ./source/flopy.mf6.modflow.mfgwfi*
   ./source/flopy.mf6.modflow.mfgwfj*
   ./source/flopy.mf6.modflow.mfgwfk*
   ./source/flopy.mf6.modflow.mfgwfl*
   ./source/flopy.mf6.modflow.mfgwfm*
   ./source/flopy.mf6.modflow.mfgwfn*
   ./source/flopy.mf6.modflow.mfgwfo*
   ./source/flopy.mf6.modflow.mfgwfp*
   ./source/flopy.mf6.modflow.mfgwfq*
   ./source/flopy.mf6.modflow.mfgwfr*
   ./source/flopy.mf6.modflow.mfgwfs*
   ./source/flopy.mf6.modflow.mfgwft*
   ./source/flopy.mf6.modflow.mfgwfu*
   ./source/flopy.mf6.modflow.mfgwfv*
   ./source/flopy.mf6.modflow.mfgwfw*
   ./source/flopy.mf6.modflow.mfgwfx*
   ./source/flopy.mf6.modflow.mfgwfy*
   ./source/flopy.mf6.modflow.mfgwfz*


MODFLOW 6 Groundwater Transport Model Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 groundwater transport models support a number of required and optional packages.
Once a MODFLOW 6 groundwater transport model object (mfgwt.ModflowGwt) has been constructed
various packages associated with the groundwater transport model can be constructed.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfgwta*
   ./source/flopy.mf6.modflow.mfgwtb*
   ./source/flopy.mf6.modflow.mfgwtc*
   ./source/flopy.mf6.modflow.mfgwtd*
   ./source/flopy.mf6.modflow.mfgwte*
   ./source/flopy.mf6.modflow.mfgwtf*
   ./source/flopy.mf6.modflow.mfgwtg*
   ./source/flopy.mf6.modflow.mfgwth*
   ./source/flopy.mf6.modflow.mfgwti*
   ./source/flopy.mf6.modflow.mfgwtj*
   ./source/flopy.mf6.modflow.mfgwtk*
   ./source/flopy.mf6.modflow.mfgwtl*
   ./source/flopy.mf6.modflow.mfgwtm*
   ./source/flopy.mf6.modflow.mfgwtn*
   ./source/flopy.mf6.modflow.mfgwto*
   ./source/flopy.mf6.modflow.mfgwtp*
   ./source/flopy.mf6.modflow.mfgwtq*
   ./source/flopy.mf6.modflow.mfgwtr*
   ./source/flopy.mf6.modflow.mfgwts*
   ./source/flopy.mf6.modflow.mfgwtt*
   ./source/flopy.mf6.modflow.mfgwtu*
   ./source/flopy.mf6.modflow.mfgwtv*
   ./source/flopy.mf6.modflow.mfgwtw*
   ./source/flopy.mf6.modflow.mfgwtx*
   ./source/flopy.mf6.modflow.mfgwty*
   ./source/flopy.mf6.modflow.mfgwtz*


MODFLOW 6 Utility Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 has several utility packages that can be associated with other packages.
This includes the obs package, which can be used to output model results specific
to its parent package, and the time series and time array series packages, which
can be used to provide time series input for other packages.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.modflow.mfutl*


MODFLOW 6 Utility Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 has a number of utilities useful for pre- or postprocessing.

Contents:

.. toctree::
   :maxdepth: 4

   ./source/flopy.mf6.utils.binaryfile_utils.rst
   ./source/flopy.mf6.utils.binarygrid_util.rst
   ./source/flopy.mf6.utils.mfobservation.rst
   ./source/flopy.mf6.utils.output_util.rst
   ./source/flopy.mf6.utils.postprocessing.rst
   ./source/flopy.mf6.utils.reference.rst
   ./source/flopy.mf6.utils.lakpak_utils.rst
   ./source/flopy.mf6.utils.model_splitter.rst


MODFLOW 6 Data
^^^^^^^^^^^^^^

FloPy for MODFLOW 6 data objects (MFDataArray, MFDataList, MFDataScalar) are
automatically constructed by FloPy when you construct a package.  These data
objects provide an interface for getting MODFLOW 6 data in different formats
and setting MODFLOW 6 data.

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mf6.data.mfdataarray
   ./source/flopy.mf6.data.mfdatalist
   ./source/flopy.mf6.data.mfdatascalar

Build MODFLOW 6 Classes
^^^^^^^^^^^^^^^^^^^^^^^

MODFLOW 6 FloPy classes can be rebuild from MODFLOW 6 definition files. This
will allow creation of MODFLOW 6 FloPy classes for development versions of
MODFLOW 6.

Contents:

.. toctree::
   :maxdepth: 4

   ./source/flopy.mf6.utils.createpackages.rst
   ./source/flopy.mf6.utils.generate_classes.rst


Previous Versions of MODFLOW
----------------------------

MODFLOW Base Classes
^^^^^^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :maxdepth: 4

   ./source/flopy.mbase
   ./source/flopy.pakbase


MODFLOW Packages
^^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.modflow.mf*


MT3DMS Packages
^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.mt3d.mt*


SEAWAT Packages
^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.seawat.swt*


MODPATH 7 Packages
^^^^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.modpath.mp7*


MODPATH 6 Packages
^^^^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.modpath.mp6*


Flopy Utilities
---------------

Model Utilities (including binary file readers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.utils.*


Plotting Utilities
^^^^^^^^^^^^^^^^^^
Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.plot.*


Export Utilities
^^^^^^^^^^^^^^^^
Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.export.*


PEST Utilities
^^^^^^^^^^^^^^
Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.pest.*


Discretization Utilities
^^^^^^^^^^^^^^^^^^^^^^^^
Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/flopy.discretization.*
