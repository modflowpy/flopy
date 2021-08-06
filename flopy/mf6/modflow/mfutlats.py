# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlats(mfpackage.MFPackage):
    """
    ModflowUtlats defines a ats package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    maxats : integer
        * maxats (integer) is the number of records in the subsequent
          perioddata block that will be used for adaptive time stepping.
    perioddata : [iperats, dt0, dtmin, dtmax, dtadj, dtfailadj]
        * iperats (integer) is the period number to designate for adaptive time
          stepping. The remaining ATS values on this line will apply to period
          iperats. iperats must be greater than zero. A warning is printed if
          iperats is greater than nper. This argument is an index variable,
          which means that it should be treated as zero-based when working with
          FloPy and Python. Flopy will automatically subtract one when loading
          index variables and add one when writing index variables.
        * dt0 (double) is the initial time step length for period iperats. If
          dt0 is zero, then the final step from the previous stress period will
          be used as the initial time step. The program will terminate with an
          error message if dt0 is negative.
        * dtmin (double) is the minimum time step length for this period. This
          value must be greater than zero and less than dtmax. dtmin must be a
          small value in order to ensure that simulation times end at the end
          of stress periods and the end of the simulation. A small value, such
          as 1.e-5, is recommended.
        * dtmax (double) is the maximum time step length for this period. This
          value must be greater than dtmin.
        * dtadj (double) is the time step multiplier factor for this period. If
          the number of outer solver iterations are less than the product of
          the maximum number of outer iterations (OUTER_MAXIMUM) and
          ATS_OUTER_MAXIMUM_FRACTION (an optional variable in the IMS input
          file with a default value of 1/3), then the time step length is
          multipled by dtadj. If the number of outer solver iterations are
          greater than the product of the maximum number of outer iterations
          and ATS_OUTER_MAXIMUM_FRACTION, then the time step length is divided
          by dtadj. dtadj must be zero, one, or greater than one. If dtadj is
          zero or one, then it has no effect on the simulation. A value between
          2.0 and 5.0 can be used as an initial estimate.
        * dtfailadj (double) is the divisor of the time step length when a time
          step fails to converge. If there is solver failure, then the time
          step will be tried again with a shorter time step length calculated
          as the previous time step length divided by dtfailadj. dtfailadj must
          be zero, one, or greater than one. If dtfailadj is zero or one, then
          time steps will not be retried with shorter lengths. In this case,
          the program will terminate with an error, or it will continue of the
          CONTINUE option is set in the simulation name file. Initial tests
          with this variable should be set to 5.0 or larger to determine if
          convergence can be achieved.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    perioddata = ListTemplateGenerator(("ats", "perioddata", "perioddata"))
    package_abbr = "utlats"
    _package_type = "ats"
    dfn_file_name = "utl-ats.dfn"

    dfn = [
        [
            "block dimensions",
            "name maxats",
            "type integer",
            "reader urword",
            "optional false",
            "default_value 1",
        ],
        [
            "block perioddata",
            "name perioddata",
            "type recarray iperats dt0 dtmin dtmax dtadj dtfailadj",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name iperats",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
        [
            "block perioddata",
            "name dt0",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name dtmin",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name dtmax",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name dtadj",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name dtfailadj",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        maxats=1,
        perioddata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "ats", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.maxats = self.build_mfdata("maxats", maxats)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True


class UtlatsPackages(mfpackage.MFChildPackages):
    """
    UtlatsPackages is a container class for the ModflowUtlats class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtlats package removing any sibling child
        packages attached to the same parent package. See ModflowUtlats init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtlats package to the container. See ModflowUtlats
        init documentation for definition of parameters.
    """

    package_abbr = "utlatspackages"

    def initialize(self, maxats=1, perioddata=None, filename=None, pname=None):
        new_package = ModflowUtlats(
            self._model,
            maxats=maxats,
            perioddata=perioddata,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._init_package(new_package, filename)

    def append_package(
        self, maxats=1, perioddata=None, filename=None, pname=None
    ):
        new_package = ModflowUtlats(
            self._model,
            maxats=maxats,
            perioddata=perioddata,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._append_package(new_package, filename)
