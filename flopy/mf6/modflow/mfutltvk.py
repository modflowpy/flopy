# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtltvk(mfpackage.MFPackage):
    """
    ModflowUtltvk defines a tvk package within a utl model.

    Parameters
    ----------
    parent_package : MFPackage
        Parent_package that this package is a part of. Package is automatically
        added to parent_package when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that information for each
          change to the hydraulic conductivity in a cell will be written to the
          model listing file.
    timeseries : {varname:data} or timeseries data
        * Contains data for the ts package. Data can be stored in a dictionary
          containing data for the ts package with variable names as keys and
          package data as values. Data just for the timeseries variable is also
          acceptable. See ts package documentation for more information.
    perioddata : [cellid, tvksetting]
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * tvksetting (keystring) line of information that is parsed into a
          property name keyword and values. Property name keywords that can be
          used to start the TVKSETTING string include: K, K22, and K33.
            k : [double]
                * k (double) is the new value to be assigned as the cell's
                  hydraulic conductivity from the start of the specified stress
                  period, as per K in the NPF package. If the OPTIONS block
                  includes a TS6 entry (see the "Time-Variable Input" section),
                  values can be obtained from a time series by entering the
                  time-series name in place of a numeric value.
            k22 : [double]
                * k22 (double) is the new value to be assigned as the cell's
                  hydraulic conductivity of the second ellipsoid axis (or the
                  ratio of K22/K if the K22OVERK NPF package option is
                  specified) from the start of the specified stress period, as
                  per K22 in the NPF package. For an unrotated case this is the
                  hydraulic conductivity in the y direction. If the OPTIONS
                  block includes a TS6 entry (see the "Time-Variable Input"
                  section), values can be obtained from a time series by
                  entering the time-series name in place of a numeric value.
            k33 : [double]
                * k33 (double) is the new value to be assigned as the cell's
                  hydraulic conductivity of the third ellipsoid axis (or the
                  ratio of K33/K if the K33OVERK NPF package option is
                  specified) from the start of the specified stress period, as
                  per K33 in the NPF package. For an unrotated case, this is
                  the vertical hydraulic conductivity. If the OPTIONS block
                  includes a TS6 entry (see the "Time-Variable Input" section),
                  values can be obtained from a time series by entering the
                  time-series name in place of a numeric value.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    ts_filerecord = ListTemplateGenerator(("tvk", "options", "ts_filerecord"))
    perioddata = ListTemplateGenerator(("tvk", "period", "perioddata"))
    package_abbr = "utltvk"
    _package_type = "tvk"
    dfn_file_name = "utl-tvk.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name ts_filerecord",
            "type record ts6 filein ts6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package ts",
            "construct_data timeseries",
            "parameter_name timeseries",
        ],
        [
            "block options",
            "name ts6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name ts6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block period",
            "name iper",
            "type integer",
            "block_variable True",
            "in_record true",
            "tagged false",
            "shape",
            "valid",
            "reader urword",
            "optional false",
        ],
        [
            "block period",
            "name perioddata",
            "type recarray cellid tvksetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name cellid",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name tvksetting",
            "type keystring k k22 k33",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name k",
            "type double precision",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name k22",
            "type double precision",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name k33",
            "type double precision",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
    ]

    def __init__(
        self,
        parent_package,
        loading_package=False,
        print_input=None,
        timeseries=None,
        perioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            parent_package, "tvk", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True


class UtltvkPackages(mfpackage.MFChildPackages):
    """
    UtltvkPackages is a container class for the ModflowUtltvk class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtltvk package removing any sibling child
        packages attached to the same parent package. See ModflowUtltvk init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtltvk package to the container. See ModflowUtltvk
        init documentation for definition of parameters.
    """

    package_abbr = "utltvkpackages"

    def initialize(
        self,
        print_input=None,
        timeseries=None,
        perioddata=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtltvk(
            self._cpparent,
            print_input=print_input,
            timeseries=timeseries,
            perioddata=perioddata,
            filename=filename,
            pname=pname,
            child_builder_call=True,
        )
        self.init_package(new_package, filename)

    def append_package(
        self,
        print_input=None,
        timeseries=None,
        perioddata=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowUtltvk(
            self._cpparent,
            print_input=print_input,
            timeseries=timeseries,
            perioddata=perioddata,
            filename=filename,
            pname=pname,
            child_builder_call=True,
        )
        self._append_package(new_package, filename)
