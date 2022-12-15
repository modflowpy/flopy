# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlspc(mfpackage.MFPackage):
    """
    ModflowUtlspc defines a spc package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of spc
          information will be written to the listing file immediately after it
          is read.
    timeseries : {varname:data} or timeseries data
        * Contains data for the ts package. Data can be stored in a dictionary
          containing data for the ts package with variable names as keys and
          package data as values. Data just for the timeseries variable is also
          acceptable. See ts package documentation for more information.
    maxbound : integer
        * maxbound (integer) integer value specifying the maximum number of spc
          cells that will be specified for use during any stress period.
    perioddata : [bndno, spcsetting]
        * bndno (integer) integer value that defines the boundary package
          feature number associated with the specified PERIOD data on the line.
          BNDNO must be greater than zero and less than or equal to MAXBOUND.
          This argument is an index variable, which means that it should be
          treated as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * spcsetting (keystring) line of information that is parsed into a
          keyword and values. Keyword values that can be used to start the
          MAWSETTING string include: CONCENTRATION.
            concentration : [double]
                * concentration (double) is the boundary concentration. If the
                  Options block includes a TIMESERIESFILE entry (see the "Time-
                  Variable Input" section), values can be obtained from a time
                  series by entering the time-series name in place of a numeric
                  value. By default, the CONCENTRATION for each boundary
                  feature is zero.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    ts_filerecord = ListTemplateGenerator(("spc", "options", "ts_filerecord"))
    perioddata = ListTemplateGenerator(("spc", "period", "perioddata"))
    package_abbr = "utlspc"
    _package_type = "spc"
    dfn_file_name = "utl-spc.dfn"

    dfn = [
        [
            "header",
            "multi-package",
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
            "block dimensions",
            "name maxbound",
            "type integer",
            "reader urword",
            "optional false",
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
            "type recarray bndno spcsetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name bndno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block period",
            "name spcsetting",
            "type keystring concentration",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name concentration",
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
        model,
        loading_package=False,
        print_input=None,
        timeseries=None,
        maxbound=None,
        perioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "spc", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
