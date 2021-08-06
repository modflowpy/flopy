# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwtapi(mfpackage.MFPackage):
    """
    ModflowGwtapi defines a api package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of api boundary cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of api
          boundary information will be written to the listing file immediately
          after it is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of api
          boundary flow rates will be printed to the listing file for every
          stress period time step in which "BUDGET PRINT" is specified in
          Output Control. If there is no Output Control option and
          "PRINT_FLOWS" is specified, then flow rates are printed for the last
          time step of each stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that api boundary flow terms
          will be written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    mover : boolean
        * mover (boolean) keyword to indicate that this instance of the api
          boundary Package can be used with the Water Mover (MVR) Package. When
          the MOVER option is specified, additional memory is allocated within
          the package to store the available, provided, and received water.
    maxbound : integer
        * maxbound (integer) integer value specifying the maximum number of api
          boundary cells that will be specified for use during any stress
          period.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    obs_filerecord = ListTemplateGenerator(
        ("gwt6", "api", "options", "obs_filerecord")
    )
    package_abbr = "gwtapi"
    _package_type = "api"
    dfn_file_name = "gwt-api.dfn"

    dfn = [
        [
            "block options",
            "name boundnames",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
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
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name obs_filerecord",
            "type record obs6 filein obs6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package obs",
            "construct_data continuous",
            "parameter_name observations",
        ],
        [
            "block options",
            "name obs6",
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
            "name obs6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block options",
            "name mover",
            "type keyword",
            "tagged true",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name maxbound",
            "type integer",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        boundnames=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        observations=None,
        mover=None,
        maxbound=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "api", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.mover = self.build_mfdata("mover", mover)
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self._init_complete = True
