# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2024 14:36:23 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowSwfgwf(mfpackage.MFPackage):
    """
    ModflowSwfgwf defines a swfgwf package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    exgtype : <string>
        * is the exchange type (GWF-GWF or GWF-GWT).
    exgmnamea : <string>
        * is the name of the first model that is part of this exchange.
    exgmnameb : <string>
        * is the name of the second model that is part of this exchange.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of exchange
          entries will be echoed to the listing file immediately after it is
          read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of exchange
          flow rates will be printed to the listing file for every stress
          period in which "SAVE BUDGET" is specified in Output Control.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    nexg : integer
        * nexg (integer) keyword and integer value specifying the number of
          SWF-GWF exchanges.
    exchangedata : [cellidm1, cellidm2, cond]
        * cellidm1 ((integer, ...)) is the cellid of the cell in model 1 as
          specified in the simulation name file. For a structured grid that
          uses the DIS input file, CELLIDM1 is the layer, row, and column
          numbers of the cell. For a grid that uses the DISV input file,
          CELLIDM1 is the layer number and CELL2D number for the two cells. If
          the model uses the unstructured discretization (DISU) input file,
          then CELLIDM1 is the node number for the cell. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * cellidm2 ((integer, ...)) is the cellid of the cell in model 2 as
          specified in the simulation name file. For a structured grid that
          uses the DIS input file, CELLIDM2 is the layer, row, and column
          numbers of the cell. For a grid that uses the DISV input file,
          CELLIDM2 is the layer number and CELL2D number for the two cells. If
          the model uses the unstructured discretization (DISU) input file,
          then CELLIDM2 is the node number for the cell. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * cond (double) is the conductance between the surface water cell and
          the groundwater cell.
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
        ("swfgwf", "options", "obs_filerecord")
    )
    exchangedata = ListTemplateGenerator(
        ("swfgwf", "exchangedata", "exchangedata")
    )
    package_abbr = "swfgwf"
    _package_type = "swfgwf"
    dfn_file_name = "exg-swfgwf.dfn"

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
            "mf6internal ipr_input",
        ],
        [
            "block options",
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal ipr_flow",
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
            "block dimensions",
            "name nexg",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name exchangedata",
            "type recarray cellidm1 cellidm2 cond",
            "shape (nexg)",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name cellidm1",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
        [
            "block exchangedata",
            "name cellidm2",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
        [
            "block exchangedata",
            "name cond",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        exgtype="SWF6-GWF6",
        exgmnamea=None,
        exgmnameb=None,
        print_input=None,
        print_flows=None,
        observations=None,
        nexg=None,
        exchangedata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            simulation, "swfgwf", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.nexg = self.build_mfdata("nexg", nexg)
        self.exchangedata = self.build_mfdata("exchangedata", exchangedata)
        self._init_complete = True
