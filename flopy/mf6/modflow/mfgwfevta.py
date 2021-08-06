# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfevta(mfpackage.MFPackage):
    """
    ModflowGwfevta defines a evta package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    readasarrays : boolean
        * readasarrays (boolean) indicates that array-based input will be used
          for the Evapotranspiration Package. This keyword must be specified to
          use array-based input.
    fixed_cell : boolean
        * fixed_cell (boolean) indicates that evapotranspiration will not be
          reassigned to a cell underlying the cell specified in the list if the
          specified cell is inactive.
    auxiliary : [string]
        * auxiliary (string) defines an array of one or more auxiliary variable
          names. There is no limit on the number of auxiliary variables that
          can be provided on this line; however, lists of information provided
          in subsequent blocks must have a column of data for each auxiliary
          variable name defined here. The number of auxiliary variables
          detected on this line determines the value for naux. Comments cannot
          be provided anywhere on this line as they will be interpreted as
          auxiliary variable names. Auxiliary variables may not be used by the
          package, but they will be available for use by other parts of the
          program. The program will terminate with an error if auxiliary
          variables are specified on more than one line in the options block.
    auxmultname : string
        * auxmultname (string) name of auxiliary variable to be used as
          multiplier of evapotranspiration rate.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of
          evapotranspiration information will be written to the listing file
          immediately after it is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of
          evapotranspiration flow rates will be printed to the listing file for
          every stress period time step in which "BUDGET PRINT" is specified in
          Output Control. If there is no Output Control option and
          "PRINT_FLOWS" is specified, then flow rates are printed for the last
          time step of each stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that evapotranspiration flow
          terms will be written to the file specified with "BUDGET FILEOUT" in
          Output Control.
    timearrayseries : {varname:data} or tas_array data
        * Contains data for the tas package. Data can be stored in a dictionary
          containing data for the tas package with variable names as keys and
          package data as values. Data just for the timearrayseries variable is
          also acceptable. See tas package documentation for more information.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    ievt : [integer]
        * ievt (integer) IEVT is the layer number that defines the layer in
          each vertical column where evapotranspiration is applied. If IEVT is
          omitted, evapotranspiration by default is applied to cells in layer
          1. If IEVT is specified, it must be specified as the first variable
          in the PERIOD block or MODFLOW will terminate with an error. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
    surface : [double]
        * surface (double) is the elevation of the ET surface (:math:`L`).
    rate : [double]
        * rate (double) is the maximum ET flux rate (:math:`LT^{-1}`).
    depth : [double]
        * depth (double) is the ET extinction depth (:math:`L`).
    aux(iaux) : [double]
        * aux(iaux) (double) is an array of values for auxiliary variable
          AUX(IAUX), where iaux is a value from 1 to NAUX, and AUX(IAUX) must
          be listed as part of the auxiliary variables. A separate array can be
          specified for each auxiliary variable. If an array is not specified
          for an auxiliary variable, then a value of zero is assigned. If the
          value specified here for the auxiliary variable is the same as
          auxmultname, then the evapotranspiration rate will be multiplied by
          this array.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(("gwf6", "evta", "options", "auxiliary"))
    tas_filerecord = ListTemplateGenerator(
        ("gwf6", "evta", "options", "tas_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwf6", "evta", "options", "obs_filerecord")
    )
    ievt = ArrayTemplateGenerator(("gwf6", "evta", "period", "ievt"))
    surface = ArrayTemplateGenerator(("gwf6", "evta", "period", "surface"))
    rate = ArrayTemplateGenerator(("gwf6", "evta", "period", "rate"))
    depth = ArrayTemplateGenerator(("gwf6", "evta", "period", "depth"))
    aux = ArrayTemplateGenerator(("gwf6", "evta", "period", "aux(iaux)"))
    package_abbr = "gwfevta"
    _package_type = "evta"
    dfn_file_name = "gwf-evta.dfn"

    dfn = [
        [
            "block options",
            "name readasarrays",
            "type keyword",
            "shape",
            "reader urword",
            "optional false",
            "default_value True",
        ],
        [
            "block options",
            "name fixed_cell",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name auxiliary",
            "type string",
            "shape (naux)",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name auxmultname",
            "type string",
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
            "name tas_filerecord",
            "type record tas6 filein tas6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package tas",
            "construct_data tas_array",
            "parameter_name timearrayseries",
        ],
        [
            "block options",
            "name tas6",
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
            "name tas6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
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
            "name obs6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
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
            "name ievt",
            "type integer",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
            "numeric_index true",
            "optional true",
        ],
        [
            "block period",
            "name surface",
            "type double precision",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
            "default_value 0.",
        ],
        [
            "block period",
            "name rate",
            "type double precision",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
            "default_value 1.e-3",
        ],
        [
            "block period",
            "name depth",
            "type double precision",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
            "default_value 1.0",
        ],
        [
            "block period",
            "name aux(iaux)",
            "type double precision",
            "shape (ncol*nrow; ncpl)",
            "reader readarray",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        readasarrays=True,
        fixed_cell=None,
        auxiliary=None,
        auxmultname=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        timearrayseries=None,
        observations=None,
        ievt=None,
        surface=0.0,
        rate=1.0e-3,
        depth=1.0,
        aux=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "evta", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.readasarrays = self.build_mfdata("readasarrays", readasarrays)
        self.fixed_cell = self.build_mfdata("fixed_cell", fixed_cell)
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.auxmultname = self.build_mfdata("auxmultname", auxmultname)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self._tas_filerecord = self.build_mfdata("tas_filerecord", None)
        self._tas_package = self.build_child_package(
            "tas", timearrayseries, "tas_array", self._tas_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.ievt = self.build_mfdata("ievt", ievt)
        self.surface = self.build_mfdata("surface", surface)
        self.rate = self.build_mfdata("rate", rate)
        self.depth = self.build_mfdata("depth", depth)
        self.aux = self.build_mfdata("aux(iaux)", aux)
        self._init_complete = True
