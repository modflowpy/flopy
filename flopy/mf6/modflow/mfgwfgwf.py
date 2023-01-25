# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfgwf(mfpackage.MFPackage):
    """
    ModflowGwfgwf defines a gwfgwf package.

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
    auxiliary : [string]
        * auxiliary (string) an array of auxiliary variable names. There is no
          limit on the number of auxiliary variables that can be provided. Most
          auxiliary variables will not be used by the GWF-GWF Exchange, but
          they will be available for use by other parts of the program. If an
          auxiliary variable with the name "ANGLDEGX" is found, then this
          information will be used as the angle (provided in degrees) between
          the connection face normal and the x axis, where a value of zero
          indicates that a normal vector points directly along the positive x
          axis. The connection face normal is a normal vector on the cell face
          shared between the cell in model 1 and the cell in model 2 pointing
          away from the model 1 cell. Additional information on "ANGLDEGX" is
          provided in the description of the DISU Package. If an auxiliary
          variable with the name "CDIST" is found, then this information will
          be used as the straight-line connection distance, including the
          vertical component, between the two cell centers. Both ANGLDEGX and
          CDIST are required if specific discharge is calculated for either of
          the groundwater models.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of GWF Exchange cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of exchange
          entries will be echoed to the listing file immediately after it is
          read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of exchange
          flow rates will be printed to the listing file for every stress
          period in which "SAVE BUDGET" is specified in Output Control.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that cell-by-cell flow terms
          will be written to the budget file for each model provided that the
          Output Control for the models are set up with the "BUDGET SAVE FILE"
          option.
    cell_averaging : string
        * cell_averaging (string) is a keyword and text keyword to indicate the
          method that will be used for calculating the conductance for
          horizontal cell connections. The text value for CELL_AVERAGING can be
          "HARMONIC", "LOGARITHMIC", or "AMT-LMK", which means "arithmetic-mean
          thickness and logarithmic-mean hydraulic conductivity". If the user
          does not specify a value for CELL_AVERAGING, then the harmonic-mean
          method will be used.
    cvoptions : [dewatered]
        * dewatered (string) If the DEWATERED keyword is specified, then the
          vertical conductance is calculated using only the saturated thickness
          and properties of the overlying cell if the head in the underlying
          cell is below its top.
    newton : boolean
        * newton (boolean) keyword that activates the Newton-Raphson
          formulation for groundwater flow between connected, convertible
          groundwater cells. Cells will not dry when this option is used.
    xt3d : boolean
        * xt3d (boolean) keyword that activates the XT3D formulation between
          the cells connected with this GWF-GWF Exchange.
    gncdata : {varname:data} or gncdata data
        * Contains data for the gnc package. Data can be stored in a dictionary
          containing data for the gnc package with variable names as keys and
          package data as values. Data just for the gncdata variable is also
          acceptable. See gnc package documentation for more information.
    perioddata : {varname:data} or perioddata data
        * Contains data for the mvr package. Data can be stored in a dictionary
          containing data for the mvr package with variable names as keys and
          package data as values. Data just for the perioddata variable is also
          acceptable. See mvr package documentation for more information.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    dev_interfacemodel_on : boolean
        * dev_interfacemodel_on (boolean) activates the interface model
          mechanism for calculating the coefficients at (and possibly near) the
          exchange. This keyword should only be used for development purposes.
    nexg : integer
        * nexg (integer) keyword and integer value specifying the number of
          GWF-GWF exchanges.
    exchangedata : [cellidm1, cellidm2, ihc, cl1, cl2, hwva, aux, boundname]
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
        * ihc (integer) is an integer flag indicating the direction between
          node n and all of its m connections. If IHC = 0 then the connection
          is vertical. If IHC = 1 then the connection is horizontal. If IHC = 2
          then the connection is horizontal for a vertically staggered grid.
        * cl1 (double) is the distance between the center of cell 1 and the its
          shared face with cell 2.
        * cl2 (double) is the distance between the center of cell 2 and the its
          shared face with cell 1.
        * hwva (double) is the horizontal width of the flow connection between
          cell 1 and cell 2 if IHC > 0, or it is the area perpendicular to flow
          of the vertical connection between cell 1 and cell 2 if IHC = 0.
        * aux (double) represents the values of the auxiliary variables for
          each GWFGWF Exchange. The values of auxiliary variables must be
          present for each exchange. The values must be specified in the order
          of the auxiliary variables specified in the OPTIONS block.
        * boundname (string) name of the GWF Exchange cell. BOUNDNAME is an
          ASCII character variable that can contain as many as 40 characters.
          If BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(("gwfgwf", "options", "auxiliary"))
    gnc_filerecord = ListTemplateGenerator(
        ("gwfgwf", "options", "gnc_filerecord")
    )
    mvr_filerecord = ListTemplateGenerator(
        ("gwfgwf", "options", "mvr_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwfgwf", "options", "obs_filerecord")
    )
    exchangedata = ListTemplateGenerator(
        ("gwfgwf", "exchangedata", "exchangedata")
    )
    package_abbr = "gwfgwf"
    _package_type = "gwfgwf"
    dfn_file_name = "exg-gwfgwf.dfn"

    dfn = [
        [
            "header",
            "multi-package",
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
            "name cell_averaging",
            "type string",
            "valid harmonic logarithmic amt-lmk",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name cvoptions",
            "type record variablecv dewatered",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name variablecv",
            "in_record true",
            "type keyword",
            "reader urword",
        ],
        [
            "block options",
            "name dewatered",
            "in_record true",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name newton",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name xt3d",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name gnc_filerecord",
            "type record gnc6 filein gnc6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package gnc",
            "construct_data gncdata",
            "parameter_name gncdata",
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
            "name gnc6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name gnc6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block options",
            "name mvr_filerecord",
            "type record mvr6 filein mvr6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package mvr",
            "construct_data perioddata",
            "parameter_name perioddata",
        ],
        [
            "block options",
            "name mvr6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name mvr6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
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
            "block options",
            "name dev_interfacemodel_on",
            "type keyword",
            "reader urword",
            "optional true",
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
            "type recarray cellidm1 cellidm2 ihc cl1 cl2 hwva aux boundname",
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
            "name ihc",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name cl1",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name cl2",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name hwva",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block exchangedata",
            "name aux",
            "type double precision",
            "in_record true",
            "tagged false",
            "shape (naux)",
            "reader urword",
            "optional true",
        ],
        [
            "block exchangedata",
            "name boundname",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        exgtype=None,
        exgmnamea=None,
        exgmnameb=None,
        auxiliary=None,
        boundnames=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        cell_averaging=None,
        cvoptions=None,
        newton=None,
        xt3d=None,
        gncdata=None,
        perioddata=None,
        observations=None,
        dev_interfacemodel_on=None,
        nexg=None,
        exchangedata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            simulation, "gwfgwf", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.cell_averaging = self.build_mfdata(
            "cell_averaging", cell_averaging
        )
        self.cvoptions = self.build_mfdata("cvoptions", cvoptions)
        self.newton = self.build_mfdata("newton", newton)
        self.xt3d = self.build_mfdata("xt3d", xt3d)
        self._gnc_filerecord = self.build_mfdata("gnc_filerecord", None)
        self._gnc_package = self.build_child_package(
            "gnc", gncdata, "gncdata", self._gnc_filerecord
        )
        self._mvr_filerecord = self.build_mfdata("mvr_filerecord", None)
        self._mvr_package = self.build_child_package(
            "mvr", perioddata, "perioddata", self._mvr_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.dev_interfacemodel_on = self.build_mfdata(
            "dev_interfacemodel_on", dev_interfacemodel_on
        )
        self.nexg = self.build_mfdata("nexg", nexg)
        self.exchangedata = self.build_mfdata("exchangedata", exchangedata)
        self._init_complete = True
