# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwtgwt(mfpackage.MFPackage):
    """
    ModflowGwtgwt defines a gwtgwt package.

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
    gwfmodelname1 : string
        * gwfmodelname1 (string) keyword to specify name of first corresponding
          GWF Model. In the simulation name file, the GWT6-GWT6 entry contains
          names for GWT Models (exgmnamea and exgmnameb). The GWT Model with
          the name exgmnamea must correspond to the GWF Model with the name
          gwfmodelname1.
    gwfmodelname2 : string
        * gwfmodelname2 (string) keyword to specify name of second
          corresponding GWF Model. In the simulation name file, the GWT6-GWT6
          entry contains names for GWT Models (exgmnamea and exgmnameb). The
          GWT Model with the name exgmnameb must correspond to the GWF Model
          with the name gwfmodelname2.
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
          provided with the list of GWT Exchange cells.
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
    adv_scheme : string
        * adv_scheme (string) scheme used to solve the advection term. Can be
          upstream, central, or TVD. If not specified, upstream weighting is
          the default weighting scheme.
    dsp_xt3d_off : boolean
        * dsp_xt3d_off (boolean) deactivate the xt3d method for the dispersive
          flux and use the faster and less accurate approximation for this
          exchange.
    dsp_xt3d_rhs : boolean
        * dsp_xt3d_rhs (boolean) add xt3d dispersion terms to right-hand side,
          when possible, for this exchange.
    filein : boolean
        * filein (boolean) keyword to specify that an input filename is
          expected next.
    perioddata : {varname:data} or perioddata data
        * Contains data for the mvt package. Data can be stored in a dictionary
          containing data for the mvt package with variable names as keys and
          package data as values. Data just for the perioddata variable is also
          acceptable. See mvt package documentation for more information.
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
          GWT-GWT exchanges.
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
          each GWTGWT Exchange. The values of auxiliary variables must be
          present for each exchange. The values must be specified in the order
          of the auxiliary variables specified in the OPTIONS block.
        * boundname (string) name of the GWT Exchange cell. BOUNDNAME is an
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

    auxiliary = ListTemplateGenerator(("gwtgwt", "options", "auxiliary"))
    mvt_filerecord = ListTemplateGenerator(
        ("gwtgwt", "options", "mvt_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwtgwt", "options", "obs_filerecord")
    )
    exchangedata = ListTemplateGenerator(
        ("gwtgwt", "exchangedata", "exchangedata")
    )
    package_abbr = "gwtgwt"
    _package_type = "gwtgwt"
    dfn_file_name = "exg-gwtgwt.dfn"

    dfn = [
        [
            "header",
            "multi-package",
        ],
        [
            "block options",
            "name gwfmodelname1",
            "type string",
            "reader urword",
            "optional false",
        ],
        [
            "block options",
            "name gwfmodelname2",
            "type string",
            "reader urword",
            "optional false",
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
            "name adv_scheme",
            "type string",
            "valid upstream central tvd",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name dsp_xt3d_off",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name dsp_xt3d_rhs",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
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
            "name mvt_filerecord",
            "type record mvt6 filein mvt6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package mvt",
            "construct_data perioddata",
            "parameter_name perioddata",
        ],
        [
            "block options",
            "name mvt6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name mvt6_filename",
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
        gwfmodelname1=None,
        gwfmodelname2=None,
        auxiliary=None,
        boundnames=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        adv_scheme=None,
        dsp_xt3d_off=None,
        dsp_xt3d_rhs=None,
        filein=None,
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
            simulation, "gwtgwt", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self.gwfmodelname1 = self.build_mfdata("gwfmodelname1", gwfmodelname1)
        self.gwfmodelname2 = self.build_mfdata("gwfmodelname2", gwfmodelname2)
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.adv_scheme = self.build_mfdata("adv_scheme", adv_scheme)
        self.dsp_xt3d_off = self.build_mfdata("dsp_xt3d_off", dsp_xt3d_off)
        self.dsp_xt3d_rhs = self.build_mfdata("dsp_xt3d_rhs", dsp_xt3d_rhs)
        self.filein = self.build_mfdata("filein", filein)
        self._mvt_filerecord = self.build_mfdata("mvt_filerecord", None)
        self._mvt_package = self.build_child_package(
            "mvt", perioddata, "perioddata", self._mvt_filerecord
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
