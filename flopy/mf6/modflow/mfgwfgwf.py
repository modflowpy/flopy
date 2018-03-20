# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


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
          the connection face normal and the x axis. Additional information on
          "ANGLDEGX" is provided in the description of the DISU Package. If an
          auxiliary variable with the name "CDIST" is found, then this
          information will be used as the straight-line connection distance
          between the two cell centers. CDIST is required if specific discharge
          is calculated for either of the groundwater models.
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
    gnc_filerecord : [gnc6_filename]
        * gnc6_filename (string) is the file name for ghost node correction
          input file. Information for the ghost nodes are provided in the file
          provided with these keywords. The format for specifying the ghost
          nodes is the same as described for the GNC Package of the GWF Model.
          This includes specifying OPTIONS, DIMENSIONS, and GNCDATA blocks. The
          order of the ghost nodes must follow the same order as the order of
          the cells in the EXCHANGEDATA block. For the GNCDATA, noden and all
          of the nodej values are assumed to be located in model 1, and nodem
          is assumed to be in model 2.
    mvr_filerecord : [mvr6_filename]
        * mvr6_filename (string) is the file name of the water mover input file
          to apply to this exchange. Information for the water mover are
          provided in the file provided with these keywords. The format for
          specifying the water mover information is the same as described for
          the Water Mover (MVR) Package of the GWF Model, with two exceptions.
          First, in the PACKAGES block, the model name must be included as a
          separate string before each package. Second, the appropriate model
          name must be included before package name 1 and package name 2 in the
          BEGIN PERIOD block. This allows providers and receivers to be located
          in both models listed as part of this exchange.
    obs_filerecord : [obs6_filename]
        * obs6_filename (string) is the file name of the observations input
          file for this exchange. See the "Observation utility" section for
          instructions for preparing observation input files. Table
          ref{table:obstype} lists observation type(s) supported by the GWF-GWF
          package.
    nexg : integer
        * nexg (integer) keyword and integer value specifying the number of
          GWF-GWF exchanges.
    exchangedata : [cellidm1, cellidm2, ihc, cl1, cl2, hwva, aux]
        * cellidm1 ((integer, ...)) is the cellid of the cell in model 1 as
          specified in the simulation name file. For a structured grid that
          uses the DIS input file, CELLIDM1 is the layer, row, and column
          numbers of the cell. For a grid that uses the DISV input file,
          CELLIDM1 is the layer number and CELL2D number for the two cells. If
          the model uses the unstructured discretization (DISU) input file,
          then CELLIDM1 is the node number for the cell.
        * cellidm2 ((integer, ...)) is the cellid of the cell in model 2 as
          specified in the simulation name file. For a structured grid that
          uses the DIS input file, CELLIDM2 is the layer, row, and column
          numbers of the cell. For a grid that uses the DISV input file,
          CELLIDM2 is the layer number and CELL2D number for the two cells. If
          the model uses the unstructured discretization (DISU) input file,
          then CELLIDM2 is the node number for the cell.
        * ihc (integer) is an integer flag indicating the direction between
          node n and all of its m connections. If IHC = 0 then the connection
          is vertical. If IHC = 1 then the connection is horizontal. If IHC = 2
          then the connection is horizontal for a vertically staggered grid.
        * cl1 (double) is the distance between the center of cell 1 and the its
          shared face with cell 2.
        * cl2 (double) is the distance between the center of cell 1 and the its
          shared face with cell 2.
        * hwva (double) is the horizontal width of the flow connection between
          cell 1 and cell 2 if IHC :math:`>` 0, or it is the area perpendicular
          to flow of the vertical connection between cell 1 and cell 2 if IHC =
          0.
        * aux (double) represents the values of the auxiliary variables for
          each GWFGWF Exchange. The values of auxiliary variables must be
          present for each exchange. The values must be specified in the order
          of the auxiliary variables specified in the OPTIONS block.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    auxiliary = ListTemplateGenerator(('gwfgwf', 'options', 'auxiliary'))
    gnc_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'gnc_filerecord'))
    mvr_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'mvr_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'obs_filerecord'))
    exchangedata = ListTemplateGenerator(('gwfgwf', 'exchangedata', 
                                          'exchangedata'))
    package_abbr = "gwfgwf"
    package_type = "gwfgwf"
    dfn_file_name = "exg-gwfgwf.dfn"

    dfn = [["block options", "name auxiliary", "type string", 
            "shape (naux)", "reader urword", "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name cell_averaging", "type string", 
            "valid harmonic logarithmic amt-lmk", "reader urword", 
            "optional true"],
           ["block options", "name cvoptions", 
            "type record variablecv dewatered", "reader urword", 
            "optional true"],
           ["block options", "name variablecv", "in_record true", 
            "type keyword", "reader urword"],
           ["block options", "name dewatered", "in_record true", 
            "type keyword", "reader urword", "optional true"],
           ["block options", "name newton", "type keyword", "reader urword", 
            "optional true"],
           ["block options", "name gnc_filerecord", 
            "type record gnc6 filein gnc6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name gnc6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name gnc6_filename", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword", "optional false"],
           ["block options", "name mvr_filerecord", 
            "type record mvr6 filein mvr6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name mvr6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name mvr6_filename", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword", "optional false"],
           ["block options", "name obs_filerecord", 
            "type record obs6 filein obs6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name obs6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name obs6_filename", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword", "optional false"],
           ["block dimensions", "name nexg", "type integer", 
            "reader urword", "optional false"],
           ["block exchangedata", "name exchangedata", 
            "type recarray cellidm1 cellidm2 ihc cl1 cl2 hwva aux", 
            "reader urword", "optional false"],
           ["block exchangedata", "name cellidm1", "type integer", 
            "in_record true", "tagged false", "reader urword", 
            "optional false", "numeric_index true"],
           ["block exchangedata", "name cellidm2", "type integer", 
            "in_record true", "tagged false", "reader urword", 
            "optional false", "numeric_index true"],
           ["block exchangedata", "name ihc", "type integer", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block exchangedata", "name cl1", "type double precision", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block exchangedata", "name cl2", "type double precision", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block exchangedata", "name hwva", "type double precision", 
            "in_record true", "tagged false", "reader urword", 
            "optional false"],
           ["block exchangedata", "name aux", "type double precision", 
            "in_record true", "tagged false", "shape (naux)", "reader urword", 
            "optional true"]]

    def __init__(self, simulation, loading_package=False, exgtype=None,
                 exgmnamea=None, exgmnameb=None, auxiliary=None,
                 print_input=None, print_flows=None, save_flows=None,
                 cell_averaging=None, cvoptions=None, newton=None,
                 gnc_filerecord=None, mvr_filerecord=None, obs_filerecord=None,
                 nexg=None, exchangedata=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfgwf, self).__init__(simulation, "gwfgwf", fname, pname,
                                            loading_package, parent_file)        

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self.auxiliary = self.build_mfdata("auxiliary",  auxiliary)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.cell_averaging = self.build_mfdata("cell_averaging", 
                                                cell_averaging)
        self.cvoptions = self.build_mfdata("cvoptions",  cvoptions)
        self.newton = self.build_mfdata("newton",  newton)
        self.gnc_filerecord = self.build_mfdata("gnc_filerecord", 
                                                gnc_filerecord)
        self.mvr_filerecord = self.build_mfdata("mvr_filerecord", 
                                                mvr_filerecord)
        self.obs_filerecord = self.build_mfdata("obs_filerecord", 
                                                obs_filerecord)
        self.nexg = self.build_mfdata("nexg",  nexg)
        self.exchangedata = self.build_mfdata("exchangedata",  exchangedata)
