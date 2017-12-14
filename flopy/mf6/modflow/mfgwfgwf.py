from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfgwf(mfpackage.MFPackage):
    """
    ModflowGwfgwf defines a gwfgwf package.

    Attributes
    ----------
    exgtype : <string>
        is the exchange type (GWF-GWF or GWF-GWT).
    exgmnamea : <string>
        is the name of the first model that is part of this exchange.
    exgmnameb : <string>
        is the name of the second model that is part of this exchange.
    auxiliary : [(auxiliary : string)]
        auxiliary : an array of auxiliary variable names. There is no limit on
          the number of auxiliary variables that can be provided. Most
          auxiliary variables will not be used by the GWF-GWF Exchange, but
          they will be available for use by other parts of the program. If an
          auxiliary variable with the name ``ANGLDEGX'' is found, then this
          information will be used as the angle (provided in degrees) between
          the connection face normal and the x axis. Additional information on
          ``ANGLDEGX'' is provided in the description of the DISU Package. If
          an auxiliary variable with the name ``CDIST'' is found, then this
          information will be used as the straight-line connection distance
          between the two cell centers. CDIST is required if specific discharge
          is calculated for either of the groundwater models.
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of exchange entries
          will be echoed to the listing file immediately after it is read.
    print_flows : (print_flows : boolean)
        print_flows : keyword to indicate that the list of exchange flow rates
          will be printed to the listing file for every stress period in which
          ``SAVE BUDGET'' is specified in Output Control.
    save_flows : (save_flows : boolean)
        save_flows : keyword to indicate that cell-by-cell flow terms will be
          written to the budget file for each model provided that the Output
          Control for the models are set up with the ``BUDGET SAVE FILE''
          option.
    cell_averaging : (cell_averaging : string)
        cell_averaging : is a keyword and text keyword to indicate the method
          that will be used for calculating the conductance for horizontal cell
          connections. The text value for cell\_averaging can be
          ``HARMONIC'', ``LOGARITHMIC'', or ``AMT-LMK'', which means
          ``arithmetic-mean thickness and logarithmic-mean hydraulic
          conductivity''. If the user does not specify a value for
          cell\_averaging, then the harmonic-mean method will be used.
    cvoptions : [(dewatered : string)]
        dewatered : If the DEWATERED keyword is specified, then the vertical
          conductance is calculated using only the saturated thickness and
          properties of the overlying cell if the head in the underlying cell
          is below its top.
    newton : (newton : boolean)
        newton : keyword that activates the Newton-Raphson formulation for
          groundwater flow between connected, convertible groundwater cells.
          Cells will not dry when this option is used.
    gnc_filerecord : [(gnc6_filename : string)]
        gnc6_filename : is the file name for ghost node correction input file.
          Information for the ghost nodes are provided in the file provided
          with these keywords. The format for specifying the ghost nodes is the
          same as described for the GNC Package of the GWF Model. This includes
          specifying OPTIONS, DIMENSIONS, and GNCDATA blocks. The order of the
          ghost nodes must follow the same order as the order of the cells in
          the EXCHANGEDATA block. For the GNCDATA, noden and all of the nodej
          values are assumed to be located in model 1, and nodem is assumed to
          be in model 2.
    mvr_filerecord : [(mvr6_filename : string)]
        mvr6_filename : is the file name of the water mover input file to apply
          to this exchange. Information for the water mover are provided in the
          file provided with these keywords. The format for specifying the
          water mover information is the same as described for the Water Mover
          (MVR) Package of the GWF Model, with two exceptions. First, in the
          PACKAGES block, the model name must be included as a separate string
          before each package. Second, the appropriate model name must be
          included before pname1 and pname2 in the BEGIN
          PERIOD block. This allows providers and receivers to be located in
          both models listed as part of this exchange.
    obs_filerecord : [(obs6_filename : string)]
        obs6_filename : is the file name of the observations input file for
          this exchange. See the ``Observation utility'' section for
          instructions for preparing observation input files. Table
          \ref{table:obstype lists observation type(s) supported by the GWF-
          GWF package.
    nexg : (nexg : integer)
        nexg : keyword and integer value specifying the number of GWF-GWF
          exchanges.
    gwfgwfrecarray : [(cellidm1 : (integer, ...)), (cellidm2 : (integer, ...)),
      (ihc : integer), (cl1 : double), (cl2 : double), (hwva : double), (aux :
      double)]
        cellidm1 : is the cellid of the cell in model 1 as specified in the
          simulation name file. For a structured grid that uses the DIS input
          file, cellidm1 is the layer, row, and column numbers of the
          cell. For a grid that uses the DISV input file, cellidm1 is
          the layer number and cell2d number for the two cells. If the model
          uses the unstructured discretization (DISU) input file, then
          cellidm1 is the node number for the cell.
        cellidm2 : is the cellid of the cell in model 2 as specified in the
          simulation name file. For a structured grid that uses the DIS input
          file, cellidm2 is the layer, row, and column numbers of the
          cell. For a grid that uses the DISV input file, cellidm2 is
          the layer number and cell2d number for the two cells. If the model
          uses the unstructured discretization (DISU) input file, then
          cellidm2 is the node number for the cell.
        ihc : is an integer flag indicating the direction between node n and
          all of its m connections. If $ihc=0$ then the connection is vertical.
          If $ihc=1$ then the connection is horizontal. If $ihc=2$ then the
          connection is horizontal for a vertically staggered grid.
        cl1 : is the distance between the center of cell nodem1 and the its
          shared face with nodem2.
        cl2 : is the distance between the center of cell nodem2 and the its
          shared face with nodem1.
        hwva : is the horizontal width of the flow connection between
          nodem1 and nodem2 if $ihc > 0$, or it is
          the area of the vertical connection between nodem1 and
          nodem2 if $ihc = 0$.
        aux : represents the values of the auxiliary variables for each GWFGWF
          Exchange. The values of auxiliary variables must be present for each
          exchange. The values must be specified in the order of the auxiliary
          variables specified in the OPTIONS block.

    """
    auxiliary = ListTemplateGenerator(('gwfgwf', 'options', 'auxiliary'))
    gnc_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'gnc_filerecord'))
    mvr_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'mvr_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwfgwf', 'options', 
                                            'obs_filerecord'))
    gwfgwfrecarray = ListTemplateGenerator(('gwfgwf', 'exchangedata', 
                                            'gwfgwfrecarray'))
    package_abbr = "gwfgwf"

    def __init__(self, simulation, add_to_package_list=True, exgtype=None,
                 exgmnamea=None, exgmnameb=None, auxiliary=None,
                 print_input=None, print_flows=None, save_flows=None,
                 cell_averaging=None, cvoptions=None, newton=None,
                 gnc_filerecord=None, mvr_filerecord=None, obs_filerecord=None,
                 nexg=None, gwfgwfrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfgwf, self).__init__(simulation, "gwfgwf", fname, pname,
                                            add_to_package_list, parent_file)        

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
        self.gwfgwfrecarray = self.build_mfdata("gwfgwfrecarray", 
                                                gwfgwfrecarray)
