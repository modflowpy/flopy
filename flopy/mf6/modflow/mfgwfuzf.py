from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfuzf(mfpackage.MFPackage):
    """
    ModflowGwfuzf defines a uzf package within a gwf6 model.

    Attributes
    ----------
    auxiliary : [(auxiliary : string)]
        defines an array of one or more auxiliary variable names. There is no limit on the number of auxiliary variables that can be provided on this line; however, lists of information provided in subsequent blocks must have a column of data for each auxiliary variable name defined here. The number of auxiliary variables detected on this line determines the value for naux. Comments cannot be provided anywhere on this line as they will be interpreted as auxiliary variable names. Auxiliary variables may not be used by the package, but they will be available for use by other parts of the program. The program will terminate with an error if auxiliary variables are specified on more than one line in the options block.
    auxmultname : (auxmultname : string)
        name of auxiliary variable to be used as multiplier of GWF cell area used by UZF cell.
    boundnames : (boundnames : keyword)
        keyword to indicate that boundary names may be provided with the list of UZF cells.
    print_input : (print_input : keyword)
        keyword to indicate that the list of UZF information will be written to the listing file immediately after it is read.
    print_flows : (print_flows : keyword)
        keyword to indicate that the list of UZF flow rates will be printed to the listing file for every stress period time step in which ``BUDGET PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_FLOWS is specified, then flow rates are printed for the last time step of each stress period.
    save_flows : (save_flows : keyword)
        keyword to indicate that UZF flow terms will be written to the file specified with ``BUDGET FILEOUT'' in Output Control.
    budget_filerecord : [(budget : keyword), (fileout : keyword), (budgetfile : string)]
        budget : keyword to specify that record corresponds to the budget.
        fileout : keyword to specify that an output filename is expected next.
        budgetfile : name of the binary output file to write budget information.
    ts_filerecord : [(ts6 : keyword), (filein : keyword), (ts6_filename : string)]
        ts6 : keyword to specify that record corresponds to a time-series file.
        filein : keyword to specify that an input filename is expected next.
        ts6_filename : defines a time-series file defining time series that can be used to assign time-varying values. See the ``Time-Variable Input'' section for instructions on using the time-series capability.
    obs_filerecord : [(obs6 : keyword), (filein : keyword), (obs6_filename : string)]
        filein : keyword to specify that an input filename is expected next.
        obs6 : keyword to specify that record corresponds to an observations file.
        obs6_filename : name of input file to define observations for the UZF package. See the ``Observation utility'' section for instructions for preparing observation input files. Table obstype lists observation type(s) supported by the UZF package.
    mover : (mover : keyword)
        keyword to indicate that this instance of the UZF Package can be used with the Water Mover (MVR) Package. When the MOVER option is specified, additional memory is allocated within the package to store the available, provided, and received water.
    simulate_et : (simulate_et : keyword)
        keyword specifying that ET in the unsaturated (UZF) and saturated zones (GWF) will be simulated. ET can be simulated in the UZF cell and not the GWF cell by emitting keywords LINEAR\_GWET and SQUARE\_GWET.
    linear_gwet : (linear_gwet : keyword)
        keyword specifying that groundwater ET will be simulated using the original ET formulation of MODFLOW-2005.
    square_gwet : (square_gwet : keyword)
        keyword specifying that groundwater ET will be simulated by assuming a constant ET rate for groundwater levels between land surface (TOP) and land surface minus the ET extinction depth (TOP-EXTDP). Groundwater ET is smoothly reduced from the PET rate to zero over a nominal interval at TOP-EXTDP.
    simulate_gwseep : (simulate_gwseep : keyword)
        keyword specifying that groundwater discharge (GWSEEP) to land surface will be simulated. Groundwater discharge is nonzero when groundwater head is greater than land surface.
    unsat_etwc : (unsat_etwc : keyword)
        keyword specifying that ET in the unsaturated zone will be simulated as a function of the specified PET rate while the water content (THETA) is greater than the ET extinction water content (extwc).
    unsat_etae : (unsat_etae : keyword)
        keyword specifying that ET in the unsaturated zone will be simulated simulated using a capillary pressure based formulation. Capillary pressure is calculated using the Brooks-Corey retention function.
    nuzfcells : (nuzfcells : integer)
        is the number of UZF cells. More than 1 UZF cell can be assigned to a GWF cell; however, only 1 GWF cell can be assigned to a single UZF cell. If the MULTILAYER option is used then UZF cells can be assigned to GWF cells below (in deeper layers than) the upper most active GWF cells.
    ntrailwaves : (ntrailwaves : integer)
        is the number of trailing waves. NTRAILWAVES has a default value of 7 and can be increased to lower mass balance error in the unsaturated zone.
    nwavesets : (nwavesets : integer)
        is the number of UZF cells specified. NWAVSETS has a default value of 40 and can be increased if more waves are required to resolve variations in water content within the unsaturated zone.
    uzfrecarray : [(iuzno : integer), (cellid : integer), (landflag : integer), (ivertcon : integer), (surfdep : double), (vks : double), (thtr : double), (thts : double), (thti : double), (eps : double), (boundname : string)]
        iuzno : integer value that defines the UZF cell number associated with the specified PACKAGEDATA data on the line. iuzno must be greater than zero and less than or equal to nuzfcells. UZF information must be specified for every UZF cell or the program will terminate with an error. The program will also terminate with an error if information for a UZF cell is specified more than once.
        cellid : is the cell identifier, and depends on the type of grid that is used for the simulation. For a structured grid that uses the DIS input file, cellid is the layer, row, and column. For a grid that uses the DISV input file, cellid is the layer and cell2d number. If the model uses the unstructured discretization (DISU) input file, then cellid is the node number for the cell.
        landflag : integer value set to one for land surface cells indicating that boundary conditions can be applied and data can be specified in the PERIOD block. A value of 0 specifies a non-land surface cell.
        ivertcon : integer value set to specify underlying UZF cell that receives water flowing to bottom of cell. If unsaturated zone flow reaches water table before the cell bottom then water is added to GWF cell instead of flowing to underlying UZF cell. A value of 0 indicates the UZF cell is not connected to an underlying UZF cell.
        surfdep : is the surface depression depth of the UZF cell.
        vks : is the vertical saturated hydraulic conductivity of the UZF cell.
        thtr : is the residual (irreducible) water content of the UZF cell.
        thts : is the saturated water content of the UZF cell.
        thti : is the initial water content of the UZF cell.
        eps : is the epsilon exponent of the UZF cell.
        boundname : name of the UZF cell cell. boundname is an ASCII character variable that can contain as many as 40 characters. If boundname contains spaces in it, then the entire name must be enclosed within single quotes.
    uzfperiodrecarray : [(iuzno : integer), (finf : string), (pet : string), (extdp : string), (extwc : string), (ha : string), (hroot : string), (rootact : string), (aux : double)]
        iuzno : integer value that defines the UZF cell number associated with the specified PERIOD data on the line.
        finf : real or character value that defines the applied infiltration rate of the UZF cell ($LT^{-1$). If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        pet : real or character value that defines the potential evapotranspiration rate of the UZF cell and specified GWF cell. Evapotranspiration is first removed from the unsaturated zone and any remaining potential evapotranspiration is applied to the saturated zone. If ivertcon is greater than zero then residual potential evapotranspiration not satisfied in the UZF cell is applied to the underlying UZF and GWF cells. pet is always specified, but is only used if SIMULATE\_ET is specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        extdp : real or character value that defines the evapotranspiration extinction depth of the UZF cell. If ivertcon is greater than zero and extdp extends below the GWF cell bottom then remaining potential evapotranspiration is applied to the underlying UZF and GWF cells. extdp is always specified, but is only used if SIMULATE\_ET is specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        extwc : real or character value that defines the evapotranspiration extinction water content of the UZF cell. extwc is always specified, but is only used if SIMULATE\_ET and UNSAT\_ETWC are specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        ha : real or character value that defines the air entry potential (head) of the UZF cell. ha is always specified, but is only used if SIMULATE\_ET and UNSAT\_ETAE are specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        hroot : real or character value that defines the root potential (head) of the UZF cell. hroot is always specified, but is only used if SIMULATE\_ET and UNSAT\_ETAE are specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        rootact : real or character value that defines the root activity function of the UZF cell. rootact is the length of roots in a given volume of soil divided by that volume. Values range from 0 to about 3 $cm^{-2$, depending on the plant community and its stage of development. rootact is always specified, but is only used if SIMULATE\_ET and UNSAT\_ETAE are specified in the OPTIONS block. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        aux : represents the values of the auxiliary variables for each UZF. The values of auxiliary variables must be present for each UZF. The values must be specified in the order of the auxiliary variables specified in the OPTIONS block. If the package supports time series and the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.

    """
    auxiliary = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'options', 'auxiliary'))
    budget_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'options', 'budget_filerecord'))
    ts_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'options', 'ts_filerecord'))
    obs_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'options', 'obs_filerecord'))
    uzfrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'packagedata', 'uzfrecarray'))
    uzfperiodrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'uzf', 'period', 'uzfperiodrecarray'))
    package_abbr = "gwfuzf"

    def __init__(self, model, add_to_package_list=True, auxiliary=None, auxmultname=None, boundnames=None,
                 print_input=None, print_flows=None, save_flows=None, budget_filerecord=None,
                 ts_filerecord=None, obs_filerecord=None, mover=None, simulate_et=None,
                 linear_gwet=None, square_gwet=None, simulate_gwseep=None, unsat_etwc=None,
                 unsat_etae=None, nuzfcells=None, ntrailwaves=None, nwavesets=None, uzfrecarray=None,
                 uzfperiodrecarray=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfuzf, self).__init__(model, "uzf", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)

        self.auxmultname = self.build_mfdata("auxmultname", auxmultname)

        self.boundnames = self.build_mfdata("boundnames", boundnames)

        self.print_input = self.build_mfdata("print_input", print_input)

        self.print_flows = self.build_mfdata("print_flows", print_flows)

        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.budget_filerecord = self.build_mfdata("budget_filerecord", budget_filerecord)

        self.ts_filerecord = self.build_mfdata("ts_filerecord", ts_filerecord)

        self.obs_filerecord = self.build_mfdata("obs_filerecord", obs_filerecord)

        self.mover = self.build_mfdata("mover", mover)

        self.simulate_et = self.build_mfdata("simulate_et", simulate_et)

        self.linear_gwet = self.build_mfdata("linear_gwet", linear_gwet)

        self.square_gwet = self.build_mfdata("square_gwet", square_gwet)

        self.simulate_gwseep = self.build_mfdata("simulate_gwseep", simulate_gwseep)

        self.unsat_etwc = self.build_mfdata("unsat_etwc", unsat_etwc)

        self.unsat_etae = self.build_mfdata("unsat_etae", unsat_etae)

        self.nuzfcells = self.build_mfdata("nuzfcells", nuzfcells)

        self.ntrailwaves = self.build_mfdata("ntrailwaves", ntrailwaves)

        self.nwavesets = self.build_mfdata("nwavesets", nwavesets)

        self.uzfrecarray = self.build_mfdata("uzfrecarray", uzfrecarray)

        self.uzfperiodrecarray = self.build_mfdata("uzfperiodrecarray", uzfperiodrecarray)


