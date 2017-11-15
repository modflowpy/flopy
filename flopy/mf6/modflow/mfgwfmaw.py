from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfmaw(mfpackage.MFPackage):
    package_abbr = "gwfmaw"
    auxiliary = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'options', 'auxiliary'))
    stage_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'options', 'stage_filerecord'))
    budget_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'options', 'budget_filerecord'))
    ts_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'options', 'ts_filerecord'))
    obs_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'options', 'obs_filerecord'))
    wellrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'packagedata', 'wellrecarray'))
    wellconnectionsrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'connectiondata', 'wellconnectionsrecarray'))
    wellperiodrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'maw', 'period', 'wellperiodrecarray'))
    """
    ModflowGwfmaw defines a maw package within a gwf6 model.

    Attributes
    ----------
    auxiliary : [(auxiliary : string)]
        defines an array of one or more auxiliary variable names. There is no limit on the number of auxiliary variables that can be provided on this line; however, lists of information provided in subsequent blocks must have a column of data for each auxiliary variable name defined here. The number of auxiliary variables detected on this line determines the value for naux. Comments cannot be provided anywhere on this line as they will be interpreted as auxiliary variable names. Auxiliary variables may not be used by the package, but they will be available for use by other parts of the program. The program will terminate with an error if auxiliary variables are specified on more than one line in the options block.
    boundnames : (boundnames : keyword)
        keyword to indicate that boundary names may be provided with the list of multi-aquifer well cells.
    print_input : (print_input : keyword)
        keyword to indicate that the list of multi-aquifer well information will be written to the listing file immediately after it is read.
    print_head : (print_head : keyword)
        keyword to indicate that the list of multi-aquifer well heads will be printed to the listing file for every stress period in which ``HEAD PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_HEAD is specified, then heads are printed for the last time step of each stress period.
    print_flows : (print_flows : keyword)
        keyword to indicate that the list of multi-aquifer well flow rates will be printed to the listing file for every stress period time step in which ``BUDGET PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_FLOWS is specified, then flow rates are printed for the last time step of each stress period.
    save_flows : (save_flows : keyword)
        keyword to indicate that multi-aquifer well flow terms will be written to the file specified with ``BUDGET FILEOUT'' in Output Control.
    stage_filerecord : [(head : keyword), (fileout : keyword), (headfile : string)]
        head : keyword to specify that record corresponds to head.
        headfile : name of the binary output file to write stage information.
        fileout : keyword to specify that an output filename is expected next.
    budget_filerecord : [(budget : keyword), (fileout : keyword), (budgetfile : string)]
        budget : keyword to specify that record corresponds to the budget.
        fileout : keyword to specify that an output filename is expected next.
        budgetfile : name of the binary output file to write budget information.
    no_well_storage : (no_well_storage : keyword)
        keyword that deactivates inclusion of well storage contributions to the multi-aquifer well package continuity equation.
    flowing_wells : (flowing_wells : keyword)
        keyword that activates the flowing wells option for the multi-aquifer well package.
    shutdown_theta : (shutdown_theta : double)
        value that defines the weight applied to discharge rate for wells that limit the water level in a discharging well (defined using the HEAD\_LIMIT keyword in the stress period data). SHUTDOWN\_THETA is used to control discharge rate oscillations when the flow rate from the aquifer is less than the specified flow rate from the aquifer to the well. Values range between 0.0 and 1.0, and larger values increase the weight (decrease under-relaxation) applied to the well discharge rate. The head\_limit option has been included to facilitate backward compatibility with previous versions of MODFLOW but use of the rate\_scaling option instead of the head\_limit option is recommended. By default, SHUTDOWN\_THETA is 0.7.
    shutdown_kappa : (shutdown_kappa : double)
        value that defines the weight applied to discharge rate for wells that limit the water level in a discharging well (defined using the HEAD\_LIMIT keyword in the stress period data). SHUTDOWN\_KAPPA is used to control discharge rate oscillations when the flow rate from the aquifer is less than the specified flow rate from the aquifer to the well. Values range between 0.0 and 1.0, and larger values increase the weight applied to the well discharge rate. The head\_limit option has been included to facilitate backward compatibility with previous versions of MODFLOW but use of the rate\_scaling option instead of the head\_limit option is recommended. By default, SHUTDOWN\_KAPPA is 0.0001.
    ts_filerecord : [(ts6 : keyword), (filein : keyword), (ts6_filename : string)]
        ts6 : keyword to specify that record corresponds to a time-series file.
        filein : keyword to specify that an input filename is expected next.
        ts6_filename : defines a time-series file defining time series that can be used to assign time-varying values. See the ``Time-Variable Input'' section for instructions on using the time-series capability.
    obs_filerecord : [(obs6 : keyword), (filein : keyword), (obs6_filename : string)]
        filein : keyword to specify that an input filename is expected next.
        obs6 : keyword to specify that record corresponds to an observations file.
        obs6_filename : name of input file to define observations for the MAW package. See the ``Observation utility'' section for instructions for preparing observation input files. Table obstype lists observation type(s) supported by the MAW package.
    mover : (mover : keyword)
        keyword to indicate that this instance of the MAW Package can be used with the Water Mover (MVR) Package. When the MOVER option is specified, additional memory is allocated within the package to store the available, provided, and received water.
    nmawwells : (nmawwells : integer)
        integer value specifying the number of multi-aquifer wells that will be simulated for all stress periods.
    wellrecarray : [(wellno : integer), (radius : double), (bottom : double), (strt : double), (condeqn : string), (ngwfnodes : integer), (aux : double), (boundname : string)]
        wellno : integer value that defines the well number associated with the specified PACKAGEDATA data on the line. wellno must be greater than zero and less than or equal to nmawwells. Multi-aquifer well information must be specified for every multi-aquifer well or the program will terminate with an error. The program will also terminate with an error if information for a multi-aquifer well is specified more than once.
        radius : radius for the multi-aquifer well.
        bottom : bottom elevation of the multi-aquifer well.
        strt : starting head for the multi-aquifer well.
        condeqn : character string that defines the conductance equation that is used to calculate the saturated conductance for the multi-aquifer well. Possible multi-aquifer well condeqn strings include: SPECIFIED--character keyword to indicate the multi-aquifer well saturated conductance will be specified. THEIM--character keyword to indicate the multi-aquifer well saturated conductance will be calculated using the Theim equation. SKIN--character keyword to indicate that the multi-aquifer well saturated conductance will be calculated using the screen top and bottom, screen hydraulic conductivity, and skin radius. CUMULATIVE--character keyword to indicate that the multi-aquifer well saturated conductance will be calculated using a combination of the Theim equation and the screen top and bottom, screen hydraulic conductivity, and skin radius. MEAN--character keyword to indicate the multi-aquifer well saturated conductance will be calculated using using the aquifer and screen top and bottom, aquifer and screen hydraulic conductivity, and well and skin radius.
        ngwfnodes : integer value that defines the number of GWF nodes connected to this (wellno) multi-aquifer well. One or more screened intervals can be connected to the same GWF node. ngwfnodes must be greater than zero.
        aux : represents the values of the auxiliary variables for each multi-aquifer well. The values of auxiliary variables must be present for each multi-aquifer well. The values must be specified in the order of the auxiliary variables specified in the OPTIONS block. If the package supports time series and the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        boundname : name of the multi-aquifer well cell. boundname is an ASCII character variable that can contain as many as 40 characters. If boundname contains spaces in it, then the entire name must be enclosed within single quotes.
    wellconnectionsrecarray : [(wellno : integer), (icon : integer), (cellid : integer), (scrn_top : double), (scrn_bot : double), (hk_skin : double), (radius_skin : double)]
        wellno : integer value that defines the well number associated with the specified CONNECTIONDATA data on the line. wellno must be greater than zero and less than or equal to nmawwells. Multi-aquifer well connection information must be specified for every multi-aquifer well connection to the GWF model (ngwfnodes) or the program will terminate with an error. The program will also terminate with an error if connection information for a multi-aquifer well connection to the GWF model is specified more than once.
        icon : integer value that defines the GWF connection number for this multi-aquifer well connection entry. iconn must be greater than zero and less than or equal to ngwfnodes for multi-aquifer well wellno.
        cellid : is the cell identifier, and depends on the type of grid that is used for the simulation. For a structured grid that uses the DIS input file, cellid is the layer, row, and column. For a grid that uses the DISV input file, cellid is the layer and cell2d number. If the model uses the unstructured discretization (DISU) input file, then cellid is the node number for the cell.
        scrn_top : value that defines the top elevation of the screen for the multi-aquifer well connection. scrn\_top can be any value if condeqn is SPECIFIED or THEIM. If the specified scrn\_top is greater than the top of the GWF cell it is set equal to the top of the cell.
        scrn_bot : value that defines the bottom elevation of the screen for the multi-aquifer well connection. scrn\_bot can be any value if condeqn is SPECIFIED or THEIM. If the specified scrn\_bot is less than the bottom of the GWF cell it is set equal to the bottom of the cell.
        hk_skin : value that defines the skin (filter pack) hydraulic conductivity (if condeqn for the multi-aquifer well is SKIN, CUMULATIVE, or MEAN) or conductance (if condeqn for the multi-aquifer well is SPECIFIED) for each GWF node connected to the multi-aquifer well (ngwfnodes). hk\_skin can be any value if condeqn is THEIM.
        radius_skin : real value that defines the skin radius (filter pack radius) for the multi-aquifer well. radius\_skin can be any value if condeqn is SPECIFIED or THEIM. Otherwise, radius\_skin must be greater than radius for the multi-aquifer well.
    wellperiodrecarray : [(wellno : integer), (mawsetting : keystring)]
        wellno : integer value that defines the well number associated with the specified PERIOD data on the line. wellno must be greater than zero and less than or equal to nmawwells.
        mawsetting : line of information that is parsed into a keyword and values. Keyword values that can be used to start the mawsetting string include: STATUS, FLOWING\_WELL, RATE, WELL\_HEAD, HEAD\_LIMIT, SHUT\_OFF, RATE\_SCALING, and AUXILIARY.
    flowing_well : (flowing_well : keyword)
        keyword to indicate the well is a flowing well. The flowing\_well option can be used to simulate flowing wells when the simulated well head exceeds the specified drainage elevation.
    fwelev : (fwelev : double)
        elevation used to determine whether or not the well is flowing.
    fwcond : (fwcond : double)
        conductance used to calculate the discharge of a free flowing well. Flow occurs when the head in the well is above the well top elevation (fwelev).
    shut_off : (shut_off : keyword)
        keyword for activating well shut off capability. Subsequent values define the minimum and maximum pumping rate that a well must exceed to shutoff or reactivate a well, respectively, during a stress period. shut\_off is only applied to discharging wells (rate$<0$) and if head\_limit is specified (not set to `off'). If head\_limit is specified, shut\_off can be deactivated by specifying a minimum value equal to zero. The shut\_off option is based on the shut\_off functionality available in the MNW2~\citep{konikow2009 package for MODFLOW-2005. The shut\_off option has been included to facilitate backward compatibility with previous versions of MODFLOW but use of the rate\_scaling option instead of the shut\_off option is recommended. By default, shut\_off is not used.
    minrate : (minrate : double)
        is the minimum rate that a well must exceed to shutoff a well during a stress period. The well will shut down during a time step if the flow rate to the well from the aquifer is less than minrate. If a well is shut down during a time step, reactivation of the well cannot occur until the next time step to reduce oscillations. minrate must be less than maxrate.
    maxrate : (maxrate : double)
        is the maximum rate that a well must exceed to reactivate a well during a stress period. The well will reactivate during a timestep if the well was shutdown during the previous time step and the flow rate to the well from the aquifer exceeds maxrate. Reactivation of the well cannot occur until the next time step if a well is shutdown to reduce oscillations. maxrate must be greater than minrate.
    rate_scaling : (rate_scaling : keyword)
        activate rate scaling. If rate\_scaling is specified, both pump\_elevation and scaling\_length must be specified. rate\_scaling cannot be used with head\_limit.
    pump_elevation : (pump_elevation : double)
        is the elevation of the multi-aquifer well pump (pump\_elevation). pump\_elevation cannot be less than the bottom elevation (bottom) of the multi-aquifer well. By default, pump\_elevation is set equal to the bottom of the largest GWF node number connected to a MAW well.
    scaling_length : (scaling_length : double)
        height above the pump elevation (scaling\_length) below which the pumping rate is reduced. The default value for scaling\_length is the well radius.
    auxname : (auxname : string)
        name for the auxiliary variable to be assigned auxval. auxname must match one of the auxiliary variable names defined in the OPTIONS block. If auxname does not match one of the auxiliary variable names defined in the OPTIONS block the data are ignored.
    auxval : (auxval : double)
        value for the auxiliary variable. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.

    """
    def __init__(self, model, add_to_package_list=True, auxiliary=None, boundnames=None, print_input=None,
                 print_head=None, print_flows=None, save_flows=None, stage_filerecord=None,
                 budget_filerecord=None, no_well_storage=None, flowing_wells=None,
                 shutdown_theta=None, shutdown_kappa=None, ts_filerecord=None, obs_filerecord=None,
                 mover=None, nmawwells=None, wellrecarray=None, wellconnectionsrecarray=None,
                 wellperiodrecarray=None, flowing_well=None, fwelev=None, fwcond=None, shut_off=None,
                 minrate=None, maxrate=None, rate_scaling=None, pump_elevation=None,
                 scaling_length=None, auxname=None, auxval=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfmaw, self).__init__(model, "maw", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)

        self.boundnames = self.build_mfdata("boundnames", boundnames)

        self.print_input = self.build_mfdata("print_input", print_input)

        self.print_head = self.build_mfdata("print_head", print_head)

        self.print_flows = self.build_mfdata("print_flows", print_flows)

        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.stage_filerecord = self.build_mfdata("stage_filerecord", stage_filerecord)

        self.budget_filerecord = self.build_mfdata("budget_filerecord", budget_filerecord)

        self.no_well_storage = self.build_mfdata("no_well_storage", no_well_storage)

        self.flowing_wells = self.build_mfdata("flowing_wells", flowing_wells)

        self.shutdown_theta = self.build_mfdata("shutdown_theta", shutdown_theta)

        self.shutdown_kappa = self.build_mfdata("shutdown_kappa", shutdown_kappa)

        self.ts_filerecord = self.build_mfdata("ts_filerecord", ts_filerecord)

        self.obs_filerecord = self.build_mfdata("obs_filerecord", obs_filerecord)

        self.mover = self.build_mfdata("mover", mover)

        self.nmawwells = self.build_mfdata("nmawwells", nmawwells)

        self.wellrecarray = self.build_mfdata("wellrecarray", wellrecarray)

        self.wellconnectionsrecarray = self.build_mfdata("wellconnectionsrecarray", wellconnectionsrecarray)

        self.wellperiodrecarray = self.build_mfdata("wellperiodrecarray", wellperiodrecarray)

        self.flowing_well = self.build_mfdata("flowing_well", flowing_well)

        self.fwelev = self.build_mfdata("fwelev", fwelev)

        self.fwcond = self.build_mfdata("fwcond", fwcond)

        self.shut_off = self.build_mfdata("shut_off", shut_off)

        self.minrate = self.build_mfdata("minrate", minrate)

        self.maxrate = self.build_mfdata("maxrate", maxrate)

        self.rate_scaling = self.build_mfdata("rate_scaling", rate_scaling)

        self.pump_elevation = self.build_mfdata("pump_elevation", pump_elevation)

        self.scaling_length = self.build_mfdata("scaling_length", scaling_length)

        self.auxname = self.build_mfdata("auxname", auxname)

        self.auxval = self.build_mfdata("auxval", auxval)


