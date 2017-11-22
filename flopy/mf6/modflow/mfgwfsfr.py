from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfsfr(mfpackage.MFPackage):
    """
    ModflowGwfsfr defines a sfr package within a gwf6 model.

    Attributes
    ----------
    auxiliary : [(auxiliary : string)]
        defines an array of one or more auxiliary variable names. There is no limit on the number of auxiliary variables that can be provided on this line; however, lists of information provided in subsequent blocks must have a column of data for each auxiliary variable name defined here. The number of auxiliary variables detected on this line determines the value for naux. Comments cannot be provided anywhere on this line as they will be interpreted as auxiliary variable names. Auxiliary variables may not be used by the package, but they will be available for use by other parts of the program. The program will terminate with an error if auxiliary variables are specified on more than one line in the options block.
    boundnames : (boundnames : keyword)
        keyword to indicate that boundary names may be provided with the list of stream reach cells.
    print_input : (print_input : keyword)
        keyword to indicate that the list of stream reach information will be written to the listing file immediately after it is read.
    print_stage : (print_stage : keyword)
        keyword to indicate that the list of stream reach stages will be printed to the listing file for every stress period in which ``HEAD PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_STAGE is specified, then stages are printed for the last time step of each stress period.
    print_flows : (print_flows : keyword)
        keyword to indicate that the list of stream reach flow rates will be printed to the listing file for every stress period time step in which ``BUDGET PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_FLOWS is specified, then flow rates are printed for the last time step of each stress period.
    save_flows : (save_flows : keyword)
        keyword to indicate that stream reach flow terms will be written to the file specified with ``BUDGET FILEOUT'' in Output Control.
    stage_filerecord : [(stage : keyword), (fileout : keyword), (stagefile : string)]
        stage : keyword to specify that record corresponds to stage.
        stagefile : name of the binary output file to write stage information.
        fileout : keyword to specify that an output filename is expected next.
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
        obs6_filename : name of input file to define observations for the SFR package. See the ``Observation utility'' section for instructions for preparing observation input files. Table obstype lists observation type(s) supported by the SFR package.
    mover : (mover : keyword)
        keyword to indicate that this instance of the SFR Package can be used with the Water Mover (MVR) Package. When the MOVER option is specified, additional memory is allocated within the package to store the available, provided, and received water.
    maximum_iterations : (maximum_iterations : double)
        value that defines an maximum number of Streamflow Routing Newton-Raphson iterations allowed for a reach. By default, maxsfrit is equal to 100.
    maximum_depth_change : (maximum_depth_change : double)
        value that defines the depth closure tolerance. By default, dmaxchg is equal to $1 \times 10^{-5$.
    unit_conversion : (unit_conversion : double)
        value (or conversion factor) that is used in calculating stream depth for stream reach. A constant of 1.486 is used for flow units of cubic feet per second, and a constant of 1.0 is used for units of cubic meters per second. The constant must be multiplied by 86,400 when using time units of days in the simulation.
    nreaches : (nreaches : integer)
        integer value specifying the number of stream reaches. There must be nreaches entries in the PACKAGEDATA block.
    sfrrecarray : [(rno : integer), (cellid : integer), (rlen : double), (rwid : double), (rgrd : double), (rtp : double), (rbth : double), (rhk : double), (man : string), (ncon : integer), (ustrf : double), (ndv : integer), (aux : double), (boundname : string)]
        rno : integer value that defines the reach number associated with the specified PACKAGEDATA data on the line. rno must be greater than zero and less than or equal to nreaches. Reach information must be specified for every reach or the program will terminate with an error. The program will also terminate with an error if information for a reach is specified more than once.
        cellid : The keyword `none' must be specified for reaches that are not connected to an underlying GWF cell. The keyword `none' is used for reaches that are in cells that have IDOMAIN values less than one or are in areas not covered by the GWF model grid. Reach-aquifer flow is not calculated if the keyword `none' is specified.
        rlen : real value that defines the reach length. rlen must be greater than zero.
        rwid : real value that defines the reach width. rwid must be greater than zero.
        rgrd : real value that defines the stream gradient (slope) across the reach. rgrd must be greater than zero.
        rtp : real value that defines the top elevation of the reach streambed.
        rbth : real value that defines the thickness of the reach streambed. rbth can be any value if cellid is `none'. Otherwise, rbth must be greater than zero.
        rhk : real value that defines the hydraulic conductivity of the reach streambed. rhk can be any positive value if cellid is `none'. Otherwise, rhk must be greater than zero.
        man : real or character value that defines the Manning's roughness coefficient for the reach. man must be greater than zero. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        ncon : integer value that defines the number of reaches connected to the reach.
        ustrf : real value that defines the fraction of upstream flow from each upstream reach that is applied as upstream inflow to the reach. The sum of all ustrf values for all reaches connected to the same upstream reach must be equal to one and ustrf must be greater than or equal to zero.
        ndv : integer value that defines the number of downstream diversions for the reach.
        aux : represents the values of the auxiliary variables for each stream reach. The values of auxiliary variables must be present for each stream reach. The values must be specified in the order of the auxiliary variables specified in the OPTIONS block. If the package supports time series and the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        boundname : name of the stream reach cell. boundname is an ASCII character variable that can contain as many as 40 characters. If boundname contains spaces in it, then the entire name must be enclosed within single quotes.
    reach_connectivityrecarray : [(rno : integer), (ic : integer)]
        rno : integer value that defines the reach number associated with the specified CONNECTIONDATA data on the line. rno must be greater than zero and less than or equal to NREACHES. Reach connection information must be specified for every reach or the program will terminate with an error. The program will also terminate with an error if connection information for a reach is specified more than once.
        ic : integer value that defines the reach number of the reach connected to the current reach and whether it is connected to the upstream or downstream end of the reach. Negative ic numbers indicate connected reaches are connected to the downstream end of the current reach. Positive ic numbers indicate connected reaches are connected to the upstream end of the current reach. The absolute value of ic must be greater than zero and less than or equal to NREACHES.
    reach_diversionsrecarray : [(rno : integer), (idv : integer), (iconr : integer), (cprior : string)]
        rno : integer value that defines the reach number associated with the specified DIVERSIONS data on the line. rno must be greater than zero and less than or equal to NREACHES. Reach diversion information must be specified for every reach with a ndv value greater than 0 or the program will terminate with an error. The program will also terminate with an error if diversion information for a given reach diversion is specified more than once.
        idv : integer value that defines the downstream diversion number for the diversion for reach rno. idv must be greater than zero and less than or equal to ndv for reach rno.
        iconr : integer value that defines the downstream reach that will receive the diverted water. idv must be greater than zero and less than or equal to NREACHES. Furthermore, reach iconr must be a downstream connection for reach rno.
        cprior : character string value that defines the the prioritization system for the diversion, such as when insufficient water is available to meet all diversion stipulations, and is used in conjunction with the value of flow value specified in the STRESS\_PERIOD\_DATA section. Available diversion options include: (1) cprior = `FRACTION', then the amount of the diversion is computed as a fraction of the streamflow leaving reach rno ($Q_{DS$); in this case, 0.0 $\le$ divflow $\le$ 1.0. (2) cprior = `EXCESS', a diversion is made only if $Q_{DS$ for reach rno exceeds the value of divflow. If this occurs, then the quantity of water diverted is the excess flow ($Q_{DS -$ divflow) and $Q_{DS$ from reach rno is set equal to divflow. This represents a flood-control type of diversion, as described by Danskin and Hanson (2002). (3) cprior = `THRESHOLD', then if $Q_{DS$ in reach rno is less than the specified diversion flow (divflow), no water is diverted from reach rno. If $Q_{DS$ in reach rno is greater than or equal to (divflow), (divflow) is diverted and $Q_{DS$ is set to the remainder ($Q_{DS -$ divflow)). This approach assumes that once flow in the stream is sufficiently low, diversions from the stream cease, and is the `priority' algorithm that originally was programmed into the STR1 Package (Prudic, 1989). (4) cprior = `UPTO' -- if $Q_{DS$ in reach rno is greater than or equal to the specified diversion flow (divflow), $Q_{DS$ is reduced by divflow. If $Q_{DS$ in reach rno is less than (divflow), divflow is set to $Q_{DS$ and there will be no flow available for reaches connected to downstream end of reach rno.
    reachperiodrecarray : [(rno : integer), (sfrsetting : keystring)]
        rno : integer value that defines the reach number associated with the specified PERIOD data on the line. rno must be greater than zero and less than or equal to NREACHES.
        sfrsetting : line of information that is parsed into a keyword and values. Keyword values that can be used to start the sfrsetting string include: STATUS, MANNING, STAGE, INFLOW, RAINFALL, EVAPORATION, RUNOFF, DIVERSION, UPSTREAM\_FRACTION, and AUXILIARY.
    diversion : (diversion : keyword)
        keyword to indicate diversion record.
    idv : (idv : integer)
        diversion number.
    divrate : (divrate : double)
        real or character value that defines the volumetric diversion (divflow) rate for the streamflow routing reach. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
    auxname : (auxname : string)
        name for the auxiliary variable to be assigned auxval. auxname must match one of the auxiliary variable names defined in the OPTIONS block. If auxname does not match one of the auxiliary variable names defined in the OPTIONS block the data are ignored.
    auxval : (auxval : double)
        value for the auxiliary variable. If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.

    """
    auxiliary = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'options', 'auxiliary'))
    stage_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'options', 'stage_filerecord'))
    budget_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'options', 'budget_filerecord'))
    ts_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'options', 'ts_filerecord'))
    obs_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'options', 'obs_filerecord'))
    sfrrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'packagedata', 'sfrrecarray'))
    reach_connectivityrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'connectiondata', 'reach_connectivityrecarray'))
    reach_diversionsrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'diversions', 'reach_diversionsrecarray'))
    reachperiodrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'sfr', 'period', 'reachperiodrecarray'))
    package_abbr = "gwfsfr"

    def __init__(self, model, add_to_package_list=True, auxiliary=None, boundnames=None, print_input=None,
                 print_stage=None, print_flows=None, save_flows=None, stage_filerecord=None,
                 budget_filerecord=None, ts_filerecord=None, obs_filerecord=None, mover=None,
                 maximum_iterations=None, maximum_depth_change=None, unit_conversion=None,
                 nreaches=None, sfrrecarray=None, reach_connectivityrecarray=None,
                 reach_diversionsrecarray=None, reachperiodrecarray=None, diversion=None, idv=None,
                 divrate=None, auxname=None, auxval=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfsfr, self).__init__(model, "sfr", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)

        self.boundnames = self.build_mfdata("boundnames", boundnames)

        self.print_input = self.build_mfdata("print_input", print_input)

        self.print_stage = self.build_mfdata("print_stage", print_stage)

        self.print_flows = self.build_mfdata("print_flows", print_flows)

        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.stage_filerecord = self.build_mfdata("stage_filerecord", stage_filerecord)

        self.budget_filerecord = self.build_mfdata("budget_filerecord", budget_filerecord)

        self.ts_filerecord = self.build_mfdata("ts_filerecord", ts_filerecord)

        self.obs_filerecord = self.build_mfdata("obs_filerecord", obs_filerecord)

        self.mover = self.build_mfdata("mover", mover)

        self.maximum_iterations = self.build_mfdata("maximum_iterations", maximum_iterations)

        self.maximum_depth_change = self.build_mfdata("maximum_depth_change", maximum_depth_change)

        self.unit_conversion = self.build_mfdata("unit_conversion", unit_conversion)

        self.nreaches = self.build_mfdata("nreaches", nreaches)

        self.sfrrecarray = self.build_mfdata("sfrrecarray", sfrrecarray)

        self.reach_connectivityrecarray = self.build_mfdata("reach_connectivityrecarray", reach_connectivityrecarray)

        self.reach_diversionsrecarray = self.build_mfdata("reach_diversionsrecarray", reach_diversionsrecarray)

        self.reachperiodrecarray = self.build_mfdata("reachperiodrecarray", reachperiodrecarray)

        self.diversion = self.build_mfdata("diversion", diversion)

        self.idv = self.build_mfdata("idv", idv)

        self.divrate = self.build_mfdata("divrate", divrate)

        self.auxname = self.build_mfdata("auxname", auxname)

        self.auxval = self.build_mfdata("auxval", auxval)


