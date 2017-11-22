from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfwel(mfpackage.MFPackage):
    """
    ModflowGwfwel defines a wel package within a gwf6 model.

    Attributes
    ----------
    auxiliary : [(auxiliary : string)]
        defines an array of one or more auxiliary variable names. There is no limit on the number of auxiliary variables that can be provided on this line; however, lists of information provided in subsequent blocks must have a column of data for each auxiliary variable name defined here. The number of auxiliary variables detected on this line determines the value for naux. Comments cannot be provided anywhere on this line as they will be interpreted as auxiliary variable names. Auxiliary variables may not be used by the package, but they will be available for use by other parts of the program. The program will terminate with an error if auxiliary variables are specified on more than one line in the options block.
    auxmultname : (auxmultname : string)
        name of auxiliary variable to be used as multiplier of well flow rate.
    boundnames : (boundnames : keyword)
        keyword to indicate that boundary names may be provided with the list of well cells.
    print_input : (print_input : keyword)
        keyword to indicate that the list of well information will be written to the listing file immediately after it is read.
    print_flows : (print_flows : keyword)
        keyword to indicate that the list of well flow rates will be printed to the listing file for every stress period time step in which ``BUDGET PRINT'' is specified in Output Control. If there is no Output Control option and PRINT\_FLOWS is specified, then flow rates are printed for the last time step of each stress period.
    save_flows : (save_flows : keyword)
        keyword to indicate that well flow terms will be written to the file specified with ``BUDGET FILEOUT'' in Output Control.
    auto_flow_reduce : (auto_flow_reduce : double)
        keyword and real value that defines the fraction of the cell thickness used as an interval for smoothly adjusting negative pumping rates to 0 in cells with head values less than or equal to the bottom of the cell. Negative pumping rates are adjusted to 0 or a smaller negative value when the head in the cell is equal to or less than the calculated interval above the cell bottom. auto\_flow\_reduce is set to 0.1 if the specified value is less than or equal to zero. By default, negative pumping rates are not reduced during a simulation.
    ts_filerecord : [(ts6 : keyword), (filein : keyword), (ts6_filename : string)]
        ts6 : keyword to specify that record corresponds to a time-series file.
        filein : keyword to specify that an input filename is expected next.
        ts6_filename : defines a time-series file defining time series that can be used to assign time-varying values. See the ``Time-Variable Input'' section for instructions on using the time-series capability.
    obs_filerecord : [(obs6 : keyword), (filein : keyword), (obs6_filename : string)]
        filein : keyword to specify that an input filename is expected next.
        obs6 : keyword to specify that record corresponds to an observations file.
        obs6_filename : name of input file to define observations for the Well package. See the ``Observation utility'' section for instructions for preparing observation input files. Table obstype lists observation type(s) supported by the Well package.
    mover : (mover : keyword)
        keyword to indicate that this instance of the Well Package can be used with the Water Mover (MVR) Package. When the MOVER option is specified, additional memory is allocated within the package to store the available, provided, and received water.
    maxbound : (maxbound : integer)
        integer value specifying the maximum number of wells cells that will be specified for use during any stress period.
    periodrecarray : [(cellid : integer), (q : double), (aux : double), (boundname : string)]
        cellid : is the cell identifier, and depends on the type of grid that is used for the simulation. For a structured grid that uses the DIS input file, cellid is the layer, row, and column. For a grid that uses the DISV input file, cellid is the layer and cell2d number. If the model uses the unstructured discretization (DISU) input file, then cellid is the node number for the cell.
        q : is the volumetric well rate. A positive value indicates recharge (injection) and a negative value indicates discharge (extraction). If the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        aux : represents the values of the auxiliary variables for each well. The values of auxiliary variables must be present for each well. The values must be specified in the order of the auxiliary variables specified in the OPTIONS block. If the package supports time series and the Options block includes a TIMESERIESFILE entry (see the ``Time-Variable Input'' section), values can be obtained from a time series by entering the time-series name in place of a numeric value.
        boundname : name of the well cell. boundname is an ASCII character variable that can contain as many as 40 characters. If boundname contains spaces in it, then the entire name must be enclosed within single quotes.

    """
    auxiliary = mfdatautil.ListTemplateGenerator(('gwf6', 'wel', 'options', 'auxiliary'))
    ts_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'wel', 'options', 'ts_filerecord'))
    obs_filerecord = mfdatautil.ListTemplateGenerator(('gwf6', 'wel', 'options', 'obs_filerecord'))
    periodrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'wel', 'period', 'periodrecarray'))
    package_abbr = "gwfwel"

    def __init__(self, model, add_to_package_list=True, auxiliary=None, auxmultname=None, boundnames=None,
                 print_input=None, print_flows=None, save_flows=None, auto_flow_reduce=None,
                 ts_filerecord=None, obs_filerecord=None, mover=None, maxbound=None,
                 periodrecarray=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfwel, self).__init__(model, "wel", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)

        self.auxmultname = self.build_mfdata("auxmultname", auxmultname)

        self.boundnames = self.build_mfdata("boundnames", boundnames)

        self.print_input = self.build_mfdata("print_input", print_input)

        self.print_flows = self.build_mfdata("print_flows", print_flows)

        self.save_flows = self.build_mfdata("save_flows", save_flows)

        self.auto_flow_reduce = self.build_mfdata("auto_flow_reduce", auto_flow_reduce)

        self.ts_filerecord = self.build_mfdata("ts_filerecord", ts_filerecord)

        self.obs_filerecord = self.build_mfdata("obs_filerecord", obs_filerecord)

        self.mover = self.build_mfdata("mover", mover)

        self.maxbound = self.build_mfdata("maxbound", maxbound)

        self.periodrecarray = self.build_mfdata("periodrecarray", periodrecarray)


