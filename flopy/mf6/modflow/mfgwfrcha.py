from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfrcha(mfpackage.MFPackage):
    """
    ModflowGwfrcha defines a rcha package within a gwf6 model.

    Attributes
    ----------
    readasarrays : (readasarrays : boolean)
        readasarrays : indicates that array-based input will be used for the
          Recharge Package. This keyword must be specified to use array-based
          input.
    fixed_cell : (fixed_cell : boolean)
        fixed_cell : indicates that recharge will not be reassigned to a cell
          underlying the cell specified in the list if the specified cell is
          inactive.
    auxiliary : [(auxiliary : string)]
        auxiliary : defines an array of one or more auxiliary variable names.
          There is no limit on the number of auxiliary variables that can be
          provided on this line; however, lists of information provided in
          subsequent blocks must have a column of data for each auxiliary
          variable name defined here. The number of auxiliary variables
          detected on this line determines the value for naux. Comments cannot
          be provided anywhere on this line as they will be interpreted as
          auxiliary variable names. Auxiliary variables may not be used by the
          package, but they will be available for use by other parts of the
          program. The program will terminate with an error if auxiliary
          variables are specified on more than one line in the options block.
    auxmultname : (auxmultname : string)
        auxmultname : name of auxiliary variable to be used as multiplier of
          recharge.
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of recharge information
          will be written to the listing file immediately after it is read.
    print_flows : (print_flows : boolean)
        print_flows : keyword to indicate that the list of recharge flow rates
          will be printed to the listing file for every stress period time step
          in which ``BUDGET PRINT'' is specified in Output Control. If there is
          no Output Control option and PRINT\_FLOWS is specified, then flow
          rates are printed for the last time step of each stress period.
    save_flows : (save_flows : boolean)
        save_flows : keyword to indicate that recharge flow terms will be
          written to the file specified with ``BUDGET FILEOUT'' in Output
          Control.
    tas_filerecord : [(tas6_filename : string)]
        tas6_filename : defines a time-array-series file defining a time-array
          series that can be used to assign time-varying values. See the Time-
          Variable Input section for instructions on using the time-array
          series capability.
    obs_filerecord : [(obs6_filename : string)]
        obs6_filename : name of input file to define observations for the
          Recharge package. See the ``Observation utility'' section for
          instructions for preparing observation input files. Table
          obstype lists observation type(s) supported by the Recharge
          package.
    irch : [(irch : integer)]
        irch : irch is the layer number that defines the layer in each
          vertical column where recharge is applied. If irch is
          omitted, recharge by default is applied to cells in layer 1.
          irch can only be used if READASARRAYS is specified in the
          OPTIONS block.
    recharge : [(recharge : double)]
        recharge : is the recharge flux rate ($LT^{-1$). This rate is
          multiplied inside the program by the surface area of the cell to
          calculate the volumetric recharge rate. The recharge array may be
          defined by a time-array series (see the "Using Time-Array Series in a
          Package" section).
    aux : [(aux : double)]
        aux : is an array of values for auxiliary variable aux(iaux), where
          iaux is a value from 1 to naux, and aux(iaux) must be listed as part
          of the auxiliary variables. A separate array can be specified for
          each auxiliary variable. If an array is not specified for an
          auxiliary variable, then a value of zero is assigned. If the value
          specified here for the auxiliary variable is the same as auxmultname,
          then the recharge array will be multiplied by this array.

    """
    auxiliary = ListTemplateGenerator(('gwf6', 'rcha', 'options', 
                                       'auxiliary'))
    tas_filerecord = ListTemplateGenerator(('gwf6', 'rcha', 'options', 
                                            'tas_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwf6', 'rcha', 'options', 
                                            'obs_filerecord'))
    irch = ArrayTemplateGenerator(('gwf6', 'rcha', 'period', 'irch'))
    recharge = ArrayTemplateGenerator(('gwf6', 'rcha', 'period', 
                                       'recharge'))
    aux = ArrayTemplateGenerator(('gwf6', 'rcha', 'period', 'aux'))
    package_abbr = "gwfrcha"

    def __init__(self, model, add_to_package_list=True, readasarrays=None,
                 fixed_cell=None, auxiliary=None, auxmultname=None,
                 print_input=None, print_flows=None, save_flows=None,
                 tas_filerecord=None, obs_filerecord=None, irch=None,
                 recharge=None, aux=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfrcha, self).__init__(model, "rcha", fname, pname,
                                             add_to_package_list, parent_file)        

        # set up variables
        self.readasarrays = self.build_mfdata("readasarrays",  readasarrays)
        self.fixed_cell = self.build_mfdata("fixed_cell",  fixed_cell)
        self.auxiliary = self.build_mfdata("auxiliary",  auxiliary)
        self.auxmultname = self.build_mfdata("auxmultname",  auxmultname)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.tas_filerecord = self.build_mfdata("tas_filerecord", 
                                                tas_filerecord)
        self.obs_filerecord = self.build_mfdata("obs_filerecord", 
                                                obs_filerecord)
        self.irch = self.build_mfdata("irch",  irch)
        self.recharge = self.build_mfdata("recharge",  recharge)
        self.aux = self.build_mfdata("aux",  aux)
