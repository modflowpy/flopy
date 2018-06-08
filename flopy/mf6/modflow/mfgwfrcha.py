# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfrcha(mfpackage.MFPackage):
    """
    ModflowGwfrcha defines a rcha package within a gwf6 model.

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
          for the Recharge Package. This keyword must be specified to use
          array-based input.
    fixed_cell : boolean
        * fixed_cell (boolean) indicates that recharge will not be reassigned
          to a cell underlying the cell specified in the list if the specified
          cell is inactive.
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
          multiplier of recharge.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of recharge
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of recharge
          flow rates will be printed to the listing file for every stress
          period time step in which "BUDGET PRINT" is specified in Output
          Control. If there is no Output Control option and "PRINT_FLOWS" is
          specified, then flow rates are printed for the last time step of each
          stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that recharge flow terms
          will be written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    tas_filerecord : [tas6_filename]
        * tas6_filename (string) defines a time-array-series file defining a
          time-array series that can be used to assign time-varying values. See
          the Time-Variable Input section for instructions on using the time-
          array series capability.
    obs_filerecord : [obs6_filename]
        * obs6_filename (string) name of input file to define observations for
          the Recharge package. See the "Observation utility" section for
          instructions for preparing observation input files. Table
          reftable:obstype lists observation type(s) supported by the Recharge
          package.
    irch : [integer]
        * irch (integer) IRCH is the layer number that defines the layer in
          each vertical column where recharge is applied. If IRCH is omitted,
          recharge by default is applied to cells in layer 1. IRCH can only be
          used if READASARRAYS is specified in the OPTIONS block. If IRCH is
          specified, it must be specified as the first variable in the PERIOD
          block or MODFLOW will terminate with an error.
    recharge : [double]
        * recharge (double) is the recharge flux rate (:math:`LT^{-1}`). This
          rate is multiplied inside the program by the surface area of the cell
          to calculate the volumetric recharge rate. The recharge array may be
          defined by a time-array series (see the "Using Time-Array Series in a
          Package" section).
    aux : [double]
        * aux (double) is an array of values for auxiliary variable aux(iaux),
          where iaux is a value from 1 to naux, and aux(iaux) must be listed as
          part of the auxiliary variables. A separate array can be specified
          for each auxiliary variable. If an array is not specified for an
          auxiliary variable, then a value of zero is assigned. If the value
          specified here for the auxiliary variable is the same as auxmultname,
          then the recharge array will be multiplied by this array.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

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
    package_type = "rcha"
    dfn_file_name = "gwf-rcha.dfn"

    dfn = [["block options", "name readasarrays", "type keyword", "shape", 
            "reader urword", "optional false", "default_value True"],
           ["block options", "name fixed_cell", "type keyword", "shape", 
            "reader urword", "optional true"],
           ["block options", "name auxiliary", "type string", 
            "shape (naux)", "reader urword", "optional true"],
           ["block options", "name auxmultname", "type string", "shape", 
            "reader urword", "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name tas_filerecord", 
            "type record tas6 filein tas6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name tas6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name tas6_filename", "type string", 
            "preserve_case true", "in_record true", "reader urword", 
            "optional false", "tagged false"],
           ["block options", "name obs_filerecord", 
            "type record obs6 filein obs6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name obs6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name obs6_filename", "type string", 
            "preserve_case true", "in_record true", "tagged false", 
            "reader urword", "optional false"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name irch", "type integer", 
            "shape (ncol*nrow; ncpl)", "reader readarray", "optional true"],
           ["block period", "name recharge", "type double precision", 
            "shape (ncol*nrow; ncpl)", "reader readarray", 
            "default_value 1.e-3"],
           ["block period", "name aux", "type double precision", 
            "shape (ncol*nrow; ncpl)", "reader readarray", "optional true"]]

    def __init__(self, model, loading_package=False, readasarrays=True,
                 fixed_cell=None, auxiliary=None, auxmultname=None,
                 print_input=None, print_flows=None, save_flows=None,
                 tas_filerecord=None, obs_filerecord=None, irch=None,
                 recharge=1.e-3, aux=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfrcha, self).__init__(model, "rcha", fname, pname,
                                             loading_package, parent_file)        

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
