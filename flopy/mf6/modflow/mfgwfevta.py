from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfevta(mfpackage.MFPackage):
    """
    ModflowGwfevta defines a evta package within a gwf6 model.

    Attributes
    ----------
    readasarrays : (readasarrays : boolean)
        readasarrays : indicates that array-based input will be used for the
          Evapotranspiration Package. This keyword must be specified to use
          array-based input.
    fixed_cell : (fixed_cell : boolean)
        fixed_cell : indicates that evapotranspiration will not be reassigned
          to a cell underlying the cell specified in the list if the specified
          cell is inactive.
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
          evapotranspiration rate.
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of evapotranspiration
          information will be written to the listing file immediately after it
          is read.
    print_flows : (print_flows : boolean)
        print_flows : keyword to indicate that the list of evapotranspiration
          flow rates will be printed to the listing file for every stress
          period time step in which ``BUDGET PRINT'' is specified in Output
          Control. If there is no Output Control option and PRINT\_FLOWS is
          specified, then flow rates are printed for the last time step of each
          stress period.
    save_flows : (save_flows : boolean)
        save_flows : keyword to indicate that evapotranspiration flow terms
          will be written to the file specified with ``BUDGET FILEOUT'' in
          Output Control.
    tas_filerecord : [(tas6_filename : string)]
        tas6_filename : defines a time-array-series file defining a time-array
          series that can be used to assign time-varying values. See the Time-
          Variable Input section for instructions on using the time-array
          series capability.
    obs_filerecord : [(obs6_filename : string)]
        obs6_filename : name of input file to define observations for the
          Evapotranspiration package. See the ``Observation utility'' section
          for instructions for preparing observation input files. Table
          obstype lists observation type(s) supported by the
          Evapotranspiration package.
    ievt : [(ievt : integer)]
        ievt : ievt is the layer number that defines the layer in each
          vertical column where evapotranspiration is applied. If ievt
          is omitted, evapotranspiration by default is applied to cells in
          layer 1.
    surface : [(surface : double)]
        surface : is the elevation of the ET surface ($L$).
    rate : [(rate : double)]
        rate : is the maximum ET flux rate ($LT^{-1$).
    depth : [(depth : double)]
        depth : is the ET extinction depth ($L$).
    aux(iaux) : [(aux(iaux) : double)]
        aux(iaux) : is an array of values for auxiliary variable aux(iaux),
          where iaux is a value from 1 to naux, and aux(iaux) must be listed as
          part of the auxiliary variables. A separate array can be specified
          for each auxiliary variable. If an array is not specified for an
          auxiliary variable, then a value of zero is assigned. If the value
          specified here for the auxiliary variable is the same as auxmultname,
          then the evapotranspiration rate will be multiplied by this array.

    """
    auxiliary = ListTemplateGenerator(('gwf6', 'evta', 'options', 
                                       'auxiliary'))
    tas_filerecord = ListTemplateGenerator(('gwf6', 'evta', 'options', 
                                            'tas_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwf6', 'evta', 'options', 
                                            'obs_filerecord'))
    ievt = ArrayTemplateGenerator(('gwf6', 'evta', 'period', 'ievt'))
    surface = ArrayTemplateGenerator(('gwf6', 'evta', 'period', 
                                      'surface'))
    rate = ArrayTemplateGenerator(('gwf6', 'evta', 'period', 'rate'))
    depth = ArrayTemplateGenerator(('gwf6', 'evta', 'period', 'depth'))
    aux = ArrayTemplateGenerator(('gwf6', 'evta', 'period', 
                                  'aux(iaux)'))
    package_abbr = "gwfevta"

    def __init__(self, model, add_to_package_list=True, readasarrays=None,
                 fixed_cell=None, auxiliary=None, auxmultname=None,
                 print_input=None, print_flows=None, save_flows=None,
                 tas_filerecord=None, obs_filerecord=None, ievt=None,
                 surface=None, rate=None, depth=None, aux=None, fname=None,
                 pname=None, parent_file=None):
        super(ModflowGwfevta, self).__init__(model, "evta", fname, pname,
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
        self.ievt = self.build_mfdata("ievt",  ievt)
        self.surface = self.build_mfdata("surface",  surface)
        self.rate = self.build_mfdata("rate",  rate)
        self.depth = self.build_mfdata("depth",  depth)
        self.aux = self.build_mfdata("aux(iaux)",  aux)
