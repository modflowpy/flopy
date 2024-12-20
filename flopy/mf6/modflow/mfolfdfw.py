# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowOlfdfw(mfpackage.MFPackage):
    """
    ModflowOlfdfw defines a dfw package within a olf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    central_in_space : boolean
        * central_in_space (boolean) keyword to indicate conductance should be
          calculated using central-in-space weighting instead of the default
          upstream weighting approach. This option should be used with caution
          as it does not work well unless all of the stream reaches are
          saturated. With this option, there is no way for water to flow into a
          dry reach from connected reaches.
    length_conversion : double
        * length_conversion (double) real value that is used to convert user-
          specified Manning's roughness coefficients from meters to model
          length units. LENGTH_CONVERSION should be set to 3.28081, 1.0, and
          100.0 when using length units (LENGTH_UNITS) of feet, meters, or
          centimeters in the simulation, respectively. LENGTH_CONVERSION does
          not need to be specified if LENGTH_UNITS are meters.
    time_conversion : double
        * time_conversion (double) real value that is used to convert user-
          specified Manning's roughness coefficients from seconds to model time
          units. TIME_CONVERSION should be set to 1.0, 60.0, 3,600.0, 86,400.0,
          and 31,557,600.0 when using time units (TIME_UNITS) of seconds,
          minutes, hours, days, or years in the simulation, respectively.
          TIME_CONVERSION does not need to be specified if TIME_UNITS are
          seconds.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that budget flow terms will
          be written to the file specified with "BUDGET SAVE FILE" in Output
          Control.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that calculated flows
          between cells will be printed to the listing file for every stress
          period time step in which "BUDGET PRINT" is specified in Output
          Control. If there is no Output Control option and "PRINT_FLOWS" is
          specified, then flow rates are printed for the last time step of each
          stress period. This option can produce extremely large list files
          because all cell-by-cell flows are printed. It should only be used
          with the DFW Package for models that have a small number of cells.
    save_velocity : boolean
        * save_velocity (boolean) keyword to indicate that x, y, and z
          components of velocity will be calculated at cell centers and written
          to the budget file, which is specified with "BUDGET SAVE FILE" in
          Output Control. If this option is activated, then additional
          information may be required in the discretization packages and the
          GWF Exchange package (if GWF models are coupled). Specifically,
          ANGLDEGX must be specified in the CONNECTIONDATA block of the DISU
          Package; ANGLDEGX must also be specified for the GWF Exchange as an
          auxiliary variable.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    export_array_ascii : boolean
        * export_array_ascii (boolean) keyword that specifies input griddata
          arrays should be written to layered ascii output files.
    dev_swr_conductance : boolean
        * dev_swr_conductance (boolean) use the conductance formulation in the
          Surface Water Routing (SWR) Process for MODFLOW-2005.
    manningsn : [double]
        * manningsn (double) mannings roughness coefficient
    idcxs : [integer]
        * idcxs (integer) integer value indication the cross section identifier
          in the Cross Section Package that applies to the reach. If not
          provided then reach will be treated as hydraulically wide. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    obs_filerecord = ListTemplateGenerator(('olf6', 'dfw', 'options',
                                            'obs_filerecord'))
    manningsn = ArrayTemplateGenerator(('olf6', 'dfw', 'griddata',
                                        'manningsn'))
    idcxs = ArrayTemplateGenerator(('olf6', 'dfw', 'griddata', 'idcxs'))
    package_abbr = "olfdfw"
    _package_type = "dfw"
    dfn_file_name = "olf-dfw.dfn"

    dfn = [
           ["header", ],
           ["block options", "name central_in_space", "type keyword",
            "reader urword", "optional true", "mf6internal icentral"],
           ["block options", "name length_conversion",
            "type double precision", "reader urword", "optional true",
            "mf6internal lengthconv"],
           ["block options", "name time_conversion",
            "type double precision", "reader urword", "optional true",
            "mf6internal timeconv"],
           ["block options", "name save_flows", "type keyword",
            "reader urword", "optional true", "mf6internal ipakcb"],
           ["block options", "name print_flows", "type keyword",
            "reader urword", "optional true", "mf6internal iprflow"],
           ["block options", "name save_velocity", "type keyword",
            "reader urword", "optional true", "mf6internal isavvelocity"],
           ["block options", "name obs_filerecord",
            "type record obs6 filein obs6_filename", "shape", "reader urword",
            "tagged true", "optional true", "construct_package obs",
            "construct_data continuous", "parameter_name observations"],
           ["block options", "name obs6", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name filein", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name obs6_filename", "type string",
            "preserve_case true", "in_record true", "tagged false",
            "reader urword", "optional false"],
           ["block options", "name export_array_ascii", "type keyword",
            "reader urword", "optional true", "mf6internal export_ascii"],
           ["block options", "name dev_swr_conductance", "type keyword",
            "reader urword", "optional true", "mf6internal iswrcond"],
           ["block griddata", "name manningsn", "type double precision",
            "shape (nodes)", "valid", "reader readarray", "layered false",
            "optional"],
           ["block griddata", "name idcxs", "type integer", "shape (nodes)",
            "valid", "reader readarray", "layered false", "optional true",
            "numeric_index true"]]

    def __init__(self, model, loading_package=False, central_in_space=None,
                 length_conversion=None, time_conversion=None, save_flows=None,
                 print_flows=None, save_velocity=None, observations=None,
                 export_array_ascii=None, dev_swr_conductance=None,
                 manningsn=None, idcxs=None, filename=None, pname=None,
                 **kwargs):
        super().__init__(model, "dfw", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.central_in_space = self.build_mfdata("central_in_space",
                                                  central_in_space)
        self.length_conversion = self.build_mfdata("length_conversion",
                                                   length_conversion)
        self.time_conversion = self.build_mfdata("time_conversion",
                                                 time_conversion)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_velocity = self.build_mfdata("save_velocity", save_velocity)
        self._obs_filerecord = self.build_mfdata("obs_filerecord",
                                                 None)
        self._obs_package = self.build_child_package("obs", observations,
                                                     "continuous",
                                                     self._obs_filerecord)
        self.export_array_ascii = self.build_mfdata("export_array_ascii",
                                                    export_array_ascii)
        self.dev_swr_conductance = self.build_mfdata("dev_swr_conductance",
                                                     dev_swr_conductance)
        self.manningsn = self.build_mfdata("manningsn", manningsn)
        self.idcxs = self.build_mfdata("idcxs", idcxs)
        self._init_complete = True
