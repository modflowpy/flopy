# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfuzf(mfpackage.MFPackage):
    """
    ModflowGwfuzf defines a uzf package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
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
          multiplier of GWF cell area used by UZF cell.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of UZF cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of UZF
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of UZF flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that UZF flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    ts_filerecord : [ts6_filename]
        * ts6_filename (string) defines a time-series file defining time series
          that can be used to assign time-varying values. See the "Time-
          Variable Input" section for instructions on using the time-series
          capability.
    obs_filerecord : [obs6_filename]
        * obs6_filename (string) name of input file to define observations for
          the UZF package. See the "Observation utility" section for
          instructions for preparing observation input files. Table
          reftable:obstype lists observation type(s) supported by the UZF
          package.
    mover : boolean
        * mover (boolean) keyword to indicate that this instance of the UZF
          Package can be used with the Water Mover (MVR) Package. When the
          MOVER option is specified, additional memory is allocated within the
          package to store the available, provided, and received water.
    simulate_et : boolean
        * simulate_et (boolean) keyword specifying that ET in the unsaturated
          (UZF) and saturated zones (GWF) will be simulated. ET can be
          simulated in the UZF cell and not the GWF cell by emitting keywords
          LINEAR_GWET and SQUARE_GWET.
    linear_gwet : boolean
        * linear_gwet (boolean) keyword specifying that groundwater ET will be
          simulated using the original ET formulation of MODFLOW-2005.
    square_gwet : boolean
        * square_gwet (boolean) keyword specifying that groundwater ET will be
          simulated by assuming a constant ET rate for groundwater levels
          between land surface (TOP) and land surface minus the ET extinction
          depth (TOP-EXTDP). Groundwater ET is smoothly reduced from the PET
          rate to zero over a nominal interval at TOP-EXTDP.
    simulate_gwseep : boolean
        * simulate_gwseep (boolean) keyword specifying that groundwater
          discharge (GWSEEP) to land surface will be simulated. Groundwater
          discharge is nonzero when groundwater head is greater than land
          surface.
    unsat_etwc : boolean
        * unsat_etwc (boolean) keyword specifying that ET in the unsaturated
          zone will be simulated as a function of the specified PET rate while
          the water content (THETA) is greater than the ET extinction water
          content (EXTWC).
    unsat_etae : boolean
        * unsat_etae (boolean) keyword specifying that ET in the unsaturated
          zone will be simulated simulated using a capillary pressure based
          formulation. Capillary pressure is calculated using the Brooks-Corey
          retention function.
    nuzfcells : integer
        * nuzfcells (integer) is the number of UZF cells. More than 1 UZF cell
          can be assigned to a GWF cell; however, only 1 GWF cell can be
          assigned to a single UZF cell. If the MULTILAYER option is used then
          UZF cells can be assigned to GWF cells below (in deeper layers than)
          the upper most active GWF cells.
    ntrailwaves : integer
        * ntrailwaves (integer) is the number of trailing waves. NTRAILWAVES
          has a default value of 7 and can be increased to lower mass balance
          error in the unsaturated zone.
    nwavesets : integer
        * nwavesets (integer) is the number of UZF cells specified. NWAVSETS
          has a default value of 40 and can be increased if more waves are
          required to resolve variations in water content within the
          unsaturated zone.
    packagedata : [iuzno, cellid, landflag, ivertcon, surfdep, vks, thtr, thts,
      thti, eps, boundname]
        * iuzno (integer) integer value that defines the UZF cell number
          associated with the specified PACKAGEDATA data on the line. IUZNO
          must be greater than zero and less than or equal to NUZFCELLS. UZF
          information must be specified for every UZF cell or the program will
          terminate with an error. The program will also terminate with an
          error if information for a UZF cell is specified more than once.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell.
        * landflag (integer) integer value set to one for land surface cells
          indicating that boundary conditions can be applied and data can be
          specified in the PERIOD block. A value of 0 specifies a non-land
          surface cell.
        * ivertcon (integer) integer value set to specify underlying UZF cell
          that receives water flowing to bottom of cell. If unsaturated zone
          flow reaches water table before the cell bottom then water is added
          to GWF cell instead of flowing to underlying UZF cell. A value of 0
          indicates the UZF cell is not connected to an underlying UZF cell.
        * surfdep (double) is the surface depression depth of the UZF cell.
        * vks (double) is the vertical saturated hydraulic conductivity of the
          UZF cell.
        * thtr (double) is the residual (irreducible) water content of the UZF
          cell.
        * thts (double) is the saturated water content of the UZF cell.
        * thti (double) is the initial water content of the UZF cell.
        * eps (double) is the epsilon exponent of the UZF cell.
        * boundname (string) name of the UZF cell cell. BOUNDNAME is an ASCII
          character variable that can contain as many as 40 characters. If
          BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    perioddata : [iuzno, finf, pet, extdp, extwc, ha, hroot, rootact, aux]
        * iuzno (integer) integer value that defines the UZF cell number
          associated with the specified PERIOD data on the line.
        * finf (string) real or character value that defines the applied
          infiltration rate of the UZF cell (:math:`LT^{-1}`). If the Options
          block includes a TIMESERIESFILE entry (see the "Time-Variable Input"
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        * pet (string) real or character value that defines the potential
          evapotranspiration rate of the UZF cell and specified GWF cell.
          Evapotranspiration is first removed from the unsaturated zone and any
          remaining potential evapotranspiration is applied to the saturated
          zone. If IVERTCON is greater than zero then residual potential
          evapotranspiration not satisfied in the UZF cell is applied to the
          underlying UZF and GWF cells. PET is always specified, but is only
          used if SIMULATE_ET is specified in the OPTIONS block. If the Options
          block includes a TIMESERIESFILE entry (see the "Time-Variable Input"
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        * extdp (string) real or character value that defines the
          evapotranspiration extinction depth of the UZF cell. If IVERTCON is
          greater than zero and EXTDP extends below the GWF cell bottom then
          remaining potential evapotranspiration is applied to the underlying
          UZF and GWF cells. EXTDP is always specified, but is only used if
          SIMULATE_ET is specified in the OPTIONS block. If the Options block
          includes a TIMESERIESFILE entry (see the "Time-Variable Input"
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        * extwc (string) real or character value that defines the
          evapotranspiration extinction water content of the UZF cell. EXTWC is
          always specified, but is only used if SIMULATE_ET and UNSAT_ETWC are
          specified in the OPTIONS block. If the Options block includes a
          TIMESERIESFILE entry (see the "Time-Variable Input" section), values
          can be obtained from a time series by entering the time-series name
          in place of a numeric value.
        * ha (string) real or character value that defines the air entry
          potential (head) of the UZF cell. HA is always specified, but is only
          used if SIMULATE_ET and UNSAT_ETAE are specified in the OPTIONS
          block. If the Options block includes a TIMESERIESFILE entry (see the
          "Time-Variable Input" section), values can be obtained from a time
          series by entering the time-series name in place of a numeric value.
        * hroot (string) real or character value that defines the root
          potential (head) of the UZF cell. HROOT is always specified, but is
          only used if SIMULATE_ET and UNSAT_ETAE are specified in the OPTIONS
          block. If the Options block includes a TIMESERIESFILE entry (see the
          "Time-Variable Input" section), values can be obtained from a time
          series by entering the time-series name in place of a numeric value.
        * rootact (string) real or character value that defines the root
          activity function of the UZF cell. ROOTACT is the length of roots in
          a given volume of soil divided by that volume. Values range from 0 to
          about 3 :math:`cm^{-2}`, depending on the plant community and its
          stage of development. ROOTACT is always specified, but is only used
          if SIMULATE\_ET and UNSAT\_ETAE are specified in the OPTIONS block.
          If the Options block includes a TIMESERIESFILE entry (see the "Time-
          Variable Input" section), values can be obtained from a time series
          by entering the time-series name in place of a numeric value.
        * aux (double) represents the values of the auxiliary variables for
          each UZF. The values of auxiliary variables must be present for each
          UZF. The values must be specified in the order of the auxiliary
          variables specified in the OPTIONS block. If the package supports
          time series and the Options block includes a TIMESERIESFILE entry
          (see the "Time-Variable Input" section), values can be obtained from
          a time series by entering the time-series name in place of a numeric
          value.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    auxiliary = ListTemplateGenerator(('gwf6', 'uzf', 'options', 
                                       'auxiliary'))
    budget_filerecord = ListTemplateGenerator(('gwf6', 'uzf', 'options', 
                                               'budget_filerecord'))
    ts_filerecord = ListTemplateGenerator(('gwf6', 'uzf', 'options', 
                                           'ts_filerecord'))
    obs_filerecord = ListTemplateGenerator(('gwf6', 'uzf', 'options', 
                                            'obs_filerecord'))
    packagedata = ListTemplateGenerator(('gwf6', 'uzf', 'packagedata', 
                                         'packagedata'))
    perioddata = ListTemplateGenerator(('gwf6', 'uzf', 'period', 
                                        'perioddata'))
    package_abbr = "gwfuzf"
    package_type = "uzf"
    dfn_file_name = "gwf-uzf.dfn"

    dfn = [["block options", "name auxiliary", "type string", 
            "shape (naux)", "reader urword", "optional true"],
           ["block options", "name auxmultname", "type string", "shape", 
            "reader urword", "optional true"],
           ["block options", "name boundnames", "type keyword", "shape", 
            "reader urword", "optional true"],
           ["block options", "name print_input", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name print_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name save_flows", "type keyword", 
            "reader urword", "optional true"],
           ["block options", "name budget_filerecord", 
            "type record budget fileout budgetfile", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name budget", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name fileout", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name budgetfile", "preserve_case true", 
            "type string", "shape", "in_record true", "reader urword", 
            "tagged false", "optional false"],
           ["block options", "name ts_filerecord", 
            "type record ts6 filein ts6_filename", "shape", "reader urword", 
            "tagged true", "optional true"],
           ["block options", "name ts6", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name filein", "type keyword", "shape", 
            "in_record true", "reader urword", "tagged true", 
            "optional false"],
           ["block options", "name ts6_filename", "type string", 
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
           ["block options", "name mover", "type keyword", "tagged true", 
            "reader urword", "optional true"],
           ["block options", "name simulate_et", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block options", "name linear_gwet", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block options", "name square_gwet", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block options", "name simulate_gwseep", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block options", "name unsat_etwc", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block options", "name unsat_etae", "type keyword", 
            "tagged true", "reader urword", "optional true"],
           ["block dimensions", "name nuzfcells", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name ntrailwaves", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name nwavesets", "type integer", 
            "reader urword", "optional false"],
           ["block packagedata", "name packagedata", 
            "type recarray iuzno cellid landflag ivertcon surfdep vks thtr " 
            "thts thti eps boundname", 
            "shape (nuzfcells)", "reader urword"],
           ["block packagedata", "name iuzno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block packagedata", "name cellid", "type integer", 
            "shape (ncelldim)", "tagged false", "in_record true", 
            "reader urword"],
           ["block packagedata", "name landflag", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name ivertcon", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block packagedata", "name surfdep", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name vks", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name thtr", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name thts", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name thti", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name eps", "type double precision", 
            "shape", "tagged false", "in_record true", "reader urword"],
           ["block packagedata", "name boundname", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "optional true"],
           ["block period", "name iper", "type integer", 
            "block_variable True", "in_record true", "tagged false", "shape", 
            "valid", "reader urword", "optional false"],
           ["block period", "name perioddata", 
            "type recarray iuzno finf pet extdp extwc ha hroot rootact aux", 
            "shape", "reader urword"],
           ["block period", "name iuzno", "type integer", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "numeric_index true"],
           ["block period", "name finf", "type string", "shape", 
            "tagged false", "in_record true", "time_series true", 
            "reader urword"],
           ["block period", "name pet", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name extdp", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name extwc", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name ha", "type string", "shape", 
            "tagged false", "in_record true", "time_series true", 
            "reader urword"],
           ["block period", "name hroot", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name rootact", "type string", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "time_series true"],
           ["block period", "name aux", "type double precision", 
            "in_record true", "tagged false", "shape (naux)", "reader urword", 
            "time_series true", "optional true"]]

    def __init__(self, model, loading_package=False, auxiliary=None,
                 auxmultname=None, boundnames=None, print_input=None,
                 print_flows=None, save_flows=None, budget_filerecord=None,
                 ts_filerecord=None, obs_filerecord=None, mover=None,
                 simulate_et=None, linear_gwet=None, square_gwet=None,
                 simulate_gwseep=None, unsat_etwc=None, unsat_etae=None,
                 nuzfcells=None, ntrailwaves=None, nwavesets=None,
                 packagedata=None, perioddata=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfuzf, self).__init__(model, "uzf", fname, pname,
                                            loading_package, parent_file)        

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary",  auxiliary)
        self.auxmultname = self.build_mfdata("auxmultname",  auxmultname)
        self.boundnames = self.build_mfdata("boundnames",  boundnames)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.print_flows = self.build_mfdata("print_flows",  print_flows)
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.budget_filerecord = self.build_mfdata("budget_filerecord", 
                                                   budget_filerecord)
        self.ts_filerecord = self.build_mfdata("ts_filerecord",  ts_filerecord)
        self.obs_filerecord = self.build_mfdata("obs_filerecord", 
                                                obs_filerecord)
        self.mover = self.build_mfdata("mover",  mover)
        self.simulate_et = self.build_mfdata("simulate_et",  simulate_et)
        self.linear_gwet = self.build_mfdata("linear_gwet",  linear_gwet)
        self.square_gwet = self.build_mfdata("square_gwet",  square_gwet)
        self.simulate_gwseep = self.build_mfdata("simulate_gwseep", 
                                                 simulate_gwseep)
        self.unsat_etwc = self.build_mfdata("unsat_etwc",  unsat_etwc)
        self.unsat_etae = self.build_mfdata("unsat_etae",  unsat_etae)
        self.nuzfcells = self.build_mfdata("nuzfcells",  nuzfcells)
        self.ntrailwaves = self.build_mfdata("ntrailwaves",  ntrailwaves)
        self.nwavesets = self.build_mfdata("nwavesets",  nwavesets)
        self.packagedata = self.build_mfdata("packagedata",  packagedata)
        self.perioddata = self.build_mfdata("perioddata",  perioddata)
