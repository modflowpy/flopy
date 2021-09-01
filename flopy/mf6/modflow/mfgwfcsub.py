# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfcsub(mfpackage.MFPackage):
    """
    ModflowGwfcsub defines a csub package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of CSUB cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of CSUB
          information will be written to the listing file immediately after it
          is read.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that cell-by-cell flow terms
          will be written to the file specified with "BUDGET SAVE FILE" in
          Output Control.
    gammaw : double
        * gammaw (double) unit weight of water. For freshwater, GAMMAW is
          9806.65 Newtons/cubic meters or 62.48 lb/cubic foot in SI and English
          units, respectively. By default, GAMMAW is 9806.65 Newtons/cubic
          meters.
    beta : double
        * beta (double) compressibility of water. Typical values of BETA are
          4.6512e-10 1/Pa or 2.2270e-8 lb/square foot in SI and English units,
          respectively. By default, BETA is 4.6512e-10 1/Pa.
    head_based : boolean
        * head_based (boolean) keyword to indicate the head-based formulation
          will be used to simulate coarse-grained aquifer materials and no-
          delay and delay interbeds. Specifying HEAD_BASED also specifies the
          INITIAL_PRECONSOLIDATION_HEAD option.
    initial_preconsolidation_head : boolean
        * initial_preconsolidation_head (boolean) keyword to indicate that
          preconsolidation heads will be specified for no-delay and delay
          interbeds in the PACKAGEDATA block. If the
          SPECIFIED_INITIAL_INTERBED_STATE option is specified in the OPTIONS
          block, user-specified preconsolidation heads in the PACKAGEDATA block
          are absolute values. Otherwise, user-specified preconsolidation heads
          in the PACKAGEDATA block are relative to steady-state or initial
          heads.
    ndelaycells : integer
        * ndelaycells (integer) number of nodes used to discretize delay
          interbeds. If not specified, then a default value of 19 is assigned.
    compression_indices : boolean
        * compression_indices (boolean) keyword to indicate that the
          recompression (CR) and compression (CC) indices are specified instead
          of the elastic specific storage (SSE) and inelastic specific storage
          (SSV) coefficients. If not specified, then elastic specific storage
          (SSE) and inelastic specific storage (SSV) coefficients must be
          specified.
    update_material_properties : boolean
        * update_material_properties (boolean) keyword to indicate that the
          thickness and void ratio of coarse-grained and interbed sediments
          (delay and no-delay) will vary during the simulation. If not
          specified, the thickness and void ratio of coarse-grained and
          interbed sediments will not vary during the simulation.
    cell_fraction : boolean
        * cell_fraction (boolean) keyword to indicate that the thickness of
          interbeds will be specified in terms of the fraction of cell
          thickness. If not specified, interbed thicknness must be specified.
    specified_initial_interbed_state : boolean
        * specified_initial_interbed_state (boolean) keyword to indicate that
          absolute preconsolidation stresses (heads) and delay bed heads will
          be specified for interbeds defined in the PACKAGEDATA block. The
          SPECIFIED_INITIAL_INTERBED_STATE option is equivalent to specifying
          the SPECIFIED_INITIAL_PRECONSOLITATION_STRESS and
          SPECIFIED_INITIAL_DELAY_HEAD. If SPECIFIED_INITIAL_INTERBED_STATE is
          not specified then preconsolidation stress (head) and delay bed head
          values specified in the PACKAGEDATA block are relative to simulated
          values of the first stress period if steady-state or initial stresses
          and GWF heads if the first stress period is transient.
    specified_initial_preconsolidation_stress : boolean
        * specified_initial_preconsolidation_stress (boolean) keyword to
          indicate that absolute preconsolidation stresses (heads) will be
          specified for interbeds defined in the PACKAGEDATA block. If
          SPECIFIED_INITIAL_PRECONSOLITATION_STRESS and
          SPECIFIED_INITIAL_INTERBED_STATE are not specified then
          preconsolidation stress (head) values specified in the PACKAGEDATA
          block are relative to simulated values if the first stress period is
          steady-state or initial stresses (heads) if the first stress period
          is transient.
    specified_initial_delay_head : boolean
        * specified_initial_delay_head (boolean) keyword to indicate that
          absolute initial delay bed head will be specified for interbeds
          defined in the PACKAGEDATA block. If SPECIFIED_INITIAL_DELAY_HEAD and
          SPECIFIED_INITIAL_INTERBED_STATE are not specified then delay bed
          head values specified in the PACKAGEDATA block are relative to
          simulated values if the first stress period is steady-state or
          initial GWF heads if the first stress period is transient.
    effective_stress_lag : boolean
        * effective_stress_lag (boolean) keyword to indicate the effective
          stress from the previous time step will be used to calculate specific
          storage values. This option can 1) help with convergence in models
          with thin cells and water table elevations close to land surface; 2)
          is identical to the approach used in the SUBWT package for
          MODFLOW-2005; and 3) is only used if the effective-stress formulation
          is being used. By default, current effective stress values are used
          to calculate specific storage values.
    strainib_filerecord : [interbedstrain_filename]
        * interbedstrain_filename (string) name of the comma-separated-values
          output file to write final interbed strain information.
    straincg_filerecord : [coarsestrain_filename]
        * coarsestrain_filename (string) name of the comma-separated-values
          output file to write final coarse-grained material strain
          information.
    compaction_filerecord : [compaction_filename]
        * compaction_filename (string) name of the binary output file to write
          compaction information.
    fileout : boolean
        * fileout (boolean) keyword to specify that an output filename is
          expected next.
    compaction_elastic_filerecord : [elastic_compaction_filename]
        * elastic_compaction_filename (string) name of the binary output file
          to write elastic interbed compaction information.
    compaction_inelastic_filerecord : [inelastic_compaction_filename]
        * inelastic_compaction_filename (string) name of the binary output file
          to write inelastic interbed compaction information.
    compaction_interbed_filerecord : [interbed_compaction_filename]
        * interbed_compaction_filename (string) name of the binary output file
          to write interbed compaction information.
    compaction_coarse_filerecord : [coarse_compaction_filename]
        * coarse_compaction_filename (string) name of the binary output file to
          write elastic coarse-grained material compaction information.
    zdisplacement_filerecord : [zdisplacement_filename]
        * zdisplacement_filename (string) name of the binary output file to
          write z-displacement information.
    package_convergence_filerecord : [package_convergence_filename]
        * package_convergence_filename (string) name of the comma spaced values
          output file to write package convergence information.
    timeseries : {varname:data} or timeseries data
        * Contains data for the ts package. Data can be stored in a dictionary
          containing data for the ts package with variable names as keys and
          package data as values. Data just for the timeseries variable is also
          acceptable. See ts package documentation for more information.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    ninterbeds : integer
        * ninterbeds (integer) is the number of CSUB interbed systems. More
          than 1 CSUB interbed systems can be assigned to a GWF cell; however,
          only 1 GWF cell can be assigned to a single CSUB interbed system.
    maxsig0 : integer
        * maxsig0 (integer) is the maximum number of cells that can have a
          specified stress offset. More than 1 stress offset can be assigned to
          a GWF cell. By default, MAXSIG0 is 0.
    cg_ske_cr : [double]
        * cg_ske_cr (double) is the initial elastic coarse-grained material
          specific storage or recompression index. The recompression index is
          specified if COMPRESSION_INDICES is specified in the OPTIONS block.
          Specified or calculated elastic coarse-grained material specific
          storage values are not adjusted from initial values if HEAD_BASED is
          specified in the OPTIONS block.
    cg_theta : [double]
        * cg_theta (double) is the initial porosity of coarse-grained
          materials.
    sgm : [double]
        * sgm (double) is the specific gravity of moist or unsaturated
          sediments. If not specified, then a default value of 1.7 is assigned.
    sgs : [double]
        * sgs (double) is the specific gravity of saturated sediments. If not
          specified, then a default value of 2.0 is assigned.
    packagedata : [icsubno, cellid, cdelay, pcs0, thick_frac, rnb, ssv_cc,
      sse_cr, theta, kv, h0, boundname]
        * icsubno (integer) integer value that defines the CSUB interbed number
          associated with the specified PACKAGEDATA data on the line. CSUBNO
          must be greater than zero and less than or equal to NINTERBEDS. CSUB
          information must be specified for every CSUB cell or the program will
          terminate with an error. The program will also terminate with an
          error if information for a CSUB interbed number is specified more
          than once. This argument is an index variable, which means that it
          should be treated as zero-based when working with FloPy and Python.
          Flopy will automatically subtract one when loading index variables
          and add one when writing index variables.
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * cdelay (string) character string that defines the subsidence delay
          type for the interbed. Possible subsidence package CDELAY strings
          include: NODELAY--character keyword to indicate that delay will not
          be simulated in the interbed. DELAY--character keyword to indicate
          that delay will be simulated in the interbed.
        * pcs0 (double) is the initial offset from the calculated initial
          effective stress or initial preconsolidation stress in the interbed,
          in units of height of a column of water. PCS0 is the initial
          preconsolidation stress if SPECIFIED_INITIAL_INTERBED_STATE or
          SPECIFIED_INITIAL_PRECONSOLIDATION_STRESS are specified in the
          OPTIONS block. If HEAD_BASED is specified in the OPTIONS block, PCS0
          is the initial offset from the calculated initial head or initial
          preconsolidation head in the CSUB interbed and the initial
          preconsolidation stress is calculated from the calculated initial
          effective stress or calculated initial geostatic stress,
          respectively.
        * thick_frac (double) is the interbed thickness or cell fraction of the
          interbed. Interbed thickness is specified as a fraction of the cell
          thickness if CELL_FRACTION is specified in the OPTIONS block.
        * rnb (double) is the interbed material factor equivalent number of
          interbeds in the interbed system represented by the interbed. RNB
          must be greater than or equal to 1 if CDELAY is DELAY. Otherwise, RNB
          can be any value.
        * ssv_cc (double) is the initial inelastic specific storage or
          compression index of the interbed. The compression index is specified
          if COMPRESSION_INDICES is specified in the OPTIONS block. Specified
          or calculated interbed inelastic specific storage values are not
          adjusted from initial values if HEAD_BASED is specified in the
          OPTIONS block.
        * sse_cr (double) is the initial elastic coarse-grained material
          specific storage or recompression index of the interbed. The
          recompression index is specified if COMPRESSION_INDICES is specified
          in the OPTIONS block. Specified or calculated interbed elastic
          specific storage values are not adjusted from initial values if
          HEAD_BASED is specified in the OPTIONS block.
        * theta (double) is the initial porosity of the interbed.
        * kv (double) is the vertical hydraulic conductivity of the delay
          interbed. KV must be greater than 0 if CDELAY is DELAY. Otherwise, KV
          can be any value.
        * h0 (double) is the initial offset from the head in cell cellid or the
          initial head in the delay interbed. H0 is the initial head in the
          delay bed if SPECIFIED_INITIAL_INTERBED_STATE or
          SPECIFIED_INITIAL_DELAY_HEAD are specified in the OPTIONS block. H0
          can be any value if CDELAY is NODELAY.
        * boundname (string) name of the CSUB cell. BOUNDNAME is an ASCII
          character variable that can contain as many as 40 characters. If
          BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    stress_period_data : [cellid, sig0]
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * sig0 (double) is the stress offset for the cell. SIG0 is added to the
          calculated geostatic stress for the cell. SIG0 is specified only if
          MAXSIG0 is specified to be greater than 0 in the DIMENSIONS block. If
          the Options block includes a TIMESERIESFILE entry (see the "Time-
          Variable Input" section), values can be obtained from a time series
          by entering the time-series name in place of a numeric value.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    strainib_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "strainib_filerecord")
    )
    straincg_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "straincg_filerecord")
    )
    compaction_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "compaction_filerecord")
    )
    compaction_elastic_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "compaction_elastic_filerecord")
    )
    compaction_inelastic_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "compaction_inelastic_filerecord")
    )
    compaction_interbed_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "compaction_interbed_filerecord")
    )
    compaction_coarse_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "compaction_coarse_filerecord")
    )
    zdisplacement_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "zdisplacement_filerecord")
    )
    package_convergence_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "package_convergence_filerecord")
    )
    ts_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "ts_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwf6", "csub", "options", "obs_filerecord")
    )
    cg_ske_cr = ArrayTemplateGenerator(
        ("gwf6", "csub", "griddata", "cg_ske_cr")
    )
    cg_theta = ArrayTemplateGenerator(("gwf6", "csub", "griddata", "cg_theta"))
    sgm = ArrayTemplateGenerator(("gwf6", "csub", "griddata", "sgm"))
    sgs = ArrayTemplateGenerator(("gwf6", "csub", "griddata", "sgs"))
    packagedata = ListTemplateGenerator(
        ("gwf6", "csub", "packagedata", "packagedata")
    )
    stress_period_data = ListTemplateGenerator(
        ("gwf6", "csub", "period", "stress_period_data")
    )
    package_abbr = "gwfcsub"
    _package_type = "csub"
    dfn_file_name = "gwf-csub.dfn"

    dfn = [
        [
            "block options",
            "name boundnames",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name gammaw",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 9806.65",
        ],
        [
            "block options",
            "name beta",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 4.6512e-10",
        ],
        [
            "block options",
            "name head_based",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name initial_preconsolidation_head",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name ndelaycells",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name compression_indices",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name update_material_properties",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name cell_fraction",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name specified_initial_interbed_state",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name specified_initial_preconsolidation_stress",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name specified_initial_delay_head",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name effective_stress_lag",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name strainib_filerecord",
            "type record strain_csv_interbed fileout interbedstrain_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name strain_csv_interbed",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name fileout",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name interbedstrain_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name straincg_filerecord",
            "type record strain_csv_coarse fileout coarsestrain_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name strain_csv_coarse",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name coarsestrain_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name compaction_filerecord",
            "type record compaction fileout compaction_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name compaction",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name fileout",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name compaction_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name compaction_elastic_filerecord",
            "type record compaction_elastic fileout elastic_compaction_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name compaction_elastic",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name elastic_compaction_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name compaction_inelastic_filerecord",
            "type record compaction_inelastic fileout "
            "inelastic_compaction_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name compaction_inelastic",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name inelastic_compaction_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name compaction_interbed_filerecord",
            "type record compaction_interbed fileout "
            "interbed_compaction_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name compaction_interbed",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name interbed_compaction_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name compaction_coarse_filerecord",
            "type record compaction_coarse fileout coarse_compaction_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name compaction_coarse",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name coarse_compaction_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name zdisplacement_filerecord",
            "type record zdisplacement fileout zdisplacement_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name zdisplacement",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name zdisplacement_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name package_convergence_filerecord",
            "type record package_convergence fileout "
            "package_convergence_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name package_convergence",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name package_convergence_filename",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name ts_filerecord",
            "type record ts6 filein ts6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package ts",
            "construct_data timeseries",
            "parameter_name timeseries",
        ],
        [
            "block options",
            "name ts6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name ts6_filename",
            "type string",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block options",
            "name obs_filerecord",
            "type record obs6 filein obs6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package obs",
            "construct_data continuous",
            "parameter_name observations",
        ],
        [
            "block options",
            "name obs6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name obs6_filename",
            "type string",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name ninterbeds",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name maxsig0",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block griddata",
            "name cg_ske_cr",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "default_value 1e-5",
        ],
        [
            "block griddata",
            "name cg_theta",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "default_value 0.2",
        ],
        [
            "block griddata",
            "name sgm",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "optional true",
        ],
        [
            "block griddata",
            "name sgs",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "optional true",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray icsubno cellid cdelay pcs0 thick_frac rnb ssv_cc "
            "sse_cr theta kv h0 boundname",
            "shape (ninterbeds)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name icsubno",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name cellid",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name cdelay",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name pcs0",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name thick_frac",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name rnb",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name ssv_cc",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name sse_cr",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name theta",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "default_value 0.2",
        ],
        [
            "block packagedata",
            "name kv",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name h0",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name boundname",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
        ],
        [
            "block period",
            "name iper",
            "type integer",
            "block_variable True",
            "in_record true",
            "tagged false",
            "shape",
            "valid",
            "reader urword",
            "optional false",
        ],
        [
            "block period",
            "name stress_period_data",
            "type recarray cellid sig0",
            "shape (maxsig0)",
            "reader urword",
        ],
        [
            "block period",
            "name cellid",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name sig0",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        boundnames=None,
        print_input=None,
        save_flows=None,
        gammaw=9806.65,
        beta=4.6512e-10,
        head_based=None,
        initial_preconsolidation_head=None,
        ndelaycells=None,
        compression_indices=None,
        update_material_properties=None,
        cell_fraction=None,
        specified_initial_interbed_state=None,
        specified_initial_preconsolidation_stress=None,
        specified_initial_delay_head=None,
        effective_stress_lag=None,
        strainib_filerecord=None,
        straincg_filerecord=None,
        compaction_filerecord=None,
        fileout=None,
        compaction_elastic_filerecord=None,
        compaction_inelastic_filerecord=None,
        compaction_interbed_filerecord=None,
        compaction_coarse_filerecord=None,
        zdisplacement_filerecord=None,
        package_convergence_filerecord=None,
        timeseries=None,
        observations=None,
        ninterbeds=None,
        maxsig0=None,
        cg_ske_cr=1e-5,
        cg_theta=0.2,
        sgm=None,
        sgs=None,
        packagedata=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "csub", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.gammaw = self.build_mfdata("gammaw", gammaw)
        self.beta = self.build_mfdata("beta", beta)
        self.head_based = self.build_mfdata("head_based", head_based)
        self.initial_preconsolidation_head = self.build_mfdata(
            "initial_preconsolidation_head", initial_preconsolidation_head
        )
        self.ndelaycells = self.build_mfdata("ndelaycells", ndelaycells)
        self.compression_indices = self.build_mfdata(
            "compression_indices", compression_indices
        )
        self.update_material_properties = self.build_mfdata(
            "update_material_properties", update_material_properties
        )
        self.cell_fraction = self.build_mfdata("cell_fraction", cell_fraction)
        self.specified_initial_interbed_state = self.build_mfdata(
            "specified_initial_interbed_state",
            specified_initial_interbed_state,
        )
        self.specified_initial_preconsolidation_stress = self.build_mfdata(
            "specified_initial_preconsolidation_stress",
            specified_initial_preconsolidation_stress,
        )
        self.specified_initial_delay_head = self.build_mfdata(
            "specified_initial_delay_head", specified_initial_delay_head
        )
        self.effective_stress_lag = self.build_mfdata(
            "effective_stress_lag", effective_stress_lag
        )
        self.strainib_filerecord = self.build_mfdata(
            "strainib_filerecord", strainib_filerecord
        )
        self.straincg_filerecord = self.build_mfdata(
            "straincg_filerecord", straincg_filerecord
        )
        self.compaction_filerecord = self.build_mfdata(
            "compaction_filerecord", compaction_filerecord
        )
        self.fileout = self.build_mfdata("fileout", fileout)
        self.compaction_elastic_filerecord = self.build_mfdata(
            "compaction_elastic_filerecord", compaction_elastic_filerecord
        )
        self.compaction_inelastic_filerecord = self.build_mfdata(
            "compaction_inelastic_filerecord", compaction_inelastic_filerecord
        )
        self.compaction_interbed_filerecord = self.build_mfdata(
            "compaction_interbed_filerecord", compaction_interbed_filerecord
        )
        self.compaction_coarse_filerecord = self.build_mfdata(
            "compaction_coarse_filerecord", compaction_coarse_filerecord
        )
        self.zdisplacement_filerecord = self.build_mfdata(
            "zdisplacement_filerecord", zdisplacement_filerecord
        )
        self.package_convergence_filerecord = self.build_mfdata(
            "package_convergence_filerecord", package_convergence_filerecord
        )
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.ninterbeds = self.build_mfdata("ninterbeds", ninterbeds)
        self.maxsig0 = self.build_mfdata("maxsig0", maxsig0)
        self.cg_ske_cr = self.build_mfdata("cg_ske_cr", cg_ske_cr)
        self.cg_theta = self.build_mfdata("cg_theta", cg_theta)
        self.sgm = self.build_mfdata("sgm", sgm)
        self.sgs = self.build_mfdata("sgs", sgs)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
