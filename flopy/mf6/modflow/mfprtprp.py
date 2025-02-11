# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 11, 2025 01:24:12 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowPrtprp(mfpackage.MFPackage):
    """
    ModflowPrtprp defines a prp package within a prt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of particle release points.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of all model
          stress package information will be written to the listing file
          immediately after it is read.
    dev_exit_solve_method : integer
        * dev_exit_solve_method (integer) the method for iterative solution of
          particle exit location and time in the generalized Pollock's method.
          0 default, 1 Brent, 2 Chandrupatla. The default is Brent's method.
    exit_solve_tolerance : double
        * exit_solve_tolerance (double) the convergence tolerance for iterative
          solution of particle exit location and time in the generalized
          Pollock's method. A value of 0.00001 works well for many problems,
          but the value that strikes the best balance between accuracy and
          runtime is problem-dependent.
    local_z : boolean
        * local_z (boolean) indicates that "zrpt" defines the local z
          coordinate of the release point within the cell, with value of 0 at
          the bottom and 1 at the top of the cell. If the cell is partially
          saturated at release time, the top of the cell is considered to be
          the water table elevation (the head in the cell) rather than the top
          defined by the user.
    extend_tracking : boolean
        * extend_tracking (boolean) indicates that particles should be tracked
          beyond the end of the simulation's final time step (using that time
          step's flows) until particles terminate or reach a specified stop
          time. By default, particles are terminated at the end of the
          simulation's final time step.
    track_filerecord : [trackfile]
        * trackfile (string) name of the binary output file to write tracking
          information.
    trackcsv_filerecord : [trackcsvfile]
        * trackcsvfile (string) name of the comma-separated value (CSV) file to
          write tracking information.
    stoptime : double
        * stoptime (double) real value defining the maximum simulation time to
          which particles in the package can be tracked. Particles that have
          not terminated earlier due to another termination condition will
          terminate when simulation time STOPTIME is reached. If the last
          stress period in the simulation consists of more than one time step,
          particles will not be tracked past the ending time of the last stress
          period, regardless of STOPTIME. If the EXTEND_TRACKING option is
          enabled and the last stress period in the simulation is steady-state,
          the simulation ending time will not limit the time to which particles
          can be tracked, but STOPTIME and STOPTRAVELTIME will continue to
          apply. If STOPTIME and STOPTRAVELTIME are both provided, particles
          will be stopped if either is reached.
    stoptraveltime : double
        * stoptraveltime (double) real value defining the maximum travel time
          over which particles in the model can be tracked. Particles that have
          not terminated earlier due to another termination condition will
          terminate when their travel time reaches STOPTRAVELTIME. If the last
          stress period in the simulation consists of more than one time step,
          particles will not be tracked past the ending time of the last stress
          period, regardless of STOPTRAVELTIME. If the EXTEND_TRACKING option
          is enabled and the last stress period in the simulation is steady-
          state, the simulation ending time will not limit the time to which
          particles can be tracked, but STOPTIME and STOPTRAVELTIME will
          continue to apply. If STOPTIME and STOPTRAVELTIME are both provided,
          particles will be stopped if either is reached.
    stop_at_weak_sink : boolean
        * stop_at_weak_sink (boolean) is a text keyword to indicate that a
          particle is to terminate when it enters a cell that is a weak sink.
          By default, particles are allowed to pass though cells that are weak
          sinks.
    istopzone : integer
        * istopzone (integer) integer value defining the stop zone number. If
          cells have been assigned IZONE values in the GRIDDATA block, a
          particle terminates if it enters a cell whose IZONE value matches
          ISTOPZONE. An ISTOPZONE value of zero indicates that there is no stop
          zone. The default value is zero.
    drape : boolean
        * drape (boolean) is a text keyword to indicate that if a particle's
          release point is in a cell that happens to be inactive at release
          time, the particle is to be moved to the topmost active cell below
          it, if any. By default, a particle is not released into the
          simulation if its release point's cell is inactive at release time.
    dry_tracking_method : string
        * dry_tracking_method (string) is a string indicating how particles
          should behave in dry-but-active cells (as can occur with the Newton
          formulation). The value can be "DROP", "STOP", or "STAY". The default
          is "DROP", which passes particles vertically and instantaneously to
          the water table. "STOP" causes particles to terminate. "STAY" causes
          particles to remain stationary but active.
    dev_forceternary : boolean
        * dev_forceternary (boolean) force use of the ternary tracking method
          regardless of cell type in DISV grids.
    release_time_tolerance : double
        * release_time_tolerance (double) real number indicating the tolerance
          within which to consider consecutive release times coincident.
          Coincident release times will be merged into a single release time.
          The default is :math:`\\epsilon \\times 10^{11}`, where
          :math:`\\epsilon` is machine precision.
    release_time_frequency : double
        * release_time_frequency (double) real number indicating the time
          frequency at which to release particles. This option can be used to
          schedule releases at a regular interval for the duration of the
          simulation, starting at the simulation start time. The release
          schedule is the union of this option, the RELEASETIMES block, and
          PERIOD block RELEASESETTING selections. If none of these are
          provided, a single release time is configured at the beginning of the
          first time step of the simulation's first stress period.
    nreleasepts : integer
        * nreleasepts (integer) is the number of particle release points.
    nreleasetimes : integer
        * nreleasetimes (integer) is the number of particle release times
          specified in the RELEASETIMES block. This is not necessarily the
          total number of release times; release times are the union of
          RELEASE_TIME_FREQUENCY, RELEASETIMES block, and PERIOD block
          RELEASESETTING selections.
    packagedata : [irptno, cellid, xrpt, yrpt, zrpt, boundname]
        * irptno (integer) integer value that defines the PRP release point
          number associated with the specified PACKAGEDATA data on the line.
          IRPTNO must be greater than zero and less than or equal to
          NRELEASEPTS. The program will terminate with an error if information
          for a PRP release point number is specified more than once. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
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
        * xrpt (double) real value that defines the x coordinate of the release
          point in model coordinates. The (x, y, z) location specified for the
          release point must lie within the cell that is identified by the
          specified cellid.
        * yrpt (double) real value that defines the y coordinate of the release
          point in model coordinates. The (x, y, z) location specified for the
          release point must lie within the cell that is identified by the
          specified cellid.
        * zrpt (double) real value that defines the z coordinate of the release
          point in model coordinates or, if the LOCAL_Z option is active, in
          local cell coordinates. The (x, y, z) location specified for the
          release point must lie within the cell that is identified by the
          specified cellid.
        * boundname (string) name of the particle release point. BOUNDNAME is
          an ASCII character variable that can contain as many as 40
          characters. If BOUNDNAME contains spaces in it, then the entire name
          must be enclosed within single quotes.
    releasetimes : time
        * time (double) real value that defines the release time with respect
          to the simulation start time.
    perioddata : releasesetting
        * releasesetting (keystring) specifies time steps at which to release a
          particle. A particle is released at the beginning of each specified
          time step. For fine control over release timing, specify times
          explicitly using the RELEASETIMES block. If the beginning of a
          specified time step coincides with a release time specified in the
          RELEASETIMES block or configured via RELEASE_TIME_FREQUENCY, only one
          particle is released at that time. Coincidence is evaluated up to the
          tolerance specified in RELEASE_TIME_TOLERANCE, or :math:`\\epsilon
          \\times 10^{11}` by default, where :math:`\\epsilon` is machine
          precision. If no release times are configured via this setting, the
          RELEASETIMES block, or the RELEASE_TIME_FREQUENCY option, a single
          release time is configured at the beginning of the first time step of
          the simulation's first stress period.
            all : [keyword]
                * all (keyword) keyword to indicate release at the start of all
                  time steps in the period.
            first : [keyword]
                * first (keyword) keyword to indicate release at the start of
                  the first time step in the period. This keyword may be used
                  in conjunction with other RELEASESETTING options.
            last : [keyword]
                * last (keyword) keyword to indicate release at the start of
                  the last time step in the period. This keyword may be used in
                  conjunction with other RELEASESETTING options.
            frequency : [integer]
                * frequency (integer) release at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  RELEASESETTING options.
            steps : [integer]
                * steps (integer) release at the start of each step specified
                  in STEPS. This option may be used in conjunction with other
                  RELEASESETTING options.
            fraction : [double]
                * fraction (double) release particles after the specified
                  fraction of the time step has elapsed. If FRACTION is not
                  set, particles are released at the start of the specified
                  time step(s). FRACTION must be a single value when used with
                  ALL, FIRST, or FREQUENCY. When used with STEPS, FRACTION may
                  be a single value or an array of the same length as STEPS. If
                  a single FRACTION value is provided with STEPS, the fraction
                  applies to all steps. NOTE: The FRACTION option has been
                  removed. For fine control over release timing, specify times
                  explicitly using the RELEASETIMES block.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    track_filerecord = ListTemplateGenerator(
        ("prt6", "prp", "options", "track_filerecord")
    )
    trackcsv_filerecord = ListTemplateGenerator(
        ("prt6", "prp", "options", "trackcsv_filerecord")
    )
    packagedata = ListTemplateGenerator(("prt6", "prp", "packagedata", "packagedata"))
    releasetimes = ListTemplateGenerator(
        ("prt6", "prp", "releasetimes", "releasetimes")
    )
    perioddata = ListTemplateGenerator(("prt6", "prp", "period", "perioddata"))
    package_abbr = "prtprp"
    _package_type = "prp"
    dfn_file_name = "prt-prp.dfn"

    dfn = [
        [
            "header",
            "multi-package",
        ],
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
            "name dev_exit_solve_method",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name exit_solve_tolerance",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 1e-5",
        ],
        [
            "block options",
            "name local_z",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name extend_tracking",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_filerecord",
            "type record track fileout trackfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name track",
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
            "name trackfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name trackcsv_filerecord",
            "type record trackcsv fileout trackcsvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name trackcsv",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name trackcsvfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name stoptime",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name stoptraveltime",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name stop_at_weak_sink",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name istopzone",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name drape",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name dry_tracking_method",
            "type string",
            "valid drop stop stay",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name dev_forceternary",
            "type keyword",
            "reader urword",
            "optional false",
            "mf6internal ifrctrn",
        ],
        [
            "block options",
            "name release_time_tolerance",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name release_time_frequency",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name nreleasepts",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name nreleasetimes",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray irptno cellid xrpt yrpt zrpt boundname",
            "shape (nreleasepts)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name irptno",
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
            "name xrpt",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name yrpt",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name zrpt",
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
            "block releasetimes",
            "name releasetimes",
            "type recarray time",
            "shape (nreleasetimes)",
            "reader urword",
        ],
        [
            "block releasetimes",
            "name time",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
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
            "name perioddata",
            "type recarray releasesetting",
            "shape",
            "reader urword",
        ],
        [
            "block period",
            "name releasesetting",
            "type keystring all first last frequency steps fraction",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name all",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name first",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name last",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name frequency",
            "type integer",
            "shape",
            "tagged true",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name steps",
            "type integer",
            "shape (<nstp)",
            "tagged true",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name fraction",
            "type double precision",
            "shape (<nstp)",
            "tagged true",
            "in_record true",
            "reader urword",
            "optional true",
            "removed 6.6.0",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        boundnames=None,
        print_input=None,
        dev_exit_solve_method=None,
        exit_solve_tolerance=1e-5,
        local_z=None,
        extend_tracking=None,
        track_filerecord=None,
        trackcsv_filerecord=None,
        stoptime=None,
        stoptraveltime=None,
        stop_at_weak_sink=None,
        istopzone=None,
        drape=None,
        dry_tracking_method=None,
        dev_forceternary=None,
        release_time_tolerance=None,
        release_time_frequency=None,
        nreleasepts=None,
        nreleasetimes=None,
        packagedata=None,
        releasetimes=None,
        perioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "prp", filename, pname, loading_package, **kwargs)

        # set up variables
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.dev_exit_solve_method = self.build_mfdata(
            "dev_exit_solve_method", dev_exit_solve_method
        )
        self.exit_solve_tolerance = self.build_mfdata(
            "exit_solve_tolerance", exit_solve_tolerance
        )
        self.local_z = self.build_mfdata("local_z", local_z)
        self.extend_tracking = self.build_mfdata("extend_tracking", extend_tracking)
        self.track_filerecord = self.build_mfdata("track_filerecord", track_filerecord)
        self.trackcsv_filerecord = self.build_mfdata(
            "trackcsv_filerecord", trackcsv_filerecord
        )
        self.stoptime = self.build_mfdata("stoptime", stoptime)
        self.stoptraveltime = self.build_mfdata("stoptraveltime", stoptraveltime)
        self.stop_at_weak_sink = self.build_mfdata(
            "stop_at_weak_sink", stop_at_weak_sink
        )
        self.istopzone = self.build_mfdata("istopzone", istopzone)
        self.drape = self.build_mfdata("drape", drape)
        self.dry_tracking_method = self.build_mfdata(
            "dry_tracking_method", dry_tracking_method
        )
        self.dev_forceternary = self.build_mfdata("dev_forceternary", dev_forceternary)
        self.release_time_tolerance = self.build_mfdata(
            "release_time_tolerance", release_time_tolerance
        )
        self.release_time_frequency = self.build_mfdata(
            "release_time_frequency", release_time_frequency
        )
        self.nreleasepts = self.build_mfdata("nreleasepts", nreleasepts)
        self.nreleasetimes = self.build_mfdata("nreleasetimes", nreleasetimes)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.releasetimes = self.build_mfdata("releasetimes", releasetimes)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
