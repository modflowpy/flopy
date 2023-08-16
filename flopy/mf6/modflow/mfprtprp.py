# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 16, 2023 03:03:23 UTC
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
          provided with the list of release-point cells.
    stoptime : double
        * stoptime (double) real value defining the maximum simulation time to
          which particles in the model can be tracked. Particles that have not
          terminated earlier due to another termination condition will
          terminate when simulation time STOPTIME is reached. If the last
          stress period in the simulation consists of more than one time step,
          particles will not be tracked past the ending time of the last stress
          period, regardless of STOPTIME. If the last stress period in the
          simulation consists of a single time step, it is assumed to be a
          steady-state stress period, and its ending time will not limit the
          simulation time to which particles can be tracked.
    stoptraveltime : double
        * stoptraveltime (double) real value defining the maximum travel time
          over which particles in the model can be tracked. Particles that have
          not terminated earlier due to another termination condition will
          terminate when their travel time reaches STOPTRAVELTIME. If the last
          stress period in the simulation consists of more than one time step,
          particles will not be tracked past the ending time of the last stress
          period, regardless of STOPTRAVELTIME. If the last stress period in
          the simulation consists of a single time step, it is assumed to be a
          steady-state stress period, and its ending time will not limit the
          travel time over which particles can be tracked.
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
        * drape (boolean) is a text keyword to indicate that a particle is to
          be moved to the topmost active cell prior to release if its release
          point is in a cell that is dry at the scheduled time of release. By
          default, a particle does not get released into the simulation if its
          release point is in a cell that is dry at the scheduled time of
          release. ??? Move to what elevation within topmost active cell ???
    nreleasepts : integer
        * nreleasepts (integer) is the number of particle release points.
    packagedata : [irptno, cellid, xrpt, yrpt, zrpt, boundname]
        * irptno (integer) integer value that defines the PRP release point
          number associated with the specified PACKAGEDATA data on the line.
          IRPTNO must be greater than zero and less than or equal to
          NRELEASEPTS. The program will terminate with an error if information
          for a PRP release point number is specified more than once. ??? DO WE
          REALLY NEED THIS ??? This argument is an index variable, which means
          that it should be treated as zero-based when working with FloPy and
          Python. Flopy will automatically subtract one when loading index
          variables and add one when writing index variables.
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
          release point must lie within the cell that corresponds to the
          specified cellid.
        * yrpt (double) real value that defines the y coordinate of the release
          point in model coordinates. The (x, y, z) location specified for the
          release point must lie within the cell that corresponds to the
          specified cellid.
        * zrpt (double) real value that defines the z coordinate of the release
          point in model coordinates. The (x, y, z) location specified for the
          release point must lie within the cell that corresponds to the
          specified cellid.
        * boundname (string) name of the release-point cell. BOUNDNAME is an
          ASCII character variable that can contain as many as 40 characters.
          If BOUNDNAME contains spaces in it, then the entire name must be
          enclosed within single quotes.
    perioddata : releasesetting
        * releasesetting (keystring) specifies the steps at the start of which
          particles will be released. The setting applies to all release points
          defined in PACKAGEDATA.
            all : [keyword]
                * all (keyword) keyword to indicate release of particles at the
                  start of all time steps in the period.
            first : [keyword]
                * first (keyword) keyword to indicate release of particles at
                  the start of the first time step in the period. This keyword
                  may be used in conjunction with other keywords to release
                  particles at the start of multiple time steps.
            frequency : [integer]
                * frequency (integer) release particles at the specified time
                  step frequency. This keyword may be used in conjunction with
                  other keywords to release particles at the start of multiple
                  time steps.
            steps : [integer]
                * steps (integer) release particles at the start of each step
                  specified in STEPS. This keyword may be used in conjunction
                  with other keywords to release particles at the start of
                  multiple time steps.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    packagedata = ListTemplateGenerator(
        ("prt6", "prp", "packagedata", "packagedata")
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
            "block dimensions",
            "name nreleasepts",
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
            "type keystring all first frequency steps",
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
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        boundnames=None,
        stoptime=None,
        stoptraveltime=None,
        stop_at_weak_sink=None,
        istopzone=None,
        drape=None,
        nreleasepts=None,
        packagedata=None,
        perioddata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "prp", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.stoptime = self.build_mfdata("stoptime", stoptime)
        self.stoptraveltime = self.build_mfdata(
            "stoptraveltime", stoptraveltime
        )
        self.stop_at_weak_sink = self.build_mfdata(
            "stop_at_weak_sink", stop_at_weak_sink
        )
        self.istopzone = self.build_mfdata("istopzone", istopzone)
        self.drape = self.build_mfdata("drape", drape)
        self.nreleasepts = self.build_mfdata("nreleasepts", nreleasepts)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
