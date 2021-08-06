# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:56:59 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowTdis(mfpackage.MFPackage):
    """
    ModflowTdis defines a tdis package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    time_units : string
        * time_units (string) is the time units of the simulation. This is a
          text string that is used as a label within model output files. Values
          for time_units may be "unknown", "seconds", "minutes", "hours",
          "days", or "years". The default time unit is "unknown".
    start_date_time : string
        * start_date_time (string) is the starting date and time of the
          simulation. This is a text string that is used as a label within the
          simulation list file. The value has no effect on the simulation. The
          recommended format for the starting date and time is described at
          https://www.w3.org/TR/NOTE-datetime.
    ats_filerecord : [ats6_filename]
        * ats6_filename (string) defines an adaptive time step (ATS) input file
          defining ATS controls. Records in the ATS file can be used to
          override the time step behavior for selected stress periods.
    nper : integer
        * nper (integer) is the number of stress periods for the simulation.
    perioddata : [perlen, nstp, tsmult]
        * perlen (double) is the length of a stress period.
        * nstp (integer) is the number of time steps in a stress period.
        * tsmult (double) is the multiplier for the length of successive time
          steps. The length of a time step is calculated by multiplying the
          length of the previous time step by TSMULT. The length of the first
          time step, :math:`\\Delta t_1`, is related to PERLEN, NSTP, and
          TSMULT by the relation :math:`\\Delta t_1= perlen \\frac{tsmult -
          1}{tsmult^{nstp}-1}`.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    ats_filerecord = ListTemplateGenerator(
        ("tdis", "options", "ats_filerecord")
    )
    perioddata = ListTemplateGenerator(("tdis", "perioddata", "perioddata"))
    package_abbr = "tdis"
    _package_type = "tdis"
    dfn_file_name = "sim-tdis.dfn"

    dfn = [
        [
            "block options",
            "name time_units",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name start_date_time",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name ats_filerecord",
            "type record ats6 filein ats6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name ats6",
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
            "name ats6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block dimensions",
            "name nper",
            "type integer",
            "reader urword",
            "optional false",
            "default_value 1",
        ],
        [
            "block perioddata",
            "name perioddata",
            "type recarray perlen nstp tsmult",
            "reader urword",
            "optional false",
            "default_value ((1.0, 1, 1.0),)",
        ],
        [
            "block perioddata",
            "name perlen",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name nstp",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block perioddata",
            "name tsmult",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        time_units=None,
        start_date_time=None,
        ats_filerecord=None,
        nper=1,
        perioddata=((1.0, 1, 1.0),),
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            simulation, "tdis", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.time_units = self.build_mfdata("time_units", time_units)
        self.start_date_time = self.build_mfdata(
            "start_date_time", start_date_time
        )
        self.ats_filerecord = self.build_mfdata(
            "ats_filerecord", ats_filerecord
        )
        self.nper = self.build_mfdata("nper", nper)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
