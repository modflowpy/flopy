# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
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
          simulation list file. The value has no affect on the simulation. The
          recommended format for the starting date and time is described at
          https://www.w3.org/TR/NOTE-datetime.
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
        nper=1,
        perioddata=((1.0, 1, 1.0),),
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowTdis, self).__init__(
            simulation, "tdis", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.time_units = self.build_mfdata("time_units", time_units)
        self.start_date_time = self.build_mfdata(
            "start_date_time", start_date_time
        )
        self.nper = self.build_mfdata("nper", nper)
        self.perioddata = self.build_mfdata("perioddata", perioddata)
        self._init_complete = True
