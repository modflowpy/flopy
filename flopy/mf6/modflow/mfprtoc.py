# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowPrtoc(mfpackage.MFPackage):
    """
    ModflowPrtoc defines a oc package within a prt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
    track_filerecord : [trackfile]
        * trackfile (string) name of the binary output file to write tracking
          information.
    trackcsv_filerecord : [trackcsvfile]
        * trackcsvfile (string) name of the comma-separated value (CSV) file to
          write tracking information.
    track_release : boolean
        * track_release (boolean) keyword to indicate that particle tracking
          output is to be written when a particle is released
    track_exit : boolean
        * track_exit (boolean) keyword to indicate that particle tracking
          output is to be written when a particle exits a cell
    track_timestep : boolean
        * track_timestep (boolean) keyword to indicate that particle tracking
          output is to be written at the end of each time step
    track_terminate : boolean
        * track_terminate (boolean) keyword to indicate that particle tracking
          output is to be written when a particle terminates for any reason
    track_weaksink : boolean
        * track_weaksink (boolean) keyword to indicate that particle tracking
          output is to be written when a particle exits a weak sink (a cell
          which removes some but not all inflow from adjacent cells)
    track_usertime : boolean
        * track_usertime (boolean) keyword to indicate that particle tracking
          output is to be written at user-specified times, provided as double
          precision values to the TRACK_TIMES or TRACK_TIMESFILE options
    track_timesrecord : [times]
        * times (double) times to track, relative to the beginning of the
          simulation.
    track_timesfilerecord : [timesfile]
        * timesfile (string) name of the tracking times file
    saverecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can only be
          BUDGET.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    printrecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can only be
          BUDGET.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    budget_filerecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "budget_filerecord")
    )
    budgetcsv_filerecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "budgetcsv_filerecord")
    )
    track_filerecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "track_filerecord")
    )
    trackcsv_filerecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "trackcsv_filerecord")
    )
    track_timesrecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "track_timesrecord")
    )
    track_timesfilerecord = ListTemplateGenerator(
        ("prt6", "oc", "options", "track_timesfilerecord")
    )
    saverecord = ListTemplateGenerator(("prt6", "oc", "period", "saverecord"))
    printrecord = ListTemplateGenerator(
        ("prt6", "oc", "period", "printrecord")
    )
    package_abbr = "prtoc"
    _package_type = "oc"
    dfn_file_name = "prt-oc.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name budget_filerecord",
            "type record budget fileout budgetfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budget",
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
            "name budgetfile",
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
            "name budgetcsv_filerecord",
            "type record budgetcsv fileout budgetcsvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budgetcsv",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name budgetcsvfile",
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
            "name track_release",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_exit",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_timestep",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_terminate",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_weaksink",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_usertime",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name track_timesrecord",
            "type record track_times times",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name track_times",
            "type keyword",
            "reader urword",
            "in_record true",
            "tagged true",
            "shape",
        ],
        [
            "block options",
            "name times",
            "type double precision",
            "shape (unknown)",
            "reader urword",
            "in_record true",
            "tagged false",
            "repeating true",
        ],
        [
            "block options",
            "name track_timesfilerecord",
            "type record track_timesfile timesfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name track_timesfile",
            "type keyword",
            "reader urword",
            "in_record true",
            "tagged true",
            "shape",
        ],
        [
            "block options",
            "name timesfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
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
            "name saverecord",
            "type record save rtype ocsetting",
            "shape",
            "reader urword",
            "tagged false",
            "optional true",
        ],
        [
            "block period",
            "name save",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block period",
            "name printrecord",
            "type record print rtype ocsetting",
            "shape",
            "reader urword",
            "tagged false",
            "optional true",
        ],
        [
            "block period",
            "name print",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block period",
            "name rtype",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block period",
            "name ocsetting",
            "type keystring all first last frequency steps",
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
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        track_filerecord=None,
        trackcsv_filerecord=None,
        track_release=None,
        track_exit=None,
        track_timestep=None,
        track_terminate=None,
        track_weaksink=None,
        track_usertime=None,
        track_timesrecord=None,
        track_timesfilerecord=None,
        saverecord=None,
        printrecord=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "oc", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.budgetcsv_filerecord = self.build_mfdata(
            "budgetcsv_filerecord", budgetcsv_filerecord
        )
        self.track_filerecord = self.build_mfdata(
            "track_filerecord", track_filerecord
        )
        self.trackcsv_filerecord = self.build_mfdata(
            "trackcsv_filerecord", trackcsv_filerecord
        )
        self.track_release = self.build_mfdata("track_release", track_release)
        self.track_exit = self.build_mfdata("track_exit", track_exit)
        self.track_timestep = self.build_mfdata(
            "track_timestep", track_timestep
        )
        self.track_terminate = self.build_mfdata(
            "track_terminate", track_terminate
        )
        self.track_weaksink = self.build_mfdata(
            "track_weaksink", track_weaksink
        )
        self.track_usertime = self.build_mfdata(
            "track_usertime", track_usertime
        )
        self.track_timesrecord = self.build_mfdata(
            "track_timesrecord", track_timesrecord
        )
        self.track_timesfilerecord = self.build_mfdata(
            "track_timesfilerecord", track_timesfilerecord
        )
        self.saverecord = self.build_mfdata("saverecord", saverecord)
        self.printrecord = self.build_mfdata("printrecord", printrecord)
        self._init_complete = True
