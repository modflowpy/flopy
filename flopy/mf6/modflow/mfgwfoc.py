# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfoc(mfpackage.MFPackage):
    """
    ModflowGwfoc defines a oc package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    head_filerecord : [headfile]
        * headfile (string) name of the output file to write head information.
    headprintrecord : [columns, width, digits, format]
        * columns (integer) number of columns for writing data.
        * width (integer) width for writing each number.
        * digits (integer) number of digits to use for writing a number.
        * format (string) write format can be EXPONENTIAL, FIXED, GENERAL, or
          SCIENTIFIC.
    saverecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can be BUDGET or
          HEAD.
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
        * rtype (string) type of information to save or print. Can be BUDGET or
          HEAD.
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
        ("gwf6", "oc", "options", "budget_filerecord")
    )
    head_filerecord = ListTemplateGenerator(
        ("gwf6", "oc", "options", "head_filerecord")
    )
    headprintrecord = ListTemplateGenerator(
        ("gwf6", "oc", "options", "headprintrecord")
    )
    saverecord = ListTemplateGenerator(("gwf6", "oc", "period", "saverecord"))
    printrecord = ListTemplateGenerator(
        ("gwf6", "oc", "period", "printrecord")
    )
    package_abbr = "gwfoc"
    _package_type = "oc"
    dfn_file_name = "gwf-oc.dfn"

    dfn = [
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
            "name head_filerecord",
            "type record head fileout headfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name head",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name headfile",
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
            "name headprintrecord",
            "type record head print_format formatrecord",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_format",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name formatrecord",
            "type record columns width digits format",
            "shape",
            "in_record true",
            "reader urword",
            "tagged",
            "optional false",
        ],
        [
            "block options",
            "name columns",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name width",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name digits",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name format",
            "type string",
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
        head_filerecord=None,
        headprintrecord=None,
        saverecord=None,
        printrecord=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwfoc, self).__init__(
            model, "oc", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.head_filerecord = self.build_mfdata(
            "head_filerecord", head_filerecord
        )
        self.headprintrecord = self.build_mfdata(
            "headprintrecord", headprintrecord
        )
        self.saverecord = self.build_mfdata("saverecord", saverecord)
        self.printrecord = self.build_mfdata("printrecord", printrecord)
        self._init_complete = True
