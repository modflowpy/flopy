# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowMvt(mfpackage.MFPackage):
    """
    ModflowMvt defines a mvt package.

    Parameters
    ----------
    parent_model_or_package : MFModel/MFPackage
        Parent_model_or_package that this package is a part of. Package is automatically
        added to parent_model_or_package when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of mover
          information will be written to the listing file immediately after it
          is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of lake flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that lake flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
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
        ("mvt", "options", "budget_filerecord")
    )
    budgetcsv_filerecord = ListTemplateGenerator(
        ("mvt", "options", "budgetcsv_filerecord")
    )
    package_abbr = "mvt"
    _package_type = "mvt"
    dfn_file_name = "gwt-mvt.dfn"

    dfn = [
        [
            "header",
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
            "name print_flows",
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
    ]

    def __init__(
        self,
        parent_model_or_package,
        loading_package=False,
        print_input=None,
        print_flows=None,
        save_flows=None,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            parent_model_or_package,
            "mvt",
            filename,
            pname,
            loading_package,
            **kwargs,
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.budgetcsv_filerecord = self.build_mfdata(
            "budgetcsv_filerecord", budgetcsv_filerecord
        )
        self._init_complete = True


class MvtPackages(mfpackage.MFChildPackages):
    """
    MvtPackages is a container class for the ModflowMvt class.

    Methods
    ----------
    initialize
        Initializes a new ModflowMvt package removing any sibling child
        packages attached to the same parent package. See ModflowMvt init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowGwtmvt package to the container. See ModflowGwtmvt
        init documentation for definition of parameters.
    """

    package_abbr = "mvtpackages"

    def initialize(
        self,
        print_input=None,
        print_flows=None,
        save_flows=None,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowMvt(
            self._cpparent,
            print_input=print_input,
            print_flows=print_flows,
            save_flows=save_flows,
            budget_filerecord=budget_filerecord,
            budgetcsv_filerecord=budgetcsv_filerecord,
            filename=filename,
            pname=pname,
            child_builder_call=True,
        )
        self.init_package(new_package, filename)

    def append_package(
        self,
        print_input=None,
        print_flows=None,
        save_flows=None,
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        filename=None,
        pname=None,
    ):
        new_package = ModflowGwtmvt(
            self._cpparent,
            print_input=print_input,
            print_flows=print_flows,
            save_flows=save_flows,
            budget_filerecord=budget_filerecord,
            budgetcsv_filerecord=budgetcsv_filerecord,
            filename=filename,
            pname=pname,
            child_builder_call=True,
        )
        self._append_package(new_package, filename)
