# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowPrtfmi(mfpackage.MFPackage):
    """
    ModflowPrtfmi defines a fmi package within a prt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that FMI flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    packagedata : [flowtype, fname]
        * flowtype (string) is the word GWFBUDGET or GWFHEAD. If GWFBUDGET is
          specified, then the corresponding file must be a budget file from a
          previous GWF Model run.
        * fname (string) is the name of the file containing flows. The path to
          the file should be included if the file is not located in the folder
          where the program was run.
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
        ("prt6", "fmi", "packagedata", "packagedata")
    )
    package_abbr = "prtfmi"
    _package_type = "fmi"
    dfn_file_name = "prt-fmi.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray flowtype filein fname",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name flowtype",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block packagedata",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block packagedata",
            "name fname",
            "in_record true",
            "type string",
            "preserve_case true",
            "tagged false",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        save_flows=None,
        packagedata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "fmi", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self._init_complete = True
