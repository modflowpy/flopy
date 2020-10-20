# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwtssm(mfpackage.MFPackage):
    """
    ModflowGwtssm defines a ssm package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of SSM flow
          rates will be printed to the listing file for every stress period
          time step in which "BUDGET PRINT" is specified in Output Control. If
          there is no Output Control option and "PRINT_FLOWS" is specified,
          then flow rates are printed for the last time step of each stress
          period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that SSM flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    sources : [pname, srctype, auxname]
        * pname (string) name of the package for which an auxiliary variable
          contains a source concentration.
        * srctype (string) type of the source. Must be AUX.
        * auxname (string) name of the auxiliary variable in the package PNAME
          that contains the source concentration.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    sources = ListTemplateGenerator(("gwt6", "ssm", "sources", "sources"))
    package_abbr = "gwtssm"
    _package_type = "ssm"
    dfn_file_name = "gwt-ssm.dfn"

    dfn = [
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
            "block sources",
            "name sources",
            "type recarray pname srctype auxname",
            "reader urword",
            "optional false",
        ],
        [
            "block sources",
            "name pname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
        ],
        [
            "block sources",
            "name srctype",
            "in_record true",
            "type string",
            "tagged false",
            "optional false",
            "reader urword",
        ],
        [
            "block sources",
            "name auxname",
            "in_record true",
            "type string",
            "tagged false",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_flows=None,
        save_flows=None,
        sources=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwtssm, self).__init__(
            model, "ssm", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.sources = self.build_mfdata("sources", sources)
        self._init_complete = True
