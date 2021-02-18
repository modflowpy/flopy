# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwtfmi(mfpackage.MFPackage):
    """
    ModflowGwtfmi defines a fmi package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    flow_imbalance_correction : boolean
        * flow_imbalance_correction (boolean) correct for an imbalance in flows
          by assuming that any residual flow error comes in or leaves at the
          concentration of the cell. When this option is activated, the GWT
          Model budget written to the listing file will contain two additional
          entries: FLOW-ERROR and FLOW-CORRECTION. These two entries will be
          equal but opposite in sign. The FLOW-CORRECTION term is a mass flow
          that is added to offset the error caused by an imprecise flow
          balance. If these terms are not relatively small, the flow model
          should be rerun with stricter convergence tolerances.
    packagedata : [flowtype, fname]
        * flowtype (string) is the word GWFBUDGET, GWFHEAD, GWFMOVER or the
          name of an advanced GWF stress package. If GWFBUDGET is specified,
          then the corresponding file must be a budget file from a previous GWF
          Model run. If an advanced GWF stress package name appears then the
          corresponding file must be the budget file saved by a LAK, SFR, MAW
          or UZF Package.
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
        ("gwt6", "fmi", "packagedata", "packagedata")
    )
    package_abbr = "gwtfmi"
    _package_type = "fmi"
    dfn_file_name = "gwt-fmi.dfn"

    dfn = [
        [
            "block options",
            "name flow_imbalance_correction",
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
        flow_imbalance_correction=None,
        packagedata=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwtfmi, self).__init__(
            model, "fmi", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.flow_imbalance_correction = self.build_mfdata(
            "flow_imbalance_correction", flow_imbalance_correction
        )
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self._init_complete = True
