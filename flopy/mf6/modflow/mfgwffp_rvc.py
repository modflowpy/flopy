# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 12, 2022 20:29:54 UTC
from flopy.mf6 import mfpackage
from flopy.mf6.data.mfdatautil import ListTemplateGenerator


class ModflowGwffp_Rvc(mfpackage.MFPackage):
    """
    ModflowGwffp_Rvc defines a fp_rvc package that is a flopy plugin extension
    of a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    maxbound : integer
        * maxbound (integer) integer value specifying the maximum number of
          rivers cells that will be specified for use during any stress period.
    stress_period_data : [pkg_name, rvc_bound, cond_up, cond_down]
        * pkg_name (string) name of riv package that rvc is modifying
        * rvc_bound (string) is the river cell bound name specified in the
          bound name of the riv package
        * cond_up (double) is the riverbed hydraulic conductance for upward
          flow.
        * cond_down (double) is the riverbed hydraulic conductance for downward
          flow.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    stress_period_data = ListTemplateGenerator(
        ("gwf6", "fp_rvc", "period", "stress_period_data")
    )
    package_abbr = "gwffp_rvc"
    _package_type = "fp_rvc"
    dfn_file_name = "gwf-fp_rvc.dfn"

    dfn = [
        [
            "header",
            "flopy-plugin rvc",
        ],
        [
            "block dimensions",
            "name maxbound",
            "type integer",
            "reader urword",
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
            "name stress_period_data",
            "type recarray pkg_name rvc_bound cond_up cond_down",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block period",
            "name pkg_name",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name rvc_bound",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name cond_up",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name cond_down",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        maxbound=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        **kwargs
    ):
        super().__init__(
            model, "fp_rvc", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
