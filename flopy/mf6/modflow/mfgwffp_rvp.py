# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 12, 2022 20:29:54 UTC
from flopy.mf6 import mfpackage
from flopy.mf6.data.mfdatautil import ListTemplateGenerator


class ModflowGwffp_Rvp(mfpackage.MFPackage):
    """
    ModflowGwffp_Rvp defines a fp_rvp package that is a flopy plugin extension
    of a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) *** ADD DESCRIPTION HERE ***
    print_flows : boolean
        * print_flows (boolean) *** ADD DESCRIPTION HERE ***
    save_flows : boolean
        * save_flows (boolean) *** ADD DESCRIPTION HERE ***
    auxiliary : [string]
        * auxiliary (string) *** ADD DESCRIPTION HERE ***
    boundnames : boolean
        * boundnames (boolean) *** ADD DESCRIPTION HERE ***
    maxbound : integer
        * maxbound (integer) maximum number...
    stress_period_data : [cellid, stage, cond_up, cond_down, rbot, aux,
      boundname]
        * cellid ((integer, ...)) *** ADD DESCRIPTION HERE *** This argument is
          an index variable, which means that it should be treated as zero-
          based when working with FloPy and Python. Flopy will automatically
          subtract one when loading index variables and add one when writing
          index variables.
        * stage (double) *** ADD DESCRIPTION HERE ***
        * cond_up (double) *** ADD DESCRIPTION HERE ***
        * cond_down (double) *** ADD DESCRIPTION HERE ***
        * rbot (double) *** ADD DESCRIPTION HERE ***
        * aux (double) *** ADD DESCRIPTION HERE ***
        * boundname (string) *** ADD DESCRIPTION HERE ***
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(
        ("gwf6", "fp_rvp", "options", "auxiliary")
    )
    stress_period_data = ListTemplateGenerator(
        ("gwf6", "fp_rvp", "period", "stress_period_data")
    )
    package_abbr = "gwffp_rvp"
    _package_type = "fp_rvp"
    dfn_file_name = "gwf-fp_rvp.dfn"

    dfn = [
        [
            "header",
            "flopy-plugin rvp",
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
            "name auxiliary",
            "type string",
            "shape (naux)",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name boundnames",
            "type keyword",
            "reader urword",
            "optional true",
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
            "type recarray cellid stage cond_up cond_down rbot aux boundname",
            "shape (maxbound)",
            "reader urword",
        ],
        [
            "block period",
            "name cellid",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name stage",
            "type double precision",
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
        [
            "block period",
            "name rbot",
            "type double precision",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name aux",
            "type double precision",
            "shape (naux)",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional True",
        ],
        [
            "block period",
            "name boundname",
            "type string",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional True",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_input=None,
        print_flows=None,
        save_flows=None,
        auxiliary=None,
        boundnames=None,
        maxbound=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        **kwargs
    ):
        super().__init__(
            model, "fp_rvp", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
