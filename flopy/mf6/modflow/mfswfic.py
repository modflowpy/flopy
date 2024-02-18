# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2024 14:36:23 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowSwfic(mfpackage.MFPackage):
    """
    ModflowSwfic defines a ic package within a swf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    strt : [double]
        * strt (double) is the initial (starting) stage---that is, stage at the
          beginning of the SWF Model simulation. STRT must be specified for all
          SWF Model simulations. One value is read for every model reach.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    strt = ArrayTemplateGenerator(("swf6", "ic", "griddata", "strt"))
    package_abbr = "swfic"
    _package_type = "ic"
    dfn_file_name = "swf-ic.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block griddata",
            "name strt",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "default_value 0.0",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        strt=0.0,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "ic", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.strt = self.build_mfdata("strt", strt)
        self._init_complete = True
