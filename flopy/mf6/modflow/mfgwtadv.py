# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage


class ModflowGwtadv(mfpackage.MFPackage):
    """
    ModflowGwtadv defines a adv package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    scheme : string
        * scheme (string) scheme used to solve the advection term. Can be
          upstream, central, or TVD. If not specified, upstream weighting is
          the default weighting scheme.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    package_abbr = "gwtadv"
    _package_type = "adv"
    dfn_file_name = "gwt-adv.dfn"

    dfn = [
        [
            "block options",
            "name scheme",
            "type string",
            "valid central upstream tvd",
            "reader urword",
            "optional true",
        ]
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        scheme=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "adv", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.scheme = self.build_mfdata("scheme", scheme)
        self._init_complete = True
