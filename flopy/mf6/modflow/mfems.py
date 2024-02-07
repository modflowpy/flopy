# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 07, 2024 20:16:08 UTC
from .. import mfpackage


class ModflowEms(mfpackage.MFPackage):
    """
    ModflowEms defines a ems package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    package_abbr = "ems"
    _package_type = "ems"
    dfn_file_name = "sln-ems.dfn"

    dfn = [
        [
            "header",
            ["solution_package", "*"],
        ],
    ]

    def __init__(
        self,
        simulation,
        loading_package=False,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            simulation, "ems", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self._init_complete = True
