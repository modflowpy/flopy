# autogenerated file, do not modify

from os import PathLike, curdir
from typing import Union

from flopy.mf6.data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator
from flopy.mf6.mfpackage import MFChildPackages, MFPackage


class ModflowEms(MFPackage):
    """
    ModflowEms defines a EMS package.

    Parameters
    ----------

    """

    package_abbr = "ems"
    _package_type = "ems"
    dfn_file_name = "sln-ems.dfn"
    dfn = [["header", ["solution_package", "*"]]]

    def __init__(
        self,
        simulation,
        loading_package=False,
        filename=None,
        pname=None,
        **kwargs,
    ):
        """
        ModflowEms defines a EMS package.

        Parameters
        ----------
        simulation
            Simulation that this package is a part of. Package is automatically
            added to simulation when it is initialized.
        loading_package : bool
            Do not set this parameter. It is intended for debugging and internal
            processing purposes only.

        filename : str
            File name for this package.
        pname : str
            Package name for this package.
        parent_file : MFPackage
            Parent package file that references this package. Only needed for
            utility packages (mfutl*). For example, mfutllaktab package must have
            a mfgwflak package parent_file.
        """

        super().__init__(simulation, "ems", filename, pname, loading_package, **kwargs)

        self._init_complete = True
