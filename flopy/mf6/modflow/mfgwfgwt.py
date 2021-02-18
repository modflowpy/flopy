# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage


class ModflowGwfgwt(mfpackage.MFPackage):
    """
    ModflowGwfgwt defines a gwfgwt package.

    Parameters
    ----------
    simulation : MFSimulation
        Simulation that this package is a part of. Package is automatically
        added to simulation when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    exgtype : <string>
        * is the exchange type (GWF-GWF or GWF-GWT).
    exgmnamea : <string>
        * is the name of the first model that is part of this exchange.
    exgmnameb : <string>
        * is the name of the second model that is part of this exchange.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    package_abbr = "gwfgwt"
    _package_type = "gwfgwt"
    dfn_file_name = "exg-gwfgwt.dfn"

    dfn = []

    def __init__(
        self,
        simulation,
        loading_package=False,
        exgtype=None,
        exgmnamea=None,
        exgmnameb=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwfgwt, self).__init__(
            simulation, "gwfgwt", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.exgtype = exgtype

        self.exgmnamea = exgmnamea

        self.exgmnameb = exgmnameb

        simulation.register_exchange_file(self)

        self._init_complete = True
