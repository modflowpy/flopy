# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 11, 2025 01:24:12 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowGwfic(mfpackage.MFPackage):
    """
    ModflowGwfic defines a ic package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    export_array_ascii : boolean
        * export_array_ascii (boolean) keyword that specifies input griddata
          arrays should be written to layered ascii output files.
    export_array_netcdf : boolean
        * export_array_netcdf (boolean) keyword that specifies input griddata
          arrays should be written to the model output netcdf file.
    strt : [double]
        * strt (double) is the initial (starting) head---that is, head at the
          beginning of the GWF Model simulation. STRT must be specified for all
          simulations, including steady-state simulations. One value is read
          for every model cell. For simulations in which the first stress
          period is steady state, the values used for STRT generally do not
          affect the simulation (exceptions may occur if cells go dry and (or)
          rewet). The execution time, however, will be less if STRT includes
          hydraulic heads that are close to the steady-state solution. A head
          value lower than the cell bottom can be provided if a cell should
          start as dry.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    strt = ArrayTemplateGenerator(("gwf6", "ic", "griddata", "strt"))
    package_abbr = "gwfic"
    _package_type = "ic"
    dfn_file_name = "gwf-ic.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name export_array_ascii",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal export_ascii",
        ],
        [
            "block options",
            "name export_array_netcdf",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal export_nc",
            "extended true",
        ],
        [
            "block griddata",
            "name strt",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "netcdf true",
            "default_value 1.0",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        export_array_ascii=None,
        export_array_netcdf=None,
        strt=1.0,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "ic", filename, pname, loading_package, **kwargs)

        # set up variables
        self.export_array_ascii = self.build_mfdata(
            "export_array_ascii", export_array_ascii
        )
        self.export_array_netcdf = self.build_mfdata(
            "export_array_netcdf", export_array_netcdf
        )
        self.strt = self.build_mfdata("strt", strt)
        self._init_complete = True
