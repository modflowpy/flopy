# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowGwecnd(mfpackage.MFPackage):
    """
    ModflowGwecnd defines a cnd package within a gwe6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    xt3d_off : boolean
        * xt3d_off (boolean) deactivate the xt3d method and use the faster and
          less accurate approximation. This option may provide a fast and
          accurate solution under some circumstances, such as when flow aligns
          with the model grid, there is no mechanical dispersion, or when the
          longitudinal and transverse dispersivities are equal. This option may
          also be used to assess the computational demand of the XT3D approach
          by noting the run time differences with and without this option on.
    xt3d_rhs : boolean
        * xt3d_rhs (boolean) add xt3d terms to right-hand side, when possible.
          This option uses less memory, but may require more iterations.
    export_array_ascii : boolean
        * export_array_ascii (boolean) keyword that specifies input griddata
          arrays should be written to layered ascii output files.
    alh : [double]
        * alh (double) longitudinal dispersivity in horizontal direction. If
          flow is strictly horizontal, then this is the longitudinal
          dispersivity that will be used. If flow is not strictly horizontal or
          strictly vertical, then the longitudinal dispersivity is a function
          of both ALH and ALV. If mechanical dispersion is represented (by
          specifying any dispersivity values) then this array is required.
    alv : [double]
        * alv (double) longitudinal dispersivity in vertical direction. If flow
          is strictly vertical, then this is the longitudinal dispsersivity
          value that will be used. If flow is not strictly horizontal or
          strictly vertical, then the longitudinal dispersivity is a function
          of both ALH and ALV. If this value is not specified and mechanical
          dispersion is represented, then this array is set equal to ALH.
    ath1 : [double]
        * ath1 (double) transverse dispersivity in horizontal direction. This
          is the transverse dispersivity value for the second ellipsoid axis.
          If flow is strictly horizontal and directed in the x direction (along
          a row for a regular grid), then this value controls spreading in the
          y direction. If mechanical dispersion is represented (by specifying
          any dispersivity values) then this array is required.
    ath2 : [double]
        * ath2 (double) transverse dispersivity in horizontal direction. This
          is the transverse dispersivity value for the third ellipsoid axis. If
          flow is strictly horizontal and directed in the x direction (along a
          row for a regular grid), then this value controls spreading in the z
          direction. If this value is not specified and mechanical dispersion
          is represented, then this array is set equal to ATH1.
    atv : [double]
        * atv (double) transverse dispersivity when flow is in vertical
          direction. If flow is strictly vertical and directed in the z
          direction, then this value controls spreading in the x and y
          directions. If this value is not specified and mechanical dispersion
          is represented, then this array is set equal to ATH2.
    ktw : [double]
        * ktw (double) thermal conductivity of the simulated fluid. Note that
          the CND Package does not account for the tortuosity of the flow paths
          when solving for the conductive spread of heat. If tortuosity plays
          an important role in the thermal conductivity calculation, its effect
          should be reflected in the value specified for KTW.
    kts : [double]
        * kts (double) thermal conductivity of the aquifer material
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    alh = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "alh"))
    alv = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "alv"))
    ath1 = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "ath1"))
    ath2 = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "ath2"))
    atv = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "atv"))
    ktw = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "ktw"))
    kts = ArrayTemplateGenerator(("gwe6", "cnd", "griddata", "kts"))
    package_abbr = "gwecnd"
    _package_type = "cnd"
    dfn_file_name = "gwe-cnd.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name xt3d_off",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name xt3d_rhs",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
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
            "block griddata",
            "name alh",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name alv",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name ath1",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name ath2",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name atv",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name ktw",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name kts",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        xt3d_off=None,
        xt3d_rhs=None,
        export_array_ascii=None,
        alh=None,
        alv=None,
        ath1=None,
        ath2=None,
        atv=None,
        ktw=None,
        kts=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "cnd", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.xt3d_off = self.build_mfdata("xt3d_off", xt3d_off)
        self.xt3d_rhs = self.build_mfdata("xt3d_rhs", xt3d_rhs)
        self.export_array_ascii = self.build_mfdata(
            "export_array_ascii", export_array_ascii
        )
        self.alh = self.build_mfdata("alh", alh)
        self.alv = self.build_mfdata("alv", alv)
        self.ath1 = self.build_mfdata("ath1", ath1)
        self.ath2 = self.build_mfdata("ath2", ath2)
        self.atv = self.build_mfdata("atv", atv)
        self.ktw = self.build_mfdata("ktw", ktw)
        self.kts = self.build_mfdata("kts", kts)
        self._init_complete = True
