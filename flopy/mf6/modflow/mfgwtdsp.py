# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowGwtdsp(mfpackage.MFPackage):
    """
    ModflowGwtdsp defines a dsp package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
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
    diffc : [double]
        * diffc (double) effective molecular diffusion coefficient.
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
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    diffc = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "diffc"))
    alh = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "alh"))
    alv = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "alv"))
    ath1 = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "ath1"))
    ath2 = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "ath2"))
    atv = ArrayTemplateGenerator(("gwt6", "dsp", "griddata", "atv"))
    package_abbr = "gwtdsp"
    _package_type = "dsp"
    dfn_file_name = "gwt-dsp.dfn"

    dfn = [
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
            "block griddata",
            "name diffc",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
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
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        xt3d_off=None,
        xt3d_rhs=None,
        diffc=None,
        alh=None,
        alv=None,
        ath1=None,
        ath2=None,
        atv=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwtdsp, self).__init__(
            model, "dsp", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.xt3d_off = self.build_mfdata("xt3d_off", xt3d_off)
        self.xt3d_rhs = self.build_mfdata("xt3d_rhs", xt3d_rhs)
        self.diffc = self.build_mfdata("diffc", diffc)
        self.alh = self.build_mfdata("alh", alh)
        self.alv = self.build_mfdata("alv", alv)
        self.ath1 = self.build_mfdata("ath1", ath1)
        self.ath2 = self.build_mfdata("ath2", ath2)
        self.atv = self.build_mfdata("atv", atv)
        self._init_complete = True
