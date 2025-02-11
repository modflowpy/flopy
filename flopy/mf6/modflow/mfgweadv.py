# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 11, 2025 01:24:12 UTC
from .. import mfpackage


class ModflowGweadv(mfpackage.MFPackage):
    """
    ModflowGweadv defines a adv package within a gwe6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    scheme : string
        * scheme (string) scheme used to solve the advection term. Can be
          upstream, central, or TVD. If not specified, upstream weighting is
          the default weighting scheme.
    ats_percel : double
        * ats_percel (double) fractional cell distance submitted by the ADV
          Package to the adaptive time stepping (ATS) package. If ATS_PERCEL is
          specified and the ATS Package is active, a time step calculation will
          be made for each cell based on flow through the cell and cell
          properties. The largest time step will be calculated such that the
          advective fractional cell distance (ATS_PERCEL) is not exceeded for
          any active cell in the grid. This time-step constraint will be
          submitted to the ATS Package, perhaps with constraints submitted by
          other packages, in the calculation of the time step. ATS_PERCEL must
          be greater than zero. If a value of zero is specified for ATS_PERCEL
          the program will automatically reset it to an internal no data value
          to indicate that time steps should not be subject to this constraint.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    package_abbr = "gweadv"
    _package_type = "adv"
    dfn_file_name = "gwe-adv.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name scheme",
            "type string",
            "valid central upstream tvd",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name ats_percel",
            "type double precision",
            "reader urword",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        scheme=None,
        ats_percel=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "adv", filename, pname, loading_package, **kwargs)

        # set up variables
        self.scheme = self.build_mfdata("scheme", scheme)
        self.ats_percel = self.build_mfdata("ats_percel", ats_percel)
        self._init_complete = True
