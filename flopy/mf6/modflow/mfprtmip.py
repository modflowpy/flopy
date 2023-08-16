# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 16, 2023 03:03:23 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowPrtmip(mfpackage.MFPackage):
    """
    ModflowPrtmip defines a mip package within a prt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    porosity : [double]
        * porosity (double) is the aquifer porosity.
    retfactor : [double]
        * retfactor (double) is a real value by which velocity is divided
          within a given cell. RETFACTOR can be used to account for solute
          retardation, i.e., the apparent effect of linear sorption on the
          velocity of particles that track solute advection. RETFACTOR may be
          assigned any real value. A RETFACTOR value greater than 1 represents
          particle retardation (slowing), and a value of 1 represents no
          retardation. The effect of specifying a RETFACTOR value for each cell
          is the same as the effect of directly multiplying the POROSITY in
          each cell by the proposed RETFACTOR value for each cell. RETFACTOR
          allows conceptual isolation of effects such as retardation from the
          effect of porosity. The default value is 1.
    izone : [integer]
        * izone (integer) is an integer zone number assigned to each cell.
          IZONE may be positive, negative, or zero. If the ISTOPZONE option is
          set to any value other than zero in a PRP Package, particles released
          by that PRP Package terminate if they enter a cell whose IZONE value
          matches ISTOPZONE. If ISTOPZONE is not specified or is set to zero in
          a PRP Package, IZONE has no effect on the termination of particles
          released by that PRP Package.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    porosity = ArrayTemplateGenerator(("prt6", "mip", "griddata", "porosity"))
    retfactor = ArrayTemplateGenerator(
        ("prt6", "mip", "griddata", "retfactor")
    )
    izone = ArrayTemplateGenerator(("prt6", "mip", "griddata", "izone"))
    package_abbr = "prtmip"
    _package_type = "mip"
    dfn_file_name = "prt-mip.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block griddata",
            "name porosity",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block griddata",
            "name retfactor",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name izone",
            "type integer",
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
        porosity=None,
        retfactor=None,
        izone=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "mip", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.porosity = self.build_mfdata("porosity", porosity)
        self.retfactor = self.build_mfdata("retfactor", retfactor)
        self.izone = self.build_mfdata("izone", izone)
        self._init_complete = True
