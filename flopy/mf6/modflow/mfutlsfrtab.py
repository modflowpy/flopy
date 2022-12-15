# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 15, 2022 12:49:36 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlsfrtab(mfpackage.MFPackage):
    """
    ModflowUtlsfrtab defines a sfrtab package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    nrow : integer
        * nrow (integer) integer value specifying the number of rows in the
          reach cross-section table. There must be NROW rows of data in the
          TABLE block.
    ncol : integer
        * ncol (integer) integer value specifying the number of columns in the
          reach cross-section table. There must be NCOL columns of data in the
          TABLE block. NCOL must be equal to 2 if MANFRACTION is not specified
          or 3 otherwise.
    table : [xfraction, height, manfraction]
        * xfraction (double) real value that defines the station (x) data for
          the cross-section as a fraction of the width (RWID) of the reach.
          XFRACTION must be greater than or equal to zero but can be greater
          than one. XFRACTION values can be used to decrease or increase the
          width of a reach from the specified reach width (RWID).
        * height (double) real value that is the height relative to the top of
          the lowest elevation of the streambed (RTP) and corresponding to the
          station data on the same line. HEIGHT must be greater than or equal
          to zero and at least one cross-section height must be equal to zero.
        * manfraction (double) real value that defines the Manning's roughness
          coefficient data for the cross-section as a fraction of the Manning's
          roughness coefficient for the reach (MAN) and corresponding to the
          station data on the same line. MANFRACTION must be greater than zero.
          MANFRACTION is applied from the XFRACTION value on the same line to
          the XFRACTION value on the next line. Although a MANFRACTION value is
          specified on the last line, any value greater than zero can be
          applied to MANFRACTION(NROW). MANFRACTION is only specified if NCOL
          is 3. If MANFRACTION is not specified, the Manning's roughness
          coefficient for the reach (MAN) is applied to the entire cross-
          section.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    table = ListTemplateGenerator(("sfrtab", "table", "table"))
    package_abbr = "utlsfrtab"
    _package_type = "sfrtab"
    dfn_file_name = "utl-sfrtab.dfn"

    dfn = [
        [
            "header",
            "multi-package",
        ],
        [
            "block dimensions",
            "name nrow",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name ncol",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block table",
            "name table",
            "type recarray xfraction height manfraction",
            "shape (nrow)",
            "reader urword",
        ],
        [
            "block table",
            "name xfraction",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block table",
            "name height",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block table",
            "name manfraction",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        nrow=None,
        ncol=None,
        table=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "sfrtab", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.nrow = self.build_mfdata("nrow", nrow)
        self.ncol = self.build_mfdata("ncol", ncol)
        self.table = self.build_mfdata("table", table)
        self._init_complete = True
