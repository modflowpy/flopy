# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on March 07, 2022 16:59:43 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtllaktab(mfpackage.MFPackage):
    """
    ModflowUtllaktab defines a laktab package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    nrow : integer
        * nrow (integer) integer value specifying the number of rows in the
          lake table. There must be NROW rows of data in the TABLE block.
    ncol : integer
        * ncol (integer) integer value specifying the number of columns in the
          lake table. There must be NCOL columns of data in the TABLE block.
          For lakes with HORIZONTAL and/or VERTICAL CTYPE connections, NCOL
          must be equal to 3. For lakes with EMBEDDEDH or EMBEDDEDV CTYPE
          connections, NCOL must be equal to 4.
    table : [stage, volume, sarea, barea]
        * stage (double) real value that defines the stage corresponding to the
          remaining data on the line.
        * volume (double) real value that defines the lake volume corresponding
          to the stage specified on the line.
        * sarea (double) real value that defines the lake surface area
          corresponding to the stage specified on the line.
        * barea (double) real value that defines the lake-GWF exchange area
          corresponding to the stage specified on the line. BAREA is only
          specified if the CLAKTYPE for the lake is EMBEDDEDH or EMBEDDEDV.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    table = ListTemplateGenerator(("laktab", "table", "table"))
    package_abbr = "utllaktab"
    _package_type = "laktab"
    dfn_file_name = "utl-laktab.dfn"

    dfn = [
        [
            "header",
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
            "type recarray stage volume sarea barea",
            "shape (nrow)",
            "reader urword",
        ],
        [
            "block table",
            "name stage",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block table",
            "name volume",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block table",
            "name sarea",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block table",
            "name barea",
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
        parent_file=None,
    ):
        super().__init__(
            model, "laktab", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.nrow = self.build_mfdata("nrow", nrow)
        self.ncol = self.build_mfdata("ncol", ncol)
        self.table = self.build_mfdata("table", table)
        self._init_complete = True


class UtllaktabPackages(mfpackage.MFChildPackages):
    """
    UtllaktabPackages is a container class for the ModflowUtllaktab class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtllaktab package removing any sibling child
        packages attached to the same parent package. See ModflowUtllaktab init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtllaktab package to the container. See ModflowUtllaktab
        init documentation for definition of parameters.
    """

    package_abbr = "utllaktabpackages"

    def initialize(
        self, nrow=None, ncol=None, table=None, filename=None, pname=None
    ):
        new_package = ModflowUtllaktab(
            self._model,
            nrow=nrow,
            ncol=ncol,
            table=table,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._init_package(new_package, filename)

    def append_package(
        self, nrow=None, ncol=None, table=None, filename=None, pname=None
    ):
        new_package = ModflowUtllaktab(
            self._model,
            nrow=nrow,
            ncol=ncol,
            table=table,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._append_package(new_package, filename)
