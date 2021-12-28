# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 22, 2021 17:36:26 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowUtlsfrtab(mfpackage.MFPackage):
    """
    ModflowUtlsfrtab defines a sfrtab package within a utl model.

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
          reach cross-section table. There must be NROW rows of data in the
          TABLE block.
    ncol : integer
        * ncol (integer) integer value specifying the number of columns in the
          reach cross-section table. There must be NCOL columns of data in the
          TABLE block. Currently, NCOL must be equal to 2.
    table : [xfraction, depth]
        * xfraction (double) real value that defines the station (x) data for
          the cross-section as a fraction of the width (RWID) of the reach.
        * depth (double) real value that defines the elevation (z) data for the
          cross-section as a depth relative to the top elevation of the reach
          (RTP) and corresponding to the station data on the same line.
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
            "type recarray xfraction depth",
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
            "name depth",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
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
            model, "sfrtab", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.nrow = self.build_mfdata("nrow", nrow)
        self.ncol = self.build_mfdata("ncol", ncol)
        self.table = self.build_mfdata("table", table)
        self._init_complete = True


class UtlsfrtabPackages(mfpackage.MFChildPackages):
    """
    UtlsfrtabPackages is a container class for the ModflowUtlsfrtab class.

    Methods
    ----------
    initialize
        Initializes a new ModflowUtlsfrtab package removing any sibling child
        packages attached to the same parent package. See ModflowUtlsfrtab init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtlsfrtab package to the container. See ModflowUtlsfrtab
        init documentation for definition of parameters.
    """

    package_abbr = "utlsfrtabpackages"

    def initialize(
        self, nrow=None, ncol=None, table=None, filename=None, pname=None
    ):
        new_package = ModflowUtlsfrtab(
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
        new_package = ModflowUtlsfrtab(
            self._model,
            nrow=nrow,
            ncol=ncol,
            table=table,
            filename=filename,
            pname=pname,
            parent_file=self._cpparent,
        )
        self._append_package(new_package, filename)
