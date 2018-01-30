# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtllaktab(mfpackage.MFPackage):
    """
    ModflowUtllaktab defines a tab package within a utl model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    add_to_package_list : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    nrow : integer
        * nrow (integer) integer value specifying the number of rows in the
          lake table. There must be NROW rows of data in the TABLE block.
    ncol : integer
        * ncol (integer) integer value specifying the number of colums in the
          lake table. There must be NCOL columns of data in the TABLE block.
          For lakes with HORIZONTAL and/or VERTICAL CTYPE connections, NROW
          must be equal to 3. For lakes with EMBEDDEDH or EMBEDDEDV CTYPE
          connections, NROW must be equal to 4.
    laktabrecarray : [stage, volume, sarea, barea]
        * stage (double) real value that defines the stage corresponding to the
          remaining data on the line.
        * volume (double) real value that defines the lake volume corresponding
          to the stage specified on the line.
        * sarea (double) real value that defines the lake surface area
          corresponding to the stage specified on the line.
        * barea (double) real value that defines the lake-GWF exchange area
          corresponding to the stage specified on the line. BAREA is only
          specified if the CLAKTYPE for the lake is EMBEDDEDH or EMBEDDEDV.
    fname : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    laktabrecarray = ListTemplateGenerator(('tab', 'table', 
                                            'laktabrecarray'))
    package_abbr = "utltab"
    package_type = "tab"
    dfn_file_name = "utl-lak-tab.dfn"

    dfn = [["block dimensions", "name nrow", "type integer", 
            "reader urword", "optional false"],
           ["block dimensions", "name ncol", "type integer", 
            "reader urword", "optional false"],
           ["block table", "name laktabrecarray", 
            "type recarray stage volume sarea barea", "shape (nrow)", 
            "reader urword"],
           ["block table", "name stage", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block table", "name volume", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block table", "name sarea", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword"],
           ["block table", "name barea", "type double precision", "shape", 
            "tagged false", "in_record true", "reader urword", 
            "optional true"]]

    def __init__(self, model, add_to_package_list=True, nrow=None, ncol=None,
                 laktabrecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowUtllaktab, self).__init__(model, "tab", fname, pname,
                                               add_to_package_list, parent_file)        

        # set up variables
        self.nrow = self.build_mfdata("nrow",  nrow)
        self.ncol = self.build_mfdata("ncol",  ncol)
        self.laktabrecarray = self.build_mfdata("laktabrecarray", 
                                                laktabrecarray)
