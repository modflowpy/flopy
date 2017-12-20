from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtltab(mfpackage.MFPackage):
    """
    ModflowUtltab defines a tab package within a utl model.

    Attributes
    ----------
    nrow : (nrow : integer)
        nrow : integer value specifying the number of rows in the lake table.
          There must be nrow rows of data in the TABLE block.
    ncol : (ncol : integer)
        ncol : integer value specifying the number of colums in the lake table.
          There must be ncol columns of data in the TABLE
          block. For lakes with HORIZONTAL and/or VERTICAL
          ctype connections, NROW must be equal to 3. For
          lakes with EMBEDDEDH or EMBEDDEDV ctype
          connections, NROW must be equal to 4.
    laktabrecarray : [(stage : double), (volume : double), (sarea : double), (barea
      : double)]
        stage : real value that defines the stage corresponding to the
          remaining data on the line.
        volume : real value that defines the lake volume corresponding to the
          stage specified on the line.
        sarea : real value that defines the lake surface area corresponding to
          the stage specified on the line.
        barea : real value that defines the lake-GWF exchange area
          corresponding to the stage specified on the line. barea is
          only specified if the claktype for the lake is
          EMBEDDEDH or EMBEDDEDV.

    """
    laktabrecarray = ListTemplateGenerator(('tab', 'table', 
                                            'laktabrecarray'))
    package_abbr = "utltab"
    package_type = "tab"
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
        super(ModflowUtltab, self).__init__(model, "tab", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.nrow = self.build_mfdata("nrow",  nrow)
        self.ncol = self.build_mfdata("ncol",  ncol)
        self.laktabrecarray = self.build_mfdata("laktabrecarray", 
                                                laktabrecarray)
