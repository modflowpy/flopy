# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtllaktab(mfpackage.MFPackage):
    """
    ModflowUtllaktab defines a tab package within a utl model.

    Parameters
    ----------
    nrow : integer
        * nrow (integer) integer value specifying the number of rows in the
          lake table. There must be texttt{nrow} rows of data in the
          texttt{TABLE} block.
    ncol : integer
        * ncol (integer) integer value specifying the number of colums in the
          lake table. There must be texttt{ncol} columns of data in the
          texttt{TABLE} block. For lakes with texttt{HORIZONTAL} and/or
          texttt{VERTICAL} texttt{ctype} connections, texttt{NROW} must be
          equal to 3. For lakes with texttt{EMBEDDEDH} or texttt{EMBEDDEDV}
          texttt{ctype} connections, texttt{NROW} must be equal to 4.
    laktabrecarray : [stage, volume, sarea, barea]
        * stage (double) real value that defines the stage corresponding to the
          remaining data on the line.
        * volume (double) real value that defines the lake volume corresponding
          to the stage specified on the line.
        * sarea (double) real value that defines the lake surface area
          corresponding to the stage specified on the line.
        * barea (double) real value that defines the lake-texttt{GWF} exchange
          area corresponding to the stage specified on the line. texttt{barea}
          is only specified if the texttt{claktype} for the lake is
          texttt{EMBEDDEDH} or texttt{EMBEDDEDV}.

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
        super(ModflowUtllaktab, self).__init__(model, "tab", fname, pname,
                                               add_to_package_list, parent_file)        

        # set up variables
        self.nrow = self.build_mfdata("nrow",  nrow)
        self.ncol = self.build_mfdata("ncol",  ncol)
        self.laktabrecarray = self.build_mfdata("laktabrecarray", 
                                                laktabrecarray)
