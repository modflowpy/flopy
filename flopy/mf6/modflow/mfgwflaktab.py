from .. import mfpackage


class ModflowGwflaktab(mfpackage.MFPackage):
    package_abbr = "gwflaktab"
    """
    ModflowGwflaktab defines a laktab package within a gwf6 model.

    Attributes
    ----------
    nrow : (nrow : integer)
        integer value specifying the number of rows in the lake table. There must be nrow rows of data in the TABLE block.
    ncol : (ncol : integer)
        integer value specifying the number of colums in the lake table. There must be ncol columns of data in the TABLE block. For lakes with HORIZONTAL and/or VERTICAL ctype connections, NROW must be equal to 3. For lakes with EMBEDDEDH or EMBEDDEDV ctype connections, NROW must be equal to 4.
    laktabrecarray : [(stage : double), (volume : double), (sarea : double), (barea : double)]
        stage : real value that defines the stage corresponding to the remaining data on the line.
        volume : real value that defines the lake volume corresponding to the stage specified on the line.
        sarea : real value that defines the lake surface area corresponding to the stage specified on the line.
        barea : real value that defines the lake-GWF exchange area corresponding to the stage specified on the line. barea is only specified if the claktype for the lake is EMBEDDEDH or EMBEDDEDV.

    """

    def __init__(self, model, add_to_package_list=True, nrow=None, ncol=None,
                 laktabrecarray=None, fname=None,
                 pname=None, parent_file=None):
        super(ModflowGwflaktab, self).__init__(model, "laktab", fname, pname,
                                               add_to_package_list,
                                               parent_file)

        # set up variables
        self.nrow = self.build_mfdata("nrow", nrow)

        self.ncol = self.build_mfdata("ncol", ncol)

        self.laktabrecarray = self.build_mfdata("laktabrecarray",
                                                laktabrecarray)
