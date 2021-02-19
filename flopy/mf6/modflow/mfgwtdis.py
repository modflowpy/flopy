# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowGwtdis(mfpackage.MFPackage):
    """
    ModflowGwtdis defines a dis package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    length_units : string
        * length_units (string) is the length units used for this model. Values
          can be "FEET", "METERS", or "CENTIMETERS". If not specified, the
          default is "UNKNOWN".
    nogrb : boolean
        * nogrb (boolean) keyword to deactivate writing of the binary grid
          file.
    xorigin : double
        * xorigin (double) x-position of the lower-left corner of the model
          grid. A default value of zero is assigned if not specified. The value
          for XORIGIN does not affect the model simulation, but it is written
          to the binary grid file so that postprocessors can locate the grid in
          space.
    yorigin : double
        * yorigin (double) y-position of the lower-left corner of the model
          grid. If not specified, then a default value equal to zero is used.
          The value for YORIGIN does not affect the model simulation, but it is
          written to the binary grid file so that postprocessors can locate the
          grid in space.
    angrot : double
        * angrot (double) counter-clockwise rotation angle (in degrees) of the
          lower-left corner of the model grid. If not specified, then a default
          value of 0.0 is assigned. The value for ANGROT does not affect the
          model simulation, but it is written to the binary grid file so that
          postprocessors can locate the grid in space.
    nlay : integer
        * nlay (integer) is the number of layers in the model grid.
    nrow : integer
        * nrow (integer) is the number of rows in the model grid.
    ncol : integer
        * ncol (integer) is the number of columns in the model grid.
    delr : [double]
        * delr (double) is the column spacing in the row direction.
    delc : [double]
        * delc (double) is the row spacing in the column direction.
    top : [double]
        * top (double) is the top elevation for each cell in the top model
          layer.
    botm : [double]
        * botm (double) is the bottom elevation for each cell.
    idomain : [integer]
        * idomain (integer) is an optional array that characterizes the
          existence status of a cell. If the IDOMAIN array is not specified,
          then all model cells exist within the solution. If the IDOMAIN value
          for a cell is 0, the cell does not exist in the simulation. Input and
          output values will be read and written for the cell, but internal to
          the program, the cell is excluded from the solution. If the IDOMAIN
          value for a cell is 1, the cell exists in the simulation. If the
          IDOMAIN value for a cell is -1, the cell does not exist in the
          simulation. Furthermore, the first existing cell above will be
          connected to the first existing cell below. This type of cell is
          referred to as a "vertical pass through" cell.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    delr = ArrayTemplateGenerator(("gwt6", "dis", "griddata", "delr"))
    delc = ArrayTemplateGenerator(("gwt6", "dis", "griddata", "delc"))
    top = ArrayTemplateGenerator(("gwt6", "dis", "griddata", "top"))
    botm = ArrayTemplateGenerator(("gwt6", "dis", "griddata", "botm"))
    idomain = ArrayTemplateGenerator(("gwt6", "dis", "griddata", "idomain"))
    package_abbr = "gwtdis"
    _package_type = "dis"
    dfn_file_name = "gwt-dis.dfn"

    dfn = [
        [
            "block options",
            "name length_units",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name nogrb",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name xorigin",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name yorigin",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name angrot",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block dimensions",
            "name nlay",
            "type integer",
            "reader urword",
            "optional false",
            "default_value 1",
        ],
        [
            "block dimensions",
            "name nrow",
            "type integer",
            "reader urword",
            "optional false",
            "default_value 2",
        ],
        [
            "block dimensions",
            "name ncol",
            "type integer",
            "reader urword",
            "optional false",
            "default_value 2",
        ],
        [
            "block griddata",
            "name delr",
            "type double precision",
            "shape (ncol)",
            "reader readarray",
            "default_value 1.0",
        ],
        [
            "block griddata",
            "name delc",
            "type double precision",
            "shape (nrow)",
            "reader readarray",
            "default_value 1.0",
        ],
        [
            "block griddata",
            "name top",
            "type double precision",
            "shape (ncol, nrow)",
            "reader readarray",
            "default_value 1.0",
        ],
        [
            "block griddata",
            "name botm",
            "type double precision",
            "shape (ncol, nrow, nlay)",
            "reader readarray",
            "layered true",
            "default_value 0.",
        ],
        [
            "block griddata",
            "name idomain",
            "type integer",
            "shape (ncol, nrow, nlay)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        length_units=None,
        nogrb=None,
        xorigin=None,
        yorigin=None,
        angrot=None,
        nlay=1,
        nrow=2,
        ncol=2,
        delr=1.0,
        delc=1.0,
        top=1.0,
        botm=0.0,
        idomain=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwtdis, self).__init__(
            model, "dis", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)
        self.nogrb = self.build_mfdata("nogrb", nogrb)
        self.xorigin = self.build_mfdata("xorigin", xorigin)
        self.yorigin = self.build_mfdata("yorigin", yorigin)
        self.angrot = self.build_mfdata("angrot", angrot)
        self.nlay = self.build_mfdata("nlay", nlay)
        self.nrow = self.build_mfdata("nrow", nrow)
        self.ncol = self.build_mfdata("ncol", ncol)
        self.delr = self.build_mfdata("delr", delr)
        self.delc = self.build_mfdata("delc", delc)
        self.top = self.build_mfdata("top", top)
        self.botm = self.build_mfdata("botm", botm)
        self.idomain = self.build_mfdata("idomain", idomain)
        self._init_complete = True
