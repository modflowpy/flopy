# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowSwfdis2D(mfpackage.MFPackage):
    """
    ModflowSwfdis2D defines a dis2d package within a swf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
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
    export_array_ascii : boolean
        * export_array_ascii (boolean) keyword that specifies input griddata
          arrays should be written to layered ascii output files.
    nrow : integer
        * nrow (integer) is the number of rows in the model grid.
    ncol : integer
        * ncol (integer) is the number of columns in the model grid.
    delr : [double]
        * delr (double) is the column spacing in the row direction.
    delc : [double]
        * delc (double) is the row spacing in the column direction.
    bottom : [double]
        * bottom (double) is the bottom elevation for each cell.
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
    delr = ArrayTemplateGenerator(('swf6', 'dis2d', 'griddata', 'delr'))
    delc = ArrayTemplateGenerator(('swf6', 'dis2d', 'griddata', 'delc'))
    bottom = ArrayTemplateGenerator(('swf6', 'dis2d', 'griddata',
                                     'bottom'))
    idomain = ArrayTemplateGenerator(('swf6', 'dis2d', 'griddata',
                                      'idomain'))
    package_abbr = "swfdis2d"
    _package_type = "dis2d"
    dfn_file_name = "swf-dis2d.dfn"

    dfn = [
           ["header", ],
           ["block options", "name length_units", "type string",
            "reader urword", "optional true"],
           ["block options", "name nogrb", "type keyword", "reader urword",
            "optional true"],
           ["block options", "name xorigin", "type double precision",
            "reader urword", "optional true"],
           ["block options", "name yorigin", "type double precision",
            "reader urword", "optional true"],
           ["block options", "name angrot", "type double precision",
            "reader urword", "optional true"],
           ["block options", "name export_array_ascii", "type keyword",
            "reader urword", "optional true", "mf6internal export_ascii"],
           ["block dimensions", "name nrow", "type integer",
            "reader urword", "optional false", "default_value 2"],
           ["block dimensions", "name ncol", "type integer",
            "reader urword", "optional false", "default_value 2"],
           ["block griddata", "name delr", "type double precision",
            "shape (ncol)", "reader readarray", "default_value 1.0"],
           ["block griddata", "name delc", "type double precision",
            "shape (nrow)", "reader readarray", "default_value 1.0"],
           ["block griddata", "name bottom", "type double precision",
            "shape (ncol, nrow)", "reader readarray", "layered false",
            "default_value 0."],
           ["block griddata", "name idomain", "type integer",
            "shape (ncol, nrow)", "reader readarray", "layered false",
            "optional true"]]

    def __init__(self, model, loading_package=False, length_units=None,
                 nogrb=None, xorigin=None, yorigin=None, angrot=None,
                 export_array_ascii=None, nrow=2, ncol=2, delr=1.0, delc=1.0,
                 bottom=0., idomain=None, filename=None, pname=None, **kwargs):
        super().__init__(model, "dis2d", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)
        self.nogrb = self.build_mfdata("nogrb", nogrb)
        self.xorigin = self.build_mfdata("xorigin", xorigin)
        self.yorigin = self.build_mfdata("yorigin", yorigin)
        self.angrot = self.build_mfdata("angrot", angrot)
        self.export_array_ascii = self.build_mfdata("export_array_ascii",
                                                    export_array_ascii)
        self.nrow = self.build_mfdata("nrow", nrow)
        self.ncol = self.build_mfdata("ncol", ncol)
        self.delr = self.build_mfdata("delr", delr)
        self.delc = self.build_mfdata("delc", delc)
        self.bottom = self.build_mfdata("bottom", bottom)
        self.idomain = self.build_mfdata("idomain", idomain)
        self._init_complete = True
