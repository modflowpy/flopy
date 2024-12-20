# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowSwfdisv2D(mfpackage.MFPackage):
    """
    ModflowSwfdisv2D defines a disv2d package within a swf6 model.

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
    nodes : integer
        * nodes (integer) is the number of cells per layer. This is a constant
          value for the grid and it applies to all layers.
    nvert : integer
        * nvert (integer) is the total number of (x, y) vertex pairs used to
          characterize the horizontal configuration of the model grid.
    bottom : [double]
        * bottom (double) is the bottom elevation for each cell.
    idomain : [integer]
        * idomain (integer) is an optional array that characterizes the
          existence status of a cell. If the IDOMAIN array is not specified,
          then all model cells exist within the solution. If the IDOMAIN value
          for a cell is 0, the cell does not exist in the simulation. Input and
          output values will be read and written for the cell, but internal to
          the program, the cell is excluded from the solution. If the IDOMAIN
          value for a cell is 1 or greater, the cell exists in the simulation.
          If the IDOMAIN value for a cell is -1, the cell does not exist in the
          simulation. Furthermore, the first existing cell above will be
          connected to the first existing cell below. This type of cell is
          referred to as a "vertical pass through" cell.
    vertices : [iv, xv, yv]
        * iv (integer) is the vertex number. Records in the VERTICES block must
          be listed in consecutive order from 1 to NVERT. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * xv (double) is the x-coordinate for the vertex.
        * yv (double) is the y-coordinate for the vertex.
    cell2d : [icell2d, xc, yc, ncvert, icvert]
        * icell2d (integer) is the CELL2D number. Records in the CELL2D block
          must be listed in consecutive order from the first to the last. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * xc (double) is the x-coordinate for the cell center.
        * yc (double) is the y-coordinate for the cell center.
        * ncvert (integer) is the number of vertices required to define the
          cell. There may be a different number of vertices for each cell.
        * icvert (integer) is an array of integer values containing vertex
          numbers (in the VERTICES block) used to define the cell. Vertices
          must be listed in clockwise order. Cells that are connected must
          share vertices. This argument is an index variable, which means that
          it should be treated as zero-based when working with FloPy and
          Python. Flopy will automatically subtract one when loading index
          variables and add one when writing index variables.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    bottom = ArrayTemplateGenerator(('swf6', 'disv2d', 'griddata',
                                     'bottom'))
    idomain = ArrayTemplateGenerator(('swf6', 'disv2d', 'griddata',
                                      'idomain'))
    vertices = ListTemplateGenerator(('swf6', 'disv2d', 'vertices',
                                      'vertices'))
    cell2d = ListTemplateGenerator(('swf6', 'disv2d', 'cell2d',
                                    'cell2d'))
    package_abbr = "swfdisv2d"
    _package_type = "disv2d"
    dfn_file_name = "swf-disv2d.dfn"

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
           ["block dimensions", "name nodes", "type integer",
            "reader urword", "optional false"],
           ["block dimensions", "name nvert", "type integer",
            "reader urword", "optional false"],
           ["block griddata", "name bottom", "type double precision",
            "shape (nodes)", "reader readarray", "layered false"],
           ["block griddata", "name idomain", "type integer",
            "shape (nodes)", "reader readarray", "layered false",
            "optional true"],
           ["block vertices", "name vertices", "type recarray iv xv yv",
            "shape (nvert)", "reader urword", "optional false"],
           ["block vertices", "name iv", "type integer", "in_record true",
            "tagged false", "reader urword", "optional false",
            "numeric_index true"],
           ["block vertices", "name xv", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"],
           ["block vertices", "name yv", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"],
           ["block cell2d", "name cell2d",
            "type recarray icell2d xc yc ncvert icvert", "shape (nodes)",
            "reader urword", "optional false"],
           ["block cell2d", "name icell2d", "type integer",
            "in_record true", "tagged false", "reader urword",
            "optional false", "numeric_index true"],
           ["block cell2d", "name xc", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"],
           ["block cell2d", "name yc", "type double precision",
            "in_record true", "tagged false", "reader urword",
            "optional false"],
           ["block cell2d", "name ncvert", "type integer", "in_record true",
            "tagged false", "reader urword", "optional false"],
           ["block cell2d", "name icvert", "type integer", "shape (ncvert)",
            "in_record true", "tagged false", "reader urword",
            "optional false", "numeric_index true"]]

    def __init__(self, model, loading_package=False, length_units=None,
                 nogrb=None, xorigin=None, yorigin=None, angrot=None,
                 export_array_ascii=None, nodes=None, nvert=None, bottom=None,
                 idomain=None, vertices=None, cell2d=None, filename=None,
                 pname=None, **kwargs):
        super().__init__(model, "disv2d", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)
        self.nogrb = self.build_mfdata("nogrb", nogrb)
        self.xorigin = self.build_mfdata("xorigin", xorigin)
        self.yorigin = self.build_mfdata("yorigin", yorigin)
        self.angrot = self.build_mfdata("angrot", angrot)
        self.export_array_ascii = self.build_mfdata("export_array_ascii",
                                                    export_array_ascii)
        self.nodes = self.build_mfdata("nodes", nodes)
        self.nvert = self.build_mfdata("nvert", nvert)
        self.bottom = self.build_mfdata("bottom", bottom)
        self.idomain = self.build_mfdata("idomain", idomain)
        self.vertices = self.build_mfdata("vertices", vertices)
        self.cell2d = self.build_mfdata("cell2d", cell2d)
        self._init_complete = True
