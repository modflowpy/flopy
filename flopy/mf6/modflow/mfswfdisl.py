# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2024 14:36:23 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowSwfdisl(mfpackage.MFPackage):
    """
    ModflowSwfdisl defines a disl package within a swf6 model.

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
    length_convert : double
        * length_convert (double) conversion factor that is used in converting
          constants used by MODFLOW into model length units. All constants are
          by default in units of "meters". Constants that use length_conversion
          include GRAVITY and STORAGE COEFFICIENT values.
    time_convert : double
        * time_convert (double) conversion factor that is used in converting
          constants used by MODFLOW into model time units. All constants are by
          default in units of "seconds". TIME_CONVERSION is used to convert the
          GRAVITY constant from seconds to the model's units.
    nogrb : boolean
        * nogrb (boolean) keyword to deactivate writing of the binary grid
          file.
    xorigin : double
        * xorigin (double) x-position of the origin used for model grid
          vertices. This value should be provided in a real-world coordinate
          system. A default value of zero is assigned if not specified. The
          value for XORIGIN does not affect the model simulation, but it is
          written to the binary grid file so that postprocessors can locate the
          grid in space.
    yorigin : double
        * yorigin (double) y-position of the origin used for model grid
          vertices. This value should be provided in a real-world coordinate
          system. If not specified, then a default value equal to zero is used.
          The value for YORIGIN does not affect the model simulation, but it is
          written to the binary grid file so that postprocessors can locate the
          grid in space.
    angrot : double
        * angrot (double) counter-clockwise rotation angle (in degrees) of the
          model grid coordinate system relative to a real-world coordinate
          system. If not specified, then a default value of 0.0 is assigned.
          The value for ANGROT does not affect the model simulation, but it is
          written to the binary grid file so that postprocessors can locate the
          grid in space.
    nodes : integer
        * nodes (integer) is the number of linear cells.
    nvert : integer
        * nvert (integer) is the total number of (x, y, z) vertex pairs used to
          characterize the model grid.
    reach_length : [double]
        * reach_length (double) length for each reach
    reach_bottom : [double]
        * reach_bottom (double) bottom elevation of surface water channel
    toreach : [integer]
        * toreach (integer) index of the downstream reach. Flow from this reach
          is passed into the dowstream reach. For reaches that do not flow to
          another reach enter a 0 to toreach. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
    idomain : [integer]
        * idomain (integer) is an optional array that characterizes the
          existence status of a cell. If the IDOMAIN array is not specified,
          then all model cells exist within the solution. If the IDOMAIN value
          for a cell is 0, the cell does not exist in the simulation. Input and
          output values will be read and written for the cell, but internal to
          the program, the cell is excluded from the solution. If the IDOMAIN
          value for a cell is 1, the cell exists in the simulation.
    vertices : [iv, xv, yv, zv]
        * iv (integer) is the vertex number. Records in the VERTICES block must
          be listed in consecutive order from 1 to NVERT. This argument is an
          index variable, which means that it should be treated as zero-based
          when working with FloPy and Python. Flopy will automatically subtract
          one when loading index variables and add one when writing index
          variables.
        * xv (double) is the x-coordinate for the vertex.
        * yv (double) is the y-coordinate for the vertex.
        * zv (double) is the z-coordinate for the vertex.
    cell2d : [icell2d, fdc, ncvert, icvert]
        * icell2d (integer) is the cell2d number. Records in the cell2d block
          must be listed in consecutive order from the first to the last. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * fdc (double) is the fractional distance to the cell center. FDC is
          relative to the first vertex in the ICVERT array. In most cases FDC
          should be 0.5, which would place the center of the line segment that
          defines the cell. If the value of FDC is 1, the cell center would
          located at the last vertex. FDC values of 0 and 1 can be used to
          place the node at either end of the cell which can be useful for
          cells with boundary conditions.
        * ncvert (integer) is the number of vertices required to define the
          cell. There may be a different number of vertices for each cell.
        * icvert (integer) is an array of integer values containing vertex
          numbers (in the VERTICES block) used to define the cell. Vertices
          must be listed in the order that defines the line representing the
          cell. Cells that are connected must share vertices. The bottom
          elevation of the cell is calculated using the ZV of the first and
          last vertex point and FDC. This argument is an index variable, which
          means that it should be treated as zero-based when working with FloPy
          and Python. Flopy will automatically subtract one when loading index
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

    reach_length = ArrayTemplateGenerator(
        ("swf6", "disl", "griddata", "reach_length")
    )
    reach_bottom = ArrayTemplateGenerator(
        ("swf6", "disl", "griddata", "reach_bottom")
    )
    toreach = ArrayTemplateGenerator(("swf6", "disl", "griddata", "toreach"))
    idomain = ArrayTemplateGenerator(("swf6", "disl", "griddata", "idomain"))
    vertices = ListTemplateGenerator(("swf6", "disl", "vertices", "vertices"))
    cell2d = ListTemplateGenerator(("swf6", "disl", "cell2d", "cell2d"))
    package_abbr = "swfdisl"
    _package_type = "disl"
    dfn_file_name = "swf-disl.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name length_units",
            "type string",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name length_convert",
            "type double precision",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name time_convert",
            "type double precision",
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
            "name nodes",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name nvert",
            "type integer",
            "reader urword",
            "optional true",
        ],
        [
            "block griddata",
            "name reach_length",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "layered false",
            "optional",
        ],
        [
            "block griddata",
            "name reach_bottom",
            "type double precision",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "layered false",
            "optional",
        ],
        [
            "block griddata",
            "name toreach",
            "type integer",
            "shape (nodes)",
            "valid",
            "reader readarray",
            "layered false",
            "optional true",
            "numeric_index true",
        ],
        [
            "block griddata",
            "name idomain",
            "type integer",
            "shape (nodes)",
            "reader readarray",
            "layered false",
            "optional true",
        ],
        [
            "block vertices",
            "name vertices",
            "type recarray iv xv yv zv",
            "shape (nvert)",
            "reader urword",
            "optional false",
        ],
        [
            "block vertices",
            "name iv",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
        [
            "block vertices",
            "name xv",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block vertices",
            "name yv",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block vertices",
            "name zv",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block cell2d",
            "name cell2d",
            "type recarray icell2d fdc ncvert icvert",
            "shape (nodes)",
            "reader urword",
            "optional false",
        ],
        [
            "block cell2d",
            "name icell2d",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
        [
            "block cell2d",
            "name fdc",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block cell2d",
            "name ncvert",
            "type integer",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block cell2d",
            "name icvert",
            "type integer",
            "shape (ncvert)",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
            "numeric_index true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        length_units=None,
        length_convert=None,
        time_convert=None,
        nogrb=None,
        xorigin=None,
        yorigin=None,
        angrot=None,
        nodes=None,
        nvert=None,
        reach_length=None,
        reach_bottom=None,
        toreach=None,
        idomain=None,
        vertices=None,
        cell2d=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "disl", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)
        self.length_convert = self.build_mfdata(
            "length_convert", length_convert
        )
        self.time_convert = self.build_mfdata("time_convert", time_convert)
        self.nogrb = self.build_mfdata("nogrb", nogrb)
        self.xorigin = self.build_mfdata("xorigin", xorigin)
        self.yorigin = self.build_mfdata("yorigin", yorigin)
        self.angrot = self.build_mfdata("angrot", angrot)
        self.nodes = self.build_mfdata("nodes", nodes)
        self.nvert = self.build_mfdata("nvert", nvert)
        self.reach_length = self.build_mfdata("reach_length", reach_length)
        self.reach_bottom = self.build_mfdata("reach_bottom", reach_bottom)
        self.toreach = self.build_mfdata("toreach", toreach)
        self.idomain = self.build_mfdata("idomain", idomain)
        self.vertices = self.build_mfdata("vertices", vertices)
        self.cell2d = self.build_mfdata("cell2d", cell2d)
        self._init_complete = True
