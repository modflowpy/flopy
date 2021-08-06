# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowGwtdisu(mfpackage.MFPackage):
    """
    ModflowGwtdisu defines a disu package within a gwt6 model.

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
    vertical_offset_tolerance : double
        * vertical_offset_tolerance (double) checks are performed to ensure
          that the top of a cell is not higher than the bottom of an overlying
          cell. This option can be used to specify the tolerance that is used
          for checking. If top of a cell is above the bottom of an overlying
          cell by a value less than this tolerance, then the program will not
          terminate with an error. The default value is zero. This option
          should generally not be used.
    nodes : integer
        * nodes (integer) is the number of cells in the model grid.
    nja : integer
        * nja (integer) is the sum of the number of connections and NODES. When
          calculating the total number of connections, the connection between
          cell n and cell m is considered to be different from the connection
          between cell m and cell n. Thus, NJA is equal to the total number of
          connections, including n to m and m to n, and the total number of
          cells.
    nvert : integer
        * nvert (integer) is the total number of (x, y) vertex pairs used to
          define the plan-view shape of each cell in the model grid. If NVERT
          is not specified or is specified as zero, then the VERTICES and
          CELL2D blocks below are not read. NVERT and the accompanying VERTICES
          and CELL2D blocks should be specified for most simulations. If the
          XT3D or SAVE_SPECIFIC_DISCHARGE options are specified in the NPF
          Package, then this information is required.
    top : [double]
        * top (double) is the top elevation for each cell in the model grid.
    bot : [double]
        * bot (double) is the bottom elevation for each cell.
    area : [double]
        * area (double) is the cell surface area (in plan view).
    idomain : [integer]
        * idomain (integer) is an optional array that characterizes the
          existence status of a cell. If the IDOMAIN array is not specified,
          then all model cells exist within the solution. If the IDOMAIN value
          for a cell is 0, the cell does not exist in the simulation. Input and
          output values will be read and written for the cell, but internal to
          the program, the cell is excluded from the solution. If the IDOMAIN
          value for a cell is 1 or greater, the cell exists in the simulation.
          IDOMAIN values of -1 cannot be specified for the DISU Package.
    iac : [integer]
        * iac (integer) is the number of connections (plus 1) for each cell.
          The sum of all the entries in IAC must be equal to NJA.
    ja : [integer]
        * ja (integer) is a list of cell number (n) followed by its connecting
          cell numbers (m) for each of the m cells connected to cell n. The
          number of values to provide for cell n is IAC(n). This list is
          sequentially provided for the first to the last cell. The first value
          in the list must be cell n itself, and the remaining cells must be
          listed in an increasing order (sorted from lowest number to highest).
          Note that the cell and its connections are only supplied for the GWT
          cells and their connections to the other GWT cells. Also note that
          the JA list input may be divided such that every node and its
          connectivity list can be on a separate line for ease in readability
          of the file. To further ease readability of the file, the node number
          of the cell whose connectivity is subsequently listed, may be
          expressed as a negative number, the sign of which is subsequently
          converted to positive by the code. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
    ihc : [integer]
        * ihc (integer) is an index array indicating the direction between node
          n and all of its m connections. If IHC = 0 then cell n and cell m are
          connected in the vertical direction. Cell n overlies cell m if the
          cell number for n is less than m; cell m overlies cell n if the cell
          number for m is less than n. If IHC = 1 then cell n and cell m are
          connected in the horizontal direction. If IHC = 2 then cell n and
          cell m are connected in the horizontal direction, and the connection
          is vertically staggered. A vertically staggered connection is one in
          which a cell is horizontally connected to more than one cell in a
          horizontal connection.
    cl12 : [double]
        * cl12 (double) is the array containing connection lengths between the
          center of cell n and the shared face with each adjacent m cell.
    hwva : [double]
        * hwva (double) is a symmetric array of size NJA. For horizontal
          connections, entries in HWVA are the horizontal width perpendicular
          to flow. For vertical connections, entries in HWVA are the vertical
          area for flow. Thus, values in the HWVA array contain dimensions of
          both length and area. Entries in the HWVA array have a one-to-one
          correspondence with the connections specified in the JA array.
          Likewise, there is a one-to-one correspondence between entries in the
          HWVA array and entries in the IHC array, which specifies the
          connection type (horizontal or vertical). Entries in the HWVA array
          must be symmetric; the program will terminate with an error if the
          value for HWVA for an n to m connection does not equal the value for
          HWVA for the corresponding n to m connection.
    angldegx : [double]
        * angldegx (double) is the angle (in degrees) between the horizontal
          x-axis and the outward normal to the face between a cell and its
          connecting cells. The angle varies between zero and 360.0 degrees,
          where zero degrees points in the positive x-axis direction, and 90
          degrees points in the positive y-axis direction. ANGLDEGX is only
          needed if horizontal anisotropy is specified in the NPF Package, if
          the XT3D option is used in the NPF Package, or if the
          SAVE_SPECIFIC_DISCHARGE option is specifed in the NPF Package.
          ANGLDEGX does not need to be specified if these conditions are not
          met. ANGLDEGX is of size NJA; values specified for vertical
          connections and for the diagonal position are not used. Note that
          ANGLDEGX is read in degrees, which is different from MODFLOW-USG,
          which reads a similar variable (ANGLEX) in radians.
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
        * icell2d (integer) is the cell2d number. Records in the CELL2D block
          must be listed in consecutive order from 1 to NODES. This argument is
          an index variable, which means that it should be treated as zero-
          based when working with FloPy and Python. Flopy will automatically
          subtract one when loading index variables and add one when writing
          index variables.
        * xc (double) is the x-coordinate for the cell center.
        * yc (double) is the y-coordinate for the cell center.
        * ncvert (integer) is the number of vertices required to define the
          cell. There may be a different number of vertices for each cell.
        * icvert (integer) is an array of integer values containing vertex
          numbers (in the VERTICES block) used to define the cell. Vertices
          must be listed in clockwise order. This argument is an index
          variable, which means that it should be treated as zero-based when
          working with FloPy and Python. Flopy will automatically subtract one
          when loading index variables and add one when writing index
          variables.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    top = ArrayTemplateGenerator(("gwt6", "disu", "griddata", "top"))
    bot = ArrayTemplateGenerator(("gwt6", "disu", "griddata", "bot"))
    area = ArrayTemplateGenerator(("gwt6", "disu", "griddata", "area"))
    idomain = ArrayTemplateGenerator(("gwt6", "disu", "griddata", "idomain"))
    iac = ArrayTemplateGenerator(("gwt6", "disu", "connectiondata", "iac"))
    ja = ArrayTemplateGenerator(("gwt6", "disu", "connectiondata", "ja"))
    ihc = ArrayTemplateGenerator(("gwt6", "disu", "connectiondata", "ihc"))
    cl12 = ArrayTemplateGenerator(("gwt6", "disu", "connectiondata", "cl12"))
    hwva = ArrayTemplateGenerator(("gwt6", "disu", "connectiondata", "hwva"))
    angldegx = ArrayTemplateGenerator(
        ("gwt6", "disu", "connectiondata", "angldegx")
    )
    vertices = ListTemplateGenerator(("gwt6", "disu", "vertices", "vertices"))
    cell2d = ListTemplateGenerator(("gwt6", "disu", "cell2d", "cell2d"))
    package_abbr = "gwtdisu"
    _package_type = "disu"
    dfn_file_name = "gwt-disu.dfn"

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
            "block options",
            "name vertical_offset_tolerance",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 0.0",
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
            "name nja",
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
            "name top",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
        ],
        [
            "block griddata",
            "name bot",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
        ],
        [
            "block griddata",
            "name area",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
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
            "block connectiondata",
            "name iac",
            "type integer",
            "shape (nodes)",
            "reader readarray",
        ],
        [
            "block connectiondata",
            "name ja",
            "type integer",
            "shape (nja)",
            "reader readarray",
            "numeric_index true",
            "jagged_array iac",
        ],
        [
            "block connectiondata",
            "name ihc",
            "type integer",
            "shape (nja)",
            "reader readarray",
            "jagged_array iac",
        ],
        [
            "block connectiondata",
            "name cl12",
            "type double precision",
            "shape (nja)",
            "reader readarray",
            "jagged_array iac",
        ],
        [
            "block connectiondata",
            "name hwva",
            "type double precision",
            "shape (nja)",
            "reader readarray",
            "jagged_array iac",
        ],
        [
            "block connectiondata",
            "name angldegx",
            "type double precision",
            "optional true",
            "shape (nja)",
            "reader readarray",
            "jagged_array iac",
        ],
        [
            "block vertices",
            "name vertices",
            "type recarray iv xv yv",
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
            "block cell2d",
            "name cell2d",
            "type recarray icell2d xc yc ncvert icvert",
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
            "name xc",
            "type double precision",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block cell2d",
            "name yc",
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
        nogrb=None,
        xorigin=None,
        yorigin=None,
        angrot=None,
        vertical_offset_tolerance=0.0,
        nodes=None,
        nja=None,
        nvert=None,
        top=None,
        bot=None,
        area=None,
        idomain=None,
        iac=None,
        ja=None,
        ihc=None,
        cl12=None,
        hwva=None,
        angldegx=None,
        vertices=None,
        cell2d=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "disu", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)
        self.nogrb = self.build_mfdata("nogrb", nogrb)
        self.xorigin = self.build_mfdata("xorigin", xorigin)
        self.yorigin = self.build_mfdata("yorigin", yorigin)
        self.angrot = self.build_mfdata("angrot", angrot)
        self.vertical_offset_tolerance = self.build_mfdata(
            "vertical_offset_tolerance", vertical_offset_tolerance
        )
        self.nodes = self.build_mfdata("nodes", nodes)
        self.nja = self.build_mfdata("nja", nja)
        self.nvert = self.build_mfdata("nvert", nvert)
        self.top = self.build_mfdata("top", top)
        self.bot = self.build_mfdata("bot", bot)
        self.area = self.build_mfdata("area", area)
        self.idomain = self.build_mfdata("idomain", idomain)
        self.iac = self.build_mfdata("iac", iac)
        self.ja = self.build_mfdata("ja", ja)
        self.ihc = self.build_mfdata("ihc", ihc)
        self.cl12 = self.build_mfdata("cl12", cl12)
        self.hwva = self.build_mfdata("hwva", hwva)
        self.angldegx = self.build_mfdata("angldegx", angldegx)
        self.vertices = self.build_mfdata("vertices", vertices)
        self.cell2d = self.build_mfdata("cell2d", cell2d)
        self._init_complete = True
