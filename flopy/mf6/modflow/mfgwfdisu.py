from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfdisu(mfpackage.MFPackage):
    package_abbr = "gwfdisu"
    top = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'griddata', 'top'))
    bot = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'griddata', 'bot'))
    area = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'griddata', 'area'))
    iac = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'iac'))
    ja = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'ja'))
    ihc = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'ihc'))
    cl12 = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'cl12'))
    hwva = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'hwva'))
    angldegx = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disu', 'connectiondata', 'angldegx'))
    verticesrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'disu', 'vertices', 'verticesrecarray'))
    cell2drecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'disu', 'cell2d', 'cell2drecarray'))
    """
    ModflowGwfdisu defines a disu package within a gwf6 model.

    Attributes
    ----------
    length_units : (length_units : string)
        is the length units used for this model. Values can be ``FEET'', ``METERS'', or ``CENTIMETERS''. If not specified, the default is ``UNKNOWN''.
    nogrb : (nogrb : keyword)
        keyword to deactivate writing of the binary grid file.
    xorigin : (xorigin : double)
        x-position of the origin used for model grid vertices. This value should be provided in a real-world coordinate system. A default value of zero is assigned if not specified. The value for xorigin does not affect the model simulation, but it is written to the binary grid file so that postprocessors can locate the grid in space.
    yorigin : (yorigin : double)
        y-position of the origin used for model grid vertices. This value should be provided in a real-world coordinate system. If not specified, then a default value equal to zero is used. The value for yorigin does not affect the model simulation, but it is written to the binary grid file so that postprocessors can locate the grid in space.
    angrot : (angrot : double)
        counter-clockwise rotation angle (in degrees) of the model grid coordinate system relative to a real-world coordinate system. If not specified, then a default value of 0.0 is assigned. The value for angrot does not affect the model simulation, but it is written to the binary grid file so that postprocessors can locate the grid in space.
    nodes : (nodes : integer)
        is the number of cells in the model grid.
    nja : (nja : integer)
        is the sum of the number of connections and nodes. When calculating the total number of connections, the connection between cell $n$ and cell $m$ is considered to be different from the connection between cell $m$ and cell $n$. Thus, nja is equal to the total number of connections, including $n$ to $m$ and $m$ to $n$, and the total number of cells.
    nvert : (nvert : integer)
        is the total number of (x, y) vertex pairs used to define the plan-view shape of each cell in the model grid. If nvert is not specified or is specified as zero, then the VERTICES and CELL2D blocks below are not read.
    top : [(top : double)]
        is the top elevation for each cell in the model grid.
    bot : [(bot : double)]
        is the bottom elevation for each cell.
    area : [(area : double)]
        is the cell surface area (in plan view).
    iac : [(iac : integer)]
        is the number of connections (plus 1) for each cell. The sum of iac must be equal to nja.
    ja : [(ja : integer)]
        is a list of cell number (n) followed by its connecting cell numbers (m) for each of the m cells connected to cell n. The number of values to provide for cell n is iac(n). This list is sequentially provided for the first to the last cell. The first value in the list must be cell n itself, and the remaining cells must be listed in an increasing order (sorted from lowest number to highest). Note that the cell and its connections are only supplied for the GWF cells and their connections to the other GWF cells. Also note that the JA list input may be chopped up to have every node number and its connectivity list on a separate line for ease in readability of the file. To further ease readability of the file, the node number of the cell whose connectivity is subsequently listed, may be expressed as a negative number the sign of which is subsequently corrected by the code.
    ihc : [(ihc : integer)]
        is an index array indicating the direction between node n and all of its m connections. If $ihc=0$ -- cell $n$ and cell $m$ are connected in the vertical direction. Cell $n$ overlies cell $m$ if the cell number for $n$ is less than $m$; cell $m$ overlies cell $n$ if the cell number for $m$ is less than $n$. If $ihc=1$ -- cell $n$ and cell $m$ are connected in the horizontal direction. If $ihc=2$ -- cell $n$ and cell $m$ are connected in the horizontal direction, and the connection is vertically staggered. A vertically staggered connection is one in which a cell is horizontally connected to more than one cell in a horizontal connection.
    cl12 : [(cl12 : double)]
        is the array containing connection lengths between the center of cell $n$ and the shared face with each adjacent $m$ cell.
    hwva : [(hwva : double)]
        is a symmetric array of size nja. For horizontal connections, entries in hwva are the horizontal width perpendicular to flow. For vertical connections, entries in hwva are the vertical area for flow. Thus, values in the hwva array contain dimensions of both length and area. Entries in the hwva array have a one-to-one correspondence with the connections specified in the ja array. Likewise, there is a one-to-one correspondence between entries in the hwva array and entries in the ihc array, which specifies the connection type (horizontal or vertical). Entries in the hwva array must be symmetric; the program will terminate with an error if the value for hwva for an $n-m$ connection does not equal the value for hwva for the corresponding $m-n$ connection.
    angldegx : [(angldegx : double)]
        is the angle (in degrees) between the horizontal x-axis and the outward normal to the face between a cell and its connecting cells (see figure 8 in the MODFLOW-USG documentation). The angle varies between zero and 360.0 degrees. angldegx is only needed if horizontal anisotropy is specified in the NPF Package or if the XT3D option is used in the NPF Package. angldegx does not need to be specified if horizontal anisotropy or the XT3D option is not used. angldegx is of size nja; values specified for vertical connections and for the diagonal position are not used. Note that angldegx is read in degrees, which is different from MODFLOW-USG, which reads a similar variable (anglex) in radians.
    verticesrecarray : [(iv : integer), (xv : double), (yv : double)]
        iv : is the vertex number. Records in the VERTICES block must be listed in consecutive order from 1 to nvert.
        xv : is the x-coordinate for the vertex.
        yv : is the y-coordinate for the vertex.
    cell2drecarray : [(icell2d : integer), (xc : double), (yc : double), (ncvert : integer), (icvert : integer)]
        icell2d : is the cell2d number. Records in the CELL2D block must be listed in consecutive order from 1 to nodes.
        xc : is the x-coordinate for the cell center.
        yc : is the y-coordinate for the cell center.
        ncvert : is the number of vertices required to define the cell. There may be a different number of vertices for each cell.
        icvert : is an array of integer values containing vertex numbers (in the VERTICES block) used to define the cell. Vertices must be listed in clockwise order.

    """
    def __init__(self, model, add_to_package_list=True, length_units=None, nogrb=None, xorigin=None, yorigin=None,
                 angrot=None, nodes=None, nja=None, nvert=None, top=None, bot=None, area=None, iac=None,
                 ja=None, ihc=None, cl12=None, hwva=None, angldegx=None, verticesrecarray=None,
                 cell2drecarray=None, fname=None, pname=None, parent_file=None):
        super(ModflowGwfdisu, self).__init__(model, "disu", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)

        self.nogrb = self.build_mfdata("nogrb", nogrb)

        self.xorigin = self.build_mfdata("xorigin", xorigin)

        self.yorigin = self.build_mfdata("yorigin", yorigin)

        self.angrot = self.build_mfdata("angrot", angrot)

        self.nodes = self.build_mfdata("nodes", nodes)

        self.nja = self.build_mfdata("nja", nja)

        self.nvert = self.build_mfdata("nvert", nvert)

        self.top = self.build_mfdata("top", top)

        self.bot = self.build_mfdata("bot", bot)

        self.area = self.build_mfdata("area", area)

        self.iac = self.build_mfdata("iac", iac)

        self.ja = self.build_mfdata("ja", ja)

        self.ihc = self.build_mfdata("ihc", ihc)

        self.cl12 = self.build_mfdata("cl12", cl12)

        self.hwva = self.build_mfdata("hwva", hwva)

        self.angldegx = self.build_mfdata("angldegx", angldegx)

        self.verticesrecarray = self.build_mfdata("verticesrecarray", verticesrecarray)

        self.cell2drecarray = self.build_mfdata("cell2drecarray", cell2drecarray)


