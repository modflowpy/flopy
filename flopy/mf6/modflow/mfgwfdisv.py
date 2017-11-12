from .. import mfpackage
from ..data import mfdatautil


class ModflowGwfdisv(mfpackage.MFPackage):
    package_abbr = "gwfdisv"
    top = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disv', 'griddata', 'top'))
    botm = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disv', 'griddata', 'botm'))
    idomain = mfdatautil.ArrayTemplateGenerator(('gwf6', 'disv', 'griddata', 'idomain'))
    verticesrecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'disv', 'vertices', 'verticesrecarray'))
    cell2drecarray = mfdatautil.ListTemplateGenerator(('gwf6', 'disv', 'cell2d', 'cell2drecarray'))
    """
    ModflowGwfdisv defines a disv package within a gwf6 model.

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
    nlay : (nlay : integer)
        is the number of layers in the model grid.
    ncpl : (ncpl : integer)
        is the number of cells per layer. This is a constant value for the grid and it applies to all layers.
    nvert : (nvert : integer)
        is the total number of (x, y) vertex pairs used to characterize the horizontal configuration of the model grid.
    top : [(top : double)]
        is the top elevation for each cell in the top model layer.
    botm : [(botm : double)]
        is the bottom elevation for each cell.
    idomain : [(idomain : integer)]
        is an optional array that characterizes the existence status of a cell. If the idomain array is not specified, then all model cells exist within the solution. If the idomain value for a cell is 0, the cell does not exist in the simulation. Input and output values will be read and written for the cell, but internal to the program, the cell is excluded from the solution. If the idomain value for a cell is 1, the cell exists in the simulation. If the idomain value for a cell is -1, the cell does not exist in the simulation. Furthermore, the first existing cell above will be connected to the first existing cell below. This type of cell is referred to as a ``vertical pass through'' cell.
    verticesrecarray : [(iv : integer), (xv : double), (yv : double)]
        iv : is the vertex number. Records in the VERTICES block must be listed in consecutive order from 1 to nvert.
        xv : is the x-coordinate for the vertex.
        yv : is the y-coordinate for the vertex.
    cell2drecarray : [(icell2d : integer), (xc : double), (yc : double), (ncvert : integer), (icvert : integer)]
        icell2d : is the cell2d number. Records in the CELL2D block must be listed in consecutive order from 1 to ncpl.
        xc : is the x-coordinate for the cell center.
        yc : is the y-coordinate for the cell center.
        ncvert : is the number of vertices required to define the cell. There may be a different number of vertices for each cell.
        icvert : is an array of integer values containing vertex numbers (in the VERTICES block) used to define the cell. Vertices must be listed in clockwise order. Cells that are connected must share vertices.

    """
    def __init__(self, model, add_to_package_list=True, length_units=None, nogrb=None, xorigin=None, yorigin=None,
                 angrot=None, nlay=None, ncpl=None, nvert=None, top=None, botm=None, idomain=None,
                 verticesrecarray=None, cell2drecarray=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfdisv, self).__init__(model, "disv", fname, pname, add_to_package_list, parent_file)        

        # set up variables
        self.length_units = self.build_mfdata("length_units", length_units)

        self.nogrb = self.build_mfdata("nogrb", nogrb)

        self.xorigin = self.build_mfdata("xorigin", xorigin)

        self.yorigin = self.build_mfdata("yorigin", yorigin)

        self.angrot = self.build_mfdata("angrot", angrot)

        self.nlay = self.build_mfdata("nlay", nlay)

        self.ncpl = self.build_mfdata("ncpl", ncpl)

        self.nvert = self.build_mfdata("nvert", nvert)

        self.top = self.build_mfdata("top", top)

        self.botm = self.build_mfdata("botm", botm)

        self.idomain = self.build_mfdata("idomain", idomain)

        self.verticesrecarray = self.build_mfdata("verticesrecarray", verticesrecarray)

        self.cell2drecarray = self.build_mfdata("cell2drecarray", cell2drecarray)


