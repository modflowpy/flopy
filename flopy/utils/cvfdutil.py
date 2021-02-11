from collections import OrderedDict
import numpy as np


def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its vertices
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in range(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0


def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    """
    import itertools as IT

    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= area * 6.0
    result_y /= area * 6.0
    return (result_x, result_y)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return


def isBetween(a, b, c, epsilon=0.001):
    crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)
    if abs(crossproduct) > epsilon:
        return False  # (or != 0 if using integers)

    dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y) * (b.y - a.y)
    if dotproduct < 0:
        return False

    squaredlengthba = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)
    if dotproduct > squaredlengthba:
        return False

    return True


def shared_face(ivlist1, ivlist2):
    for i in range(len(ivlist1) - 1):
        iv1 = ivlist1[i]
        iv2 = ivlist1[i + 1]
        for i2 in range(len(ivlist2) - 1):
            if ivlist2[i2 : i2 + 1] == [iv2, iv1]:
                return True
    return False


def segment_face(ivert, ivlist1, ivlist2, vertices):
    """
    Check the vertex lists for cell 1 and cell 2.  Add a new vertex to cell 1
    if necessary.

    Parameters
    ----------
    ivert : int
        vertex number to check
    ivlist1 : list
        list of vertices for cell 1.  Add a new vertex to this cell if needed.
    ivlist2 : list
        list of vertices for cell2.
    vertices : ndarray
        array of x, y vertices

    Returns
    -------
    segmented : bool
        Return True if a face in cell 1 was split up by adding a new vertex

    """

    # go through ivlist1 and find faces that have ivert
    faces_to_check = []
    for ipos in range(len(ivlist1) - 1):
        face = (ivlist1[ipos], ivlist1[ipos + 1])
        if ivert in face:
            faces_to_check.append(face)

    # go through ivlist2 and find points to check
    points_to_check = []
    for ipos in range(len(ivlist2) - 1):
        if ivlist2[ipos] == ivert:
            points_to_check.append(ivlist2[ipos + 1])
        elif ivlist2[ipos + 1] == ivert:
            points_to_check.append(ivlist2[ipos])

    for face in faces_to_check:
        iva, ivb = face
        x, y = vertices[iva]
        a = Point(x, y)
        x, y = vertices[ivb]
        b = Point(x, y)
        for ivc in points_to_check:
            if ivc not in face:
                x, y = vertices[ivc]
                c = Point(x, y)
                if isBetween(a, b, c):
                    ipos = ivlist1.index(ivb)
                    if ipos == 0:
                        ipos = len(ivlist1) - 1
                    ivlist1.insert(ipos, ivc)
                    return True

    return False


def to_cvfd(
    vertdict,
    nodestart=None,
    nodestop=None,
    skip_hanging_node_check=False,
    verbose=False,
):
    """
    Convert a vertex dictionary into verts and iverts

    Parameters
    ----------
    vertdict
        vertdict is a dictionary {icell: [(x1, y1), (x2, y2), (x3, y3), ...]}

    nodestart : int
        starting node number. (default is zero)

    nodestop : int
        ending node number up to but not including. (default is len(vertdict))

    skip_hanging_node_check : bool
        skip the hanging node check.  this may only be necessary for quad-based
        grid refinement. (default is False)

    verbose : bool
        print messages to the screen. (default is False)

    Returns
    -------
    verts : ndarray
        array of x, y vertices

    iverts : list
        list containing a list for each cell

    """

    if nodestart is None:
        nodestart = 0
    if nodestop is None:
        nodestop = len(vertdict)
    ncells = nodestop - nodestart

    # First create vertexdict {(x1, y1): ivert1, (x2, y2): ivert2, ...} and
    # vertexlist [[ivert1, ivert2, ...], [ivert9, ivert10, ...], ...]
    # In the process, filter out any duplicate vertices
    vertexdict = OrderedDict()
    vertexlist = []
    xcyc = np.empty((ncells, 2), dtype=float)
    iv = 0
    nvertstart = 0
    if verbose:
        print("Converting vertdict to cvfd representation.")
        print("Number of cells in vertdict is: {}".format(len(vertdict)))
        print(
            "Cell {} up to {} (but not including) will be processed.".format(
                nodestart, nodestop
            )
        )
    for icell in range(nodestart, nodestop):
        points = vertdict[icell]
        nvertstart += len(points)
        xc, yc = centroid_of_polygon(points)
        xcyc[icell, 0] = xc
        xcyc[icell, 1] = yc
        ivertlist = []
        for p in points:
            pt = tuple(p)
            if pt in vertexdict:
                ivert = vertexdict[pt]
            else:
                vertexdict[pt] = iv
                ivert = iv
                iv += 1
            ivertlist.append(ivert)
        if ivertlist[0] != ivertlist[-1]:
            raise Exception("Cell {} not closed".format(icell))
        vertexlist.append(ivertlist)

    # next create vertex_cell_dict = {}; for each vertex, store list of cells
    # that use it
    nvert = len(vertexdict)
    if verbose:
        print("Started with {} vertices.".format(nvertstart))
        print("Ended up with {} vertices.".format(nvert))
        print(
            "Reduced total number of vertices by {}".format(nvertstart - nvert)
        )
        print("Creating dict of vertices with their associated cells")
    vertex_cell_dict = OrderedDict()
    for icell in range(nodestart, nodestop):
        ivertlist = vertexlist[icell]
        for ivert in ivertlist:
            if ivert in vertex_cell_dict:
                if icell not in vertex_cell_dict[ivert]:
                    vertex_cell_dict[ivert].append(icell)
            else:
                vertex_cell_dict[ivert] = [icell]
    if verbose:
        print("Done creating dict of vertices with their associated cells")

    # Now, go through each vertex and look at the cells that use the vertex.
    # For quadtree-like grids, there may be a need to add a new hanging node
    # vertex to the larger cell.
    if not skip_hanging_node_check:
        if verbose:
            print("Checking for hanging nodes.")
        vertexdict_keys = list(vertexdict.keys())
        finished = False
        while not finished:
            finished = True
            for ivert, cell_list in vertex_cell_dict.items():
                for icell1 in cell_list:
                    for icell2 in cell_list:

                        # skip if same cell
                        if icell1 == icell2:
                            continue

                        # skip if share face already
                        ivertlist1 = vertexlist[icell1]
                        ivertlist2 = vertexlist[icell2]
                        if shared_face(ivertlist1, ivertlist2):
                            continue

                        # don't share a face, so need to segment if necessary
                        segmented = segment_face(
                            ivert, ivertlist1, ivertlist2, vertexdict_keys
                        )
                        if segmented:
                            finished = False
        if verbose:
            print("Done checking for hanging nodes.")

    verts = np.array(vertexdict_keys)
    iverts = vertexlist

    return verts, iverts


def shapefile_to_cvfd(shp, **kwargs):
    import shapefile

    print("Translating shapefile ({}) into cvfd format".format(shp))
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    vertdict = {}
    for icell, shape in enumerate(shapes):
        points = shape.points
        vertdict[icell] = points
    verts, iverts = to_cvfd(vertdict, **kwargs)
    return verts, iverts


def shapefile_to_xcyc(shp):
    """

    Get cell centroid coordinates

    Parameters
    ----------
    shp : string
        Name of shape file

    Returns
    -------
    xcyc : ndarray
        x, y coordinates of all polygons in shp

    """
    import shapefile

    print("Translating shapefile ({}) into cell centroids".format(shp))
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    ncells = len(shapes)
    xcyc = np.empty((ncells, 2), dtype=float)
    for icell, shape in enumerate(shapes):
        points = shape.points
        xc, yc = centroid_of_polygon(points)
        xcyc[icell, 0] = xc
        xcyc[icell, 1] = yc
    return xcyc


def gridlist_to_verts(gridlist):
    """

    Take a list of flopy structured model grids and convert them into vertices.
    The idomain can be set to remove cells in a parent grid.  Cells from a
    child grid will patched in to make a single set of vertices.  Cells will
    be numbered according to consecutive numbering of active cells in the
    grid list.

    Parameters
    ----------
    gridlist : list
        List of flopy.discretization.modelgrid.  Must be of type structured
        grids

    Returns
    -------
    verts, iverts : np.ndarray, list
        vertices and list of cells and which vertices comprise the cells
    """
    vertdict = {}
    icell = 0
    for sg in gridlist:
        ilays, irows, icols = np.where(sg.idomain > 0)
        for _, i, j in zip(ilays, irows, icols):
            v = sg.get_cell_vertices(i, j)
            vertdict[icell] = v + [v[0]]
            icell += 1
    verts, iverts = to_cvfd(vertdict, verbose=False)
    return verts, iverts


def get_disv_gridprops(verts, iverts, xcyc=None):
    """

    Calculate disv grid properties from verts and iverts

    Parameters
    ----------
    verts : ndarray
        2d array of x, y vertices

    iverts : list
        list of size ncpl, with a list of vertex numbers for each cell

    Returns
    -------
    gridprops : dict
        Dictionary containing entries that can be passed directly into the
        modflow6 disv package.

    """
    nvert = verts.shape[0]
    ncpl = len(iverts)
    if xcyc is None:
        xcyc = np.empty((ncpl, 2), dtype=float)
        for icell in range(ncpl):
            vlist = [
                (verts[ivert, 0], verts[ivert, 1]) for ivert in iverts[icell]
            ]
            xcyc[icell, 0], xcyc[icell, 1] = centroid_of_polygon(vlist)
    else:
        assert xcyc.shape == (ncpl, 2)
    vertices = []
    for i in range(nvert):
        vertices.append((i, verts[i, 0], verts[i, 1]))
    cell2d = []
    for i in range(ncpl):
        cell2d.append(
            [i, xcyc[i, 0], xcyc[i, 1], len(iverts[i])]
            + [iv for iv in iverts[i]]
        )
    gridprops = {}
    gridprops["ncpl"] = ncpl
    gridprops["nvert"] = nvert
    gridprops["vertices"] = vertices
    gridprops["cell2d"] = cell2d
    return gridprops


def gridlist_to_disv_gridprops(gridlist):
    """

    Take a list of flopy structured model grids and convert them into a
    dictionary that can be passed into the modflow6 disv package.  Cells from a
    child grid will patched in to make a single set of vertices.  Cells will
    be numbered according to consecutive numbering of active cells in the
    grid list.

    Parameters
    ----------
    gridlist : list
        List of flopy.discretization.modelgrid.  Must be of type structured
        grids

    Returns
    -------
    gridprops : dict
        Dictionary containing entries that can be passed directly into the
        modflow6 disv package.

    """
    verts, iverts = gridlist_to_verts(gridlist)
    gridprops = get_disv_gridprops(verts, iverts)
    return gridprops
