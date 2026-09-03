import numpy as np
import pandas as pd

from .geometry import distance
from .geospatial_utils import GeoSpatialUtil
from .gridintersect import GridIntersect


def _min_distance_index(x0, x1, y0, y1):
    """
    Method to get the index of the minimum distance between points

    Parameters
    ----------
    x0 : np.ndarray
        array of x coordinates
    x1 : float
        x coordinate of a point
    y0 : np.ndarray
        array of y coordinates
    y1 : float
        y coordinate of a point

    Returns
    -------
        int : index location
    """
    dist = distance(x0, y0, x1, y1)
    idx = np.where(dist == np.min(dist))[0]
    return idx


def _edge_length_lut(xyverts):
    """
    Method to create a look-up table based on index of
    grid cell edge lengths

    Parameters
    ----------
    xyverts: np.ndarray
        numpy array of xy coordinate pairs

    Returns
    -------
        dict
    """
    xy0 = xyverts.T[:, :-1]
    xy1 = xyverts.T[:, 1:]
    dist = distance(xy0[0], xy0[1], xy1[0], xy1[1])
    dist_lu = {}
    for v, d in enumerate(dist):
        if v < len(dist) - 1:
            dist_lu[(v, v + 1)] = d
        else:
            dist_lu[(0, v)] = d
    return dist_lu


def _minimize_hfb_deviance(idxs, xyverts, pts):
    """
    Method to minimumize the distance of the HFB fault trace from
    the intersection points of the fault and model grid cell. Used
    primarily for tie-breakers where there is not a clear "routing"
    option based on vertex to vertex edge length distance

    Parameters
    ----------
    idxs : iterable
        list, tuple, or numpy array of polygon vertex indices for test hfb path
    xyverts : np.array
        numpy array of (x,y) polygon vertices
    pts : np.array
        numpy array of (x,y) points of intersection between line and polygon

    Returns
    -------
        float : sum of minimum distances between edge centroids across a potential
        fault routing option and the intersection points between the fault line
        and grid node
    """
    lcs = []
    xyverts = xyverts.T
    pts = pts.T
    for ix in range(1, len(idxs)):
        ixx = (idxs[ix - 1], idxs[ix])
        xc = np.mean(xyverts[0, ixx])
        yc = np.mean(xyverts[1, ixx])
        lcs.append([xc, yc])

    mins = []
    for xc, yc in lcs:
        dists = distance(pts[0], pts[1], xc, yc)
        mins.append(np.min(dists))

    return np.sum(mins)


def _perturb_intersection_coords(idxs, xyverts, ipt, epsilon=1e-06):
    """
    Method to perturb an intersection location by a small amount which is used
    to handle colinear intersection and intersection at the midpoint of a cell edge.

    Parameters
    ----------
    idxs : list
        list of vertex indices
    xyverts : np.array
        numpy array of x,y vertices for a cell
    ipt : iterable
        intersection point x,y coordinate
    epsilon : float
        perturbation value

    Returns
    -------
        ipt : list holding x, y coordinate pair of the perturbed intersection
    """
    xyverts = xyverts.T
    xverts = xyverts[0][idxs]
    yverts = xyverts[1][idxs]

    if (xverts[0] - xverts[1]) == 0:
        # vertical line
        new_vrt = [xverts[0], np.mean(yverts) - epsilon]
    elif (yverts[0] - yverts[1]) == 0:
        # horizontal line
        new_vrt = [np.mean(xverts) - epsilon, yverts[0]]
    else:
        # need to adjust across the line
        m = (yverts[1] - yverts[0]) / [xverts[1] - xverts[0]]
        if xverts[0] != 0:
            vidx = 0
        else:
            vidx = 1
        b = yverts[vidx] / (m * xverts[vidx])
        cx = ipt[0] - epsilon
        cy = m * cx + b
        new_vrt = [cx, cy]

    ipt = new_vrt
    return ipt


def _edge_neighbors(modelgrid):
    """
    Method to get a dictionary of unique node edges (by ivert) and the nodes the
    edge is shared between

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object

    Returns
    -------
        dict : dictionary of {edge iverts : [nodes]}
    """
    node_num = 0
    geoms = []
    node_nums = []

    for poly in modelgrid.iverts:
        poly = [int(i) for i in poly]
        if poly[0] == poly[-1]:
            poly = poly[:-1]
        for v in range(len(poly)):
            geoms.append(tuple(sorted([poly[v - 1], poly[v]])))
        node_nums += [node_num] * len(poly)
        node_num += 1

    edge_nodes = {}
    for i, item in enumerate(geoms):
        if item not in edge_nodes:
            edge_nodes[item] = {
                node_nums[i],
            }
        else:
            edge_nodes[item].add(node_nums[i])

    return edge_nodes


def make_hfb_array(modelgrid, geom):
    """
    Method to make a HFB recarray from geospatial linestring information.
    Note: this method was developed to only accept a single linestring at a
    time to give the user an opportunity to save unique fault information
    for calibration purposes.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object
        FloPy StructuredGrid, VertexGrid, and UnstructuredGrid are supported
    geom : geospatial object
        geom parameter is a geospatial LineString representation.
        Shapely.geometry.LineString, GeoJson, List of vertices,
        shapefile.shape types are supported.

    Returns
    -------
        np.recarray
        numpy recarray of cellid1, cellid2, and hydchr. The hydchr field must
        be set by the user, is set to NaN, and is provided for convenience.
    """
    gu = GeoSpatialUtil(geom, "LineString")
    if gu.shapetype.lower() != "linestring":
        raise AssertionError(
            f"{gu.shapetype} is not supported, only LineStrings are supported"
        )

    geom = gu.points

    if modelgrid.idomain is not None:
        idomain = modelgrid.idomain.ravel()
    else:
        idomain = np.ones((modelgrid.nnodes,), dtype=int)
    edge_set = _edge_neighbors(modelgrid)
    vert_ivert = {tuple(i): cnt for cnt, i in enumerate(modelgrid.verts)}
    iverts = modelgrid.iverts
    xverts, yverts = modelgrid.cross_section_vertices
    verts = []
    for node, xv in enumerate(xverts):
        yv = yverts[node]
        vrts = list(zip(xv, yv))
        if len(vrts) == len(iverts[node]):
            if iverts[node][0] != iverts[node][-1]:
                vrts.append(vrts[0])
        verts.append(np.array(vrts))

    ixs = GridIntersect(modelgrid)
    result = ixs.intersect(geom, shapetype="LineString")
    result_adj = []
    for record in result:
        node = record.cellid
        ixshp = record.ixshapes
        coords = np.array(ixshp.coords.xy).T
        x0, y0 = coords[0, 0], coords[0, 1]
        x1, y1 = coords[-1, 0], coords[-1, 1]

        xycell = verts[node]
        xcell = xycell.T[0][:-1]
        ycell = xycell.T[1][:-1]

        vidx0 = _min_distance_index(xcell, x0, ycell, y0)
        if len(vidx0) > 1:
            # perturb line by small epsilon
            coords[0] = _perturb_intersection_coords(vidx0, xycell, coords[0])
            x0, y0 = coords[0, 0], coords[0, 1]
            vidx0 = _min_distance_index(xcell, x0, ycell, y0)

        vidx0 = vidx0[0]

        vidx1 = _min_distance_index(xcell, x1, ycell, y1)
        if len(vidx1) > 1:
            # perturb line by small epsilon
            coords[-1] = _perturb_intersection_coords(vidx1, xycell, coords[-1])
            x1, y1 = coords[-1, 0], coords[-1, 1]
            vidx1 = _min_distance_index(xcell, x1, ycell, y1)

        vidx1 = vidx1[0]

        if vidx0 == vidx1:
            continue

        tmp = tuple(sorted([vidx0, vidx1]))

        if tmp[1] - tmp[0] > 1:
            nvert = len(xycell) - 1
            elens = _edge_length_lut(xycell)
            # construct line routing options
            o1 = list(range(tmp[0], tmp[1] + 1))
            o2 = list(range(tmp[1], nvert)) + list(range(0, tmp[0] + 1))
            # calculate routing distance
            d1 = np.sum(
                [elens[tuple(sorted([o1[ix - 1], o1[ix]]))] for ix in range(1, len(o1))]
            )
            d2 = np.sum(
                [elens[tuple(sorted([o2[ix - 1], o2[ix]]))] for ix in range(1, len(o2))]
            )

            # evaluate distance and break ties if necessary
            if d1 < d2:
                tmp = o1
            elif d2 < d1:
                tmp = o2
            else:
                om1 = _minimize_hfb_deviance(o1, xycell, coords)
                om2 = _minimize_hfb_deviance(o2, xycell, coords)
                if om1 <= om2:
                    tmp = o1
                else:
                    tmp = o2

        edges = []
        for ix in range(1, len(tmp)):
            eix0 = tmp[ix - 1]
            eix1 = tmp[ix]

            xyv0 = tuple(xycell[eix0])
            xyv1 = tuple(xycell[eix1])

            iv0 = vert_ivert[xyv0]
            iv1 = vert_ivert[xyv1]

            edges.append(tuple(sorted([iv0, iv1])))

        hfb_neighs = []
        for edge in edges:
            for n in edge_set[edge]:
                if n == node:
                    continue
                hfb_neighs.append(n)

        res = [int(record.cellid), hfb_neighs]
        result_adj.append(res)

    hfb_data = []
    visited = []
    if modelgrid.nlay is not None and modelgrid.grid_type != "unstructured":
        for lay in range(modelgrid.nlay):
            ncpl_adj = lay * modelgrid.ncpl
            for cid0, hfb_neighs in result_adj:
                if not idomain[cid0 + ncpl_adj]:
                    continue

                if modelgrid.grid_type == "structured":
                    cellid0 = modelgrid.get_lrc(cid0 + ncpl_adj)[0]
                else:
                    cellid0 = (lay, cid0)

                for cid1 in hfb_neighs:
                    if not idomain[cid1 + ncpl_adj]:
                        continue

                    if modelgrid.grid_type == "structured":
                        cellid1 = modelgrid.get_lrc(cid1 + ncpl_adj)[0]
                    else:
                        cellid1 = (lay, cid1)

                    if cellid0 + cellid1 in visited:
                        continue
                    elif cellid1 + cellid0 in visited:
                        continue

                    visited.append(cellid0 + cellid1)
                    hfb_data.append((cellid0, cellid1))
    else:
        for cellid0, hfb_neighs in result_adj:
            if not idomain[cellid0]:
                continue

            for cellid1 in hfb_neighs:
                if not idomain[cellid1]:
                    continue

                if (cellid0, cellid1) in visited:
                    continue
                elif (cellid1, cellid0) in visited:
                    continue

                visited.append((cellid0, cellid1))
                hfb_data.append(((cellid0,), (cellid1,)))

    df = pd.DataFrame(hfb_data, columns=["cellid1", "cellid2"])
    df["hydchr"] = np.nan
    return df.to_records(index=False)
