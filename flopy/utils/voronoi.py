import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from .cvfdutil import get_disv_gridprops


def get_sorted_vertices(icell_vertices, vertices):
    centroid = vertices[icell_vertices].mean(axis=0)
    tlist = []
    for i, iv in enumerate(icell_vertices):
        x, y = vertices[iv]
        dx = x - centroid[0]
        dy = y - centroid[1]
        tlist.append((np.arctan2(-dy, dx), iv))
    tlist.sort()
    return [iv for angl, iv in tlist]


def get_valid_faces(vor):
    nvalid_faces = np.zeros(vor.npoints, dtype=int)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ip1, ip2 = pointidx
            nvalid_faces[ip1] += 1
            nvalid_faces[ip2] += 1
    return nvalid_faces


# todo: send this to point in polygon method defined in Rasters
def point_in_cell(point, vertices):
    try:
        from shapely.geometry import Point, Polygon
    except:
        raise ModuleNotFoundError("shapely is not installed")

    p = Point(point)
    poly = Polygon(vertices)
    if p.intersects(poly):
        return True
    else:
        return False


# todo: find out how this is different from get_sorted_vertices()
def sort_vertices(vlist):
    x, y = zip(*vlist)
    x = np.array(x)
    y = np.array(y)
    xc = x.mean()
    yc = y.mean()
    tlist = []
    for i, (x, y) in enumerate(vlist):
        dx = x - xc
        dy = y - yc
        tlist.append((np.arctan2(-dy, dx), i))
    tlist.sort()
    return [vlist[i] for angl, i in tlist]


def get_voronoi_grid(points, **kwargs):

    # Create the voronoi object
    # Note for a circular region, may need to set qhull_options='Qz'
    vor = Voronoi(points, **kwargs)

    # Go through and replace -1 values in ridge_vertices
    # with a new point on the boundary
    new_ridge_vertices = []
    new_vor_vertices = [(x, y) for x, y in vor.vertices]

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            new_ridge_vertices.append(list(simplex))
        else:
            # i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            i1 = simplex[0]
            i2 = simplex[1]

            midpoint = vor.points[pointidx].mean(axis=0)
            x, y = midpoint
            new_vor_vertices.append((x, y))
            ipt = len(new_vor_vertices) - 1
            if i1 > 0:
                new_pair = [i1, ipt]
            else:
                new_pair = [ipt, i2]
            new_ridge_vertices.append(new_pair)

    # create iverts list
    iverts = []
    for n in range(vor.points.shape[0]):
        iverts.append([])
    for pointidx, simplex in zip(vor.ridge_points, new_ridge_vertices):
        for ipt in simplex:
            if ipt not in iverts[pointidx[0]]:
                iverts[pointidx[0]].append(ipt)
            if ipt not in iverts[pointidx[1]]:
                iverts[pointidx[1]].append(ipt)

    # If a cell doesn't have any valid faces, then it must be a corner,
    # so add the cell point itself as a vertex
    if True:
        nvalid_faces = get_valid_faces(vor)
        for n in np.where(nvalid_faces == 0)[0]:
            x, y = points[n]
            new_vor_vertices.append((x, y))
            iv = len(new_vor_vertices) - 1
            iverts[n].append(iv)

    # if cell center is not inside polygon, then add cell center as vertex
    if True:
        for n in range(points.shape[0]):
            poly = []
            ivs = iverts[n]
            for iv in ivs:
                x, y = new_vor_vertices[iv]
                poly.append((x, y))
            poly = sort_vertices(poly)
            xc, yc = points[n]
            if not point_in_cell((xc, yc), poly):
                new_vor_vertices.append((xc, yc))
                iv = len(new_vor_vertices) - 1
                iverts[n].append(iv)

    verts = np.array(new_vor_vertices)
    for icell in range(len(iverts)):
        iverts[icell] = get_sorted_vertices(iverts[icell], verts)

    return verts, iverts


class VoronoiGrid:
    """
    FloPy VoronoiGrid helper class for creating a voronoi model grid from
    an array of input points that define cell centers.  The class handles
    boundary cells by closing polygons along the edge, something that cannot
    be done directly with the scipy.spatial.Voronoi class.

    Parameters
    ----------
    points : ndarray
        Two dimensional array of points with x in column 0 and y in column 1.
        These points will become cell centers in the voronoi grid.
    kwargs : dict
        List of additional keyword arguments that will be passed through to
        scipy.spatial.Voronoi.  For circular shaped model grids, the
        qhull_options='Qz' option has been found to work well.

    Notes
    -----
    The points passed into this class are marked as the cell center locations.
    Along the edges, these cell centers do not correspond to the centroid
    location of the cell polygon.  Instead, the cell centers are along the
    edge.

    This class does not yet support holes, which are supported by the Triangle
    class.  This is a feature that could be added in the future.

    """

    def __init__(self, points, **kwargs):
        verts, iverts = get_voronoi_grid(points, **kwargs)
        self.points = points
        self.verts = verts
        self.iverts = iverts
        self.ncpl = len(iverts)
        self.nverts = verts.shape[0]
        return

    def get_disv_gridprops(self):
        """
        Get a dictionary of arguments that can be passed in to the
        flopy.mf6.ModflowGwfdisv class.

        Returns
        -------
        disv_gridprops : dict
            Dictionary of arguments than can be unpacked into the
            flopy.mf6.ModflowGwfdisv constructor

        """
        disv_gridprops = get_disv_gridprops(
            self.verts, self.iverts, xcyc=self.points
        )
        return disv_gridprops

    def get_disu5_gridprops(self):
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def get_disu6_gridprops(self):
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def get_gridprops_vertexgrid(self):
        """
        Get a dictionary of information needed to create a flopy VertexGrid.
        The returned dictionary can be unpacked directly into the
        flopy.discretization.VertexGrid() constructor.

        Returns
        -------
        gridprops : dict

        """
        gridprops = self.get_disv_gridprops()
        del gridprops["nvert"]
        return gridprops

    def get_gridprops_unstructuredgrid(self):
        """
        Get a dictionary of information needed to create a flopy
        UnstructuredGrid.  The returned dictionary can be unpacked directly
        into the flopy.discretization.UnstructuredGrid() constructor.

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        disv_gridprops = self.get_disv_gridprops()
        vertices = disv_gridprops["vertices"]
        iverts = self.iverts
        ncpl = self.ncpl
        xcenters = self.points[:, 0]
        ycenters = self.points[:, 1]

        gridprops["vertices"] = vertices
        gridprops["iverts"] = iverts
        gridprops["ncpl"] = ncpl
        gridprops["xcenters"] = xcenters
        gridprops["ycenters"] = ycenters

        return gridprops

    def get_patch_collection(self, ax=None, **kwargs):
        """
        Get a matplotlib patch collection representation of the voronoi grid

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            axes to plot the patch collection
        kwargs : dict
            Additional keyward arguments to pass to the flopy.plot.plot_cvfd
            function that returns a patch collection from verts and iverts

        Returns
        -------
        pc : matplotlib.collections.PatchCollection
            patch collection of model

        """
        from ..discretization import VertexGrid
        from ..plot import PlotMapView

        modelgrid = VertexGrid(**self.get_gridprops_vertexgrid())
        pmv = PlotMapView(modelgrid=modelgrid, ax=ax)
        pc = pmv.plot_grid(**kwargs)
        return pc

    def plot(self, ax=None, plot_title=True, **kwargs):
        """
        Plot the voronoi model grid

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            axes to plot the voronoi grid
        plot_title : bool
            Add the number of cells and number of vertices as a plot title
        kwargs : dict
            Additional keyword arguments to pass to self.get_patch_collection

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            axes that contains the voronoi model grid

        """
        if ax is None:
            ax = plt.subplot(1, 1, 1, aspect="equal")
        pc = self.get_patch_collection(ax, **kwargs)
        if plot_title:
            ax.set_title(
                "ncells: {}; nverts: {}".format(self.ncpl, self.nverts)
            )
        return ax
