import numpy as np

from .cvfdutil import get_disv_gridprops
from .geometry import point_in_polygon
from .utl_import import import_optional_dependency


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
    shapely_geo = import_optional_dependency("shapely.geometry")

    p = shapely_geo.Point(point)
    poly = shapely_geo.Polygon(vertices)
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


def tri2vor(tri, **kwargs):
    """
    This is the workhorse for the VoronoiGrid class for creating a voronoi
    grid from a constructed and built flopy Triangle grid.

    Parameters
    ----------
    tri : flopy.utils.Triangle
    Flopy triangle object is used to construct the complementary voronoi
    diagram.

    Returns
    -------
    verts, iverts : ndarray, list of lists

    """
    import_optional_dependency(
        "scipy.spatial",
        error_message="Voronoi requires SciPy.",
    )
    from scipy.spatial import Voronoi

    # assign local variables
    tri_verts = tri.verts
    tri_iverts = tri.iverts
    tri_edge = tri.edge
    npoints = tri_verts.shape[0]
    ntriangles = len(tri_iverts)
    nedges = tri_edge.shape[0]

    # check to make sure there are no duplicate points
    tri_verts_unique = np.unique(tri_verts, axis=0)
    if tri_verts.shape != tri_verts_unique.shape:
        npoints_unique = tri_verts_unique.shape[0]
        errmsg = (
            f"There are duplicate points in the triangular mesh. "
            f"These can be caused by overlapping regions, holes, and "
            f"refinement features.  The triangular mesh has {npoints} "
            f"points but only {npoints_unique} are unique."
        )
        raise Exception(errmsg)

    # construct the voronoi grid
    vor = Voronoi(tri_verts, **kwargs)
    ridge_points = vor.ridge_points
    ridge_vertices = vor.ridge_vertices

    # test the voronoi vertices, and mark those outside of the domain
    nvertices = vor.vertices.shape[0]
    xc = vor.vertices[:, 0].reshape((nvertices, 1))
    yc = vor.vertices[:, 1].reshape((nvertices, 1))
    domain_polygon = [(x, y) for x, y in tri._polygons[0]]
    vor_vert_indomain = point_in_polygon(xc, yc, domain_polygon)
    vor_vert_indomain = vor_vert_indomain.flatten()
    nholes = len(tri._holes)
    if nholes > 0:
        for ihole in range(nholes):
            ipolygon = ihole + 1
            polygon = [(x, y) for x, y in tri._polygons[ipolygon]]
            vor_vert_notindomain = point_in_polygon(xc, yc, polygon)
            vor_vert_notindomain = vor_vert_notindomain.flatten()
            idx = np.where(vor_vert_notindomain == True)
            vor_vert_indomain[idx] = False

    idx_vertindex = -1 * np.ones((nvertices), int)
    idx_filtered = np.where(vor_vert_indomain == True)
    nvalid_vertices = len(idx_filtered[0])
    # renumber valid vertices consecutively
    idx_vertindex[idx_filtered] = np.arange(nvalid_vertices)

    # Create new lists for the voronoi grid vertices and the
    # voronoi grid incidence list.  There should be one voronoi
    # cell for each vertex point in the triangular grid
    vor_verts = [(x, y) for x, y in vor.vertices[idx_filtered]]
    vor_iverts = [[] for i in range(npoints)]

    # step 1 -- go through voronoi ridge vertices
    # and add valid vertices to vor_verts and to the
    # vor_iverts incidence list
    if True:
        for ips, irvs in zip(ridge_points, ridge_vertices):
            ip0, ip1 = ips
            irv0, irv1 = irvs
            if irv0 >= 0:
                point_in_domain = vor_vert_indomain[irv0]
                if point_in_domain:
                    ivert = idx_vertindex[irv0]
                    if ivert not in vor_iverts[ip0]:
                        vor_iverts[ip0].append(ivert)
                    if ivert not in vor_iverts[ip1]:
                        vor_iverts[ip1].append(ivert)
            if irv1 >= 0:
                point_in_domain = vor_vert_indomain[irv1]
                if point_in_domain:
                    ivert = idx_vertindex[irv1]
                    if ivert not in vor_iverts[ip0]:
                        vor_iverts[ip0].append(ivert)
                    if ivert not in vor_iverts[ip1]:
                        vor_iverts[ip1].append(ivert)

    # step 2 -- along the edge, add points
    if True:
        # Count number of boundary markers that correspond to the outer
        # polygon domain or to holes.  These segments will be used to add
        # new vertices for edge cells.
        nexterior_boundary_markers = len(tri._polygons[0])
        for ihole in range(nholes):
            polygon = tri._polygons[ihole + 1]
            nexterior_boundary_markers += len(polygon)
        idx = (tri_edge["boundary_marker"] > 0) & (
            tri_edge["boundary_marker"] <= nexterior_boundary_markers
        )
        inewvert = len(vor_verts)
        for _, ip0, ip1, _ in tri_edge[idx]:
            midpoint = tri_verts[[ip0, ip1]].mean(axis=0)
            px, py = midpoint
            vor_verts.append((px, py))

            # add midpoint to each voronoi cell
            vor_iverts[ip0].append(inewvert)
            vor_iverts[ip1].append(inewvert)
            inewvert += 1

            # add ip0 triangle vertex to voronoi cell
            px, py = tri_verts[ip0]
            vor_verts.append((px, py))
            vor_iverts[ip0].append(inewvert)
            inewvert += 1

            # add ip1 triangle vertex to voronoi cell
            px, py = tri_verts[ip1]
            vor_verts.append((px, py))
            vor_iverts[ip1].append(inewvert)
            inewvert += 1

    # Last step -- sort vertices in correct order
    if True:
        vor_verts = np.array(vor_verts)
        for icell in range(len(vor_iverts)):
            iverts_cell = vor_iverts[icell]
            vor_iverts[icell] = get_sorted_vertices(iverts_cell, vor_verts)

    return vor_verts, vor_iverts


class VoronoiGrid:
    """
    FloPy VoronoiGrid helper class for creating a voronoi model grid from
    an array of input points that define cell centers.  The class handles
    boundary cells by closing polygons along the edge, something that cannot
    be done directly with the scipy.spatial.Voronoi class.

    Parameters
    ----------
    input : flopy.utils.Triangle
        Constructred and built flopy Triangle object.
    kwargs : dict
        List of additional keyword arguments that will be passed through to
        scipy.spatial.Voronoi.  For circular shaped model grids, the
        qhull_options='Qz' option has been found to work well.

    Notes
    -----
    When using VoronoiGrid, the construction order used for the Triangle
    grid matters.  The first add_polygon() call must be to add the model
    domain.  Then add_polygon() must be used to add any holes.  Lastly,
    add_polygon() can be used to add regions.  This VoronoiGrid class uses
    this order to find model edges that require further work for defining and
    closing edge model cells.

    """

    def __init__(self, tri, **kwargs):
        from .triangle import Triangle

        if isinstance(tri, Triangle):
            points = tri.verts
            verts, iverts = tri2vor(tri, **kwargs)
        else:
            raise TypeError(
                "The tri argument must be of type flopy.utils.Triangle"
            )
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
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot(1, 1, 1, aspect="equal")
        pc = self.get_patch_collection(ax, **kwargs)
        if plot_title:
            ax.set_title(f"ncells: {self.ncpl}; nverts: {self.nverts}")
        return ax
