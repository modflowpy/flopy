import numpy as np
from .grid import Grid, CachedData


class UnstructuredGrid(Grid):
    """
    Class for an unstructured model grid

    Parameters
    ----------
    vertices
        list of vertices that make up the grid
    cell2d
        list of cells and their vertices

    Properties
    ----------
    vertices
        returns list of vertices that make up the grid
    cell2d
        returns list of cells and their vertices

    Methods
    ----------
    get_cell_vertices(cellid)
        returns vertices for a single cell at cellid.
    """
    def __init__(self, vertices=None, iverts=None, xcenters=None, ycenters=None,
                 top=None, botm=None, idomain=None, lenuni=None,
                 ncpl=None, epsg=None, proj4=None, prj=None,
                 xoff=0., yoff=0., angrot=0., layered=True, nodes=None):
        super(UnstructuredGrid, self).__init__(self.grid_type, top, botm, idomain,
                                               lenuni, epsg, proj4, prj,
                                               xoff, yoff, angrot)

        self._vertices = vertices
        self._iverts = iverts
        self._top = top
        self._botm = botm
        self._ncpl = ncpl
        self._layered = layered
        self._xc = xcenters
        self._yc = ycenters
        self._nodes = nodes

        if iverts is not None:
            if self.layered:
                assert np.all([n == len(iverts) for n in ncpl])
                assert np.array(self.xcellcenters).shape[0] == self.ncpl[0]
                assert np.array(self.ycellcenters).shape[0] == self.ncpl[0]
            else:
                msg = ('Length of iverts must equal ncpl.sum '
                       '({} {})'.format(len(iverts), ncpl))
                assert len(iverts) == np.sum(ncpl), msg
                assert np.array(self.xcellcenters).shape[0] == self.ncpl
                assert np.array(self.ycellcenters).shape[0] == self.ncpl

    @property
    def is_valid(self):
        if self._nodes is not None:
            return True
        return False

    @property
    def is_complete(self):
        if self._nodes is not None and \
                super(UnstructuredGrid, self).is_complete:
            return True
        return False

    @property
    def grid_type(self):
        return "unstructured"

    @property
    def nlay(self):
        if self.layered:
            try:
                return len(self.ncpl)
            except TypeError:
                return 1
        else:
            return 1

    @property
    def layered(self):
        return self._layered

    @property
    def nnodes(self):
        if self._nodes is not None:
            return self._nodes
        else:
            return self.nlay * self.ncpl

    @property
    def ncpl(self):
        if self._ncpl is None:
            return len(self._iverts)
        return self._ncpl

    @property
    def shape(self):
        if isinstance(self.ncpl, (list, np.ndarray)):
            return self.nlay, self.ncpl[0]
        else:
            return self.nlay, self.ncpl

    @property
    def extent(self):
        self._copy_cache = False
        xvertices = np.hstack(self.xvertices)
        yvertices = np.hstack(self.yvertices)
        self._copy_cache = True
        return (np.min(xvertices),
                np.max(xvertices),
                np.min(yvertices),
                np.max(yvertices))

    @property
    def grid_lines(self):
        """
        Creates a series of grid line vertices for drawing
        a model grid line collection

        Returns:
            list: grid line vertices
        """
        self._copy_cache = False
        xgrid = self.xvertices
        ygrid = self.yvertices

        lines = []
        for ncell, verts in enumerate(xgrid):
            for ix, vert in enumerate(verts):
                lines.append([(xgrid[ncell][ix - 1], ygrid[ncell][ix - 1]),
                              (xgrid[ncell][ix], ygrid[ncell][ix])])
        self._copy_cache = True
        return lines

    @property
    def xyzcellcenters(self):
        """
        Internal method to get cell centers and set to grid
        """
        cache_index = 'cellcenters'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_grid_geometry_info()
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyzvertices(self):
        """
        Internal method to get model grid verticies

        Returns:
            list of dimension ncpl by nvertices
        """
        cache_index = 'xyzgrid'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_grid_geometry_info()
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    def intersect(self, x, y, local=False, forgive=False):
        x, y = super(UnstructuredGrid, self).intersect(x, y, local, forgive)
        raise Exception('Not implemented yet')

    def get_cell_vertices(self, cellid):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param cellid: (int) cellid number
        :return: list of x,y cell vertices
        """
        self._copy_cache = False
        cell_vert = list(zip(self.xvertices[cellid],
                             self.yvertices[cellid]))
        self._copy_cache = True
        return cell_vert

    def _build_grid_geometry_info(self):
        cache_index_cc = 'cellcenters'
        cache_index_vert = 'xyzgrid'

        vertexdict = {ix: list(v[-2:])
                      for ix, v in enumerate(self._vertices)}

        xcenters = self._xc
        ycenters = self._yc
        xvertices = []
        yvertices = []

        # build xy vertex and cell center info
        for iverts in self._iverts:

            xcellvert = []
            ycellvert = []
            for ix in iverts:
                xcellvert.append(vertexdict[ix][0])
                ycellvert.append(vertexdict[ix][1])

            xvertices.append(xcellvert)
            yvertices.append(ycellvert)

        zvertices, zcenters = self._zcoords()

        if self._has_ref_coordinates:
            # transform x and y
            xcenters, ycenters = self.get_coords(xcenters, ycenters)
            xvertxform = []
            yvertxform = []
            # vertices are a list within a list
            for xcellvertices, ycellvertices in zip(xvertices, yvertices):
                xcellvertices, \
                ycellvertices = self.get_coords(xcellvertices, ycellvertices)
                xvertxform.append(xcellvertices)
                yvertxform.append(ycellvertices)
            xvertices = xvertxform
            yvertices = yvertxform

        self._cache_dict[cache_index_cc] = CachedData([xcenters,
                                                       ycenters,
                                                       zcenters])
        self._cache_dict[cache_index_vert] = CachedData([xvertices,
                                                         yvertices,
                                                         zvertices])

    @classmethod
    def from_argus_export(cls, fname, nlay=1):
        """
        Create a new SpatialReferenceUnstructured grid from an Argus One
        Trimesh file

        Parameters
        ----------
        fname : string
            File name

        nlay : int
            Number of layers to create

        Returns
        -------
            sru : flopy.utils.reference.SpatialReferenceUnstructured

        """
        from ..utils.geometry import get_polygon_centroid
        f = open(fname, 'r')
        line = f.readline()
        ll = line.split()
        ncells, nverts = ll[0:2]
        ncells = int(ncells)
        nverts = int(nverts)
        verts = np.empty((nverts, 2), dtype=np.float)
        xc = np.empty((ncells), dtype=np.float)
        yc = np.empty((ncells), dtype=np.float)

        # read the vertices
        f.readline()
        for ivert in range(nverts):
            line = f.readline()
            ll = line.split()
            c, iv, x, y = ll[0:4]
            verts[ivert, 0] = x
            verts[ivert, 1] = y

        # read the cell information and create iverts, xc, and yc
        iverts = []
        for icell in range(ncells):
            line = f.readline()
            ll = line.split()
            ivlist = []
            for ic in ll[2:5]:
                ivlist.append(int(ic) - 1)
            if ivlist[0] != ivlist[-1]:
                ivlist.append(ivlist[0])
            iverts.append(ivlist)
            xc[icell], yc[icell] = get_polygon_centroid(verts[ivlist, :])

        # close file and return spatial reference
        f.close()
        return cls(verts, iverts, xc, yc, ncpl=np.array(nlay * [len(iverts)]))
