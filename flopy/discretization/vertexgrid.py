import numpy as np
import itertools
from .grid import Grid, CachedData


class VertexGrid(Grid):
    """
    class for a vertex model grid

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

    def __init__(self, vertices, cell2d, top=None, botm=None, idomain=None,
                 lenuni=None, epsg=None, proj4=None, prj=None, xoff=0.0,
                 yoff=0.0, angrot=0.0, grid_type='vertex'):
        super(VertexGrid, self).__init__(grid_type, top, botm, idomain, lenuni,
                                         epsg, proj4, prj, xoff, yoff, angrot)
        self._vertices = vertices
        self._cell2d = cell2d
        self._top = top
        self._botm = botm
        self._idomain = idomain

    @property
    def nlay(self):
        if self._botm is not None:
            return len(self._botm + 1)

    @property
    def ncpl(self):
        return len(self._botm[0])

    @property
    def shape(self):
        return self.nlay, self.ncpl

    @property
    def extent(self):
        xvertices = np.hstack(self.xvertices)
        yvertices = np.hstack(self.yvertices)
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
        xgrid = self.xvertices
        ygrid = self.yvertices

        lines = []
        for ncell, verts in enumerate(xgrid):
            for ix, vert in enumerate(verts):
                lines.append([(xgrid[ncell][ix - 1], ygrid[ncell][ix - 1]),
                              (xgrid[ncell][ix], ygrid[ncell][ix])])
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
        return self._cache_dict[cache_index].data

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

        return self._cache_dict[cache_index].data

    def intersect(self, x, y, local=True):
        x, y = super(VertexGrid, self).intersect(x, y, local)
        raise Exception('Not implemented yet')

    def get_cell_vertices(self, cellid):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param cellid: (int) cellid number
        :return: list of x,y cell vertices
        """
        return list(zip(self.xvertices[cellid],
                        self.yvertices[cellid]))

    def _build_grid_geometry_info(self):
        cache_index_cc = 'cellcenters'
        cache_index_vert = 'xyzgrid'

        vertexdict = {v[0]: [v[1], v[2]]
                      for v in self._vertices}
        xcenters = []
        ycenters = []
        xvertices = []
        yvertices = []

        # build xy vertex and cell center info
        for cell2d in self._cell2d:
            cell2d = tuple(cell2d)
            xcenters.append(cell2d[1])
            ycenters.append(cell2d[2])

            vert_number = []
            for i in cell2d[4:]:
                if i is not None:
                    vert_number.append(int(i))

            xcellvert = []
            ycellvert = []
            for ix in vert_number:
                xcellvert.append(vertexdict[ix][0])
                ycellvert.append(vertexdict[ix][1])
            xvertices.append(xcellvert)
            yvertices.append(ycellvert)

        # build z cell centers
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


if __name__ == "__main__":
    import os
    import flopy as fp

    ws = "../../examples/data/mf6/test003_gwfs_disv"
    name = "mfsim.nam"

    sim = fp.mf6.modflow.MFSimulation.load(sim_name=name, sim_ws=ws)

    print(sim.model_names)
    ml = sim.get_model("gwf_1")

    dis = ml.dis

    t = VertexGrid(dis.vertices.array, dis.cell2d.array, top=dis.top.array,
                   botm=dis.botm.array, idomain=dis.idomain.array,
                   epsg=26715, xoff=0, yoff=0, angrot=45)

    sr_x = t.xvertices
    sr_y = t.yvertices
    sr_xc = t.xcellcenters
    sr_yc = t.ycellcenters
    sr_lc = t.grid_lines
    sr_e = t.extent

    print('break')

    t.use_ref_coords = False
    x = t.xvertices
    y = t.yvertices
    z = t.zvertices
    xc = t.xcellcenters
    yc = t.ycellcenters
    zc = t.zcellcenters
    lc = t.grid_lines
    e = t.extent

    print('break')