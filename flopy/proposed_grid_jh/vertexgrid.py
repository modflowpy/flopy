import numpy as np
from flopy.grid.grid import Grid, CachedData


class VertexGrid(Grid):
    def __init__(self, vertices, cell2d, top=None, botm=None, idomain=None,
                 epsg=None, proj4=None, lenuni=2, xoff=None, yoff=None,
                 angrot=0.0, grid_type='layered_vertex'):
        super(VertexGrid, self).__init__(grid_type, top, botm, idomain,
                                         epsg, proj4, lenuni, xoff, yoff,
                                         angrot)
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
        ncpl_list = []
        for layer in self._botm:
            ncpl_list.append(len(layer))
        return ncpl_list

    @property
    def extent(self):
        return (min(np.ravel(self.xgrid)),
                max(np.ravel(self.xgrid)),
                min(np.ravel(self.ygrid)),
                max(np.ravel(self.ygrid)))

    @property
    def xgridlength(self):
        return max(np.ravel(self.xgrid)) - min(np.ravel(self.xgrid))

    @property
    def ygridlength(self):
        return max(np.ravel(self.ygrid)) - min(np.ravel(self.ygrid))

    @property
    def grid_lines(self):
        """
        Creates a series of grid line vertices for drawing
        a model grid line collection

        Returns:
            list: grid line vertices
        """
        xgrid = self.xgrid
        ygrid = self.ygrid

        lines = []
        for ncell, verts in enumerate(xgrid):
            for ix, vert in enumerate(verts):
                lines.append([(xgrid[ncell][ix - 1], ygrid[ncell][ix - 1]),
                              (xgrid[ncell][ix], ygrid[ncell][ix])])
        return lines

    def get_model_dim_arrays(self):
        if self.grid_type() == 'layered_vertex':
            return [np.arange(1, self.nlay + 1, 1, np.int),
                    np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == 'unlayered_vertex':
            return [np.arange(1, self.num_cells() + 1, 1, np.int)]

    def num_cells_per_layer(self):
        if self.grid_type() == 'layered_vertex':
            return max(self.ncpl)
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Model is unstructured and does not ' \
                         'have a consistant number of cells per ' \
                         'layer.'
            print(except_str)
            raise Exception(except_str)

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError(
                'this feature is not yet implemented')
        else:
            if self.grid_type() == 'layered_vertex':
                total_cells = 0
                for layer_cells in self.ncpl:
                    total_cells += layer_cells
                return total_cells
            elif self.grid_type() == 'unlayered_vertex':
                return self.ncpl

    @property
    def cellcenters(self):
        """
        Internal method to get cell centers and set to grid
        """
        cache_index = 'cellcenters'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_grid_geometry_info()
        return self._cache_dict[cache_index].data

    @property
    def xyzgrid(self):
        """
        Internal method to get model grid verticies

        Returns:
            list of dimension ncpl by nvertices
        """
        cache_index = 'xyzvertices'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_grid_geometry_info()

        return self._cache_dict[cache_index].data

    def _build_grid_geometry_info(self):
        cache_index_cc = 'cellcenters'
        cache_index_vert = 'xyzvertices'

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

            vert_number = [int(i) for i in cell2d[4:]]
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
            xcenters, ycenters = self.transform(xcenters, ycenters)
            xvertxform = []
            yvertxform = []
            # vertices are a list within a list
            for xcellvertices, ycellvertices in zip(xvertices, yvertices):
                xcellvertices, ycellvertices = self.transform(xcellvertices,
                                                              ycellvertices)
                xvertxform.append(xcellvertices)
                yvertxform.append(ycellvertices)
            xvertices = xvertxform
            yvertices = yvertxform

        self._cache_dict[cache_index_cc] = CachedData([xcenters, ycenters,
                                                       zcenters])
        self._cache_dict[cache_index_vert] = CachedData([xvertices, yvertices,
                                                         zvertices])


if __name__ == "__main__":
    import os
    import flopy as fp
    from flopy.proposed_grid_srp.reference import SpatialReference

    ws = "../../examples/data/mf6/test003_gwfs_disv"
    name = "mfsim.nam"

    sim = fp.mf6.modflow.MFSimulation.load(sim_name=name, sim_ws=ws)

    print(sim.model_names)
    ml = sim.get_model("gwf_1")

    dis = ml.dis

    t = VertexGrid(dis.vertices.array, dis.cell2d.array, top=dis.top.array,
                   botm=dis.botm.array, idomain=dis.idomain.array,
                   epsg=26715, xoff=0, yoff=0, angrot=45)

    sr_x = t.xgrid
    sr_y = t.ygrid
    sr_xc = t.xcenters
    sr_yc = t.ycenters
    sr_lc = t.grid_lines
    sr_e = t.extent

    print('break')

    t.use_ref_coords = False
    x = t.xgrid
    y = t.ygrid
    z = t.zgrid
    xc = t.xcenters
    yc = t.ycenters
    zc = t.zcenters
    lc = t.grid_lines
    e = t.extent

    print('break')