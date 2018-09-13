from collections import OrderedDict
import numpy as np
from flopy.proposed_grid_srp.modelgrid import ModelGrid, MFGridException, \
    CachedData, CachedDataType


class VertexModelGrid(ModelGrid):
    # comment JL: layered_vertex and unlayered_vertex should be changed to
    # vertex and unstructured in my opinion. This is because modflow
    # defines specific grid types as Structured, Vertex, and Unstructured.
    # By defining the grid_types differently in flopy, it creates unnecessary
    # confusion.

    # comment JL: lenuni should default to unspecified to avoid unwanted
    # issues with model grid transforms using a spatial reference. This argument
    # should either be explicitly set by the user or passed in from the dis file.

    # comment JL: origin_x & origin_y. Consider renaming to xorigin and yorigin
    # for consistency with modflow6 naming conventions.
    def __init__(self, vertices, cell2d, top=None, botm=None, idomain=None,
                 simulation_time=None, lenuni=2, sr=None, origin_loc='ul',
                 origin_x=None, origin_y=None, rotation=0.0,
                 grid_type='layered_vertex'):
        super(VertexModelGrid, self).__init__(grid_type,  top, botm, idomain,
                                              sr, simulation_time, lenuni,
                                              origin_loc, origin_x, origin_y,
                                              rotation)
        self._vertices = vertices
        self._cell2d = cell2d
        self._top = top
        self._botm = botm
        self._idomain = idomain

    @property
    def nlay(self):
        if self._botm is not None:
            # this should be len(self._botm) since there is one botm array per layer
            # return len(self._botm) + 1
            return len(self._botm)

    @property
    def ncpl(self):
        # comment JL: ncpl should be an integer, because it is a constant for all layers (pg 30 mf6 io documentation)
        # and is defined as a single constant in the DISV package ex. in Modflow6 ncpl = 100 means
        # that for all layers there is 100 cells. Vertex model grids have regular structure in the z-direction....
        # this chage would also allow the user to get number of cells from an unstructured grid using the
        # self.ncpl property!

        # ncpl_list = []
        # for layer in self._botm:
        #     ncpl_list.append(len(layer))
        # return ncpl_list
        return len(self._cell2d)

    @property
    def extent(self):
        return (min(np.ravel(self.xgrid)),
                max(np.ravel(self.xgrid)),
                min(np.ravel(self.ygrid)),
                max(np.ravel(self.ygrid)))

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
                    np.arange(1, self.ncpl + 1, 1, np.int)]
        elif self.grid_type() == 'unlayered_vertex':
            return [np.arange(1, self.num_cells() + 1, 1, np.int)]

    def num_cells_per_layer(self):
        # comment JL: This method should be removed since ncpl_i for i in n layers is a contant value
        if self.grid_type() == 'layered_vertex':
            return self.ncpl # max(self.ncpl)
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Model is unstructured and does not ' \
                         'have a consistant number of cells per ' \
                         'layer.'
            print(except_str)
            raise MFGridException(except_str)

    def num_cells(self, active_only=False):
        if active_only:
            if self.idomain is not None:
                return self.ncpl * self.nlay - np.count_nonzero(self.idomain==0)
            else:
                err_msg = "idomain has not been suplied to VertexModelGrid"
                raise AttributeError(err_msg)

        else:
            # comment JL: this should be simplified to self.ncpl * nlay
            if self.grid_type() == 'layered_vertex':
                # total_cells = 0
                #for layer_cells in self.ncpl:
                #    total_cells += layer_cells
                return self.ncpl * self.nlay # total_cells
            elif self.grid_type() == 'unlayered_vertex':
                return self.ncpl

    def get_model_dim(self):
        if self.grid_type() == 'layered_vertex':
            #comment JL: remove max(), it is a redundant operation.
            return [self.nlay, self.ncpl]
        elif self.grid_type() == 'unlayered_vertex':
            return [self.num_cells()]

    def get_model_dim_names(self):
        # this method does not need to include 'structured'
        # method may be better suited in base class.
        if self.grid_type() == 'structured':
            return ['layer', 'row', 'column']
        elif self.grid_type() == 'layered_vertex':
            # layer cell num should be named cellid to be consistent with modflow6 output names
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == 'unlayered_vertex':
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        if self.grid_type() == 'layered_vertex':
            # this should just be cellid to be consistent with modflow6 output names.
            return ['layer_cell_num']
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Can not get layer dimension name for DISU ' \
                         'grid. DISU grids do not support layers.'
            print(except_str)
            raise MFGridException(except_str)

    def get_horizontal_cross_section_dim_arrays(self):
        if self.grid_type() == 'layered_vertex':
            # not sure what this will be used for... if the user has ncpl they can use np.arange(1, ncpl + 1)
            # also not sure why the numpy array is nested in a list, which adds an unused dimension to the array
            return [np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Can not get horizontal plane arrays for DISU' \
                         ' grid. DISU grids do not support individual layers.'
            print(except_str)
            raise MFGridException(except_str)

    def get_all_model_cells(self):
        model_cells = []
        if self.grid_type() == 'layered_vertex':
            for layer in range(0, self.nlay):
                # comment JL: removed the layer call from self.ncpl[layer]
                for layer_cellid in range(0, self.ncpl):
                    model_cells.append((layer + 1, layer_cellid + 1))
            return model_cells
        else:
            # comment JL: remove the [0] call from self.ncpl[0]
            for node in range(0, self.ncpl):
                model_cells.append(node + 1)
            return model_cells

    @classmethod
    # move to export folder
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
        return cls(xc, yc, verts, iverts, np.array(nlay * [len(iverts)]))

    @property
    def cellcenters(self):
        """
        Internal method to get cell centers and set to grid
        """
        cache_index = (CachedDataType.cell_centers.value,
                       self._use_ref_coordinates)
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
        cache_index = (CachedDataType.xyvertices.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_grid_geometry_info()

        return self._cache_dict[cache_index].data

    def _build_grid_geometry_info(self):
        cache_index_cc = (CachedDataType.cell_centers.value,
                          self._use_ref_coordinates)
        cache_index_vert = (CachedDataType.xyvertices.value,
                            self._use_ref_coordinates)

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

        if self._use_ref_coordinates:
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

    sr = SpatialReference(epsg=26715)
    t = VertexModelGrid(dis.vertices.array, dis.cell2d.array, top=dis.top.array,
                        botm=dis.botm.array, idomain=dis.idomain.array, sr=sr,
                        origin_x=0, origin_y=0, rotation=45, origin_loc="ul")

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