from collections import OrderedDict
import numpy as np
from .modelgrid import ModelGrid, MFGridException


class VertexModelGrid(ModelGrid):
    def __init__(self, vertices, cell2d, top=None, botm=None, idomain=None,
                 sr=None, simulation_time=None):
        if nlay is None:
            grid_type = 'unlayered_vertex'
        else:
            grid_type = 'layered_vertex'
        super(VertexModelGrid, self).__init__(grid_type, sr, simulation_time,
                                              model_name, steady)

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
        return len(self._cell2d)

    @property
    def top(self):
        return self._top

    @property
    def botm(self):
        return self._botm

    @property
    def idomain(self):
        return self._idomain

    def get_model_dim_arrays(self):
        if self.grid_type() == 'layered_vertex':
            return [np.arange(1, self._nlay + 1, 1, np.int),
                    np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == 'unlayered_vertex':
            return [np.arange(1, self.num_cells() + 1, 1, np.int)]

    def num_cells_per_layer(self):
        if self.grid_type() == 'layered_vertex':
            return max(self._ncpl)
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Model is unstructured and does not ' \
                         'have a consistant number of cells per ' \
                         'layer.'
            print(except_str)
            raise MFGridException(except_str)

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError(
                'this feature is not yet implemented')
        else:
            if self.grid_type() == 'layered_vertex':
                total_cells = 0
                for layer_cells in self._ncpl:
                    total_cells += layer_cells
                return total_cells
            elif self.grid_type() == 'unlayered_vertex':
                return self._ncpl

    def get_model_dim(self):
        if self.grid_type() == 'layered_vertex':
            return [self._nlay, max(self._ncpl)]
        elif self.grid_type() == 'unlayered_vertex':
            return [self.num_cells()]

    def get_model_dim_names(self):
        if self.grid_type() == 'structured':
            return ['layer', 'row', 'column']
        elif self.grid_type() == 'layered_vertex':
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == 'unlayered_vertex':
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        if self.grid_type() == 'layered_vertex':
            return ['layer_cell_num']
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Can not get layer dimension name for DISU ' \
                         'grid. DISU grids do not support layers.'
            print(except_str)
            raise MFGridException(except_str)

    def get_horizontal_cross_section_dim_arrays(self):
        if self.grid_type() == 'layered_vertex':
            return [np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == 'unlayered_vertex':
            except_str = 'ERROR: Can not get horizontal plane arrays for DISU' \
                         ' grid. DISU grids do not support individual layers.'
            print(except_str)
            raise MFGridException(except_str)

    def get_all_model_cells(self):
        model_cells = []
        if self.grid_type() == 'layered_vertex':
            for layer in range(0, self._nlay):
                for layer_cellid in range(0, self._ncpl):
                    model_cells.append((layer + 1, layer_cellid + 1))
            return model_cells
        else:
            for node in range(0, self._ncpl):
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

    def __set_vertex_dictionaries(self):
        """
        Set method to create vertex dictionary by node number
        """
        vertexdict = {v[0]: [v[1], v[2]]
                      for v in self._vertices}
        xcenters = {}
        ycenters = {}
        xverticies = {}
        yverticies = {}

        for cell2d in self._cell2d:
            cell2d = tuple(cell2d)
            xcenters[int(cell2d[0])] = cell2d[1]
            ycenters[int(cell2d[0])] = cell2d[2]

            vert_number = [int(i) for i in cell2d[4:]]
            xverticies[int(cell2d[0])] = (vertexdict[ix][0]
                                          for ix in vert_number)
            yverticies[int(cell2d[0])] = (vertexdict[ix][1]
                                          for ix in vert_number)

        self.__xcenters_dict = xcenters
        self.__ycenters_dict = ycenters
        self.__xvertex_dict = xverticies
        self.__yvertex_dict = yverticies

    def __get_grid(self, dimension="x"):
        """
        Internal method to get model grid verticies for x

        Returns:
            list of dimension ncpl by nvertices
        """
        grid = []
        if dimension.lower() == "x":
            if self.__xvertex_dict is None:
                self.__set_vertex_dictionaries()

            for key, verts in sorted(self.__xvertex_dict.items()):
                grid.append(tuple([i for i in verts]))

        elif dimension.lower() == "y":
            if self.__yvertex_dict is None:
                self.__set_vertex_dictionaries()

            for key, verts in sorted(self.__yvertex_dict.items()):
                grid.append(tuple([i for i in verts]))

        elif dimension.lower() == "z":
            if self._top is None and self._botm is None:
                err_msg = "Model top and botm array must be specified" \
                          "for zcenters to be calculated"
                raise AssertionError(err_msg)
            else:
                grid = np.concatenate((self._top,
                                       self._botm), axis=0)

        else:
            err_msg = "get_grid method not implemented" \
                      " for {} dimension".format(dimension)
            raise NotImplementedError(err_msg)

        return np.array(grid)

    def cellcenters(self):
        """
        Internal method to get cell centers and set to grid
        """
        cache_index = (CachedDataType.cell_centers.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            self._build_vertex_dictionaries()
        return self._cache_dict[cache_index].data

    def _build_vertex_dictionaries():
        vertexdict = {v[0]: [v[1], v[2]]
                      for v in self.__verticies}
        xcenters = {}
        ycenters = {}
        xverticies = {}
        yverticies = {}

        for cell2d in self.__cell2d:
            cell2d = tuple(cell2d)
            xcenters[int(cell2d[0])] = cell2d[1]
            ycenters[int(cell2d[0])] = cell2d[2]

            vert_number = [int(i) for i in cell2d[4:]]
            xverticies[int(cell2d[0])] = (vertexdict[ix][0]
                                          for ix in vert_number)
            yverticies[int(cell2d[0])] = (vertexdict[ix][1]
                                          for ix in vert_number)

        self.__xcenters_dict = xcenters
        self.__ycenters_dict = ycenters
        self.__xvertex_dict = xverticies
        self.__yvertex_dict = yverticies
