import numpy as np
from .modelgrid import ModelGrid, GridType, MFGridException


class VertexModelGrid(ModelGrid):
    def __init__(self, top, botm, idomain, vertices, cell2d, nlay=None,
                 ncpl=None, sr=None, simulation_time=None, model_name='',
                 steady=False):
        if nlay is None:
            grid_type = GridType.unlayered_vertex
        else:
            grid_type = GridType.layered_vertex
        super(VertexModelGrid, self).__init__(grid_type, sr, simulation_time,
                                              model_name, steady)
        self.top = top
        self.botm = botm
        self.idomain = idomain
        self._vertices = vertices
        self.cell2d = cell2d
        self._nlay = nlay
        self._ncpl = ncpl

    def get_model_dim_arrays(self):
        if self.grid_type() == GridType.layered_vertex:
            return [np.arange(1, self._nlay + 1, 1, np.int),
                    np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [np.arange(1, self.num_cells() + 1, 1, np.int)]

    def num_cells_per_layer(self):
        if self.grid_type() == GridType.layered_vertex:
            return max(self._ncpl)
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Model "{}" is unstructured and does not ' \
                         'have a consistant number of cells per ' \
                         'layer.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError(
                'this feature is not yet implemented')
        else:
            if self.grid_type() == GridType.layered_vertex:
                total_cells = 0
                for layer_cells in self._ncpl:
                    total_cells += layer_cells
                return total_cells
            elif self.grid_type() == GridType.unlayered_vertex:
                return self._ncpl

    def get_model_dim(self):
        if self.grid_type() == GridType.layered_vertex:
            return [self._nlay, max(self._ncpl)]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [self.num_cells()]

    def get_model_dim_names(self):
        if self.grid_type() == GridType.structured:
            return ['layer', 'row', 'column']
        elif self.grid_type() == GridType.layered_vertex:
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        if self.grid_type() == GridType.layered_vertex:
            return ['layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Can not get layer dimension name for model ' \
                         '"{}" DISU grid. DISU grids do not support ' \
                         'layers.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    def get_horizontal_cross_section_dim_arrays(self):
        if self.grid_type() == GridType.layered_vertex:
            return [np.arange(1, self.num_cells_per_layer() + 1, 1, np.int)]
        elif self.grid_type() == GridType.unlayered_vertex:
            except_str = 'ERROR: Can not get horizontal plane arrays for ' \
                         'model "{}" DISU grid.  DISU grids do not support ' \
                         'individual layers.'.format(self.model_name)
            print(except_str)
            raise MFGridException(except_str)

    def get_all_model_cells(self):
        model_cells = []
        if self.grid_type() == GridType.layered_vertex:
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