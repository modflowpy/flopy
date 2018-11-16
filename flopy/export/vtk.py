from __future__ import print_function, division
import os
from ..discretization import StructuredGrid


def start_tag(f, tag, indent_level, indent_char='  '):
    s = indent_level * indent_char + tag
    indent_level += 1
    f.write(s + '\n')
    return indent_level


def end_tag(f, tag, indent_level, indent_char='  '):
    indent_level -= 1
    s = indent_level * indent_char + tag
    f.write(s + '\n')
    return indent_level


class Vtk(object):
    """
    Support for writing a model to a vtk file

    """
    def __init__(self, output_filename, model, verbose=None):

        assert output_filename.lower().endswith(".vtu")
        if verbose is None:
            verbose = model.verbose
        self.verbose = verbose

        if os.path.exists(output_filename):
            if self.verbose:
                print('removing existing vtk file: ' + output_filename)
            os.remove(output_filename)
        self.output_filename = output_filename

        self.model = model
        self.modelgrid = model.modelgrid
        self.shape = (self.modelgrid.nlay, self.modelgrid.nrow,
                      self.modelgrid.ncol)

        self.arrays = {}

        return

    def add_array(self, name, a):
        assert a.shape == self.shape
        self.arrays[name] = a
        return

    def write(self, shared_vertex=False, ibound_filter=False):
        """
        Write the vtk file

        """

        indent_level = 0
        if self.verbose:
            print('writing vtk file')
        f = open(self.output_filename, 'w')

        # calculate number of active cells
        nlay, nrow, ncol = self.shape
        ncells = nlay * nrow * ncol
        ibound = None
        if ibound_filter:
            ibound = self.modelgrid.idomain
            ncells = (ibound != 0).sum()
        if shared_vertex:
            npoints = (nrow + 1) * (ncol + 1) * (nlay + 1)
        else:
            npoints = ncells * 8
        if self.verbose:
            s = 'Number of point is {}\n ' \
                'Number of cells is {}\n'.format(npoints, ncells)
            print(s)

        # xml
        s = '<?xml version="1.0"?>'
        f.write(s + '\n')
        indent_level = start_tag(f, '<VTKFile type="UnstructuredGrid">',
                                 indent_level)

        # unstructured grid
        indent_level = start_tag(f, '<UnstructuredGrid>', indent_level)

        # piece
        s = '<Piece NumberOfPoints="{}" ' \
            'NumberOfCells="{}">'.format(npoints, ncells)
        indent_level = start_tag(f, s, indent_level)

        # points
        s = '<Points>'
        indent_level = start_tag(f, s, indent_level)

        s = '<DataArray type="Float64" NumberOfComponents="3">'
        indent_level = start_tag(f, s, indent_level)
        assert(isinstance(self.modelgrid, StructuredGrid))
        z = np.vstack([self.modelgrid.top.reshape(1, self.modelgrid.nrow,
                                                  self.modelgrid.ncol),
                       self.modelgrid.botm])
        if shared_vertex:
            verts, iverts = self.get_3d_shared_vertex_connectivity(
                self.modelgrid)
        else:
            verts, iverts = self.get_3d_vertex_connectivity(self.modelgrid)

        for row in verts:
            s = indent_level * '  ' + '{} {} {} \n'.format(*row)
            f.write(s)
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)

        s = '</Points>'
        indent_level = end_tag(f, s, indent_level)

        # cells
        s = '<Cells>'
        indent_level = start_tag(f, s, indent_level)

        s = '<DataArray type="Int32" Name="connectivity">'
        indent_level = start_tag(f, s, indent_level)
        for row in iverts:
            s = indent_level * '  ' + ' '.join([str(i) for i in row]) + '\n'
            f.write(s)
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)

        s = '<DataArray type="Int32" Name="offsets">'
        indent_level = start_tag(f, s, indent_level)
        icount = 0
        for row in iverts:
            icount += len(row)
            s = indent_level * '  ' + '{} \n'.format(icount)
            f.write(s)
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)

        s = '<DataArray type="UInt8" Name="types">'
        indent_level = start_tag(f, s, indent_level)
        for row in iverts:
            s = indent_level * '  ' + '{} \n'.format(11)
            f.write(s)
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)

        s = '</Cells>'
        indent_level = end_tag(f, s, indent_level)

        # add cell data
        s = '<CellData Scalars="scalars">'
        indent_level = start_tag(f, s, indent_level)

        self._write_data_array(f, indent_level, 'top', z[0:-1], ibound)

        for name, a in self.arrays.items():
            self._write_data_array(f, indent_level, name, a, ibound)

        s = '</CellData>'
        indent_level = end_tag(f, s, indent_level)

        # end piece
        indent_level = end_tag(f, '</Piece>', indent_level)

        # end unstructured grid
        indent_level = end_tag(f, '</UnstructuredGrid>', indent_level)

        # end xml
        indent_level = end_tag(f, '</VTKFile>', indent_level)

        # end file
        f.close()
        return

    def _write_data_array(self, f, indent_level, name, a, ibound):
        """
        Write a numpy array to the vtk file

        """

        # header tag
        s = '<DataArray type="Float64" Name="{}" format="ascii">'.format(name)
        indent_level = start_tag(f, s, indent_level)

        # data
        nlay = a.shape[0]

        # combine ibound with laycbd when model supports laycbd
        if ibound is not None and hasattr(self.model, 'dis') and \
                hasattr(self.model.dis, 'laycbd'):
            cbd = np.where(self.model.dis.laycbd.array > 0)
            ibound = np.insert(ibound, cbd[0]+1, ibound[cbd[0],:,:], axis=0)

        for k in range(nlay):
            s = indent_level * '  '
            f.write(s)
            if ibound is None:
                ak = a[k].flatten()
            else:
                idx = (ibound[k] != 0)
                ak = a[k][idx].flatten()
            for v in ak:
                s = ' {}'.format(v)
                f.write(s)
            f.write('\n')

        # ending tag
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)
        return

    @staticmethod
    def get_3d_shared_vertex_connectivity(mg):

        # get the x and y points for the grid
        x, y, z = mg.xyzvertices
        x = x.flatten()
        y = y.flatten()

        # set the size of the vertex grid
        nrowvert = mg.nrow + 1
        ncolvert = mg.ncol + 1
        nlayvert = mg.__nlay + 1
        nrvncv = nrowvert * ncolvert
        npoints = nrvncv * nlayvert

        # create and fill a 3d points array for the grid
        verts = np.empty((npoints, 3), dtype=np.float)
        verts[:, 0] = np.tile(x, nlayvert)
        verts[:, 1] = np.tile(y, nlayvert)
        istart = 0
        istop = nrvncv
        top_botm = mg.top_botm
        for k in range(mg.__nlay + 1):
            verts[istart:istop, 2] = mg.interpolate(top_botm[k],
                                                      verts[istart:istop, :2],
                                                      method='linear')
            istart = istop
            istop = istart + nrvncv

        # create the list of points comprising each cell. points must be
        # listed a specific way according to vtk requirements.
        iverts = []
        for k in range(mg.__nlay):
            koffset = k * nrvncv
            for i in range(mg.nrow):
                for j in range(mg.ncol):
                    if mg._idomain is not None:
                        if self._idomain[k, i, j] == 0:
                            continue
                    iv1 = i * ncolvert + j + koffset
                    iv2 = iv1 + 1
                    iv4 = (i + 1) * ncolvert + j + koffset
                    iv3 = iv4 + 1
                    iverts.append([iv4 + nrvncv, iv3 + nrvncv,
                                   iv1 + nrvncv, iv2 + nrvncv,
                                   iv4, iv3, iv1, iv2])

        return verts, iverts

    @staticmethod
    def get_3d_vertex_connectivity(mg):
        if mg.idomain is None:
            ncells = mg.__nlay * mg.nrow * mg.ncol
            ibound = np.ones((mg.__nlay, mg.nrow, mg.ncol), dtype=np.int)
        else:
            ncells = (mg.idomain != 0).sum()
            ibound = mg.idomain
        npoints = ncells * 8
        verts = np.empty((npoints, 3), dtype=np.float)
        iverts = []
        ipoint = 0
        top_botm = mg.top_botm
        for k in range(mg.__nlay):
            for i in range(mg.nrow):
                for j in range(mg.ncol):
                    if ibound[k, i, j] == 0:
                        continue

                    ivert = []
                    pts = mg._cell_vert_list(i, j)
                    pt0, pt1, pt2, pt3, pt0 = pts

                    z = top_botm[k + 1, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    z = top_botm[k, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    iverts.append(ivert)

        return verts, iverts


if __name__ == '__main__':
    import flopy
    import numpy as np
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=3, nrow=3, ncol=3, top=0,
                                   botm=[-1., -2., -3.])
    vtkfile = Vtk('test.vtu', ml)
    vtkfile.write()
