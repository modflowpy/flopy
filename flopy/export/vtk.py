from __future__ import print_function, division
import os
import numpy as np


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

        assert model.dis is not None
        self.model = model
        self.shape = (self.model.nlay, self.model.nrow, self.model.ncol)

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
            assert self.model.bas6, 'Cannot find basic (BAS6) package ' \
                'and ibound_filter is set to True.'
            ibound = self.model.bas6.ibound.array
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
        dis = self.model.dis
        z = np.vstack([dis.top.array.reshape(1, dis.nrow, dis.ncol),
                       dis.botm.array])
        if shared_vertex:
            verts, iverts = dis.sr.get_3d_shared_vertex_connectivity(dis.nlay,
                                                            z, ibound=ibound)
        else:
            verts, iverts = dis.sr.get_3d_vertex_connectivity(dis.nlay, z,
                                                              ibound=ibound)

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

        if ibound is not None:
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

if __name__ == '__main__':
    import flopy
    import numpy as np
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=3, nrow=3, ncol=3, top=0,
                                   botm=[-1., -2., -3.])
    vtkfile = Vtk('test.vtu', ml)
    vtkfile.write()
