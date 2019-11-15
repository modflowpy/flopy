from __future__ import print_function, division
import os
import numpy as np
from ..discretization import StructuredGrid
from ..datbase import DataType, DataInterface
import flopy.utils.binaryfile as bf
from flopy.utils import HeadFile
import numpy.ma as ma
import struct
import sys

# Module for exporting vtk from flopy

# BINARY *********************************************
np_to_struct = {'int8': 'b',
                'uint8': 'B',
                'int16': 'h',
                'uint16': 'H',
                'int32': 'i',
                'uint32': 'I',
                'int64': 'q',
                'uint64': 'Q',
                'float32': 'f',
                'float64': 'd'}


class BinaryXml:
    """
    Helps write binary vtk files

    Parameters
    ----------

    file_path : str
        output file path

    """
    def __init__(self, file_path):
        self.stream = open(file_path, "wb")
        self.open_tag = False
        self.current = []
        self.stream.write(b'<?xml version="1.0"?>')
        if sys.byteorder == "little":
            self.byte_order = '<'
        else:
            self.byte_order = '>'

    def write_size(self, block_size):
        # size is a 64 bit unsigned integer
        byte_order = self.byte_order + 'Q'
        block_size = struct.pack(byte_order, block_size)
        self.stream.write(block_size)

    def write_array(self, data):
        assert (data.flags['C_CONTIGUOUS'] or data.flags['F_CONTIGUOUS'])

        # ravel in fortran order
        dd = np.ravel(data, order='F')

        data_format = self.byte_order + str(data.size) + np_to_struct[
            data.dtype.name]
        binary_data = struct.pack(data_format, *dd)
        self.stream.write(binary_data)

    def write_coord_arrays(self, x, y, z):
        # check that arrays are the same shape and data type
        assert (x.size == y.size == z.size)
        assert (x.dtype.itemsize == y.dtype.itemsize == z.dtype.itemsize)

        # check if arrays are contiguous
        assert (x.flags['C_CONTIGUOUS'] or x.flags['F_CONTIGUOUS'])
        assert (y.flags['C_CONTIGUOUS'] or y.flags['F_CONTIGUOUS'])
        assert (z.flags['C_CONTIGUOUS'] or z.flags['F_CONTIGUOUS'])

        data_format = self.byte_order + str(1) + \
            np_to_struct[x.dtype.name]

        xrav = np.ravel(x, order='F')
        yrav = np.ravel(y, order='F')
        zrav = np.ravel(z, order='F')

        for idx in range(x.size):
            bx = struct.pack(data_format, xrav[idx])
            by = struct.pack(data_format, yrav[idx])
            bz = struct.pack(data_format, zrav[idx])
            self.stream.write(bx)
            self.stream.write(by)
            self.stream.write(bz)

    def close(self):
        assert (not self.open_tag)
        self.stream.close()

    def open_element(self, tag):
        if self.open_tag:
            self.stream.write(b">")
        tag_string = "\n<%s" % tag
        self.stream.write(str.encode(tag_string))
        self.open_tag = True
        self.current.append(tag)
        return self

    def close_element(self, tag=None):
        if tag:
            assert (self.current.pop() == tag)
            if self.open_tag:
                self.stream.write(b">")
                self.open_tag = False
            string = "\n</%s>" % tag
            self.stream.write(str.encode(string))
        else:
            self.stream.write(b"/>")
            self.open_tag = False
            self.current.pop()
        return self

    def add_text(self, text):
        if self.open_tag:
            self.stream.write(b">\n")
            self.open_tag = False
        self.stream.write(str.encode(text))
        return self

    def add_attributes(self, **kwargs):
        assert self.open_tag
        for key in kwargs:
            st = ' %s="%s"' % (key, kwargs[key])
            self.stream.write(str.encode(st))
        return self

# END BINARY *********************************************


def start_tag(f, tag, indent_level, indent_char='  '):
    # starts xml tag
    s = indent_level * indent_char + tag
    indent_level += 1
    f.write(s + '\n')
    return indent_level


def end_tag(f, tag, indent_level, indent_char='  '):
    # ends xml tag
    indent_level -= 1
    s = indent_level * indent_char + tag
    f.write(s + '\n')
    return indent_level


class _Array(object):
    # class to store array and tell if array is 2d
    def __init__(self, array, array2d):
        self.array = array
        self.array2d = array2d


def _get_basic_modeltime(perlen_list):
    modeltim = 0
    totim = []
    for tim in perlen_list:
        totim.append(modeltim)
        modeltim += tim
    return totim


class Vtk(object):
    """
    Class to build VTK object for exporting flopy vtk

    Parameters
    ----------

    model : MFModel
        flopy model instance
    verbose : bool
        if True, stdout is verbose
    nanval : float
        no data value, default is -1e20
    smooth : bool
        If True will create smooth output surface
    point_scalars : bool
        if True will output point scalar values, this will set smooth to True.

    Attributes
    ----------

    arrays : dict
        Stores data arrays added to VTK object

    """

    def __init__(self, model, verbose=None, nanval=-1e+20, smooth=False,
                 point_scalars=False):

        if point_scalars:
            smooth = True

        if verbose is None:
            verbose = model.verbose
        self.verbose = verbose

        # set up variables
        self.model = model
        self.modelgrid = model.modelgrid
        self.arrays = {}
        self.shape = (self.modelgrid.nlay, self.modelgrid.nrow,
                      self.modelgrid.ncol)
        self.shape2d = (self.shape[1], self.shape[2])
        self.nlay = self.modelgrid.nlay
        self.nrow = self.modelgrid.nrow
        self.ncol = self.modelgrid.ncol
        self.ncol = self.modelgrid.ncol
        self.nanval = nanval

        self.cell_type = 11
        self.arrays = {}

        self.smooth = smooth
        self.point_scalars = point_scalars

        # check if structured grid, vtk only supports structured grid
        assert (isinstance(self.modelgrid, StructuredGrid))

        # cbd
        self.cbd_on = False

        # get ibound
        if self.modelgrid.idomain is None:
            # ibound = None
            ibound = np.ones(self.shape)
        else:
            ibound = self.modelgrid.idomain
            # build cbd ibound
            if ibound is not None and hasattr(self.model, 'dis') and \
                    hasattr(self.model.dis, 'laycbd'):

                self.cbd = np.where(self.model.dis.laycbd.array > 0)
                ibound = np.insert(ibound, self.cbd[0] + 1, ibound[self.cbd[
                                                                    0], :, :],
                                   axis=0)
                self.cbd_on = True

        self.ibound = ibound

        # build the model vertices
        self.verts, self.iverts, self.zverts = \
            self.get_3d_vertex_connectivity()

        return

    def add_array(self, name, a, array2d=False):

        """

        Adds an array to the vtk object

        Parameters
        ----------

        name : str
            name of the array
        a : flopy array
            the array to be added to the vtk object
        array2d : bool
            True if the array is 2d

        """

        if name == 'ibound':
            return

        # if array is 2d reformat to 3d array
        if array2d:
            assert a.shape == self.shape2d
            array = np.full(self.shape, self.nanval)
            array[0, :, :] = a
            a = array

        try:

            assert a.shape == self.shape
        except AssertionError:
            return

        a = np.where(a == self.nanval, np.nan, a)
        a = a.astype(float)
        # idxs = np.argwhere(a == self.nanval)

        # add array to self.arrays
        self.arrays[name] = a
        return

    def write(self, output_file, timeval=None):
        """

        writes the stored arrays to vtk file

        Parameters
        ----------

        output_file : str
            output file name to write the vtk data

        timeval : scalar
            model time value to be stored in the time section of the vtk
            file, default is None
        """

        # make sure file ends with vtu
        assert output_file.lower().endswith(".vtu")

        # get the active data cells based on the data arrays and ibound
        actwcells3d = self._configure_data_arrays()
        actwcells = actwcells3d.ravel()

        # get the indexes of the active cells
        idxs = np.argwhere(actwcells != 0).ravel()

        # get the verts and iverts to be output
        verts = [self.verts[idx] for idx in idxs]
        iverts = self._build_iverts(verts)

        # get the total number of cells and vertices
        ncells = len(iverts)
        npoints = ncells * 8

        if self.verbose:
            print('Writing vtk file: ' + output_file)
            print('Number of point is {}, Number of cells is {}\n'.format(
                npoints, ncells))

        # open output file for writing
        f = open(output_file, 'w')

        # write xml
        indent_level = 0
        s = '<?xml version="1.0"?>'
        f.write(s + '\n')
        indent_level = start_tag(f, '<VTKFile type="UnstructuredGrid">',
                                 indent_level)

        indent_level = start_tag(f, '<UnstructuredGrid>', indent_level)

        # if time value write time section
        if timeval:

            indent_level = start_tag(f, '<FieldData>', indent_level)

            s = '<DataArray type="Float64" Name="TimeValue"' \
                ' NumberOfTuples="1" ' \
                'format="ascii" RangeMin="{0}" RangeMax="{0}">'
            indent_level = start_tag(f, s, indent_level)

            f.write(indent_level * '  ' + '{}\n'.format(timeval))

            indent_level = end_tag(f, '</DataArray>', indent_level)

            indent_level = end_tag(f, '</FieldData>', indent_level)

        # piece
        s = '<Piece NumberOfPoints="{}" ' \
            'NumberOfCells="{}">'.format(npoints, ncells)
        indent_level = start_tag(f, s, indent_level)

        # points
        s = '<Points>'
        indent_level = start_tag(f, s, indent_level)

        s = '<DataArray type="Float64" NumberOfComponents="3">'
        indent_level = start_tag(f, s, indent_level)
        assert (isinstance(self.modelgrid, StructuredGrid))
        for cell in verts:
            for row in cell:
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
            s = indent_level * '  ' + '{} \n'.format(self.cell_type)
            f.write(s)
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)

        s = '</Cells>'
        indent_level = end_tag(f, s, indent_level)

        # add cell data
        s = '<CellData Scalars="scalars">'
        indent_level = start_tag(f, s, indent_level)

        # write data arrays to file
        for arrayName, arrayValues in self.arrays.items():
            self.write_data_array(f, indent_level, arrayName,
                                  arrayValues, actwcells3d)

        s = '</CellData>'
        indent_level = end_tag(f, s, indent_level)

        # write arrays as point data if point scalars is set to True
        if self.point_scalars:
            s = '<PointData Scalars="scalars">'
            indent_level = start_tag(f, s, indent_level)
            for array_name, array_values in self.arrays.items():
                self.write_point_value(f, indent_level, array_values,
                                       array_name, actwcells3d)

            s = '</PointData>'
            indent_level = end_tag(f, s, indent_level)

        else:
            pass

        # end piece
        indent_level = end_tag(f, '</Piece>', indent_level)

        # end unstructured grid
        indent_level = end_tag(f, '</UnstructuredGrid>', indent_level)

        # end xml
        indent_level = end_tag(f, '</VTKFile>', indent_level)

        # end file
        f.close()
        self.arrays.clear()
        return

    def write_binary(self, output_file):

        """

        outputs binary .vtu file

        Parameters
        ----------

        output_file : str
            vtk output file

        """

        # make sure file ends with vtu
        assert output_file.lower().endswith(".vtu")

        if self.verbose:
            print('writing binary vtk file')

        xml = BinaryXml(output_file)
        offset = 0
        grid_type = 'UnstructuredGrid'

        # get the active data cells based on the data arrays and ibound
        actwcells3d = self._configure_data_arrays()
        actwcells = actwcells3d.ravel()

        # get the indexes of the active cells
        idxs = np.argwhere(actwcells != 0).ravel()

        # get the verts and iverts to be output
        verts = [self.verts[idx] for idx in idxs]
        iverts = self._build_iverts(verts)

        # check if there is data to be written out
        if len(verts) == 0:
            # if not cannot write binary .vtu file
            return

        # get the total number of cells and vertices
        ncells = len(iverts)
        npoints = ncells * 8

        if self.verbose:
            print('Writing vtk file: ' + output_file)
            print('Number of point is {}, Number of cells is {}\n'.format(
                npoints, ncells))

        # format verts and iverts
        verts = np.array(verts)
        verts.reshape(npoints, 3)
        iverts = np.ascontiguousarray(iverts, np.float64)

        # write xml file info
        xml.open_element("VTKFile"). \
            add_attributes(type=grid_type, version="1.0",
                           byte_order=self._get_byte_order(),
                           header_type="UInt64")
        # unstructured grid
        xml.open_element(grid_type)

        # piece
        xml.open_element('Piece')
        xml.add_attributes(NumberOfPoints=npoints, NumberOfCells=ncells)

        # points
        xml.open_element('Points')

        xml.open_element('DataArray')
        xml.add_attributes(Name='points', NumberOfComponents='3',
                           type='Float64',
                           format='appended', offset=offset)

        # calculate the offset of the start of the next piece of data
        # offset is calculated from beginning of data section
        points_size = verts.size * verts[0].dtype.itemsize
        offset += points_size + 8

        xml.close_element('DataArray')

        xml.close_element('Points')

        # cells
        xml.open_element('Cells')

        # connectivity
        xml.open_element('DataArray')
        xml.add_attributes(Name='connectivity', NumberOfComponents='1',
                           type='Float64',
                           format='appended', offset=offset)
        conn_size = iverts.size * iverts[0].dtype.itemsize
        offset += conn_size + 8

        xml.close_element('DataArray')

        xml.open_element('DataArray')
        xml.add_attributes(Name='offsets', NumberOfComponents='1',
                           type='Float64',
                           format='appended', offset=offset)
        offsets_size = iverts.shape[0] * iverts[0].dtype.itemsize
        offset += offsets_size + 8

        xml.close_element('DataArray')

        xml.open_element('DataArray')
        xml.add_attributes(Name='types', NumberOfComponents='1',
                           type='Float64',
                           format='appended', offset=offset)
        types_size = iverts.shape[0] * iverts[0].dtype.itemsize
        offset += types_size + 8

        xml.close_element('DataArray')

        xml.close_element('Cells')

        xml.open_element('CellData')
        xml.add_attributes(Scalars='scalars')

        # format data arrays and store for later output
        processed_arrays = []
        for name, a in self.arrays.items():
            a = a.ravel()[idxs]
            xml.open_element('DataArray')
            xml.add_attributes(Name=name, NumberOfComponents='1',
                               type='Float64',
                               format='appended', offset=offset)
            a = np.ascontiguousarray(a, np.float64)
            processed_arrays.append([a, a.size * a[0].dtype.itemsize])
            offset += processed_arrays[-1][-1] + 8
            xml.close_element('DataArray')

        xml.close_element('CellData')

        # for data array point scalars
        if self.point_scalars:

            xml.open_element('PointData')
            xml.add_attributes(Scalars='scalars')

            # get output point arrays
            # loop through stored arrays
            for name, a in self.arrays.items():
                # get the array values onto vertices
                verts_info = self.get_3d_vertex_connectivity(
                    actwcells=actwcells3d, zvalues=a)
                # get values
                point_values_dict = verts_info[2]
                a = np.array([point_values_dict[cellid] for cellid in
                              sorted(point_values_dict.keys())]).ravel()

                xml.open_element('DataArray')
                xml.add_attributes(Name=name, NumberOfComponents='1',
                                   type='Float64',
                                   format='appended', offset=offset)
                a = np.ascontiguousarray(a, np.float64)
                processed_arrays.append([a, a.size * a[0].dtype.itemsize])
                offset += processed_arrays[-1][-1] + 8

                xml.close_element('DataArray')
            xml.close_element('PointData')

        # end piece
        xml.close_element('Piece')

        # end unstructured grid
        xml.close_element('UnstructuredGrid')

        # build data section
        xml.open_element("AppendedData").add_attributes(
            encoding="raw").add_text("_")

        xml.write_size(points_size)
        # format verts for output
        verts_x = np.ascontiguousarray(np.ravel(verts[:, :, 0]),
                                       np.float64)
        verts_y = np.ascontiguousarray(np.ravel(verts[:, :, 1]),
                                       np.float64)
        verts_z = np.ascontiguousarray(np.ravel(verts[:, :, 2]),
                                       np.float64)
        # write coordinates
        xml.write_coord_arrays(verts_x, verts_y, verts_z)

        # write iverts
        xml.write_size(conn_size)
        rav_iverts = np.ascontiguousarray(np.ravel(iverts), np.float64)
        xml.write_array(rav_iverts)

        xml.write_size(offsets_size)
        data = np.empty((iverts.shape[0]), np.float64)
        icount = 0
        for index, row in enumerate(iverts):
            icount += len(row)
            data[index] = icount
        xml.write_array(data)

        # write cell types (11)
        xml.write_size(types_size)
        data = np.empty((iverts.shape[0]), np.float64)
        data.fill(self.cell_type)
        xml.write_array(data)

        # write out the array scalars and array point scalars
        for a, block_size in processed_arrays:
            xml.write_size(block_size)
            xml.write_array(a)

        # end xml
        xml.close_element("AppendedData")
        xml.close_element('VTKFile')
        xml.close()
        # clear arrays
        self.arrays.clear()

    def _configure_data_arrays(self):
        """
        Compares arrays and active cells to find where active data
        exists, and what cells to output.
        """

        # get 1d shape
        shape1d = self.shape[0] * self.shape[1] * self.shape[2]

        # build index array
        ot_idx_array = np.zeros(shape1d, dtype=np.int)
        # loop through arrays
        for name in self.arrays:
            array = self.arrays[name]
            # make array 1d
            a = array.ravel()
            # find where no data
            where_nan = np.isnan(a)
            # where no data set to the class nan val
            a[where_nan] = self.nanval
            # get the indexes where there is data
            idxs = np.argwhere(a != self.nanval)
            # set the active array to 1
            ot_idx_array[idxs] = 1

        # reset the shape of the active data array
        ot_idx_array = ot_idx_array.reshape(self.shape)
        # where the ibound is 0 set the active array to 0
        ot_idx_array[self.ibound == 0] = 0

        return ot_idx_array

    def get_3d_vertex_connectivity(self, actwcells=None, zvalues=None):

        """

        Builds x,y,z vertices

        Parameters
        ----------
        actwcells : array
            array of where data exists
        zvalues: array
            array of values to be used instead of the zvalues of
            the vertices.  This allows point scalars to be interpolated.

        Returns
        -------
        vertsdict : dict
            dictionary of verts
        ivertsdict : dict
            dictionary of iverts
        zvertsdict : dict
            dictionary of zverts

        """
        # set up active cells
        if actwcells is None:
            actwcells = self.ibound

        ipoint = 0

        vertsdict = {}
        ivertsdict = {}
        zvertsdict = {}

        # if smoothing interpolate the z values
        if self.smooth:
            if zvalues is not None:
                # use the given data array values
                zVertices = self.extendedDataArray(zvalues)
            else:
                zVertices = self.extendedDataArray(self.modelgrid.top_botm)
        else:
            zVertices = None

        # model cellid based on 1darray
        # build the vertices
        cellid = -1
        for k in range(self.nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    cellid += 1
                    if actwcells[k, i, j] == 0:
                        continue
                    verts = []
                    ivert = []
                    zverts = []
                    pts = self.modelgrid._cell_vert_list(i, j)
                    pt0, pt1, pt2, pt3, pt0 = pts

                    if not self.smooth:
                        cellBot = self.modelgrid.top_botm[k + 1, i, j]
                        cellTop = self.modelgrid.top_botm[k, i, j]
                        celElev = [cellBot, cellTop]
                        for elev in celElev:
                            # verts[ipoint, :] = np.append(pt1,elev)
                            verts.append([pt1[0], pt1[1], elev])
                            # verts[ipoint+1, :] = np.append(pt2,elev)
                            verts.append([pt2[0], pt2[1], elev])
                            # verts[ipoint+2, :] = np.append(pt0,elev)
                            verts.append([pt0[0], pt0[1], elev])
                            # verts[ipoint+3, :] = np.append(pt3,elev)
                            verts.append([pt3[0], pt3[1], elev])
                            ivert.extend([ipoint, ipoint+1, ipoint+2,
                                          ipoint+3])
                            zverts.extend([elev, elev, elev, elev])
                            ipoint += 4
                        vertsdict[cellid] = verts
                        ivertsdict[cellid] = ivert
                        zvertsdict[cellid] = zverts
                    else:
                        layers = [k+1, k]
                        for lay in layers:
                            verts.append([pt1[0], pt1[1], zVertices[lay, i+1,
                                                                    j]])
                            verts.append([pt2[0], pt2[1], zVertices[lay, i+1,
                                                                    j+1]])

                            verts.append([pt0[0], pt0[1], zVertices[lay, i,
                                                                    j]])

                            verts.append([pt3[0], pt3[1], zVertices[lay, i,
                                                                    j+1]])
                            ivert.extend([ipoint, ipoint+1, ipoint+2,
                                          ipoint+3])
                            zverts.extend([zVertices[lay, i+1, j], zVertices[
                                lay, i+1, j+1],
                                        zVertices[lay, i, j], zVertices[lay, i,
                                                                        j+1]])
                            ipoint += 4
                        vertsdict[cellid] = verts
                        ivertsdict[cellid] = ivert
                        zvertsdict[cellid] = zverts
        return vertsdict, ivertsdict, zvertsdict

    def extendedDataArray(self, dataArray):

        if dataArray.shape[0] == self.nlay+1:
            dataArray = dataArray
        else:
            listArray = [dataArray[0]]
            for lay in range(dataArray.shape[0]):
                listArray.append(dataArray[lay])
            dataArray = np.stack(listArray)

        matrix = np.zeros([self.nlay+1, self.nrow+1, self.ncol+1])
        for lay in range(self.nlay+1):
            for row in range(self.nrow+1):
                for col in range(self.ncol+1):

                    indexList = [[row-1, col-1], [row-1, col], [row, col-1],
                                 [row, col]]
                    neighList = []
                    for index in indexList:
                        if index[0] in range(self.nrow) and index[1] in \
                                range(self.ncol):
                            neighList.append(dataArray[lay, index[0],
                                                       index[1]])
                    neighList = np.array(neighList)
                    if neighList[neighList != self.nanval].shape[0] > 0:
                        headMean = neighList[neighList != self.nanval].mean()
                    else:
                        headMean = self.nanval
                    matrix[lay, row, col] = headMean
        return matrix

    @staticmethod
    def _get_byte_order():
        if sys.byteorder == "little":
            return "LittleEndian"
        else:
            return "BigEndian"

    @staticmethod
    def write_data_array(f, indent_level, arrayName, arrayValues,
                         actWCells):
        """

        Writes the data array to the output vtk file

        Parameters
        ----------
        f : file object
            output vtk file
        indent_level : int
            current indent of the xml
        arrayName : str
            name of the output array
        arrayValues : array
            the data array being output
        actWCells : array
            array of the active cells

        """

        s = '<DataArray type="Float64" Name="{}" format="ascii">'.format(
            arrayName)
        indent_level = start_tag(f, s, indent_level)

        # data
        nlay = arrayValues.shape[0]

        for lay in range(nlay):
            s = indent_level * '  '
            f.write(s)
            idx = (actWCells[lay] != 0)
            arrayValuesLay = arrayValues[lay][idx].flatten()
            for layValues in arrayValuesLay:
                s = ' {}'.format(layValues)
                f.write(s)
            f.write('\n')

        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)
        return

    def write_point_value(self, f, indent_level, data_array, array_name,
                          actwcells):
        """
        Writes the data array to the output vtk file as point scalars
        """
        # header tag
        s = '<DataArray type="Float64" Name="{}" format="ascii">'.format(
            array_name)
        indent_level = start_tag(f, s, indent_level)

        # data
        verts_info = self.get_3d_vertex_connectivity(
            actwcells=actwcells, zvalues=data_array)

        zverts = verts_info[2]

        for cellid in sorted(zverts):
            for z in zverts[cellid]:
                s = indent_level * '  '
                f.write(s)
                s = ' {}'.format(z)
                f.write(s)
                f.write('\n')

        # ending tag
        s = '</DataArray>'
        indent_level = end_tag(f, s, indent_level)
        return

    @staticmethod
    def _build_iverts(verts):
        """

        Builds the iverts based on the vertices being output

        Parameters
        ----------
        verts : array
            vertices being output

        Returns
        -------

        iverts : array
            array of ivert values

        """
        ncells = len(verts)
        npoints = ncells * 8
        iverts = []
        ivert = []
        count = 1
        for i in range(npoints):
            ivert.append(i)
            if count == 8:
                iverts.append(ivert)
                ivert = []
                count = 0
            count += 1
        iverts = np.array(iverts)

        return iverts


def _get_names(in_list):
    ot_list = []
    for x in in_list:
        if isinstance(x, bytes):
            ot_list.append(str(x.decode('UTF-8')))
        else:
            ot_list.append(x)
    return ot_list


def export_cbc(model, cbcfile, otfolder, precision='single', nanval=-1e+20,
               kstplist=None, kperlist=None, keylist=None, smooth=False,
               point_scalars=False, binary=False):
    """

    Exports cell by cell file to vtk

    Parameters
    ----------

    model : flopy model instance
        the flopy model instance
    cbcfile : str
        the cell by cell file
    otfolder : str
        output folder to write the data
    precision : str:
        binary file precision, default is 'single'
    nanval : scalar
        no data value
    kstplist : list
        list of timesteps
    kperlist : list
        list of stress periods
    keylist : list
        list of flow term names
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """

    mg = model.modelgrid
    shape = (mg.nlay, mg.nrow, mg.ncol)

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    # set up the pvd file to make the output files time enabled
    pvdfile = open(
        os.path.join(otfolder, '{}_Heads.pvd'.format(model.name)),
        'w')

    pvdfile.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         byte_order="LittleEndian"
         compressor="vtkZLibDataCompressor">
  <Collection>\n""")

    # load cbc

    cbb = bf.CellBudgetFile(cbcfile, precision=precision)

    # totim_dict = dict(zip(cbb.get_kstpkper(), model.dis.get_totim()))

    # get records
    records = _get_names(cbb.get_unique_record_names())

    # build imeth lookup
    imeth_dict = {record: imeth for (record, imeth) in zip(records,
                                                           cbb.imethlist)}
    # get list of packages to export
    if not keylist:
        keylist = records

    if not kperlist:
        kperlist = list(set([x[1] for x in cbb.get_kstpkper() if x[1] > -1]))
    else:
        kperlist = [kper - 1 for kper in kperlist]

    if not kstplist:
        kstplist = list(set([x[0] for x in cbb.get_kstpkper() if x[0] > -1]))
    else:
        kstplist = [kstp - 1 for kstp in kstplist]

    # get model name
    model_name = model.name

    vtk = Vtk(model, nanval=nanval, smooth=smooth, point_scalars=point_scalars)

    # export data
    addarray = False
    count = 1
    for kper in kperlist:
        for kstp in kstplist:

            ot_base = '{}_CBC_KPER{}_KSTP{}.vtu'.format(
                model_name, kper + 1, kstp + 1)
            otfile = os.path.join(otfolder, ot_base)
            pvdfile.write("""<DataSet timestep="{}" group="" part="0"
                         file="{}"/>\n""".format(count, ot_base))
            for name in keylist:

                try:
                    rec = cbb.get_data(kstpkper=(kstp, kper), text=name,
                                       full3D=True)

                    if len(rec) > 0:
                        array = rec[0]  # need to fix for multiple pak
                        addarray = True

                except ValueError:

                    rec = cbb.get_data(kstpkper=(kstp, kper), text=name)[0]

                    if imeth_dict[name] == 6:
                        array = np.full(shape, nanval)
                        # rec array
                        for [node, q] in zip(rec['node'], rec['q']):
                            lyr, row, col = np.unravel_index(node - 1, shape)

                            array[lyr, row, col] = q

                        addarray = True
                    else:
                        raise Exception('Data type not currently supported '
                                        'for cbc output')
                        # print('Data type not currently supported '
                        #       'for cbc output')

                if addarray:

                    # set the data to no data value
                    if ma.is_masked(array):
                        array = np.where(array.mask, nanval, array)

                    # add array to vtk
                    vtk.add_array(name.strip(), array)  # need to adjust for

            # write the vtk data to the output file
            if binary:
                vtk.write_binary(otfile)
            else:
                vtk.write(otfile)
            count += 1
    # finish writing the pvd file
    pvdfile.write("""  </Collection>
</VTKFile>""")

    pvdfile.close()
    return


def export_heads(model, hdsfile, otfolder, nanval=-1e+20, kstplist=None,
                 kperlist=None, smooth=False, point_scalars=False,
                 binary=False):
    """

    Exports binary head file to vtk

    Parameters
    ----------

    model : MFModel
        the flopy model instance
    hdsfile : str
        binary heads file
    otfolder : str
        output folder to write the data
    nanval : scalar
        no data value, default value is -1e20
    kstplist : list
        list of timesteps
    kperlist : list
        list of stress periods
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """

    # setup output folder
    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    # start writing the pvd file to make the data time aware
    pvdfile = open(os.path.join(otfolder, '{}_Heads.pvd'.format(model.name)),
                   'w')

    pvdfile.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         byte_order="LittleEndian"
         compressor="vtkZLibDataCompressor">
  <Collection>\n""")
    # get teh ehads
    hds = HeadFile(hdsfile)

    # get the model time
    # totim_dict = dict(zip(hds.get_kstpkper(), model.dis.get_totim()))

    # set up time step and stress periods for output
    if not kperlist:
        kperlist = list(set([x[1] for x in hds.get_kstpkper() if x[1] > -1]))
    else:
        kperlist = [kper - 1 for kper in kperlist]

    if not kstplist:
        kstplist = list(set([x[0] for x in hds.get_kstpkper() if x[0] > -1]))
    else:
        kstplist = [kstp - 1 for kstp in kstplist]

    # set upt the vtk
    vtk = Vtk(model, smooth=smooth, point_scalars=point_scalars, nanval=nanval)

    # output data
    count = 0
    for kper in kperlist:
        for kstp in kstplist:
            hdarr = hds.get_data((kstp, kper))
            vtk.add_array('head', hdarr)
            ot_base = '{}_Heads_KPER{}_KSTP{}.vtu'.format(
                model.name, kper + 1, kstp + 1)
            otfile = os.path.join(otfolder, ot_base)
            # vtk.write(otfile, timeval=totim_dict[(kstp, kper)])
            if binary:
                vtk.write_binary(otfile)
            else:
                vtk.write(otfile)
            pvdfile.write("""<DataSet timestep="{}" group="" part="0"
             file="{}"/>\n""".format(count, ot_base))
            count += 1

    pvdfile.write("""  </Collection>
</VTKFile>""")

    pvdfile.close()


def export_array(model, array, output_folder, name, nanval=-1e+20,
                 array2d=False, smooth=False, point_scalars=False,
                 binary=False):

    """

    Export array to vtk

    Parameters
    ----------

    model : flopy model instance
        the flopy model instance
    array : flopy array
        flopy 2d or 3d array
    output_folder : str
        output folder to write the data
    name : str
        name of array
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        True if array is 2d, default is False
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vtk = Vtk(model, nanval=nanval, smooth=smooth, point_scalars=point_scalars)
    vtk.add_array(name, array, array2d=array2d)
    otfile = os.path.join(output_folder, '{}.vtu'.format(name))
    if binary:
        vtk.write_binary(otfile)
    else:
        vtk.write(otfile)

    return


def export_transient(model, array, output_folder, name, nanval=-1e+20,
                     array2d=False, smooth=False, point_scalars=False,
                     binary=False):
    """

    Export transient 2d array to vtk

    Parameters
    ----------

    model : MFModel
        the flopy model instance
    array : Transient instance
        flopy transient array
    output_folder : str
        output folder to write the data
    name : str
        name of array
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        True if array is 2d, default is False
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    to_tim = model.dis.get_totim()

    vtk = Vtk(model, nanval=nanval, smooth=smooth, point_scalars=point_scalars)

    if array2d:

        for kper in range(array.shape[0]):

            t2d_array_kper = array[kper]
            t2d_array_kper_shape = t2d_array_kper.shape
            t2d_array_input = t2d_array_kper.reshape(t2d_array_kper_shape[1],
                                                     t2d_array_kper_shape[2])

            vtk.add_array(name, t2d_array_input, array2d=True)

            ot_name = '{}_0{}'.format(name, kper + 1)
            ot_file = os.path.join(output_folder, '{}.vtu'.format(ot_name))
            vtk.write(ot_file, timeval=to_tim[kper])

    else:

        for kper in range(array.shape[0]):
            vtk.add_array(name, array[kper])

            ot_name = '{}_0{}'.format(name, kper + 1)
            ot_file = os.path.join(output_folder, '{}.vtu'.format(ot_name))
            if binary:
                vtk.write_binary(ot_file)
            else:
                vtk.write(ot_file, timeval=to_tim[kper])
    return


def trans_dict(in_dict, name, trans_array, array2d=False):
    """
    Builds or adds to dictionary trans_array
    """
    if not in_dict:
        in_dict = {}
    for kper in range(trans_array.shape[0]):
        if kper not in in_dict:
            in_dict[kper] = {}
        in_dict[kper][name] = _Array(trans_array[kper], array2d=array2d)

    return in_dict


def export_package(pak_model, pak_name, ot_folder, vtkobj=None,
                   nanval=-1e+20, smooth=False, point_scalars=False,
                   binary=False):

    """
    Exports package to vtk

    Parameters
    ----------

    pak_model : flopy model instance
        the model of the package
    pak_name : str
        the name of the package
    ot_folder : str
        output folder to write the data
    vtkobj : VTK instance
        a vtk object (allows export_package to be called from
        export_model)
    nanval : scalar
        no data value, default value is -1e20
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """

    # see if there is vtk object being supplied by export_model
    if not vtkobj:
        # if not build one
        vtk = Vtk(pak_model, nanval=nanval, smooth=smooth,
                  point_scalars=point_scalars)
    else:
        # otherwise use the vtk object that was supplied
        vtk = vtkobj

    if not os.path.exists(ot_folder):
        os.mkdir(ot_folder)

    # is there output data
    has_output = False
    # is there output transient data
    vtk_trans_dict = None

    # get package
    if isinstance(pak_name, list):
        pak_name = pak_name[0]

    pak = pak_model.get_package(pak_name)

    shape_check_3d = (pak_model.modelgrid.nlay, pak_model.modelgrid.nrow,
                      pak_model.modelgrid.ncol)
    shape_check_2d = (shape_check_3d[1], shape_check_3d[2])

    # loop through the items in the package
    for item, value in pak.__dict__.items():

        if value is None or not hasattr(value, 'data_type'):
            continue

        if isinstance(value, list):
            raise NotImplementedError("LIST")

        elif isinstance(value, DataInterface):
            # if transiet list data add to the vtk_trans_dict for later output
            if value.data_type == DataType.transientlist:

                try:
                    list(value.masked_4D_arrays_itr())
                except AttributeError:

                    continue
                except ValueError:
                    continue
                has_output = True
                for name, array in value.masked_4D_arrays_itr():

                    vtk_trans_dict = trans_dict(vtk_trans_dict, name, array)

            elif value.data_type == DataType.array3d:
                # if 3d array add array to the vtk and set has_output to True
                if value.array is not None:
                    has_output = True

                    vtk.add_array(item, value.array)

            elif value.data_type == DataType.array2d and value.array.shape ==\
                    shape_check_2d:
                # if 2d array add array to vtk object and turn on has output
                if value.array is not None:
                    has_output = True
                    vtk.add_array(item, value.array, array2d=True)

            elif value.data_type == DataType.transient2d:
                # if transient data add data to vtk_trans_dict for later output
                if value.array is not None:
                    has_output = True
                    vtk_trans_dict = trans_dict(vtk_trans_dict, item,
                                                value.array, array2d=True)

            elif value.data_type == DataType.list:
                # this data type is not being output
                if value.array is not None:
                    has_output = True
                    if isinstance(value.array, np.recarray):
                        pass

                    else:
                        raise Exception('Data type not understond in data '
                                        'list')

            elif value.data_type == DataType.transient3d:
                # add to transient dictionary for output
                if value.array is not None:
                    has_output = True
                    # vtk_trans_dict = _export_transient_3d(vtk, value.array,
                    #                 vtkdict=vtk_trans_dict)
                    vtk_trans_dict = trans_dict(vtk_trans_dict, item,
                                                value.array)

            else:
                pass
        else:
            pass

    if not has_output:
        # there is no data to output
        pass

    else:

        # write out data
        # write array data
        if len(vtk.arrays) > 0:
            ot_file = os.path.join(ot_folder, '{}.vtu'.format(pak_name))
            if binary:
                vtk.write_binary(ot_file)
            else:
                vtk.write(ot_file)

        # write transient data
        if vtk_trans_dict:

            # get model time
            # to_tim = _get_basic_modeltime(pak_model.modeltime.perlen)
            # loop through the transient array data that was stored in the
            # trans_array_dict and output
            for kper, array_dict in vtk_trans_dict.items():
                # if to_tim:
                #     time = to_tim[kper]
                # else:
                #     time = None
                # set up output file
                ot_file = os.path.join(ot_folder, '{} _0{}.vtu'.format(
                    pak_name, kper + 1))
                for name, array in sorted(array_dict.items()):
                    if array.array2d:
                        array_shape = array.array.shape
                        a = array.array.reshape(array_shape[1], array_shape[2])
                    else:
                        a = array.array
                    vtk.add_array(name, a, array.array2d)
                # vtk.write(ot_file, timeval=time)
                if binary:
                    vtk.write_binary(ot_file)
                else:
                    vtk.write(ot_file)
    return


def export_model(model, ot_folder, package_names=None, nanval=-1e+20,
                 smooth=False, point_scalars=False, binary=False):
    """

    model : flopy model instance
        flopy model
    ot_folder : str
        output folder
    package_names : list
        list of package names to be exported
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        True if array is 2d, default is False
    smooth : bool
        If true a smooth surface will be output, default is False
    point_scalars : bool
        If True point scalar values will be written, default is False
    binary : bool
        if True the output .vtu file will be binary, default is
        False.

    """
    vtk = Vtk(model, nanval=nanval, smooth=smooth, point_scalars=point_scalars)

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
    else:
        package_names = [pak.name[0] for pak in ml.packagelist]

    if not os.path.exists(ot_folder):
        os.mkdir(ot_folder)

    for pak_name in package_names:
        export_package(model, pak_name, ot_folder, vtkobj=vtk, nanval=nanval,
                       smooth=smooth, point_scalars=point_scalars,
                       binary=binary)
