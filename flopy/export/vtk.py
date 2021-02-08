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

np_to_vtk_type = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
}

np_to_struct = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
}


class XmlWriterInterface:
    """
    Helps writing vtk files.

    Parameters
    ----------

    file_path : str
        output file path
    """

    def __init__(self, file_path):
        # class attributes
        self.open_tag = False
        self.current = []
        self.indent_level = 0
        self.indent_char = "  "

        # open file and initialize
        self.f = self._open_file(file_path)
        self.write_string('<?xml version="1.0"?>')

        # open VTKFile element
        self.open_element("VTKFile").add_attributes(version="0.1")

    def _open_file(self, file_path):
        """
        Open the file for writing.

        Return
        ------
        File object.
        """
        raise NotImplementedError("must define _open_file in child class")

    def write_string(self, string):
        """
        Write a string to the file.
        """
        raise NotImplementedError("must define write_string in child class")

    def open_element(self, tag):
        if self.open_tag:
            self.write_string(">")
        indent = self.indent_level * self.indent_char
        self.indent_level += 1
        tag_string = "\n" + indent + "<%s" % tag
        self.write_string(tag_string)
        self.open_tag = True
        self.current.append(tag)
        return self

    def close_element(self, tag=None):
        self.indent_level -= 1
        if tag:
            assert self.current.pop() == tag
            if self.open_tag:
                self.write_string(">")
                self.open_tag = False
            indent = self.indent_level * self.indent_char
            tag_string = "\n" + indent + "</%s>" % tag
            self.write_string(tag_string)
        else:
            self.write_string("/>")
            self.open_tag = False
            self.current.pop()
        return self

    def add_attributes(self, **kwargs):
        assert self.open_tag
        for key in kwargs:
            st = ' %s="%s"' % (key, kwargs[key])
            self.write_string(st)
        return self

    def write_line(self, text):
        if self.open_tag:
            self.write_string(">")
            self.open_tag = False
        self.write_string("\n")
        indent = self.indent_level * self.indent_char
        self.write_string(indent)
        self.write_string(text)
        return self

    def write_array(self, array, actwcells=None, **kwargs):
        """
        Write an array to the file.

        Parameters
        ----------
        array : ndarray
            the data array being output
        actwcells : array
            array of the active cells
        kwargs : dictionary
            Attributes to be added to the DataArray element
        """
        raise NotImplementedError("must define write_array in child class")

    def final(self):
        """
        Finalize the file. Must be called.
        """
        self.close_element("VTKFile")
        assert not self.open_tag
        self.f.close()


class XmlWriterAscii(XmlWriterInterface):
    """
    Helps writing ascii vtk files.

    Parameters
    ----------

    file_path : str
        output file path
    """

    def __init__(self, file_path):
        super(XmlWriterAscii, self).__init__(file_path)

    def _open_file(self, file_path):
        """
        Open the file for writing.

        Return
        ------
        File object.
        """
        return open(file_path, "w")

    def write_string(self, string):
        """
        Write a string to the file.
        """
        self.f.write(string)

    def write_array(self, array, actwcells=None, **kwargs):
        """
        Write an array to the file.

        Parameters
        ----------
        array : ndarray
            the data array being output
        actwcells : array
            array of the active cells
        kwargs : dictionary
            Attributes to be added to the DataArray element
        """
        # open DataArray element with relevant attributes
        self.open_element("DataArray")
        vtk_type = np_to_vtk_type[array.dtype.name]
        self.add_attributes(type=vtk_type)
        self.add_attributes(**kwargs)
        self.add_attributes(format="ascii")

        # write the data
        nlay = array.shape[0]
        for lay in range(nlay):
            if actwcells is not None:
                idx = actwcells[lay] != 0
                array_lay_flat = array[lay][idx].flatten()
            else:
                array_lay_flat = array[lay].flatten()
            # replace NaN values by -1e9 as there is a bug is Paraview when
            # reading NaN in ASCII mode
            # https://gitlab.kitware.com/paraview/paraview/issues/19042
            # this may be removed in the future if they fix the bug
            array_lay_flat[np.isnan(array_lay_flat)] = -1e9
            s = " ".join(["{}".format(val) for val in array_lay_flat])
            self.write_line(s)

        # close DataArray element
        self.close_element("DataArray")
        return


class XmlWriterBinary(XmlWriterInterface):
    """
    Helps writing binary vtk files.

    Parameters
    ----------

    file_path : str
        output file path

    """

    def __init__(self, file_path):
        super(XmlWriterBinary, self).__init__(file_path)

        if sys.byteorder == "little":
            self.byte_order = "<"
            self.add_attributes(byte_order="LittleEndian")
        else:
            self.byte_order = ">"
            self.add_attributes(byte_order="BigEndian")
        self.add_attributes(header_type="UInt64")

        # class attributes
        self.offset = 0
        self.byte_count_size = 8
        self.processed_arrays = []

    def _open_file(self, file_path):
        """
        Open the file for writing.

        Return
        ------
        File object.
        """
        return open(file_path, "wb")

    def write_string(self, string):
        """
        Write a string to the file.
        """
        self.f.write(str.encode(string))

    def write_array(self, array, actwcells=None, **kwargs):
        """
        Write an array to file.

        Parameters
        ----------
        array : ndarray
            the data array being output
        actwcells : array
            array of the active cells
        kwargs : dictionary
            Attributes to be added to the DataArray element
        """
        # open DataArray element with relevant attributes
        self.open_element("DataArray")
        vtk_type = np_to_vtk_type[array.dtype.name]
        self.add_attributes(type=vtk_type)
        self.add_attributes(**kwargs)
        self.add_attributes(format="appended", offset=self.offset)

        # store array for later writing (appended data section)
        if actwcells is not None:
            array = array[actwcells != 0]
        a = np.ascontiguousarray(array.ravel())
        array_size = array.size * array[0].dtype.itemsize
        self.processed_arrays.append([a, array_size])

        # calculate the offset of the start of the next piece of data
        # offset is calculated from beginning of data section
        self.offset += array_size + self.byte_count_size

        # close DataArray element
        self.close_element("DataArray")
        return

    def _write_size(self, block_size):
        # size is a 64 bit unsigned integer
        byte_order = self.byte_order + "Q"
        block_size = struct.pack(byte_order, block_size)
        self.f.write(block_size)

    def _append_array_binary(self, data):
        # see vtk documentation and more details here:
        # https://vtk.org/Wiki/VTK_XML_Formats#Appended_Data_Section
        assert data.flags["C_CONTIGUOUS"] or data.flags["F_CONTIGUOUS"]
        assert data.ndim == 1
        data_format = (
            self.byte_order + str(data.size) + np_to_struct[data.dtype.name]
        )
        binary_data = struct.pack(data_format, *data)
        self.f.write(binary_data)

    def final(self):
        """
        Finalize the file. Must be called.
        """
        # build data section
        self.open_element("AppendedData")
        self.add_attributes(encoding="raw")
        self.write_line("_")
        for a, block_size in self.processed_arrays:
            self._write_size(block_size)
            self._append_array_binary(a)
        self.close_element("AppendedData")

        # call super final
        super(XmlWriterBinary, self).final()


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
        If True, stdout is verbose
    nanval : float
        no data value, default is -1e20
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
            if True the output file will be binary, default is False

    Attributes
    ----------

    arrays : dict
        Stores data arrays added to VTK object
    """

    def __init__(
        self,
        model,
        verbose=None,
        nanval=-1e20,
        smooth=False,
        point_scalars=False,
        vtk_grid_type="auto",
        true2d=False,
        binary=False,
    ):

        if point_scalars:
            smooth = True

        if verbose is None:
            verbose = model.verbose
        self.verbose = verbose

        # set up variables
        self.model = model
        self.modelgrid = model.modelgrid
        self.nlay = self.modelgrid.nlay
        if hasattr(self.model, "dis") and hasattr(self.model.dis, "laycbd"):
            self.nlay = self.nlay + np.sum(self.model.dis.laycbd.array > 0)
        self.nrow = self.modelgrid.nrow
        self.ncol = self.modelgrid.ncol
        self.shape = (self.nlay, self.nrow, self.ncol)
        self.shape2d = (self.shape[1], self.shape[2])
        self.shape_verts = (
            self.shape[0] + 1,
            self.shape[1] + 1,
            self.shape[2] + 1,
        )
        self.shape_verts2d = (self.shape_verts[1], self.shape_verts[2])
        self.nanval = nanval

        self.arrays = {}
        self.vectors = {}

        self.smooth = smooth
        self.point_scalars = point_scalars
        self.has_cell_data = False
        self.has_point_data = False

        # check if structured grid, vtk only supports structured grid
        assert isinstance(self.modelgrid, StructuredGrid)

        # cbd
        self.cbd_on = False

        # get ibound
        if self.modelgrid.idomain is None:
            # ibound = None
            ibound = np.ones(self.shape)
        else:
            ibound = self.modelgrid.idomain
            # build cbd ibound
            if (
                ibound is not None
                and hasattr(self.model, "dis")
                and hasattr(self.model.dis, "laycbd")
            ):

                self.cbd = np.where(self.model.dis.laycbd.array > 0)
                ibound = np.insert(
                    ibound, self.cbd[0] + 1, ibound[self.cbd[0], :, :], axis=0
                )
                self.cbd_on = True

        self.ibound = ibound

        self.true2d = true2d
        self.nx = self.modelgrid.ncol
        self.ny = self.modelgrid.nrow
        self.nz = self.modelgrid.nlay
        if self.true2d:
            if self.nz == 1:
                self.nz = 0
            elif self.ny == 1:
                self.ny = 0
            elif self.nx == 1:
                self.nx = 0
            else:
                raise ValueError(
                    "The option true2d was used but the model is " "not 2d."
                )
            self.cell_type = 8
        else:
            self.cell_type = 11

        self.vtk_grid_type, self.file_extension = self._vtk_grid_type(
            vtk_grid_type
        )

        self.binary = binary

        return

    def _vtk_grid_type(self, vtk_grid_type="auto"):
        """
        Determines the vtk grid type and corresponding file extension.

        Parameters
        ----------
        vtk_grid_type : str
            Specific vtk_grid_type or 'auto'. Possible specific values are
            'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
            If 'auto', the grid type is automatically determined. Namely:
                * A regular grid (in all three directions) will be saved as an
                  'ImageData'.
                * A rectilinear (in all three directions), non-regular grid
                  will be saved as a 'RectilinearGrid'.
                * Other grids will be saved as 'UnstructuredGrid'.

        Returns
        ----------
        (vtk_grid_type, file_extension) : tuple of two strings
        """
        # if 'auto', determine the vtk grid type automatically
        if vtk_grid_type == "auto":
            if self.modelgrid.grid_type == "structured":
                if (
                    self.modelgrid.is_regular
                    or (self.modelgrid.is_regular_xy and self.nz == 0)
                    or (self.modelgrid.is_regular_xz and self.ny == 0)
                    or (self.modelgrid.is_regular_yz and self.nx == 0)
                ):
                    vtk_grid_type = "ImageData"
                elif self.modelgrid.is_rectilinear or self.nz == 0:
                    vtk_grid_type = "RectilinearGrid"
                else:
                    vtk_grid_type = "UnstructuredGrid"
            else:
                vtk_grid_type = "UnstructuredGrid"
        # otherwise, check the validity of the passed vtk_grid_type
        else:
            allowable_types = [
                "ImageData",
                "RectilinearGrid",
                "UnstructuredGrid",
            ]
            if not any(vtk_grid_type in s for s in allowable_types):
                raise ValueError(
                    '"' + vtk_grid_type + '" is not a correct '
                    "vtk_grid_type."
                )
            if (
                vtk_grid_type == "ImageData"
                or vtk_grid_type == "RectilinearGrid"
            ) and not self.modelgrid.grid_type == "structured":
                raise NotImplementedError(
                    'vtk_grid_type cannot be "'
                    + vtk_grid_type
                    + '" for a grid '
                    "that is not structured"
                )
            if (
                vtk_grid_type == "ImageData"
                and not self.modelgrid.is_regular
                and not (self.modelgrid.is_regular_xy and self.nz == 0)
                and not (self.modelgrid.is_regular_xz and self.ny == 0)
                and not (self.modelgrid.is_regular_yz and self.nx == 0)
            ):
                raise ValueError(
                    'vtk_grid_type cannot be "ImageData" for a '
                    "non-regular grid spacing"
                )
            if (
                vtk_grid_type == "RectilinearGrid"
                and not self.modelgrid.is_rectilinear
                and not self.nz == 0
            ):
                raise ValueError(
                    'vtk_grid_type cannot be "RectilinearGrid" '
                    "for a non-rectilinear grid spacing"
                )

        # determine the file extension
        if vtk_grid_type == "ImageData":
            file_extension = ".vti"
        elif vtk_grid_type == "RectilinearGrid":
            file_extension = ".vtr"
        # else vtk_grid_type == 'UnstructuredGrid'
        else:
            file_extension = ".vtu"

        # return vtk grid type and file extension
        return (vtk_grid_type, file_extension)

    def _format_array(self, a, array2d=False):
        """
        Formats array for vtk output.

        Parameters
        ----------

        name : str
            name of the array
        a : flopy array
            the array to be added to the vtk object
        array2d : bool
            True if the array is 2d

        Return
        ------
        Formatted array (note a copy is made)
        """
        # if array is 2d reformat to 3d array
        if array2d:
            if a.shape == self.shape2d:
                array = np.full(self.shape, self.nanval)
            elif a.shape == self.shape_verts2d:
                array = np.full(self.shape_verts, self.nanval)
            else:
                raise ValueError("Incompatible array size")
            array[0, :, :] = a
            a = array

        # deal with inactive cells
        inactive3d = self.ibound == 0
        if a.shape == self.shape:
            # set to nan where nanval or where ibound==0
            where_to_nan = np.logical_or(a == self.nanval, inactive3d)
            self.has_cell_data = True
        elif a.shape == self.shape_verts:
            # set to nan where ibound==0 at all 8 neighbors
            where_to_nan = np.full(self.shape_verts, True)
            where_to_nan[:-1, :-1, :-1] = inactive3d
            where_to_nan[:-1, :-1, 1:] = np.logical_and(
                where_to_nan[:-1, :-1, 1:], inactive3d
            )
            where_to_nan[:-1, 1:, :-1] = np.logical_and(
                where_to_nan[:-1, 1:, :-1], inactive3d
            )
            where_to_nan[:-1, 1:, 1:] = np.logical_and(
                where_to_nan[:-1, 1:, 1:], inactive3d
            )
            where_to_nan[1:, :-1, :-1] = np.logical_and(
                where_to_nan[1:, :-1, :-1], inactive3d
            )
            where_to_nan[1:, :-1, 1:] = np.logical_and(
                where_to_nan[1:, :-1, 1:], inactive3d
            )
            where_to_nan[1:, 1:, :-1] = np.logical_and(
                where_to_nan[1:, 1:, :-1], inactive3d
            )
            where_to_nan[1:, 1:, 1:] = np.logical_and(
                where_to_nan[1:, 1:, 1:], inactive3d
            )
            self.has_point_data = True
            self.smooth = True
        else:
            # incompatible size, skip this array
            return None
        a = np.where(where_to_nan, np.nan, a)

        return a

    def add_array(self, name, a, array2d=False):
        """
        Adds an array to the vtk object.

        Parameters
        ----------

        name : str
            Name of the array.
        a : flopy array
            The array to be added to the vtk object.
            The shape should match either grid cells or grid vertices.
        array2d : bool
            true if the array is 2d and represents the first layer,
            default is False
        """
        # format array
        a = self._format_array(a, array2d)

        # add to self.arrays
        if a is not None:
            self.arrays[name] = a

        return

    def add_vector(self, name, v, array2d=False):
        """
        Adds a vector (i.e., a tuple of arrays) to the vtk object.

        Parameters
        ----------

        name : str
            Name of the vector.
        v : tuple of arrays
            The vector to be added to the vtk object. The shape of each
            component should match either grid cells or grid vertices.
        array2d : bool
            true if the vector components are 2d arrays and represent the first
            layer, default is False

        Notes
        -----
        If the grid is rotated, the vector will be rotated too, assuming that
        the first and second components are along x and y directions,
        respectively.
        """
        # format each component of the vector
        vf = ()
        for vcomp in v:
            vcomp = self._format_array(vcomp, array2d=array2d)
            if vcomp is None:
                return
            vf = vf + (vcomp,)

        # rotate the vector according to grid
        if self.modelgrid.angrot_radians != 0.0:
            from ..utils import geometry

            vf = list(vf)
            vf[0], vf[1] = geometry.rotate(
                vf[0], vf[1], 0.0, 0.0, self.modelgrid.angrot_radians
            )
            vf = tuple(vf)

        # add to self.vectors
        self.vectors[name] = vf

        return

    def write(self, output_file, timeval=None):
        """
        Writes the stored arrays to vtk file in XML format.

        Parameters
        ----------

        output_file : str
            output file name without extension (extension is determined
            automatically)
        timeval : scalar
            model time value to be stored in the time section of the vtk
            file, default is None
        """
        # output file
        output_file = output_file + self.file_extension
        if self.verbose:
            print("Writing vtk file: " + output_file)

        # initialize xml file
        if self.binary:
            xml = XmlWriterBinary(output_file)
        else:
            xml = XmlWriterAscii(output_file)
        xml.add_attributes(type=self.vtk_grid_type)

        # grid type
        xml.open_element(self.vtk_grid_type)

        # if time value write time section
        if timeval:
            xml.open_element("FieldData")
            xml.write_array(
                np.array([timeval]),
                Name="TimeValue",
                NumberOfTuples="1",
                RangeMin="{0}",
                RangeMax="{0}",
            )
            xml.close_element("FieldData")

        if self.vtk_grid_type == "UnstructuredGrid":
            # get the active data cells based on the data arrays and ibound
            actwcells3d = self._configure_data_arrays()

            # get the verts and iverts to be output
            verts, iverts, _ = self._get_3d_vertex_connectivity(
                actwcells=actwcells3d
            )

            # check if there is data to be written out
            if len(verts) == 0:
                # if nothing, cannot write file
                return

            # get the total number of cells and vertices
            ncells = len(iverts)
            if self.true2d:
                npoints = ncells * 4
            else:
                npoints = ncells * 8
            if self.verbose:
                print(
                    "Number of point is {}, Number of cells is {}\n".format(
                        npoints, ncells
                    )
                )

            # piece
            xml.open_element("Piece")
            xml.add_attributes(NumberOfPoints=npoints, NumberOfCells=ncells)

            # points
            xml.open_element("Points")
            verts = np.array(list(verts.values()))
            verts.reshape(npoints, 3)
            xml.write_array(verts, Name="points", NumberOfComponents="3")
            xml.close_element("Points")

            # cells
            xml.open_element("Cells")

            # connectivity
            iverts = np.array(list(iverts.values()))
            xml.write_array(
                iverts, Name="connectivity", NumberOfComponents="1"
            )

            # offsets
            offsets = np.empty((iverts.shape[0]), np.int32)
            icount = 0
            for index, row in enumerate(iverts):
                icount += len(row)
                offsets[index] = icount
            xml.write_array(offsets, Name="offsets", NumberOfComponents="1")

            # types
            types = np.full((iverts.shape[0]), self.cell_type, dtype=np.uint8)
            xml.write_array(types, Name="types", NumberOfComponents="1")

            # end cells
            xml.close_element("Cells")

        elif self.vtk_grid_type == "ImageData":
            # note: in vtk, "extent" actually means indices of grid lines
            vtk_extent_str = (
                "0"
                + " "
                + str(self.nx)
                + " "
                + "0"
                + " "
                + str(self.ny)
                + " "
                + "0"
                + " "
                + str(self.nz)
            )
            xml.add_attributes(WholeExtent=vtk_extent_str)
            grid_extent = self.modelgrid.xyzextent
            vtk_origin_str = (
                str(grid_extent[0])
                + " "
                + str(grid_extent[2])
                + " "
                + str(grid_extent[4])
            )
            xml.add_attributes(Origin=vtk_origin_str)
            vtk_spacing_str = (
                str(self.modelgrid.delr[0])
                + " "
                + str(self.modelgrid.delc[0])
                + " "
                + str(self.modelgrid.top[0, 0] - self.modelgrid.botm[0, 0, 0])
            )
            xml.add_attributes(Spacing=vtk_spacing_str)

            # piece
            xml.open_element("Piece").add_attributes(Extent=vtk_extent_str)

        elif self.vtk_grid_type == "RectilinearGrid":
            # note: in vtk, "extent" actually means indices of grid lines
            vtk_extent_str = (
                "0"
                + " "
                + str(self.nx)
                + " "
                + "0"
                + " "
                + str(self.ny)
                + " "
                + "0"
                + " "
                + str(self.nz)
            )
            xml.add_attributes(WholeExtent=vtk_extent_str)

            # piece
            xml.open_element("Piece").add_attributes(Extent=vtk_extent_str)

            # grid coordinates
            xml.open_element("Coordinates")

            # along x
            xedges = self.modelgrid.xyedges[0]
            xml.write_array(xedges, Name="coord_x", NumberOfComponents="1")

            # along y
            yedges = np.flip(self.modelgrid.xyedges[1])
            xml.write_array(yedges, Name="coord_y", NumberOfComponents="1")

            # along z
            zedges = np.flip(self.modelgrid.zedges)
            xml.write_array(zedges, Name="coord_z", NumberOfComponents="1")

            # end coordinates
            xml.close_element("Coordinates")

        if self.has_cell_data:
            # cell data
            xml.open_element("CellData")

            # loop through stored arrays
            for name, a in self.arrays.items():
                if a.shape == self.shape_verts:
                    # these are dealt with later
                    continue
                if self.vtk_grid_type == "UnstructuredGrid":
                    xml.write_array(
                        a,
                        actwcells=actwcells3d,
                        Name=name,
                        NumberOfComponents="1",
                    )
                else:
                    # flip "a" so coordinates increase along with indices as in
                    # vtk
                    a = np.flip(a, axis=[0, 1])
                    xml.write_array(a, Name=name, NumberOfComponents="1")

            # loop through stored vectors
            for name, v in self.vectors.items():
                if v[0].shape == self.shape_verts:
                    # these are dealt with later
                    continue
                ncomp = len(v)
                v_as_array = np.moveaxis(np.array(v), 0, -1)
                if self.vtk_grid_type == "UnstructuredGrid":
                    shape4d = actwcells3d.shape + (ncomp,)
                    actwcells4d = actwcells3d.reshape(actwcells3d.shape + (1,))
                    actwcells4d = np.broadcast_to(actwcells4d, shape4d)
                    xml.write_array(
                        v_as_array,
                        actwcells=actwcells4d,
                        Name=name,
                        NumberOfComponents=ncomp,
                    )
                else:
                    # flip "v" so coordinates increase along with indices as in
                    # vtk
                    v_as_array = np.flip(v_as_array, axis=[0, 1])
                    xml.write_array(
                        v_as_array, Name=name, NumberOfComponents=ncomp
                    )

            # end cell data
            xml.close_element("CellData")

        if self.point_scalars or self.has_point_data:
            # point data (i.e., values at vertices)
            xml.open_element("PointData")

            # loop through stored arrays
            for name, a in self.arrays.items():
                if a.shape == self.shape:
                    if not self.point_scalars:
                        continue
                    # get the array values onto vertices
                    if self.vtk_grid_type == "UnstructuredGrid":
                        _, _, averts = self._get_3d_vertex_connectivity(
                            actwcells=actwcells3d, zvalues=a
                        )
                        a = np.array(list(averts.values()))
                    else:
                        a = self.modelgrid.array_at_verts(a)
                        a = np.flip(a, axis=[0, 1])
                        # deal with true2d
                        if self.true2d:
                            if self.nz == 0:
                                a = a[0, :, :]
                            elif self.ny == 0:
                                a = a[:, 0, :]
                            elif self.nz == 0:
                                a = a[:, :, 0]
                else:
                    if self.vtk_grid_type == "UnstructuredGrid":
                        # still need to do this to be consistent with
                        # connectivity (i.e. 8 points for every cell)
                        _, _, averts = self._get_3d_vertex_connectivity(
                            actwcells=actwcells3d, zvalues=a
                        )
                        a = np.array(list(averts.values()))
                    else:
                        # flip "a" so coordinates increase along with indices
                        # as in vtk
                        a = np.flip(a, axis=[0, 1])
                        # deal with true2d
                        if self.true2d:
                            if self.nz == 0:
                                a = a[0, :, :]
                            elif self.ny == 0:
                                a = a[:, 0, :]
                            elif self.nz == 0:
                                a = a[:, :, 0]
                xml.write_array(a, Name=name, NumberOfComponents="1")

            # loop through stored vectors
            for name, v in self.vectors.items():
                if v[0].shape == self.shape:
                    if not self.point_scalars:
                        continue
                    # get the vector values onto vertices
                    v_verts = ()
                    for vcomp in v:
                        if self.vtk_grid_type == "UnstructuredGrid":
                            _, _, averts = self._get_3d_vertex_connectivity(
                                actwcells=actwcells3d, zvalues=vcomp
                            )
                            vcomp = np.array(list(averts.values()))
                        else:
                            vcomp = self.modelgrid.array_at_verts(vcomp)
                            vcomp = np.flip(vcomp, axis=[0, 1])
                            # deal with true2d
                            if self.true2d:
                                if self.nz == 0:
                                    vcomp = vcomp[0, :, :]
                                elif self.ny == 0:
                                    vcomp = vcomp[:, 0, :]
                                elif self.nz == 0:
                                    vcomp = vcomp[:, :, 0]
                        v_verts = v_verts + (vcomp,)
                    v = v_verts
                else:
                    v_verts = ()
                    for vcomp in v:
                        if self.vtk_grid_type == "UnstructuredGrid":
                            # still need to do this to be consistent with
                            # connectivity (i.e. 8 points for every cell)
                            _, _, averts = self._get_3d_vertex_connectivity(
                                actwcells=actwcells3d, zvalues=vcomp
                            )
                            vcomp = np.array(list(averts.values()))
                        else:
                            vcomp = np.flip(vcomp, axis=[0, 1])
                            # deal with true2d
                            if self.true2d:
                                if self.nz == 0:
                                    vcomp = vcomp[0, :, :]
                                elif self.ny == 0:
                                    vcomp = vcomp[:, 0, :]
                                elif self.nz == 0:
                                    vcomp = vcomp[:, :, 0]
                        v_verts = v_verts + (vcomp,)
                    v = v_verts
                # write to file
                ncomp = len(v)
                v_as_array = np.moveaxis(np.array(v), 0, -1)
                xml.write_array(
                    v_as_array, Name=name, NumberOfComponents=ncomp
                )

            # end point data
            xml.close_element("PointData")

        # end piece
        xml.close_element("Piece")

        # end vtk_grid_type
        xml.close_element(self.vtk_grid_type)

        # finalize and close xml file
        xml.final()

        # clear arrays
        self.arrays.clear()
        self.vectors.clear()

    def _configure_data_arrays(self):
        """
        Compares arrays and active cells to find where active data
        exists, and what cells to output.
        """
        # build 1d index array
        shape1d = self.shape[0] * self.shape[1] * self.shape[2]
        actwcells1d = np.zeros(shape1d, dtype=int)
        if self.has_point_data:
            shape1d_verts = (
                self.shape_verts[0] * self.shape_verts[1] * self.shape_verts[2]
            )
            actwcells1d_verts = np.zeros(shape1d_verts, dtype=int)

        # loop through arrays
        for a in self.arrays.values():
            # make array 1d
            a1d = a.ravel()
            # get the indexes where there is data
            idxs = np.argwhere(np.logical_not(np.isnan(a1d)))
            # set the active array to 1
            if a.shape == self.shape:
                actwcells1d[idxs] = 1
            elif self.has_point_data:
                actwcells1d_verts[idxs] = 1

        # loop through vectors
        for v in self.vectors.values():
            for vcomp in v:
                # make array 1d
                vcomp1d = vcomp.ravel()
                # get the indexes where there is data
                idxs = np.argwhere(np.logical_not(np.isnan(vcomp1d)))
                # set the active array to 1
                if vcomp.shape == self.shape:
                    actwcells1d[idxs] = 1
                elif self.has_point_data:
                    actwcells1d_verts[idxs] = 1

        # reshape to 3D array
        actwcells3d = actwcells1d.reshape(self.shape)
        if self.has_point_data:
            actwcells3d_verts = actwcells1d_verts.reshape(self.shape_verts)
            # activate cells that are neighbor of 8 active vertices
            activate = np.full(self.shape, True)
            activate[actwcells3d_verts[:-1, :-1, :-1] == 0] = False
            activate[actwcells3d_verts[:-1, :-1, 1:] == 0] = False
            activate[actwcells3d_verts[:-1, 1:, :-1] == 0] = False
            activate[actwcells3d_verts[:-1, 1:, 1:] == 0] = False
            activate[actwcells3d_verts[1:, :-1, :-1] == 0] = False
            activate[actwcells3d_verts[1:, :-1, 1:] == 0] = False
            activate[actwcells3d_verts[1:, 1:, :-1] == 0] = False
            activate[actwcells3d_verts[1:, 1:, 1:] == 0] = False
            activate[self.ibound == 0] = False
            actwcells3d[activate] = 1

        return actwcells3d

    def _get_3d_vertex_connectivity(self, actwcells=None, zvalues=None):
        """
        Builds x,y,z vertices.

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
                if zvalues.shape == self.shape:
                    # interpolate using the given values
                    zVertices = self.modelgrid.array_at_verts(zvalues)
                else:
                    # in this case the given values are already at vertices
                    zVertices = zvalues
            else:
                zVertices = self.modelgrid.zverts_smooth
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

                    # determine z values
                    if self.nz == 0 and zvalues is None:
                        elev = np.nanmin(
                            self.modelgrid.top_botm_withnan[k + 1, :, :]
                        )
                        zvals = [
                            [elev, elev, elev, elev],
                            [elev, elev, elev, elev],
                        ]
                    elif not self.smooth:
                        zbot = self.modelgrid.top_botm[k + 1, i, j]
                        ztop = self.modelgrid.top_botm[k, i, j]
                        zvals = [
                            [zbot, zbot, zbot, zbot],
                            [ztop, ztop, ztop, ztop],
                        ]
                    else:
                        zvals = [
                            [
                                zVertices[k + 1, i + 1, j],
                                zVertices[k + 1, i + 1, j + 1],
                                zVertices[k + 1, i, j],
                                zVertices[k + 1, i, j + 1],
                            ],
                            [
                                zVertices[k, i + 1, j],
                                zVertices[k, i + 1, j + 1],
                                zVertices[k, i, j],
                                zVertices[k, i, j + 1],
                            ],
                        ]

                    # fill in the output lists
                    if self.nz == 0:
                        verts.append([pt1[0], pt1[1], zvals[0][0]])
                        verts.append([pt2[0], pt2[1], zvals[0][1]])
                        verts.append([pt0[0], pt0[1], zvals[0][2]])
                        verts.append([pt3[0], pt3[1], zvals[0][3]])
                        ivert.extend(
                            [ipoint, ipoint + 1, ipoint + 2, ipoint + 3]
                        )
                        zverts.extend(
                            [
                                zvals[0][0],
                                zvals[0][1],
                                zvals[0][2],
                                zvals[0][3],
                            ]
                        )
                        ipoint += 4
                    elif self.ny == 0:
                        verts.append([pt1[0], pt1[1], zvals[0][0]])
                        verts.append([pt2[0], pt2[1], zvals[0][1]])
                        verts.append([pt1[0], pt1[1], zvals[1][0]])
                        verts.append([pt2[0], pt2[1], zvals[1][1]])
                        ivert.extend(
                            [ipoint, ipoint + 1, ipoint + 2, ipoint + 3]
                        )
                        zverts.extend(
                            [
                                zvals[0][0],
                                zvals[0][1],
                                zvals[1][0],
                                zvals[1][1],
                            ]
                        )
                        ipoint += 4
                    elif self.nx == 0:
                        verts.append([pt1[0], pt1[1], zvals[0][0]])
                        verts.append([pt0[0], pt0[1], zvals[0][2]])
                        verts.append([pt1[0], pt1[1], zvals[1][0]])
                        verts.append([pt0[0], pt0[1], zvals[1][2]])
                        ivert.extend(
                            [ipoint, ipoint + 1, ipoint + 2, ipoint + 3]
                        )
                        zverts.extend(
                            [
                                zvals[0][0],
                                zvals[0][2],
                                zvals[1][0],
                                zvals[1][2],
                            ]
                        )
                        ipoint += 4
                    else:
                        for zvals_l in zvals:
                            verts.append([pt1[0], pt1[1], zvals_l[0]])
                            verts.append([pt2[0], pt2[1], zvals_l[1]])
                            verts.append([pt0[0], pt0[1], zvals_l[2]])
                            verts.append([pt3[0], pt3[1], zvals_l[3]])
                            ivert.extend(
                                [ipoint, ipoint + 1, ipoint + 2, ipoint + 3]
                            )
                            zverts.extend(zvals_l)
                            ipoint += 4
                    vertsdict[cellid] = verts
                    ivertsdict[cellid] = ivert
                    zvertsdict[cellid] = zverts
        return vertsdict, ivertsdict, zvertsdict


def _get_names(in_list):
    ot_list = []
    for x in in_list:
        if isinstance(x, bytes):
            ot_list.append(str(x.decode("UTF-8")))
        else:
            ot_list.append(x)
    return ot_list


def export_cbc(
    model,
    cbcfile,
    otfolder,
    precision="single",
    verbose=False,
    nanval=-1e20,
    kstpkper=None,
    text=None,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
):
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
    precision : str
        Precision of data in the cell by cell file: 'single' or 'double'.
        Default is 'single'.
    verbose : bool
        If True, write information to the screen. Default is False.
    nanval : scalar
        no data value
    kstpkper : tuple of ints or list of tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    text : str or list of str
        The text identifier for the record.  Examples include
        'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """

    mg = model.modelgrid
    shape = (mg.nlay, mg.nrow, mg.ncol)

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    # set up the pvd file to make the output files time enabled
    pvdfilename = model.name + "_CBC.pvd"
    pvdfile = open(os.path.join(otfolder, pvdfilename), "w")

    pvdfile.write(
        """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         byte_order="LittleEndian"
         compressor="vtkZLibDataCompressor">
  <Collection>\n"""
    )

    # load cbc
    cbb = bf.CellBudgetFile(cbcfile, precision=precision, verbose=verbose)

    # totim_dict = dict(zip(cbb.get_kstpkper(), model.dis.get_totim()))

    # get records
    records = _get_names(cbb.get_unique_record_names())

    # build imeth lookup
    imeth_dict = {
        record: imeth for (record, imeth) in zip(records, cbb.imethlist)
    }
    # get list of packages to export
    if text is not None:
        # build keylist
        if isinstance(text, str):
            keylist = [text]
        elif isinstance(text, list):
            keylist = text
        else:
            raise Exception("text must be type str or list of str")
    else:
        keylist = records

    # get the export times
    if kstpkper is not None:
        if isinstance(kstpkper, tuple):
            kstpkper = [kstpkper]
        elif not isinstance(kstpkper, list) or not isinstance(
            kstpkper[0], tuple
        ):
            raise Exception(
                "kstpkper must be a tuple (kstp, kper) or a list " "of tuples"
            )
    else:
        kstpkper = cbb.get_kstpkper()

    # get model name
    model_name = model.name

    vtk = Vtk(
        model,
        nanval=nanval,
        smooth=smooth,
        point_scalars=point_scalars,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )

    # export data
    addarray = False
    count = 1
    for kstpkper_i in kstpkper:
        ot_base = "{}_CBC_KPER{}_KSTP{}".format(
            model_name, kstpkper_i[1] + 1, kstpkper_i[0] + 1
        )
        otfile = os.path.join(otfolder, ot_base)
        pvdfile.write(
            """<DataSet timestep="{}" group="" part="0"
                     file="{}"/>\n""".format(
                count, ot_base
            )
        )
        for name in keylist:

            try:
                rec = cbb.get_data(kstpkper=kstpkper_i, text=name, full3D=True)

                if len(rec) > 0:
                    array = rec[0]  # need to fix for multiple pak
                    addarray = True

            except ValueError:

                rec = cbb.get_data(kstpkper=kstpkper_i, text=name)[0]

                if imeth_dict[name] == 6:
                    array = np.full(shape, nanval)
                    # rec array
                    for [node, q] in zip(rec["node"], rec["q"]):
                        lyr, row, col = np.unravel_index(node - 1, shape)

                        array[lyr, row, col] = q

                    addarray = True
                else:
                    raise Exception(
                        "Data type not currently supported " "for cbc output"
                    )
                    # print('Data type not currently supported '
                    #       'for cbc output')

            if addarray:

                # set the data to no data value
                if ma.is_masked(array):
                    array = np.where(array.mask, nanval, array)

                # add array to vtk
                vtk.add_array(name.strip(), array)  # need to adjust for

        # write the vtk data to the output file
        vtk.write(otfile)
        count += 1
    # finish writing the pvd file
    pvdfile.write(
        """  </Collection>
</VTKFile>"""
    )

    pvdfile.close()
    return


def export_heads(
    model,
    hdsfile,
    otfolder,
    text="head",
    precision="auto",
    verbose=False,
    nanval=-1e20,
    kstpkper=None,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
):
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
    text : string
        Name of the text string in the head file.  Default is 'head'.
    precision : str
        Precision of data in the head file: 'auto', 'single' or 'double'.
        Default is 'auto'.
    verbose : bool
        If True, write information to the screen. Default is False.
    nanval : scalar
        no data value, default value is -1e20
    kstpkper : tuple of ints or list of tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """

    # setup output folder
    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    # start writing the pvd file to make the data time aware
    pvdfilename = model.name + "_" + text + ".pvd"
    pvdfile = open(os.path.join(otfolder, pvdfilename), "w")

    pvdfile.write(
        """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         byte_order="LittleEndian"
         compressor="vtkZLibDataCompressor">
  <Collection>\n"""
    )

    # get the heads
    hds = HeadFile(hdsfile, text=text, precision=precision, verbose=verbose)

    # get the export times
    if kstpkper is not None:
        if isinstance(kstpkper, tuple):
            kstpkper = [kstpkper]
        elif not isinstance(kstpkper, list) or not isinstance(
            kstpkper[0], tuple
        ):
            raise Exception(
                "kstpkper must be a tuple (kstp, kper) or a list " "of tuples"
            )
    else:
        kstpkper = hds.get_kstpkper()

    # set upt the vtk
    vtk = Vtk(
        model,
        smooth=smooth,
        point_scalars=point_scalars,
        nanval=nanval,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )

    # output data
    count = 0
    for kstpkper_i in kstpkper:
        hdarr = hds.get_data(kstpkper_i)
        vtk.add_array(text, hdarr)
        ot_base = ("{}_" + text + "_KPER{}_KSTP{}").format(
            model.name, kstpkper_i[1] + 1, kstpkper_i[0] + 1
        )
        otfile = os.path.join(otfolder, ot_base)
        # vtk.write(otfile, timeval=totim_dict[(kstp, kper)])
        vtk.write(otfile)
        pvdfile.write(
            """<DataSet timestep="{}" group="" part="0"
         file="{}"/>\n""".format(
                count, ot_base
            )
        )
        count += 1

    pvdfile.write(
        """  </Collection>
</VTKFile>"""
    )

    pvdfile.close()


def export_array(
    model,
    array,
    output_folder,
    name,
    nanval=-1e20,
    array2d=False,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
):
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
        true if the array is 2d and represents the first layer, default is
        False
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vtk = Vtk(
        model,
        nanval=nanval,
        smooth=smooth,
        point_scalars=point_scalars,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )
    vtk.add_array(name, array, array2d=array2d)
    otfile = os.path.join(output_folder, "{}".format(name))
    vtk.write(otfile)

    return


def export_vector(
    model,
    vector,
    output_folder,
    name,
    nanval=-1e20,
    array2d=False,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
):

    """

    Export vector (i.e., a tuple of arrays) to vtk

    Parameters
    ----------

    model : flopy model instance
        the flopy model instance
    vector : tuple of arrays
        vector to be exported
    output_folder : str
        output folder to write the data
    name : str
        name of vector
    nanval : scalar
        no data value, default value is -1e20
    array2d : bool
        true if the vector components are 2d arrays and represent the first
        layer, default is False
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vtk = Vtk(
        model,
        nanval=nanval,
        smooth=smooth,
        point_scalars=point_scalars,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )
    vtk.add_vector(name, vector, array2d=array2d)
    otfile = os.path.join(output_folder, "{}".format(name))
    vtk.write(otfile)

    return


def export_transient(
    model,
    array,
    output_folder,
    name,
    nanval=-1e20,
    array2d=False,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
    kpers=None,
):
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
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    to_tim = model.dis.get_totim()

    vtk = Vtk(
        model,
        nanval=nanval,
        smooth=smooth,
        point_scalars=point_scalars,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )

    if name.endswith("_"):
        separator = ""
    else:
        separator = "_"

    if kpers is None:
        kpers = range(array.shape[0])
    else:
        assert isinstance(kpers, list) or isinstance(kpers, np.ndarray)

    if array2d:
        for kper in kpers:

            t2d_array_kper = array[kper]
            t2d_array_kper_shape = t2d_array_kper.shape
            t2d_array_input = t2d_array_kper.reshape(
                t2d_array_kper_shape[1], t2d_array_kper_shape[2]
            )

            vtk.add_array(name, t2d_array_input, array2d=True)

            otname = "{}".format(name) + separator + "0{}".format(kper + 1)
            otfile = os.path.join(output_folder, "{}".format(otname))
            vtk.write(otfile, timeval=to_tim[kper])

    else:
        for kper in kpers:
            vtk.add_array(name, array[kper])

            otname = "{}".format(name) + separator + "0{}".format(kper + 1)
            otfile = os.path.join(output_folder, "{}".format(otname))
            vtk.write(otfile, timeval=to_tim[kper])
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


def export_package(
    pak_model,
    pak_name,
    otfolder,
    vtkobj=None,
    nanval=-1e20,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
    kpers=None,
):
    """
    Exports package to vtk

    Parameters
    ----------

    pak_model : flopy model instance
        the model of the package
    pak_name : str
        the name of the package
    otfolder : str
        output folder to write the data
    vtkobj : VTK instance
        a vtk object (allows export_package to be called from
        export_model)
    nanval : scalar
        no data value, default value is -1e20
    smooth : bool
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """

    # see if there is vtk object being supplied by export_model
    if not vtkobj:
        # if not build one
        vtk = Vtk(
            pak_model,
            nanval=nanval,
            smooth=smooth,
            point_scalars=point_scalars,
            vtk_grid_type=vtk_grid_type,
            true2d=true2d,
            binary=binary,
        )
    else:
        # otherwise use the vtk object that was supplied
        vtk = vtkobj

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    # is there output data
    has_output = False
    # is there output transient data
    vtk_trans_dict = None

    # get package
    if isinstance(pak_name, list):
        pak_name = pak_name[0]

    pak = pak_model.get_package(pak_name)

    shape_check_3d = (
        pak_model.modelgrid.nlay,
        pak_model.modelgrid.nrow,
        pak_model.modelgrid.ncol,
    )
    shape_check_2d = (shape_check_3d[1], shape_check_3d[2])

    # loop through the items in the package
    for item, value in pak.__dict__.items():

        if value is None or not hasattr(value, "data_type"):
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

            elif (
                value.data_type == DataType.array2d
                and value.array.shape == shape_check_2d
            ):
                # if 2d array add array to vtk object and turn on has output
                if value.array is not None:
                    has_output = True
                    vtk.add_array(item, value.array, array2d=True)

            elif value.data_type == DataType.transient2d:
                # if transient data add data to vtk_trans_dict for later output
                if value.array is not None:
                    has_output = True
                    vtk_trans_dict = trans_dict(
                        vtk_trans_dict, item, value.array, array2d=True
                    )

            elif value.data_type == DataType.list:
                # this data type is not being output
                if value.array is not None:
                    has_output = True
                    if isinstance(value.array, np.recarray):
                        pass

                    else:
                        raise Exception(
                            "Data type not understond in data " "list"
                        )

            elif value.data_type == DataType.transient3d:
                # add to transient dictionary for output
                if value.array is not None:
                    has_output = True
                    # vtk_trans_dict = _export_transient_3d(vtk, value.array,
                    #                 vtkdict=vtk_trans_dict)
                    vtk_trans_dict = trans_dict(
                        vtk_trans_dict, item, value.array
                    )

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
            otfile = os.path.join(otfolder, "{}".format(pak_name))
            vtk.write(otfile)

        # write transient data
        if vtk_trans_dict:

            # only retain requested stress periods
            if kpers is not None:
                assert isinstance(kpers, list) or isinstance(kpers, np.ndarray)
                vtk_trans_dict = {kper: vtk_trans_dict[kper] for kper in kpers}

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
                otfile = os.path.join(
                    otfolder, "{}_0{}".format(pak_name, kper + 1)
                )
                for name, array in sorted(array_dict.items()):
                    if array.array2d:
                        array_shape = array.array.shape
                        a = array.array.reshape(array_shape[1], array_shape[2])
                    else:
                        a = array.array
                    vtk.add_array(name, a, array.array2d)
                # vtk.write(otfile, timeval=time)
                vtk.write(otfile)
    return


def export_model(
    model,
    otfolder,
    package_names=None,
    nanval=-1e20,
    smooth=False,
    point_scalars=False,
    vtk_grid_type="auto",
    true2d=False,
    binary=False,
    kpers=None,
):
    """
    Exports model to vtk

    Parameters
    ----------

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
        if True, will create smooth layer elevations, default is False
    point_scalars : bool
        if True, will also output array values at cell vertices, default is
        False; note this automatically sets smooth to True
    vtk_grid_type : str
        Specific vtk_grid_type or 'auto' (default). Possible specific values
        are 'ImageData', 'RectilinearGrid', and 'UnstructuredGrid'.
        If 'auto', the grid type is automatically determined. Namely:
            * A regular grid (in all three directions) will be saved as an
              'ImageData'.
            * A rectilinear (in all three directions), non-regular grid
              will be saved as a 'RectilinearGrid'.
            * Other grids will be saved as 'UnstructuredGrid'.
    true2d : bool
        If True, the model is expected to be 2d (1 layer, 1 row or 1 column)
        and the data will be exported as true 2d data, default is False.
    binary : bool
        if True the output file will be binary, default is False
    kpers : iterable of int
        Stress periods to export. If None (default), all stress periods will be
        exported.
    """
    vtk = Vtk(
        model,
        nanval=nanval,
        smooth=smooth,
        point_scalars=point_scalars,
        vtk_grid_type=vtk_grid_type,
        true2d=true2d,
        binary=binary,
    )

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
    else:
        package_names = [pak.name[0] for pak in model.packagelist]

    if not os.path.exists(otfolder):
        os.mkdir(otfolder)

    for pak_name in package_names:
        export_package(
            model,
            pak_name,
            otfolder,
            vtkobj=vtk,
            nanval=nanval,
            smooth=smooth,
            point_scalars=point_scalars,
            vtk_grid_type=vtk_grid_type,
            true2d=true2d,
            binary=binary,
            kpers=kpers,
        )
