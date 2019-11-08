from __future__ import print_function, division
import os
import numpy as np
from ..discretization import StructuredGrid
from ..datbase import DataType, DataInterface
import flopy.utils.binaryfile as bf
from flopy.utils import HeadFile
import numpy.ma as ma

# Module for exporting vtk from flopy


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

    def __init__(self, model, verbose=None, nanval=-1e+20, smooth=False,
                 point_scalars=False):

        """
        Make Vtk object for exporting flopy vtk

        :param model: flopy model instance
        :param verbose: if True, stdout is verbose
        :param nanval: no data value
        :param smooth: if True will create a smooth output surface
        :param point_scalars: if True will output poin scalar values,
        this will set smooth to True.
        """

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

        self.cbd_on = False

        # get ibound
        if self.modelgrid.idomain is None:
            # ibound = None
            ibound = np.ones(self.shape)
        else:
            ibound = self.modelgrid.idomain

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
        :param name: name of the array
        :param a: the array to be added to the vtk object
        :param array2d: True if the array is 2d

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
            # raise AssertionError
        # if a.dtype == int:
        a = a.astype(float)
        # add array to self.arrays
        self.arrays[name] = a
        return

    def write(self, output_file, timeval=None):
        """
        writes the arrays from self.arrays to vtk file

        :param output_file: output file name to write the vtk data
        :param timeval: model time value to be stored in the time section of
        the vtk file
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

    def _configure_data_arrays(self):
        """
        Compares all the stored arrays int the vtk class along with the
        ibound to figure out what cells and points need to be written to the
        output vtk file.
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

        :param actwcells: array of where data exists
        :param zvalues: array of values to be used instead of the zvalues of
        the vertices.  This allows point scalars to be interpolated.
        :return: dictionaries of verts, iverts, and zvalues
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
                    if neighList[neighList > self.nanval].shape[0] > 0:
                        headMean = neighList[neighList > self.nanval].mean()
                    else:
                        headMean = self.nanval
                    matrix[lay, row, col] = headMean
        return matrix

    @staticmethod
    def write_data_array(f, indent_level, arrayName, arrayValues,
                         actWCells):
        """

        Writes the data array to the output vtk file

        :param f: output vtk file
        :param indent_level: current indent of the xml
        :param arrayName: name of the output array
        :param arrayValues: the data array being output
        :param actWCells: array of the active cells

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

        :param verts: vertices being output
        :return: iverts for output
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


def export_cbc(model, cbcfile, otfolder, precision='single', kstplist=None,
               kperlist=None, keylist=None, smooth=False, point_scalars=False,
               nanval=-1e+20):
    """

    Exports cell by cell file to vtk

    :param model: the flopy model instance
    :param cbcfile: the cell by cell file
    :param otfolder: output folder to write the data to
    :param precision: bindary file precision
    :param kstplist: list of timesteps to be written
    :param kperlist: list of stress periods to be writeen
    :param keylist: list of flow term names to be output
    :param smooth: If true a smooth surface will be output
    :param point_scalars: If True point scalar values will be written
    :param nanval: the no data value

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
                        raise Exception('Data type not currenlty supported '
                                        'for cbc output')

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
    pvdfile.write("""  </Collection>
</VTKFile>""")

    pvdfile.close()
    return


def export_heads(model, hdsfile, otfolder, kstplist=None, kperlist=None,
                 smooth=False, point_scalars=False, nanval=-1e+20):
    """
    Exports heads to vtk files by timestep and stressperiod

    :param model: the model instance
    :param hdsfile: the binary heads file
    :param otfolder: the output folder to write the .vtu files
    :param kstplist: list of time steps to output
    :param kperlist: list of stress periods to output
    :param smooth: If set to True a smooth surface will be output
    :param point_scalars: If set to True the heads will be written to the
    vertices as point scalars as well as cell values
    :param nanval: The no data value
    :return: Heads will be written to files named by stress period and
    timestep to the otfolder
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
            vtk.write(otfile)
            pvdfile.write("""<DataSet timestep="{}" group="" part="0"
             file="{}"/>\n""".format(count, ot_base))
            count += 1

    pvdfile.write("""  </Collection>
</VTKFile>""")

    pvdfile.close()


def export_array(model, array, output_folder, name, nanval=-1e+20,
                 array2d=False, smooth=False, point_scalars=False):

    """

    :param model: array the model belongs to
    :param array: array to be exported
    :param arrayname: name of the array to be used in vtk file
    :param output_folder: output folder to store the output .vtu file
    :param array2d: True if array is 2d
    :nanval nan value
    :return: outputs a .vtu file of array

    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vtk = Vtk(model, nanval=nanval, smooth=smooth, point_scalars=point_scalars)
    vtk.add_array(name, array, array2d=array2d)
    otfile = os.path.join(output_folder, '{}.vtu'.format(name))
    vtk.write(otfile)

    return


def export_transient(model, array, output_folder, name, nanval=-1e+20,
                     array2d=False, smooth=False, point_scalars=False):

    """

    :param model: model of transient array
    :param array: transient array to export
    :param name: name of the data
    :param output_folder: output folder location
    :param array2d: True if array is 2d
    :param nanval: nan value
    :return: ouputs .vtu files of transient data to output folder
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
                   nanval=-1e+20, smooth=False, point_scalars=False):

    """
    Exports package to vtk

    :param pak_model: the model of the package
    :param pak_name: the name of the package
    :param ot_folder: the folder to output the package data
    :param vtkobj: a vtk object (allows export_package to be called from
    export_model)
    :param nanval: no data value
    :param smooth: If True the output will be a smooth represenation
    :param point_scalars: If True the package data will be written as point
    values as well as cell values.
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
                vtk.write(ot_file)
    return


def export_model(model, ot_folder, package_names=None, nanval=-1e+20,
                 smooth=False, point_scalars=False):
    """

    :param model: flopy model instance
    :param ot_folder: output folder
    :param package_names: list ofpackage names to be exported
    :param nanval: no data value
    :param smooth: smoothing
    :param point_scalars: If True array data will be written as point scalars
    as well as cell scalars

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
                       smooth=smooth, point_scalars=point_scalars)
