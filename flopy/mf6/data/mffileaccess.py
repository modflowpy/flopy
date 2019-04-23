import sys, inspect
from copy import deepcopy
import numpy as np
from ..mfbase import MFDataException, VerbosityLevel
from ...utils.datautil import PyListUtil, find_keyword, DatumUtil, MultiListIter
from .mfdatautil import convert_data, to_string, MFComment
from ...utils.binaryfile import BinaryHeader
from ...utils import datautil
from ..data.mfstructure import DatumType, MFDataStructure, DataType


class MFFileAccess(object):
    def __init__(self, structure, data_dimensions, simulation_data, path,
                 current_key):
        self.structure = structure
        self._data_dimensions = data_dimensions
        self._simulation_data = simulation_data
        self._path = path
        self._current_key = current_key

    def write_binary_file(self, data, fname, text, modelgrid=None,
                          modeltime=None, stress_period=0,
                          precision='double', write_multi_layer=False):
        fd = self._open_ext_file(fname, binary=True, write=True)
        if write_multi_layer:
            for layer, value in enumerate(data):
                self._write_layer(fd, value, modelgrid, modeltime,
                                  stress_period, precision, text, fname,
                                  layer+1)
        else:
            self._write_layer(fd, data, modelgrid, modeltime, stress_period,
                              precision, text, fname)
        data.tofile(fd)
        fd.close()

    def _write_layer(self, fd, data, modelgrid, modeltime, stress_period,
                     precision, text, fname, ilay=None):
        header_data = self._get_header(modelgrid, modeltime, stress_period,
                                       precision, text, fname, ilay)
        header_data.tofile(fd)
        data.tofile(fd)

    @staticmethod
    def _get_bintype(modelgrid):
        if modelgrid.grid_type == 'vertex':
            return 'vardisv'
        elif modelgrid.grid_type == 'unstructured':
            return 'vardisu'
        else:
            return 'vardis'

    def _get_header(self, modelgrid, modeltime, stress_period, precision, text,
                    fname, ilay=None):
        # handle dis (row, col, lay), disv (ncpl, lay), and disu (nodes) cases
        if modelgrid is not None and modeltime is not None:
            pertim = modeltime.perlen[stress_period]
            totim = modeltime.perlen.sum()
            if ilay is None:
                ilay = modelgrid.nlay
            if modelgrid.grid_type == 'structured':
                return BinaryHeader.create(
                    bintype='vardis', precision=precision, text=text,
                    nrow=modelgrid.nrow, ncol=modelgrid.ncol,
                    ilay=ilay, pertim=pertim,
                    totim=totim, kstp=1, kper=stress_period+1)
            elif modelgrid.grid_type == 'vertex':
                if ilay is None:
                    ilay = modelgrid.nlay
                return BinaryHeader.create(
                    bintype='vardisv', precision=precision, text=text,
                    ncpl=modelgrid.ncpl, ilay=ilay, m3=1,
                    pertim=pertim, totim=totim, kstp=1,
                    kper=stress_period)
            elif modelgrid.grid_type == 'unstructured':
                return BinaryHeader.create(
                    bintype='vardisu', precision=precision, text=text,
                    nodes=modelgrid.idomain.size, m2=1, m3=1,
                    pertim=pertim, totim=totim, kstp=1, kper=stress_period)
            else:
                if ilay is None:
                    ilay = 1
                header = BinaryHeader.create(
                    bintype='vardis', precision=precision, text=text,
                    nrow=1, ncol=1, ilay=ilay, pertim=pertim,
                    totim=totim, kstp=1, kper=stress_period)
                if self._simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('Model grid does not have a valid type. Using '
                          'default spatial discretization header values for '
                          'binary file {}.'.format(fname))
        else:
            pertim = np.float64(1.0)
            header = BinaryHeader.create(
                bintype='vardis', precision=precision, text=text,
                nrow=1, ncol=1, ilay=1, pertim=pertim,
                totim=pertim, kstp=1, kper=stress_period)
            if self._simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('Binary file data not part of a model. Using default '
                      'spatial discretization header values for binary file '
                      '{}.'.format(fname))
        return header

    def _get_next_data_line(self, file_handle):
        end_of_file = False
        while not end_of_file:
            line = file_handle.readline()
            if line == '':
                message = 'More data expected when reading {} from file ' \
                          '{}'.format(self.structure.name, file_handle.name)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self.structure.path, 'reading data from file',
                    self.structure.name, inspect.stack()[0][3],
                    type_, value_, traceback_, message,
                    self._simulation_data.debug)
            clean_line = line.strip()
            # If comment or empty line
            if not MFComment.is_comment(clean_line, True):
                return datautil.PyListUtil.split_data_line(clean_line)

    def _read_pre_data_comments(self, line, file_handle, pre_data_comments,
                                storage):
        line_num = 0
        if pre_data_comments:
            storage.pre_data_comments = MFComment(pre_data_comments.text,
                                                  self._path,
                                                  self._simulation_data,
                                                  line_num)
        else:
            storage.pre_data_comments = None

        # read through any fully commented or empty lines
        arr_line = PyListUtil.split_data_line(line)
        while MFComment.is_comment(arr_line, True) and line != '':
            if storage.pre_data_comments:
                storage.pre_data_comments.add_text('\n')
                storage.pre_data_comments.add_text(' '.join(arr_line))
            else:
                storage.pre_data_comments = MFComment(arr_line, self._path,
                                                      self._simulation_data,
                                                      line_num)

            storage.add_data_line_comment(arr_line, line_num)

            line = file_handle.readline()
            arr_line = PyListUtil.split_data_line(line)
        return line

    def _get_aux_var_index(self, aux_name):
        aux_var_index = None
        # confirm whether the keyword found is an auxiliary variable name
        aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
        if aux_var_names:
            for aux_var_name, index in zip(aux_var_names[0],
                                           range(0,len(aux_var_names[0]))):
                if aux_name.lower() == aux_var_name.lower():
                    aux_var_index = index - 1
        return aux_var_index

    def _load_keyword(self, arr_line, index_num, keyword):
        aux_var_index = None
        if keyword != '':
            # verify keyword
            keyword_found = arr_line[index_num].lower()
            keyword_match = keyword.lower() == keyword_found
            aux_var_names = None
            if not keyword_match:
                aux_var_index = self._get_aux_var_index(keyword_found)
            if not keyword_match and aux_var_index is None:
                aux_text = ''
                if aux_var_names is not None:
                    aux_text = ' or auxiliary variables ' \
                               '{}'.format(aux_var_names[0])
                message = 'Error reading variable "{}".  Expected ' \
                             'variable keyword "{}"{} not found ' \
                             'at line "{}". {}'.format(self.structure.name,
                                                       keyword,
                                                       aux_text,
                                                       ' '.join(arr_line),
                                                       self._path)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self.structure.path, 'loading keyword',
                    self.structure.name, inspect.stack()[0][3],
                    type_, value_, traceback_, message,
                    self._simulation_data.debug)
            return (index_num + 1, aux_var_index)
        return (index_num, aux_var_index)

    def _open_ext_file(self, fname, binary=False, write=False):
        model_dim = self._data_dimensions.package_dim.model_dim[0]
        read_file = self._simulation_data.mfpath.resolve_path(
            fname, model_dim.model_name)
        if write:
            options = 'w'
        else:
            options = 'r'
        if binary:
            options = '{}b'.format(options)
        try:
            fd = open(read_file, options)
            return fd
        except:
            message = 'Unable to open file {} in mode {}.  Make sure the ' \
                      'file is not locked and the folder exists' \
                      '.'.format(read_file, options)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self._data_dimensions.structure.get_model(),
                self._data_dimensions.structure.get_package(),
                self._data_dimensions.structure.path,
                'opening external file for writing',
                self._data_dimensions.structure.name, inspect.stack()[0][3],
                type_, value_, traceback_, message,
                self._simulation_data.debug)

    @staticmethod
    def datum_to_numpy_type(datum_type):
        if datum_type == DatumType.integer:
            return np.int32, 'int'
        elif datum_type == DatumType.double_precision:
            return np.float64, 'double'
        elif datum_type == DatumType.string or \
            datum_type == DatumType.keyword:
            return np.str, 'str'
        else:
            return None, None


class MFFileAccessArray(MFFileAccess):
    def __init__(self, structure, data_dimensions, simulation_data, path,
                 current_key):
        super(MFFileAccessArray, self).__init__(
            structure, data_dimensions, simulation_data, path, current_key)

    def write_text_file(self, data, fp, data_type, data_size):
        try:
            fd = open(fp, 'w')
        except:
            message = 'Unable to open file {}.  Make sure the file ' \
                      'is not locked and the folder exists' \
                      '.'.format(fp)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self._data_dimensions.structure.get_model(),
                self._data_dimensions.structure.get_package(),
                self._data_dimensions.structure.path,
                'opening external file for writing',
                self.structure.name, inspect.stack()[0][3], type_,
                value_, traceback_, message,
                self._simulation_data.debug)
        current_size = 0
        for data_item in MultiListIter(data, True):
            if data_item[2] and current_size > 0:
                # new list/dimension, add appropriate formatting to
                # the file
                fd.write('\n')
            fd.write('{} '.format(to_string(data_item[0], data_type,
                                            self._simulation_data,
                                            self._data_dimensions)))
            current_size += 1
        if current_size != data_size:
            message = 'Not enough data for "{}" provided for file' \
                      ' {}.  Expected data size is {}, actual data ' \
                      'size is' \
                      '{}.'.format(self.structure.path, fd.name,
                                   data_size, current_size)
            type_, value_, traceback_ = sys.exc_info()
            fd.close()
            raise MFDataException(
                self._data_dimensions.structure.get_model(),
                self._data_dimensions.structure.get_package(),
                self._data_dimensions.structure.path,
                'storing external data', self.structure.name,
                inspect.stack()[0][3], type_, value_, traceback_,
                message, self._simulation_data.debug)
        fd.close()

    def read_binary_data_from_file(self, fname, data_shape, data_size,
                                   data_type, modelgrid,
                                   read_multi_layer=False):
        import flopy.utils.binaryfile as bf
        fd = self._open_ext_file(fname, True)
        numpy_type, name = self.datum_to_numpy_type(data_type)
        header_dtype = bf.BinaryHeader.set_dtype(
            bintype=self._get_bintype(modelgrid),
            precision=name)
        if read_multi_layer and len(data_shape) > 1:
            all_data = np.empty(data_shape, numpy_type)
            headers = []
            layer_shape = data_shape[1:]
            data_size = int(data_size / data_shape[0])
            for index in range(0, data_shape[0]):
                layer_data = self._read_binary_file_layer(
                    fd, fname, header_dtype, numpy_type, data_size, layer_shape)
                all_data[index, :] = layer_data[0]
                headers.append(layer_data[1])
            fd.close()
            return all_data, headers
        else:
            bin_data = self._read_binary_file_layer(
                fd, fname, header_dtype, numpy_type, data_size, data_shape)
            fd.close()
            return bin_data

    def _read_binary_file_layer(self, fd, fname, header_dtype, numpy_type,
                                data_size, data_shape):
        header_data = np.fromfile(fd, dtype=header_dtype, count=1)
        data = np.fromfile(fd, dtype=numpy_type, count=data_size)
        if data.size != data_size:
            message = 'Binary file {} does not contain expected data. ' \
                      'Expected array size {} but found size ' \
                      '{}.'.format(fname, data_size, data.size)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self._data_dimensions.structure.get_model(),
                self._data_dimensions.structure.get_package(),
                self._data_dimensions.structure.path,
                'opening external file for writing',
                self.structure.name, inspect.stack()[0][3], type_,
                value_, traceback_, message,
                self._simulation_data.debug)
        return data.reshape(data_shape), header_data

    def read_text_data_from_file(self, data_size, data_type, data_dim, layer,
                                  fname=None, fd=None, data_item=None):
        # load variable data from file
        current_size = 0
        data_out = []
        if layer is None:
            layer = 0
        close_file = False
        if fd is None:
            close_file = True
            fd = self._open_ext_file(fname)
        line = ' '
        PyListUtil.reset_delimiter_used()
        while line != '':
            line = fd.readline()
            arr_line = PyListUtil.split_data_line(line, True)
            for data in arr_line:
                if data != '':
                    if current_size == data_size:
                        if self._simulation_data.verbosity_level.value >= \
                                VerbosityLevel.normal.value:
                            path = self._data_dimensions.structure.path
                            print(
                                'WARNING: More data found than expected in '
                                'file {} for data '
                                '"{}".'.format(fd.name,
                                               path))
                        break
                    data_out.append(convert_data(data, self._data_dimensions,
                                                 data_type, data_item))
                    current_size += 1
            if current_size == data_size:
                break

        if current_size != data_size:
            message = 'Not enough data in file {} for data "{}".  ' \
                      'Expected data size {} but only found ' \
                      '{}.'.format(fd.name,
                                   self._data_dimensions.structure.name,
                                   data_size, current_size)
            type_, value_, traceback_ = sys.exc_info()
            if close_file:
                fd.close()
            raise MFDataException(
                self._data_dimensions.structure.get_model(),
                self._data_dimensions.structure.get_package(),
                self._data_dimensions.structure.path,
                'reading data file',
                self._data_dimensions.structure.name,
                inspect.stack()[0][3], type_, value_,
                traceback_, message,
                self._simulation_data.debug)

        if close_file:
            fd.close()

        data_out = np.reshape(data_out, data_dim)
        return data_out, current_size

    def load_from_package(self, first_line, file_handle, layer_shape,
                          storage, keyword, pre_data_comments=None):
        # read in any pre data comments
        current_line = self._read_pre_data_comments(first_line, file_handle,
                                                    pre_data_comments, storage)
        datautil.PyListUtil.reset_delimiter_used()
        arr_line = datautil.PyListUtil.\
            split_data_line(current_line)
        package_dim = self._data_dimensions.package_dim
        if len(arr_line) > 2:
            # check for time array series
            if arr_line[1].upper() == 'TIMEARRAYSERIES':
                storage.set_tas(arr_line[2], arr_line[1], self._current_key)
                return layer_shape, [False, None]
        if not self.structure.data_item_structures[0].just_data:
            # verify keyword
            index_num, aux_var_index = self._load_keyword(arr_line, 0, keyword)
        else:
            index_num = 0
            aux_var_index = None

        # TODO: Add species support
        # if layered supported, look for layered flag
        if self.structure.layered or aux_var_index is not None:
            if (len(arr_line) > index_num and
                    arr_line[index_num].lower() == 'layered'):
                storage.layered = True
                try:
                    layers = layer_shape
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'resolving layer dimensions',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, None,
                                          self._simulation_data.debug, ex)
                if len(layers) > 0:
                    storage.init_layers(layers)
            elif aux_var_index is not None:
                # each layer stores a different aux variable
                layers = len(package_dim.get_aux_variables()[0]) - 1
                layer_shape = (layers,)
                storage.layered = True
                while storage.layer_storage.list_shape[0] < layers:
                    storage.add_layer()
            else:
                storage.flatten()
        try:
            dimensions = storage.get_data_dimensions(
                layer_shape)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment = 'Could not get data shape for key "{}".'.format(
                self._current_key)
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'getting data shape',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, comment,
                                  self._simulation_data.debug, ex)
        layer_size = 1
        for dimension in dimensions:
            layer_size *= dimension

        if aux_var_index is None:
            # loop through the number of layers
            for layer in storage.layer_storage.indexes():
                self._load_layer(layer, layer_size, storage, arr_line,
                                 file_handle, layer_shape)
        else:
            # write the aux var to it's unique index
            self._load_layer((aux_var_index,), layer_size, storage, arr_line,
                             file_handle, layer_shape)
        return layer_shape, [False, None]

    def _load_layer(self, layer, layer_size, storage, arr_line, file_handle,
                    layer_shape):
        di_struct = self.structure.data_item_structures[0]
        if not di_struct.just_data or datautil.max_tuple_abs_size(layer) > 0:
            arr_line = self._get_next_data_line(file_handle)

        layer_storage = storage.layer_storage[layer]
        # if constant
        if arr_line[0].upper() == 'CONSTANT':
            if len(arr_line) < 2:
                message = 'MFArray "{}" contains a CONSTANT that is not ' \
                          'followed by a number.'.format(self.structure.name)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'loading data layer from file',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
            # store data
            layer_storage.set_internal_constant()
            try:
                storage.store_internal([convert_data(
                    arr_line[1], self._data_dimensions, self.structure.type,
                    di_struct)], layer, const=True, multiplier=[1.0])
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'storing data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            # store anything else as a comment
            if len(arr_line) > 2:
                layer_storage.comments = \
                        MFComment(' '.join(arr_line[2:]), self._path,
                                  self._simulation_data, layer)
        # if internal
        elif arr_line[0].upper() == 'INTERNAL':
            if len(arr_line) < 2:
                message = 'Data array "{}" contains a INTERNAL that is not ' \
                          'followed by a multiplier' \
                          '.'.format(self.structure.name)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'loading data layer from file',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)

            try:
                multiplier, print_format = \
                    storage.process_internal_line(arr_line)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'processing line of data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            storage.layer_storage[layer].set_internal_array()

            # store anything else as a comment
            if len(arr_line) > 5:
                layer_storage.comments = \
                        MFComment(' '.join(arr_line[5:]), self._path,
                                  self._simulation_data, layer)

            try:
                # load variable data from current file
                if multiplier is not None:
                    storage.layer_storage[layer].factor = multiplier
                if print_format is not None:
                    storage.layer_storage[layer].iprn = print_format
                data_from_file = self.read_text_data_from_file(
                    storage.get_data_size(layer), storage._data_type,
                    storage.get_data_dimensions(layer), layer,
                    fd=file_handle)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'reading data from file '
                                      '{}'.format(file_handle.name),
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            data_shaped = self._resolve_data_shape(data_from_file[0],
                                                   layer_shape, storage)
            try:
                storage.store_internal(data_shaped, layer, const=False,
                                       multiplier=[multiplier],
                                       print_format=print_format)
            except Exception as ex:
                comment = 'Could not store data: "{}"'.format(data_shaped)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'storing data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, ex)
            # verify correct size
            if layer_size > 0 and layer_size != data_from_file[1]:
                message = 'Data array "{}" does not contain the expected ' \
                          'amount of INTERNAL data. expected {},' \
                          ' found {}.'.format(self.structure.name,
                                              layer_size,
                                              data_from_file[1])
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'loading data layer from file',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
        elif arr_line[0].upper() == 'OPEN/CLOSE':
            try:
                storage.process_open_close_line(arr_line, layer)
            except Exception as ex:
                comment = 'Could not open open/close file specified by' \
                          ' "{}".'.format(' '.join(arr_line))
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'storing data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, ex)

    def _resolve_data_shape(self, data, layer_shape, storage):
        try:
            dimensions = storage.get_data_dimensions(layer_shape)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment = 'Could not get data shape for key "{}".'.format(
                self._current_key)
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'getting data shape',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, comment,
                                  self._simulation_data.debug, ex)
        if isinstance(data, list) or isinstance(data, np.ndarray):
            try:
                return np.reshape(data, dimensions).tolist()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                comment = 'Could not reshape data to dimensions ' \
                          '"{}".'.format(dimensions)
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'reshaping data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, ex)
        else:
            return data


class MFFileAccessList(MFFileAccess):
    def __init__(self, structure, data_dimensions, simulation_data, path,
                 current_key):
        super(MFFileAccessList, self).__init__(
            structure, data_dimensions, simulation_data, path, current_key)

    def load_from_package(self, first_line, file_handle, storage,
                          pre_data_comments=None):
        # lock things to maximize performance
        self._data_dimensions.lock()
        self._last_line_info = []
        self._data_line = None

        # read in any pre data comments
        current_line = self._read_pre_data_comments(first_line, file_handle,
                                                    pre_data_comments, storage)
        # reset data line delimiter so that the next split_data_line will
        # automatically determine the delimiter
        datautil.PyListUtil.reset_delimiter_used()
        arr_line = datautil.PyListUtil.split_data_line(current_line)
        if arr_line and (len(arr_line[0]) >= 2 and
                arr_line[0][:3].upper() == 'END'):
            return [False, arr_line]
        if len(arr_line) >= 2 and arr_line[0].upper() == 'OPEN/CLOSE':
            try:
                storage.process_open_close_line(arr_line, 0)
            except Exception as ex:
                message = 'An error occurred while processing the following' \
                          'open/close line: {}'.format(current_line)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(), self._path,
                                      'processing open/close line',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
        else:
            have_newrec_line, newrec_line, self._data_line =\
                self.read_list_data_from_file(file_handle, storage,
                                              self._current_key,
                                              current_line, self._data_line)
            return [have_newrec_line, newrec_line]

        # loop until end of block
        line = ' '
        while line != '':
            arr_line = self._get_next_data_line(file_handle)
            if arr_line and (len(arr_line[0]) >= 2 and
                    arr_line[0][:3].upper() == 'END'):
                # end of block
                self._data_dimensions.unlock()
                return [False, line]
        self._data_dimensions.unlock()
        return [False, None]

    def read_list_data_from_file(self, file_handle, storage, current_key,
                                 current_line=None, data_line=None,
                                 store_internal=True):
        data_rec = None
        data_loaded = []
        self._temp_dict = {}
        self._last_line_info = []
        store_data = False
        simple_line = False
        struct = self.structure
        if current_line is None:
            current_line = file_handle.readline()
        arr_line = PyListUtil.split_data_line(current_line)
        line_num = 0

        try:
            simple_line, data_line = self._load_list_line(
                storage, arr_line, line_num, data_loaded, True,
                current_key=current_key, data_line=data_line)[1:]
            line_num += 1
            store_data = True
        except MFDataException as err:
            # this could possibly be a constant line.
            line = file_handle.readline()
            arr_line = PyListUtil.split_data_line(line)
            if len(arr_line) >= 2 and arr_line[0].upper() == 'CONSTANT' \
                    and len(struct.data_item_structures) >= 2 and \
                    struct.data_item_structures[0].name.upper() \
                    == 'CELLID':
                # store first line as a comment
                if storage.pre_data_comments is None:
                    storage.pre_data_comments = \
                            MFComment(current_line, struct.path,
                                      self._simulation_data, 0)
                else:
                    storage.pre_data_comments.add_text(current_line)
                # store constant value for all cellids
                    storage.layer_storage.first_item().set_internal_constant()
                    if store_internal:
                        storage.store_internal(
                            convert_data(arr_line[1], self._data_dimensions,
                                         struct.data_item_structures[1].type,
                                         struct.data_item_structures[0]),
                            0, const=True, multiplier=[1.0])
                    else:
                        data_rec = storage._build_recarray(arr_line[1], None,
                                                           True)
                line = ' '
                while line != '':
                    line = file_handle.readline()
                    arr_line = PyListUtil.split_data_line(line)
                    if arr_line and (len(arr_line[0]) >= 2 and
                            arr_line[0][:3].upper() == 'END'):
                        return [False, line, data_line]
            else:
                # not a constant or open/close line, exception is valid
                comment = 'Unable to process line 1 of data list: ' \
                          '"{}"'.format(current_line)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(struct.get_model(), struct.get_package(),
                                      struct.path,
                                      'loading data list from '
                                      'package file',
                                      struct.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, err)

        if struct.type == DatumType.record or struct.type == DatumType.string:
            # records only contain a single line
            if data_loaded == [False]:
                print('break')
            storage.append_data(data_loaded)
            storage.data_dimensions.unlock()
            return [False, None, data_line]

        # get block recarray list for later processing
        recarrays = []
        parent_block = struct.parent_block
        if parent_block is not None:
            recarrays = parent_block.get_all_recarrays()
        recarray_len = len(recarrays)

        # loop until end of block
        line = ' '
        while line != '':
            line = file_handle.readline()
            arr_line = PyListUtil.split_data_line(line)
            if arr_line and (len(arr_line[0]) >= 2 and
                    arr_line[0][:3].upper() == 'END'):
                # end of block
                if store_data:
                    if store_internal:
                        # store as rec array
                        storage.store_internal(data_loaded, None, False,
                                               current_key)
                    else:
                        data_rec = storage._build_recarray(data_loaded,
                                                           current_key, True)
                storage.data_dimensions.unlock()
                return [False, line, data_line]
            if recarray_len != 1 and \
                    not MFComment.is_comment(arr_line, True):
                key = find_keyword(arr_line, struct.get_keywords())
                if key is None:
                    # unexpected text, may be start of another record
                    if store_data:
                        if store_internal:
                            storage.store_internal(data_loaded, None, False,
                                                   current_key)
                        else:
                            data_rec = storage._build_recarray(data_loaded,
                                                               current_key,
                                                               True)
                    storage.data_dimensions.unlock()
                    return [True, line, data_line]
            if simple_line and struct.num_optional == 0:
                if MFComment.is_comment(arr_line,
                                        True):
                    arr_line.insert(0, '\n')
                    storage.add_data_line_comment(arr_line, line_num)
                else:
                    # do higher performance quick load
                    self._data_line = ()
                    cellid_index = 0
                    cellid_tuple = ()
                    for index, entry in enumerate(self._last_line_info):
                        for sub_entry in entry:
                            if sub_entry[1] is not None:
                                data_structs = struct.data_item_structures
                                if sub_entry[2] > 0:
                                    # is a cellid
                                    cell_num = convert_data(
                                            arr_line[sub_entry[0]],
                                            self._data_dimensions,
                                            sub_entry[1],
                                            data_structs[index])
                                    cellid_tuple += (cell_num - 1,)
                                    # increment index
                                    cellid_index += 1
                                    if cellid_index == sub_entry[2]:
                                        # end of current cellid
                                        self._data_line += (cellid_tuple,)
                                        cellid_index = 0
                                        cellid_tuple = ()
                                else:
                                    # not a cellid
                                    self._data_line += (convert_data(
                                            arr_line[sub_entry[0]],
                                            self._data_dimensions,
                                            sub_entry[1],
                                            data_structs[index]),)
                            else:
                                self._data_line += (None,)
                    data_loaded.append(self._data_line)

            else:
                try:
                    data_line = self._load_list_line(
                        storage, arr_line, line_num, data_loaded, False,
                        current_key=current_key, data_line=data_line)[2]
                except Exception as ex:
                    comment = 'Unable to process line {} of data list: ' \
                              '"{}"'.format(line_num + 1, line)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(struct.get_model(),
                                          struct.get_package(),
                                          struct.path,
                                          'loading data list from '
                                          'package file',
                                          struct.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, comment,
                                          self._simulation_data.debug, ex)
            line_num += 1
        if store_data:
            # store as rec array
            storage.store_internal(data_loaded, None, False, current_key)
            storage.data_dimensions.unlock()
        if not store_internal:
            return data_rec
        else:
            return [False, None, data_line]

    def _load_list_line(self, storage, arr_line, line_num, data_loaded,
                        build_type_list, current_key, data_index_start=0,
                        data_set=None, ignore_optional_vars=False,
                        data_line=None):
        data_item_ks = None
        struct = self.structure
        org_data_line = data_line
        simple_line = True
        # only initialize if we are at the start of a new line
        if data_index_start == 0:
            data_set = struct
            # new line of data
            data_line = ()
            # determine if at end of block
            if arr_line and arr_line[0][:3].upper() == 'END':
                self.enabled = True
                return 0, simple_line, data_line
        data_index = data_index_start
        arr_line_len = len(arr_line)
        if MFComment.is_comment(arr_line, True) and data_index_start == 0:
            arr_line.insert(0, '\n')
            storage.add_data_line_comment(arr_line, line_num)
        else:
            # read variables
            var_index = 0
            data = ''
            for data_item_index, data_item in \
                    enumerate(data_set.data_item_structures):
                if not data_item.optional or not ignore_optional_vars:
                    if data_item.name == 'aux':
                        data_index, data_line = \
                            self._process_aux(storage, arr_line, arr_line_len,
                                              data_item, data_index, var_index,
                                              current_key, data_line)[0:2]
                    # optional mname data items are only specified if the
                    # package is part of a model
                    elif not data_item.optional or \
                            data_item.name[0:5] != 'mname' or \
                            not storage.in_model:
                        if data_item.type == DatumType.keyword:
                            data_index += 1
                            simple_line = False
                        elif data_item.type == DatumType.record:
                            # this is a record within a record, recurse into
                            # _load_line to load it
                            data_index, simple_line, data_line = \
                                self._load_list_line(
                                    storage, arr_line, line_num, data_loaded,
                                    build_type_list, current_key, data_index,
                                    data_item, False, data_line=data_line)
                            simple_line = False
                        elif data_item.name != 'boundname' or \
                                self._data_dimensions.package_dim.boundnames():
                            if data_item.optional and data == '#':
                                # comment mark found and expecting optional
                                # data_item, we are done
                                break
                            if data_index >= arr_line_len and data_item.optional:
                                break
                            more_data_expected = True
                            unknown_repeats = False
                            repeat_count = 0
                            while more_data_expected or unknown_repeats:
                                if data_index >= arr_line_len:
                                    if data_item.optional or unknown_repeats:
                                        break
                                    elif struct.num_optional >= \
                                            len(data_set.data_item_structures)\
                                            - data_item_index:
                                        # there are enough optional variables
                                        # to account for the lack of data
                                        # reload line with all optional
                                        # variables ignored
                                        data_line = org_data_line
                                        return self._load_list_line(
                                            storage, arr_line, line_num,
                                            data_loaded, build_type_list,
                                            current_key, data_index_start,
                                            data_set, True, data_line=data_line)
                                    else:
                                        comment = 'Not enough data provided ' \
                                                  'for {}. Data for required ' \
                                                  'data item "{}" not ' \
                                                  'found'.format(struct.name,
                                                                 data_item.
                                                                 name)
                                        type_, value_, \
                                        traceback_ = sys.exc_info()
                                        raise MFDataException(
                                            struct.get_model(),
                                            struct.get_package(),
                                            struct.path,
                                            'loading data list from '
                                            'package file',
                                            struct.name,
                                            inspect.stack()[0][3], type_,
                                            value_, traceback_, comment,
                                            self._simulation_data.debug)

                                data = arr_line[data_index]
                                repeat_count += 1
                                if data_item.type == DatumType.keystring:
                                    simple_line = False
                                    if repeat_count <= 1:  # only process the
                                        # keyword on the first repeat find
                                        #  data item associated with correct
                                        # keystring
                                        name_data = data.lower()
                                        if name_data not in \
                                                data_item.keystring_dict:
                                            name_data = '{}record'.format(
                                                    name_data)
                                            if name_data not in \
                                                    data_item.keystring_dict:
                                                # data does not match any
                                                # expected keywords
                                                if self._simulation_data.\
                                                    verbosity_level.value >= \
                                                    VerbosityLevel.normal.\
                                                        value:
                                                    print('WARNING: Failed to '
                                                          'process line {}.  '
                                                          'Line does not match'
                                                          ' expected keystring'
                                                          ' {}'.format(
                                                          ' '.join(arr_line),
                                                          data_item.name))
                                                break
                                        data_item_ks = \
                                            data_item.keystring_dict[
                                            name_data]
                                        if data_item_ks == 0:
                                            comment = 'Could not find ' \
                                                      'keystring ' \
                                                      '{}.'.format(name_data)
                                            type_, value_, \
                                            traceback_ = sys.exc_info()
                                            raise MFDataException(
                                                struct.get_model(),
                                                struct.get_package(),
                                                struct.path,
                                                'loading data list from '
                                                'package file',
                                                struct.name,
                                                inspect.stack()[0][3], type_,
                                                value_, traceback_, comment,
                                                self._simulation_data.debug)

                                        # keyword is always implied in a
                                        # keystring and should be stored,
                                        # add a string data_item for the
                                        # keyword
                                        if data_item.name in \
                                                self._temp_dict:
                                            # used cached data item for
                                            # performance
                                            keyword_data_item = \
                                                self._temp_dict[data_item.name]
                                        else:
                                            keyword_data_item = \
                                                deepcopy(data_item)
                                            keyword_data_item.type = \
                                                DatumType.string
                                            self._temp_dict[data_item.name] \
                                                = keyword_data_item
                                        data_index, more_data_expected, \
                                            data_line, unknown_repeats = \
                                            self._append_data_list(
                                                    storage,
                                                    keyword_data_item,
                                                    arr_line,
                                                    arr_line_len,
                                                    data_index,
                                                    var_index,
                                                    repeat_count,
                                                    current_key,
                                                    data_line)
                                    if isinstance(data_item_ks,
                                                  MFDataStructure):
                                        dis = \
                                        data_item_ks.data_item_structures
                                        for ks_data_item in dis:
                                            if ks_data_item.type != \
                                                    DatumType.keyword \
                                                    and data_index < \
                                                            arr_line_len:
                                                # data item contains additional
                                                # information
                                                data_index, more_data_expected, \
                                                    data_line, unknown_repeats = \
                                                    self._append_data_list(
                                                        storage,
                                                        ks_data_item,
                                                        arr_line,
                                                        arr_line_len,
                                                        data_index,
                                                        var_index,
                                                        repeat_count,
                                                        current_key,
                                                        data_line)
                                        while data_index < arr_line_len:
                                            try:
                                                # append remaining data
                                                # (temporary fix)
                                                data_index, more_data_expected, \
                                                    data_line, unknown_repeats = \
                                                    self._append_data_list(
                                                        storage,
                                                        ks_data_item,
                                                        arr_line,
                                                        arr_line_len,
                                                        data_index,
                                                        var_index,
                                                        repeat_count,
                                                        current_key,
                                                        data_line)
                                            except MFDataException:
                                                break
                                    else:
                                        if data_item_ks.type != \
                                                DatumType.keyword:
                                            data_index, more_data_expected, \
                                                data_line, unknown_repeats = \
                                            self._append_data_list(
                                                storage, data_item_ks, arr_line,
                                                arr_line_len, data_index,
                                                var_index, repeat_count,
                                                current_key, data_line)
                                        else:
                                            # append empty data as a placeholder.
                                            # this is necessarily to keep the
                                            # recarray a consistent shape
                                            data_line = \
                                                data_line + (None,)
                                            data_index += 1
                                else:
                                    if data_item.tagged and repeat_count == 1:
                                        # data item tagged, include data item
                                        # name as a keyword
                                        di_type = data_item.type
                                        data_item.type = DatumType.keyword
                                        data_index, more_data_expected, \
                                            data_line, unknown_repeats = \
                                            self._append_data_list(
                                                storage, data_item, arr_line,
                                                arr_line_len, data_index,
                                                var_index, repeat_count,
                                                current_key, data_line)
                                        data_item.type = di_type
                                    data_index, more_data_expected, \
                                        data_line, unknown_repeats = \
                                        self._append_data_list(
                                            storage, data_item, arr_line,
                                            arr_line_len, data_index, var_index,
                                            repeat_count, current_key,
                                            data_line)
                                if more_data_expected is None:
                                    # indeterminate amount of data expected.
                                    # keep reading data until eoln
                                    more_data_expected = \
                                        (data_index < arr_line_len)
                                simple_line = simple_line and \
                                        not unknown_repeats and \
                                        len(data_item.shape) == 0
                    var_index += 1

            # populate unused optional variables with None type
            for data_item in data_set.data_item_structures[var_index:]:
                simple_line = False
                if data_item.name == 'aux':
                    data_line = self._process_aux(
                        storage, arr_line, arr_line_len, data_item, data_index,
                        var_index, current_key, data_line)[1]
                elif data_item.name != 'boundname' or \
                        self._data_dimensions.package_dim.boundnames():
                    data_index, more_data_expected, data_line, \
                    unknown_repeats = self._append_data_list(
                        storage, data_item, None, 0, data_index, var_index, 1,
                        current_key, data_line)

            # only do final processing on outer-most record
            if data_index_start == 0:
                # if more pieces exist
                if arr_line_len > data_index + 1:
                    # FEATURE: Keep number of white space characters used in
                    # comments section
                    storage.comments[line_num] = MFComment(
                            ' '.join(arr_line[data_index+1:]), struct.path,
                            self._simulation_data, line_num)
                data_loaded.append(data_line)
        return data_index, simple_line, data_line

    def _process_aux(self, storage, arr_line, arr_line_len, data_item, data_index,
                     var_index, current_key, data_line):
        aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
        more_data_expected = False
        if aux_var_names is not None:
            for var_name in aux_var_names[0]:
                if var_name.lower() != 'auxiliary':
                    if data_index >= arr_line_len:
                        # store placeholder None
                        data_index, more_data_expected, data_line = \
                            self._append_data_list(
                                storage, data_item, None, 0, data_index,
                                var_index, 1, current_key, data_line)[0:3]
                    else:
                        # read in aux variables
                        data_index, more_data_expected, data_line = \
                            self._append_data_list(
                                storage, data_item, arr_line, arr_line_len,
                                data_index, var_index, 0, current_key,
                                data_line)[0:3]
        return data_index, data_line, more_data_expected

    def _append_data_list(self, storage, data_item, arr_line, arr_line_len,
                          data_index, var_index, repeat_count, current_key,
                          data_line):
        # append to a 2-D list which will later be converted to a numpy
        # rec array
        struct = self.structure
        self._last_line_info.append([])
        if data_item.is_cellid or (data_item.possible_cellid and
                                   storage._validate_cellid(
                                       arr_line, data_index)):
            if self._data_dimensions is None:
                comment = 'CellID field specified in for data ' \
                          '"{}" field "{}" which does not contain a model '\
                          'grid. This could be due to a problem with ' \
                          'the flopy definition files. Please get the ' \
                          'latest flopy definition files' \
                          '.'.format(struct.name, data_item.name)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(struct.get_model(),
                                      struct.get_package(), struct.path,
                                      'loading data list from package file',
                                      struct.name,
                                      inspect.stack()[0][3], type_, value_,
                                      traceback_, comment,
                                      self._simulation_data.debug)
            # read in the entire cellid
            model_grid = self._data_dimensions.get_model_grid()
            cellid_size = model_grid.get_num_spatial_coordinates()
            cellid_tuple = ()
            if not DatumUtil.is_int(arr_line[data_index]) and \
                    arr_line[data_index].lower() == 'none':
                # special case where cellid is 'none', store as tuple of
                # 'none's
                cellid_tuple = ('none',) * cellid_size
                self._last_line_info[-1].append([data_index, DatumType.string,
                                                 cellid_size])
                new_index = data_index + 1
            else:
                # handle regular cellid
                if cellid_size + data_index > arr_line_len:
                    comment = 'Not enough data found when reading cell ID ' \
                              'in data "{}" field "{}". Expected {} items ' \
                              'and found {} items'\
                              '.'.format(struct.name,
                                         data_item.name, cellid_size,
                                         arr_line_len - data_index)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(struct.get_model(),
                                          struct.get_package(),
                                          struct.path,
                                          'loading data list from package '
                                          'file', struct.name,
                                          inspect.stack()[0][3], type_, value_,
                                          traceback_, comment,
                                          self._simulation_data.debug)
                for index in range(data_index, cellid_size + data_index):
                    if not DatumUtil.is_int(arr_line[index]) or \
                            int(arr_line[index]) < 0:
                        comment = 'Expected a integer or cell ID in ' \
                                  'data "{}" field "{}".  Found {} ' \
                                  'in line "{}"' \
                                  '. '.format(struct.name,
                                              data_item.name, arr_line[index],
                                              arr_line)
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(struct.get_model(),
                                              struct.get_package(),
                                              struct.path,
                                              'loading data list from package '
                                              'file', struct.name,
                                              inspect.stack()[0][3], type_,
                                              value_,
                                              traceback_, comment,
                                              self._simulation_data.debug)

                    data_converted = convert_data(arr_line[index],
                                                  self._data_dimensions,
                                                  data_item.type)
                    cellid_tuple = cellid_tuple + (int(data_converted) - 1,)
                    self._last_line_info[-1].append([index, DatumType.integer,
                                                     cellid_size])
                new_index = data_index + cellid_size
            data_line = data_line + (cellid_tuple,)
            if data_item.shape is not None and len(data_item.shape) > 0 and \
                    data_item.shape[0] == 'ncelldim':
                # shape is the coordinate shape, which has already been read
                more_data_expected = False
                unknown_repeats = False
            else:
                more_data_expected, unknown_repeats = \
                    storage.resolve_shape_list(
                        data_item, repeat_count, current_key, data_line)
            return new_index, more_data_expected, data_line, unknown_repeats
        else:
            if arr_line is None:
                data_converted = None
                self._last_line_info[-1].append([data_index, None, 0])
            else:
                if arr_line[data_index].lower() in \
                        self._data_dimensions.package_dim.get_tsnames():
                    # references a time series, store as is
                    data_converted = arr_line[data_index].lower()
                    # override recarray data type to support writing
                    # string values
                    storage.override_data_type(var_index, object)
                    self._last_line_info[-1].append([data_index,
                                                     DatumType.string, 0])
                else:
                    data_converted = convert_data(arr_line[data_index],
                                                  self._data_dimensions,
                                                  data_item.type,
                                                  data_item)
                    self._last_line_info[-1].append([data_index,
                                                     data_item.type, 0])
            data_line = data_line + (data_converted,)
            more_data_expected, unknown_repeats = \
                storage.resolve_shape_list(
                    data_item, repeat_count, current_key, data_line)
            return data_index + 1, more_data_expected, data_line, \
                   unknown_repeats


class MFFileAccessScalar(MFFileAccess):
    def __init__(self, structure, data_dimensions, simulation_data, path,
                 current_key):
        super(MFFileAccessScalar, self).__init__(
            structure, data_dimensions, simulation_data, path, current_key)

    def load_from_package(self, first_line, file_handle, storage, data_type,
                          keyword, pre_data_comments=None):
        # read in any pre data comments
        current_line = self._read_pre_data_comments(first_line, file_handle,
                                                    pre_data_comments, storage)

        datautil.PyListUtil.reset_delimiter_used()
        arr_line = datautil.PyListUtil.\
            split_data_line(current_line)
        # verify keyword
        index_num = self._load_keyword(arr_line, 0, keyword)[0]

        # store data
        datatype = self.structure.get_datatype()
        if self.structure.type == DatumType.record:
            index = 0
            for data_item_type in self.structure.get_data_item_types():
                optional = self.structure.data_item_structures[index].optional
                if len(arr_line) <= index + 1 or \
                        data_item_type[0] != DatumType.keyword or (index > 0
                        and optional == True):
                    break
                index += 1
            first_type = self.structure.get_data_item_types()[0]
            if first_type[0] == DatumType.keyword:
                converted_data = [True]
            else:
                converted_data = []
            if first_type[0] != DatumType.keyword or index == 1:
                if self.structure.get_data_item_types()[1] != \
                        DatumType.keyword or arr_line[index].lower == \
                        self.structure.data_item_structures[index].name:
                    try:
                        converted_data.append(convert_data(
                            arr_line[index],
                            self._data_dimensions,
                            self.structure.data_item_structures[index].type,
                            self.structure.data_item_structures[0]))
                    except Exception:
                        message = 'Could not convert "{}" of type "{}" ' \
                                  'to a string.'.format(
                                    arr_line[index],
                                    self.structure.data_item_structures[index].
                                        type)
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(self.structure.get_model(),
                                              self.structure.get_package(),
                                              self._path,
                                              'converting data to string',
                                              self.structure.name,
                                              inspect.stack()[0][3], type_,
                                              value_, traceback_, message,
                                              self._simulation_data.debug)
            try:
                storage.set_data(converted_data, key=self._current_key)
                index_num += 1
            except Exception:
                message = 'Could not set data "{}" with key ' \
                          '"{}".'.format(converted_data, self._current_key)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
        elif datatype == DataType.scalar_keyword or \
                datatype == DataType.scalar_keyword_transient:
            # store as true
            try:
                storage.set_data(True, key=self._current_key)
            except Exception:
                message = 'Could not set data "True" with key ' \
                          '"{}".'.format(self._current_key)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
        else:
            data_item_struct = self.structure.data_item_structures[0]
            if len(arr_line) < 1 + index_num:
                message = 'Error reading variable "{}".  Expected data ' \
                             'after label "{}" not found at line ' \
                             '"{}".'.format(self.structure.name,
                                            data_item_struct.name.lower(),
                                            current_line)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'loading data from file',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
            try:
                converted_data = convert_data(arr_line[index_num],
                                              self._data_dimensions,
                                              data_type, data_item_struct)
            except Exception as ex:
                message = 'Could not convert "{}" of type "{}" ' \
                          'to a string.'.format(arr_line[index_num],
                                                data_type)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'converting data to string',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
            try:
                # read next word as data
                storage.set_data(converted_data, key=self._current_key)
            except Exception as ex:
                message = 'Could not set data "{}" with key ' \
                          '"{}".'.format(converted_data, self._current_key)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
            index_num += 1

        if len(arr_line) > index_num:
            # save remainder of line as comment
            storage.add_data_line_comment(arr_line[index_num:], 0)
        return [False, None]
