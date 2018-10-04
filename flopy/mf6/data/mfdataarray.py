import sys, inspect, copy
import numpy as np
from collections import OrderedDict
from ..data.mfstructure import DatumType
from ..data import mfstructure, mfdatautil, mfdata
from ..data.mfdatautil import MultiList
from ..mfbase import ExtFileAction, MFDataException
from ..utils.mfenums import DiscretizationType

class MFArray(mfdata.MFMultiDimVar):
    """
    Provides an interface for the user to access and update MODFLOW array data.

    Parameters
    ----------
    sim_data : MFSimulationData
        data contained in the simulation
    structure : MFDataStructure
        describes the structure of the data
    data : list or ndarray
        actual data
    enable : bool
        enable/disable the array
    path : tuple
        path in the data dictionary to this MFArray
    dimensions : MFDataDimensions
        dimension information related to the model, package, and array

    Methods
    -------
    new_simulation : (sim_data : MFSimulationData)
        initialize MFArray object for a new simulation
    supports_layered : bool
        Returns whether this MFArray supports layered data
    set_layered_data : (layered_data : bool)
        Sets whether this MFArray supports layered data
    store_as_external_file : (external_file_path : string, multiplier : float,
        layer_num : int)
        Stores data from layer "layer_num" to an external file at
        "external_file_path" with a multiplier "multiplier".  For unlayered
        data do not pass in "layer".
    store_as_internal_array : (multiplier : float, layer_num : int)
        Stores data from layer "layer_num" internally within the MODFLOW file
        with a multiplier "multiplier". For unlayered data do not pass in
        "layer".
    has_data : (layer_num : int) : bool
        Returns whether layer "layer_num" has any data associated with it.
        For unlayered data do not pass in "layer".
    get_data : (layer_num : int) : ndarray
        Returns the data associated with layer "layer_num".  If "layer_num" is
        None, returns all data.
    set_data : (data : ndarray/list, multiplier : float, layer_num : int)
        Sets the contents of the data at layer "layer_num" to "data" with
        multiplier "multiplier". For unlayered
        data do not pass in "layer_num".  data can have the following formats:
        1) ndarray - numpy ndarray containing all of the data
        2) [data] - python list containing all of the data
        3) val - a single constant value to be used for all of the data
        4) {'filename':filename, 'factor':fct, 'iprn':print, 'data':data} -
        dictionary defining external file information
        5) {'data':data, 'factor':fct, 'iprn':print) - dictionary defining
        internal information. Data that is layered can also be set by defining
        a list with a length equal to the number of layers in the model.
        Each layer in the list contains the data as defined in the
        formats above:
            [layer_1_val, [layer_2_array_vals],
            {'filename':file_with_layer_3_data, 'factor':fct, 'iprn':print}]

    load : (first_line : string, file_handle : file descriptor,
            block_header : MFBlockHeader, pre_data_comments : MFComment) :
            tuple (bool, string)
        Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read and
        the second item being the last line of text read from the file.
    get_file_entry : (layer : int) : string
        Returns a string containing the data in layer "layer".  For unlayered
        data do not pass in "layer".

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, data=None, enable=True, path=None,
                 dimensions=None):
        super(MFArray, self).__init__(sim_data, structure, enable, path,
                                      dimensions)
        if self.structure.layered:
            try:
                self._layer_shape = self.layer_shape()
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
        else:
            self._layer_shape = (1,)
        if self._layer_shape[0] is None:
            self._layer_shape = (1,)
        self._data_type = structure.data_item_structures[0].type
        try:
            shp_ml = MultiList(shape=self._layer_shape)
            self._data_storage = self._new_storage(shp_ml.get_total_size()
                                                   != 1)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(structure.get_model(),
                                  structure.get_package(), path,
                                  'creating storage', structure.name,
                                  inspect.stack()[0][3],
                                  type_, value_, traceback_, None,
                                  sim_data.debug, ex)
        self._last_line_info = []
        if self.structure.type == DatumType.integer:
            multiplier = [1]
        else:
            multiplier = [1.0]
        if data is not None:
            try:
                self._get_storage_obj().set_data(data, key=self._current_key,
                                                 multiplier=multiplier)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)

    def __setattr__(self, name, value):
        if name == '__setstate__':
            raise AttributeError(name)
        elif name == 'fname':
            self._get_storage_obj().layer_storage.first_item().fname = value
        elif name == 'factor':
            self._get_storage_obj().layer_storage.first_item().factor = value
        elif name == 'iprn':
            self._get_storage_obj().layer_storage.first_item().iprn = value
        elif name == 'binary':
            self._get_storage_obj().layer_storage.first_item().binary = value
        else:
            super(MFArray, self).__setattr__(name, value)

    def __getitem__(self, k):
        if isinstance(k, int):
            k = (k,)
        storage = self._get_storage_obj()
        if storage.layered and (isinstance(k, tuple) or isinstance(k, list)):
            if not storage.layer_storage.in_shape(k):
                comment = 'Could not retrieve layer {} of "{}". There' \
                          'are only {} layers available' \
                          '.'.format(k, self.structure.name,
                                     len(storage.layer_storage))
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug)
            # for layered data treat k as layer number(s)
            return storage.layer_storage[k]
        else:
            # for non-layered data treat k as an array/list index of the data
            if isinstance(k, int):
                try:
                    if len(self.get_data(apply_mult=True).shape) == 1:
                        return self.get_data(apply_mult=True)[k]
                    elif self.get_data(apply_mult=True).shape[0] == 1:
                        return self.get_data(apply_mult=True)[0, k]
                    elif self.get_data(apply_mult=True).shape[1] == 1:
                        return self.get_data(apply_mult=True)[k, 0]
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'setting data',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, None,
                                          self._simulation_data.debug, ex)

                comment = 'Unable to resolve index "{}" for ' \
                          'multidimensional data.'.format(k)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug)
            else:
                try:
                    if isinstance(k, tuple):
                        if len(k) == 3:
                            return self.get_data(apply_mult=True)[k[0], k[1], k[2]]
                        elif len(k) == 2:
                            return self.get_data(apply_mult=True)[k[0], k[1]]
                        if len(k) == 1:
                            return self.get_data(apply_mult=True)[k]
                    else:
                        return self.get_data(apply_mult=True)[(k,)]
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'setting data',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, None,
                                          self._simulation_data.debug, ex)

    def __setitem__(self, k, value):
        storage = self._get_storage_obj()
        if storage.layered:
            if isinstance(k, int):
                k = (k,)
            # for layered data treat k as a layer number
            try:
               storage.layer_storage[k].set_data(value)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)

        else:
            try:
                # for non-layered data treat k as an array/list index of the data
                a = self.get_data()
                a[k] = value
                a = a.astype(self.get_data().dtype)
                layer_storage = storage.layer_storage.first_item()
                self._get_storage_obj().set_data(a, key=self._current_key,
                                                 multiplier=layer_storage.factor)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'setting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)

    def new_simulation(self, sim_data):
        super(MFArray, self).new_simulation(sim_data)
        self._data_storage = self._new_storage(False)
        self._layer_shape = (1,)

    def supports_layered(self):
        try:
            model_grid = self._data_dimensions.get_model_grid()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'getting model grid',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, None,
                                  self._simulation_data.debug, ex)
        return self.structure.layered and \
            model_grid.grid_type() != DiscretizationType.DISU

    def set_layered_data(self, layered_data):
        if layered_data is True and self.structure.layered is False:
            if self._data_dimensions.get_model_grid().grid_type() == \
                    DiscretizationType.DISU:
                comment = 'Layered option not available for unstructured ' \
                             'grid. {}'.format(self._path)
            else:
                comment = 'Data "{}" does not support layered option. ' \
                             '{}'.format(self._data_name, self._path)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'setting layered data', self.structure.name,
                                  inspect.stack()[0][3], type_, value_,
                                  traceback_, comment,
                                  self._simulation_data.debug)
        self._get_storage_obj().layered = layered_data

    def make_layered(self):
        if self.supports_layered():
            try:
                self._get_storage_obj().make_layered()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'making data layered',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
        else:
            if self._data_dimensions.get_model_grid().grid_type() == \
                    DiscretizationType.DISU:
                comment = 'Layered option not available for unstructured ' \
                             'grid. {}'.format(self._path)
            else:
                comment = 'Data "{}" does not support layered option. ' \
                             '{}'.format(self._data_name, self._path)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'converting data to layered',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_, value_,
                                  traceback_, comment,
                                  self._simulation_data.debug)

    def store_as_external_file(self, external_file_path, multiplier=[1.0],
                               layer=None):
        if isinstance(layer, int):
            layer = (layer,)
        storage = self._get_storage_obj()
        if storage is None:
            self._set_storage_obj(self._new_storage(False, True))
        ds_index = self._resolve_layer_index(layer)

        try:
            # move data to file
            if storage.layer_storage[ds_index[0]].data_storage_type == \
                    mfdata.DataStorageType.external_file:
                storage.external_to_external(external_file_path, multiplier,
                                             layer)
            else:
                storage.internal_to_external(external_file_path, multiplier,
                                             layer)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'storing data in external file '
                                  '{}'.format(external_file_path),
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, None,
                                  self._simulation_data.debug, ex)

        # update data storage
        self._get_storage_obj().layer_storage[ds_index[0]].data_storage_type \
            = mfdata.DataStorageType.external_file
        self._get_storage_obj().layer_storage[ds_index[0]].fname = \
            external_file_path
        if multiplier is not None:
            self._get_storage_obj().layer_storage[ds_index[0]].multiplier = \
                    multiplier[0]

    def has_data(self, layer=None):
        storage = self._get_storage_obj()
        if storage is None:
            return False
        if isinstance(layer, int):
            layer = (layer,)
        try:
            return storage.has_data(layer)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'checking for data',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, None,
                                  self._simulation_data.debug, ex)

    def get_data(self, layer=None, apply_mult=False, **kwargs):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        if isinstance(layer, int):
            layer = (layer,)
        storage = self._get_storage_obj()
        if storage is not None:
            try:
                data = self._get_storage_obj().get_data(layer, apply_mult)
                if 'array' in kwargs and kwargs['array'] \
                        and isinstance(self, MFTransientArray):
                    data = np.expand_dims(data, 0)
                return data
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
        return None

    def set_data(self, data, multiplier=[1.0], layer=None):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        if isinstance(layer, int):
            layer = (layer,)
        try:
            self._get_storage_obj().set_data(data, layer, multiplier,
                                             key=self._current_key)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'setting data',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, None,
                                  self._simulation_data.debug, ex)
        self._layer_shape = self._get_storage_obj().layer_storage.list_shape

    def load(self, first_line, file_handle, block_header,
             pre_data_comments=None):
        super(MFArray, self).load(first_line, file_handle, block_header,
                                  pre_data_comments=None)

        if self.structure.layered:
            try:
                model_grid = self._data_dimensions.get_model_grid()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting model grid',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            if self._layer_shape[-1] != model_grid.num_layers():
                if model_grid.grid_type() == DiscretizationType.DISU:
                    self._layer_shape = (1,)
                else:
                    self._layer_shape = (model_grid.num_layers(),)
                    if self._layer_shape[-1] is None:
                        self._layer_shape = (1,)
                shape_ml = MultiList(shape=self._layer_shape)
                self._set_storage_obj(self._new_storage(
                    shape_ml.get_total_size() != 1, True))
        storage = self._get_storage_obj()
        # read in any pre data comments
        current_line = self._read_pre_data_comments(first_line, file_handle,
                                                    pre_data_comments)
        mfdatautil.ArrayUtil.reset_delimiter_used()
        arr_line = mfdatautil.ArrayUtil.\
            split_data_line(current_line)
        package_dim = self._data_dimensions.package_dim
        if len(arr_line) > 2:
            # check for time array series
            if arr_line[1].upper() == 'TIMEARRAYSERIES':
                tas_names = package_dim.get_tasnames()
                if arr_line[2].lower() in tas_names:
                    # this is a time series array with a valid tas variable
                    storage.data_structure_type = \
                            mfdata.DataStructureType.scalar
                    try:
                        storage.set_data(' '.join(arr_line[1:3]), 0,
                                         key=self._current_key)
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
                    return [False, None]
                else:
                    message = 'TIMEARRAYSERIES keyword not ' \
                              'followed by a valid TAS variable. '
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'loading data from file',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, message,
                                          self._simulation_data.debug)
        if not self.structure.data_item_structures[0].just_data:
            # verify keyword
            index_num, aux_var_index = self._load_keyword(arr_line, 0)
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
                    layers = self.layer_shape()
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
                self._layer_shape = layers
            elif aux_var_index is not None:
                # each layer stores a different aux variable
                layers = len(package_dim.get_aux_variables()[0]) - 1
                self._layer_shape = (layers,)
                storage.layered = True
                while storage.layer_storage.list_shape[0] < layers:
                    storage.add_layer()
            else:
                storage.flatten()
        try:
            dimensions = self._get_storage_obj().get_data_dimensions(
                self._layer_shape)
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
                                 file_handle)
        else:
            # write the aux var to it's unique index
            self._load_layer((aux_var_index,), layer_size, storage, arr_line,
                             file_handle)
        return [False, None]

    def _load_layer(self, layer, layer_size, storage, arr_line, file_handle):
        di_struct = self.structure.data_item_structures[0]
        if not di_struct.just_data or mfdatautil.max_tuple_abs_size(layer) > 0:
            arr_line = mfdatautil.ArrayUtil.\
                split_data_line(file_handle.readline())
        layer_storage = storage.layer_storage[layer]
        # if constant
        if arr_line[0].upper() == 'CONSTANT':
            if len(arr_line) < 2:
                message = 'MFArray "{}" contains a CONSTANT that is not ' \
                          'followed by a number.'.format(self._data_name)
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
            layer_storage.data_storage_type = \
                    mfdata.DataStorageType.internal_constant
            try:
                storage.store_internal([storage.convert_data(arr_line[1],
                                                             self._data_type,
                                                             di_struct)],
                                       layer, const=True, multiplier=[1.0])
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
                        mfdata.MFComment(' '.join(arr_line[2:]),
                                         self._path, self._simulation_data,
                                         layer)
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
                multiplier, print_format, flags_found = \
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
            storage.layer_storage[layer].data_storage_type = \
                mfdata.DataStorageType.internal_array

            # store anything else as a comment
            if len(arr_line) > 5:
                layer_storage.comments = \
                        mfdata.MFComment(' '.join(arr_line[5:]), self._path,
                                         self._simulation_data, layer)

            try:
                # load variable data from current file
                data_from_file = storage.read_data_from_file(layer, file_handle,
                                                             multiplier,
                                                             print_format,
                                                             di_struct)
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
            data_shaped = self._resolve_data_shape(data_from_file[0])
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

    def get_file_entry(self, layer=None,
                       ext_file_action=ExtFileAction.copy_relative_paths):
        if isinstance(layer, int):
            layer = (layer,)
        data_storage = self._get_storage_obj()
        if data_storage is None or \
                data_storage.layer_storage.get_total_size() == 0 \
                or not data_storage.has_data():
            return ''

        # determine if this is the special aux variable case
        if self.structure.name.lower() == 'aux' and data_storage.layered:
            layered_aux = True
        else:
            layered_aux = False
        # prepare indent
        indent = self._simulation_data.indent_string
        shape_ml = MultiList(shape=self._layer_shape)
        if shape_ml.get_total_size() == 1:
            data_indent = indent
        else:
            data_indent = '{}{}'.format(indent,
                                        self._simulation_data.indent_string)

        file_entry_array = []
        if data_storage.data_structure_type == mfdata.DataStructureType.scalar:
            # scalar data, like in the case of a time array series gets written
            # on a single line
            try:
                data = data_storage.get_data()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            file_entry_array.append('{}{}{}{}\n'.format(indent,
                                                        self.structure.name,
                                                        indent,
                                                        data))
        elif data_storage.layered:
            if not layered_aux:
                if not self.structure.data_item_structures[0].just_data:
                    name = self.structure.name
                    file_entry_array.append('{}{}{}{}\n'.format(indent, name,
                                                                indent,
                                                                'LAYERED'))
                else:
                    file_entry_array.append('{}{}\n'.format(indent, 'LAYERED'))

            if layer is None:
                layer_min = shape_ml.first_index()
                layer_max = copy.deepcopy(self._layer_shape)
            else:
                # set layer range
                if not shape_ml.in_shape(layer):
                    comment = 'Layer {} for variable "{}" does not exist' \
                              '.'.format(layer, self._data_name)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'getting file entry',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_, value_,
                                          traceback_, comment,
                                          self._simulation_data.debug)

                layer_min = layer
                layer_max = shape_ml.inc_shape_idx(layer)
            for layer in shape_ml.indexes(layer_min, layer_max):
                file_entry_array.append(
                        self._get_file_entry_layer(layer, data_indent,
                                                   data_storage.layer_storage[
                                                   layer].data_storage_type,
                                                   ext_file_action,
                                                   layered_aux))
        else:
            # data is not layered
            if not self.structure.data_item_structures[0].just_data:
                file_entry_array.append('{}{}\n'.format(indent,
                                                        self.structure.name))

            data_storage_type = data_storage.layer_storage[0].data_storage_type
            file_entry_array.append(
                    self._get_file_entry_layer(None, data_indent,
                                               data_storage_type,
                                               ext_file_action))

        return ''.join(file_entry_array)

    def _new_storage(self, set_layers=True, base_storage=False):
        if set_layers:
            return mfdata.DataStorage(self._simulation_data,
                                      self._data_dimensions,
                                      self.get_file_entry,
                                      mfdata.DataStorageType.internal_array,
                                      mfdata.DataStructureType.ndarray,
                                      self._layer_shape)
        else:
            return mfdata.DataStorage(self._simulation_data,
                                      self._data_dimensions,
                                      self.get_file_entry,
                                      mfdata.DataStorageType.internal_array,
                                      mfdata.DataStructureType.ndarray)

    def _get_storage_obj(self):
        return self._data_storage

    def _set_storage_obj(self, storage):
        self._data_storage = storage

    def _get_file_entry_layer(self, layer, data_indent, storage_type,
                              ext_file_action, layered_aux=False):
        if not self.structure.data_item_structures[0].just_data and \
                not layered_aux:
            indent_string = '{}{}'.format(self._simulation_data.indent_string,
                                          self._simulation_data.indent_string)
        else:
            indent_string = self._simulation_data.indent_string

        file_entry = ''
        if layered_aux:
            try:
                # display aux name
                file_entry = '{}{}\n'.format(indent_string,
                                             self._get_aux_var_name(layer))
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting aux variables',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            indent_string = '{}{}'.format(indent_string,
                                          self._simulation_data.indent_string)

        data_storage = self._get_storage_obj()
        if storage_type == mfdata.DataStorageType.internal_array:
            # internal data header + data
            format_str = self._get_internal_formatting_string(layer).upper()
            lay_str = self._get_data_layer_string(layer, data_indent).upper()
            file_entry = '{}{}{}\n{}{}'.format(file_entry, indent_string,
                                               format_str, indent_string,
                                               lay_str)
        elif storage_type == mfdata.DataStorageType.internal_constant:
            #  constant data
            try:
                const_val = data_storage.get_const_val(layer)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting constant value',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, None,
                                      self._simulation_data.debug, ex)
            const_str = self._get_constant_formatting_string(
                const_val, layer, self._data_type).upper()
            file_entry = '{}{}{}'.format(file_entry, indent_string,
                                         const_str)
        else:
            #  external data
            ext_str = self._get_external_formatting_string(layer,
                                                           ext_file_action)
            file_entry = '{}{}{}'.format(file_entry, indent_string,
                                         ext_str)
            #  add to active list of external files
            try:
                file_path = data_storage.get_external_file_path(layer)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                comment = 'Could not get external file path for layer ' \
                          '"{}"'.format(layer),
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'getting external file path',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, ex)
            package_dim = self._data_dimensions.package_dim
            model_name = package_dim.model_dim[0].model_name
            self._simulation_data.mfpath.add_ext_file(file_path, model_name)
        return file_entry

    def _get_data_layer_string(self, layer, data_indent):
        layer_data_string = ['']
        line_data_count = 0
        # iterate through data layer
        try:
            data = self._get_storage_obj().get_data(layer, False)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment =  'Could not get data for layer "{}"'.format(layer)
            raise MFDataException(self.structure.get_model(),
                                  self.structure.get_package(),
                                  self._path,
                                  'getting data',
                                  self.structure.name,
                                  inspect.stack()[0][3], type_,
                                  value_, traceback_, comment,
                                  self._simulation_data.debug, ex)
        data_iter = mfdatautil.ArrayUtil.next_item(data)
        indent_str = self._simulation_data.indent_string
        for item, last_item, new_list, nesting_change in data_iter:
            # increment data/layer counts
            line_data_count += 1
            try:
                data_lyr = self._get_storage_obj().to_string(item, self._data_type)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                comment = 'Could not convert data "{}" of type "{}" to a ' \
                          'string.'.format(item, self._data_type)
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'converting data',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, comment,
                                      self._simulation_data.debug, ex)
            layer_data_string[-1] = '{}{}{}'.format(layer_data_string[-1],
                                                    indent_str,
                                                    data_lyr)
            if self._simulation_data.wrap_multidim_arrays and \
                    (line_data_count == self._simulation_data.
                        max_columns_of_data or last_item):
                layer_data_string.append('{}'.format(data_indent))
                line_data_count = 0
        if len(layer_data_string) > 0:
            # clean up the text at the end of the array
            layer_data_string[-1] = layer_data_string[-1].strip()
        if len(layer_data_string) == 1:
            return '{}{}\n'.format(data_indent, layer_data_string[0].rstrip())
        else:
            return '\n'.join(layer_data_string)

    def _resolve_data_shape(self, data):
        try:
            dimensions = self._get_storage_obj().get_data_dimensions(
                self._layer_shape)
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

    def _resolve_layer_index(self, layer, allow_multiple_layers=False):
        # handle layered vs non-layered data
        storage = self._get_storage_obj()
        if storage.layered:
            if layer is None:
                if allow_multiple_layers:
                    layer_index = storage.get_active_layer_indices()
                else:
                    comment = 'Data "{}" is layered but no ' \
                                 'layer_num was specified' \
                                 '.'.format(self._data_name)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.structure.get_model(),
                                          self.structure.get_package(),
                                          self._path,
                                          'resolving layer index',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_, value_,
                                          traceback_, comment,
                                          self._simulation_data.debug)

            else:
                layer_index = [layer]
        else:
            layer_index = [[0]]
        return layer_index

    def _verify_data(self, data_iter, layer_num):
        # TODO: Implement
        #size = self._data_dimensions.model_subspace_size(self.structure.shape)
        return True


class MFTransientArray(MFArray, mfdata.MFTransient):
    """
    Provides an interface for the user to access and update MODFLOW transient
    array data.

    Parameters
    ----------
    sim_data : MFSimulationData
        data contained in the simulation
    structure : MFDataStructure
        describes the structure of the data
    data : list or ndarray
        actual data
    enable : bool
        enable/disable the array
    path : tuple
        path in the data dictionary to this MFArray
    dimensions : MFDataDimensions
        dimension information related to the model, package, and array

    Methods
    -------
    add_transient_key : (transient_key : int)
        Adds a new transient time allowing data for that time to be stored and
        retrieved using the key "transient_key"
    get_data : (layer_num : int, key : int) : ndarray
        Returns the data associated with layer "layer_num" during time "key".
        If "layer_num" is None, returns all data for time "key".
    set_data : (data : ndarray/list, multiplier : float, layer_num : int,
        key : int)
        Sets the contents of the data at layer "layer_num" and time "key" to
        "data" with multiplier "multiplier". For unlayered data do not pass
        in "layer_num".
    load : (first_line : string, file_handle : file descriptor,
            block_header : MFBlockHeader, pre_data_comments : MFComment) :
            tuple (bool, string)
        Loads data from first_line (the first line of data) and open file
        handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
    get_file_entry : (layer : int, key : int) : string
        Returns a string containing the data in layer "layer" at time "key".
        For unlayered data do not pass in "layer".

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, enable=True, path=None,
                 dimensions=None):
        super(MFTransientArray, self).__init__(sim_data=sim_data,
                                              structure=structure,
                                              data=None,
                                              enable=enable,
                                              path=path,
                                              dimensions=dimensions)
        self._transient_setup(self._data_storage)
        self.repeating = True

    def add_transient_key(self, transient_key):
        super(MFTransientArray, self).add_transient_key(transient_key)
        self._data_storage[transient_key] = super(MFTransientArray,
                                                  self)._new_storage()

    def get_data(self, key=None, apply_mult=True, **kwargs):
        if self._data_storage is not None and len(self._data_storage) > 0:
            if key is None:
                output = None
                sim_time = self._data_dimensions.package_dim.model_dim[
                    0].simulation_time
                num_sp = sim_time.get_num_stress_periods()
                data = None
                for sp in range(0, num_sp):
                #for key in self._data_storage.keys():
                    if sp in self._data_storage:
                        self.get_data_prep(sp)
                        data = super(MFTransientArray, self).get_data(
                            apply_mult=apply_mult, **kwargs)
                        data = np.expand_dims(data, 0)
                    else:
                        if data is None:
                            # get any data
                            self.get_data_prep(self._data_storage.key()[0])
                            data = super(MFTransientArray, self).get_data(
                                apply_mult=apply_mult, **kwargs)
                            data = np.expand_dims(data, 0)
                        if self.structure.type == DatumType.integer:
                            data = np.full_like(data, 0)
                        else:
                            data = np.full_like(data, 0.0)
                    if output is None:
                        output = data
                    else:
                        output = np.concatenate((output, data))
                return output
            else:
                self.get_data_prep(key)
                return super(MFTransientArray, self).get_data(
                    apply_mult=apply_mult)
        else:
            return None

    def set_data(self, data, multiplier=[1.0], layer=None, key=None):
        if isinstance(data, dict) or isinstance(data, OrderedDict):
            # each item in the dictionary is a list for one stress period
            # the dictionary key is the stress period the list is for
            for key, list_item in data.items():
                self._set_data_prep(list_item, key)
                super(MFTransientArray, self).set_data(list_item, multiplier,
                                                       layer)
        else:
            if key is None:
                # search for a key
                new_key_index = self.structure.first_non_keyword_index()
                if new_key_index is not None and hasattr(data, '__len__') and \
                        len(data) > new_key_index:
                    key = data[new_key_index]
                else:
                    key = 0
            self._set_data_prep(data, key)
            super(MFTransientArray, self).set_data(data, multiplier, layer)

    def get_file_entry(self, key=0,
                       ext_file_action=ExtFileAction.copy_relative_paths):
        self._get_file_entry_prep(key)
        return super(MFTransientArray, self).get_file_entry(ext_file_action=
                                                            ext_file_action)

    def load(self, first_line, file_handle, block_header,
             pre_data_comments=None):
        self._load_prep(block_header)
        return super(MFTransientArray, self).load(first_line, file_handle,
                                                  pre_data_comments)

    def _new_storage(self, set_layers=True, base_storage=False):
        if base_storage:
            return super(MFTransientArray, self)._new_storage(set_layers,
                                                              base_storage)
        else:
            return OrderedDict()

    def _set_storage_obj(self, storage):
        self._data_storage[self._current_key] = storage

    def _get_storage_obj(self):
        if self._current_key is None or \
                self._current_key not in self._data_storage:
            return None
        return self._data_storage[self._current_key]