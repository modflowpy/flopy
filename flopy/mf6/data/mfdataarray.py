import numpy as np
from collections import OrderedDict
from ..data.mfstructure import DatumType
from ..data import mfstructure, mfdatautil, mfdata
from ..mfbase import ExtFileAction
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
        4) {'filename':filename, 'factor':fct, 'iprn':print} - dictionary
        defining external file information
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
            model_grid = self._data_dimensions.get_model_grid()
            if model_grid.grid_type() == DiscretizationType.DISU:
                self._number_of_layers = 1
            else:
                self._number_of_layers = model_grid.num_layers()
                if self._number_of_layers is None:
                    self._number_of_layers = 1
        else:
            self._number_of_layers = 1
        self._data_type = structure.data_item_structures[0].type
        self._data_storage = self._new_storage(self._number_of_layers != 1)
        if self.structure.type == DatumType.integer:
            multiplier = [1]
        else:
            multiplier = [1.0]
        if data is not None:
            self._get_storage_obj().set_data(data, key=self._current_key,
                                             multiplier=multiplier)

    def __setattr__(self, name, value):
        if name == 'fname':
            self._get_storage_obj().layer_storage[0].fname = value
        elif name == 'factor':
            self._get_storage_obj().layer_storage[0].factor = value
        elif name == 'iprn':
            self._get_storage_obj().layer_storage[0].iprn = value
        elif name == 'binary':
            self._get_storage_obj().layer_storage[0].binary = value
        else:
            super(MFArray, self).__setattr__(name, value)

    def __getitem__(self, k):
        storage = self._get_storage_obj()
        if storage.layered and isinstance(k, int):
            # for layered data treat k as a layer number
            return storage.layer_storage[k]
        else:
            # for non-layered data treat k as an array/list index of the data
            if isinstance(k, int):
                if len(self.get_data(apply_mult=True).shape) == 1:
                    return self.get_data(apply_mult=True)[k]
                elif self.get_data(apply_mult=True).shape[0] == 1:
                    return self.get_data(apply_mult=True)[0, k]
                elif self.get_data(apply_mult=True).shape[1] == 1:
                    return self.get_data(apply_mult=True)[k, 0]
                else:
                    raise Exception(
                        "mfdataarray.__getitem__() error: an integer was " +
                        "passed, self.shape > 1 in both dimensions")
            else:
                if isinstance(k, tuple):
                    if len(k) == 3:
                        return self.get_data(apply_mult=True)[k[0], k[1], k[2]]
                    elif len(k) == 2:
                        return self.get_data(apply_mult=True)[k[0], k[1]]
                    if len(k) == 1:
                        return self.get_data(apply_mult=True)[k]
                else:
                    return self.get_data(apply_mult=True)[(k,)]

    def __setitem__(self, k, value):
        storage = self._get_storage_obj()
        if storage.layered:
            # for layered data treat k as a layer number
            storage.layer_storage[k].set_data(value)
        else:
            # for non-layered data treat k as an array/list index of the data
            a = self.get_data()
            a[k] = value
            a = a.astype(self.get_data().dtype)
            layer_storage = storage.layer_storage[0]
            self._get_storage_obj().set_data(a, key=self._current_key,
                                             multiplier=layer_storage.factor)

    def new_simulation(self, sim_data):
        super(MFArray, self).new_simulation(sim_data)
        self._data_storage = self._new_storage(False)
        self._number_of_layers = 1

    def supports_layered(self):
        model_grid = self._data_dimensions.get_model_grid()
        return self.structure.layered and \
               model_grid.grid_type() != DiscretizationType.DISU

    def set_layered_data(self, layered_data):
        if layered_data is True and self.structure.layered is False:
            if self._data_dimensions.get_model_grid().grid_type() == \
                    DiscretizationType.DISU:
                except_str = 'Layered option not available for unstructured ' \
                             'grid. {}'.format(self._path)
            else:
                except_str = 'Data "{}" does not support layered option. ' \
                             '{}'.format(self._data_name, self._path)
            print(except_str)
            raise mfstructure.MFDataException(except_str)
        self._get_storage_obj().layered = layered_data

    def make_layered(self):
        if self.supports_layered():
            self._get_storage_obj().make_layered()
        else:
            if self._data_dimensions.get_model_grid().grid_type() == \
                    DiscretizationType.DISU:
                except_str = 'Layered option not available for unstructured ' \
                             'grid. {}'.format(self._path)
            else:
                except_str = 'Data "{}" does not support layered option. ' \
                             '{}'.format(self._data_name, self._path)
            print(except_str)
            raise mfstructure.MFDataException(except_str)

    def store_as_external_file(self, external_file_path, multiplier=[1.0],
                               layer_num=None):
        storage = self._get_storage_obj()
        if storage is None:
            self._data_storage = self._new_storage(False)
        ds_index = self._resolve_layer_index(layer_num)

        # move data to file
        if storage.layer_storage[ds_index[0]].data_storage_type == \
                mfdata.DataStorageType.external_file:
            storage.external_to_external(external_file_path, multiplier,
                                         layer_num)
        else:
            storage.internal_to_external(external_file_path, multiplier,
                                         layer_num)

        # update data storage
        self._get_storage_obj().layer_storage[ds_index[0]].data_storage_type \
                = mfdata.DataStorageType.external_file
        self._get_storage_obj().layer_storage[ds_index[0]].fname = \
                external_file_path
        if multiplier is not None:
            self._get_storage_obj().layer_storage[ds_index[0]].multiplier = \
                    multiplier[0]

    def has_data(self, layer_num=None):
        if self._get_storage_obj() is None:
            return False
        return self._get_storage_obj().has_data(layer_num)

    def get_data(self, layer_num=None, apply_mult=False):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        return self._get_storage_obj().get_data(layer_num, apply_mult)

    def set_data(self, data, multiplier=[1.0], layer_num=None):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        self._get_storage_obj().set_data(data, layer_num, multiplier,
                                         key=self._current_key)
        self._number_of_layers = self._get_storage_obj().num_layers

    def load(self, first_line, file_handle, block_header,
             pre_data_comments=None):
        super(MFArray, self).load(first_line, file_handle, block_header,
                                  pre_data_comments=None)

        if self.structure.layered and self._number_of_layers != \
                self._data_dimensions.get_model_grid().num_layers():
            model_grid = self._data_dimensions.get_model_grid()
            if model_grid.grid_type() == DiscretizationType.DISU:
                self._number_of_layers = 1
            else:
                self._number_of_layers = model_grid.num_layers()
                if self._number_of_layers is None:
                    self._number_of_layers = 1
            self._data_storage = self._new_storage(self._number_of_layers != 1)
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
                    storage.set_data(' '.join(arr_line[1:3]), 0,
                                     key=self._current_key)
                    return [False, None]
                else:
                    except_str = 'ERROR: "timearrayseries" keyword not ' \
                                 'followed by a valid TAS variable. ' \
                                 '{}'.format(self._path)
                    print(except_str)
                    raise mfstructure.MFFileParseException(except_str)

        if not self.structure.data_item_structures[0].just_data:
            # verify keyword
            index_num, aux_var_index = self._load_keyword(arr_line, 0)
        else:
            index_num = 0
            aux_var_index = None

        # if layered supported, look for layered flag
        if self.structure.layered or aux_var_index is not None:
            if (len(arr_line) > index_num and
                    arr_line[index_num].lower() == 'layered'):
                storage.layered = True
                layers = self._data_dimensions.get_model_grid().num_layers()
            elif aux_var_index is not None:
                # each layer stores a different aux variable
                layers = len(package_dim.get_aux_variables()[0]) - 1
                self._number_of_layers = layers
                storage.layered = True
                while storage.num_layers < layers:
                    storage.add_layer()
            else:
                layers = 1
                storage.flatten()

        else:
            layers = 1
        total_size = \
                self._data_dimensions.model_subspace_size(self.structure.shape)
        if aux_var_index is not None:
            layer_size = total_size
        else:
            layer_size = int(total_size / layers)

        if aux_var_index is None:
            # loop through the number of layers
            for layer in range(0, layers):
                self._load_layer(layer, layer_size, storage, arr_line,
                                 file_handle)
        else:
            # write the aux var to it's unique index
            self._load_layer(aux_var_index, layer_size, storage, arr_line,
                             file_handle)
        return [False, None]

    def _load_layer(self, layer, layer_size, storage, arr_line, file_handle):
        if not self.structure.data_item_structures[0].just_data or layer > 0:
            arr_line = \
                    mfdatautil.ArrayUtil.\
                        split_data_line(file_handle.readline())
        layer_storage = storage.layer_storage[layer]
        # if constant
        if arr_line[0].upper() == 'CONSTANT':
            if len(arr_line) < 2:
                except_str = 'ERROR: MFArray "{}" contains a CONSTANT that ' \
                             'is not followed by a number. ' \
                             '{}'.format(self._data_name, self._path)
                print(except_str)
                raise mfstructure.MFFileParseException(except_str)
            # store data
            layer_storage.data_storage_type = \
                    mfdata.DataStorageType.internal_constant
            storage.store_internal([storage.convert_data(arr_line[1],
                                                         self._data_type)],
                                   layer, const=True, multiplier=[1.0])
            # store anything else as a comment
            if len(arr_line) > 2:
                layer_storage.comments = \
                        mfdata.MFComment(' '.join(arr_line[2:]),
                                         self._path, self._simulation_data,
                                         layer)
        # if internal
        elif arr_line[0].upper() == 'INTERNAL':
            if len(arr_line) < 2:
                except_str = 'ERROR: Data array "{}" contains a INTERNAL ' \
                             'that is not followed by a multiplier. ' \
                             '{}'.format(self.structure.name, self._path)
                print(except_str)
                raise mfstructure.MFFileParseException(except_str)
            multiplier, print_format, flags_found = \
                    storage.process_internal_line(arr_line)

            storage.layer_storage[layer].data_storage_type = \
                    mfdata.DataStorageType.internal_array

            # store anything else as a comment
            if len(arr_line) > 5:
                layer_storage.comments = \
                        mfdata.MFComment(' '.join(arr_line[5:]), self._path,
                                         self._simulation_data, layer)

            # load variable data from current file
            data_from_file = storage.read_data_from_file(layer, file_handle,
                                                         multiplier,
                                                         print_format)
            data_shaped = self._resolve_data_shape(data_from_file[0])
            storage.store_internal(data_shaped, layer, const=False,
                                   multiplier=[multiplier],
                                   print_format=print_format)

            # verify correct size
            if layer_size > 0 and layer_size != data_from_file[1]:
                except_str = 'ERROR: Data array "{}" does not contain the ' \
                             'expected amount of INTERNAL data. expected {},' \
                             ' found {}.  {}'.format(self.structure.name,
                                                     layer_size,
                                                     data_from_file[1],
                                                     self._path)
                print(except_str)
                raise mfstructure.MFFileParseException(except_str)
        elif arr_line[0].upper() == 'OPEN/CLOSE':
            storage.process_open_close_line(arr_line, layer)

    def get_file_entry(self, layer=None,
                       ext_file_action=ExtFileAction.copy_relative_paths):
        data_storage = self._get_storage_obj()
        if data_storage is None or data_storage.num_layers == 0 or \
                not data_storage.has_data():
            return ''

        # determine if this is the special aux variable case
        if self.structure.name.lower() == 'aux' and data_storage.layered:
            layered_aux = True
        else:
            layered_aux = False
        # prepare indent
        indent = self._simulation_data.indent_string
        if self._number_of_layers == 1:
            data_indent = indent
        else:
            data_indent = '{}{}'.format(indent,
                                        self._simulation_data.indent_string)

        file_entry_array = []
        if data_storage.data_structure_type == mfdata.DataStructureType.scalar:
            # scalar data, like in the case of a time array series gets written
            # on a single line
            data = data_storage.get_data()
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
                layer_min = 0
                layer_max = self._number_of_layers
            else:
                # set layer range
                if layer >= self._number_of_layers - 1:
                    except_str = 'Layer {} for variable "{}" does not exist.' \
                                 ' {}'.format(layer, self._data_name,
                                              self._path)
                    print(except_str)
                    raise mfstructure.MFDataException(except_str)

                layer_min = layer
                layer_max = layer + 1

            for layer in range(layer_min, layer_max):
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

    def _new_storage(self, set_layers=True):
        if set_layers:
            return mfdata.DataStorage(self._simulation_data,
                                      self._data_dimensions,
                                      mfdata.DataStorageType.internal_array,
                                      mfdata.DataStructureType.ndarray,
                                      self._number_of_layers)
        else:
            return mfdata.DataStorage(self._simulation_data,
                                      self._data_dimensions,
                                      mfdata.DataStorageType.internal_array,
                                      mfdata.DataStructureType.ndarray)

    def _get_storage_obj(self):
        return self._data_storage

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
            # display aux name
            file_entry = '{}{}\n'.format(indent_string,
                                         self._get_aux_var_name(layer))
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
            const_str = self._get_constant_formatting_string(
                    data_storage.get_const_val(layer), layer,
                    self._data_type).upper()
            file_entry = '{}{}{}'.format(file_entry, indent_string,
                                         const_str)
        else:
            #  external data
            ext_str = self._get_external_formatting_string(layer,
                                                           ext_file_action)
            file_entry = '{}{}{}'.format(file_entry, indent_string,
                                         ext_str)
            #  add to active list of external files
            file_path = data_storage.get_external_file_path(layer)
            package_dim = self._data_dimensions.package_dim
            model_name = package_dim.model_dim[0].model_name
            self._simulation_data.mfpath.add_ext_file(file_path, model_name)
        return file_entry

    def _get_data_layer_string(self, layer, data_indent):
        layer_data_string = ['']
        line_data_count = 0
        # iterate through data layer
        data = self._get_storage_obj().get_data(layer, False)
        data_iter = mfdatautil.ArrayUtil.next_item(data)
        indent_str = self._simulation_data.indent_string
        for item, last_item, new_list, nesting_change in data_iter:
            # increment data/layer counts
            line_data_count += 1
            data_lyr = self._get_storage_obj().to_string(item, self._data_type)
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
        data_dim = self._data_dimensions
        dimensions, shape_rule = data_dim.get_data_shape(repeating_key=
                                                         self._current_key)
        if self._get_storage_obj().layered:
            dimensions = dimensions[1:]
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return np.reshape(data, dimensions).tolist()
        else:
            return data

    def _resolve_layer_index(self, layer_num, allow_multiple_layers=False):
        # handle layered vs non-layered data
        storage = self._get_storage_obj()
        if storage.layered:
            if layer_num is None:
                if allow_multiple_layers:
                    layer_index = storage.get_active_layer_indices()
                else:
                    except_str = 'Data "{}" is layered but no ' \
                                 'layer_num was specified. ' \
                                 '{}'.format(self._data_name, self._path)
                    print(except_str)
                    raise mfstructure.MFDataException(except_str)
            else:
                layer_index = [layer_num]
        else:
            layer_index = [0]
        return layer_index

    def _verify_data(self, data_iter, layer_num):
        # TODO: Implement
        size = self._data_dimensions.model_subspace_size(self.structure.shape)

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

    def get_data(self, key=None, apply_mult=True):
        if key is None:
            key = self._current_key
        self.get_data_prep(key)
        return super(MFTransientArray, self).get_data(apply_mult=apply_mult)

    def set_data(self, data, multiplier=[1.0], layer_num=None, key=None):
        if isinstance(data, dict) or isinstance(data, OrderedDict):
            # each item in the dictionary is a list for one stress period
            # the dictionary key is the stress period the list is for
            for key, list_item in data.items():
                self._set_data_prep(list_item, key)
                super(MFTransientArray, self).set_data(list_item, multiplier,
                                                       layer_num)
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
            super(MFTransientArray, self).set_data(data, multiplier, layer_num)

    def get_file_entry(self, key=0,
                       ext_file_action=ExtFileAction.copy_relative_paths):
        self._get_file_entry_prep(key)
        return super(MFTransientArray, self).get_file_entry(ext_file_action=
                                                            ext_file_action)

    def load(self, first_line, file_handle, block_header,
             pre_data_comments=None):
        self._load_prep(first_line, file_handle, block_header,
                        pre_data_comments)
        return super(MFTransientArray, self).load(first_line, file_handle,
                                                  pre_data_comments)

    def _new_storage(self, set_layers=True):
        return OrderedDict()

    def _get_storage_obj(self):
        if self._current_key is None or \
                self._current_key not in self._data_storage:
            return None
        return self._data_storage[self._current_key]