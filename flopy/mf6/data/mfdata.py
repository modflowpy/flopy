from operator import itemgetter
from copy import deepcopy
import numpy as np
from shutil import copyfile
from collections import OrderedDict
from enum import Enum
from ..data.mfstructure import MFDataException, MFFileParseException, \
                               MFInvalidTransientBlockHeaderException, \
                               MFDataItemStructure
from ..data.mfdatautil import DatumUtil, FileIter, MultiListIter, ArrayUtil, \
                              ConstIter, ArrayIndexIter
from ..coordinates.modeldimensions import DataDimensions


class MFComment(object):
    """
    Represents a variable in a MF6 input file


    Parameters
    ----------
    comment : string or list
        comment to be displayed in output file
    path : string
        tuple representing location in the output file
    line_number : integer
        line number to display comment in output file

    Attributes
    ----------
    comment : string or list
        comment to be displayed in output file
    path : string
        tuple representing location in the output file
    line_number : integer
        line number to display comment in output file

    Methods
    -------
    write : (file)
        writes the comment to file
    add_text(additional_text)
        adds text to the comment
    get_file_entry(eoln_suffix=True)
        returns the comment text in the format to write to package files
    is_empty(include_whitespace=True)
        checks to see if comment is just an empty string ''.  if
        include_whitespace is set to false a string with only whitespace is
        considered empty
    is_comment(text, include_empty_line=False) : boolean
        returns true if text is a comment.  an empty line is considered a
        comment if include_empty_line is true.

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, comment, path, sim_data, line_number=0):
        assert(isinstance(comment, str) or isinstance(comment, list) or comment
               is None)
        self.text = comment
        self.path = path
        self.line_number = line_number
        self.sim_data = sim_data

    """
    Add text to the comment string.

    Parameters
    ----------
    additional_text: string
        text to add
    """
    def add_text(self, additional_text):
        if additional_text:
            if isinstance(self.text, list):
                self.text.append(additional_text)
            else:
                self.text = '{} {}'.format(self.text, additional_text)

    """
    Get the comment text in the format to write to package files.

    Parameters
    ----------
    eoln_suffix: boolean
        have comment text end with end of line character
    Returns
    -------
    string : comment text
    """
    def get_file_entry(self, eoln_suffix=True):
        file_entry = ''
        if self.text and self.sim_data.comments_on:
            if not isinstance(self.text, str) and isinstance(self.text, list):
                file_entry = self._recursive_get(self.text)
            else:
                if self.text.strip():
                    file_entry = self.text
            if eoln_suffix:
                file_entry = '{}\n'.format(file_entry)
        return file_entry

    def _recursive_get(self, base_list):
        file_entry = ''
        if base_list and self.sim_data.comments_on:
            for item in base_list:
                if not isinstance(item, str) and isinstance(item, list):
                    file_entry = '{}{}'.format(file_entry,
                                               self._recursive_get(item))
                else:
                    file_entry = '{} {}'.format(file_entry, item)
        return file_entry

    """
    Write the comment text to a file.

    Parameters
    ----------
    fd : file
        file to write to
    eoln_suffix: boolean
        have comment text end with end of line character
    """
    def write(self, fd, eoln_suffix=True):
        if self.text and self.sim_data.comments_on:
            if not isinstance(self.text, str) and isinstance(self.text, list):
                self._recursive_write(fd, self.text)
            else:
                if self.text.strip():
                    fd.write(self.text)
            if eoln_suffix:
                fd.write('\n')

    """
    Check for comment text

    Parameters
    ----------
    include_whitespace : boolean
        include whitespace as text
    Returns
    -------
    boolean : True if comment text exists
    """
    def is_empty(self, include_whitespace=True):
        if include_whitespace:
            if self.text():
                return True
            return False
        else:
            if self.text.strip():
                return True
            return False

    """
    Check text to see if it is valid comment text

    Parameters
    ----------
    text : string
        potential comment text
    include_empty_line : boolean
        allow empty line to be valid
    Returns
    -------
    boolean : True if text is valid comment text
    """
    @staticmethod
    def is_comment(text, include_empty_line=False):
        if not text:
            return include_empty_line
        if text and isinstance(text, list):
            # look for comment mark in first item of list
            text_clean = text[0].strip()
        else:
            text_clean = text.strip()
        if include_empty_line and not text_clean:
            return True
        if text_clean and (text_clean[0] == '#' or text_clean[0] == '!' or
                           text_clean[0] == '//'):
            return True
        return False

    # recursively writes a nested list to a file
    def _recursive_write(self, fd, base_list):
        if base_list:
            for item in base_list:
                if not isinstance(item, str) and isinstance(item, list):
                    self._recursive_write(fd, item)
                else:
                    fd.write(' {}'.format(item))


class DataStorageType(Enum):
    """
    Enumeration of different ways that data can be stored
    """
    internal_array = 1
    internal_constant = 2
    external_file = 3
    open_close = 4


class DataStructureType(Enum):
    """
    Enumeration of different data structures used to store data
    """
    ndarray = 1
    recarray = 2
    scalar = 3


class LayerStorage(object):
    def __init__(self, data_storage, lay_num,
                 data_storage_type=DataStorageType.internal_array,
                 data_structure_type=DataStructureType.ndarray):
        self._data_storage_parent = data_storage
        self._lay_num = lay_num
        self._internal_data = None
        self._data_const_value = None
        self._data_structure_type = data_structure_type
        self.data_storage_type = data_storage_type
        self.fname = None
        self.factor = 1.0
        self.iprn = None
        self.binary = False

    def __repr__(self):
        if self.data_storage_type == DataStorageType.internal_constant:
            return 'constant {}'.format(self._get_data_const_val())
        else:
            return repr(self.get_data())

    def __str__(self):
        if self.data_storage_type == DataStorageType.internal_constant:
            return '{}'.format(self._get_data_const_val())
        else:
            return str(self.get_data())

    def __getattribute__(self, name):
        if name == 'array':
            return self._data_storage_parent.get_data(self._lay_num, True)
        return object.__getattribute__(self, name)

    def set_data(self, data):
        self._data_storage_parent.set_data(data, self._lay_num, [self.factor])

    def get_data(self):
        return self._data_storage_parent.get_data(self._lay_num, False)

    def _get_data_const_val(self):
        if isinstance(self._data_const_value, list):
            return self._data_const_value[0]
        else:
            return self._data_const_value


class DataStorage(object):
    """
    Stores and retrieves data.


    Parameters
    ----------
    sim_data : simulation data class
        reference to the simulation data class
    data_dimensions : data dimensions class
        a data dimensions class for the data being stored
    data_storage_type : enum
        how the data will be stored (internally, as a constant, as an external
        file)
    data_structure_type : enum
        what internal type is the data stored in (ndarray, recarray, scalar)
    num_layers : int
        number of layers data has
    layered : boolean
        is the data layered

    Attributes
    ----------
    data_storage_type : list
        list of data storage types, one for each layer
    data_const_value : list
        list of data constants, one for each layer
    external_file_path : list
        list of external file paths, one for each layer
    multiplier : list
        list of multipliers, one for each layer
    print_format : list
        list of print formats, one for each layer
    num_layers : int
        number of data layers
    data_structure_type :
        what internal type is the data stored in (ndarray, recarray, scalar)
    layered : boolean
        is the data layered
    pre_data_comments : string
        any comments before the start of the data
    comments : OrderedDict
        any comments mixed in with the data, dictionary keys are data lines
    post_data_comments : string
        any comments after the end of the data

    Methods
    -------
    override_data_type : (index, data_type)
        overrides the data type used in a recarray at index "index" with data
        type "data_type"
    get_external_file_path(layer)
        gets the path to an external file for layer "layer"
    get_const_val(layer)
        gets the constant value of a given layer.  data storage type for layer
        must be "internal_constant".
    has_data(layer) : boolean
        returns true if data exists for the specified layer, false otherwise
    get_data(layer) : ndarray/recarray/string
        returns the data for the specified layer
    update_item(data, key_index)
        updates the data in a recarray at index "key_index" with data "data".
        data is a list containing all data for a single record in the
        recarray.  .  data structure type must be recarray
    append_data(data)
        appends data "data" to the end of a recarray.  data structure type must
        be recarray
    set_data(data, layer=None, multiplier=[1.0]
        sets the data being stored to "data" for layer "layer", replacing all
        data for that layer.  a multiplier can be specified.
    get_active_layer_indices() : list
        returns the indices of all layers expected to contain data
    store_internal(data, layer=None, const=False, multiplier=[1.0])
        store data "data" at layer "layer" internally
    store_external(file_path, layer=None, multiplier=[1.0], print_format=None,
        data=None, do_not_verify=False) store data "data" at layer "layer"
        externally in file "file_path"
    external_to_external(new_external_file, multiplier=None, layer=None)
        copies existing external data to the new file location and points to
        the new file
    external_to_internal(layer_num=None, store_internal=False) :
      ndarray/recarray
        loads existing external data for layer "layer_num" and returns it.  if
        store_internal is True it also storages the data internally,
        changing the storage type for "layer_num" layer to internal.
    internal_to_external(new_external_file, multiplier=None, layer=None,
                         print_format=None)
        stores existing internal data for layer "layer" to external file
        "new_external_file"
    read_data_from_file(layer, fd=None, multiplier=None) : (ndarray, int)
        reads in data from a given file "fd" as data from layer "layer".
        returns data as an ndarray along with the size of the data
    to_string(val, type, is_cellid=False, possible_cellid=False)
        converts data "val" of type "type" to a string.  is_cellid is True if
        the data type is known to be a cellid and is treated as such.  when
        possible_cellid is True the data is checked to see if it matches the
        shape/dimensions of a cellid before using it as one.
    resolve_data_size(index) : int
        resolves the size of a given data element in a recarray based on the
        names in the existing rec_array.  assumes repeating data element
        names follow the format <data_element_name>_X.  returns the number of
        times the data element repeats.
    convert_data(data, type) : type
        converts data "data" to type "type" and returns the converted data

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, data_dimensions,
                 data_storage_type=DataStorageType.internal_array,
                 data_structure_type=DataStructureType.ndarray, num_layers=1,
                 layered=False):
        self.data_dimensions = data_dimensions
        self._simulation_data = sim_data
        self._data_type_overrides = {}
        self.layer_storage = [LayerStorage(self, x, data_storage_type)
                              for x in range(num_layers)]
        self.num_layers = num_layers
        self.data_structure_type = data_structure_type
        package_dim = self.data_dimensions.package_dim
        self.in_model = self.data_dimensions is not None and \
                        len(package_dim.package_path) > 1 and \
                        package_dim.model_dim[0].model_name.lower() == \
                        package_dim.package_path[0]

        if data_structure_type == DataStructureType.recarray:
            self.build_type_list(resolve_data_shape=False)
            self._data_type = None
        else:
            self._data_type = self.data_dimensions.structure.get_datum_type()
        self.layered = layered

        # initialize comments
        self.pre_data_comments = None
        self.post_data_comments = None
        self.comments = OrderedDict()

    def __repr__(self):
        return self.get_data_str(True)

    def __str__(self):
        return self.get_data_str(False)

    def flatten(self):
        self.layered = False
        storage_type = self.layer_storage[0].data_storage_type
        self.layer_storage = [LayerStorage(self, 0, storage_type)]
        self.num_layers = 1

    def make_layered(self):
        if not self.layered:
            if self.data_structure_type != DataStructureType.ndarray:
                except_str = 'Data structure type "{}" does not support ' \
                             'layered data.'.format(self.data_structure_type)
                print(except_str)
                raise MFDataException(except_str)
            if self.layer_storage[0].data_storage_type == \
                    DataStorageType.external_file:
                except_str = 'Converting external file data into layered ' \
                             'data currently not support.'
                print(except_str)
                raise MFDataException(except_str)

            previous_storage = self.layer_storage[0]
            data = previous_storage.get_data()
            storage_type = previous_storage.data_storage_type
            data_dim = self.get_data_dimensions(None)
            self.layer_storage = [LayerStorage(self, x, storage_type)
                                  for x in range(data_dim[0])]
            if previous_storage.data_storage_type == \
                    DataStorageType.internal_constant:
                for storage in self.layer_storage:
                    storage._data_const_value = \
                        previous_storage._data_const_value
            elif previous_storage.data_storage_type == \
                    DataStorageType.internal_array:
                assert(len(data) == len(self.layer_storage))
                for data_layer, storage in zip(data, self.layer_storage):
                    storage._internal_data = data_layer
                    storage.factor = previous_storage.factor
                    storage.iprn = previous_storage.iprn
            self.layered = True
            self.num_layers = len(self.layer_storage)

    def get_data_str(self, formal):
        layer_strs= []
        data_str = ''
        # Assemble strings for internal array data
        for index, storage in enumerate(self.layer_storage):
            if storage.data_storage_type == DataStorageType.internal_array:
                if storage._internal_data is not None:
                    header = self._get_layer_header_str(index)
                    if formal:
                        data_str = '{}Layer_{}{{{}}}' \
                                   '\n({})\n'.format(data_str, index + 1,
                                                     header, repr(storage))
                    else:
                        data_str = '{}{{{}}}\n({})\n'.format(data_str, header,
                                                             str(storage))
            elif storage.data_storage_type == \
                    DataStorageType.internal_constant:
                if storage._data_const_value is not None:
                    data_str = '{}{{{}}}' \
                               '\n'.format(data_str,
                                           self._get_layer_header_str(index))

        return data_str

    def _get_layer_header_str(self, layer):
        header_list = []
        if self.layer_storage[layer].data_storage_type == \
                DataStorageType.external_file:
            header_list.append('open/close '
                               '{}'.format(self.layer_storage[layer].fname))
        elif self.layer_storage[layer].data_storage_type == \
                DataStorageType.internal_constant:
            header_list.append('constant {}'.format(self.layer_storage[layer]))
        else:
            header_list.append('internal')
        if self.layer_storage[layer].factor != 1.0 and \
                self.layer_storage[layer].factor != 1:
            header_list.append('factor '
                               '{}'.format(self.layer_storage[layer].factor))
        if self.layer_storage[layer].iprn is not None:
            header_list.append('iprn '
                               '{}'.format(self.layer_storage[layer].iprn))
        if len(header_list) > 0:
            return ', '.join(header_list)
        else:
            return ''

    def add_layer(self):
        storage_zero = self.layer_storage[0]
        self.layer_storage.append(LayerStorage(self, len(self.layer_storage),
                                               storage_zero.data_storage_type))
        self.num_layers += 1

    def override_data_type(self, index, data_type):
        self._data_type_overrides[index] = data_type

    def get_external_file_path(self, layer):
        if layer is None:
            return self.layer_storage[0].fname
        else:
            return self.layer_storage[layer].fname

    def get_const_val(self, layer=None):
        if layer is None:
            assert(len(self.layer_storage) >= 1)
            assert(self.layer_storage[0].data_storage_type ==
                   DataStorageType.internal_constant)
            return self.layer_storage[0]._data_const_value
        else:
            assert(len(self.layer_storage) > layer)
            assert(self.layer_storage[layer].data_storage_type ==
                   DataStorageType.internal_constant)
            return self.layer_storage[layer]._data_const_value

    def has_data(self, layer=None):
        ret_val = self._access_data(layer, False)
        return ret_val is not None and ret_val != False

    def get_data(self, layer=None, apply_mult=True):
        return self._access_data(layer, True, apply_mult=apply_mult)

    def _access_data(self, layer, return_data=False, apply_mult=True):
        layer_check = self._resolve_layer(layer)
        if self.layer_storage[layer_check].data_storage_type == \
                DataStorageType.external_file:
            if return_data:
                return self.external_to_internal(layer)
            else:
                return True
        else:
            if (self.layer_storage[layer_check]._internal_data is None and
              self.layer_storage[layer_check].data_storage_type ==
                    DataStorageType.internal_array) or \
              (self.layer_storage[layer_check]._data_const_value is None and
              self.layer_storage[layer_check].data_storage_type ==
                    DataStorageType.internal_constant):
                return None
            if self.data_structure_type == DataStructureType.ndarray and \
               self.layer_storage[layer_check]._data_const_value is None and \
               self.layer_storage[layer_check]._internal_data is None:
                return None
            assert(layer is None or layer < self.num_layers)
            if layer is None:
                if self.data_structure_type == DataStructureType.ndarray or \
                  self.data_structure_type == DataStructureType.scalar:
                    if return_data:
                        data = self._build_full_data(apply_mult)
                        if data is None:
                            if self.layer_storage[0].data_storage_type == \
                                    DataStorageType.internal_constant:
                                return self.layer_storage[0].get_data()[0]
                        else:
                            return data
                    else:
                        if self.data_structure_type == DataStructureType.scalar:
                            return self.layer_storage[0]._internal_data \
                                   is not None
                        check_storage = self.layer_storage[layer_check]
                        return (check_storage._data_const_value is not None and
                                check_storage.data_storage_type ==
                                DataStorageType.internal_constant) or (
                                check_storage._internal_data is not None and
                                check_storage.data_storage_type ==
                                DataStorageType.internal_array)
                else:
                    if self.layer_storage[layer_check].data_storage_type == \
                            DataStorageType.internal_constant:
                        if return_data:
                            # recarray stored as a constant.  currently only
                            # support grid-based constant recarrays.  build
                            # a recarray of all cells
                            data_list = []
                            model_grid = self.data_dimensions.get_model_grid()
                            structure = self.data_dimensions.structure
                            package_dim = self.data_dimensions.package_dim
                            for cellid in model_grid.get_all_model_cells():
                                data_line = (cellid,) + \
                                            (self.layer_storage[0].
                                             _data_const_value,)
                                if len(structure.data_item_structures) > 2:
                                    # append None any expected optional data
                                    for data_item_struct in \
                                            structure.data_item_structures[2:]:
                                        if (data_item_struct.name !=
                                                'boundname' or
                                                package_dim.boundnames()):
                                            data_line = data_line + (None,)
                                data_list.append(data_line)
                            return np.rec.array(data_list,
                                                self._recarray_type_list)
                        else:
                            return self.layer_storage[layer_check
                                   ]._data_const_value is not None
                    else:
                        if return_data:
                            return self.layer_storage[0]._internal_data
                        else:
                            return True
            elif self.layer_storage[layer].data_storage_type == \
                    DataStorageType.internal_array:
                if return_data:
                    return self.layer_storage[layer]._internal_data
                else:
                    return self.layer_storage[layer]._internal_data is not None
            elif self.layer_storage[layer].data_storage_type == \
                    DataStorageType.internal_constant:
                layer_storage = self.layer_storage[layer]
                if return_data:
                    data = self._fill_const_layer(layer)
                    if data is None:
                        if layer_storage.data_storage_type == \
                                DataStructureType.internal_constant:
                            return layer_storage._data_const_value[0]
                    else:
                        return data
                else:
                    return layer_storage._data_const_value is not None
            else:
                if return_data:
                    return self.get_external(layer)
                else:
                    return True

    def append_data(self, data):
        # currently only support appending to recarrays
        assert(self.data_structure_type == DataStructureType.recarray)
        internal_data = self.layer_storage[0]._internal_data
        if internal_data is None:
            if len(data[0]) != len(self._recarray_type_list):
                # rebuild type list using existing data as a guide
                self.build_type_list(data=data)
            self.set_data(np.rec.array(data, self._recarray_type_list))
        else:
            if len(self.layer_storage[0]._internal_data[0]) < len(data[0]):
                # Rebuild recarray to fit larger size
                for index in range(len(internal_data[0]), len(data[0])):
                    self._duplicate_last_item()
                internal_data_list = internal_data.tolist()
                for data_item in data:
                    internal_data_list.append(data_item)
                self._add_placeholders(internal_data_list)
                self.set_data(np.rec.array(internal_data_list,
                                           self._recarray_type_list))
            else:
                if len(self.layer_storage[0]._internal_data[0]) > len(data[0]):
                    # Add placeholders to data
                    self._add_placeholders(data)
                self.set_data(np.hstack(
                    (internal_data, np.rec.array(data,
                                                 self._recarray_type_list))))

    def set_data(self, data, layer=None, multiplier=[1.0], key=None,
                 autofill=False):
        if self.data_structure_type == DataStructureType.recarray or \
          self.data_structure_type == DataStructureType.scalar:
            self._set_list(data, layer, multiplier, key, autofill)
        else:
            data_dim = self.data_dimensions
            struct = data_dim.structure
            if struct.name == 'aux':
                # make a list out of a single item
                if isinstance(data, int) or isinstance(data, float) or \
                        isinstance(data, str):
                    data = [[data]]
                # handle special case of aux variables in an array
                self.layered = True
                aux_var_names = data_dim.package_dim.get_aux_variables()
                if len(data) == len(aux_var_names[0]) - 1:
                    for layer, aux_var_data in enumerate(data):
                        if layer > 0:
                            self.add_layer()
                        self._set_array(aux_var_data, layer, multiplier, key,
                                        autofill)
                else:
                    except_str = 'Unable to set data for aux variable.  ' \
                                 'Expected {} aux variables ' \
                                 'but got {}. {}'.format(len(aux_var_names[0]),
                                                         len(data),
                                                         struct.path)
                    print(except_str)
                    raise MFDataException(except_str)
            else:
                self._set_array(data, layer, multiplier, key, autofill)

    def _set_list(self, data, layer, multiplier, key, autofill):
        if isinstance(data, dict):
            if 'filename' in data:
                self.process_open_close_line(data, layer)
                return
        self.store_internal(data, layer, multiplier, key=key,
                            autofill=autofill)

    def _set_array(self, data, layer, multiplier, key, autofill):
        # make a list out of a single item
        if isinstance(data, int) or isinstance(data, float) or isinstance(data, str):
            data = [data]

        # try to set as a single layer
        if not self._set_array_layer(data, layer, multiplier, key):
            # check for possibility of multi-layered data
            success = False
            if layer is None and self.data_structure_type == \
                    DataStructureType.ndarray and len(data) == self.num_layers:
                self.layered = True
                # loop through list and try to store each list entry as a layer
                success = True
                for layer_num, layer_data in enumerate(data):
                    if not isinstance(layer_data, list) and \
                            not isinstance(layer_data, dict) and \
                            not isinstance(layer_data, np.ndarray):
                        layer_data = [layer_data]
                    success = success and self._set_array_layer(layer_data,
                                                                layer_num,
                                                                multiplier,
                                                                key)
            if not success:
                except_str = 'Unable to set data for variable "{}" layer ' \
                             '{}.  Enable verbose mode for more information.' \
                             '{} '.format(self.data_dimensions.structure.name,
                                          layer,
                                          self.data_dimensions.structure.path)
                print(except_str)
                raise MFDataException(except_str)
        elif layer is None:
            self.layered = False
            self.num_layers = 1

    def _set_array_layer(self, data, layer, multiplier, key):
        # look for a single constant value
        data_type = self.data_dimensions.structure.get_datum_type()
        if len(data) == 1 and self._is_type(data[0], data_type):
            # store data as const
            self.store_internal(data, layer, True, multiplier, key=key)
            return True

        # look for internal and open/close data
        if isinstance(data, dict):
            if isinstance(data['data'], int) or \
                    isinstance(data['data'], float) or \
                    isinstance(data['data'], str):
                # data should always in in a list/array
                data['data'] = [data['data']]

            if 'filename' in data:
                self.process_open_close_line(data, layer)
                return True
            elif 'data' in data:
                multiplier, iprn, flags_found = \
                    self.process_internal_line(data)
                if len(data['data']) == 1:
                    # merge multiplier with single value and make constant
                    if DatumUtil.is_float(multiplier):
                        mult = 1.0
                    else:
                        mult = 1
                    self.store_internal([data['data'][0] * multiplier], layer,
                                        True, [mult], key=key,
                                        print_format=iprn)
                else:
                    self.store_internal(data['data'], layer, False,
                                        [multiplier], key=key,
                                        print_format=iprn)
                return True
        elif isinstance(data[0], str):
            if data[0].lower() == 'internal':
                multiplier, iprn, \
                        flags_found = self.process_internal_line(data)
                self.store_internal(data[-1], layer, False, [multiplier],
                                    key=key, print_format=iprn)
                return True
            elif data[0].lower() != 'open/close':
                # assume open/close is just omitted, though test data file to
                # be sure
                new_data = data[:]
                new_data.insert(0, 'open/close')
            else:
                new_data = data[:]
            multiplier, iprn, binary = self.process_open_close_line(new_data,
                                                                    layer,
                                                                    False)
            model_name = \
                    self.data_dimensions.package_dim.model_dim[0].model_name
            resolved_path = \
                    self._simulation_data.mfpath.resolve_path(new_data[1],
                                                              model_name)
            if self._verify_data(FileIter(resolved_path), layer):
                # store location to file
                self.store_external(new_data[1], layer, [multiplier],
                                    print_format=iprn, binary=binary,
                                    do_not_verify=True)
                return True
        # try to resolve as internal array
        layer_storage = self.layer_storage[self._resolve_layer(layer)]
        if not (layer_storage.data_storage_type ==
                DataStorageType.internal_constant and
                ArrayUtil.has_one_item(data)) and \
                self._verify_data(MultiListIter(data), layer):
            # store data as is
            self.store_internal(data, layer, False, multiplier, key=key)
            return True
        return False

    def get_active_layer_indices(self):
        layer_index = []
        for index in range(0, self.num_layers):
            if self.layer_storage[index].fname is not None or \
                    self.layer_storage[index]._internal_data is not None:
                layer_index.append(index)
        return layer_index

    def get_external(self, layer=None):
        assert(layer is None or layer < self.num_layers)

    def store_internal(self, data, layer=None, const=False, multiplier=[1.0],
                       key=None, autofill=False,
                       print_format=None):
        if self.data_structure_type == DataStructureType.recarray:
            if self.layer_storage[0].data_storage_type == \
                    DataStorageType.internal_constant:
                self.layer_storage[0]._data_const_value = data
            else:
                self.layer_storage[0].data_storage_type = \
                        DataStorageType.internal_array
                if isinstance(data, np.recarray):
                    self.layer_storage[0]._internal_data = data
                else:
                    if autofill and data is not None:
                        if not isinstance(data, list):
                            # put data in a list format for recarray
                            data = [(data,)]
                        # auto-fill tagged keyword
                        structure = self.data_dimensions.structure
                        data_item_structs = structure.data_item_structures
                        if data_item_structs[0].tagged and not \
                                data_item_structs[0].type == 'keyword':
                            for data_index, data_entry in enumerate(data):
                                if (data_item_structs[0].type == 'string' and
                                        data_entry[0].lower() ==
                                        data_item_structs[0].name.lower()):
                                    break
                                data[data_index] = \
                                        (data_item_structs[0].name.lower(),) \
                                        + data[data_index]
                    self.build_type_list(data=data, key=key)
                    if autofill and data is not None:
                        # resolve any fields with data types that do not agree
                        # with the expected type list
                        self._resolve_multitype_fields(data)
                    if isinstance(data, list):
                        # data needs to be stored as tuples within a list.
                        # if this is not the case try to fix it
                        self._tupleize_data(data)
                        # add placeholders to data so it agrees with expected
                        # dimensions of recarray
                        self._add_placeholders(data)
                    self.set_data(np.rec.array(data, self._recarray_type_list))
        elif self.data_structure_type == DataStructureType.scalar:
            self.layer_storage[0]._internal_data = data
        else:
            layer, multiplier = self._store_prep(layer, multiplier)
            dimensions = self.get_data_dimensions(layer)
            if const:
                self.layer_storage[layer].data_storage_type = \
                        DataStorageType.internal_constant
                self.layer_storage[layer]._data_const_value = data
            else:
                self.layer_storage[layer].data_storage_type = \
                        DataStorageType.internal_array
                self.layer_storage[layer]._internal_data = \
                        np.reshape(data, dimensions)
            self.layer_storage[layer].factor = multiplier
            self.layer_storage[layer].iprn = print_format

    def _resolve_multitype_fields(self, data):
        # find any data fields where the data is not a consistent type
        itype_len = len(self._recarray_type_list)
        for data_entry in data:
            for index, data_val in enumerate(data_entry):
                if index < itype_len and \
                        self._recarray_type_list[index][1] != object and \
                        type(data_val) != self._recarray_type_list[index][1] \
                        and (type(data_val) != int or
                        self._recarray_type_list[index][1] != float):
                    # for inconsistent types use generic object type
                    self._recarray_type_list[index] = \
                            (self._recarray_type_list[index][0], object)

    def store_external(self, file_path, layer=None, multiplier=[1.0],
                       print_format=None, data=None, do_not_verify=False,
                       binary=False):
        layer, multiplier = self._store_prep(layer, multiplier)
        self.layer_storage[layer].fname = file_path
        self.layer_storage[layer].iprn = print_format
        self.layer_storage[layer].binary = binary
        self.layer_storage[layer].data_storage_type = \
                DataStorageType.external_file
        if self.data_structure_type == DataStructureType.recarray:
            self.layer_storage[0]._internal_data = None
        else:
            self.layer_storage[layer].factor = multiplier
            self.layer_storage[layer]._internal_data = None
            if data is not None:
                data_size = self._get_data_size(layer)
                current_size = 0
                data_dim = self.data_dimensions
                data_type = data_dim.structure.data_item_structures[0].type
                model_name = data_dim.package_dim.model_dim[0].model_name
                fd = open(self._simulation_data.mfpath.resolve_path(file_path,
                                                                    model_name)
                          , 'w')
                for data_item in MultiListIter(data, True):
                    if data_item[2] and current_size > 0:
                        # new list/dimension, add appropriate formatting to
                        # the file
                        fd.write('\n')
                    fd.write('{} '.format(self.to_string(data_item[0],
                                                         data_type)))
                    current_size += 1
                if current_size != data_size:
                    except_str = 'Not enough data for "{}" provided for file' \
                                 ' {}.  Expected {} but only received ' \
                                 '{}.'.format(data_dim.structure.path, fd.name,
                                              data_size, current_size)
                    print(except_str)
                    fd.close()
                    raise MFDataException(except_str)
                fd.close()
            elif self._simulation_data.verify_external_data and \
                    do_not_verify is None:
                self.external_to_internal(layer)

    def external_to_external(self, new_external_file, multiplier=None,
                             layer=None):
        # currently only support files containing ndarrays
        assert(self.data_structure_type == DataStructureType.ndarray)
        assert((layer is None and self.num_layers == 1) or
               (layer is not None and layer < self.num_layers))
        # get data storage
        if layer is None:
            layer = 1
        assert(self.layer_storage[layer].fname is not None)
        # copy file to new location
        copyfile(self.layer_storage[layer].fname, new_external_file)

        # update
        self.store_external(new_external_file, layer,
                            [self.layer_storage[layer].factor],
                            self.layer_storage[layer].iprn,
                            binary=self.layer_storage[layer].binary)

    def external_to_internal(self, layer_num=None, store_internal=False):
        # currently only support files containing ndarrays
        assert(self.data_structure_type == DataStructureType.ndarray)

        if layer_num is None:
            data_out = self._build_full_data(store_internal)
        else:
            # load data from external file
            data_out, current_size = self.read_data_from_file(layer_num)
            if self.layer_storage[layer_num].factor is not None:
                data_out = data_out * self.layer_storage[layer_num].factor

        if store_internal:
            self.store_internal(data_out, layer_num)
        return data_out

    def internal_to_external(self, new_external_file, multiplier=None,
                             layer=None, print_format=None):
        if layer is None:
            self.store_external(new_external_file, layer, multiplier,
                                print_format,
                                self.layer_storage[0]._internal_data)
        else:
            self.store_external(new_external_file, layer, multiplier,
                                print_format,
                                self.layer_storage[layer]._internal_data)

    def read_data_from_file(self, layer, fd=None, multiplier=None,
                            print_format=None):
        if multiplier is not None:
            self.layer_storage[layer].factor = multiplier
        if print_format is not None:
            self.layer_storage[layer].iprn = print_format
        data_size = self._get_data_size(layer)
        # load variable data from file
        current_size = 0
        data_out = []
        if layer is None:
            layer = 0
        close_file = False
        if fd is None:
            close_file = True
            model_dim = self.data_dimensions.package_dim.model_dim[0]
            fd = open(self._simulation_data.mfpath.resolve_path(
                    self.layer_storage[layer].fname, model_dim.model_name),
                    'r')
        line = ' '
        while line != '':
            line = fd.readline()
            arr_line = ArrayUtil.split_data_line(line, True)
            for data in arr_line:
                if data != '':
                    if current_size == data_size:
                        path = self.data_dimensions.structure.path
                        print('WARNING: More data found than expected in file'
                              ' {} for data '
                              '"{}".'.format(fd.name,
                                             path))
                        break
                    data_out.append(self.convert_data(data, self._data_type))
                    current_size += 1
            if current_size == data_size:
                break
        if current_size != data_size:
            except_str = 'Not enough data in file {} for data "{}".  ' \
                         'Expected {} but only found ' \
                         '{}.'.format(fd.name,
                                      self.data_dimensions.structure.path,
                                      data_size, current_size)
            print(except_str)
            if close_file:
                fd.close()
            raise MFDataException(except_str)

        if close_file:
            fd.close()

        dimensions = self.get_data_dimensions(layer)
        data_out = np.reshape(data_out, dimensions)
        return data_out, current_size

    def to_string(self, val, type, is_cellid=False, possible_cellid=False,
                  force_upper_case=False):
        while isinstance(val, list) or isinstance(val, np.ndarray):
            # extract item from list
            assert(len(val) == 1)
            val = val[0]

        if is_cellid or (possible_cellid and isinstance(val, tuple)):
            if len(val) > 0 and val[0] == 'none':
                # handle case that cellid is 'none'
                return val[0]
            string_val = []
            for item in val:
                string_val.append(str(item+1))
            return ' '.join(string_val)
        elif type == 'float':
            upper = self._simulation_data.scientific_notation_upper_threshold
            lower = self._simulation_data.scientific_notation_lower_threshold
            if (abs(val) > upper or abs(val) < lower) and val != 0:
                format_str = '{:.%dE}' % self._simulation_data.float_precision
            else:
                format_str = '{:%d' \
                             '.%df}' % (self._simulation_data.float_characters,
                                        self._simulation_data.float_precision)

            return format_str.format(val)
        elif type == 'int' or type == 'integer':
            return str(val)
        elif type == 'string':
            arr_val = val.split()
            if len(arr_val) > 1:
                # quote any string with spaces
                string_val = "'{}'".format(val)
                if force_upper_case:
                    return string_val.upper()
                else:
                    return string_val
        if force_upper_case:
            return str(val).upper()
        else:
            return str(val)

    def process_internal_line(self, arr_line):
        internal_modifiers_found = False
        if self._data_type == 'integer':
            multiplier = 1
        else:
            multiplier = 1.0
        print_format = None
        if isinstance(arr_line, list):
            if len(arr_line) < 2:
                except_str = 'ERROR: Data array "{}" contains a INTERNAL ' \
                             'that is not followed by a multiplier. ' \
                             '{}'.format(self.data_dimensions.structure.name,
                                         self.data_dimensions.structure.path)
                print(except_str)
                raise MFFileParseException(except_str)

            index = 1
            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    if arr_line[index].lower() == 'factor' and \
                            index + 1 < len(arr_line):
                        multiplier = self.convert_data(arr_line[index+1],
                                                       self._data_type)
                        internal_modifiers_found = True
                        index += 2
                    elif arr_line[index].lower() == 'iprn' and \
                            index + 1 < len(arr_line):
                        print_format = arr_line[index+1]
                        index += 2
                        internal_modifiers_found = True
                    else:
                        break
                else:
                    break
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == 'factor':
                    multiplier = self.convert_data(value, self._data_type)
                    internal_modifiers_found = True
                if key.lower() == 'iprn':
                    print_format = value
                    internal_modifiers_found = True
        return multiplier, print_format, internal_modifiers_found

    def process_open_close_line(self, arr_line, layer, store=True):
        # process open/close line
        index = 2
        if self._data_type == 'integer':
            multiplier = 1
        else:
            multiplier = 1.0
        print_format = None
        binary = False
        data_file = None

        data_dim = self.data_dimensions
        if isinstance(arr_line, list):
            if len(arr_line) < 2 and store:
                except_str = 'ERROR: Data array "{}" contains a OPEN/CLOSE ' \
                             'that is not followed by a file. ' \
                             '{}'.format(data_dim.structure.name,
                                         data_dim.structure.path)
                print(except_str)
                raise MFFileParseException(except_str)

            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    if arr_line[index].lower() == 'factor' and \
                            index + 1 < len(arr_line):
                        multiplier = self.convert_data(arr_line[index+1],
                                                       self._data_type)
                        index += 2
                    elif arr_line[index].lower() == 'iprn' and \
                            index + 1 < len(arr_line):
                        print_format = arr_line[index+1]
                        index += 2
                    elif arr_line[index].lower() == 'binary':
                        binary = True
                        index += 1
                    else:
                        break
                else:
                    break
                # save comments
            if index < len(arr_line):
                self.layer_storage[layer].comments = MFComment(
                        ' '.join(arr_line[index:]),
                        self.data_dimensions.structure.path,
                        self._simulation_data, layer)
            if arr_line[0].lower() == 'open/close':
                data_file = arr_line[1]
            else:
                data_file = arr_line[0]
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == 'factor':
                    multiplier = self.convert_data(value, self._data_type)
                if key.lower() == 'iprn':
                    print_format = value
                if key.lower() == 'binary':
                    binary = bool(value)
            if 'filename' in arr_line:
                data_file = arr_line['filename']

        if store:
            if data_file is None:
                except_str = 'Unable to set data for variable "{}".  ' \
                             'No data file specified. ' \
                             '{}'.format(data_dim.structure.name,
                                         data_dim.structure.path)
                print(except_str)
                raise MFDataException(except_str)
            # store external info
            self.store_external(data_file, layer, [multiplier], print_format,
                                binary=binary)

            #  add to active list of external files
            model_name = data_dim.package_dim.model_dim[0].model_name
            self._simulation_data.mfpath.add_ext_file(data_file, model_name)

        return multiplier, print_format, binary

    def _tupleize_data(self, data):
        for index, data_line in enumerate(data):
            if type(data_line) != tuple:
                if type(data_line) == list:
                    data[index] = tuple(data_line)
                else:
                    data[index] = (data_line,)

    def _add_placeholders(self, data):
        idx = 0
        for data_line in data:
            data_line_len = len(data_line)
            if data_line_len < len(self._recarray_type_list):
                for index in range(data_line_len,
                                   len(self._recarray_type_list)):
                    if self._recarray_type_list[index][1] == int:
                        self._recarray_type_list[index] = \
                                (self._recarray_type_list[index][0], object)
                        data_line += (None,)
                    elif self._recarray_type_list[index][1] == float:
                        data_line += (np.nan,)
                    else:
                        data_line += (None,)
                data[idx] = data_line
            idx += 1

    def _duplicate_last_item(self):
        last_item = self._recarray_type_list[-1]
        arr_item_name = last_item[0].split('_')
        if DatumUtil.is_int(arr_item_name[-1]):
            new_item_num = int(arr_item_name[-1]) + 1
            new_item_name = '_'.join(arr_item_name[0:-1])
            new_item_name = '{}_{}'.format(new_item_name, new_item_num)
        else:
            new_item_name = '{}_1'.format(last_item[0])
        self._recarray_type_list.append((new_item_name, last_item[1]))

    def _build_full_data(self, apply_multiplier=False):
        if self.data_structure_type == DataStructureType.scalar:
            return self.layer_storage[0]._internal_data
        dimensions = self.get_data_dimensions(None)
        if dimensions[0] < 0:
            return None
        full_data = np.full(dimensions, np.nan,
                            self.data_dimensions.structure.get_datum_type(True)
                            )

        if not self.layered:
            layers_to_process = 1
        else:
            layers_to_process = self.num_layers
        for layer in range(0, layers_to_process):
            if self.layer_storage[layer].factor is not None and \
                    apply_multiplier:
                mult = self.layer_storage[layer].factor
            elif self._data_type == 'integer':
                mult = 1
            else:
                mult = 1.0

            if self.layer_storage[layer].data_storage_type == \
                    DataStorageType.internal_array:
                if len(self.layer_storage[layer]._internal_data) > layer and \
                       self.layer_storage[layer]._internal_data[layer] is None:
                    return None
                if self.num_layers == 1 or not self.layered:
                    full_data = self.layer_storage[layer]._internal_data * mult
                else:
                    full_data[layer] = \
                            self.layer_storage[layer]._internal_data * mult
            elif self.layer_storage[layer].data_storage_type == \
                    DataStorageType.internal_constant:
                if self.num_layers == 1 or not self.layered:
                    full_data = self._fill_const_layer(layer) * mult
                else:
                    full_data[layer] = self._fill_const_layer(layer) * mult
            else:
                if self.num_layers == 1 or not self.layered:
                    full_data = self.read_data_from_file(layer)[0] * mult
                else:
                    full_data[layer] = self.read_data_from_file(layer)[0]*mult
        return full_data

    def _resolve_layer(self, layer):
        if layer is None:
            return 0
        else:
            return layer

    def _verify_data(self, data_iter, layer):
        # get expected size
        data_dimensions = self.get_data_dimensions(layer)
        # get expected data types
        if self.data_dimensions.structure.type == 'recarray' or \
                self.data_dimensions.structure.type == 'record':
            data_types = self.data_dimensions.structure.get_data_item_types()
            # check to see if data contains the correct types and is a possibly
            # correct size
            record_loc = 0
            actual_data_size = 0
            rows_of_data = 0
            for data_item in data_iter:
                if self._is_type(data_item, data_types[record_loc]):
                    actual_data_size += 1
                if record_loc == len(data_types) - 1:
                    record_loc = 0
                    rows_of_data += 1
                else:
                    record_loc += 1
            return rows_of_data > 0 and (rows_of_data < data_dimensions[0] or
                                         data_dimensions[0] == -1)
        else:
            expected_data_size = 1
            for dimension in data_dimensions:
                expected_data_size = expected_data_size * dimension
            data_type = self.data_dimensions.structure.get_datum_type()
            # check to see if data can fit dimensions
            actual_data_size = 0
            for data_item in data_iter:
                if self._is_type(data_item, data_type):
                    actual_data_size += 1
                if actual_data_size >= expected_data_size:
                    return True
            return False

    def _fill_const_layer(self, layer):
        data_dimensions = self.get_data_dimensions(layer)
        if data_dimensions[0] < 0:
            return self.layer_storage[layer]._data_const_value
        else:
            data_iter = ConstIter(self.layer_storage[layer]._data_const_value)
            return self._fill_dimensions(data_iter, data_dimensions)

    def _is_type(self, data_item, data_type):
        if data_type == 'string' or data_type == 'keyword':
            return True
        elif data_type == 'integer':
            return DatumUtil.is_int(data_item)
        elif data_type == 'float' or data_type == 'double':
            return DatumUtil.is_float(data_item)
        elif data_type == 'keystring':
            # TODO: support keystring type
            print('Keystring type currently not supported.')
            return True
        else:
            print('{} type checking currently not supported'.format(data_type))
            return True

    def _fill_dimensions(self, data_iter, dimensions):
        if self.data_structure_type == DataStructureType.ndarray:
            # initialize array
            data_array = np.ndarray(shape=dimensions, dtype=float)
            # fill array
            for index in ArrayIndexIter(dimensions):
                data_array.itemset(index, data_iter.__next__()[0])
            return data_array
        elif self.data_structure_type == DataStructureType.scalar:
            return data_iter.__next__()
        else:
            data_array = None
            data_line = ()
            # fill array
            array_index_iter = ArrayIndexIter(dimensions)
            current_col = 0
            for index in array_index_iter:
                data_line += (index,)
                if current_col == dimensions[1] - 1:
                    if data_array is None:
                        data_array = np.rec.array(data_line,
                                                  self._recarray_type_list)
                    else:
                        rec_array = np.rec.array(data_line,
                                                 self._recarray_type_list)
                        data_array = np.hstack((data_array,
                                                rec_array))
                    current_col = 0
                    data_line = ()
                data_array[index] = data_iter.next()
            return data_array

    def resolve_data_size(self, index):
        # Resolves the size of a given data element based on the names in the
        # existing rec_array. Assumes repeating data element names follow the
        #  format <data_element_name>_X
        assert(self.data_structure_type == DataStructureType.recarray)
        if len(self.layer_storage[0]._internal_data[0]) <= index:
            return 0
        label = self.layer_storage[0]._internal_data.dtype.names[index]
        label_list = label.split('_')
        if len(label_list) == 1:
            return 1
        internal_data = self.layer_storage[0]._internal_data
        for forward_index in range(index+1, len(internal_data.dtype.names)):
            forward_label = internal_data.dtype.names[forward_index]
            forward_label_list = forward_label.split('_')
            if forward_label_list[0] != label_list[0]:
                return forward_index - index
        return len(internal_data.dtype.names) - index

    def build_type_list(self, data_set=None, record_with_record=False,
                        data=None, resolve_data_shape=True, key=None,
                        nseg=None):
        if data_set is None:
            self._recarray_type_list = []
            data_set = self.data_dimensions.structure
        initial_keyword = True
        package_dim = self.data_dimensions.package_dim
        for data_item, index in zip(data_set.data_item_structures,
                                    range(0,
                                          len(data_set.data_item_structures))):
            # handle optional mnames
            if not data_item.optional or len(data_item.name) < 5 or \
                    data_item.name.lower()[0:5] != 'mname' \
                    or not self.in_model:
                overrides = self._data_type_overrides
                if len(self._recarray_type_list) in overrides:
                    data_type = overrides[len(self._recarray_type_list)]
                elif isinstance(data_item, MFDataItemStructure):
                    data_type = data_item.get_rec_type()
                else:
                    data_type = None
                if data_item.name.lower() == 'aux' and resolve_data_shape:
                    aux_var_names = package_dim.get_aux_variables()
                    if aux_var_names is not None:
                        for aux_var_name in aux_var_names[0]:
                            if aux_var_name.lower() != 'auxiliary':
                                self._recarray_type_list.append((aux_var_name,
                                                                 data_type))

                elif data_item.type == 'record':
                    # record within a record, recurse
                    self.build_type_list(data_item, True, data)
                elif data_item.type == 'keystring':
                    self._recarray_type_list.append((data_item.name,
                                                     data_type))
                    # add potential data after keystring to type list
                    ks_data_item = deepcopy(data_item)
                    ks_data_item.type = 'string'
                    ks_data_item.name = '{}_data'.format(ks_data_item.name)
                    ks_rec_type = ks_data_item.get_rec_type()
                    self._recarray_type_list.append((ks_data_item.name,
                                                     ks_rec_type))
                    if index == len(data_set.data_item_structures) - 1:
                        idx = 1
                        data_line_max_size = self._get_max_data_line_size(data)
                        while data is not None and \
                                len(self._recarray_type_list) < \
                                data_line_max_size:
                            # keystrings at the end of a line can contain items
                            # of variable length. assume everything at the
                            # end of the data line is related to the last
                            # keystring
                            self._recarray_type_list.append(
                                    ('{}_{}'.format(ks_data_item.name, idx),
                                                    ks_rec_type))
                            idx += 1

                elif data_item.name != 'boundname' or \
                        self.data_dimensions.package_dim.boundnames():
                    # don't include initial keywords
                    if data_item.type != 'keyword' or initial_keyword == False:
                        initial_keyword = False
                        shape_rule = None
                        if data_item.tagged:
                            if data_item.type != 'string' and \
                                    data_item.type != 'keyword':
                                self._recarray_type_list.append(
                                        ('{}_label'.format(data_item.name),
                                                           object))
                        if nseg is not None and len(data_item.shape) > 0 and \
                                isinstance(data_item.shape[0], str) and \
                                data_item.shape[0][0:4] == 'nseg':
                            # nseg explicitly specified.  resolve any formula
                            # nseg is in
                            model_dim = \
                                    self.data_dimensions.get_model_dim(None)
                            expression_array = \
                                    model_dim.build_shape_expression(data_item.
                                                                     shape)
                            if isinstance(expression_array, list) and \
                                    len(expression_array) == 1:
                                exp = expression_array[0]
                                resolved_shape = \
                                        [model_dim.resolve_exp(exp, nseg)]
                            else:
                                resolved_shape = [1]
                        else:
                            if resolve_data_shape:
                                data_dim = self.data_dimensions
                                resolved_shape, shape_rule = \
                                        data_dim.get_data_shape(data_item,
                                                                data_set,
                                                                data, key)
                            else:
                                resolved_shape = [1]
                        if not resolved_shape or len(resolved_shape) == 0 or \
                                resolved_shape[0] == -1:
                            # could not resolve shape
                            resolved_shape = [1]
                        elif resolved_shape[0] == -9999 or \
                                shape_rule is not None:
                            if data is not None:
                                # shape is an indeterminate 1-d array and
                                # should consume the remainder of the data
                                max_s = ArrayUtil.max_multi_dim_list_size(data)
                                resolved_shape[0] = \
                                    max_s - len(self._recarray_type_list)
                            else:
                                # shape is indeterminate 1-d array and no data
                                # provided to resolve
                                resolved_shape[0] = 1
                        if data_item.is_cellid:
                            if data_item.shape is not None and \
                                    len(data_item.shape) > 0 and \
                              data_item.shape[0] == 'ncelldim':
                                # A cellid is a single entry (tuple) in the
                                # recarray.  Adjust dimensions accordingly.
                                data_dim = self.data_dimensions
                                model_grid = data_dim.get_model_grid()
                                size = model_grid.get_num_spatial_coordinates()
                                data_item.remove_cellid(resolved_shape,
                                                        size)
                        for index in range(0, resolved_shape[0]):
                            if resolved_shape[0] > 1:
                                # type list fields must have unique names
                                self._recarray_type_list.append(
                                        ('{}_{}'.format(data_item.name,
                                                        index), data_type))
                            else:
                                self._recarray_type_list.append(
                                        (data_item.name, data_type))
        return self._recarray_type_list

    @staticmethod
    def _get_max_data_line_size(data):
        max_size = 0
        if data is not None:
            for index in range(0, len(data)):
                if len(data[index]) > max_size:
                    max_size = len(data[index])
        return max_size

    def get_data_dimensions(self, layer):
        data_dimensions, shape_rule = self.data_dimensions.get_data_shape()
        if layer is not None and self.num_layers > 1:
            # this currently does not support jagged arrays
            data_dimensions = data_dimensions[1:]
        return data_dimensions

    def _store_prep(self, layer, multiplier):
        assert(layer is None or layer < self.num_layers)
        if layer is None:
            # layer is none means the data provided is for all layers or this
            # is not layered data
            layer = 0
            self.num_layers = 1
            self._internal_data = [None]
        if layer > len(multiplier) - 1:
            if multiplier[0] is None:
                multiplier = 1.0
            else:
                multiplier = multiplier[0]
        else:
            if multiplier[layer] is None:
                multiplier = 1.0
            elif isinstance(multiplier, list):
                multiplier = multiplier[0]

        return layer, multiplier

    def _get_data_size(self, layer):
        dimensions = self.get_data_dimensions(layer)
        data_size = 1
        for dimension in dimensions:
            data_size = data_size * dimension
        return data_size

    def convert_data(self, data, type, data_item=None):
        if type == 'float' or type == 'double':
            try:
                if isinstance(data, str):
                    # fix any scientific formatting that python can't handle
                    data = data.replace('d', 'e')

                return float(ArrayUtil.clean_numeric(data))
            except ValueError:
                except_str = 'Variable "{}" with value "{}" can ' \
                             'not be converted to float. ' \
                             '{}'.format(self.data_dimensions.structure.name,
                                         data,
                                         self.data_dimensions.structure.path)
                print(except_str)
                raise MFDataException(except_str)
        elif type == 'int' or type == 'integer':
            try:
                return int(ArrayUtil.clean_numeric(data))
            except ValueError:
                except_str = 'Variable "{}" with value "{}" can ' \
                             'not be converted to int. ' \
                             '{}'.format(self.data_dimensions.structure.name,
                                         data,
                                         self.data_dimensions.structure.path)
                print(except_str)
                raise MFDataException(except_str)
        elif type == 'string' and data is not None:
            if data_item is None or not data_item.preserve_case:
                # keep strings lower case
                return data.lower()
        return data


class MFTransient(object):
    """
    Parent class for transient data.  This class contains internal objects and
    methods that most end users will not need to access directly.

    Parameters
    ----------
        *args, **kwargs
            Parameters present to support multiple child class interfaces

    Attributes
    ----------
    _current_key : str
        current key defining specific transient dataset to be accessed
    _data_storage : dict
        dictionary of DataStorage objects
    _data_struct_type : str
        type of storage (numpy array type in most cases) used to store data

    Methods
    -------
    add_transient_key(transient_key)
        verifies the validity of the transient key about to be added
    get_data_prep(transient_key)
        called prior to the child class getting data.  ensures that the data
        retrieved will come from the dataset of a specific transient_key
    _set_data_prep(transient_key)
        called prior to the child class setting data.  ensures that the data
        set will go to the dataset of a specific transient_key
    _get_file_entry_prep(transient_key)
        called prior to the child class getting the file entry.  ensures that
        the file entry only reflects the data from a specific transient_key
    _load_prep(first_line, file_handle, block_header, pre_data_comments)
        called prior to the child class loading data from a file.  figures out
        what transient_key to store the data under
    _append_list_as_record_prep(record, transient_key)
        called prior to the child class appending a list to a record.  ensures
        that the list gets appended to the record associated with the key
        transient_key
    _update_record_prep(transient_key)
        called prior to the child class updating a record.  ensures that the
        record being updated is the one associated with the key transient_key
    get_active_key_list() : list
        returns a list of the active transient keys
    _verify_sp(sp_num) : bool
        returns true of the stress period sp_num is within the expected range
        of stress periods for this model

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, *args, **kwargs):
        self._current_key = None
        self._data_storage = None
        self._data_struct_type = None

    def add_transient_key(self, transient_key):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))

    def update_transient_key(self, old_transient_key, new_transient_key):
        if old_transient_key in self._data_storage:
            # replace dictionary key
            self._data_storage[new_transient_key] = \
                self._data_storage[old_transient_key]
            del self._data_storage[old_transient_key]
            if self._current_key == old_transient_key:
                # update current key
                self._current_key = new_transient_key

    def _transient_setup(self, data_storage, data_struct_type):
        self._data_storage = data_storage
        self._data_struct_type = data_struct_type

    def get_data_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))
        self._current_key = transient_key
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)

    def _set_data_prep(self, data, transient_key=0):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))
        self._current_key = transient_key
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)

    def _get_file_entry_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))
        self._current_key = transient_key

    def _load_prep(self, first_line, file_handle, block_header,
                   pre_data_comments=None):
        # transient key is first non-keyword block variable
        transient_key = block_header.get_transient_key()
        if isinstance(transient_key, int):
            if not self._verify_sp(transient_key):
                except_str = 'Invalid transient key "{}" in block' \
                             ' "{}"'.format(transient_key, block_header.name)
                raise MFInvalidTransientBlockHeaderException(except_str)
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)
        self._current_key = transient_key

    def _append_list_as_record_prep(self, record, transient_key=0):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))
        self._current_key = transient_key
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)

    def _update_record_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            assert(self._verify_sp(transient_key))
        self._current_key = transient_key

    def get_active_key_list(self):
        return sorted(self._data_storage.items(), key=itemgetter(0))

    def _verify_sp(self, sp_num):
        if self._path[0].lower() == 'nam':
            return True
        assert ('tdis', 'dimensions', 'nper') in self._simulation_data.mfdata
        nper = self._simulation_data.mfdata[('tdis', 'dimensions', 'nper')]
        return (sp_num <= nper.get_data())


class MFData(object):
    """
    Base class for all data.  This class contains internal objects and methods
    that most end users will not need to access directly.

    Parameters
    ----------
    sim_data : MFSimulationData
        container class for all data for a MF6 simulation
    structure : MFDataStructure
        defines the structure of the data
    enable : bool
        whether this data is currently being used
    path : tuple
        tuple describing path to the data generally in the format (<model>,
        <package>, <block>, <data>)
    dimensions : DataDimensions
        object used to retrieve dimension information about data
    *arges, **kwargs : exists to support different child class parameter sets
        with extra init parameters

    Attributes
    ----------
    _current_key : str
        current key defining specific transient dataset to be accessed

    Methods
    -------
    new_simulation(sim_data)
        points data object to a new simulation

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, enable=True, path=None,
                 dimensions=None, *args, **kwargs):
        # initialize
        self._current_key = None
        self._simulation_data = sim_data
        self.structure = structure
        self.enabled = enable
        self.repeating = False
        if path is None:
            self._path = structure.path
        else:
            self._path = path
        self._data_name = structure.name
        self._data_storage = None
        self._data_type = structure.type
        self._indent_level = 1
        self._internal_data = None
        self._keyword = ''
        if self._simulation_data is not None:
            self._data_dimensions = DataDimensions(dimensions, structure)
            # build a unique path in the simulation dictionary
            self._org_path = self._path
            index = 0
            while self._path in self._simulation_data.mfdata:
                self._path = self._org_path[:-1] + \
                             ('{}_{}'.format(self._org_path[-1], index),)
                index += 1
        self._structure_init()
        # tie this to the simulation dictionary
        sim_data.mfdata[self._path] = self

    def __getattribute__(self, name):
         if name == 'array':
            return self.get_data(apply_mult=True)
         return object.__getattribute__(self, name)

    def __repr__(self):
        return repr(self._get_storage_obj())

    def __str__(self):
        return str(self._get_storage_obj())

    def new_simulation(self, sim_data):
        self._simulation_data = sim_data
        self._data_storage = None

    def aux_var_names(self):
        parent_path = self._path[:-1]
        result = self._simulation_data.mfdata.find_in_path(parent_path,
                                                           'auxnames')
        if result[0] is not None:
            return result.get_data()
        else:
            return []

    def get_description(self, description=None, data_set=None):
        if data_set is None:
            data_set = self.structure
        for index, data_item in data_set.data_items.items():
            if data_item.type == 'record':
                # record within a record, recurse
                description = self.get_description(description, data_item)
            else:
                if data_item.description:
                    if description:
                        description = '{}\n{}'.format(description,
                                                      data_item.description)
                    else:
                        description = data_item.description
        return description

    def load(self, first_line, file_handle, block_header,
             pre_data_comments=None):
        self.enabled = True

    def is_valid(self):
        # TODO: Implement for each data type
        return True

    def _structure_init(self, data_set=None):
        if data_set is None:
            # Initialize variables
            self._num_optional = 0
            self._num_required = 0
            self._num_heading_required = 0
            self._num_loads = 0
            data_set = self.structure
        for data_item_struct in data_set.data_item_structures:
            if data_item_struct.type == 'record':
                # this is a record within a record, recurse
                self._structure_init(data_item_struct)
            else:
                if len(self.structure.data_item_structures) == 1:
                    # data item name is a keyword to look for
                    self._keyword = data_item_struct.name
                if data_item_struct.optional:
                    self._num_optional += 1
                else:
                    self._num_required += 1
                    if data_item_struct.reader == 'readarray':
                        self._num_heading_required += 1

    def _get_constant_formatting_string(self, const_val, layer, data_type,
                                        suffix='\n'):
        sim_data = self._simulation_data
        const_format = list(sim_data.constant_formatting)
        const_format[1] = self._get_storage_obj().to_string(const_val,
                                                            data_type)
        return '{}{}'.format(sim_data.indent_string.join(const_format), suffix)

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

    def _get_aux_var_name(self, aux_var_index):
        aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
        return aux_var_names[0][aux_var_index+1]

    def _load_keyword(self, arr_line, index_num):
        aux_var_index = None
        if self._keyword != '':
            # verify keyword
            keyword_found = arr_line[index_num].lower()
            keyword_match = self._keyword.lower() == keyword_found
            aux_var_names = None
            if not keyword_match:
                aux_var_index = self._get_aux_var_index(keyword_found)
            if not keyword_match and aux_var_index is None:
                aux_text = ''
                if aux_var_names is not None:
                    aux_text = ' or auxiliary variables ' \
                               '{}'.format(aux_var_names[0])
                except_str = 'Error reading variable "{}".  Expected ' \
                             'variable keyword "{}"{} not found ' \
                             'at line "{}". {}'.format(self._data_name,
                                                       self._keyword,
                                                       aux_text,
                                                       ' '.join(arr_line),
                                                       self._path)
                print(except_str)
                raise MFFileParseException(except_str)
            return (index_num + 1, aux_var_index)
        return (index_num, aux_var_index)

    def _read_pre_data_comments(self, line, file_handle, pre_data_comments):
        line_num = 0
        storage = self._get_storage_obj()
        if pre_data_comments:
            storage.pre_data_comments = MFComment(pre_data_comments.text,
                                                  self._path,
                                                  self._simulation_data,
                                                  line_num)
        else:
            storage.pre_data_comments = None

        # read through any fully commented or empty lines
        arr_line = ArrayUtil.split_data_line(line)
        while MFComment.is_comment(arr_line, True) and line != '':
            if storage.pre_data_comments:
                storage.pre_data_comments.add_text('\n')
                storage.pre_data_comments.add_text(' '.join(arr_line))
            else:
                storage.pre_data_comments = MFComment(arr_line, self._path,
                                                      self._simulation_data,
                                                      line_num)

            self._add_data_line_comment(arr_line, line_num)

            line = file_handle.readline()
            arr_line = ArrayUtil.split_data_line(line)
        return line

    def _add_data_line_comment(self, comment, line_num):
        storage = self._get_storage_obj()
        if line_num in storage.comments:
            storage.comments[line_num].add_text('\n')
            storage.comments[line_num].add_text(' '.join(comment))
        else:
            storage.comments[line_num] = MFComment(' '.join(comment),
                                                   self._path,
                                                   self._simulation_data,
                                                   line_num)

    def _build_type_array(self, axis_list, var_type):
        # calculate size
        data_size = 1
        for axis in axis_list:
            data_size = data_size * axis
        self._min_data_length += data_size

        # build flat array
        type_array = []
        for index in range(0, data_size):
            type_array.append((var_type, True))

        # reshape
        return np.reshape(type_array, (2,) + axis_list)

    def _get_storage_obj(self):
        return self._data_storage


class MFMultiDimVar(MFData):
    def __init__(self, sim_data, structure, enable=True, path=None,
                 dimensions=None):
        super(MFMultiDimVar, self).__init__(sim_data, structure, enable, path,
                                            dimensions)

    def _get_internal_formatting_string(self, layer):
        if layer is None:
            layer = 0
        int_format = ['INTERNAL', 'FACTOR']
        data_type = self.structure.get_datum_type()
        layer_storage = self._get_storage_obj().layer_storage[layer]
        if layer_storage.factor is not None:
            int_format.append(str(layer_storage.factor))
        else:
            if data_type == 'float' or data_type == 'double' or \
                    data_type == 'double precision':
                int_format.append('1.0')
            else:
                int_format.append('1')
        if layer_storage.iprn is not None:
            int_format.append('IPRN')
            int_format.append(str(layer_storage.iprn))
        return self._simulation_data.indent_string.join(int_format)

    def _get_external_formatting_string(self, layer, ext_file_action):
        if layer is None:
            layer = 0
        # resolve external file path
        file_mgmt = self._simulation_data.mfpath
        layer_storage = self._get_storage_obj().layer_storage[layer]
        model_name = self._data_dimensions.package_dim.model_dim[0].model_name
        ext_file_path = file_mgmt.get_updated_path(layer_storage.fname,
                                                   model_name,
                                                   ext_file_action)
        ext_format = ['OPEN/CLOSE', "'{}'".format(ext_file_path)]
        ext_format.append('FACTOR')
        if layer_storage.factor is not None:
            ext_format.append(str(layer_storage.factor))
        else:
            if self.structure.type.lower() == 'float' \
                    or self.structure.type.lower() == 'double':
                ext_format.append('1.0')
            else:
                ext_format.append('1')
        if layer_storage.binary:
            ext_format.append('BINARY')
        if layer_storage.iprn is not None:
            ext_format.append('IPRN')
            ext_format.append(str(layer_storage.iprn))
        return '{}\n'.format(
                self._simulation_data.indent_string.join(ext_format))
