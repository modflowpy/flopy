import os, sys, inspect
import numpy as np
from copy import deepcopy
from ..mfbase import MFDataException


def get_first_val(arr):
    while isinstance(arr, list) or isinstance(arr, np.ndarray):
        arr = arr[0]
    return arr


def clean_name(name):
    # remove bad characters
    clean_string = name.replace(' ', '_')
    clean_string = clean_string.replace('-', '_')
    # remove anything after a parenthesis
    index = clean_string.find('(')
    if index != -1:
        clean_string = clean_string[0:index]

    return clean_string


def find_keyword(arr_line, keyword_dict):
    # convert to lower case
    arr_line_lower = []
    for word in arr_line:
        # integers and floats are not keywords
        if not DatumUtil.is_int(word) and not DatumUtil.is_float(word):
            arr_line_lower.append(word.lower())
    # look for constants in order of most words to least words
    key = ''
    for num_words in range(len(arr_line_lower), -1, -1):
        key = tuple(arr_line_lower[0:num_words])
        if len(key) > 0 and key in keyword_dict:
            return key
    return None


def max_tuple_abs_size(some_tuple):
    max_size = 0
    for item in some_tuple:
        item_abs = abs(item)
        if item_abs > max_size:
            max_size = item_abs
    return max_size


class TemplateGenerator(object):
    """
    Abstract base class for building a data template for different data types.
    This is a generic class that is initialized with a path that identifies
    the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)
    """
    def __init__(self, path):
        self.path = path

    def _get_data_dimensions(self, model):
        from ..data import mfstructure
        from ..coordinates import modeldimensions

        # get structure info
        sim_struct = mfstructure.MFStructure().sim_struct
        package_struct = sim_struct.get_data_structure(self.path[0:-2])

        # get dimension info
        data_struct = sim_struct.get_data_structure(self.path)
        package_dim = modeldimensions.PackageDimensions([model.dimensions],
                                                        package_struct,
                                                        self.path[0:-1])
        return data_struct, modeldimensions.DataDimensions(package_dim,
                                                           data_struct)

    def build_type_header(self, type, data=None):
        from ..data.mfdata import DataStorageType

        if type == DataStorageType.internal_array:
            if isinstance(self, ArrayTemplateGenerator):
                return {'factor':1.0, 'iprn':1, 'data':data}
            else:
                return None
        elif type == DataStorageType.internal_constant:
            return data
        elif type == DataStorageType.external_file:
            return {'filename':'', 'factor':1.0, 'iprn':1}
        return None


class ArrayTemplateGenerator(TemplateGenerator):
    """
    Class that builds a data template for MFArrays.  This is a generic class
    that is initialized with a path that identifies the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)

    Methods
    -------
    empty: (model: MFModel, layered: boolean, data_storage_type_list: boolean,
            default_value: int/float) : variable
        Builds a template for the data you need to specify for a specific data
        type (ie. "hk") in a specific model.  The data type and dimensions
        is determined by "path" during initialization of this class and the
        model is passed in to this method as the "model" parameter.  If the
        data is transient a dictionary containing a single stress period
        will be returned.  If "layered" is set to true, data will be returned
        as a list ndarrays, one for each layer.  data_storage_type_list is a
        list of DataStorageType, one type for each layer.  If "default_value"
        is specified the data template will be populated with that value,
        otherwise each ndarray in the data template will be populated with
        np.empty (0 or 0.0 if the DataStorageType is a constant).
    """
    def __init__(self, path):
        super(ArrayTemplateGenerator, self).__init__(path)

    def empty(self, model=None, layered=False, data_storage_type_list=None,
              default_value=None):
        from ..data import mfdata, mfstructure
        from ..data.mfdata import DataStorageType

        # get the expected dimensions of the data
        data_struct, data_dimensions = self._get_data_dimensions(model)
        datum_type = data_struct.get_datum_type()
        data_type = data_struct.get_datatype()
        # build a temporary data storge object
        data_storage = mfdata.DataStorage(
                model.simulation_data, data_dimensions, None,
                mfdata.DataStorageType.internal_array,
                mfdata.DataStructureType.recarray)
        dimension_list = data_storage.get_data_dimensions(None)

        # if layered data
        if layered and dimension_list[0] > 1:
            if data_storage_type_list is not None and \
                    len(data_storage_type_list) != dimension_list[0]:
                comment = 'data_storage_type_list specified with the ' \
                          'wrong size.  Size {} but expected to be ' \
                          'the same as the number of layers, ' \
                          '{}.'.format(len(data_storage_type_list),
                                       dimension_list[0])
                type_, value_, traceback_ = sys.exc_info()

                raise MFDataException(data_struct.get_model(),
                                      data_struct.get_package(),
                                      data_struct.path,
                                      'generating array template',
                                      data_struct.name,
                                      inspect.stack()[0][3],
                                      type_, value_, traceback_, comment,
                                      model.simulation_data.debug)
            # build each layer
            data_with_header = []
            for layer in range(0, dimension_list[0]):
                # determine storage type
                if data_storage_type_list is None:
                    data_storage_type = DataStorageType.internal_array
                else:
                    data_storage_type = data_storage_type_list[layer]
                # build data type header
                data_with_header.append(self._build_layer(datum_type,
                                                          data_storage_type,
                                                          default_value,
                                                          dimension_list))
        else:
            if data_storage_type_list is None or \
                    data_storage_type_list[0] == \
                            DataStorageType.internal_array:
                data_storage_type = DataStorageType.internal_array
            else:
                data_storage_type = data_storage_type_list[0]
            # build data type header
            data_with_header = self._build_layer(datum_type,
                                                 data_storage_type,
                                                 default_value,
                                                 dimension_list, True)

        # if transient/multiple list
        if data_type == mfstructure.DataType.array_transient:
            # Return as dictionary
            return {0:data_with_header}
        else:
            return data_with_header

    def _build_layer(self, data_type, data_storage_type, default_value,
                     dimension_list, all_layers=False):
        from ..data.mfdata import DataStorageType

        # build data
        if data_storage_type == DataStorageType.internal_array:
            if default_value is None:
                if all_layers:
                    data = np.empty(dimension_list, data_type)
                else:
                    data = np.empty(dimension_list[1:], data_type)
            else:
                if all_layers:
                    data = np.full(dimension_list, default_value, data_type)
                else:
                    data = np.full(dimension_list[1:], default_value,
                                   data_type)
        elif data_storage_type == DataStorageType.internal_constant:
            if default_value is None:
                if data_type == np.int:
                    data = 0
                else:
                    data = 0.0
            else:
                data = default_value
        else:
            data = None
        # build data type header
        return self.build_type_header(data_storage_type, data)


class ListTemplateGenerator(TemplateGenerator):
    """
    Class that builds a data template for MFLists.  This is a generic class
    that is initialized with a path that identifies the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)

    Methods
    -------
    empty: (maxbound: int, aux_vars: list, boundnames: boolean, nseg: int) :
            dictionary
        Builds a template for the data you need to specify for a specific data
        type (ie. "stress_period_data") in a specific model.  The data type is
        determined by "path" during initialization of this class.  If the data
        is transient a dictionary containing a single stress period will be
        returned.  The number of entries in the recarray are determined by
        the "maxbound" parameter.  The "aux_vars" parameter is a list of aux
        var names to be used in this data list.  If boundnames is set to
        true and boundname field will be included in the recarray.  nseg is
        only used on list data that contains segments.  If timeseries is true,
        a template that is compatible with time series data is returned.
    """
    def __init__(self, path):
        super(ListTemplateGenerator, self).__init__(path)

    def _build_template_data(self, type_list):
        template_data = []
        for type in type_list:
            if type[1] == int:
                template_data.append(0)
            elif type[1] == float:
                template_data.append(np.nan)
            else:
                template_data.append(None)
        return tuple(template_data)

    def empty(self, model, maxbound=None, aux_vars=None, boundnames=False,
              nseg=None, timeseries=False, stress_periods=None):
        from ..data import mfdata, mfstructure

        data_struct, data_dimensions = self._get_data_dimensions(model)
        data_type = data_struct.get_datatype()
        # build a temporary data storge object
        data_storage = mfdata.DataStorage(
                model.simulation_data, data_dimensions, None,
                mfdata.DataStorageType.internal_array,
                mfdata.DataStructureType.recarray)

        # build type list
        type_list = data_storage.build_type_list(nseg=nseg)
        if aux_vars is not None:
            if len(aux_vars) > 0 and (isinstance(aux_vars[0], list) or
                    isinstance(aux_vars[0], tuple)):
                aux_vars = aux_vars[0]
            for aux_var in aux_vars:
                type_list.append((aux_var, object))
        if boundnames:
            type_list.append(('boundnames', object))

        if timeseries:
            # fix type list to make all types objects
            for index in range(0, len(type_list)):
                type_list[index] = (type_list[index][0], object)

        # build rec array
        template_data = self._build_template_data(type_list)
        rec_array_data = []
        if maxbound is not None:
            for index in range(0, maxbound):
                rec_array_data.append(template_data)
        else:
            rec_array_data.append(template_data)
        rec_array = np.rec.array(rec_array_data, type_list)

        # if transient/multiple list
        if data_type == mfstructure.DataType.list_transient or \
                data_type == mfstructure.DataType.list_multiple:
            # Return as dictionary
            if stress_periods is None:
                return {0:rec_array}
            else:
                template = {}
                for stress_period in stress_periods:
                    template[stress_period] = deepcopy(rec_array)
                return template
        else:
            return rec_array


class DatumUtil(object):
    @ staticmethod
    def is_int(str):
        try:
            int(str)
            return True
        except TypeError:
            return False
        except ValueError:
            return False

    @ staticmethod
    def is_float(str):
        try:
            float(str)
            return True
        except TypeError:
            return False
        except ValueError:
            return False


class ArrayUtil(object):
    """
    Class contains miscellaneous methods to work with and compare arrays

    Parameters
    ----------
    path : string
        file path to read/write to
    max_error : float
        maximum acceptable error when doing a compare of floating point numbers

    Methods
    -------
    is_empty_list : (current_list : list) : boolean
        determines if an n-dimensional list is empty
    con_convert : (data : string, data_type : type that has conversion
                   operation) : boolean
        returns true if data can be converted into data_type
    max_multi_dim_list_size : (current_list : list) : boolean
        determines the max number of items in a multi-dimensional list
        'current_list'
    first_item : (current_list : list) : variable
        returns the first item in the list 'current_list'
    next_item : (current_list : list) : variable
        returns the next item in the list 'current_list'
    array_comp : (first_array : list, second_array : list) : boolean
        compares two lists, returns true if they are identical (with max_error)
    spilt_data_line : (line : string) : list
        splits a string apart (using split) and then cleans up the results
        dealing with various MODFLOW input file releated delimiters.  returns
        the delimiter type used.
    clean_numeric : (text : string) : string
        returns a cleaned up version of 'text' with only numeric characters
    save_array_diff : (first_array : list, second_array : list,
                       first_array_name : string, second_array_name : string)
        saves lists 'first_array' and 'second_array' to files first_array_name
        and second_array_name and then saves the difference of the two
        arrays to 'debug_array_diff.txt'
    save_array(filename : string, multi_array : list)
        saves 'multi_array' to the file 'filename'
    """
    numeric_chars = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,
                     '6': 0, '7': 0, '8': 0, '9': 0, '.': 0, '-': 0}
    quote_list = {"'", '"'}
    delimiter_list = {',': 0, '\t': 0, ' ': 0}
    delimiter_used = None
    line_num = 0
    consistent_delim = False

    def __init__(self, path=None, max_error=0.01):
        self.max_error = max_error
        if path:
            self.path = path
        else:
            self.path = os.getcwd()

    @staticmethod
    def has_one_item(current_list):
        if not isinstance(current_list, list) and not isinstance(current_list,
                                                                 np.ndarray):
            return True
        if len(current_list) != 1:
            return False
        if (isinstance(current_list[0], list) or
                isinstance(current_list, np.ndarray)) and \
                len(current_list[0] != 0):
            return False
        return True

    @staticmethod
    def is_empty_list(current_list):
        if not isinstance(current_list, list):
            return not current_list

        for item in current_list:
            if isinstance(item, list):
                # still in a list of lists, recurse
                if not ArrayUtil.is_empty_list(item):
                    return False
            else:
                return False

        return True

    @staticmethod
    def max_multi_dim_list_size(current_list):
        max_length = -1
        for item in current_list:
            if len(item) > max_length:
                max_length = len(item)
        return max_length

    @staticmethod
    def first_item(current_list):
        if not isinstance(current_list, list):
            return current_list

        for item in current_list:
            if isinstance(item, list):
                # still in a list of lists, recurse
                return ArrayUtil.first_item(item)
            else:
                return item

    @staticmethod
    def next_item(current_list, new_list=True, nesting_change=0,
                  end_of_list=True):
        # returns the next item in a nested list along with other information:
        # (<next item>, <end of list>, <entering new list>,
        #  <change in nesting level>
        if not isinstance(current_list, list) and \
                not isinstance(current_list, np.ndarray):
            yield (current_list, end_of_list, new_list, nesting_change)
        else:
            list_size = 1
            for item in current_list:
                if isinstance(item, list) or isinstance(current_list,
                                                        np.ndarray):
                    # still in a list of lists, recurse
                    for item in ArrayUtil.next_item(item, list_size == 1,
                                                    nesting_change + 1,
                                                    list_size ==
                                                    len(current_list)):
                        yield item
                    nesting_change = -(nesting_change + 1)
                else:
                    yield (item, list_size == len(current_list),
                           list_size == 1, nesting_change)
                    nesting_change = 0
                list_size += 1

    @staticmethod
    def next_list(current_list):
        if not isinstance(current_list[0], list):
            yield current_list
        else:
            for lst in current_list:
                if isinstance(lst[0], list):
                    for lst in ArrayUtil.next_list(lst):
                        yield lst
                else:
                    yield lst

    def array_comp(self, first_array, second_array):
        diff = first_array - second_array
        max = np.max(np.abs(diff))
        if max > self.max_error:
            return False
        return True

    @staticmethod
    def reset_delimiter_used():
        ArrayUtil.delimiter_used = None
        ArrayUtil.line_num = 0
        ArrayUtil.consistent_delim = True

    @staticmethod
    def split_data_line(line, external_file=False, delimiter_conf_length=15):
        if ArrayUtil.line_num > delimiter_conf_length and \
                ArrayUtil.consistent_delim:
            # consistent delimiter has been found.  continue using that
            # delimiter without doing further checks
            if ArrayUtil.delimiter_used == None:
                clean_line = line.strip().split()
            else:
                clean_line = line.strip().split(ArrayUtil.delimiter_used)
        else:
            clean_line = line.strip().split()
            if external_file:
                # try lots of different delimitiers for external files and use the
                # one the breaks the data apart the most
                max_split_size = len(clean_line)
                max_split_type = None
                for delimiter in ArrayUtil.delimiter_list:
                    alt_split = line.strip().split(delimiter)
                    if len(alt_split) > max_split_size:
                        max_split_size = len(alt_split)
                        max_split_type = delimiter
                if max_split_type is not None:
                    clean_line = line.strip().split(max_split_type)
                    if ArrayUtil.line_num == 0:
                        ArrayUtil.delimiter_used = max_split_type
                    elif ArrayUtil.delimiter_used != max_split_type:
                        ArrayUtil.consistent_delim = False
        ArrayUtil.line_num += 1

        arr_fixed_line = []
        index = 0
        # loop through line to fix quotes and delimiters
        while index < len(clean_line):
            item = clean_line[index]
            if item and item not in ArrayUtil.delimiter_list:
                if item and item[0] in ArrayUtil.quote_list:
                    # starts with a quote, handle quoted text
                    if item[-1] in ArrayUtil.quote_list:
                        arr_fixed_line.append(item[1:-1])
                    else:
                        arr_fixed_line.append(item[1:])
                        # loop until trailing quote found
                        while index < len(clean_line):
                            index += 1
                            if index < len(clean_line):
                                item = clean_line[index]
                                if item[-1] in ArrayUtil.quote_list:
                                    arr_fixed_line[-1] = \
                                        '{} {}'.format(arr_fixed_line[-1],
                                                       item[:-1])
                                    break
                                else:
                                    arr_fixed_line[-1] = \
                                        '{} {}'.format(arr_fixed_line[-1],
                                                       item)
                else:
                    # no quote, just append
                    arr_fixed_line.append(item)
            index += 1

        return arr_fixed_line

    @staticmethod
    def clean_numeric(text):
        if isinstance(text, str):
            # remove all non-numeric text from leading and trailing positions
            # of text
            if text:
                while text and (text[0] not in ArrayUtil.numeric_chars or text[-1]
                                not in ArrayUtil.numeric_chars):
                    if text[0] not in ArrayUtil.numeric_chars:
                        text = text[1:]
                    if text and text[-1] not in ArrayUtil.numeric_chars:
                        text = text[:-1]
        return text

    def save_array_diff(self, first_array, second_array, first_array_name,
                        second_array_name):
        try:
            diff = first_array - second_array
            self.save_array(first_array_name, first_array)
            self.save_array(second_array_name, second_array)
            self.save_array('debug_array_diff.txt', diff)
        except:
            print("An error occurred while outputting array differences.")
            return False
        return True

    # Saves an array with up to three dimensions
    def save_array(self, filename, multi_array):
        file_path = os.path.join(self.path, filename)
        with open(file_path, 'w') as outfile:
            outfile.write('{}\n'.format(str(multi_array.shape)))
            if len(multi_array.shape) == 4:
                for slice in multi_array:
                    for second_slice in slice:
                        for third_slice in second_slice:
                            for item in third_slice:
                                outfile.write(' {:10.3e}'.format(item))
                            outfile.write('\n')
                        outfile.write('\n')
                    outfile.write('\n')
            elif len(multi_array.shape) == 3:
                for slice in multi_array:
                    np.savetxt(outfile, slice, fmt='%10.3e')
                    outfile.write('\n')
            else:
                np.savetxt(outfile, multi_array, fmt='%10.3e')


class MultiList():
    """
    Class for storing objects in an n-dimensional list which can be iterated
    through as a single list.

    Parameters
    ----------
    mdlist : list
        multi-dimensional list to initialize the multi-list.  either mdlist
        or both shape and callback must be specified
    shape : tuple
        shape of the multi-list
    callback : method
        callback method that takes a location in the multi-list (tuple) and
        returns an object to be stored at that location in the multi-list

    Methods
    -------
    increment_dimension : (dimension, callback)
        increments the size of one of the two dimensions of the multi-list
    build_list : (callback)
        builds a multi-list of shape self.list_shape, constructing objects
        for the list using the supplied callback method
    first_item : () : object
        gets the first entry in the multi-list
    get_total_size : () : int
        returns the total number of entries in the multi-list
    in_shape : (indexes) : boolean
        returns whether a tuple of indexes are valid indexes for the shape of
        the multi-list
    inc_shape_idx : (indexes) : tuple
        given a tuple of indexes pointing to an entry in the multi-list,
        returns a tuple of indexes pointing to the next entry in the multi-list
    first_index : () : tuple
        returns a tuple of indexes pointing to the first entry in the
        multi-list
    indexes : (start_indexes=None, end_indexes=None) : iter(tuple)
        returns an iterator that iterates from the location in the
        multi-list defined by start_indexes to the location in the
        multi-list defined by end_indexes
    elements : () : iter(object)
        returns an iterator that iterates over each object stored in the
        multi-list
    """
    def __init__(self, mdlist=None, shape=None, callback=None):
        if mdlist is not None:
            self.multi_dim_list = mdlist
            self.list_shape = MultiList._calc_shape(mdlist)
        elif shape is not None:
            self.list_shape = shape
            self.multi_dim_list = []
            if callback is not None:
                self.build_list(callback)
        else:
            raise Exception('MultiList requires either a mdlist or a shape '
                            'at initialization.')

    def __getitem__(self, k):
        if isinstance(k, list) or isinstance(k, tuple):
            item_ptr = self.multi_dim_list
            for index in k:
                item_ptr = item_ptr[index]
            return item_ptr
        else:
            return self.multi_dim_list[k]

    @staticmethod
    def _calc_shape(current_list):
        shape = []
        if isinstance(current_list, list):
            shape.append(len(current_list))
            sub_list = current_list[0]
            if isinstance(sub_list, list):
                shape += MultiList._calc_shape(sub_list)
        elif isinstance(current_list, np.ndarray):
            shape.append(current_list.shape[0])
        else:
            return 1
        return tuple(shape)

    def increment_dimension(self, dimension, callback):
        # ONLY SUPPORTS 1 OR 2 DIMENSIONAL MULTI-LISTS
        # TODO: REWRITE TO SUPPORT N-DIMENSIONAL MULTI-LISTS
        if len(self.list_shape) > 2:
            raise Exception('Increment_dimension currently only supports 1 '
                            'or 2 dimensional multi-lists')
        if len(self.list_shape) == 1:
            self.multi_dim_list.append(callback(len(self.list_shape)))
            self.list_shape = (self.list_shape[0] + 1,)
        else:
            if dimension == 1:
                new_row_idx = len(self.multi_dim_list)
                self.multi_dim_list.append([])
                for index in range(0, self.list_shape[1]):
                    self.multi_dim_list[-1].append(callback((new_row_idx,
                                                             index)))
                self.list_shape = (self.list_shape[0] + 1, self.list_shape[1])
            elif dimension == 2:
                new_col_idx = len(self.multi_dim_list[0])
                for index in range(0, self.list_shape[0]):
                    self.multi_dim_list[index].append(callback((index,
                                                                new_col_idx)))
                self.list_shape = (self.list_shape[0], self.list_shape[1] + 1)
            else:
                raise Exception('For two dimensional lists "dimension" must '
                                'be 1 or 2.')

    def build_list(self, callback):
        entry_points = [(self.multi_dim_list, self.first_index())]
        shape_len = len(self.list_shape)
        # loop through each dimension
        for index, shape_size in enumerate(self.list_shape):
            new_entry_points = []
            # loop through locations to add to the list
            for entry_point in entry_points:
                # loop through the size of current dimension
                for val in range(0, shape_size):
                    if index < (shape_len - 1):
                        # this is a multi-dimensional multi-list, build out
                        # first dimension
                        entry_point[0].append([])
                        if entry_point[1] is None:
                            new_location = (len(entry_point) - 1,)
                        else:
                            new_location = ((len(entry_point[0]) - 1), val)
                        new_entry_points.append((entry_point[0][-1],
                                                 new_location))
                    else:
                        entry_point[0].append(callback(entry_point[1]))
            entry_points = new_entry_points

    def first_item(self):
        return ArrayUtil.first_item(self.multi_dim_list)

    def get_total_size(self):
        shape_size = 1
        for item in self.list_shape:
            if item is None:
                return 0
            else:
                shape_size *= item
        return shape_size

    def in_shape(self, indexes):
        for index, item in zip(indexes, self.list_shape):
            if index > item:
                return False
        return True

    def inc_shape_idx(self, indexes):
        new_indexes = []
        incremented = False
        for index, item in zip(indexes, self.list_shape):
            if index == item:
                new_indexes.append(0)
            elif incremented:
                new_indexes.append(index)
            else:
                incremented = True
                new_indexes.append(index+1)
        if not incremented:
            new_indexes[-1] += 1
        return tuple(new_indexes)

    def first_index(self):
        first_index = []
        for index in self.list_shape:
            first_index.append(0)
        return tuple(first_index)

    def nth_index(self, n):
        index = None
        aii = ArrayIndexIter(self.list_shape, True)
        index_num = 0
        while index_num <= n:
            index = aii.next()
            index_num += 1
        return index

    def indexes(self, start_indexes=None, end_indexes=None):
        aii = ArrayIndexIter(self.list_shape, True)
        if start_indexes is not None:
            aii.current_location = list(start_indexes)
            aii.current_index = len(aii.current_location) - 1
        if end_indexes is not None:
            aii.end_location = list(end_indexes)
        return aii

    def elements(self):
        return MultiListIter(self.multi_dim_list, False)


class ArrayIndexIter(object):
    def __init__(self, array_shape, index_as_tuple=False):
        self.array_shape = array_shape
        self.current_location = []
        self.end_location = []
        self.first_item = True
        self.index_as_tuple = index_as_tuple
        for item in array_shape:
            self.current_location.append(0)
            self.end_location.append(item)
        self.current_index = len(self.current_location) - 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_item:
            self.first_item = False
            if self.current_location[self.current_index] < \
                    self.end_location[self.current_index]:
                if len(self.current_location) > 1 or self.index_as_tuple:
                    return tuple(self.current_location)
                else:
                    return self.current_location[0]
        while self.current_index >= 0:
            location = self.current_location[self.current_index]
            if location < self.end_location[self.current_index] - 1:
                self.current_location[self.current_index] += 1
                self.current_index = len(self.current_location) - 1
                if len(self.current_location) > 1 or self.index_as_tuple:
                    return tuple(self.current_location)
                else:
                    return self.current_location[0]
            else:
                self.current_location[self.current_index] = 0
                self.current_index -= 1
        raise StopIteration()

    next = __next__  # Python 2 support


class MultiListIter(object):
    def __init__(self, multi_list, detailed_info=False, iter_leaf_lists=False):
        self.multi_list = multi_list
        self.detailed_info = detailed_info
        if iter_leaf_lists:
            self.val_iter = ArrayUtil.next_list(self.multi_list)
        else:
            self.val_iter = ArrayUtil.next_item(self.multi_list)

    def __iter__(self):
        return self

    def __next__(self):
        next_val = next(self.val_iter)
        if self.detailed_info:
            return next_val
        else:
            return next_val[0]

    next = __next__  # Python 2 support


class ConstIter(object):
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return self

    def __next__(self):
        return self.value

    next = __next__  # Python 2 support


class FileIter(object):
    def __init__(self, file_path):
        self.eof = False
        try:
            self._fd = open(file_path, 'r')
        except:
            self.eof = True
        self._current_data = None
        self._data_index = 0
        self._next_line()

    def __iter__(self):
        return self

    def __next__(self):
        if self.eof:
            raise StopIteration()
        else:
            while self._current_data is not None and \
                  self._data_index >= len(self._current_data):
                self._next_line()
                self._data_index = 0
                if self.eof:
                    raise StopIteration()
            self._data_index += 1
            return self._current_data[self._data_index-1]

    def close(self):
        self._fd.close()

    def _next_line(self):
        if self.eof:
            return
        data_line = self._fd.readline()
        if data_line is None:
            self.eof = True
            return
        self._current_data = ArrayUtil.split_data_line(data_line)

    next = __next__  # Python 2 support


class NameIter(object):
    def __init__(self, name, first_not_numbered=True):
        self.name = name
        self.iter_num = -1
        self.first_not_numbered = first_not_numbered

    def __iter__(self):
        return self

    def __next__(self):
        self.iter_num += 1
        if self.iter_num == 0 and self.first_not_numbered:
            return self.name
        else:
            return '{}_{}'.format(self.name, self.iter_num)

    next = __next__  # Python 2 support


class PathIter(object):
    def __init__(self, path, first_not_numbered=True):
        self.path = path
        self.name_iter = NameIter(path[-1], first_not_numbered)

    def __iter__(self):
        return self

    def __next__(self):
        return self.path[0:-1] + (self.name_iter.__next__(),)

    next = __next__  # Python 2 support


class MFDocString(object):
    """
    Helps build a python class doc string

    Parameters
    ----------
    description : string
        description of the class

    Attributes
    ----------
    indent: string
        indent to use in doc string
    description : string
        description of the class
    parameter_header : string
        header for parameter section of doc string
    parameters : list
        list of docstrings for class parameters

    Methods
    -------
    add_parameter : (param_descr : string, beginning_of_list : bool)
        adds doc string for a parameter with description 'param_descr' to the
        end of the list unless beginning_of_list is True
    get_doc_string : () : string
        builds and returns the docstring for the class
    """
    def __init__(self, description):
        self.indent = '    '
        self.description = description
        self.parameter_header = '{}Parameters\n{}' \
                                '----------'.format(self.indent, self.indent)
        self.parameters = []
        self.model_parameters = []

    def add_parameter(self, param_descr, beginning_of_list=False,
                      model_parameter=False):
        if beginning_of_list:
            self.parameters.insert(0, param_descr)
            if model_parameter:
                self.model_parameters.insert(0, param_descr)
        else:
            self.parameters.append(param_descr)
            if model_parameter:
                self.model_parameters.append(param_descr)

    def get_doc_string(self, model_doc_string=False):
        doc_string = '{}"""\n{}{}\n\n{}\n'.format(self.indent, self.indent,
                                                  self.description,
                                                  self.parameter_header)
        if model_doc_string:
            param_list = self.model_parameters
            doc_string = '{}    modelname : string\n        name of the ' \
                         'model\n    model_nam_file : string\n' \
                         '        relative path to the model name file from ' \
                         'model working folder\n    version : string\n' \
                         '        version of modflow\n    exe_name : string\n'\
                         '        model executable name\n' \
                         '    model_ws : string\n' \
                         '        model working folder path' \
                         '\n'.format(doc_string)
        else:
            param_list = self.parameters
        for parameter in param_list:
            doc_string += '{}\n'.format(parameter)
        if not model_doc_string:
            doc_string += '\n{}"""'.format(self.indent)
        return doc_string
