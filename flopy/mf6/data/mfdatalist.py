from collections import OrderedDict
import math
from copy import deepcopy
from ..data import mfstructure, mfdatautil, mfdata
from ..mfbase import ExtFileAction


class MFList(mfdata.MFMultiDimVar):
    """
    Provides an interface for the user to access and update MODFLOW scalar data.

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
    data_exists : bool
        Returns true of data exists in this list, false otherwise.
    has_data : (layer_num : int) : bool
        Returns whether layer "layer_num" has any data associated with it.  For unlayered data do not pass in "layer".
    get_data : (layer_num : int) : ndarray
        Returns the data associated with layer "layer_num".  If "layer_num" is None, returns all data.
    set_data : (data : ndarray/list/dict, multiplier : float, layer_num : int)
        Sets the contents of the data at layer "layer_num" to "data" with multiplier "multiplier".  For unlayered
        data do not pass in "layer_num".  data can have the following formats:
            1) ndarray - ndarray containing the datalist
            2) [(line_one), (line_two), ...] - list where each like of the datalist is a tuple within the list
            3) {'filename':filename, factor=fct, iprn=print_code} - dictionary defining the external file containing
               the datalist.
        If the data is transient, a dictionary can be used to specify each stress period where the dictionary
        key is <stress period> - 1 and the dictionary value is the datalist data defined above:
            {0:ndarray, 1:[(line_one), (line_two), ...], 2:{'filename':filename})
    append_data : (data : list(tuple))
        Appends "data" to the end of this list.  Assumes data is in a format that can be appended directly
        to a numpy recarray.
    append_list_as_record : (data : list)
        Appends the list "data" as a single record in this list's recarray.  Assumes "data" has the correct
        dimensions.
    update_record : (record : list, key_index : int)
        Updates a record at index "key_index" with the contents of "record".  If the index does not exist
        update_record appends the contents of "record" to this list's recarray.
    search_data : (search_term : string, col : int)
        Searches the list data at column "col" for "search_term".  If col is None search_data searches
        the entire list.
    load : (first_line : string, file_handle : file descriptor, block_header : MFBlockHeader,
            pre_data_comments : MFComment) : tuple (bool, string)
        Loads data from first_line (the first line of data) and open file file_handle which is pointing to
        the second line of data.  Returns a tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
    get_file_entry : (layer : int) : string
        Returns a string containing the data in layer "layer".  For unlayered data do not pass in "layer".

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, data=None, enable=True, path=None, dimensions=None):
        super(MFList, self).__init__(sim_data, structure, enable, path, dimensions)
        self._data_storage = self._new_storage()
        self._last_line_info = []
        self._aux_vars = []
        self._data_line = None
        if data is not None:
            self.set_data(data, True)

    def new_simulation(self, sim_data):
        super(MFList, self).new_simulation(sim_data)
        self._data_storage = self._new_storage()
        self._aux_vars = []
        self._data_line = None

    def data_exists(self):
        if self._get_storage_obj() is None:
            return None
        return self._get_storage_obj().has_data()

    def has_data(self):
        if self._get_storage_obj() is None:
            return False
        return self._get_storage_obj().has_data()

    def get_data(self, apply_mult=False):
        if self._get_storage_obj() is None:
            return None
        return self._get_storage_obj().get_data()

    def set_data(self, data, autofill=False):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage()
        # error check data shape

        # store data
        self._get_storage_obj().set_data(data, autofill=autofill)

    def append_data(self, data):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage()
        # store data
        self._get_storage_obj().append_data(data)

    def append_list_as_record(self, record):
        # convert to tuple
        tuple_record = ()
        for item in record:
            tuple_record += (item,)
        # store
        self._get_storage_obj().append_data([tuple_record])

    def update_record(self, record, key_index):
        #if not self._get_storage_obj().update_item(record, key_index):
        self.append_list_as_record(record)

    def search_data(self, search_term, col=None):
        data = self._get_storage_obj().get_data()
        if data is not None:
            search_term = search_term.lower()
            for row in data:
                col_num = 0
                for val in row:
                    if val is not None and val.lower() == search_term and (col == None or col == col_num):
                        return (row, col)
                    col_num += 1
        return None

    def get_file_entry(self, values_only=False, ext_file_action=ExtFileAction.copy_relative_paths):
        # freeze model grid to boost performance
        self._data_dimensions.lock()
        # init
        file_entry = ''
        storage = self._get_storage_obj()
        if storage is None or not storage.has_data():
            return file_entry

        # write out initial comments
        if storage.pre_data_comments:
            file_entry = storage.pre_data_comments.get_file_entry()

        if storage.layer_storage[0].data_storage_type == mfdata.DataStorageType.external_file:
            file_entry = '{}{}{}'.format(self._simulation_data.indent_string,
                                         self._simulation_data.indent_string,
                                         self._get_external_formatting_string(0, ext_file_action).upper())
        else:
            data_complete = storage.get_data()
            if storage.layer_storage[0].data_storage_type == mfdata.DataStorageType.internal_constant:
                data_lines = 1
            else:
                data_lines = len(data_complete)
            # loop through list line by line - assumes first data_item size is representative
            for mflist_line in range(0, data_lines):
                text_line = []
                index = 0
                self._get_file_entry_record(data_complete, mflist_line, text_line, index, self.structure)

                # include comments
                if mflist_line in storage.comments and storage.comments[mflist_line].text:
                    text_line.append(storage.comments[mflist_line].text)

                if len(file_entry) > 0:
                    file_entry = '{}{}{}\n'.format(file_entry, self._simulation_data.indent_string,
                                                   self._simulation_data.indent_string.join(text_line))
                else:
                    file_entry = '{}{}\n'.format(self._simulation_data.indent_string,
                                                 self._simulation_data.indent_string.join(text_line))

        # unfreeze model grid
        self._data_dimensions.unlock()

        return file_entry

    def _get_file_entry_record(self, data_complete, mflist_line, text_line, index, data_set, record_in_record=False):
        storage = self._get_storage_obj()
        if storage.layer_storage[0].data_storage_type == mfdata.DataStorageType.internal_constant:
            #  constant data
            data_type = self.structure.data_item_structures[1].type
            text_line.append('{}{}{}'.format(self._simulation_data.indent_string,
                                             self._simulation_data.indent_string,
                                             self._get_constant_formatting_string(storage.get_const_val(0),
                                                                                  0, data_type, '').upper()))
        else:
            for data_item in data_set.data_item_structures:
                if data_item.name == 'aux':
                    aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
                    if aux_var_names is not None:
                        for aux_var_name in aux_var_names[0]:
                            if aux_var_name.lower() != 'auxiliary':
                                data_val = data_complete[mflist_line][index]
                                text_line.append(self._get_storage_obj().to_string(data_val,
                                                                                   data_item.type_obj,
                                                                                   data_item.is_cellid,
                                                                                   data_item.possible_cellid,
                                                                                   data_item.ucase))
                                index += 1
                elif data_item.type == 'record':
                    # record within a record, recurse
                    self._get_file_entry_record(data_complete, mflist_line, text_line, index, data_item, True)
                elif (data_item.name != 'boundname' or self._data_dimensions.package_dim.boundnames()) and \
                  (not data_item.optional or len(data_item.name) < 5 or data_item.name[0:5] != 'mname'
                  or not storage.in_model):
                    if len(data_complete[mflist_line]) <= index:
                        if data_item.optional == False:
                            except_str = 'ERROR: Not enough data provided for {}. Data for required data ' \
                                         'item "{}" not found'.format(self._path, data_item.name)
                            print(except_str)
                            raise mfstructure.MFFileParseException(except_str)
                        else:
                            break
                    # resolve size of data
                    data_val = data_complete[mflist_line][index]
                    data_line = [data_complete[mflist_line]]
                    resolved_shape, shape_rule = self._data_dimensions.get_data_shape(data_item, self.structure,
                                                                                      data_line,
                                                                                      repeating_key=self._current_key)
                    if data_item.is_cellid or (data_item.possible_cellid and self._validate_cellid([data_val], 0)):
                        if data_item.shape is not None and len(data_item.shape) > 0 and \
                          data_item.shape[0] == 'ncelldim':
                            cellid_size = self._data_dimensions.get_model_grid().get_num_spatial_coordinates()
                            data_item.remove_cellid(resolved_shape, cellid_size)
                    data_size = 1
                    if len(resolved_shape) == 1 and mfdatautil.DatumUtil.is_int(resolved_shape[0]):
                        data_size = int(resolved_shape[0])
                        if data_size < 0:
                            # unable to resolve data size based on shape, use the data heading names
                            # to resolve data size
                            data_size = self._get_storage_obj().resolve_data_size(index)
                    data_complete_len = len(data_complete[mflist_line])
                    for data_index in range(0, data_size):
                        if data_complete_len > index:
                            data_val = data_complete[mflist_line][index]
                            if data_item.type == 'keyword':
                                text_line.append(data_item.name.upper())
                                if self.structure.block_variable:
                                    # block variables behave differently for now.  this needs to be resolved
                                    # more consistently at some point
                                    index += 1
                            elif data_val is not None and (not mfdatautil.DatumUtil.is_float(data_val) or
                                    not math.isnan(float(data_val))):
                                if data_item.tagged and data_index == 0:
                                    # data item tagged, include data item name as a keyword
                                    text_line.append(storage.to_string(data_val,
                                                                       'string',
                                                                       False,
                                                                       force_upper_case=data_item.ucase))
                                    index += 1
                                    data_val = data_complete[mflist_line][index]
                                text_line.append(storage.to_string(data_val,
                                                                   data_item.type_obj,
                                                                   data_item.is_cellid,
                                                                   data_item.possible_cellid,
                                                                   data_item.ucase))
                                index += 1
                            if data_item.type == 'keystring':
                                # keystring must be at the end of the line so everything else is part of the
                                # keystring data
                                for data_index in range(index, data_complete_len):
                                    data_val = data_complete[mflist_line][data_index]
                                    if data_val is not None:
                                        text_line.append(storage.to_string(data_val,
                                                         data_item.type_obj,
                                                         data_item.is_cellid,
                                                         data_item.possible_cellid,
                                                         data_item.ucase))
                                index = data_index
                        elif not data_item.optional and shape_rule is None:
                            except_str = 'ERROR: Not enough data provided for {}. Data for required data ' \
                                         'item "{}" not found'.format(self._path, data_item.name)
                            print(except_str)
                            raise mfstructure.MFFileParseException(except_str)

    def load(self, first_line, file_handle, block_header, pre_data_comments=None):
        super(MFList, self).load(first_line, file_handle, block_header, pre_data_comments=None)

        self._data_dimensions.lock()
        self._last_line_info = []
        self._aux_vars = []
        simple_line = False
        data_loaded = []
        storage = self._get_storage_obj()

        # read in any pre data comments
        current_line = self._read_pre_data_comments(first_line, file_handle, pre_data_comments)

        arr_line = mfdatautil.ArrayUtil.split_data_line(current_line)
        if arr_line and (len(arr_line[0]) >= 2 and arr_line[0][:3].upper() == 'END'):
            return [False, arr_line]
        store_data = False
        if len(arr_line) >= 2 and arr_line[0].upper() == 'OPEN/CLOSE':
            line_num = 0
            self._get_storage_obj().process_open_close_line(arr_line, 0)
        else:
            try:
                line_num = 0
                simple_line = self._load_line(arr_line, line_num, data_loaded, True)[1]
                line_num += 1
                store_data = True
            except mfstructure.MFFileParseException as err:
                # this could possibly be a constant line.
                line = file_handle.readline()
                arr_line = mfdatautil.ArrayUtil.split_data_line(line)
                if len(arr_line) >= 2 and arr_line[0].upper() == 'CONSTANT' and \
                  len(self.structure.data_item_structures) >= 2 and \
                  self.structure.data_item_structures[0].name.upper() == 'CELLID':
                    # store first line as a comment
                    if storage.pre_data_comments is None:
                        storage.pre_data_comments = mfdata.MFComment(current_line, self._path,
                                                                     self._simulation_data, 0)
                    else:
                        storage.pre_data_comments.add_text(current_line)
                    # store constant value for all cellids
                    storage = self._get_storage_obj()
                    storage.layer_storage[0].data_storage_type = mfdata.DataStorageType.internal_constant
                    storage.store_internal(storage.convert_data(arr_line[1],
                                                                self.structure.data_item_structures[1].type),
                                                                0, const=True, multiplier=[1.0])
                    line = ' '
                    while line != '':
                        line = file_handle.readline()
                        arr_line = mfdatautil.ArrayUtil.split_data_line(line)
                        if arr_line and (len(arr_line[0]) >= 2 and arr_line[0][:3].upper() == 'END'):
                            return [False, line]
                else:
                    # not a constant or open/close line, exception is valid
                    raise err

        if self.structure.type == 'record' or self.structure.type == 'string':
            # records only contain a single line
            storage.append_data(data_loaded)
            self._data_dimensions.unlock()
            return [False, None]

        # get block recarray list for later processing
        recarrays = []
        parent_block = self.structure.parent_block
        if parent_block is not None:
            recarrays = parent_block.get_all_recarrays()
        recarray_len = len(recarrays)

        # loop until end of block
        line = ' '
        while line != '':
            line = file_handle.readline()
            arr_line = mfdatautil.ArrayUtil.split_data_line(line)
            if arr_line and (len(arr_line[0]) >= 2 and arr_line[0][:3].upper() == 'END'):
                # end of block
                if store_data:
                    # store as rec array
                    storage.set_data(data_loaded, self._current_key)
                self._data_dimensions.unlock()
                return [False, line]
            if recarray_len != 1 and not mfdata.MFComment.is_comment(arr_line, True):
                key = mfdatautil.find_keyword(arr_line, self.structure.get_keywords())
                if key is None:
                    # unexpected text, may be start of another record
                    if store_data:
                        storage.set_data(data_loaded, self._current_key)
                    self._data_dimensions.unlock()
                    return [True, line]
            if simple_line and self.structure.num_optional == 0:
                # do higher performance quick load
                self._data_line = ()
                cellid_index = 0
                cellid_tuple = ()
                for entry in self._last_line_info:
                    for sub_entry in entry:
                        if sub_entry[1] is not None:
                            if sub_entry[2] > 0:
                                # is a cellid
                                cell_num = storage.convert_data(arr_line[sub_entry[0]], sub_entry[1])
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
                                self._data_line += (storage.convert_data(arr_line[sub_entry[0]], sub_entry[1]),)
                        else:
                            self._data_line += (None,)
                data_loaded.append(self._data_line)

            else:
                self._load_line(arr_line, line_num, data_loaded, False)
            line_num += 1
        if store_data:
            # store as rec array
            storage.set_data(data_loaded, self._current_key)
        self._data_dimensions.unlock()
        return [False, None]

    def _new_storage(self):
        return mfdata.DataStorage(self._simulation_data,
                                  self._data_dimensions,
                                  mfdata.DataStorageType.internal_array,
                                  mfdata.DataStructureType.recarray)

    def _get_storage_obj(self):
        return self._data_storage

    def _load_line(self, arr_line, line_num, data_loaded, build_type_list,
                   data_index_start=0, data_set=None, ignore_optional_vars=False):
        org_data_line = self._data_line
        storage = self._get_storage_obj()
        simple_line = True
        # only initialize if we are at the start of a new line
        if data_index_start == 0:
            data_set = self.structure
            if build_type_list:
                self._data_item_struct_list = []
            # new line of data
            self._data_line = ()
            # determine if at end of block
            if arr_line and arr_line[0][:3].upper() == 'END':
                self.enabled = True
                return arr_line
        data_index = data_index_start
        arr_line_len = len(arr_line)
        if mfdata.MFComment.is_comment(arr_line, True) and data_index_start == 0:
            arr_line.insert(0, '\n')
            self._add_data_line_comment(arr_line, line_num)
        else:
            # read variables
            var_index = 0
            data = ''
            for data_item, data_item_index in zip(data_set.data_item_structures,
                                                  range(0, len(data_set.data_item_structures))):
                if not data_item.optional or not ignore_optional_vars:
                    if data_item.name == 'aux':
                        data_index = self._process_aux(arr_line, arr_line_len, data_item, data_index, var_index)
                    # optional mname data items are only specified if the package is part of a model
                    elif not data_item.optional or data_item.name[0:5] != 'mname' or not storage.in_model:
                        if data_item.type == 'keyword':
                            data_index += 1
                            simple_line = False
                        elif data_item.type == 'record':
                            # this is a record within a record, recurse into _load_line to load it
                            data_index = self._load_line(arr_line, line_num, data_loaded,
                                                         build_type_list, data_index, data_item)[0]
                            simple_line = False
                        elif data_item.name != 'boundname' or self._data_dimensions.package_dim.boundnames():
                            if data_item.optional and data == '#':
                                # comment mark found and expecting optional data_item, we are done
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
                                    elif self.structure.num_optional >= len(data_set.data_item_structures) - \
                                        data_item_index:
                                        # there are enough optional variables to account for the lack of data
                                        # reload line with all optional variables ignored
                                        # TODO: Decide whether optional variables should be allowed to exist anywhere
                                        self._data_line = org_data_line
                                        return self._load_line(arr_line, line_num, data_loaded, build_type_list,
                                                               data_index_start, data_set, True)
                                    else:
                                        except_str = 'ERROR: Not enough data provided for {}. Data for required data ' \
                                                     'item "{}" not found'.format(self._path, data_item.name)
                                        print(except_str)
                                        raise mfstructure.MFFileParseException(except_str)

                                data = arr_line[data_index]
                                repeat_count += 1
                                if data_item.type == 'keystring':
                                    simple_line = False
                                    if repeat_count <= 1:  # only process the keyword on the first repeat
                                        # find data item associated with correct keystring
                                        name_data = data.lower()
                                        if name_data not in data_item.keystring_dict:
                                            name_data = '{}record'.format(name_data)
                                            if name_data not in data_item.keystring_dict:
                                                # data does not match any expected keywords
                                                print('WARNING: Failed to process line {}.  Line does not match expected '
                                                      'keystring {}'.format(' '.join(arr_line), data_item.name))
                                                break
                                        data_item_ks = data_item.keystring_dict[name_data]
                                        assert(data_item_ks != 0)
                                        # keyword is always implied in a keystring and should be stored,
                                        # add a string data_item for the keyword
                                        keyword_data_item = deepcopy(data_item)
                                        keyword_data_item.type = 'string'
                                        data_index, more_data_expected, unknown_repeats = self._append_data(keyword_data_item,
                                                                                                            arr_line,
                                                                                                            arr_line_len,
                                                                                                            data_index,
                                                                                                            var_index,
                                                                                                            repeat_count)
                                    if data_item_ks.type != 'keyword':
                                        # data item contains additional information
                                        data_index, more_data_expected, unknown_repeats = self._append_data(data_item_ks,
                                                                                                            arr_line,
                                                                                                            arr_line_len,
                                                                                                            data_index,
                                                                                                            var_index,
                                                                                                            repeat_count)
                                    else:
                                        # append empty data as a placeholder.  this is necessarily to keep the recarray
                                        # a consistent shape
                                        self._data_line = self._data_line + (None,)
                                        data_index += 1
                                else:
                                    if data_item.tagged and repeat_count == 1:
                                        # data item tagged, include data item name as a keyword
                                        di_type = data_item.type
                                        data_item.type = 'keyword'
                                        data_index, more_data_expected, unknown_repeats = self._append_data(data_item,
                                                                                                            arr_line,
                                                                                                            arr_line_len,
                                                                                                            data_index,
                                                                                                            var_index,
                                                                                                            repeat_count)
                                        data_item.type = di_type
                                    data_index, more_data_expected, unknown_repeats = self._append_data(data_item,
                                                                                                        arr_line,
                                                                                                        arr_line_len,
                                                                                                        data_index,
                                                                                                        var_index,
                                                                                                        repeat_count)
                                if more_data_expected is None:
                                    # indeterminate amount of data expected.  keep reading data until eoln
                                    more_data_expected = (data_index < arr_line_len)
                                simple_line = simple_line and not unknown_repeats and len(data_item.shape) == 0
                    var_index += 1

            # populate unused optional variables with None type
            for data_item in data_set.data_item_structures[var_index:]:
                simple_line = False
                if data_item.name == 'aux':
                    self._process_aux(arr_line, arr_line_len, data_item, data_index, var_index)
                elif data_item.name != 'boundname' or self._data_dimensions.package_dim.boundnames():
                    data_index, more_data_expected, unknown_repeats = self._append_data(data_item, None, 0, data_index,
                                                                                        var_index, 1)

            # only do final processing on outer-most record
            if data_index_start == 0:
                # if more pieces exist
                if arr_line_len > data_index + 1:
                    # FEATURE: Keep number of white space characters used in comments section
                    storage.comments[line_num] = mfdata.MFComment(' '.join(arr_line[data_index+1:]), self._path,
                                                                  self._simulation_data, line_num)
                data_loaded.append(self._data_line)
        return data_index, simple_line

    def _process_aux(self, arr_line, arr_line_len, data_item, data_index, var_index):
        aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
        if aux_var_names is not None:
            for var_name in aux_var_names[0]:
                if var_name.lower() != 'auxiliary':
                    if data_index >= arr_line_len:
                        # store placeholder None
                        data_index, more_data_expected, unknown_repeats = self._append_data(data_item, None, 0,
                                                                                            data_index, var_index, 1)
                    else:
                        # read in aux variables
                        data_index, more_data_expected, unknown_repeats = self._append_data(data_item, arr_line,
                                                                                            arr_line_len, data_index,
                                                                                            var_index, 0)
        return data_index

    def _append_data(self, data_item, arr_line, arr_line_len, data_index, var_index, repeat_count):
        # append to a 2-D list which will later be converted to a numpy rec array
        self._last_line_info.append([])
        if data_item.is_cellid or (data_item.possible_cellid and self._validate_cellid(arr_line, data_index)):
            if self._data_dimensions is None:
                except_str = 'ERROR: CellID field specified in MFDataList "{}" field "{}" which ' \
                             'does not contain a model grid.  Can not resolve model grid type for ' \
                             'cellid.  {}'.format(self.structure.name, data_item.name, self._path)
                print(except_str)
                raise mfstructure.MFFileParseException(except_str)

            #resolved_shape = self._data_dimensions.get_data_shape(data_item, self.structure, [self._data_line])

            # read in the entire cellid
            cellid_size = self._data_dimensions.get_model_grid().get_num_spatial_coordinates()
            cellid_tuple = ()
            if not mfdatautil.DatumUtil.is_int(arr_line[data_index]) and arr_line[data_index].lower() == 'none':
                # special case where cellid is 'none', store as tuple of 'none's
                cellid_tuple = ('none',) * cellid_size
                self._last_line_info[-1].append([data_index, 'string', cellid_size])
                new_index = data_index + 1
            else:
                # handle regular cellid
                if cellid_size + data_index > arr_line_len:
                    except_str = 'ERROR: Not enough data found to represent CellID in MFDataList ' \
                                 '"{}" field "{}".  Expected {} items and found {} in line "{}". ' \
                                 '{}'.format(self.structure.name, data_item.name, cellid_size,
                                              arr_line_len - data_index, arr_line, self._path)
                    print(except_str)
                    raise mfstructure.MFFileParseException(except_str)
                for index in range(data_index, cellid_size + data_index):
                    if not mfdatautil.DatumUtil.is_int(arr_line[index]) or int(arr_line[index]) < 0:
                        except_str = 'ERROR: Expected a integer or CellID in MFDataList ' \
                                     '"{}" field "{}".  Found {} in line "{}". ' \
                                     '{}'.format(self.structure.name, data_item.name, arr_line[index],
                                                 arr_line, self._path)
                        print(except_str)
                        raise mfstructure.MFFileParseException(except_str)

                    data_converted = self._get_storage_obj().convert_data(arr_line[index], data_item.type)
                    cellid_tuple = cellid_tuple + (int(data_converted)-1,)
                    self._last_line_info[-1].append([index, 'integer', cellid_size])
                new_index = data_index + cellid_size
            self._data_line = self._data_line + (cellid_tuple,)
            if data_item.shape is not None and len(data_item.shape) > 0 and data_item.shape[0] == 'ncelldim':
                # shape is the coordinate shape, which has already been read
                more_data_expected = False
                unknown_repeats = False
            else:
                more_data_expected, unknown_repeats = self._resolve_shape(data_item, repeat_count)
            return new_index, more_data_expected, unknown_repeats
        else:
            if arr_line is None:
                data_converted = None
                self._last_line_info[-1].append([data_index, None, 0])
            else:
                if arr_line[data_index].lower() in self._data_dimensions.package_dim.get_tsnames():
                    # references a time series, store as is
                    data_converted = arr_line[data_index].lower()
                    # override recarray data type to support writing string values
                    self._get_storage_obj().override_data_type(var_index, object)
                    self._last_line_info[-1].append([data_index, 'string', 0])
                else:
                    data_converted = self._get_storage_obj().convert_data(arr_line[data_index], data_item.type)
                    self._last_line_info[-1].append([data_index, data_item.type, 0])
            self._data_line = self._data_line + (data_converted,)
            more_data_expected, unknown_repeats = self._resolve_shape(data_item, repeat_count)
            return data_index + 1, more_data_expected, unknown_repeats

    def _resolve_shape(self, data_item, repeat_count, cellid_size=None):
        resolved_shape, shape_rule = self._data_dimensions.get_data_shape(data_item, self.structure, [self._data_line],
                                                                          repeating_key=self._current_key)
        if cellid_size is not None:
            data_item.remove_cellid(resolved_shape, cellid_size)

        if len(resolved_shape) == 1:
            if mfdatautil.DatumUtil.is_int(resolved_shape[0]):
                if repeat_count < int(resolved_shape[0]):
                    return True, shape_rule is not None
                elif int(resolved_shape[0]) == -9999:
                    # repeating unknown number of times in 1-D array
                    return False, True
            else:
                return False, False
        return False, False

    def _validate_cellid(self, arr_line, data_index):
        if arr_line is None:
            return False
        cellid_size = self._data_dimensions.get_model_grid().get_num_spatial_coordinates()
        if cellid_size + data_index > len(arr_line):
            return False
        for index in range(data_index, cellid_size + data_index):
            if not mfdatautil.DatumUtil.is_int(arr_line[index]):
                return False
            if int(arr_line[index]) <= 0:
                return False
        return True


class MFTransientList(MFList, mfdata.MFTransient):
    """
    Provides an interface for the user to access and update MODFLOW transient list data.

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
        Adds a new transient time allowing data for that time to be stored and retrieved using the key
        "transient_key"
    data_exists (key : int) : bool
        Returns true of data exists at key "key", false otherwise.
    add_one :(transient_key : int)
        Adds one to the data stored at key "transient_key"
    get_data : (layer_num : int, key : int) : ndarray
        Returns the data associated with layer "layer_num" during time "key".  If "layer_num" is None,
        returns all data for time "key".
    set_data : (data : ndarray/list, multiplier : float, layer_num : int, key : int)
        Sets the contents of the data at layer "layer_num" and time "key" to "data" with multiplier "multiplier".
        For unlayered data do not pass in "layer_num".
    load : (first_line : string, file_handle : file descriptor, block_header : MFBlockHeader,
            pre_data_comments : MFComment) : tuple (bool, string)
        Loads data from first_line (the first line of data) and open file file_handle which is pointing to
        the second line of data.  Returns a tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
    get_file_entry : (layer : int, key : int) : string
        Returns a string containing the data in layer "layer" at time "key".  For unlayered data do not pass in "layer".
    append_list_as_record : (data : list, key : int)
        Appends the list "data" as a single record in this list's recarray at time "key".  Assumes
        "data" has the correct dimensions.
    update_record : (record : list, key_index : int, key : int)
        Updates a record at index "key_index" and time "key" with the contents of "record".  If the index
        does not exist update_record appends the contents of "record" to this list's recarray.
    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, enable=True, path=None, dimensions=None):
        super(MFTransientList, self).__init__(sim_data=sim_data,
                                              structure=structure,
                                              data=None,
                                              enable=enable,
                                              path=path,
                                              dimensions=dimensions)
        self._transient_setup(self._data_storage, mfdata.DataStructureType.recarray)
        self.repeating = True

    def add_transient_key(self, transient_key):
        super(MFTransientList, self).add_transient_key(transient_key)
        self._data_storage[transient_key] = super(MFTransientList, self)._new_storage()

    def data_exists(self, key=None):
        if key is None:
            key = self._current_key
        self.get_data_prep(key)
        return super(MFTransientList, self).data_exists()

    def get_data(self, key=None, apply_mult=False):
        if key is None:
            key = self._current_key
        self.get_data_prep(key)
        return super(MFTransientList, self).get_data(apply_mult=apply_mult)

    def set_data(self, data, key=None, autofill=False):
        if (isinstance(data, dict) or isinstance(data, OrderedDict)) and 'filename' not in data:
            # each item in the dictionary is a list for one stress period
            # the dictionary key is the stress period the list is for
            for key, list_item in data.items():
                self._set_data_prep(list_item, key)
                super(MFTransientList, self).set_data(list_item, autofill=autofill)
        else:
            if key is None:
                # search for a key
                new_key_index = self.structure.first_non_keyword_index()
                if new_key_index is not None and len(data) > new_key_index:
                    key = data[new_key_index]
                else:
                    key = 0
            self._set_data_prep(data, key)
            super(MFTransientList, self).set_data(data)

    def get_file_entry(self, key=0, ext_file_action=ExtFileAction.copy_relative_paths):
        self._get_file_entry_prep(key)
        return super(MFTransientList, self).get_file_entry(ext_file_action=ext_file_action)

    def load(self, first_line, file_handle, block_header, pre_data_comments=None):
        self._load_prep(first_line, file_handle, block_header, pre_data_comments)
        return super(MFTransientList, self).load(first_line, file_handle, pre_data_comments)

    def append_list_as_record(self, record, key=0):
        self._append_list_as_record_prep(record, key)
        super(MFTransientList, self).append_list_as_record(record)

    def update_record(self, record, key_index, key=0):
        self._update_record_prep(key)
        super(MFTransientList, self).update_record(record, key_index)

    def _new_storage(self):
        return OrderedDict()

    def _get_storage_obj(self):
        if self._current_key is None or self._current_key not in self._data_storage:
            return None
        return self._data_storage[self._current_key]


class MFMultipleList(MFTransientList):
    """
    Provides an interface for the user to access and update MODFLOW multiple list data.  This is list data
    that is in the same format as the MFTransientList, but is not time based.

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

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, sim_data, structure, enable=True, path=None, dimensions=None):
        super(MFMultipleList, self).__init__(sim_data=sim_data,
                                            structure=structure,
                                            enable=enable,
                                            path=path,
                                            dimensions=dimensions)
