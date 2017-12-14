"""
mfstructure module.  Contains classes related to package structure


"""
import os
import ast
import keyword
from enum import Enum
from collections import OrderedDict
import numpy as np


class DfnType(Enum):
    common = 1
    sim_name_file = 2
    sim_tdis_file = 3
    ims_file = 4
    exch_file = 5
    gwf_name_file = 6
    gwf_model_file = 7
    gnc_file = 8
    mvr_file = 9
    utl = 10
    unknown = 999


class ReaderType(Enum):
    urword = 1
    readarray = 2
    u1ddbl = 3
    u2ddbl = 4


class ItemType(Enum):
    recarray = 1
    integer = 2


class BlockType(Enum):
    scalers = 1
    single_data_list = 2
    single_data_array = 3
    multi_data_array = 4


class Dfn(object):
    def __init__(self):
        self.file_order = ['sim-nam',  # dfn completed  tex updated
                           'sim-tdis',  # dfn completed  tex updated
                           'exg-gwfgwf',  # dfn completed  tex updated
                           'sln-ims',  # dfn completed  tex updated
                           'gwf-nam',  # dfn completed  tex updated
                           'gwf-dis',  # dfn completed  tex updated
                           'gwf-disv',  # dfn completed  tex updated
                           'gwf-disu',  # dfn completed  tex updated
                           'gwf-ic',  # dfn completed  tex updated
                           'gwf-npf',  # dfn completed  tex updated
                           'gwf-sto',  # dfn completed  tex updated
                           'gwf-hfb',  # dfn completed  tex updated
                           'gwf-chd',  # dfn completed  tex updated
                           'gwf-wel',  # dfn completed  tex updated
                           'gwf-drn',  # dfn completed  tex updated
                           'gwf-riv',  # dfn completed  tex updated
                           'gwf-ghb',  # dfn completed  tex updated
                           'gwf-rch',  # dfn completed  tex updated
                           'gwf-rcha',  # dfn completed  tex updated
                           'gwf-evt',  # dfn completed  tex updated
                           'gwf-evta',  # dfn completed  tex updated
                           'gwf-maw',  # dfn completed  tex updated
                           'gwf-sfr',  # dfn completed  tex updated
                           'gwf-lak',  # dfn completed  tex updated
                           'gwf-uzf',  # dfn completed  tex updated
                           'gwf-mvr',  # dfn completed  tex updated
                           'gwf-gnc',  # dfn completed  tex updated
                           'gwf-oc',  # dfn completed  tex updated
                           'utl-obs',
                           'utl-ts',
                           'utl-tab',
                           'utl-tas']

        # directories
        self.dfndir = os.path.join('.', 'dfn')
        self.texdir = os.path.join('.', 'tex')
        self.mddir = os.path.join('.', 'md')
        self.common = os.path.join(self.dfndir, 'common.dfn')

    def get_file_list(self):
        dfn_path, tail = os.path.split(os.path.realpath(__file__))
        dfn_path = os.path.join(dfn_path, 'dfn')
        # construct list of dfn files to process in the order of file_order
        files = os.listdir(dfn_path)
        for f in files:
            if 'common' in f:
                continue
            package_abbr = os.path.splitext(f)[0]
            if package_abbr not in self.file_order:
                self.file_order.append(package_abbr)
                # raise Exception('File not in file_order: ', f)
        return [fname + '.dfn' for fname in self.file_order if
                fname + '.dfn' in files]


class DfnFile(object):
    def __init__(self, file):
        self.multi_package = {'gwf-mvr': 0, 'exg-gwfgwf': 0, 'gwf-chd': 0,
                              'gwf-rch': 0,
                              'gwf-drn': 0, 'gwf-riv': 0, 'utl-obs': 0,
                              'utl-ts': 0, 'utl-tas': 0}

        dfn_path, tail = os.path.split(os.path.realpath(__file__))
        dfn_path = os.path.join(dfn_path, 'dfn')
        self._file_path = os.path.join(dfn_path, file)
        self.dfn_type = self._file_type(file)
        self.package_type = os.path.splitext(file[4:])[0]
        self.package_group = file[:3]
        self.file = file
        self.dataset_items_needed_dict = {}

    def multi_package_support(self):
        base_file = os.path.splitext(self.file)[0]
        return base_file in self.multi_package

    def dict_by_name(self):
        name_dict = OrderedDict()
        name = None
        dfn_fp = open(self._file_path, 'r')
        for line in dfn_fp:
            if self._valid_line(line):
                arr_line = line.strip().split()
                if arr_line[0] == 'name':
                    name = arr_line[1]
                elif arr_line[0] == 'description' and name is not None:
                    name_dict[name] = ' '.join(arr_line[1:])
        dfn_fp.close()
        return name_dict

    def get_block_structure_dict(self, path, common, model_file):
        block_dict = OrderedDict()
        dataset_items_in_block = {}
        self.dataset_items_needed_dict = {}
        keystring_items_needed_dict = {}
        current_block = None
        dfn_fp = open(self._file_path, 'r')

        for line in dfn_fp:
            if self._valid_line(line):
                # load next data item
                new_data_item_struct = MFDataItemStructure()
                new_data_item_struct.set_value(line, common)
                for next_line in dfn_fp:
                    if self._empty_line(next_line):
                        break
                    if self._valid_line(next_line):
                        new_data_item_struct.set_value(next_line, common)

                # if block does not exist
                if current_block is None or \
                        current_block.name != new_data_item_struct.block_name:
                    # create block
                    current_block = MFBlockStructure(
                        new_data_item_struct.block_name, path, model_file)
                    # put block in block_dict
                    block_dict[current_block.name] = current_block
                    # init dataset item lookup
                    self.dataset_items_needed_dict = {}
                    dataset_items_in_block = {}

                # resolve block type
                if len(current_block.block_header_structure) > 0:
                    if len(current_block.block_header_structure[
                               0].data_item_structures) > 0 and \
                                    current_block.block_header_structure[
                                        0].data_item_structures[
                                        0].type.lower() == 'integer':
                        block_type = BlockType.transient
                    else:
                        block_type = BlockType.multiple
                else:
                    block_type = BlockType.single

                if new_data_item_struct.block_variable:
                    block_dataset_struct = MFDataStructure(
                        new_data_item_struct, model_file)
                    block_dataset_struct.parent_block = current_block
                    self._process_needed_data_items(block_dataset_struct,
                                                    dataset_items_in_block)
                    block_dataset_struct.set_path(
                        path + (new_data_item_struct.block_name,))
                    block_dataset_struct.add_item(new_data_item_struct)
                    current_block.add_dataset(block_dataset_struct, True)
                else:
                    new_data_item_struct.block_type = block_type
                    dataset_items_in_block[
                        new_data_item_struct.name] = new_data_item_struct

                    # if data item belongs to existing dataset(s)
                    item_location_found = False
                    if new_data_item_struct.name in \
                            self.dataset_items_needed_dict:
                        if new_data_item_struct.type == 'record':
                            # record within a record - create a data set in
                            # place of the data item
                            new_data_item_struct = self._new_dataset(
                                new_data_item_struct, current_block,
                                dataset_items_in_block, path,
                                model_file, False)
                            new_data_item_struct.record_within_record = True

                        for dataset in self.dataset_items_needed_dict[
                            new_data_item_struct.name]:
                            item_added = dataset.add_item(new_data_item_struct,
                                                          record=True)
                            item_location_found = item_location_found or \
                                                  item_added
                    # if data item belongs to an existing keystring
                    if new_data_item_struct.name in \
                            keystring_items_needed_dict:
                        new_data_item_struct.set_path(
                            keystring_items_needed_dict[
                                new_data_item_struct.name].path)
                        keystring_items_needed_dict[
                            new_data_item_struct.name].keystring_dict[
                            new_data_item_struct.name] \
                            = new_data_item_struct
                        item_location_found = True

                    if new_data_item_struct.type == 'keystring':
                        # add keystrings to search list
                        for key, val in \
                                new_data_item_struct.keystring_dict.items():
                            keystring_items_needed_dict[
                                key] = new_data_item_struct

                    # if data set does not exist
                    if not item_location_found:
                        self._new_dataset(new_data_item_struct, current_block,
                                          dataset_items_in_block,
                                          path, model_file, True)
                        if current_block.name.upper() == 'SOLUTIONGROUP' and \
                                len(current_block.block_header_structure) == 0:
                            # solution_group a special case for now
                            block_data_item_struct = MFDataItemStructure()
                            block_data_item_struct.name = 'order_num'
                            block_data_item_struct.data_items = ['order_num']
                            block_data_item_struct.type = 'integer'
                            block_data_item_struct.longname = 'order_num'
                            block_data_item_struct.description = \
                                'internal variable to keep track of ' \
                                'solution group number'
                            block_dataset_struct = MFDataStructure(
                                block_data_item_struct, model_file)
                            block_dataset_struct.parent_block = current_block
                            block_dataset_struct.set_path(
                                path + (new_data_item_struct.block_name,))
                            block_dataset_struct.add_item(
                                block_data_item_struct)
                            current_block.add_dataset(block_dataset_struct,
                                                      True)
        dfn_fp.close()
        return block_dict

    def _new_dataset(self, new_data_item_struct, current_block,
                     dataset_items_in_block,
                     path, model_file, add_to_block=True):
        current_dataset_struct = MFDataStructure(new_data_item_struct,
                                                 model_file)
        current_dataset_struct.set_path(
            path + (new_data_item_struct.block_name,))
        self._process_needed_data_items(current_dataset_struct,
                                        dataset_items_in_block)
        if add_to_block:
            # add dataset
            current_block.add_dataset(current_dataset_struct)
            current_dataset_struct.parent_block = current_block
        current_dataset_struct.add_item(new_data_item_struct)
        return current_dataset_struct

    def _process_needed_data_items(self, current_dataset_struct,
                                   dataset_items_in_block):
        # add data items needed to dictionary
        for item_name, val in \
                current_dataset_struct.expected_data_items.items():
            if item_name in dataset_items_in_block:
                current_dataset_struct.add_item(
                    dataset_items_in_block[item_name])
            else:
                if item_name in self.dataset_items_needed_dict:
                    self.dataset_items_needed_dict[item_name].append(
                        current_dataset_struct)
                else:
                    self.dataset_items_needed_dict[item_name] = [
                        current_dataset_struct]

    def _valid_line(self, line):
        if len(line.strip()) > 1 and line[0] != '#':
            return True
        return False

    def _empty_line(self, line):
        if len(line.strip()) <= 1:
            return True
        return False

    def _file_type(self, file_name):
        # determine file type
        if len(file_name) >= 6 and file_name[0:6] == 'common':
            return DfnType.common
        elif file_name[0:3] == 'sim':
            if file_name[4:7] == 'nam':
                return DfnType.sim_name_file
            elif file_name[4:8] == 'tdis':
                return DfnType.sim_tdis_file
            else:
                return DfnType.unknown
        elif file_name[0:3] == 'sln':
            return DfnType.ims_file
        elif file_name[0:3] == 'exg':
            return DfnType.exch_file
        elif file_name[0:3] == 'gwf':
            if file_name[4:7] == 'nam':
                return DfnType.gwf_name_file
            elif file_name[4:7] == 'gnc':
                return DfnType.gnc_file
            elif file_name[4:7] == 'mvr':
                return DfnType.mvr_file
            else:
                return DfnType.gwf_model_file
        elif file_name[0:3] == 'utl':
            return DfnType.utl
        else:
            return DfnType.unknown


class DataType(Enum):
    """
    Types of data that can be found in a package file
    """
    scalar_keyword = 1
    scalar = 2
    array = 3
    array_transient = 4
    list = 5
    list_transient = 6
    list_multiple = 7
    scalar_transient = 8
    scalar_keyword_transient = 9


class BlockType(Enum):
    """
    Types of blocks that can be found in a package file
    """
    single = 1
    multiple = 2
    transient = 3


class StructException(Exception):
    """
    Exception related to the package file structure
    """

    def __init__(self, error, location):
        Exception.__init__(self,
                           "StructException: {} ({})".format(error, location))
        self.location = location


class MFFileParseException(Exception):
    """
    Exception related to parsing MODFLOW input files
    """

    def __init__(self, error):
        Exception.__init__(self, "MFFileParseException: {}".format(error))


class MFInvalidTransientBlockHeaderException(MFFileParseException):
    """
    Exception related to parsing a transient block header
    """

    def __init__(self, error):
        Exception.__init__(self,
                           "MFInvalidTransientBlockHeaderException: {}".format(
                               error))


class MFFileWriteException(Exception):
    """
    Exception related to the writing MODFLOW input files
    """

    def __init__(self, error):
        Exception.__init__(self, "MFFileWriteException: {}".format(error))


class MFDataException(Exception):
    """
    Exception related to MODFLOW input/output data
    """

    def __init__(self, error):
        Exception.__init__(self, "MFDataException: {}".format(error))


class MFFileExistsException(Exception):
    """
    MODFLOW input file requested does not exist
    """

    def __init__(self, error):
        Exception.__init__(self, "MFFileExistsException: {}".format(error))


class ReadAsArraysException(Exception):
    """
    Attempted to load ReadAsArrays package as non-ReadAsArraysPackage
    """

    def __init__(self, error):
        Exception.__init__(self, "ReadAsArraysException: {}".format(error))


class MFDataItemStructure(object):
    """
    Defines the structure of a single MF6 data item in a dfn file

    Attributes
    ----------
    block_name : str
        name of block that data item is in
    name : str
        name of data item
    name_list : list
        list of alternate names for the data item, includes data item's main
        name "name"
    python_name : str
        name of data item referenced in python, with illegal python characters
        removed
    type : str
        type of the data item as it appears in the dfn file
    type_obj : python type
        type of the data item as a python type
    valid_values : list
        list of valid values for the data item.  if empty, this constraint does
        not apply
    data_items : list
        list of data items contained in this data_item, including itself
    in_record : bool
        in_record attribute as appears in dfn file
    tagged : bool
        whether data item is tagged.  if the data item is tagged its name is
        included in the MF6 input file
    just_data : bool
        when just_data is true only data appears in the MF6 input file.
        otherwise, name information appears
    shape : list
        describes the shape of the data
    reader : basestring
        reader that MF6 uses to read the data
    optional : bool
        whether data item is optional or required as part of the MFData in the
        MF6 input file
    longname : str
        long name of the data item
    description : str
        description of the data item
    path : tuple
        a tuple describing the data item's location within the simulation
        (<model>,<package>,<block>,<data>)
    repeating : bool
        whether or not the data item can repeat in the MF6 input file
    block_variable : bool
        if true, this data item is part of the block header
    block_type : BlockType
        whether the block containing this item is a single non-repeating block,
        a multiple repeating block, or a transient repeating block
    keystring_dict : dict
        dictionary containing acceptable keystrings if this data item is of
        type keystring
    is_cellid : bool
        true if this data item is definitely of type cellid
    possible_cellid : bool
        true if this data item may be of type cellid
    ucase : bool
        this data item must be displayed in upper case in the MF6 input file

    Methods
    -------
    remove_cellid : (resolved_shape : list, cellid_size : int)
        removes the cellid size from the shape of a data item
    resolve_shape : (simulation_data : SimulationData)
        resolves the shape of this data item based on the simulation data
        contained in simulation_data
    set_path : (path : tuple)
        sets the path to this data item to path
    get_rec_type : () : object type
        gets the type of object of this data item to be used in a numpy
        recarray
    valid_type : (value : any)
        returns true of value is an acceptable type for this data item

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self):
        self.block_name = None
        self.name = None
        self.name_list = []
        self.python_name = None
        self.type = None
        self.type_obj = None
        self.valid_values = []
        self.data_items = None
        self.in_record = False
        self.tagged = True
        self.just_data = False
        self.shape = []
        self.reader = None
        self.optional = False
        self.longname = None
        self.description = ''
        self.path = None
        self.repeating = False
        self.block_variable = False
        self.block_type = BlockType.single
        self.keystring_dict = {}
        self.is_cellid = False
        self.possible_cellid = False
        self.ucase = False
        self.preserve_case = False

    def set_value(self, line, common):
        arr_line = line.strip().split()
        if len(arr_line) > 1:
            if arr_line[0] == 'block':
                self.block_name = ' '.join(arr_line[1:])
            elif arr_line[0] == 'name':
                self.name = ' '.join(arr_line[1:]).lower()
                self.name_list.append(self.name)
                if len(self.name) >= 6 and self.name[0:6] == 'cellid':
                    self.is_cellid = True
                if self.name and self.name[0:2] == 'id':
                    self.possible_cellid = True
                self.python_name = self.name.replace('-', '_').lower()
                # don't allow name to be a python keyword
                if keyword.iskeyword(self.name):
                    self.python_name = '{}_'.format(self.python_name)
            elif arr_line[0] == 'other_names':
                arr_names = ' '.join(arr_line[1:]).lower().split(',')
                for name in arr_names:
                    self.name_list.append(name)
            elif arr_line[0] == 'type':
                type_line = arr_line[1:]
                assert (len(type_line) > 0)
                self.type = type_line[0].lower()
                if self.type == 'recarray' or self.type == 'record' or \
                        self.type == 'repeating_record' \
                        or self.type == 'keystring':
                    self.data_items = type_line[1:]
                    if self.type == 'keystring':
                        for item in self.data_items:
                            self.keystring_dict[item.lower()] = 0
                else:
                    self.data_items = [self.name]
                self.type_obj = self._get_type()
            elif arr_line[0] == 'valid':
                for value in arr_line[1:]:
                    self.valid_values.append(value)
            elif arr_line[0] == 'in_record':
                self.in_record = self._get_boolean_val(arr_line)
            elif arr_line[0] == 'tagged':
                self.tagged = self._get_boolean_val(arr_line)
            elif arr_line[0] == 'just_data':
                self.just_data = self._get_boolean_val(arr_line)
            elif arr_line[0] == 'shape':
                if len(arr_line) > 1:
                    self.shape = []
                    for dimension in arr_line[1:]:
                        if dimension[-1] != ';':
                            dimension = dimension.replace('(', '')
                            dimension = dimension.replace(')', '')
                            dimension = dimension.replace(',', '')
                            self.shape.append(dimension)
                        else:
                            # only process what is after the last ; which by
                            # convention is the most generalized form of the
                            # shape
                            self.shape = []
                if len(self.shape) > 0:
                    self.repeating = True
            elif arr_line[0] == 'reader':
                self.reader = ' '.join(arr_line[1:])
            elif arr_line[0] == 'optional':
                self.optional = self._get_boolean_val(arr_line)
            elif arr_line[0] == 'longname':
                self.longname = ' '.join(arr_line[1:])
            elif arr_line[0] == 'description':
                if arr_line[1] == 'REPLACE':
                    self.description = self._resolve_common(arr_line, common)
                elif len(arr_line) > 1 and arr_line[1].strip():
                    self.description = ' '.join(arr_line[1:])
            elif arr_line[0] == 'block_variable':
                if len(arr_line) > 1:
                    self.block_variable = bool(arr_line[1])
            elif arr_line[0] == 'ucase':
                if len(arr_line) > 1:
                    self.ucase = bool(arr_line[1])
            elif arr_line[0] == 'preserve_case':
                self.preserve_case = self._get_boolean_val(arr_line)

    @staticmethod
    def remove_cellid(resolved_shape, cellid_size):
        # remove the cellid size from the shape
        for dimension, index in zip(resolved_shape,
                                    range(0, len(resolved_shape))):
            if dimension == cellid_size:
                resolved_shape[index] = 1
                break

    def resolve_shape(self, simulation_data):
        shape_dimensions = []
        parent_path = self.path[:-2]
        for item in self.shape:
            if item == 'naux':
                # shape is number of aux variables
                result = simulation_data.mfdata.find_in_path(parent_path,
                                                             'auxnames')
                if result[0]:
                    shape_dimensions.append(len(result[0].get_data()))
                else:
                    shape_dimensions.append(0)
            else:
                result = simulation_data.mfdata.find_in_path(parent_path, item)
                if result[0]:
                    shape_dimensions.append(int(result[0].get_data()))
                else:
                    shape_dimensions.append(item)
        return shape_dimensions

    @staticmethod
    def _get_boolean_val(bool_option_line):
        if len(bool_option_line) <= 1:
            return False
        if bool_option_line[1].lower() == 'true':
            return True
        return False

    @staticmethod
    def _resolve_common(arr_line, common):
        assert (arr_line[2] in common and len(arr_line) >= 4)
        resolved_str = common[arr_line[2]]
        find_replace_str = ' '.join(arr_line[3:])
        find_replace_dict = ast.literal_eval(find_replace_str)
        for find_str, replace_str in find_replace_dict.items():
            resolved_str = resolved_str.replace(find_str, replace_str)
        # clean up formatting
        resolved_str = resolved_str.replace('\\texttt', '')
        resolved_str = resolved_str.replace('{', '')
        resolved_str = resolved_str.replace('}', '')

        return resolved_str

    def set_path(self, path):
        self.path = path + (self.name,)
        mfstruct = MFStructure()
        for dimension in self.shape:
            dim_path = path + (dimension,)
            if dim_path in mfstruct.dimension_dict:
                mfstruct.dimension_dict[dim_path].append(self)
            else:
                mfstruct.dimension_dict[dim_path] = [self]

    def _get_type(self):
        if self.type == 'float' or self.type == 'double':
            return float
        elif self.type == 'int' or self.type == 'integer':
            return int
        elif self.type == 'constant':
            return bool
        elif self.type == 'string':
            return str
        elif self.type == 'list-defined':
            return str
        return str

    def get_rec_type(self):
        item_type = self.type_obj
        if item_type == str or self.is_cellid:
            return object
        return item_type

    def valid_type(self, value):
        if self.type == 'float':
            if not isinstance(value, float):
                return False
        elif self.type == 'int' or self.type == 'integer':
            if not isinstance(value, int):
                return False
        elif self.type == 'constant':
            if not isinstance(value, bool):
                return False
        elif self.type == 'string':
            if not isinstance(value, str):
                return False
        elif self.type == 'list-defined':
            if not isinstance(value, list):
                return False
        return True


class MFDataStructure(object):
    """
    Defines the structure of a single MF6 data item in a dfn file

    Parameters
    ----------
    data_item : MFDataItemStructure
        base data item associated with this data structure
    model_data : bool
        whether or not this is part of a model

    Attributes
    ----------
    type : str
        type of the data as it appears in the dfn file
    path : tuple
        a tuple describing the data's location within the simulation
        (<model>,<package>,<block>,<data>)
    optional : bool
        whether data is optional or required as part of the MFBlock in the MF6
        input file
    name : str
        name of data item
    name_list : list
        list of alternate names for the data, includes data item's main name
        "name"
    python_name : str
        name of data referenced in python, with illegal python characters
        removed
    longname : str
        long name of the data
    repeating : bool
        whether or not the data can repeat in the MF6 input file
    layered : bool
        whether this data can appear by layer
    num_data_items : int
        number of data item structures contained in this MFDataStructure,
        including itself
    record_within_record : bool
        true if this MFDataStructure is a record within a container
        MFDataStructure
    file_data : bool
        true if data points to a file
    block_type : BlockType
        whether the block containing this data is a single non-repeating block,
        a multiple repeating block, or a transient repeating block
    block_variable : bool
        if true, this data is part of the block header
    model_data : bool
        if true, data is part of a model
    num_optional : int
        number of optional data items
    parent_block : MFBlockStructure
        parent block structure object
    data_item_structures : list
        list of data item structures contained in this MFDataStructure
    expected_data_items : dict
        dictionary of expected data item names for quick lookup
    shape : tuple
        shape of first data item

    Methods
    -------
    get_keywords : () : list
        returns a list of all keywords associated with this data
    supports_aux : () : bool
        returns true of this data supports aux variables
    add_item : (item : MFDataItemStructure, record : bool)
        adds a data item to this MFDataStructure
    set_path : (path : tuple)
        sets the path describing the data's location within the simulation
        (<model>,<package>,<block>,<data>)
    get_datatype : () : DataType
        returns the DataType of this data (array, list, scalar, ...)
    get_record_size : () : int
        gets the number of data items, excluding keyword data items, in this
        MFDataStructure
    all_keywords : () : bool
        returns true of all data items are keywords
    get_type_string : () : str
        returns descriptive string of the data types in this MFDataStructure
    get_description : () : str
        returns a description of the data
    get_type_array : (type_array : list):
        builds an array of data type information in type_array
    get_datum_type : (numpy_type : bool):
        returns the object type of the first data item in this MFDataStructure
        with a standard type.  if numpy_type is true returns the type as a
        numpy type
    get_data_item_types: () : list
        returns a list of object type for every data item in this
        MFDataStructure
    first_non_keyword_index : () : int
        return the index of the first data item in this MFDataStructure that is
        not a keyword

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, data_item, model_data):
        self.type = data_item.type
        self.path = None
        self.optional = data_item.optional
        self.name = data_item.name
        self.name_list = data_item.name_list
        self.python_name = data_item.python_name
        self.longname = data_item.longname
        self.repeating = False
        self.layered = ('nlay' in data_item.shape or
                        'nodes' in data_item.shape)
        self.num_data_items = len(data_item.data_items)
        self.record_within_record = False
        self.file_data = False
        self.block_type = data_item.block_type
        self.block_variable = data_item.block_variable
        self.model_data = model_data
        self.num_optional = 0
        self.parent_block = None

        # self.data_item_structures_dict = OrderedDict()
        self.data_item_structures = []
        self.expected_data_items = OrderedDict()
        self.shape = data_item.shape
        if self.type == 'recarray' or self.type == 'record' or \
                self.type == 'repeating_record':
            # record expected data for later error checking
            for data_item_name in data_item.data_items:
                self.expected_data_items[data_item_name] = len(
                    self.expected_data_items)
        else:
            self.expected_data_items[data_item.name] = len(
                self.expected_data_items)

    def get_item(self, item_name):
        for item in self.data_item_structures:
            if item.name.lower() == item_name.lower():
                return item
        return None

    def get_keywords(self):
        keywords = []
        if self.type == 'recarray' or self.type == 'record' or \
                self.type == 'repeating_record':
            for data_item_struct in self.data_item_structures:
                if data_item_struct.type == 'keyword':
                    if len(keywords) == 0:
                        # create first keyword tuple
                        for name in data_item_struct.name_list:
                            keywords.append((name,))
                    else:
                        # update all keyword tuples with latest keyword found
                        new_keywords = []
                        for keyword_tuple in keywords:
                            for name in data_item_struct.name_list:
                                new_keywords.append(keyword_tuple + (name,))
                        if data_item_struct.optional:
                            keywords = keywords + new_keywords
                        else:
                            keywords = new_keywords
                elif data_item_struct.type == 'keystring':
                    for keyword_item in data_item_struct.data_items:
                        keywords.append((keyword_item,))
                elif len(keywords) == 0:
                    if len(data_item_struct.valid_values) > 0:
                        new_keywords = []
                        # loop through all valid values and append to the end
                        # of each keyword tuple
                        for valid_value in data_item_struct.valid_values:
                            if len(keywords) == 0:
                                new_keywords.append((valid_value,))
                            else:
                                for keyword_tuple in keywords:
                                    new_keywords.append(
                                        keyword_tuple + (valid_value,))
                        keywords = new_keywords
                    else:
                        for name in data_item_struct.name_list:
                            keywords.append((name,))
        else:
            for name in self.name_list:
                keywords.append((name,))
        return keywords

    def supports_aux(self):
        for data_item_struct in self.data_item_structures:
            if data_item_struct.name.lower() == 'aux':
                return True
        return False

    def add_item(self, item, record=False):
        item_added = False
        if item.type != 'recarray' and ((item.type != 'record' and
                                         item.type != 'repeating_record') or
           record == True):
            assert (item.name in self.expected_data_items)
            item.set_path(self.path)
            if len(self.data_item_structures) == 0:
                self.keyword = item.name
            # insert data item into correct location in array
            location = self.expected_data_items[item.name]
            if len(self.data_item_structures) > location:
                # TODO: ask about this condition and remove
                if self.data_item_structures[location] is None:
                    # verify that this is not a placeholder value
                    assert (self.data_item_structures[location] is None)
                    self.file_data = self.file_data or (
                    item.name.lower() == 'filein' or
                    item.name.lower() == 'fileout')
                    # replace placeholder value
                    self.data_item_structures[location] = item
                    item_added = True
            else:
                for index in range(0,
                                   location - len(self.data_item_structures)):
                    # insert placeholder in array
                    self.data_item_structures.append(None)
                self.file_data = self.file_data or (
                item.name.lower() == 'filein' or
                item.name.lower() == 'fileout')
                self.data_item_structures.append(item)
                item_added = True
            self.optional = self.optional and item.optional
            if item.optional:
                self.num_optional += 1

        return item_added

    def set_path(self, path):
        self.path = path + (self.name,)

    def get_datatype(self):
        if self.type == 'recarray':
            if self.block_type != BlockType.single and not self.block_variable:
                if self.block_type == BlockType.transient:
                    return DataType.list_transient
                else:
                    return DataType.list_multiple
            else:
                return DataType.list
        if self.type == 'record' or self.type == 'repeating_record':
            record_size, repeating_data_item = self.get_record_size()
            if (record_size >= 1 and not self.all_keywords()) or \
                    repeating_data_item:
                if self.block_type != BlockType.single and \
                        not self.block_variable:
                    if self.block_type == BlockType.transient:
                        return DataType.list_transient
                    else:
                        return DataType.list_multiple
                else:
                    return DataType.list
            else:
                if self.block_type != BlockType.single and \
                        not self.block_variable:
                    return DataType.scalar_transient
                else:
                    return DataType.scalar
        elif len(self.data_item_structures) > 0 and \
                self.data_item_structures[0].repeating:
            if self.data_item_structures[0].type.lower() == 'string':
                return DataType.list
            else:
                if self.block_type == BlockType.single:
                    return DataType.array
                else:
                    return DataType.array_transient
        elif len(self.data_item_structures) > 0 and \
                self.data_item_structures[0].type.lower() == 'keyword':
            if self.block_type != BlockType.single and not self.block_variable:
                return DataType.scalar_keyword_transient
            else:
                return DataType.scalar_keyword
        else:
            if self.block_type != BlockType.single and not self.block_variable:
                return DataType.scalar_transient
            else:
                return DataType.scalar

    def get_record_size(self):
        count = 0
        repeating = False
        for data_item_structure in self.data_item_structures:
            if data_item_structure.type == 'record':
                count += data_item_structure.get_record_size()[0]
            else:
                if data_item_structure.type != 'keyword' or count > 0:
                    if data_item_structure.repeating:
                        # count repeats as one extra record
                        repeating = True
                    count += 1
        return count, repeating

    def all_keywords(self):
        for data_item_structure in self.data_item_structures:
            if data_item_structure.type == 'record':
                if not data_item_structure.all_keywords():
                    return False
            else:
                if data_item_structure.type != 'keyword':
                    return False
        return True

    def get_type_string(self):
        type_array = []
        self.get_type_array(type_array)
        type_string = ', '.join(type_array)
        type_header = ''
        type_footer = ''
        if len(self.data_item_structures) > 1 or \
                self.data_item_structures[
            0].repeating:
            type_header = '['
            type_footer = ']'
            if self.repeating:
                type_footer = '] ... [{}]'.format(type_string)

        return '{}{}{}'.format(type_header, type_string, type_footer)

    def get_description(self):
        description = ''
        for index, item in enumerate(self.data_item_structures):
            if item is None:
                continue
            if item.type == 'record':
                description = '{}\n{}'.format(description,
                                              item.get_description())
            elif self.display_item(index):
                if len(description.strip()) > 0:
                    description = '{}\n{} : {}'.format(description,
                                                       item.name,
                                                       item.description)
                else:
                    description = '{} : {}'.format(item.name,
                                                   item.description)
        return description.strip()

    def get_type_array(self, type_array):
        for index, item in enumerate(self.data_item_structures):
            if item.type == 'record':
                item.get_type_array(type_array)
            else:
                item_type = item.type
                first_nk_idx = self.first_non_keyword_index()
                if self.display_item(index):
                    # single keyword is type boolean
                    if item_type == 'keyword' and \
                      len(self.data_item_structures) == 1:
                        item_type = 'boolean'
                    if item.is_cellid:
                        item_type = '(integer, ...)'
                    # two keywords
                    if len(self.data_item_structures) == 2 and \
                            first_nk_idx is None:
                        # keyword type is string
                        item_type = 'string'
                    type_array.append('({} : {})'.format(item.name, item_type))

    def display_item(self, item_num):
        item = self.data_item_structures[item_num]
        first_nk_idx = self.first_non_keyword_index()
        # all keywords excluded if there is a non-keyword
        if not (item.type == 'keyword' and first_nk_idx is not None):
            # ignore first keyword if there are two keywords
            if len(self.data_item_structures) == 2 and first_nk_idx is None \
              and item_num == 0:
                return False
            return True
        return False

    def get_datum_type(self, numpy_type=False):
        data_item_types = self.get_data_item_types()
        for var_type in data_item_types:
            if var_type == 'float' or var_type == 'double' or var_type == \
              'int' or var_type == 'integer' or var_type == 'string':
                if numpy_type:
                    if var_type == 'float' or var_type == 'double':
                        return np.float
                    elif var_type == 'int' or var_type == 'integer':
                        return np.int
                    else:
                        return np.object
                else:
                    return var_type
        return None

    def get_data_item_types(self):
        data_item_types = []
        for data_item in self.data_item_structures:
            if data_item.type == 'record':
                # record within a record
                data_item_types += data_item.get_data_item_types()
            else:
                data_item_types.append(data_item.type)
        return data_item_types

    def first_non_keyword_index(self):
        for data_item, index in zip(self.data_item_structures,
                                    range(0, len(self.data_item_structures))):
            if data_item.type != 'keyword':
                return index
        return None


class MFBlockStructure(object):
    """
    Defines the structure of a MF6 block.


    Parameters
    ----------
    name : string
        block name
    path : tuple
        tuple that describes location of block within simulation
        (<model>, <package>, <block>)
    model_block : bool
        true if this block is part of a model

    Attributes
    ----------
    name : string
        block name
    path : tuple
        tuple that describes location of block within simulation
        (<model>, <package>, <block>)
    model_block : bool
        true if this block is part of a model
    data_structures : OrderedDict
        dictionary of data items in this block, with the data item name as
        the key
    block_header_structure : list
        list of data items that are part of this block's "header"

    Methods
    -------
    repeating() : bool
        Returns true if more than one instance of this block can appear in a
        MF6 package file
    add_dataset(dataset : MFDataStructure, block_header_dataset : bool)
        Adds dataset to this block, as a header dataset of block_header_dataset
        is true
    number_non_optional_data() : int
        Returns the number of non-optional non-header data structures in
        this block
    number_non_optional_block_header_data() : int
        Returns the number of non-optional block header data structures in
        this block
    get_data_structure(path : tuple) : MFDataStructure
        Returns the data structure in this block with name defined by path[0].
        If name does not exist, returns None.
    get_all_recarrays() : list
        Returns all data non-header data structures in this block that are of
        type recarray

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(self, name, path, model_block):
        # initialize
        self.data_structures = OrderedDict()
        self.block_header_structure = []
        self.name = name
        self.path = path + (self.name,)
        self.model_block = model_block

    def repeating(self):
        if len(self.block_header_structure) > 0:
            return True
        return False

    def add_dataset(self, dataset, block_header_dataset=False):
        dataset.set_path(self.path)
        if dataset.block_variable:
            self.block_header_structure.append(dataset)
        else:
            self.data_structures[dataset.name] = dataset

    def number_non_optional_data(self):
        num = 0
        for key, data_structure in self.data_structures.items():
            if not data_structure.optional:
                num += 1
        return num

    def number_non_optional_block_header_data(self):
        if len(self.block_header_structure) > 0 and not \
        self.block_header_structure[0].optional:
            return 1
        else:
            return 0

    def get_data_structure(self, path):
        if path[0] in self.data_structures:
            return self.data_structures[path[0]]
        else:
            return None

    def get_all_recarrays(self):
        recarray_list = []
        for ds_key, item in self.data_structures.items():
            if item.type == 'recarray':
                recarray_list.append(item)
        return recarray_list


class MFInputFileStructure(object):
    """
    MODFLOW Input File Stucture class.  Loads file
    structure information for individual input file
    types.


    Parameters
    ----------
    dfn_file : string
        the definition file used to define the structure of this input file
    path : tuple
        path defining the location of the container of this input file
        structure within the overall simulation structure
    common : bool
        is this the common dfn file
    model_file : bool
        this file belongs to a specific model type

    Attributes
    ----------
    valid : boolean
        simulation structure validity
    path : tuple
        path defining the location of this input file structure within the
        overall simulation structure
    read_as_arrays : bool
        if this input file structure is the READASARRAYS version of a package

    Methods
    -------
    is_valid() : bool
        Checks all structures objects within the file for validity
    get_data_structure(path : string)
        Returns a data structure of it exists, otherwise returns None.  Data
        structure type returned is based on the tuple/list "path"

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, dfn_file, path, common, model_file):
        # initialize
        self.valid = True
        self.package_group = dfn_file.package_group
        self.file_type = dfn_file.package_type
        self.dfn_type = dfn_file.dfn_type
        self.package_plot_dictionary = {}
        self.path = path + (self.file_type,)
        # TODO: Get package description from somewhere (tex file?)
        self.description = ''
        self.model_file = model_file  # file belongs to a specific model
        self.read_as_arrays = False

        self.multi_package_support = dfn_file.multi_package_support()
        # self.description = dfn_file.description
        self.blocks = dfn_file.get_block_structure_dict(self.path, common,
                                                        model_file)

    def is_valid(self):
        valid = True
        for block in self.blocks:
            valid = valid and block.is_valid()
        return valid

    def get_data_structure(self, path):
        if isinstance(path, tuple) or isinstance(path, list):
            if path[0] in self.blocks:
                return self.blocks[path[0]].get_data_structure(path[1:])
            else:
                return None
        else:
            for block in self.blocks:
                if path in block.data_structures:
                    return block.data_structures[path]
            return None


class MFModelStructure(object):
    """
    Defines the structure of a MF6 model and its packages

    Parameters
    ----------
    model_type : string
        abbreviation of model type

    Attributes
    ----------
    valid : boolean
        simulation structure validity
    name_file_struct_obj : MFInputFileStructure
        describes the structure of the simulation name file
    package_struct_objs : OrderedDict
        describes the structure of the simulation packages
    model_type : string
        dictionary containing simulation package structure

    Methods
    -------
    add_namefile : (dfn_file : DfnFile, model_file=True : bool)
        Adds a namefile structure object to the model
    add_package(dfn_file : DfnFile, model_file=True : bool)
        Adds a package structure object to the model
    is_valid() : bool
        Checks all structures objects within the model for validity
    get_data_structure(path : string)
        Returns a data structure of it exists, otherwise returns None.  Data
        structure type returned is based on the tuple/list "path"

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, model_type, utl_struct_objs):
        # add name file structure
        self.model_type = model_type
        self.name_file_struct_obj = None
        self.package_struct_objs = OrderedDict()
        self.utl_struct_objs = utl_struct_objs
        self.package_plot_dictionary = {}

    def add_namefile(self, dfn_file, common):
        self.name_file_struct_obj = MFInputFileStructure(dfn_file,
                                                         (self.model_type,),
                                                         common, True)

    def add_package(self, dfn_file, common):
        self.package_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (self.model_type,), common, True)

    def get_package_struct(self, package_type):
        if package_type in self.package_struct_objs:
            return self.package_struct_objs[package_type]
        elif package_type in self.utl_struct_objs:
            return self.utl_struct_objs[package_type]
        else:
            return None

    def is_valid(self):
        valid = True
        for package_struct in self.package_struct_objs:
            valid = valid and package_struct.is_valid()
        return valid

    def get_data_structure(self, path):
        if path[0] in self.package_struct_objs:
            if len(path) > 1:
                return self.package_struct_objs[path[0]].get_data_structure(
                    path[1:])
            else:
                return self.package_struct_objs[path[0]]
        elif path[0] == 'nam':
            if len(path) > 1:
                return self.name_file_struct_obj.get_data_structure(path[1:])
            else:
                return self.name_file_struct_obj
        else:
            return None


class MFSimulationStructure(object):
    """
    Defines the structure of a MF6 simulation and its packages
    and models.

    Parameters
    ----------

    Attributes
    ----------
    name_file_struct_obj : MFInputFileStructure
        describes the structure of the simulation name file
    package_struct_objs : OrderedDict
        describes the structure of the simulation packages
    model_struct_objs : OrderedDict
        describes the structure of the supported model types
    utl_struct_objs : OrderedDict
        describes the structure of the supported utility packages
    common : OrderedDict
        common file information
    model_type : string
        placeholder

    Methods
    -------
    process_dfn : (dfn_file : DfnFile)
        reads in the contents of a dfn file, storing that contents in the
        appropriate object
    add_namefile : (dfn_file : DfnFile, model_file=True : bool)
        Adds a namefile structure object to the simulation
    add_util : (dfn_file : DfnFile)
        Adds a utility package structure object to the simulation
    add_package(dfn_file : DfnFile, model_file=True : bool)
        Adds a package structure object to the simulation
    store_common(dfn_file : DfnFile)
        Stores the contents of the common dfn file
    add_model(model_type : string)
        Adds a model structure object to the simulation
    is_valid() : bool
        Checks all structures objects within the simulation for validity
    get_data_structure(path : string)
        Returns a data structure of it exists, otherwise returns None.  Data
        structure type returned is based on the tuple/list "path"
    tag_read_as_arrays
        Searches through all packages and tags any packages with a name that
        indicates they are the READASARRAYS version of a package.

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self):
        # initialize
        self.name_file_struct_obj = None
        self.package_struct_objs = OrderedDict()
        self.utl_struct_objs = OrderedDict()
        self.model_struct_objs = OrderedDict()
        self.common = None
        self.model_type = ''

    def process_dfn(self, dfn_file):
        if dfn_file.dfn_type == DfnType.common:
            self.store_common(dfn_file)
        elif dfn_file.dfn_type == DfnType.sim_name_file:
            self.add_namefile(dfn_file, False)
        elif dfn_file.dfn_type == DfnType.sim_tdis_file or \
                dfn_file.dfn_type == DfnType.exch_file or \
                dfn_file.dfn_type == DfnType.ims_file:
            self.add_package(dfn_file, False)
        elif dfn_file.dfn_type == DfnType.utl:
            self.add_util(dfn_file)
        elif dfn_file.dfn_type == DfnType.gwf_model_file or \
                dfn_file.dfn_type == DfnType.gwf_name_file or \
                dfn_file.dfn_type == DfnType.gnc_file or \
                dfn_file.dfn_type == DfnType.mvr_file:
            gwf_ver = 'gwf{}'.format(MFStructure().get_version_string())
            if gwf_ver not in self.model_struct_objs:
                self.add_model(gwf_ver)
            if dfn_file.dfn_type == DfnType.gwf_model_file:
                self.model_struct_objs[gwf_ver].add_package(dfn_file,
                                                            self.common)
            elif dfn_file.dfn_type == DfnType.gnc_file or \
                    dfn_file.dfn_type == DfnType.mvr_file:
                # gnc and mvr files belong both on the simulation and model
                # level
                self.model_struct_objs[gwf_ver].add_package(dfn_file,
                                                            self.common)
                self.add_package(dfn_file, False)
            else:
                self.model_struct_objs[gwf_ver].add_namefile(dfn_file,
                                                             self.common)

    def add_namefile(self, dfn_file, model_file=True):
        self.name_file_struct_obj = MFInputFileStructure(dfn_file, (),
                                                         self.common,
                                                         model_file)

    def add_util(self, dfn_file):
        self.utl_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (), self.common, True)

    def add_package(self, dfn_file, model_file=True):
        self.package_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (), self.common, model_file)

    def store_common(self, dfn_file):
        # store common stuff
        self.common = dfn_file.dict_by_name()

    def add_model(self, model_type):
        self.model_struct_objs[model_type] = MFModelStructure(
            model_type, self.utl_struct_objs)

    def is_valid(self):
        valid = True
        for package_struct in self.package_struct_objs:
            valid = valid and package_struct.is_valid()
        for model_struct in self.model_struct_objs:
            valid = valid and model_struct.is_valid()
        return valid

    def get_data_struct_mpd(self, model_type, package_type, data_name):
        package_struct = None
        if model_type in self.model_struct_objs:
            model_struct = self.model_struct_objs[model_type]
            if package_type in model_struct.package_struct_objs:
                package_struct = model_struct.package_struct_objs[package_type]
            elif package_type in model_struct.utl_struct_objs:
                package_struct = model_struct.package_struct_objs[package_type]
        elif package_type in self.package_struct_objs:
            package_struct = self.package_struct_objs[package_type]
        elif package_type in self.utl_struct_objs:
            package_struct = self.utl_struct_objs[package_type]
        if package_struct:
            return package_struct.get_data_structure(data_name)
        else:
            return None

    def get_data_structure(self, path):
        if path[0] in self.package_struct_objs:
            if len(path) > 1:
                return self.package_struct_objs[path[0]].get_data_structure(
                    path[1:])
            else:
                return self.package_struct_objs[path[0]]
        elif path[0] in self.model_struct_objs:
            if len(path) > 1:
                return self.model_struct_objs[path[0]].get_data_structure(
                    path[1:])
            else:
                return self.model_struct_objs[path[0]]
        elif path[0] in self.utl_struct_objs:
            if len(path) > 1:
                return self.utl_struct_objs[path[0]].get_data_structure(
                    path[1:])
            else:
                return self.utl_struct_objs[path[0]]
        elif path[0] == 'nam':
            if len(path) > 1:
                return self.name_file_struct_obj.get_data_structure(path[1:])
            else:
                return self.name_file_struct_obj
        else:
            return None

    def tag_read_as_arrays(self):
        for key, package_struct in self.package_struct_objs.items():
            if key[0:-1] in self.package_struct_objs and key[-1] == 'a':
                package_struct.read_as_arrays = True
        for model_key, model_struct in self.model_struct_objs.items():
            for key, package_struct in \
                    model_struct.package_struct_objs.items():
                if key[0:-1] in model_struct.package_struct_objs and \
                        key[-1] == 'a':
                    package_struct.read_as_arrays = True


class MFStructure(object):
    """
    Singleton class for accessing the contents of the json structure file
    (only one instance of this class can exist, which loads the json file on
    initialization)

    Parameters
    ----------
    mf_version : int
        version of MODFLOW
    valid : bool
        whether the structure information loaded from the dfn files is valid
    sim_struct : MFSimulationStructure
        Object containing file structure for all simulation files
    dimension_dict : dict
        Dictionary mapping paths to dimension information to the dataitem whose
        dimension information is being described
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MFStructure, cls).__new__(cls)

            # Initialize variables
            cls._instance.mf_version = 6
            cls._instance.valid = True
            cls._instance.sim_struct = None
            cls._instance.dimension_dict = {}

            # Read metadata from file
            if not cls._instance.__load_structure():
                cls._instance.valid = False

        return cls._instance

    def get_version_string(self):
        return format(str(self.mf_version))

    def __load_structure(self):
        mf_dfn = Dfn()
        dfn_files = mf_dfn.get_file_list()

        # set up structure classes
        self.sim_struct = MFSimulationStructure()

        # get common
        common_dfn = DfnFile('common.dfn')
        self.sim_struct.process_dfn(common_dfn)

        # process each file
        for file in dfn_files:
            self.sim_struct.process_dfn(DfnFile(file))
        self.sim_struct.tag_read_as_arrays()

        return True
