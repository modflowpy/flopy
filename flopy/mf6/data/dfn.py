import os
from collections import OrderedDict
from enum import Enum
from ..data import mfstructure


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
        self.file_order = ['sim-nam',     # dfn completed  tex updated
                      'sim-tdis',    # dfn completed  tex updated
                      'exg-gwfgwf',  # dfn completed  tex updated
                      'sln-ims',     # dfn completed  tex updated
                      'gwf-nam',     # dfn completed  tex updated
                      'gwf-dis',     # dfn completed  tex updated
                      'gwf-disv',    # dfn completed  tex updated
                      'gwf-disu',    # dfn completed  tex updated
                      'gwf-ic',      # dfn completed  tex updated
                      'gwf-npf',     # dfn completed  tex updated
                      'gwf-sto',     # dfn completed  tex updated
                      'gwf-hfb',     # dfn completed  tex updated
                      'gwf-chd',     # dfn completed  tex updated
                      'gwf-wel',     # dfn completed  tex updated
                      'gwf-drn',     # dfn completed  tex updated
                      'gwf-riv',     # dfn completed  tex updated
                      'gwf-ghb',     # dfn completed  tex updated
                      'gwf-rch',     # dfn completed  tex updated
                      'gwf-rcha',    # dfn completed  tex updated
                      'gwf-evt',     # dfn completed  tex updated
                      'gwf-evta',    # dfn completed  tex updated
                      'gwf-maw',     # dfn completed  tex updated
                      'gwf-sfr',     # dfn completed  tex updated
                      'gwf-lak',     # dfn completed  tex updated
                      'gwf-uzf',     # dfn completed  tex updated
                      'gwf-mvr',     # dfn completed  tex updated
                      'gwf-gnc',     # dfn completed  tex updated
                      'gwf-oc',      # dfn completed  tex updated
                      'utl-obs',
                      'utl-ts',
                      'utl-tab',
                      'utl-tas']


        # directories
        self.dfndir = os.path.join('.', 'dfn')
        self.texdir = os.path.join('.', 'tex')
        self.mddir  = os.path.join('.', 'md')
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
                #raise Exception('File not in file_order: ', f)
        return [fname + '.dfn' for fname in self.file_order if fname + '.dfn' in files]


class DfnFile(object):
    def __init__(self, file):
        self.multi_package = {'gwf-mvr':0, 'exg-gwfgwf':0, 'gwf-chd':0, 'gwf-rch':0,
                              'gwf-drn':0, 'gwf-riv':0, 'utl-obs':0, 'utl-ts':0, 'utl-tas':0}

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
                new_data_item_struct = mfstructure.MFDataItemStructure()
                new_data_item_struct.set_value(line, common)
                for next_line in dfn_fp:
                    if self._empty_line(next_line):
                        break
                    if self._valid_line(next_line):
                        new_data_item_struct.set_value(next_line, common)

                # if block does not exist
                if current_block is None or current_block.name != new_data_item_struct.block_name:
                    # create block
                    current_block = mfstructure.MFBlockStructure(new_data_item_struct.block_name, path, model_file)
                    # put block in block_dict
                    block_dict[current_block.name] = current_block
                    # init dataset item lookup
                    self.dataset_items_needed_dict = {}
                    dataset_items_in_block = {}

                # resolve block type
                if len(current_block.block_header_structure) > 0:
                    if len(current_block.block_header_structure[0].data_item_structures) > 0 and \
                      current_block.block_header_structure[0].data_item_structures[0].type.lower() == 'integer':
                        block_type = mfstructure.BlockType.transient
                    else:
                        block_type = mfstructure.BlockType.multiple
                else:
                    block_type = mfstructure.BlockType.single

                if new_data_item_struct.block_variable:
                    block_dataset_struct = mfstructure.MFDataStructure(new_data_item_struct, model_file)
                    block_dataset_struct.parent_block = current_block
                    self._process_needed_data_items(block_dataset_struct, dataset_items_in_block)
                    block_dataset_struct.set_path(path + (new_data_item_struct.block_name,))
                    block_dataset_struct.add_item(new_data_item_struct)
                    current_block.add_dataset(block_dataset_struct, True)
                else:
                    new_data_item_struct.block_type = block_type
                    dataset_items_in_block[new_data_item_struct.name] = new_data_item_struct

                    # if data item belongs to existing dataset(s)
                    item_location_found = False
                    if new_data_item_struct.name in self.dataset_items_needed_dict:
                        if new_data_item_struct.type == 'record':
                            # record within a record - create a data set in place of the data item
                            new_data_item_struct = self._new_dataset(new_data_item_struct, current_block,
                                                                     dataset_items_in_block, path,
                                                                     model_file, False)
                            new_data_item_struct.record_within_record = True

                        for dataset in self.dataset_items_needed_dict[new_data_item_struct.name]:
                            item_added = dataset.add_item(new_data_item_struct, record=True)
                            item_location_found = item_location_found or item_added
                    # if data item belongs to an existing keystring
                    if new_data_item_struct.name in keystring_items_needed_dict:
                        new_data_item_struct.set_path(keystring_items_needed_dict[new_data_item_struct.name].path)
                        keystring_items_needed_dict[new_data_item_struct.name].keystring_dict[new_data_item_struct.name] \
                            = new_data_item_struct
                        item_location_found = True

                    if new_data_item_struct.type == 'keystring':
                        # add keystrings to search list
                        for key, val in new_data_item_struct.keystring_dict.items():
                            keystring_items_needed_dict[key] = new_data_item_struct

                    # if data set does not exist
                    if not item_location_found:
                        self._new_dataset(new_data_item_struct, current_block, dataset_items_in_block,
                                          path, model_file, True)
                        if current_block.name.upper() == 'SOLUTIONGROUP' and \
                           len(current_block.block_header_structure) == 0:
                            # solution_group a special case for now
                            block_data_item_struct = mfstructure.MFDataItemStructure()
                            block_data_item_struct.name = 'order_num'
                            block_data_item_struct.data_items = ['order_num']
                            block_data_item_struct.type = 'integer'
                            block_data_item_struct.longname = 'order_num'
                            block_data_item_struct.description = 'internal variable to keep track or solution group number'
                            block_dataset_struct = mfstructure.MFDataStructure(block_data_item_struct, model_file)
                            block_dataset_struct.parent_block = current_block
                            block_dataset_struct.set_path(path + (new_data_item_struct.block_name,))
                            block_dataset_struct.add_item(block_data_item_struct)
                            current_block.add_dataset(block_dataset_struct, True)
        dfn_fp.close()
        return block_dict

    def _new_dataset(self, new_data_item_struct, current_block, dataset_items_in_block,
                     path, model_file, add_to_block=True):
        current_dataset_struct = mfstructure.MFDataStructure(new_data_item_struct, model_file)
        current_dataset_struct.set_path(path + (new_data_item_struct.block_name,))
        self._process_needed_data_items(current_dataset_struct, dataset_items_in_block)
        if add_to_block:
            # add dataset
            current_block.add_dataset(current_dataset_struct)
            current_dataset_struct.parent_block = current_block
        current_dataset_struct.add_item(new_data_item_struct)
        return current_dataset_struct

    def _process_needed_data_items(self, current_dataset_struct, dataset_items_in_block):
        # add data items needed to dictionary
        for item_name, val in current_dataset_struct.expected_data_items.items():
            if item_name in dataset_items_in_block:
                current_dataset_struct.add_item(dataset_items_in_block[item_name])
            else:
                if item_name in self.dataset_items_needed_dict:
                    self.dataset_items_needed_dict[item_name].append(current_dataset_struct)
                else:
                    self.dataset_items_needed_dict[item_name] = [current_dataset_struct]


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

