"""
mfstructure module.  Contains classes related to package structure


"""
import os
import traceback
import ast
import keyword
from enum import Enum
from textwrap import TextWrapper
from collections import OrderedDict
import numpy as np
from ..mfbase import PackageContainer, StructException


numeric_index_text = (
    "This argument is an index variable, which means that "
    "it should be treated as zero-based when working with "
    "FloPy and Python. Flopy will automatically subtract "
    "one when loading index variables and add one when "
    "writing index variables."
)


class DfnType(Enum):
    common = 1
    sim_name_file = 2
    sim_tdis_file = 3
    ims_file = 4
    exch_file = 5
    model_name_file = 6
    model_file = 7
    gnc_file = 8
    mvr_file = 9
    utl = 10
    unknown = 999


class Dfn(object):
    """
    Base class for package file definitions

    Attributes
    ----------
    dfndir : path
        folder containing package definition files (dfn)
    common : path
        file containing common information
    multi_package : dict
        contains the names of all packages that are allowed to have multiple
        instances in a model/simulation

    Methods
    -------
    get_file_list : () : list
        returns all of the dfn files found in dfndir.  files are returned in
        a specified order defined in the local variable file_order

    See Also
    --------

    Notes
    -----

    Examples
    ----
    """

    def __init__(self):
        # directories
        self.dfndir = os.path.join(".", "dfn")
        self.common = os.path.join(self.dfndir, "common.dfn")
        # FIX: Transport - multi packages are hard coded
        self.multi_package = {
            "exggwfgwf": 0,
            "gwfchd": 0,
            "gwfwel": 0,
            "gwfdrn": 0,
            "gwfriv": 0,
            "gwfghb": 0,
            "gwfrch": 0,
            "gwfevt": 0,
            "gwfmaw": 0,
            "gwfsfr": 0,
            "gwflak": 0,
            "gwfuzf": 0,
            "lnfcgeo": 0,
            "lnfrgeo": 0,
            "lnfngeo": 0,
            "utlobs": 0,
            "utlts": 0,
            "utltas": 0,
        }

    def get_file_list(self):
        file_order = [
            "sim-nam",  # dfn completed  tex updated
            "sim-tdis",  # dfn completed  tex updated
            "exg-gwfgwf",  # dfn completed  tex updated
            "sln-ims",  # dfn completed  tex updated
            "gwf-nam",  # dfn completed  tex updated
            "gwf-dis",  # dfn completed  tex updated
            "gwf-disv",  # dfn completed  tex updated
            "gwf-disu",  # dfn completed  tex updated
            "lnf-disl",  # dfn completed  tex updated
            "gwf-ic",  # dfn completed  tex updated
            "gwf-npf",  # dfn completed  tex updated
            "gwf-sto",  # dfn completed  tex updated
            "gwf-hfb",  # dfn completed  tex updated
            "gwf-chd",  # dfn completed  tex updated
            "gwf-wel",  # dfn completed  tex updated
            "gwf-drn",  # dfn completed  tex updated
            "gwf-riv",  # dfn completed  tex updated
            "gwf-ghb",  # dfn completed  tex updated
            "gwf-rch",  # dfn completed  tex updated
            "gwf-rcha",  # dfn completed  tex updated
            "gwf-evt",  # dfn completed  tex updated
            "gwf-evta",  # dfn completed  tex updated
            "gwf-maw",  # dfn completed  tex updated
            "gwf-sfr",  # dfn completed  tex updated
            "gwf-lak",  # dfn completed  tex updated
            "gwf-uzf",  # dfn completed  tex updated
            "gwf-mvr",  # dfn completed  tex updated
            "gwf-gnc",  # dfn completed  tex updated
            "gwf-oc",  # dfn completed  tex updated
            "utl-obs",
            "utl-ts",
            "utl-tab",
            "utl-tas",
        ]

        dfn_path, tail = os.path.split(os.path.realpath(__file__))
        dfn_path = os.path.join(dfn_path, "dfn")
        # construct list of dfn files to process in the order of file_order
        files = os.listdir(dfn_path)
        for f in files:
            if "common" in f or "flopy" in f:
                continue
            package_abbr = os.path.splitext(f)[0]
            if package_abbr not in file_order:
                file_order.append(package_abbr)
        return [
            fname + ".dfn" for fname in file_order if fname + ".dfn" in files
        ]

    def _file_type(self, file_name):
        # determine file type
        if len(file_name) >= 6 and file_name[0:6] == "common":
            return DfnType.common, None
        elif file_name[0:3] == "sim":
            if file_name[3:6] == "nam":
                return DfnType.sim_name_file, None
            elif file_name[3:7] == "tdis":
                return DfnType.sim_tdis_file, None
            else:
                return DfnType.unknown, None
        elif file_name[0:3] == "nam":
            return DfnType.sim_name_file, None
        elif file_name[0:4] == "tdis":
            return DfnType.sim_tdis_file, None
        elif file_name[0:3] == "sln" or file_name[0:3] == "ims":
            return DfnType.ims_file, None
        elif file_name[0:3] == "exg":
            return DfnType.exch_file, file_name[3:6]
        elif file_name[0:3] == "utl":
            return DfnType.utl, None
        else:
            model_type = file_name[0:3]
            if file_name[3:6] == "nam":
                return DfnType.model_name_file, model_type
            elif file_name[3:6] == "gnc":
                return DfnType.gnc_file, model_type
            elif file_name[3:6] == "mvr":
                return DfnType.mvr_file, model_type
            else:
                return DfnType.model_file, model_type


class DfnPackage(Dfn):
    """
    Dfn child class that loads dfn information from a list structure stored
    in the auto-built package classes

    Attributes
    ----------
    package : MFPackage
        MFPackage subclass that contains dfn information

    Methods
    -------
    multi_package_support : () : bool
        returns flag for multi-package support
    get_block_structure_dict : (path : tuple, common : bool, model_file :
            bool) : dict
        returns a dictionary of block structure information for the package

    See Also
    --------

    Notes
    -----

    Examples
    ----
    """

    def __init__(self, package):
        super(DfnPackage, self).__init__()
        self.package = package
        self.package_type = package._package_type
        self.dfn_file_name = package.dfn_file_name
        # the package type is always the text after the last -
        package_name = self.package_type.split("-")
        self.package_type = package_name[-1]
        if not isinstance(package_name, str) and len(package_name) > 1:
            self.package_prefix = "".join(package_name[:-1])
        else:
            self.package_prefix = ""
        self.dfn_type, self.model_type = self._file_type(
            self.dfn_file_name.replace("-", "")
        )
        self.dfn_list = package.dfn

    def multi_package_support(self):
        return self.package.package_abbr in self.multi_package

    def get_block_structure_dict(self, path, common, model_file):
        block_dict = OrderedDict()
        dataset_items_in_block = {}
        self.dataset_items_needed_dict = {}
        keystring_items_needed_dict = {}
        current_block = None

        for dfn_entry in self.dfn_list:
            # load next data item
            new_data_item_struct = MFDataItemStructure()
            for next_line in dfn_entry:
                new_data_item_struct.set_value(next_line, common)
            # if block does not exist
            if (
                current_block is None
                or current_block.name != new_data_item_struct.block_name
            ):
                # create block
                current_block = MFBlockStructure(
                    new_data_item_struct.block_name, path, model_file
                )
                # put block in block_dict
                block_dict[current_block.name] = current_block
                # init dataset item lookup
                self.dataset_items_needed_dict = {}
                dataset_items_in_block = {}

            # resolve block type
            if len(current_block.block_header_structure) > 0:
                if (
                    len(
                        current_block.block_header_structure[
                            0
                        ].data_item_structures
                    )
                    > 0
                    and current_block.block_header_structure[0]
                    .data_item_structures[0]
                    .type
                    == DatumType.integer
                ):
                    block_type = BlockType.transient
                else:
                    block_type = BlockType.multiple
            else:
                block_type = BlockType.single

            if new_data_item_struct.block_variable:
                block_dataset_struct = MFDataStructure(
                    new_data_item_struct,
                    model_file,
                    self.package_type,
                    self.dfn_list,
                )
                block_dataset_struct.parent_block = current_block
                self._process_needed_data_items(
                    block_dataset_struct, dataset_items_in_block
                )
                block_dataset_struct.set_path(
                    path + (new_data_item_struct.block_name,)
                )
                block_dataset_struct.add_item(new_data_item_struct)
                current_block.add_dataset(block_dataset_struct)
            else:
                new_data_item_struct.block_type = block_type
                dataset_items_in_block[
                    new_data_item_struct.name
                ] = new_data_item_struct

                # if data item belongs to existing dataset(s)
                item_location_found = False
                if new_data_item_struct.name in self.dataset_items_needed_dict:
                    if new_data_item_struct.type == DatumType.record:
                        # record within a record - create a data set in
                        # place of the data item
                        new_data_item_struct = self._new_dataset(
                            new_data_item_struct,
                            current_block,
                            dataset_items_in_block,
                            path,
                            model_file,
                            False,
                        )
                        new_data_item_struct.record_within_record = True

                    for dataset in self.dataset_items_needed_dict[
                        new_data_item_struct.name
                    ]:
                        item_added = dataset.add_item(
                            new_data_item_struct, record=True
                        )
                        item_location_found = item_location_found or item_added
                # if data item belongs to an existing keystring
                if new_data_item_struct.name in keystring_items_needed_dict:
                    new_data_item_struct.set_path(
                        keystring_items_needed_dict[
                            new_data_item_struct.name
                        ].path
                    )
                    if new_data_item_struct.type == DatumType.record:
                        # record within a keystring - create a data set in
                        # place of the data item
                        new_data_item_struct = self._new_dataset(
                            new_data_item_struct,
                            current_block,
                            dataset_items_in_block,
                            path,
                            model_file,
                            False,
                        )
                    keystring_items_needed_dict[
                        new_data_item_struct.name
                    ].keystring_dict[
                        new_data_item_struct.name
                    ] = new_data_item_struct
                    item_location_found = True

                if new_data_item_struct.type == DatumType.keystring:
                    # add keystrings to search list
                    for (
                        key,
                        val,
                    ) in new_data_item_struct.keystring_dict.items():
                        keystring_items_needed_dict[key] = new_data_item_struct

                # if data set does not exist
                if not item_location_found:
                    self._new_dataset(
                        new_data_item_struct,
                        current_block,
                        dataset_items_in_block,
                        path,
                        model_file,
                        True,
                    )
                    if (
                        current_block.name.upper() == "SOLUTIONGROUP"
                        and len(current_block.block_header_structure) == 0
                    ):
                        # solution_group a special case for now
                        block_data_item_struct = MFDataItemStructure()
                        block_data_item_struct.name = "order_num"
                        block_data_item_struct.data_items = ["order_num"]
                        block_data_item_struct.type = DatumType.integer
                        block_data_item_struct.longname = "order_num"
                        block_data_item_struct.description = (
                            "internal variable to keep track of "
                            "solution group number"
                        )
                        block_dataset_struct = MFDataStructure(
                            block_data_item_struct,
                            model_file,
                            self.package_type,
                            self.dfn_list,
                        )
                        block_dataset_struct.parent_block = current_block
                        block_dataset_struct.set_path(
                            path + (new_data_item_struct.block_name,)
                        )
                        block_dataset_struct.add_item(block_data_item_struct)
                        current_block.add_dataset(block_dataset_struct)
        return block_dict

    def _new_dataset(
        self,
        new_data_item_struct,
        current_block,
        dataset_items_in_block,
        path,
        model_file,
        add_to_block=True,
    ):
        current_dataset_struct = MFDataStructure(
            new_data_item_struct, model_file, self.package_type, self.dfn_list
        )
        current_dataset_struct.set_path(
            path + (new_data_item_struct.block_name,)
        )
        self._process_needed_data_items(
            current_dataset_struct, dataset_items_in_block
        )
        if add_to_block:
            # add dataset
            current_block.add_dataset(current_dataset_struct)
            current_dataset_struct.parent_block = current_block
        current_dataset_struct.add_item(new_data_item_struct)
        return current_dataset_struct

    def _process_needed_data_items(
        self, current_dataset_struct, dataset_items_in_block
    ):
        # add data items needed to dictionary
        for (
            item_name,
            val,
        ) in current_dataset_struct.expected_data_items.items():
            if item_name in dataset_items_in_block:
                current_dataset_struct.add_item(
                    dataset_items_in_block[item_name]
                )
            else:
                if item_name in self.dataset_items_needed_dict:
                    self.dataset_items_needed_dict[item_name].append(
                        current_dataset_struct
                    )
                else:
                    self.dataset_items_needed_dict[item_name] = [
                        current_dataset_struct
                    ]


class DfnFile(Dfn):
    """
    Dfn child class that loads dfn information from a package definition (dfn)
    file

    Attributes
    ----------
    file : str
        name of the file to be loaded

    Methods
    -------
    multi_package_support : () : bool
        returns flag for multi-package support
    dict_by_name : {} : dict
        returns a dictionary of data item descriptions from the dfn file with
        the data item name as the dictionary key
    get_block_structure_dict : (path : tuple, common : bool, model_file :
            bool) : dict
        returns a dictionary of block structure information for the package

    See Also
    --------

    Notes
    -----

    Examples
    ----
    """

    def __init__(self, file):
        super(DfnFile, self).__init__()

        dfn_path, tail = os.path.split(os.path.realpath(__file__))
        dfn_path = os.path.join(dfn_path, "dfn")
        self._file_path = os.path.join(dfn_path, file)
        self.dfn_file_name = file
        self.dfn_type, self.model_type = self._file_type(
            self.dfn_file_name.replace("-", "")
        )
        self.package_type = os.path.splitext(file[4:])[0]
        # the package type is always the text after the last -
        package_name = self.package_type.split("-")
        self.package_type = package_name[-1]
        if not isinstance(package_name, str) and len(package_name) > 1:
            self.package_prefix = "".join(package_name[:-1])
        else:
            self.package_prefix = ""
        self.file = file
        self.dataset_items_needed_dict = {}
        self.dfn_list = []

    def multi_package_support(self):
        base_file = os.path.splitext(self.file)[0]
        base_file = base_file.replace("-", "")
        return base_file in self.multi_package

    def dict_by_name(self):
        name_dict = OrderedDict()
        name = None
        dfn_fp = open(self._file_path, "r")
        for line in dfn_fp:
            if self._valid_line(line):
                arr_line = line.strip().split()
                if arr_line[0] == "name":
                    name = arr_line[1]
                elif arr_line[0] == "description" and name is not None:
                    name_dict[name] = " ".join(arr_line[1:])
        dfn_fp.close()
        return name_dict

    def get_block_structure_dict(self, path, common, model_file):
        self.dfn_list = []
        block_dict = OrderedDict()
        dataset_items_in_block = {}
        self.dataset_items_needed_dict = {}
        keystring_items_needed_dict = {}
        current_block = None
        dfn_fp = open(self._file_path, "r")

        for line in dfn_fp:
            if self._valid_line(line):
                # load next data item
                new_data_item_struct = MFDataItemStructure()
                new_data_item_struct.set_value(line, common)
                self.dfn_list.append([line])
                for next_line in dfn_fp:
                    if self._empty_line(next_line):
                        break
                    if self._valid_line(next_line):
                        new_data_item_struct.set_value(next_line, common)
                        self.dfn_list[-1].append(next_line)

                # if block does not exist
                if (
                    current_block is None
                    or current_block.name != new_data_item_struct.block_name
                ):
                    # create block
                    current_block = MFBlockStructure(
                        new_data_item_struct.block_name, path, model_file
                    )
                    # put block in block_dict
                    block_dict[current_block.name] = current_block
                    # init dataset item lookup
                    self.dataset_items_needed_dict = {}
                    dataset_items_in_block = {}

                # resolve block type
                if len(current_block.block_header_structure) > 0:
                    if (
                        len(
                            current_block.block_header_structure[
                                0
                            ].data_item_structures
                        )
                        > 0
                        and current_block.block_header_structure[0]
                        .data_item_structures[0]
                        .type
                        == DatumType.integer
                    ):
                        block_type = BlockType.transient
                    else:
                        block_type = BlockType.multiple
                else:
                    block_type = BlockType.single

                if new_data_item_struct.block_variable:
                    block_dataset_struct = MFDataStructure(
                        new_data_item_struct,
                        model_file,
                        self.package_type,
                        self.dfn_list,
                    )
                    block_dataset_struct.parent_block = current_block
                    self._process_needed_data_items(
                        block_dataset_struct, dataset_items_in_block
                    )
                    block_dataset_struct.set_path(
                        path + (new_data_item_struct.block_name,)
                    )
                    block_dataset_struct.add_item(
                        new_data_item_struct, False, self.dfn_list
                    )
                    current_block.add_dataset(block_dataset_struct)
                else:
                    new_data_item_struct.block_type = block_type
                    dataset_items_in_block[
                        new_data_item_struct.name
                    ] = new_data_item_struct

                    # if data item belongs to existing dataset(s)
                    item_location_found = False
                    if (
                        new_data_item_struct.name
                        in self.dataset_items_needed_dict
                    ):
                        if new_data_item_struct.type == DatumType.record:
                            # record within a record - create a data set in
                            # place of the data item
                            new_data_item_struct = self._new_dataset(
                                new_data_item_struct,
                                current_block,
                                dataset_items_in_block,
                                path,
                                model_file,
                                False,
                            )
                            new_data_item_struct.record_within_record = True

                        for dataset in self.dataset_items_needed_dict[
                            new_data_item_struct.name
                        ]:
                            item_added = dataset.add_item(
                                new_data_item_struct, True, self.dfn_list
                            )
                            item_location_found = (
                                item_location_found or item_added
                            )
                    # if data item belongs to an existing keystring
                    if (
                        new_data_item_struct.name
                        in keystring_items_needed_dict
                    ):
                        new_data_item_struct.set_path(
                            keystring_items_needed_dict[
                                new_data_item_struct.name
                            ].path
                        )
                        if new_data_item_struct.type == DatumType.record:
                            # record within a keystring - create a data set in
                            # place of the data item
                            new_data_item_struct = self._new_dataset(
                                new_data_item_struct,
                                current_block,
                                dataset_items_in_block,
                                path,
                                model_file,
                                False,
                            )
                        keystring_items_needed_dict[
                            new_data_item_struct.name
                        ].keystring_dict[
                            new_data_item_struct.name
                        ] = new_data_item_struct
                        item_location_found = True

                    if new_data_item_struct.type == DatumType.keystring:
                        # add keystrings to search list
                        for (
                            key,
                            val,
                        ) in new_data_item_struct.keystring_dict.items():
                            keystring_items_needed_dict[
                                key
                            ] = new_data_item_struct

                    # if data set does not exist
                    if not item_location_found:
                        self._new_dataset(
                            new_data_item_struct,
                            current_block,
                            dataset_items_in_block,
                            path,
                            model_file,
                            True,
                        )
                        if (
                            current_block.name.upper() == "SOLUTIONGROUP"
                            and len(current_block.block_header_structure) == 0
                        ):
                            # solution_group a special case for now
                            block_data_item_struct = MFDataItemStructure()
                            block_data_item_struct.name = "order_num"
                            block_data_item_struct.data_items = ["order_num"]
                            block_data_item_struct.type = DatumType.integer
                            block_data_item_struct.longname = "order_num"
                            block_data_item_struct.description = (
                                "internal variable to keep track of "
                                "solution group number"
                            )
                            block_dataset_struct = MFDataStructure(
                                block_data_item_struct,
                                model_file,
                                self.package_type,
                                self.dfn_list,
                            )
                            block_dataset_struct.parent_block = current_block
                            block_dataset_struct.set_path(
                                path + (new_data_item_struct.block_name,)
                            )
                            block_dataset_struct.add_item(
                                block_data_item_struct, False, self.dfn_list
                            )
                            current_block.add_dataset(block_dataset_struct)
        dfn_fp.close()
        return block_dict

    def _new_dataset(
        self,
        new_data_item_struct,
        current_block,
        dataset_items_in_block,
        path,
        model_file,
        add_to_block=True,
    ):
        current_dataset_struct = MFDataStructure(
            new_data_item_struct, model_file, self.package_type, self.dfn_list
        )
        current_dataset_struct.set_path(
            path + (new_data_item_struct.block_name,)
        )
        self._process_needed_data_items(
            current_dataset_struct, dataset_items_in_block
        )
        if add_to_block:
            # add dataset
            current_block.add_dataset(current_dataset_struct)
            current_dataset_struct.parent_block = current_block
        current_dataset_struct.add_item(
            new_data_item_struct, False, self.dfn_list
        )
        return current_dataset_struct

    def _process_needed_data_items(
        self, current_dataset_struct, dataset_items_in_block
    ):
        # add data items needed to dictionary
        for (
            item_name,
            val,
        ) in current_dataset_struct.expected_data_items.items():
            if item_name in dataset_items_in_block:
                current_dataset_struct.add_item(
                    dataset_items_in_block[item_name], False, self.dfn_list
                )
            else:
                if item_name in self.dataset_items_needed_dict:
                    self.dataset_items_needed_dict[item_name].append(
                        current_dataset_struct
                    )
                else:
                    self.dataset_items_needed_dict[item_name] = [
                        current_dataset_struct
                    ]

    def _valid_line(self, line):
        if len(line.strip()) > 1 and line[0] != "#":
            return True
        return False

    def _empty_line(self, line):
        if len(line.strip()) <= 1:
            return True
        return False


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


class DatumType(Enum):
    """
    Types of individual pieces of data
    """

    keyword = 1
    integer = 2
    double_precision = 3
    string = 4
    constant = 5
    list_defined = 6
    keystring = 7
    record = 8
    repeating_record = 9
    recarray = 10


class BlockType(Enum):
    """
    Types of blocks that can be found in a package file
    """

    single = 1
    multiple = 2
    transient = 3


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
    layer_dims : list
        which dimensions in the shape function as layers, if None defaults to
        "layer"
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
    set_path : (path : tuple)
        sets the path to this data item to path
    get_rec_type : () : object type
        gets the type of object of this data item to be used in a numpy
        recarray

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self):
        self.file_name_keywords = {"filein": False, "fileout": False}
        self.contained_keywords = {"file_name": True}
        self.block_name = None
        self.name = None
        self.display_name = None
        self.name_length = None
        self.is_aux = False
        self.is_boundname = False
        self.is_mname = False
        self.name_list = []
        self.python_name = None
        self.type = None
        self.type_string = None
        self.type_obj = None
        self.valid_values = []
        self.data_items = None
        self.in_record = False
        self.tagged = True
        self.just_data = False
        self.shape = []
        self.layer_dims = ["nlay"]
        self.reader = None
        self.optional = False
        self.longname = None
        self.description = ""
        self.path = None
        self.repeating = False
        self.block_variable = False
        self.block_type = BlockType.single
        self.keystring_dict = {}
        self.is_cellid = False
        self.possible_cellid = False
        self.ucase = False
        self.preserve_case = False
        self.default_value = None
        self.numeric_index = False
        self.support_negative_index = False
        self.construct_package = None
        self.construct_data = None
        self.parameter_name = None
        self.one_per_pkg = False
        self.jagged_array = None

    def set_value(self, line, common):
        arr_line = line.strip().split()
        if len(arr_line) > 1:
            if arr_line[0] == "block":
                self.block_name = " ".join(arr_line[1:])
            elif arr_line[0] == "name":
                if self.type == DatumType.keyword:
                    # display keyword names in upper case
                    self.display_name = " ".join(arr_line[1:]).upper()
                else:
                    self.display_name = " ".join(arr_line[1:]).lower()
                self.name = " ".join(arr_line[1:]).lower()
                self.name_list.append(self.name)
                if len(self.name) >= 6 and self.name[0:6] == "cellid":
                    self.is_cellid = True
                if self.name and self.name[0:2] == "id":
                    self.possible_cellid = True
                self.python_name = self.name.replace("-", "_").lower()
                # don't allow name to be a python keyword
                if keyword.iskeyword(self.name):
                    self.python_name = "{}_".format(self.python_name)
                # performance optimizations
                if self.name == "aux":
                    self.is_aux = True
                if self.name == "boundname":
                    self.is_boundname = True
                if self.name[0:5] == "mname":
                    self.is_mname = True
                self.name_length = len(self.name)
            elif arr_line[0] == "other_names":
                arr_names = " ".join(arr_line[1:]).lower().split(",")
                for name in arr_names:
                    self.name_list.append(name)
            elif arr_line[0] == "type":
                if self.support_negative_index:
                    # type already automatically set when
                    # support_negative_index flag is set
                    return
                type_line = arr_line[1:]
                if len(type_line) <= 0:
                    raise StructException(
                        'Data structure "{}" does not have '
                        "a type specified"
                        ".".format(self.name),
                        self.path,
                    )
                self.type_string = type_line[0].lower()
                self.type = self._str_to_enum_type(type_line[0])
                if (
                    self.type == DatumType.recarray
                    or self.type == DatumType.record
                    or self.type == DatumType.repeating_record
                    or self.type == DatumType.keystring
                ):
                    self.data_items = type_line[1:]
                    if self.type == DatumType.keystring:
                        for item in self.data_items:
                            self.keystring_dict[item.lower()] = 0
                else:
                    self.data_items = [self.name]
                self.type_obj = self._get_type()
                if self.type == DatumType.keyword:
                    # display keyword names in upper case
                    if self.display_name is not None:
                        self.display_name = self.display_name.upper()
            elif arr_line[0] == "valid":
                for value in arr_line[1:]:
                    self.valid_values.append(value)
            elif arr_line[0] == "in_record":
                self.in_record = self._get_boolean_val(arr_line)
            elif arr_line[0] == "tagged":
                self.tagged = self._get_boolean_val(arr_line)
            elif arr_line[0] == "just_data":
                self.just_data = self._get_boolean_val(arr_line)
            elif arr_line[0] == "shape":
                if len(arr_line) > 1:
                    self.shape = []
                    for dimension in arr_line[1:]:
                        if dimension[-1] != ";":
                            dimension = dimension.replace("(", "")
                            dimension = dimension.replace(")", "")
                            dimension = dimension.replace(",", "")
                            if dimension[0] == "*":
                                dimension = dimension.replace("*", "")
                                # set as a "layer" dimension
                                self.layer_dims.insert(0, dimension)
                            self.shape.append(dimension)
                        else:
                            # only process what is after the last ; which by
                            # convention is the most generalized form of the
                            # shape
                            self.shape = []
                if len(self.shape) > 0:
                    self.repeating = True
            elif arr_line[0] == "reader":
                self.reader = " ".join(arr_line[1:])
            elif arr_line[0] == "optional":
                self.optional = self._get_boolean_val(arr_line)
            elif arr_line[0] == "longname":
                self.longname = " ".join(arr_line[1:])
            elif arr_line[0] == "description":
                if arr_line[1] == "REPLACE":
                    self.description = self._resolve_common(arr_line, common)
                elif len(arr_line) > 1 and arr_line[1].strip():
                    self.description = " ".join(arr_line[1:])

                # clean self.description
                replace_pairs = [
                    ("``", '"'),  # double quotes
                    ("''", '"'),
                    ("`", "'"),  # single quotes
                    ("~", " "),  # non-breaking space
                    (r"\mf", "MODFLOW 6"),
                    (r"\citep{konikow2009}", "(Konikow et al., 2009)"),
                    (r"\citep{hill1990preconditioned}", "(Hill, 1990)"),
                    (r"\ref{table:ftype}", "in mf6io.pdf"),
                    (r"\ref{table:gwf-obstypetable}", "in mf6io.pdf"),
                ]
                for s1, s2 in replace_pairs:
                    if s1 in self.description:
                        self.description = self.description.replace(s1, s2)

                # massage latex equations
                self.description = self.description.replace("$<$", "<")
                self.description = self.description.replace("$>$", ">")
                if "$" in self.description:
                    descsplit = self.description.split("$")
                    mylist = [
                        i.replace("\\", "")
                        + ":math:`"
                        + j.replace("\\", "\\\\")
                        + "`"
                        for i, j in zip(descsplit[::2], descsplit[1::2])
                    ]
                    mylist.append(descsplit[-1].replace("\\", ""))
                    self.description = "".join(mylist)
                else:
                    self.description = self.description.replace("\\", "")
            elif arr_line[0] == "block_variable":
                if len(arr_line) > 1:
                    self.block_variable = bool(arr_line[1])
            elif arr_line[0] == "ucase":
                if len(arr_line) > 1:
                    self.ucase = bool(arr_line[1])
            elif arr_line[0] == "preserve_case":
                self.preserve_case = self._get_boolean_val(arr_line)
            elif arr_line[0] == "default_value":
                self.default_value = " ".join(arr_line[1:])
            elif arr_line[0] == "numeric_index":
                self.numeric_index = self._get_boolean_val(arr_line)
            elif arr_line[0] == "support_negative_index":
                self.support_negative_index = self._get_boolean_val(arr_line)
                # must be double precision to support 0 and -0
                self.type_string = "double_precision"
                self.type = self._str_to_enum_type(self.type_string)
                self.type_obj = self._get_type()
            elif arr_line[0] == "construct_package":
                self.construct_package = arr_line[1]
            elif arr_line[0] == "construct_data":
                self.construct_data = arr_line[1]
            elif arr_line[0] == "parameter_name":
                self.parameter_name = arr_line[1]
            elif arr_line[0] == "one_per_pkg":
                self.one_per_pkg = bool(arr_line[1])
            elif arr_line[0] == "jagged_array":
                self.jagged_array = arr_line[1]

    def get_type_string(self):
        return "[{}]".format(self.type_string)

    def get_description(self, line_size, initial_indent, level_indent):
        item_desc = "* {} ({}) {}".format(
            self.name, self.type_string, self.description
        )
        if self.numeric_index or self.is_cellid:
            # append zero-based index text
            item_desc = "{} {}".format(item_desc, numeric_index_text)
        twr = TextWrapper(
            width=line_size,
            initial_indent=initial_indent,
            drop_whitespace=True,
            subsequent_indent="  {}".format(initial_indent),
        )
        item_desc = "\n".join(twr.wrap(item_desc))
        return item_desc

    def get_doc_string(self, line_size, initial_indent, level_indent):
        description = self.get_description(
            line_size, initial_indent + level_indent, level_indent
        )
        param_doc_string = "{} : {}".format(
            self.python_name, self.get_type_string()
        )
        twr = TextWrapper(
            width=line_size,
            initial_indent=initial_indent,
            subsequent_indent="  {}".format(initial_indent),
            drop_whitespace=True,
        )
        param_doc_string = "\n".join(twr.wrap(param_doc_string))
        param_doc_string = "{}\n{}".format(param_doc_string, description)
        return param_doc_string

    def get_keystring_desc(self, line_size, initial_indent, level_indent):
        if self.type != DatumType.keystring:
            raise StructException(
                'Can not get keystring description for "{}" '
                "because it is not a keystring"
                ".".format(self.name),
                self.path,
            )

        # get description of keystring elements
        description = ""
        for key, item in self.keystring_dict.items():
            if description:
                description = "{}\n".format(description)
            description = "{}{}".format(
                description,
                item.get_doc_string(line_size, initial_indent, level_indent),
            )
        return description

    def indicates_file_name(self):
        if self.name.lower() in self.file_name_keywords:
            return True
        for key, item in self.contained_keywords.items():
            if self.name.lower().find(key) != -1:
                return True
        return False

    def is_file_name(self):
        if (
            self.name.lower() in self.file_name_keywords
            and self.file_name_keywords[self.name.lower()] == True
        ):
            return True
        for key, item in self.contained_keywords.items():
            if self.name.lower().find(key) != -1 and item == True:
                return True
        return False

    @staticmethod
    def remove_cellid(resolved_shape, cellid_size):
        # remove the cellid size from the shape
        for dimension, index in zip(
            resolved_shape, range(0, len(resolved_shape))
        ):
            if dimension == cellid_size:
                resolved_shape[index] = 1
                break

    @staticmethod
    def _get_boolean_val(bool_option_line):
        if len(bool_option_line) <= 1:
            return False
        if bool_option_line[1].lower() == "true":
            return True
        return False

    @staticmethod
    def _find_close_bracket(arr_line):
        for index, word in enumerate(arr_line):
            word = word.strip()
            if len(word) > 0 and word[-1] == "}":
                return index
        return None

    @staticmethod
    def _resolve_common(arr_line, common):
        if common is None:
            return arr_line
        if not (arr_line[2] in common and len(arr_line) >= 4):
            raise StructException(
                'Could not find line "{}" in common dfn' ".".format(arr_line)
            )
        close_bracket_loc = MFDataItemStructure._find_close_bracket(
            arr_line[2:]
        )
        resolved_str = common[arr_line[2]]
        if close_bracket_loc is None:
            find_replace_str = " ".join(arr_line[3:])
        else:
            close_bracket_loc += 3
            find_replace_str = " ".join(arr_line[3:close_bracket_loc])
        find_replace_dict = ast.literal_eval(find_replace_str)
        for find_str, replace_str in find_replace_dict.items():
            resolved_str = resolved_str.replace(find_str, replace_str)
        # clean up formatting
        resolved_str = resolved_str.replace("\\texttt", "")
        resolved_str = resolved_str.replace("{", "")
        resolved_str = resolved_str.replace("}", "")

        return resolved_str

    def set_path(self, path):
        self.path = path + (self.name,)
        mfstruct = MFStructure(True)
        for dimension in self.shape:
            dim_path = path + (dimension,)
            if dim_path in mfstruct.dimension_dict:
                mfstruct.dimension_dict[dim_path].append(self)
            else:
                mfstruct.dimension_dict[dim_path] = [self]

    def _get_type(self):
        if self.type == DatumType.double_precision:
            return float
        elif self.type == DatumType.integer:
            return int
        elif self.type == DatumType.constant:
            return bool
        elif self.type == DatumType.string:
            return str
        elif self.type == DatumType.list_defined:
            return str
        return str

    def _str_to_enum_type(self, type_string):
        if type_string.lower() == "keyword":
            return DatumType.keyword
        elif type_string.lower() == "integer":
            return DatumType.integer
        elif (
            type_string.lower() == "double_precision"
            or type_string.lower() == "double"
        ):
            return DatumType.double_precision
        elif type_string.lower() == "string":
            return DatumType.string
        elif type_string.lower() == "constant":
            return DatumType.constant
        elif type_string.lower() == "list-defined":
            return DatumType.list_defined
        elif type_string.lower() == "keystring":
            return DatumType.keystring
        elif type_string.lower() == "record":
            return DatumType.record
        elif type_string.lower() == "recarray":
            return DatumType.recarray
        elif type_string.lower() == "repeating_record":
            return DatumType.repeating_record
        else:
            exc_text = 'Data item type "{}" not supported.'.format(type_string)
            raise StructException(exc_text, self.path)

    def get_rec_type(self):
        item_type = self.type_obj
        if item_type == str or self.is_cellid:
            return object
        return item_type


class MFDataStructure(object):
    """
    Defines the structure of a single MF6 data item in a dfn file

    Parameters
    ----------
    data_item : MFDataItemStructure
        base data item associated with this data structure
    model_data : bool
        whether or not this is part of a model
    package_type : str
        abbreviated package type

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
    get_min_record_entries : () : int
        gets the minimum number of entries, as entered in a package file,
        for a single record. excludes optional data items
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

    def __init__(self, data_item, model_data, package_type, dfn_list):
        self.type = data_item.type
        self.package_type = package_type
        self.path = None
        self.optional = data_item.optional
        self.name = data_item.name
        self.block_name = data_item.block_name
        self.name_length = len(self.name)
        self.is_aux = data_item.is_aux
        self.is_boundname = data_item.is_boundname
        self.name_list = data_item.name_list
        self.python_name = data_item.python_name
        self.longname = data_item.longname
        self.default_value = data_item.default_value
        self.repeating = False
        self.layered = (
            "nlay" in data_item.shape
            or "nodes" in data_item.shape
            or len(data_item.layer_dims) > 1
        )
        self.num_data_items = len(data_item.data_items)
        self.record_within_record = False
        self.file_data = False
        self.block_type = data_item.block_type
        self.block_variable = data_item.block_variable
        self.model_data = model_data
        self.num_optional = 0
        self.parent_block = None
        self._fpmerge_data_item(data_item, dfn_list)
        self.construct_package = data_item.construct_package
        self.construct_data = data_item.construct_data
        self.parameter_name = data_item.parameter_name
        self.one_per_pkg = data_item.one_per_pkg

        # self.data_item_structures_dict = OrderedDict()
        self.data_item_structures = []
        self.expected_data_items = OrderedDict()
        self.shape = data_item.shape
        if (
            self.type == DatumType.recarray
            or self.type == DatumType.record
            or self.type == DatumType.repeating_record
        ):
            # record expected data for later error checking
            for data_item_name in data_item.data_items:
                self.expected_data_items[data_item_name] = len(
                    self.expected_data_items
                )
        else:
            self.expected_data_items[data_item.name] = len(
                self.expected_data_items
            )

    @property
    def is_mname(self):
        for item in self.data_item_structures:
            if item.is_mname:
                return True
        return False

    def get_item(self, item_name):
        for item in self.data_item_structures:
            if item.name.lower() == item_name.lower():
                return item
        return None

    def get_keywords(self):
        keywords = []
        if (
            self.type == DatumType.recarray
            or self.type == DatumType.record
            or self.type == DatumType.repeating_record
        ):
            for data_item_struct in self.data_item_structures:
                if data_item_struct.type == DatumType.keyword:
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
                elif data_item_struct.type == DatumType.keystring:
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
                                        keyword_tuple + (valid_value,)
                                    )
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
            if data_item_struct.name.lower() == "aux":
                return True
        return False

    def add_item(self, item, record=False, dfn_list=None):
        item_added = False
        if item.type != DatumType.recarray and (
            (
                item.type != DatumType.record
                and item.type != DatumType.repeating_record
            )
            or record == True
        ):
            if item.name not in self.expected_data_items:
                raise StructException(
                    'Could not find data item "{}" in '
                    "expected data items of data structure "
                    "{}.".format(item.name, self.name),
                    self.path,
                )
            item.set_path(self.path)
            if len(self.data_item_structures) == 0:
                self.keyword = item.name
            # insert data item into correct location in array
            location = self.expected_data_items[item.name]
            if len(self.data_item_structures) > location:
                # TODO: ask about this condition and remove
                if self.data_item_structures[location] is None:
                    # verify that this is not a placeholder value
                    if self.data_item_structures[location] is not None:
                        raise StructException(
                            'Data structure "{}" already '
                            'has the item named "{}"'
                            ".".format(self.name, item.name),
                            self.path,
                        )
                    if isinstance(item, MFDataItemStructure):
                        self.file_data = (
                            self.file_data or item.indicates_file_name()
                        )
                    # replace placeholder value
                    self.data_item_structures[location] = item
                    item_added = True
            else:
                for index in range(
                    0, location - len(self.data_item_structures)
                ):
                    # insert placeholder in array
                    self.data_item_structures.append(None)
                if isinstance(item, MFDataItemStructure):
                    self.file_data = (
                        self.file_data or item.indicates_file_name()
                    )
                self.data_item_structures.append(item)
                item_added = True
            self.optional = self.optional and item.optional
            if item.optional:
                self.num_optional += 1
        if item_added:
            self._fpmerge_data_item(item, dfn_list)
        return item_added

    def _fpmerge_data_item(self, item, dfn_list):
        mfstruct = MFStructure()
        # check for flopy-specific dfn data
        if item.name.lower() in mfstruct.flopy_dict:
            # read flopy-specific dfn data
            for name, value in mfstruct.flopy_dict[item.name.lower()].items():
                line = "{} {}".format(name, value)
                item.set_value(line, None)
                if dfn_list is not None:
                    dfn_list[-1].append(line)

    def set_path(self, path):
        self.path = path + (self.name,)

    def get_datatype(self):
        if self.type == DatumType.recarray:
            if self.block_type != BlockType.single and not self.block_variable:
                if self.block_type == BlockType.transient:
                    return DataType.list_transient
                else:
                    return DataType.list_multiple
            else:
                return DataType.list
        if (
            self.type == DatumType.record
            or self.type == DatumType.repeating_record
        ):
            record_size, repeating_data_item = self.get_record_size()
            if (
                record_size >= 1 and not self.all_keywords()
            ) or repeating_data_item:
                if (
                    self.block_type != BlockType.single
                    and not self.block_variable
                ):
                    if self.block_type == BlockType.transient:
                        return DataType.list_transient
                    else:
                        return DataType.list_multiple
                else:
                    return DataType.list
            else:
                if (
                    self.block_type != BlockType.single
                    and not self.block_variable
                ):
                    return DataType.scalar_transient
                else:
                    return DataType.scalar
        elif (
            len(self.data_item_structures) > 0
            and self.data_item_structures[0].repeating
        ):
            if self.data_item_structures[0].type == DatumType.string:
                return DataType.list
            else:
                if self.block_type == BlockType.single:
                    return DataType.array
                else:
                    return DataType.array_transient
        elif (
            len(self.data_item_structures) > 0
            and self.data_item_structures[0].type == DatumType.keyword
        ):
            if self.block_type != BlockType.single and not self.block_variable:
                return DataType.scalar_keyword_transient
            else:
                return DataType.scalar_keyword
        else:
            if self.block_type != BlockType.single and not self.block_variable:
                return DataType.scalar_transient
            else:
                return DataType.scalar

    def is_mult_or_trans(self):
        data_type = self.get_datatype()
        if (
            data_type == DataType.scalar_keyword_transient
            or data_type == DataType.array_transient
            or data_type == DataType.list_transient
            or data_type == DataType.list_multiple
        ):
            return True
        return False

    def get_min_record_entries(self):
        count = 0
        for data_item_structure in self.data_item_structures:
            if not data_item_structure.optional:
                if data_item_structure.type == DatumType.record:
                    count += data_item_structure.get_record_size()[0]
                else:
                    if data_item_structure.type != DatumType.keyword:
                        count += 1
        return count

    def get_record_size(self):
        count = 0
        repeating = False
        for data_item_structure in self.data_item_structures:
            if data_item_structure.type == DatumType.record:
                count += data_item_structure.get_record_size()[0]
            else:
                if data_item_structure.type != DatumType.keyword or count > 0:
                    if data_item_structure.repeating:
                        # count repeats as one extra record
                        repeating = True
                    count += 1
        return count, repeating

    def all_keywords(self):
        for data_item_structure in self.data_item_structures:
            if data_item_structure.type == DatumType.record:
                if not data_item_structure.all_keywords():
                    return False
            else:
                if data_item_structure.type != DatumType.keyword:
                    return False
        return True

    def get_type_string(self):
        type_array = []
        self.get_docstring_type_array(type_array)
        type_string = ", ".join(type_array)
        type_header = ""
        type_footer = ""
        if (
            len(self.data_item_structures) > 1
            or self.data_item_structures[0].repeating
        ):
            type_header = "["
            type_footer = "]"
            if self.repeating:
                type_footer = "] ... [{}]".format(type_string)

        return "{}{}{}".format(type_header, type_string, type_footer)

    def get_docstring_type_array(self, type_array):
        for index, item in enumerate(self.data_item_structures):
            if item.type == DatumType.record:
                item.get_docstring_type_array(type_array)
            else:
                if self.display_item(index):
                    if (
                        self.type == DatumType.recarray
                        or self.type == DatumType.record
                        or self.type == DatumType.repeating_record
                    ):
                        type_array.append("{}".format(item.name))
                    else:
                        type_array.append(
                            "{}".format(self._resolve_item_type(item))
                        )

    def get_description(
        self, line_size=79, initial_indent="        ", level_indent="    "
    ):
        type_array = []
        self.get_type_array(type_array)
        description = ""
        for datastr, index, itype in type_array:
            item = datastr.data_item_structures[index]
            if item is None:
                continue
            if item.type == DatumType.record:
                item_desc = item.get_description(
                    line_size, initial_indent + level_indent, level_indent
                )
                description = "{}\n{}".format(description, item_desc)
            elif datastr.display_item(index):
                if len(description.strip()) > 0:
                    description = "{}\n".format(description)
                item_desc = item.description
                if item.numeric_index or item.is_cellid:
                    # append zero-based index text
                    item_desc = "{} {}".format(item_desc, numeric_index_text)

                item_desc = "* {} ({}) {}".format(item.name, itype, item_desc)
                twr = TextWrapper(
                    width=line_size,
                    initial_indent=initial_indent,
                    subsequent_indent="  {}".format(initial_indent),
                )
                item_desc = "\n".join(twr.wrap(item_desc))
                description = "{}{}".format(description, item_desc)
                if item.type == DatumType.keystring:
                    keystr_desc = item.get_keystring_desc(
                        line_size, initial_indent + level_indent, level_indent
                    )
                    description = "{}\n{}".format(description, keystr_desc)
        return description

    def get_subpackage_description(
        self, line_size=79, initial_indent="        ", level_indent="    "
    ):
        item_desc = (
            "* Contains data for the {} package. Data can be "
            "stored in a dictionary containing data for the {} "
            "package with variable names as keys and package data as "
            "values. Data just for the {} variable is also "
            "acceptable. See {} package documentation for more "
            "information"
            ".".format(
                self.construct_package,
                self.construct_package,
                self.parameter_name,
                self.construct_package,
            )
        )
        twr = TextWrapper(
            width=line_size,
            initial_indent=initial_indent,
            subsequent_indent="  {}".format(initial_indent),
        )
        return "\n".join(twr.wrap(item_desc))

    def get_doc_string(
        self, line_size=79, initial_indent="    ", level_indent="    "
    ):
        if self.parameter_name is not None:
            description = self.get_subpackage_description(
                line_size, initial_indent + level_indent, level_indent
            )
            var_name = self.parameter_name
            type_name = "{}varname:data{} or {} data".format(
                "{", "}", self.construct_data
            )
        else:
            description = self.get_description(
                line_size, initial_indent + level_indent, level_indent
            )
            var_name = self.python_name
            type_name = self.get_type_string()

        param_doc_string = "{} : {}".format(var_name, type_name)
        twr = TextWrapper(
            width=line_size,
            initial_indent=initial_indent,
            subsequent_indent="  {}".format(initial_indent),
        )
        param_doc_string = "\n".join(twr.wrap(param_doc_string))
        param_doc_string = "{}\n{}".format(param_doc_string, description)
        return param_doc_string

    def get_type_array(self, type_array):
        for index, item in enumerate(self.data_item_structures):
            if item.type == DatumType.record:
                item.get_type_array(type_array)
            else:
                if self.display_item(index):
                    type_array.append(
                        (
                            self,
                            index,
                            "{}".format(self._resolve_item_type(item)),
                        )
                    )

    def _resolve_item_type(self, item):
        item_type = item.type_string
        first_nk_idx = self.first_non_keyword_index()
        # single keyword is type boolean
        if item_type == "keyword" and len(self.data_item_structures) == 1:
            item_type = "boolean"
        if item.is_cellid:
            item_type = "(integer, ...)"
        # two keywords
        if len(self.data_item_structures) == 2 and first_nk_idx is None:
            # keyword type is string
            item_type = "string"
        return item_type

    def display_item(self, item_num):
        item = self.data_item_structures[item_num]
        first_nk_idx = self.first_non_keyword_index()
        # all keywords excluded if there is a non-keyword
        if not (item.type == DatumType.keyword and first_nk_idx is not None):
            # ignore first keyword if there are two keywords
            if (
                len(self.data_item_structures) == 2
                and first_nk_idx is None
                and item_num == 0
            ):
                return False
            return True
        return False

    def get_datum_type(self, numpy_type=False, return_enum_type=False):
        data_item_types = self.get_data_item_types()
        for var_type in data_item_types:
            if (
                var_type[0] == DatumType.double_precision
                or var_type[0] == DatumType.integer
                or var_type[0] == DatumType.string
            ):
                if return_enum_type:
                    return var_type[0]
                else:
                    if numpy_type:
                        if var_type[0] == DatumType.double_precision:
                            return np.float64
                        elif var_type[0] == DatumType.integer:
                            return np.int32
                        else:
                            return object
                    else:
                        return var_type[2]
        return None

    def get_data_item_types(self):
        data_item_types = []
        for data_item in self.data_item_structures:
            if data_item.type == DatumType.record:
                # record within a record
                data_item_types += data_item.get_data_item_types()
            else:
                data_item_types.append(
                    [data_item.type, data_item.type_string, data_item.type_obj]
                )
        return data_item_types

    def first_non_keyword_index(self):
        for data_item, index in zip(
            self.data_item_structures, range(0, len(self.data_item_structures))
        ):
            if data_item.type != DatumType.keyword:
                return index
        return None

    def get_model(self):
        if self.model_data:
            if len(self.path) >= 1:
                return self.path[0]
        return None

    def get_package(self):
        if self.model_data:
            if len(self.path) >= 2:
                return self.path[1]
        else:
            if len(self.path) >= 1:
                return self.path[0]
        return ""


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

    def add_dataset(self, dataset):
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
        if (
            len(self.block_header_structure) > 0
            and not self.block_header_structure[0].optional
        ):
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
            if item.type == DatumType.recarray:
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
        self.file_type = dfn_file.package_type
        self.file_prefix = dfn_file.package_prefix
        self.dfn_type = dfn_file.dfn_type
        self.dfn_file_name = dfn_file.dfn_file_name
        self.description = ""
        self.path = path + (self.file_type,)
        self.model_file = model_file  # file belongs to a specific model
        self.read_as_arrays = False

        self.multi_package_support = dfn_file.multi_package_support()
        self.blocks = dfn_file.get_block_structure_dict(
            self.path, common, model_file
        )
        self.dfn_list = dfn_file.dfn_list

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

    def add_namefile(self, dfn_file, common):
        self.name_file_struct_obj = MFInputFileStructure(
            dfn_file, (self.model_type,), common, True
        )

    def add_package(self, dfn_file, common):
        self.package_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (self.model_type,), common, True
        )

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
                    path[1:]
                )
            else:
                return self.package_struct_objs[path[0]]
        elif path[0] == "nam":
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
        self.model_type = ""

    @property
    def model_types(self):
        model_type_list = []
        for model in self.model_struct_objs.values():
            model_type_list.append(model.model_type[:-1])
        return model_type_list

    def process_dfn(self, dfn_file):
        if dfn_file.dfn_type == DfnType.common:
            self.store_common(dfn_file)
        elif dfn_file.dfn_type == DfnType.sim_name_file:
            self.add_namefile(dfn_file, False)
        elif (
            dfn_file.dfn_type == DfnType.sim_tdis_file
            or dfn_file.dfn_type == DfnType.exch_file
            or dfn_file.dfn_type == DfnType.ims_file
        ):
            self.add_package(dfn_file, False)
        elif dfn_file.dfn_type == DfnType.utl:
            self.add_util(dfn_file)
        elif (
            dfn_file.dfn_type == DfnType.model_file
            or dfn_file.dfn_type == DfnType.model_name_file
            or dfn_file.dfn_type == DfnType.gnc_file
            or dfn_file.dfn_type == DfnType.mvr_file
        ):
            model_ver = "{}{}".format(
                dfn_file.model_type, MFStructure(True).get_version_string()
            )
            if model_ver not in self.model_struct_objs:
                self.add_model(model_ver)
            if dfn_file.dfn_type == DfnType.model_file:
                self.model_struct_objs[model_ver].add_package(
                    dfn_file, self.common
                )
            elif (
                dfn_file.dfn_type == DfnType.gnc_file
                or dfn_file.dfn_type == DfnType.mvr_file
            ):
                # gnc and mvr files belong both on the simulation and model
                # level
                self.model_struct_objs[model_ver].add_package(
                    dfn_file, self.common
                )
                self.add_package(dfn_file, False)
            else:
                self.model_struct_objs[model_ver].add_namefile(
                    dfn_file, self.common
                )

    def add_namefile(self, dfn_file, model_file=True):
        self.name_file_struct_obj = MFInputFileStructure(
            dfn_file, (), self.common, model_file
        )

    def add_util(self, dfn_file):
        self.utl_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (), self.common, True
        )

    def add_package(self, dfn_file, model_file=True):
        self.package_struct_objs[dfn_file.package_type] = MFInputFileStructure(
            dfn_file, (), self.common, model_file
        )

    def store_common(self, dfn_file):
        # store common stuff
        self.common = dfn_file.dict_by_name()

    def add_model(self, model_type):
        self.model_struct_objs[model_type] = MFModelStructure(
            model_type, self.utl_struct_objs
        )

    def is_valid(self):
        valid = True
        for package_struct in self.package_struct_objs:
            valid = valid and package_struct.is_valid()
        for model_struct in self.model_struct_objs:
            valid = valid and model_struct.is_valid()
        return valid

    def get_data_structure(self, path):
        if path[0] in self.package_struct_objs:
            if len(path) > 1:
                return self.package_struct_objs[path[0]].get_data_structure(
                    path[1:]
                )
            else:
                return self.package_struct_objs[path[0]]
        elif path[0] in self.model_struct_objs:
            if len(path) > 1:
                return self.model_struct_objs[path[0]].get_data_structure(
                    path[1:]
                )
            else:
                return self.model_struct_objs[path[0]]
        elif path[0] in self.utl_struct_objs:
            if len(path) > 1:
                return self.utl_struct_objs[path[0]].get_data_structure(
                    path[1:]
                )
            else:
                return self.utl_struct_objs[path[0]]
        elif path[0] == "nam":
            if len(path) > 1:
                return self.name_file_struct_obj.get_data_structure(path[1:])
            else:
                return self.name_file_struct_obj
        else:
            return None

    def tag_read_as_arrays(self):
        for key, package_struct in self.package_struct_objs.items():
            if key[0:-1] in self.package_struct_objs and key[-1] == "a":
                package_struct.read_as_arrays = True
        for model_key, model_struct in self.model_struct_objs.items():
            for (
                key,
                package_struct,
            ) in model_struct.package_struct_objs.items():
                if (
                    key[0:-1] in model_struct.package_struct_objs
                    and key[-1] == "a"
                ):
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

    def __new__(cls, internal_request=False, load_from_dfn_files=False):
        if cls._instance is None:
            cls._instance = super(MFStructure, cls).__new__(cls)

            # Initialize variables
            cls._instance.mf_version = 6
            cls._instance.valid = True
            cls._instance.sim_struct = None
            cls._instance.dimension_dict = {}
            cls._instance.load_from_dfn_files = load_from_dfn_files
            cls._instance.flopy_dict = {}

            # Read metadata from file
            cls._instance.valid = cls._instance.__load_structure()
        elif not cls._instance.valid and not internal_request:
            if cls._instance.__load_structure():
                cls._instance.valid = True

        return cls._instance

    def get_version_string(self):
        return format(str(self.mf_version))

    def __load_structure(self):
        # set up structure classes
        self.sim_struct = MFSimulationStructure()

        if self.load_from_dfn_files:
            mf_dfn = Dfn()
            dfn_files = mf_dfn.get_file_list()

            # load flopy-specific settings
            self.__load_flopy()

            # get common
            common_dfn = DfnFile("common.dfn")
            self.sim_struct.process_dfn(common_dfn)

            # process each file
            for file in dfn_files:
                self.sim_struct.process_dfn(DfnFile(file))
            self.sim_struct.tag_read_as_arrays()
        else:
            package_list = PackageContainer.package_factory(None, None)
            for package in package_list:
                self.sim_struct.process_dfn(DfnPackage(package))
            self.sim_struct.tag_read_as_arrays()

        return True

    def __load_flopy(self):
        current_variable = None
        var_info = {}
        dfn_path, tail = os.path.split(os.path.realpath(__file__))
        flopy_path = os.path.join(dfn_path, "dfn", "flopy.dfn")
        dfn_fp = open(flopy_path, "r")
        for line in dfn_fp:
            if self.__valid_line(line):
                lst_line = line.strip().split()
                if lst_line[0].lower() == "name":
                    # store current variable
                    self.flopy_dict[current_variable] = var_info
                    # reset var_info dict
                    var_info = {}
                    current_variable = lst_line[1].lower()
                else:
                    var_info[lst_line[0].lower()] = lst_line[1].lower()
        # store last variable
        self.flopy_dict[current_variable] = var_info

    @staticmethod
    def __valid_line(line):
        if len(line.strip()) > 1 and line[0] != "#":
            return True
        return False
