import os
import sys
import errno
import inspect
import numpy as np
from collections import OrderedDict

from .mfbase import PackageContainer, ExtFileAction, PackageContainerType
from .mfbase import MFFileMgmt, MFDataException, ReadAsArraysException, \
                    MFInvalidTransientBlockHeaderException, VerbosityLevel, \
                    FlopyException
from .data.mfstructure import DatumType
from .data import mfstructure, mfdata
from ..utils import datautil
from .data import mfdataarray, mfdatalist, mfdatascalar
from .coordinates import modeldimensions
from ..pakbase import PackageInterface


class MFBlockHeader(object):
    """
    Represents the header of a block in a MF6 input file

    Parameters
    ----------
    name : string
        block name
    variable_strings : list
        list of strings that appear after the block name
    comment : MFComment
        comment text in the block header

    Attributes
    ----------
    name : string
        block name
    variable_strings : list
        list of strings that appear after the block name
    comment : MFComment
        comment text in the block header
    data_items : list
        list of MFVariable of the variables contained in this block

    Methods
    -------
    write_header : (fd : file object)
        writes block header to file object 'fd'
    write_footer : (fd : file object)
        writes block footer to file object 'fd'
    """
    def __init__(self, name, variable_strings, comment, simulation_data=None,
                 path=None):
        self.name = name
        self.variable_strings = variable_strings
        if not ((simulation_data is None and path is None) or
               (simulation_data is not None and path is not None)):
            raise FlopyException('Block header must be initialized with both '
                                 'simulation_data and path or with neither.')
        if simulation_data is None:
            self.comment = comment
            self.simulation_data = None
            self.path = None
            self.comment_path = None
        else:
            self.connect_to_dict(simulation_data, path, comment)
        # TODO: Get data_items from dictionary
        self.data_items = []

    def build_header_variables(self, simulation_data, block_header_structure,
                               block_path, data, dimensions):
        self.data_items = []
        var_path = block_path + (block_header_structure[0].name,)

        # fix up data
        fixed_data = []
        if block_header_structure[0].data_item_structures[0].type == \
                DatumType.keyword:
            data_item = block_header_structure[0].data_item_structures[0]
            fixed_data.append(data_item.name)
        if type(data) == tuple:
            data = list(data)
        if type(data) == list:
            fixed_data = fixed_data + data
        else:
            fixed_data.append(data)
        if len(fixed_data) > 0:
            fixed_data = [tuple(fixed_data)]
        # create data object
        new_data = MFBlock.data_factory(simulation_data, None,
                                        block_header_structure[0], True,
                                        var_path, dimensions, fixed_data)
        self.data_items.append(new_data)

    def is_same_header(self, block_header):
        if len(self.data_items) == 0 or \
                        len(block_header.variable_strings) == 0:
            return True
        typ_obj = self.data_items[0].structure.data_item_structures[0].type_obj
        if typ_obj == int or typ_obj == float:
            if self.variable_strings[0] == block_header.variable_strings[0]:
                return True
            else:
                return False
        else:
            return True

    def get_comment(self):
        if self.simulation_data is None:
            return self.comment
        else:
            return self.simulation_data.mfdata[self.comment_path]

    def connect_to_dict(self, simulation_data, path, comment=None):
        self.simulation_data = simulation_data
        self.path = path
        self.comment_path = path + ('blk_hdr_comment',)
        if comment is None:
            simulation_data.mfdata[self.comment_path] = self.comment
        else:
            simulation_data.mfdata[self.comment_path] = comment
        self.comment = None

    def write_header(self, fd):
        fd.write('BEGIN {}'.format(self.name))
        if len(self.data_items) > 0:
            if isinstance(self.data_items[0], mfdatascalar.MFScalar):
                one_based = self.data_items[0].structure.type == \
                            DatumType.integer
                entry = self.data_items[0].get_file_entry(values_only=True,
                                                          one_based=one_based)
            else:
                entry = self.data_items[0].get_file_entry()
            fd.write('{}'.format(entry.rstrip()))
            if len(self.data_items) > 1:
                for data_item in self.data_items[1:]:
                    entry = data_item.get_file_entry(values_only=True)
                    fd.write('%s' % (entry.rstrip()))
        if self.get_comment().text:
            fd.write(' ')
            self.get_comment().write(fd)
        fd.write('\n')

    def write_footer(self, fd):
        fd.write('END {}'.format(self.name))
        if len(self.data_items) > 0:
            one_based = self.data_items[0].structure.type == \
                        DatumType.integer
            if isinstance(self.data_items[0], mfdatascalar.MFScalar):
                entry = self.data_items[0].get_file_entry(values_only=True,
                                                          one_based=one_based)
            else:
                entry = self.data_items[0].get_file_entry()
            fd.write('{}'.format(entry.rstrip()))
        fd.write('\n')

    def get_transient_key(self):
        transient_key = None
        for index in range(0, len(self.data_items)):
            if self.data_items[index].structure.type != DatumType.keyword:
                transient_key = self.data_items[index].get_data()
                if isinstance(transient_key, np.recarray):
                    item_struct = self.data_items[index].structure
                    key_index = item_struct.first_non_keyword_index()
                    if not (key_index is not None and
                           len(transient_key[0]) > key_index):
                        if key_index is None:
                            raise FlopyException('Block header index could '
                                                 'not be determined.')
                        else:
                            raise FlopyException('Block header index "{}" '
                                                 'must be less than "{}"'
                                                 '.'.format(
                                key_index, len(transient_key[0])))
                    transient_key = transient_key[0][key_index]
                break
        return transient_key


class MFBlock(object):
    """
    Represents a block in a MF6 input file


    Parameters
    ----------
    simulation_data : MFSimulationData
        data specific to this simulation
    dimensions : MFDimensions
        describes model dimensions including model grid and simulation time
    structure : MFVariableStructure
        structure describing block
    path : tuple
        unique path to block

    Attributes
    ----------
    block_headers : MFBlockHeaderIO
        block header text (BEGIN/END), header variables, comments in the header
    structure : MFBlockStructure
        structure describing block
    path : tuple
        unique path to block
    datasets : OrderDict
        dictionary of dataset objects with keys that are the name of the
        dataset
    datasets_keyword : dict
        dictionary of dataset objects with keys that are key words to identify
        start of dataset
    enabled : boolean
        block is being used

    Methods
    -------
    get_block_header_info : (line : string, path : tuple)
        static method that parses a line as a block header and returns a
        MFBlockHeader class representing the block header in that line
    load : (block_header : MFBlockHeader, fd : file, strict : boolean)
        loads block from file object.  file object must be advanced to
        beginning of block before calling
    write : (fd : file)
        writes block to a file object
    is_valid : ()
        returns true of the block is valid

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, simulation_data, dimensions, structure, path,
                 model_or_sim, container_package):
        self._simulation_data = simulation_data
        self._dimensions = dimensions
        self._model_or_sim = model_or_sim
        self._container_package = container_package
        self.block_headers = [MFBlockHeader(structure.name, [],
                                            mfdata.MFComment('', path,
                                                             simulation_data,
                                                             0),
                                            simulation_data, path)]
        self.structure = structure
        self.path = path
        self.datasets = OrderedDict()
        self.datasets_keyword = {}
        self.blk_trailing_comment_path = path + ('blk_trailing_comment',)
        self.blk_post_comment_path = path + ('blk_post_comment',)
        if self.blk_trailing_comment_path not in simulation_data.mfdata:
            simulation_data.mfdata[self.blk_trailing_comment_path] = \
              mfdata.MFComment('', '', simulation_data, 0)
        if self.blk_post_comment_path not in simulation_data.mfdata:
            simulation_data.mfdata[self.blk_post_comment_path] = \
              mfdata.MFComment('\n', '', simulation_data, 0)
        # initially disable if optional
        self.enabled = structure.number_non_optional_data() > 0
        self.loaded = False
        self.external_file_name = None
        self._structure_init()

    def __repr__(self):
        return self._get_data_str(True)

    def __str__(self):
        return self._get_data_str(False)

    def _get_data_str(self, formal):
        data_str = ''
        for idx, dataset in self.datasets.items():
            if formal:
                ds_repr = repr(dataset)
                if len(ds_repr.strip()) > 0:
                    data_str = '{}{}\n{}\n'.format(data_str,
                                                   dataset.structure.name,
                                                   repr(dataset))
            else:
                ds_str = str(dataset)
                if len(ds_str.strip()) > 0:
                    data_str = '{}{}\n{}\n'.format(data_str,
                                                   dataset.structure.name,
                                                   str(dataset))
        return data_str

    # return an MFScalar, MFList, or MFArray
    @staticmethod
    def data_factory(sim_data, model_or_sim, structure, enable, path, dimensions,
                     data=None, package=None):
        data_type = structure.get_datatype()
        # examine the data structure and determine the data type
        if data_type == mfstructure.DataType.scalar_keyword or \
          data_type == mfstructure.DataType.scalar:
            return mfdatascalar.MFScalar(sim_data, model_or_sim, structure, data,
                                         enable, path, dimensions)
        elif data_type == mfstructure.DataType.scalar_keyword_transient or \
          data_type == mfstructure.DataType.scalar_transient:
            trans_scalar = mfdatascalar.MFScalarTransient(sim_data,
                                                          model_or_sim,
                                                          structure,
                                                          enable, path,
                                                          dimensions)
            if data is not None:
                trans_scalar.set_data(data, key=0)
            return trans_scalar
        elif data_type == mfstructure.DataType.array:
            return mfdataarray.MFArray(sim_data, model_or_sim, structure, data,
                                       enable, path, dimensions)
        elif data_type == mfstructure.DataType.array_transient:
            trans_array = mfdataarray.MFTransientArray(sim_data, model_or_sim,
                                                       structure, enable, path,
                                                       dimensions)
            if data is not None:
                trans_array.set_data(data, key=0)
            return trans_array
        elif data_type == mfstructure.DataType.list:
            return mfdatalist.MFList(sim_data, model_or_sim, structure, data,
                                     enable,path, dimensions, package)
        elif data_type == mfstructure.DataType.list_transient:
            trans_list = mfdatalist.MFTransientList(sim_data, model_or_sim,
                                                    structure, enable, path,
                                                    dimensions, package)
            if data is not None:
                trans_list.set_data(data, key=0, autofill=True)
            return trans_list
        elif data_type == mfstructure.DataType.list_multiple:
            mult_list = mfdatalist.MFMultipleList(sim_data, model_or_sim,
                                                  structure, enable, path,
                                                  dimensions, package)
            if data is not None:
                mult_list.set_data(data, key=0, autofill=True)
            return mult_list

    def _structure_init(self):
        # load datasets keywords into dictionary
        for key, dataset_struct in self.structure.data_structures.items():
            for keyword in dataset_struct.get_keywords():
                self.datasets_keyword[keyword] = dataset_struct
        # load block header data items into dictionary
        for dataset in self.structure.block_header_structure:
            self._new_dataset(dataset.name, dataset, True, None)

    def set_model_relative_path(self, model_ws):
        # update datasets
        for key, dataset in self.datasets.items():
            if dataset.structure.file_data:
                try:
                    file_data = dataset.get_data()
                except MFDataException as mfde:
                    raise MFDataException(mfdata_except=mfde,
                                          model=
                                          self._container_package.model_name,
                                          package=
                                          self._container_package._get_pname(),
                                          message='Error occurred while '
                                                  'getting file data from '
                                                  '"{}"'.format(
                                                  dataset.structure.name))
                if file_data:
                    # update file path location for all file paths
                    for file_line in file_data:
                        old_file_path, old_file_name = \
                                os.path.split(file_line[0])
                        file_line[0] = os.path.join(model_ws, old_file_name)
        # update block headers
        for block_header in self.block_headers:
            for dataset in block_header.data_items:
                if dataset.structure.file_data:
                    try:
                        file_data = dataset.get_data()
                    except MFDataException as mfde:
                        raise MFDataException(mfdata_except=mfde,
                                              model=self._container_package.
                                              model_name,
                                              package=self._container_package.
                                              _get_pname(),
                                              message='Error occurred while '
                                                      'getting file data from '
                                                      '"{}"'.format(
                                                      dataset.structure.name))

                    if file_data:
                        # update file path location for all file paths
                        for file_line in file_data:
                            old_file_path, old_file_name = \
                                os.path.split(file_line[1])
                            new_file_path = os.path.join(model_ws,
                                                         old_file_name)
                            # update transient keys of datasets within the
                            # block
                            for key, idataset in self.datasets.items():
                                if isinstance(idataset, mfdata.MFTransient):
                                    idataset.update_transient_key(file_line[1],
                                                                 new_file_path)
                            file_line[1] = os.path.join(model_ws,
                                                        old_file_name)

    def add_dataset(self, dataset_struct, data, var_path):
        try:
            self.datasets[var_path[-1]] = self.data_factory(
                self._simulation_data, self._model_or_sim, dataset_struct,
                True, var_path, self._dimensions, data,
                self._container_package)
        except MFDataException as mfde:
            raise MFDataException(mfdata_except=mfde,
                                  model=self._container_package.model_name,
                                  package=self._container_package._get_pname(),
                                  message='Error occurred while adding'
                                  ' dataset "{}" to block '
                                  '"{}"'.format(dataset_struct.name,
                                                self.structure.name))

        self._simulation_data.mfdata[var_path] = self.datasets[var_path[-1]]
        dtype = dataset_struct.get_datatype()
        if dtype == mfstructure.DataType.list_transient or \
           dtype == mfstructure.DataType.list_multiple or \
           dtype == mfstructure.DataType.array_transient:
            # build repeating block header(s)
            if isinstance(data, dict):
                # Add block headers for each dictionary key
                for index in data:
                    if isinstance(index, tuple):
                        header_list = list(index)
                    else:
                        header_list = [index]
                    self._build_repeating_header(header_list)
            elif isinstance(data, list):
                # Add a single block header of value 0
                self._build_repeating_header([0])
            elif dtype != mfstructure.DataType.list_multiple and \
              data is not None:
                self._build_repeating_header([[0]])

        return self.datasets[var_path[-1]]

    def _build_repeating_header(self, header_data):
        if self._header_exists(header_data[0]):
            return
        if len(self.block_headers[-1].data_items) == 1 and \
           self.block_headers[-1].data_items[0].get_data() is not None:
            block_header_path = self.path + (len(self.block_headers) + 1,)
            block_header = MFBlockHeader(self.structure.name, [],
                                         mfdata.MFComment('', self.path,
                                         self._simulation_data, 0),
                                         self._simulation_data,
                                         block_header_path)
            self.block_headers.append(block_header)
        else:
            block_header_path = self.path + (len(self.block_headers),)
        struct = self.structure
        last_header = self.block_headers[-1]
        try:
            last_header.build_header_variables(self._simulation_data,
                                               struct.block_header_structure,
                                               block_header_path,
                                               header_data,
                                               self._dimensions)
        except MFDataException as mfde:
            raise MFDataException(mfdata_except=mfde,
                                  model=self._container_package.model_name,
                                  package=self._container_package._get_pname(),
                                  message='Error occurred while building'
                                  ' block header variables for block '
                                  '"{}"'.format(last_header.name))

    def _new_dataset(self, key, dataset_struct, block_header=False,
                     initial_val=None):
        dataset_path = self.path + (key,)
        if block_header:
            if dataset_struct.type == DatumType.integer and \
              initial_val is not None \
              and len(initial_val) >= 1 and \
              dataset_struct.get_record_size()[0] == 1:
                # stress periods are stored 0 based
                initial_val = int(initial_val[0]) - 1
            if isinstance(initial_val, list):
                initial_val = [tuple(initial_val)]
            try:
                new_data = MFBlock.data_factory(self._simulation_data,
                                                self._model_or_sim,
                                                dataset_struct, True,
                                                dataset_path, self._dimensions,
                                                initial_val,
                                                self._container_package)
            except MFDataException as mfde:
                raise MFDataException(mfdata_except=mfde,
                                      model=self._container_package.model_name,
                                      package=self._container_package._get_pname(),
                                      message='Error occurred while adding'
                                              ' dataset "{}" to block '
                                              '"{}"'.format(
                                          dataset_struct.name,
                                          self.structure.name))
            self.block_headers[-1].data_items.append(new_data)
        else:
            try:
                self.datasets[key] = self.data_factory(self._simulation_data,
                                                       self._model_or_sim,
                                                       dataset_struct, True,
                                                       dataset_path,
                                                       self._dimensions,
                                                       initial_val,
                                                       self._container_package)
            except MFDataException as mfde:
                raise MFDataException(mfdata_except=mfde,
                                      model=self._container_package.model_name,
                                      package=self._container_package._get_pname(),
                                      message='Error occurred while adding'
                                              ' dataset "{}" to block '
                                              '"{}"'.format(
                                          dataset_struct.name,
                                          self.structure.name))
        for keyword in dataset_struct.get_keywords():
            self.datasets_keyword[keyword] = dataset_struct

    def is_empty(self):
        for key, dataset in self.datasets.items():
            try:
                has_data = dataset.has_data()
            except MFDataException as mfde:
                raise MFDataException(mfdata_except=mfde,
                                      model=self._container_package.model_name,
                                      package=self._container_package._get_pname(),
                                      message='Error occurred while verifying'
                                              ' data of dataset "{}" in block '
                                              '"{}"'.format(
                                          dataset.structure.name,
                                          self.structure.name))

            if has_data is not None and has_data:
                return False
        return True

    def load(self, block_header, fd, strict=True):
        # verify number of header variables
        if len(block_header.variable_strings) < \
          self.structure.number_non_optional_block_header_data():
            if self._simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                warning_str = 'WARNING: Block header for block "{}" does not ' \
                              'contain the correct number of ' \
                              'variables {}'.format(block_header.name, self.path)
                print(warning_str)
            return

        if self.loaded:
            # verify header has not already been loaded
            for bh_current in self.block_headers:
                if bh_current.is_same_header(block_header):
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        warning_str = 'WARNING: Block header for block "{}" is ' \
                                      'not a unique block header ' \
                                      '{}'.format(block_header.name, self.path)
                        print(warning_str)
                    return

        # init
        self.enabled = True
        if not self.loaded:
            self.block_headers = []
        self.block_headers.append(block_header)

        # process any header variable
        if len(self.structure.block_header_structure) > 0:
            dataset = self.structure.block_header_structure[0]
            self._new_dataset(dataset.name, dataset, True,
                              self.block_headers[-1].variable_strings)

        # handle special readasarrays case
        if self._container_package.structure.read_as_arrays:
            # auxiliary variables may appear with aux variable name as keyword
            aux_vars = self._container_package.auxiliary.get_data()
            if aux_vars is not None:
                for var_name in list(aux_vars[0])[1:]:
                    self.datasets_keyword[(var_name,)] = \
                      self._container_package.aux.structure

        comments = []

        # capture any initial comments
        initial_comment = mfdata.MFComment('', '', 0)
        fd_block = fd
        line = fd_block.readline()
        datautil.PyListUtil.reset_delimiter_used()
        arr_line = datautil.PyListUtil.split_data_line(line)
        while mfdata.MFComment.is_comment(line, True):
            initial_comment.add_text(line)
            line = fd_block.readline()
            arr_line = datautil.PyListUtil.split_data_line(line)

        # if block not empty
        if not (len(arr_line[0]) > 2 and arr_line[0][:3].upper() == 'END'):
            if arr_line[0].lower() == 'open/close':
                # open block contents from external file
                fd_block.readline()
                fd_path, filename = os.path.split(
                  os.path.realpath(fd_block.name))
                try:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        opening external file "{}"..'
                              '.'.format(arr_line[1]))
                    self.external_file_name = arr_line[1]
                    fd_block = open(os.path.join(fd_path, self.external_file_name),
                                    'r')
                    # read first line of external file
                    line = fd_block.readline()
                    arr_line = datautil.PyListUtil.split_data_line(line)
                except:
                    type_, value_, traceback_ = sys.exc_info()
                    message = 'Error reading external file specified in ' \
                              'line "{}"'.format(line)
                    raise MFDataException(self._container_package.model_name,
                                          self._container_package._get_pname(),
                                          self.path, 'reading external file',
                                          self.structure.name,
                                          inspect.stack()[0][3], type_,
                                          value_, traceback_, message,
                                          self._simulation_data.debug)
            if len(self.structure.data_structures) <= 1:
                # load a single data set
                dataset = self.datasets[next(iter(self.datasets))]
                try:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        loading data {}..'
                              '.'.format(dataset.structure.name))
                    next_line = dataset.load(line, fd_block,
                                             self.block_headers[-1],
                                             initial_comment)
                except MFDataException as mfde:
                    raise MFDataException(
                        mfdata_except=mfde, model=self._container_package.
                        model_name,
                        package=self._container_package._get_pname(),
                        message='Error occurred while loading data "{}" in '
                                'block "{}" from file "{}"'
                                '.'.format(dataset.structure.name,
                                           self.structure.name,
                                           fd_block.name))
                package_info_list = self._get_package_info(dataset)
                if package_info_list is not None:
                    for package_info in package_info_list:
                        if self._simulation_data.verbosity_level.value >= \
                                VerbosityLevel.verbose.value:
                            print('        loading child package {}..'
                                  '.'.format(package_info[0]))
                        pkg = self._model_or_sim.load_package(
                            package_info[0], package_info[1],
                            package_info[1], True, package_info[2],
                            package_info[3], self._container_package)
                        if hasattr(self._container_package, package_info[0]):
                            package_group = getattr(self._container_package,
                                                    package_info[0])
                            package_group._append_package(pkg, pkg.filename,
                                                          False)

                if next_line[1] is not None:
                    arr_line = datautil.PyListUtil.split_data_line(
                      next_line[1])
                else:
                    arr_line = ''
                # capture any trailing comments
                post_data_comments = mfdata.MFComment('', '',
                                                      self._simulation_data, 0)
                dataset.post_data_comments = post_data_comments
                while arr_line and (len(next_line[1]) <= 2 or
                                    arr_line[0][:3].upper() != 'END'):
                    next_line[1] = fd_block.readline().strip()
                    arr_line = datautil.PyListUtil.split_data_line(
                        next_line[1])
                    if arr_line and (len(next_line[1]) <= 2 or
                                     arr_line[0][:3].upper() != 'END'):
                        post_data_comments.add_text(' '.join(arr_line))
            else:
                # look for keyword and store line as data or comment
                try:
                    key, results = self._find_data_by_keyword(line, fd_block,
                                                              initial_comment)
                except MFInvalidTransientBlockHeaderException as e:
                    warning_str = 'WARNING: {}'.format(e)
                    print(warning_str)
                    self.block_headers.pop()
                    return

                self._save_comments(arr_line, line, key, comments)
                if results[1] is None or results[1][:3].upper() != 'END':
                    # block consists of unordered datasets
                    # load the data sets out of order based on
                    # initial constants
                    line = ' '
                    while line != '':
                        line = fd_block.readline()
                        arr_line = datautil.PyListUtil.\
                            split_data_line(line)
                        if arr_line:
                            # determine if at end of block
                            if len(arr_line[0]) > 2 and \
                               arr_line[0][:3].upper() == 'END':
                                break
                            # look for keyword and store line as data o
                            # r comment
                            key, result = self._find_data_by_keyword(
                                line, fd_block, initial_comment)
                            self._save_comments(arr_line, line, key, comments)
                            if result[1] is not None and \
                               result[1][:3].upper() == 'END':
                                break

        self._simulation_data.mfdata[self.blk_trailing_comment_path].text = \
            comments
        self.loaded = True
        self.is_valid()

    def _find_data_by_keyword(self, line, fd, initial_comment):
        first_key = None
        nothing_found = False
        next_line = [True, line]
        while next_line[0] and not nothing_found:
            arr_line = datautil.PyListUtil.\
                split_data_line(next_line[1])
            key = datautil.find_keyword(arr_line, self.datasets_keyword)
            if key is not None:
                ds_name = self.datasets_keyword[key].name
                try:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        loading data {}...'.format(ds_name))
                    next_line = self.datasets[ds_name].load(
                        next_line[1], fd, self.block_headers[-1],
                        initial_comment)
                except MFDataException as mfde:
                    raise MFDataException(mfdata_except=mfde,
                                          model=self._container_package.
                                          model_name,
                                          package=self._container_package.
                                          _get_pname(),
                                          message='Error occurred while '
                                                  'loading data "{}" in '
                                                  'block "{}" from file "{}"'
                                                  '.'.format(
                                              ds_name, self.structure.name,
                                              fd.name))

                # see if first item's name indicates a reference to
                # another package
                package_info_list = self._get_package_info(self.datasets[
                                                           ds_name])
                if package_info_list is not None:
                    for package_info in package_info_list:
                        if self._simulation_data.verbosity_level.value >= \
                                VerbosityLevel.verbose.value:
                            print('        loading child package {}..'
                                  '.'.format(package_info[1]))
                        pkg = self._model_or_sim.load_package(
                            package_info[0], package_info[1], package_info[1],
                            True, package_info[2], package_info[3],
                            self._container_package)
                        if hasattr(self._container_package, package_info[0]):
                            package_group = getattr(self._container_package,
                                                    package_info[0])
                            package_group._append_package(pkg, pkg.filename,
                                                          False)
                if first_key is None:
                    first_key = key
                nothing_found = False
            elif arr_line[0].lower() == 'readasarrays' and \
              self.path[-1].lower() == 'options' and \
              self._container_package.structure.read_as_arrays == False:
                error_msg = 'ERROR: Attempting to read a ReadAsArrays ' \
                            'package as a non-ReadAsArrays ' \
                            'package {}'.format(self.path)
                raise ReadAsArraysException(error_msg)
            else:
                nothing_found = True

        if first_key is None:
            # look for recarrays.  if there is a lone recarray in this block,
            # use it by default
            recarrays = self.structure.get_all_recarrays()
            if len(recarrays) != 1:
                return key, [None, None]
            dataset = self.datasets[recarrays[0].name]
            ds_result = dataset.load(line, fd, self.block_headers[-1],
                                     initial_comment)

            # see if first item's name indicates a reference to another package
            package_info_list = self._get_package_info(dataset)
            if package_info_list is not None:
                for package_info in package_info_list:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        loading child package {}..'
                              '.'.format(package_info[0]))
                    pkg = self._model_or_sim.load_package(
                        package_info[0], package_info[1], None, True,
                        package_info[2], package_info[3],
                        self._container_package)
                    if hasattr(self._container_package, package_info[0]):
                        package_group = getattr(self._container_package,
                                                package_info[0])
                        package_group._append_package(pkg, pkg.filename,
                                                      False)

            return recarrays[0].keyword, ds_result
        else:
            return first_key, next_line

    def _get_package_info(self, dataset):
        if not dataset.structure.file_data:
            return None
        for index in range(0, len(dataset.structure.data_item_structures)):
            data_item = dataset.structure.data_item_structures[index]
            if data_item.type == DatumType.keyword or data_item.type == \
                    DatumType.string:
                item_name = data_item.name
                package_type = item_name[:-1]
                model_type = self._model_or_sim.structure.model_type
                if PackageContainer.package_factory(package_type,
                                                    model_type) is not None:
                    try:
                        data = dataset.get_data()
                    except MFDataException as mfde:
                        raise MFDataException(mfdata_except=mfde,
                                              model=self._container_package.
                                              model_name,
                                              package=self._container_package.
                                              _get_pname(),
                                              message='Error occurred while '
                                                      'getting data from "{}" '
                                                      'in block "{}".'.format(
                                                  dataset.structure.name,
                                                  self.structure.name))
                    if isinstance(data, np.recarray):
                        file_location = data[-1][index]
                    else:
                        file_location = data
                    package_info_list = []
                    file_path, file_name = os.path.split(file_location)
                    dict_package_name = '{}_{}'.format(package_type,
                                                       self.path[-2])
                    package_info_list.append((package_type, file_name,
                                              file_path,
                                              dict_package_name))
                    return package_info_list
                return None
        return None

    def _save_comments(self, arr_line, line, key, comments):
        # FIX: Save these comments somewhere in the data set
        if not key in self.datasets_keyword:
            if mfdata.MFComment.is_comment(key, True):
                if comments:
                    comments.append('\n')
                comments.append(arr_line)

    def write(self, fd, ext_file_action=ExtFileAction.copy_relative_paths):
        # never write an empty block
        is_empty = self.is_empty()
        if is_empty and self.structure.name.lower() != 'exchanges' and \
          self.structure.name.lower() != 'options':
            return
        if self.structure.repeating():
            repeating_datasets = self._find_repeating_datasets()
            for repeating_dataset in repeating_datasets:
                # resolve any missing block headers
                self._add_missing_block_headers(repeating_dataset)
            if len(repeating_datasets) > 0:
                # loop through all block headers
                for block_header in self.block_headers:
                    self._write_block(fd, block_header, ext_file_action)
            else:
                # write out block
                self._write_block(fd, self.block_headers[0], ext_file_action)

        else:
            self._write_block(fd, self.block_headers[0], ext_file_action)

    def _add_missing_block_headers(self, repeating_dataset):
        for key in repeating_dataset.get_active_key_list():
            if not self._header_exists(key[0]):
                self._build_repeating_header([key[0]])

    def _header_exists(self, key):
        if not isinstance(key, list):
            comp_key_list = [key]
        else:
            comp_key_list = key
        for block_header in self.block_headers:
            transient_key = block_header.get_transient_key()
            for comp_key in comp_key_list:
                if transient_key is not None and transient_key == comp_key:
                    return True
        return False

    def _find_repeating_datasets(self):
        repeating_datasets = []
        for key, dataset in self.datasets.items():
            if dataset.repeating:
                repeating_datasets.append(dataset)
        return repeating_datasets

    def _write_block(self, fd, block_header, ext_file_action):
        # write block header
        block_header.write_header(fd)
        transient_key = None
        if len(block_header.data_items) > 0:
            transient_key = block_header.get_transient_key()

        if self.external_file_name is not None:
            # write block contents to external file
            indent_string = self._simulation_data.indent_string
            fd.write('{}open/close {}\n'.format(indent_string,
                                                self.external_file_name))
            fd_main = fd
            fd_path, filename = os.path.split(os.path.realpath(fd.name))
            try:
                fd = open(os.path.join(fd_path, self.external_file_name), 'w')
            except:
                type_, value_, traceback_ = sys.exc_info()
                message = 'Error reading external file ' \
                          '"{}"'.format(self.external_file_name)
                raise MFDataException(self._container_package.model_name,
                                      self._container_package._get_pname(),
                                      self.path, 'reading external file',
                                      self.structure.name,
                                      inspect.stack()[0][3], type_,
                                      value_, traceback_, message,
                                      self._simulation_data.debug)
        # write data sets
        for key, dataset in self.datasets.items():
            try:
                if transient_key is None:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        writing data {}..'
                              '.'.format(dataset.structure.name))
                    fd.write(dataset.get_file_entry(
                        ext_file_action=ext_file_action))
                else:
                    if self._simulation_data.verbosity_level.value >= \
                            VerbosityLevel.verbose.value:
                        print('        writing data {} ({})..'
                              '.'.format(dataset.structure.name,
                                         transient_key))
                    if dataset.repeating:
                        fd.write(dataset.get_file_entry(
                            transient_key, ext_file_action=ext_file_action))
                    else:
                        fd.write(dataset.get_file_entry(
                            ext_file_action=ext_file_action))
            except MFDataException as mfde:
                raise MFDataException(mfdata_except=mfde,
                                      model=self._container_package.model_name,
                                      package=self._container_package.
                                      _get_pname(),
                                      message='Error occurred while writing '
                                              'data "{}" in block "{}" to file'
                                              ' "{}".'.format(
                                                  dataset.structure.name,
                                                  self.structure.name,
                                                  fd.name))
        # write trailing comments
        self._simulation_data.mfdata[self.blk_trailing_comment_path].write(fd)

        if self.external_file_name is not None:
            # switch back writing to package file
            fd.close()
            fd = fd_main

        # write block footer
        block_header.write_footer(fd)

        # write post block comments
        self._simulation_data.mfdata[self.blk_post_comment_path].write(fd)

        # write extra line if comments are off
        if not self._simulation_data.comments_on:
            fd.write('\n')

    def is_allowed(self):
        if self.structure.variable_dependant_path:
            # fill in empty part of the path with the current path
            if len(self.structure.variable_dependant_path) == 3:
                dependant_var_path = (self.path[0],) + \
                                     self.structure.variable_dependant_path
            elif len(self.structure.variable_dependant_path) == 2:
                dependant_var_path = (self.path[0], self.path[1]) + \
                                     self.structure.variable_dependant_path
            elif len(self.structure.variable_dependant_path) == 1:
                dependant_var_path = (self.path[0], self.path[1],
                                      self.path[2]) + \
                                     self.structure.variable_dependant_path
            else:
                dependant_var_path = None

            # get dependency
            dependant_var = None
            mf_data = self._simulation_data.mfdata
            if dependant_var_path in mf_data:
                dependant_var = mf_data[dependant_var_path]

            # resolve dependency
            if self.structure.variable_value_when_active[0] == 'Exists':
                exists = self.structure.variable_value_when_active[1]
                if dependant_var and exists.lower() == 'true':
                    return True
                elif not dependant_var and exists.lower() == 'false':
                    return True
                else:
                    return False
            elif not dependant_var:
                return False
            elif self.structure.variable_value_when_active[0] == '>':
                min_val = self.structure.variable_value_when_active[1]
                if dependant_var > float(min_val):
                    return True
                else:
                    return False
            elif self.structure.variable_value_when_active[0] == '<':
                max_val = self.structure.variable_value_when_active[1]
                if dependant_var < float(max_val):
                    return True
                else:
                    return False
        return True

    def is_valid(self):
        # check data sets
        for key, dataset in self.datasets.items():
            # Non-optional datasets must be enabled
            if not dataset.structure.optional and not dataset.enabled:
                return False
            # Enabled blocks must be valid
            if dataset.enabled and not dataset.is_valid:
                return False
        # check variables
        for block_header in self.block_headers:
            for dataset in block_header.data_items:
                # Non-optional datasets must be enabled
                if not dataset.structure.optional and not dataset.enabled:
                    return False
                # Enabled blocks must be valid
                if dataset.enabled and not dataset.is_valid():
                    return False


class MFPackage(PackageContainer, PackageInterface):
    """
    Provides an interface for the user to specify data to build a package.

    Parameters
    ----------
    model_or_sim : MFModel of MFSimulation
        the parent model or simulation containing this package
    package_type : string
        string defining the package type
    filename : string
        filename of file where this package is stored
    pname : string
        package name
    loading_package : bool
        whether or not to add this package to the parent container's package
        list during initialization
    parent_file : MFPackage
        parent package that contains this package

    Attributes
    ----------
    blocks : OrderedDict
        dictionary of blocks contained in this package by block name
    path : tuple
        data dictionary path to this package
    structure : PackageStructure
        describes the blocks and data contain in this package
    dimensions : PackageDimension
        resolves data dimensions for data within this package

    Methods
    -------
    build_mfdata : (var_name : variable name, data : data contained in this
      object) : MFData subclass
        Returns the appropriate data type object (mfdatalist, mfdataarray, or
        mfdatascalar) giving that object the appropriate structure (looked
        up based on var_name) and any data supplied
    load : (strict : bool) : bool
        Loads the package from file
    is_valid : bool
        Returns whether or not this package is valid
    write
        Writes the package to a file
    get_file_path : string
        Returns the package file's path
    remove
        Removes package from the simulation/model it is currently a part of

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """
    def __init__(self, model_or_sim, package_type, filename=None, pname=None,
                 loading_package=False, parent_file=None):

        self._model_or_sim = model_or_sim
        self._data_list = []
        self._package_type = package_type
        if model_or_sim.type == 'Model' and package_type.lower() != 'nam':
            self.model_name = model_or_sim.name
        else:
            self.model_name = None

        if model_or_sim.type != 'Model' and model_or_sim.type != 'Simulation':
            message = 'Invalid model_or_sim parameter. Expecting either a ' \
                      'model or a simulation. Instead type "{}" was ' \
                      'given.'.format(type(model_or_sim))
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.model_name, pname, '', 'initializing '
                                  'package', None, inspect.stack()[0][3],
                                  type_, value_, traceback_, message,
                                  model_or_sim.simulation_data.debug)

        super(MFPackage, self).__init__(model_or_sim.simulation_data,
                                        self.model_name)

        self.parent = model_or_sim
        self._simulation_data = model_or_sim.simulation_data
        self.parent_file = parent_file
        self.blocks = OrderedDict()
        self.container_type = []
        self.loading_package = loading_package
        if pname is not None:
            if not isinstance(pname, str):
                message = 'Invalid pname parameter. Expecting type str. ' \
                          'Instead type "{}" was ' \
                          'given.'.format(type(pname))
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.model_name, pname, '',
                                      'initializing package', None,
                                      inspect.stack()[0][3], type_, value_,
                                      traceback_, message,
                                      model_or_sim.simulation_data.debug)

            self.package_name = pname.lower()
        else:
            self.package_name = None

        if filename is None:
            self._filename = MFFileMgmt.string_to_file_path('{}.{}'.format(
              self._model_or_sim.name, package_type))
        else:
            if not isinstance(filename, str):
                message = 'Invalid fname parameter. Expecting type str. ' \
                          'Instead type "{}" was ' \
                          'given.'.format(type(filename))
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.model_name, pname, '',
                                      'initializing package', None,
                                      inspect.stack()[0][3], type_, value_,
                                      traceback_, message,
                                      model_or_sim.simulation_data.debug)

            self._filename = MFFileMgmt.string_to_file_path(filename)

        self.path, \
            self.structure = model_or_sim.register_package(self,
                                                           not loading_package,
                                                           pname is None,
                                                           filename is None)
        self.dimensions = self.create_package_dimensions()

        if self.path is None:
            if self._simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('WARNING: Package type {} failed to register property.'
                      ' {}'.format(self._package_type, self.path))
        if parent_file is not None:
            self.container_type.append(PackageContainerType.package)
        # init variables that may be used later
        self.post_block_comments = None
        self.last_error = None
        self.bc_color = "black"
        self.__inattr = False
        self._child_package_groups = {}

    def __setattr__(self, name, value):
        if hasattr(self, name) and getattr(self, name) is not None:
            attribute = object.__getattribute__(self, name)
            if attribute is not None and isinstance(attribute, mfdata.MFData):
                self.parent._mg_resync = True
                try:
                    if isinstance(attribute, mfdatalist.MFList):
                        attribute.set_data(value, autofill=True)
                    else:
                        attribute.set_data(value)
                except MFDataException as mfde:
                    raise MFDataException(mfdata_except=mfde,
                                          model=self.model_name,
                                          package=self._get_pname())
                return
        super(MFPackage, self).__setattr__(name, value)

    def __repr__(self):
        return self._get_data_str(True)

    def __str__(self):
        return self._get_data_str(False)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, fname):
        if isinstance(self.parent_file, MFPackage) and \
                self.structure.file_type in \
                self.parent_file._child_package_groups:
            try:
                child_pkg_group = self.parent_file._child_package_groups[
                    self.structure.file_type]
                child_pkg_group._update_filename(self._filename, fname)
            except:
                print('WARNING: Unable to update file name for parent'
                      'package of {}.'.format(self.name))
        self._filename = fname

    @property
    def package_type(self):
        return self._package_type

    @property
    def name(self):
        return [self.package_name]

    @name.setter
    def name(self, name):
        self.package_name = name

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def plotable(self):
        if self._model_or_sim.type == "Simulation":
            return False
        else:
            return True

    @property
    def data_list(self):
        # return [data_object, data_object, ...]
        return self._data_list

    def export(self, f, **kwargs):
        from flopy import export
        return export.utils.package_export(f, self, **kwargs)

    def _get_data_str(self, formal, show_data=True):
        data_str = 'package_name = {}\nfilename = {}\npackage_type = {}' \
                   '\nmodel_or_simulation_package = {}' \
                   '\n{}_name = {}' \
                   '\n'.format(self._get_pname(), self._filename,
                               self.package_type,
                               self._model_or_sim.type.lower(),
                               self._model_or_sim.type.lower(),
                               self._model_or_sim.name)
        if self.parent_file is not None and formal:
            data_str = '{}parent_file = ' \
                       '{}\n\n'.format(data_str, self.parent_file._get_pname())
        else:
            data_str = '{}\n'.format(data_str)
        if show_data:
            for idx, block in self.blocks.items():
                if formal:
                    bl_repr = repr(block)
                    if len(bl_repr.strip()) > 0:
                        data_str = '{}Block {}\n--------------------\n{}' \
                                   '\n'.format(data_str, block.structure.name,
                                               repr(block))
                else:
                    bl_str = str(block)
                    if len(bl_str.strip()) > 0:
                        data_str = '{}Block {}\n--------------------\n{}' \
                                   '\n'.format(data_str, block.structure.name,
                                               str(block))
        return data_str

    def _get_pname(self):
        if self.package_name is not None:
            return '{}'.format(self.package_name)
        else:
            return '{}'.format(self._filename)

    def _get_block_header_info(self, line, path):
        # init
        header_variable_strs = []
        arr_clean_line = line.strip().split()
        header_comment = mfdata.MFComment('', path + (arr_clean_line[1],),
                                          self._simulation_data, 0)
        # break header into components
        if len(arr_clean_line) < 2:
            message = 'Block header does not contain a name. Name ' \
                      'expected in line "{}".'.format(line)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.model_name,
                                  self._get_pname(),
                                  self.path,
                                  'parsing block header', None,
                                  inspect.stack()[0][3],
                                  type_, value_, traceback_, message,
                                  self._simulation_data.debug)
        elif len(arr_clean_line) == 2:
            return MFBlockHeader(arr_clean_line[1], header_variable_strs,
                                 header_comment)
        else:
            # process text after block name
            comment = False
            for entry in arr_clean_line[2:]:
                # if start of comment
                if mfdata.MFComment.is_comment(entry.strip()[0]):
                    comment = True
                if comment:
                    header_comment.text = ' '.join([header_comment.text,
                                                    entry])
                else:
                    header_variable_strs.append(entry)
            return MFBlockHeader(arr_clean_line[1], header_variable_strs,
                                 header_comment)

    def _update_size_defs(self):
        # build temporary data lookup by name
        data_lookup = {}
        for name, block in self.blocks.items():
            for ds_index, dataset in block.datasets.items():
                data_lookup[dataset.structure.name] = dataset

        # loop through all data
        for name, block in self.blocks.items():
            for ds_index, dataset in block.datasets.items():
                # if data shape is 1-D
                if dataset.structure.shape and \
                        len(dataset.structure.shape) == 1:
                    # if shape name is data in this package
                    if dataset.structure.shape[0] in data_lookup:
                        size_def = data_lookup[dataset.structure.shape[0]]
                        size_def_name = size_def.structure.name

                        if isinstance(dataset, mfdata.MFTransient):
                            # for transient data always use the maximum size
                            new_size = -1
                            for key in dataset.get_active_key_list():
                                try:
                                    data = dataset.get_data(key=key[0])
                                except (IOError,
                                        OSError,
                                        MFDataException):
                                    # TODO: Handle case where external file
                                    # path has been moved
                                    data = None
                                if data is not None:
                                    data_len = len(data)
                                    if data_len > new_size:
                                        new_size = data_len
                        else:
                            # for all other data set max to size
                            new_size = -1
                            try:
                                data = dataset.get_data()
                            except (IOError,
                                    OSError,
                                    MFDataException):
                                # TODO: Handle case where external file
                                # path has been moved
                                data = None
                            if data is not None:
                                new_size = len(dataset.get_data())
                        if size_def.get_data() != new_size >= 0:
                            # store current size
                            size_def.set_data(new_size)

                            # informational message to the user
                            if self._simulation_data.verbosity_level.value >= \
                                    VerbosityLevel.normal.value:
                                print('INFORMATION: {} in {} changed to {} '
                                      'based on size of '
                                      '{}'.format(size_def_name,
                                                  size_def.structure.path[:-1],
                                                  new_size,
                                                  dataset.structure.name))

    def remove(self):
        self._model_or_sim.remove_package(self)

    def build_child_packages_container(self, pkg_type, filerecord):
        # get package class
        package_obj = self.package_factory(pkg_type,
                                           self._model_or_sim.model_type)
        # create child package object
        child_pkgs_name = 'utl{}packages'.format(pkg_type)
        child_pkgs_obj = self.package_factory(child_pkgs_name, '')
        child_pkgs = child_pkgs_obj(self._model_or_sim, self, pkg_type,
                                    filerecord, None, package_obj)
        setattr(self, pkg_type, child_pkgs)
        self._child_package_groups[pkg_type] = child_pkgs

    def build_child_package(self, pkg_type, data, parameter_name, filerecord):
        if not hasattr(self, pkg_type):
            self.build_child_packages_container(pkg_type, filerecord)
        if data is not None:
            package_group = getattr(self, pkg_type)
            # build child package file name
            child_path = package_group._next_default_file_path()
            # create new empty child package
            package_obj = self.package_factory(pkg_type,
                                               self._model_or_sim.model_type)
            package = package_obj(self._model_or_sim, filename=child_path,
                                  parent_file=self)
            assert hasattr(package, parameter_name)

            if isinstance(data, dict):
                # evaluate and add data to package
                unused_data = {}
                for key, value in data.items():
                    # if key is an attribute of the child package
                    if isinstance(key, str) and hasattr(package, key):
                        # set child package attribute
                        child_data_attr = getattr(package, key)
                        if isinstance(child_data_attr, mfdatalist.MFList):
                            child_data_attr.set_data(value, autofill=True)
                        elif isinstance(child_data_attr, mfdata.MFData):
                            child_data_attr.set_data(value)
                        elif key == 'fname' or key == 'filename':
                            child_path = value
                            package._filename = value
                        else:
                            setattr(package, key, value)
                    else:
                        unused_data[key] = value
                if unused_data:
                    setattr(package, parameter_name, unused_data)
            else:
                setattr(package, parameter_name, data)

            # append package to list
            package_group._init_package(package, child_path)

    def build_mfdata(self, var_name, data=None):
        if self.loading_package:
            data = None
        for key, block in self.structure.blocks.items():
            if var_name in block.data_structures:
                if block.name not in self.blocks:
                    self.blocks[block.name] = MFBlock(self._simulation_data,
                                                      self.dimensions, block,
                                                      self.path + (key,),
                                                      self._model_or_sim, self)
                dataset_struct = block.data_structures[var_name]
                var_path = self.path + (key, var_name)
                ds = self.blocks[block.name].add_dataset(dataset_struct,
                                                         data, var_path)
                self._data_list.append(ds)
                return ds

        message = 'Unable to find variable "{}" in package ' \
                  '"{}".'.format(var_name, self.package_type)
        type_, value_, traceback_ = sys.exc_info()
        raise MFDataException(self.model_name, self._get_pname(),
                              self.path, 'building data objects',
                              None, inspect.stack()[0][3],
                              type_, value_, traceback_, message,
                              self._simulation_data.debug)

    def set_model_relative_path(self, model_ws):
        # update blocks
        for key, block in self.blocks.items():
            block.set_model_relative_path(model_ws)
        # update sub-packages
        for package in self._packagelist:
            package.set_model_relative_path(model_ws)

    def load(self, strict=True):
        # open file
        try:
            fd_input_file = open(self.get_file_path(), 'r')
        except OSError as e:
            if e.errno == errno.ENOENT:
                message = 'File {} of type {} could not be opened' \
                          '.'.format(self.get_file_path(), self.package_type)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.model_name,
                                      self.package_name,
                                      self.path, 'loading package file',
                                      None, inspect.stack()[0][3],
                                      type_, value_, traceback_, message,
                                      self._simulation_data.debug)

        try:
            self._load_blocks(fd_input_file, strict)
        except ReadAsArraysException as err:
            fd_input_file.close()
            raise ReadAsArraysException(err)
        # close file
        fd_input_file.close()

        if self.simulation_data.auto_set_sizes:
            self._update_size_defs()

        # return validity of file
        return self.is_valid()

    def is_valid(self):
        # Check blocks
        for key, block in self.blocks.items():
            # Non-optional blocks must be enabled
            if block.structure.number_non_optional_data() > 0 and \
               not block.enabled and block.is_allowed():
                self.last_error = 'Required block "{}" not ' \
                                  'enabled'.format(block.block_header.name)
                return False
            # Enabled blocks must be valid
            if block.enabled and not block.is_valid:
                self.last_error = 'Invalid block ' \
                                  '"{}"'.format(block.block_header.name)
                return False

        return True

    def _load_blocks(self, fd_input_file, strict=True, max_blocks=sys.maxsize):
        # init
        self._simulation_data.mfdata[self.path + ('pkg_hdr_comments',)] = \
          mfdata.MFComment('', self.path, self._simulation_data)
        self.post_block_comments = mfdata.MFComment('', self.path,
                                                    self._simulation_data)

        blocks_read = 0
        found_first_block = False
        line = ' '
        while line != '':
            line = fd_input_file.readline()
            clean_line = line.strip()
            # If comment or empty line
            if mfdata.MFComment.is_comment(clean_line, True):
                self._store_comment(line, found_first_block)
            elif len(clean_line) > 4 and clean_line[:5].upper() == 'BEGIN':
                # parse block header
                try:
                    block_header_info = self._get_block_header_info(line,
                                                                    self.path)
                except MFDataException as mfde:
                    message = 'An error occurred while loading block header ' \
                              'in line "{}".'.format(line)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(self.model_name, self._get_pname(),
                                          self.path, 'loading block header',
                                          None, inspect.stack()[0][3],
                                          type_, value_, traceback_, message,
                                          self._simulation_data.debug)

                # if there is more than one possible block with the same name,
                # resolve the correct block to use
                block_key = block_header_info.name.lower()
                block_num = 1
                possible_key = '{}-{}'.format(block_header_info.name.lower(),
                                              block_num)
                if possible_key in self.blocks:
                    block_key = possible_key
                    block_header_name = block_header_info.name.lower()
                    while block_key in self.blocks and \
                      not self.blocks[block_key].is_allowed():
                        block_key = '{}-{}'.format(block_header_name,
                                                   block_num)
                        block_num += 1

                if block_key not in self.blocks:
                    # block name not recognized, load block as comments and
                    # issue a warning
                    if self.simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        warning_str = 'WARNING: Block "{}" is not a valid block ' \
                                      'name for file type ' \
                                      '{}.'.format(block_key, self.package_type)
                        print(warning_str)
                    self._store_comment(line, found_first_block)
                    while line != '':
                        line = fd_input_file.readline()
                        self._store_comment(line, found_first_block)
                        arr_line = datautil.PyListUtil.split_data_line(line)
                        if arr_line and (len(arr_line[0]) <= 2 or
                           arr_line[0][:3].upper() == 'END'):
                            break
                else:
                    found_first_block = True
                    self.post_block_comments = \
                      mfdata.MFComment('', self.path, self._simulation_data)
                    skip_block = False
                    if self.blocks[block_key].loaded:
                        # Only blocks defined as repeating are allowed to have
                        # multiple entries
                        header_name = block_header_info.name
                        if not self.structure.blocks[header_name.lower()].\
                                repeating():
                            # warn and skip block
                            if self._simulation_data.verbosity_level.value >= \
                                    VerbosityLevel.normal.value:
                                warning_str = 'WARNING: Block "{}" has ' \
                                              'multiple entries and is not ' \
                                              'intended to be a repeating ' \
                                              'block ({} package' \
                                              ')'.format(header_name,
                                                         self.package_type)
                                print(warning_str)
                            skip_block = True

                    if not skip_block:
                        if self.simulation_data.verbosity_level.value >= \
                                VerbosityLevel.verbose.value:
                            print('      loading block {}...'.format(
                                self.blocks[block_key].structure.name))

                        self.blocks[block_key].load(block_header_info,
                                                    fd_input_file, strict)
                        self._simulation_data.mfdata[self.blocks[block_key].
                                                     blk_post_comment_path] = \
                            self.post_block_comments

                        blocks_read += 1
                        if blocks_read >= max_blocks:
                            break
            else:
                if not (len(clean_line) == 0 or (len(line) > 2 and
                                                 line[:3].upper() == 'END')):
                    # Record file location of beginning of unresolved text
                    # treat unresolved text as a comment for now
                    self._store_comment(line, found_first_block)

    def write(self, ext_file_action=ExtFileAction.copy_relative_paths):
        if self.simulation_data.auto_set_sizes:
            self._update_size_defs()

        # create any folders in path
        package_file_path = self.get_file_path()
        package_folder = os.path.split(package_file_path)[0]
        if package_folder and not os.path.isdir(package_folder):
            os.makedirs(os.path.split(package_file_path)[0])

        # open file
        fd = open(package_file_path, 'w')

        # write blocks
        self._write_blocks(fd, ext_file_action)

        fd.close()

    def create_package_dimensions(self):
        model_dims = None
        if self.container_type[0] == PackageContainerType.model:
            model_dims = [modeldimensions.ModelDimensions(
                self.path[0], self._simulation_data)]
        else:
            # this is a simulation file that does not correspond to a specific
            # model.  figure out which model to use and return a dimensions
            # object for that model
            if self.dfn_file_name[0:3] == 'exg':
                exchange_rec_array = self._simulation_data.mfdata[
                    ('nam', 'exchanges', 'exchanges')].get_data()
                if exchange_rec_array is None:
                    return None
                for exchange in exchange_rec_array:
                    if exchange[1].lower() == self._filename.lower():
                        model_dims = [modeldimensions.ModelDimensions(
                                          exchange[2], self._simulation_data),
                                      modeldimensions.ModelDimensions(
                                          exchange[3], self._simulation_data)]
                        break
            elif self.parent_file is not None:
                model_dims = []
                for md in self.parent_file.dimensions.model_dim:
                    model_name = md.model_name
                    model_dims.append(modeldimensions.ModelDimensions(
                        model_name, self._simulation_data))
            else:
                model_dims = [modeldimensions.ModelDimensions(
                    None, self._simulation_data)]
        return modeldimensions.PackageDimensions(model_dims, self.structure,
                                                 self.path)

    def _store_comment(self, line, found_first_block):
        # Store comment
        if found_first_block:
            self.post_block_comments.text += line
        else:
            self._simulation_data.mfdata[self.path +
                                         ('pkg_hdr_comments',)].text += line

    def _write_blocks(self, fd, ext_file_action):
        # verify that all blocks are valid
        if not self.is_valid():
            message = 'Unable to write out model file "{}" due to the ' \
                        'following error: ' \
                        '{} ({})'.format(self._filename, self.last_error,
                                         self.path)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(self.model_name, self._get_pname(),
                                  self.path, 'writing package blocks',
                                  None, inspect.stack()[0][3],
                                  type_, value_, traceback_, message,
                                  self._simulation_data.debug)

        # write initial comments
        pkg_hdr_comments_path = self.path + ('pkg_hdr_comments',)
        if pkg_hdr_comments_path in self._simulation_data.mfdata:
            self._simulation_data.mfdata[self.path +
                                         ('pkg_hdr_comments',)].write(fd,
                                                                      False)

        # loop through blocks
        block_num = 1
        for index, block in self.blocks.items():
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.verbose.value:
                print('      writing block {}...'.format(block.structure.name))
            # write block
            block.write(fd, ext_file_action=ext_file_action)
            block_num += 1

    def get_file_path(self):
        if self.path[0] in self._simulation_data.mfpath.model_relative_path:
            return os.path.join(self._simulation_data.mfpath.get_model_path(
                self.path[0]), self._filename)
        else:
            return os.path.join(self._simulation_data.mfpath.get_sim_path(),
                                self._filename)

    def export(self, f, **kwargs):
        from flopy import export
        return export.utils.package_export(f, self, **kwargs)

    def plot(self, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        package input data

        Parameters
        ----------
        package: flopy.pakbase.Package instance supplied for plotting

        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return. (default is
                zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        ----------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.

        """
        from flopy.plot.plotutil import PlotUtilities

        if not self.plotable:
            raise TypeError("Simulation level packages are not plotable")

        axes = PlotUtilities._plot_package_helper(self,
                                                  **kwargs)
        return axes


class MFChildPackages(object):
    def __init__(self, model, parent, pkg_type, filerecord, package=None,
                 package_class=None):
        self._packages = []
        self._filerecord = filerecord
        if package is not None:
            self._packages.append(package)
        self._model = model
        self._cpparent = parent
        self._pkg_type = pkg_type
        self._package_class = package_class
        self._inattr = False

    def __getattribute__(self, name):
        if name == '_MFChildPackages_packages' or name == \
                '_MFChildPackages_inattr' or name == '_packages' or \
                name == '_inattr':
            return super(MFChildPackages, self).__getattribute__(name)

        if self._inattr is not None and not self._inattr:
            if self._packages and hasattr(self._packages[0], name):
                self._inattr = True
                item = getattr(self._packages[0], name)
                self._inattr = False
                return item
        return super(MFChildPackages, self).__getattribute__(name)

    def __getattr__(self, attr):
        if attr == '_MFChildPackages_inattr' or attr == '_inattr':
            return None
        raise AttributeError(attr)

    def __getitem__(self, k):
        if isinstance(k, int):
            if k < len(self._packages):
                return self._packages[k]
        raise Exception('Package index {} does not exist.'.format(k))

    def __setattr__(self, key, value):
        if key != '_packages' and key != '_model' and key != '_cpparent' and \
                key != '_inattr' and key != '_filerecord' and \
                key != '_package_class' and key != '_pkg_type':
            if len(self._packages) == 0:
                raise Exception('No {} package is currently attached to package'
                                ' {}. Use the initialize method to create a(n) '
                                '{} package before attempting to access its '
                                'properties.'.format(self._pkg_type,
                                                     self._cpparent.filename,
                                                     self._pkg_type))
            package = self._packages[0]
            setattr(package, key, value)
            return
        super(MFChildPackages, self).__setattr__(key, value)

    def __default_file_path_base(self, file_path, suffix=''):
        root, stem = os.path.split(file_path)
        stem_lst = stem.split('.')
        file_name = '.'.join(stem_lst[:-1])
        if len(stem_lst) > 1:
            file_ext = stem_lst[-1]
            return '{}.{}{}.{}'.format(file_name, file_ext, suffix,
                                     self._pkg_type)
        elif suffix != '':
            return '{}.{}'.format(stem, self._pkg_type)
        else:
            return '{}.{}.{}'.format(stem, suffix, self._pkg_type)

    def __file_path_taken(self, possible_path):
        for package in self._packages:
            # Do case insensitive compare
            if package.filename.lower() == possible_path.lower():
                return True
        return False

    def _next_default_file_path(self):
        possible_path = self.__default_file_path_base(self._cpparent.filename)
        suffix = 0
        while self.__file_path_taken(possible_path):
            possible_path = self.__default_file_path_base(
                self._cpparent.filename, suffix)
            suffix += 1
        return possible_path

    def _init_package(self, package, fname):
        # clear out existing packages
        self._remove_packages()
        if fname is None:
            # build a file name
            fname = self._next_default_file_path()
            package._filename = fname
        # set file record variable
        self._filerecord.set_data(fname, autofill=True)
        # add the package to the list
        self._packages.append(package)

    def _update_filename(self, old_fname, new_fname):
        file_record = self._filerecord.get_data()
        new_file_record_data = []
        if file_record is not None:
            file_record_data = file_record[0]
            for item in file_record_data:
                if item.lower() == old_fname.lower():
                    new_file_record_data.append((new_fname,))
                else:
                    new_file_record_data.append((item,))
        else:
            new_file_record_data.append((new_fname,))
        self._filerecord.set_data(new_file_record_data)

    def _append_package(self, package, fname, update_frecord=True):
        if fname is None:
            # build a file name
            fname = self._next_default_file_path()
            package._filename = fname

        if update_frecord:
            # set file record variable
            file_record = self._filerecord.get_data()
            file_record_data = file_record[0]
            new_file_record_data = []
            for item in file_record_data:
                new_file_record_data.append((item,))
            new_file_record_data.append((fname,))
            self._filerecord.set_data(new_file_record_data)

        # add the package to the list
        self._packages.append(package)

    def _remove_packages(self):
        for package in self._packages:
            self._model.remove_package(package)
        self._packages = []