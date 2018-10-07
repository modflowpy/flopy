"""
mfsimulation module.  contains the MFSimulation class


"""
import errno, sys, inspect
import collections
import os.path
import numpy as np
from flopy.mbase import run_model
from flopy.mf6.mfbase import PackageContainer, MFFileMgmt, ExtFileAction, \
                             PackageContainerType, MFDataException, \
                             FlopyException, VerbosityLevel
from flopy.mf6.mfmodel import MFModel
from flopy.mf6.mfpackage import MFPackage
from flopy.mf6.data.mfstructure import DatumType
from flopy.mf6.data import mfstructure, mfdata
from flopy.mf6.utils import binaryfile_utils
from flopy.mf6.utils import mfobservation
from flopy.mf6.modflow import mfnam, mfims, mftdis, mfgwfgnc, mfgwfmvr


class SimulationDict(collections.OrderedDict):
    """
    Class containing custom dictionary for MODFLOW simulations.  Behaves as an
    OrderedDict with some additional features described below.

    Parameters
    ----------
    path : MFFileMgmt
        object containing path information for the simulation

    Methods
    -------
    output_keys : (print_keys: boolean) : list
        returns a list of output data keys the dictionary supports for output
        data, print_keys allows those keys to be printed to output.
    input_keys : ()
        prints all input data keys
    observation_keys : ()
        prints observation keys
    keys : ()
        print all keys, input and output
    plot : (key : string, **kwargs)
        plots data with key 'key' using **kwargs for plot options
    shapefile : (key : string, **kwargs)
        create shapefile from data with key 'key' and with additional fields
        in **kwargs
    """
    def __init__(self, path, *args):
        self._path = path
        collections.OrderedDict.__init__(self)

    def __getitem__(self, key):
        # check if the key refers to a binary output file, or an observation
        # output file, if so override the dictionary request and call output
        #  requester classes

        # FIX: Transport - Include transport output files
        if key[1] in ('CBC', 'HDS', 'DDN', 'UCN'):
            val = binaryfile_utils.MFOutput(self, self._path, key)
            return val.data

        elif key[-1] == 'Observations':
            val = mfobservation.MFObservation(self, self._path, key)
            return val.data

        val = collections.OrderedDict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        collections.OrderedDict.__setitem__(self, key, val)

    def find_in_path(self, key_path, key_leaf):
        key_path_size = len(key_path)
        for key, item in self.items():
            if key[:key_path_size] == key_path:
                if key[-1] == key_leaf:
                    # found key_leaf as a key in the dictionary
                    return item, None
                if not isinstance(item, mfdata.MFComment):
                    data_item_index = 0
                    data_item_structures = item.structure.data_item_structures
                    for data_item_struct in data_item_structures:
                        if data_item_struct.name == key_leaf:
                            # found key_leaf as a data item name in the data in
                            # the dictionary
                            return item, data_item_index
                        if data_item_struct.type != DatumType.keyword:
                            data_item_index += 1
        return None, None

    def output_keys(self, print_keys=True):
        # get keys to request binary output
        x = binaryfile_utils.MFOutputRequester.getkeys(self, self._path,
                                                       print_keys=print_keys)
        return [key for key in x.dataDict]

    def input_keys(self):
        # get keys to request input ie. package data
        for key in self:
            print(key)

    def observation_keys(self):
        # get keys to request observation file output
        mfobservation.MFObservationRequester.getkeys(self, self._path)

    def keys(self):
        # overrides the built in keys to print all keys, input and output
        self.input_keys()
        try:
            self.output_keys()
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
        try:
            self.observation_keys()
        except KeyError:
            pass


class MFSimulationData(object):
    """
    Class containing MODFLOW simulation data and file formatting data.

    Parameters
    ----------
    path : string
        path on disk to the simulation

    Attributes
    ----------
    indent_string : string
        string used to define how much indent to use (file formatting)
    internal_formatting : list
        list defining string to use for internal formatting
    external_formatting : list
        list defining string to use for external formatting
    open_close_formatting : list
        list defining string to use for open/close
    max_columns_of_data : int
        maximum columns of data before line wraps
    wrap_multidim_arrays : bool
        whether to wrap line for multi-dimensional arrays at the end of a
        row/column/layer
    float_precision : int
        number of decimal points to write for a floating point number
    float_characters : int
        number of characters a floating point number takes up
    sci_note_upper_thres : float
        numbers greater than this threshold are written in scientific notation
    sci_note_lower_thres : float
        numbers less than this threshold are written in scientific notation
    mfpath : MFFileMgmt
        file path location information for the simulation
    model_dimensions : OrderedDict
        dictionary containing discretization information for each model
    mfdata : SimulationDict
        custom dictionary containing all model data for the simulation
    """
    def __init__(self, path):
        # --- formatting variables ---
        self.indent_string = '  '
        self.constant_formatting = ['constant', '']
        self.max_columns_of_data = 20
        self.wrap_multidim_arrays = True
        self.float_precision = 8
        self.float_characters = 15
        self._sci_note_upper_thres = 100000
        self._sci_note_lower_thres = 0.001
        self.fast_write = True
        self.verify_external_data = True
        self.comments_on = False
        self.auto_set_sizes = True
        self.debug = False
        self.verbose = True
        self.verbosity_level = VerbosityLevel.normal

        self._update_str_format()

        # --- file path ---
        self.mfpath = MFFileMgmt(path)

        # --- ease of use variables to make working with modflow input and
        # output data easier --- model dimension class for each model
        self.model_dimensions = collections.OrderedDict()

        # --- model data ---
        self.mfdata = SimulationDict(self.mfpath)

        # --- temporary variables ---
        # other external files referenced
        self.referenced_files = collections.OrderedDict()

    def set_sci_note_upper_thres(self, value):
        self._sci_note_upper_thres = value
        self._update_str_format()

    def set_sci_note_lower_thres(self, value):
        self._sci_note_lower_thres = value
        self._update_str_format()

    def _update_str_format(self):
        self.reg_format_str = '{:.%dE}' % \
                               self.float_precision
        self.sci_format_str = '{:%d.%df' \
                              '}' % (self.float_characters,
                                     self.float_precision)


class MFSimulation(PackageContainer):
    """
    MODFLOW Simulation Class.  Entry point into any MODFLOW simulation.

    Parameters
    ----------
    sim_name : string
        name of the simulation.
    sim_nam_file : string
        relative to the simulation name file from the simulation working
        folder.
    version : string
        MODFLOW version
    exe_name : string
        relative path to MODFLOW executable from the simulation working folder
    sim_ws : string
        path to simulation working folder
    sim_tdis_file : string
        relative path to MODFLOW TDIS file

    Attributes
    ----------
    sim_name : string
        name of the simulation
    models : OrderedDict
        all models in the simulation
    exchanges : list
        all exchange packages in the simulation
    imsfiles : list
        all ims packages in the simulation
    mfdata : OrderedDict
        all variables defined in the simulation.  the key for a variable is
        defined as a tuple.  for "simulation level" packages the tuple
        starts with the package type, followed by the block name, followed
        by the variable name ("TDIS", "DIMENSIONS", "nper").  for "model level"
        packages the tuple starts with the model name, followed by the package
        name, followed by the block name, followed by the variable name (
        "MyModelName", "DIS6", "OPTIONS", "length_units").
    name_file : MFPackage
        simulation name file
    tdis_file
        simulation tdis file

    Methods
    -------
    load : (sim_name : string, sim_name_file : string, version : string,
            exe_name : string, sim_ws : string, strict : boolean,
            verbosity_level : VerbosityLevel) :
            MFSimulation
        a class method that loads a simulation from files
    write_simulation
        writes the simulation to files
    set_sim_path : (path : string)
        set the file path to the root simulation folder and updates all model
        file paths
    get_model : (model_name : string)
              : [MFModel]
        returns the models in the simulation with a given model name, name file
        name, or model type
    add_model : (model : MFModel, sln_group : integer)
        add model to the simulation
    remove_model : (model_name : string)
        remove model from the simulation
    get_package : (type : string)
        returns a simulation package based on package type
    add_package : (package : MFPackage)
        adds a simulation package to the simulation
    remove_package : (package_name : string)
        removes package from the simulation.  package_name can be the
        package's name, type, or package object to be removed from
        the model
    is_valid : () : boolean
        checks the validity of the solution and all of its models and packages

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> s = flopy6.mfsimulation.load('my simulation', 'simulation.nam')

    """
    def __init__(self, sim_name='sim', version='mf6',
                 exe_name='mf6.exe', sim_ws='.'):
        super(MFSimulation, self).__init__(MFSimulationData(sim_ws), sim_name)
        # verify metadata
        fpdata = mfstructure.MFStructure()
        if not fpdata.valid:
            excpt_str = 'Invalid package metadata.  Unable to load MODFLOW ' \
                        'file structure metadata.'
            raise FlopyException(excpt_str)

        # initialize
        self.dimensions = None
        self.type = 'Simulation'

        self.version = version
        self.exe_name = exe_name
        self._models = collections.OrderedDict()
        self._tdis_file = None
        self._exchange_files = collections.OrderedDict()
        self._ims_files = collections.OrderedDict()
        self._ghost_node_files = {}
        self._mover_files = {}
        self._other_files = collections.OrderedDict()
        self.structure = fpdata.sim_struct

        self._exg_file_num = {}
        self._gnc_file_num = 0
        self._mvr_file_num = 0

        self.simulation_data.mfpath.set_last_accessed_path()

        # build simulation name file
        self.name_file = mfnam.ModflowNam(self, fname='mfsim.nam')

        # try to build directory structure
        try:
            os.makedirs(self.simulation_data.mfpath.get_sim_path())
        except OSError as e:
            if e.errno == errno.EEXIST and \
                self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('Directory structure already exists for simulation path '
                      '{}'.format(self.simulation_data.mfpath.get_sim_path()))

        # set simulation validity initially to false since the user must first
        # add at least one model to the simulation and fill out the name and
        #  tdis files
        self.valid = False

    def __getattr__(self, item):
        """
        __getattr__ - used to allow for getting models and packages as if
        they are attributes

        Parameters
        ----------
        item : str
            model or package name


        Returns
        -------
        md : Model or package object
            Model or package object of type :class:flopy6.mfmodel or
            :class:flopy6.mfpackage

        """

        models = []
        if item in self.structure.model_types:
            # get all models of this type
            for model in self._models.values():
                if model.model_type == item:
                    models.append(model)

        if len(models) > 0:
            return models
        elif item in self._models:
            return self.get_model(item)
        else:
            return self.get_package(item)

    def __repr__(self):
        return self._get_data_str(True)

    def __str__(self):
        return self._get_data_str(False)

    def _get_data_str(self, formal):
        file_mgt = self.simulation_data.mfpath
        data_str = 'sim_name = {}\nsim_path = {}\nexe_name = ' \
                   '{}\n\n'.format(self.name, file_mgt.get_sim_path(),
                                   self.exe_name)

        for package in self.packagelist:
            pk_str = package._get_data_str(formal, False)
            if formal:
                if len(pk_str.strip()) > 0:
                    data_str = '{}###################\nPackage {}\n' \
                               '###################\n\n' \
                               '{}\n'.format(data_str, package._get_pname(),
                                             pk_str)
            else:
                if len(pk_str.strip()) > 0:
                    data_str = '{}###################\nPackage {}\n' \
                               '###################\n\n' \
                               '{}\n'.format(data_str, package._get_pname(),
                                             pk_str)
        for idx, model in self._models.items():
            if formal:
                mod_repr = repr(model)
                if len(mod_repr.strip()) > 0:
                    data_str = '{}@@@@@@@@@@@@@@@@@@@@\nModel {}\n' \
                               '@@@@@@@@@@@@@@@@@@@@\n\n' \
                               '{}\n'.format(data_str, model.name, mod_repr)
            else:
                mod_str = str(model)
                if len(mod_str.strip()) > 0:
                    data_str = '{}@@@@@@@@@@@@@@@@@@@@\nModel {}\n' \
                               '@@@@@@@@@@@@@@@@@@@@\n\n' \
                               '{}\n'.format(data_str, model.name, mod_str)
        return data_str

    @classmethod
    def load(cls, sim_name='modflowsim', version='mf6', exe_name='mf6.exe',
             sim_ws='.', strict=True, verbosity_level=VerbosityLevel.normal):
        """
        Load an existing model.

        Parameters
        ----------
        sim_name : string
            name of the simulation.
        sim_nam_file : string
            relative to the simulation name file from the simulation working
            folder.
        version : string
            MODFLOW version
        exe_name : string
            relative path to MODFLOW executable from the simulation working
            folder
        sim_ws : string
            path to simulation working folder
        strict : boolean
            strict enforcement of file formatting
        verbosity_level : VerbosityLevel
            verbosity level of console output messages
        Returns
        -------
        sim : MFSimulation object

        Examples
        --------
        >>> s = flopy6.mfsimulation.load('my simulation')
        """
        # initialize
        instance = cls(sim_name, version, exe_name, sim_ws)
        instance.simulation_data.verbosity_level = verbosity_level

        if verbosity_level.value >= VerbosityLevel.normal.value:
            print('loading simulation...')

        # load simulation name file
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print('  loading simulation name file...')
        instance.name_file.load(strict)

        # load TDIS file
        tdis_pkg = 'tdis{}'.format(mfstructure.MFStructure().
                                   get_version_string())
        tdis_attr = getattr(instance.name_file, tdis_pkg)
        instance._tdis_file = mftdis.ModflowTdis(instance,
                                                 fname=tdis_attr.get_data())

        instance._tdis_file.filename = instance.simulation_data.mfdata[
            ('nam', 'timing', tdis_pkg)].get_data()
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print('  loading tdis package...')
        instance._tdis_file.load(strict)

        # load models
        try:
            model_recarray = instance.simulation_data.mfdata[('nam', 'models',
                                                              'models')]
            models = model_recarray.get_data()
        except MFDataException as mfde:
            message = 'Error occurred while loading model names from the ' \
                      'simulation name file.'
            raise MFDataException(mfdata_except=mfde,
                                  model=instance.name,
                                  package='nam',
                                  message=message)

        for item in models:
            # resolve model working folder and name file
            path, name_file = os.path.split(item[1])
            model_obj = PackageContainer.model_factory(item[0][:-1].lower())
            # load model
            if verbosity_level.value >= VerbosityLevel.normal.value:
                print('  loading model {}...'.format(item[0].lower()))
            instance._models[item[2]] = model_obj.load(
                instance,
                instance.structure.model_struct_objs[item[0].lower()], item[2],
                name_file, version, exe_name, strict, path)

        # load exchange packages and dependent packages
        try:
            exchange_recarray = instance.name_file.exchanges
            has_exch_data = exchange_recarray.has_data()
        except MFDataException as mfde:
            message = 'Error occurred while loading exchange names from the ' \
                      'simulation name file.'
            raise MFDataException(mfdata_except=mfde,
                                  model=instance.name,
                                  package='nam',
                                  message=message)
        if has_exch_data:
            try:
                exch_data = exchange_recarray.get_data()
            except MFDataException as mfde:
                message = 'Error occurred while loading exchange names from the ' \
                          'simulation name file.'
                raise MFDataException(mfdata_except=mfde,
                                      model=instance.name,
                                      package='nam',
                                      message=message)
            for exgfile in exch_data:
                # get exchange type by removing numbers from exgtype
                exchange_type = ''.join([char for char in exgfile[0] if
                                         not char.isdigit()]).upper()
                # get exchange number for this type
                if not exchange_type in instance._exg_file_num:
                    exchange_file_num = 0
                    instance._exg_file_num[exchange_type] = 1
                else:
                    exchange_file_num = instance._exg_file_num[exchange_type]
                    instance._exg_file_num[exchange_type] += 1

                exchange_name = '{}_EXG_{}'.format(exchange_type,
                                                   exchange_file_num)
                # find package class the corresponds to this exchange type
                package_obj = instance.package_factory(
                    exchange_type.replace('-', '').lower(), '')
                if not package_obj:
                    message = 'An error occurred while loading the ' \
                              'simulation name file.  Invalid exchange type ' \
                              '"{}" specified.'.format(exchange_type)
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(instance.name,
                                          'nam',
                                          'nam',
                                          'loading simulation name file',
                                          exchange_recarray.structure.name,
                                          inspect.stack()[0][3],
                                          type_, value_, traceback_, message,
                                          instance._simulation_data.debug)

                # build and load exchange package object
                exchange_file = package_obj(instance, exgtype=exgfile[0],
                                            exgmnamea=exgfile[2],
                                            exgmnameb=exgfile[3],
                                            fname=exgfile[1],
                                            pname=exchange_name,
                                            loading_package=True)
                if verbosity_level.value >= VerbosityLevel.normal.value:
                    print('  loading exchange package {}..'
                          '.'.format(exchange_file._get_pname()))
                exchange_file.load(strict)
                instance._exchange_files[exgfile[1]] = exchange_file

        # load simulation packages
        solution_recarray = instance.simulation_data.mfdata[('nam',
                                                             'solutiongroup',
                                                             'solutiongroup'
                                                             )]

        try:
            solution_group_list = solution_recarray.get_data()
        except MFDataException as mfde:
            message = 'Error occurred while loading solution groups from ' \
                      'the simulation name file.'
            raise MFDataException(mfdata_except=mfde,
                                  model=instance.name,
                                  package='nam',
                                  message=message)
        for solution_group in solution_group_list:
            for solution_info in solution_group:
                ims_file = mfims.ModflowIms(instance, fname=solution_info[1],
                                            pname=solution_info[2])
                if verbosity_level.value >= VerbosityLevel.normal.value:
                    print('  loading ims package {}..'
                          '.'.format(ims_file._get_pname()))
                ims_file.load(strict)

        instance.simulation_data.mfpath.set_last_accessed_path()
        return instance

    def load_package(self, ftype, fname, pname, strict, ref_path,
                     dict_package_name=None, parent_package=None):
        """
        loads a package from a file

        Parameters
        ----------
        ftype : string
            the file type
        fname : string
            the name of the file containing the package input
        pname : string
            the user-defined name for the package
        strict : bool
            strict mode when loading the file
        ref_path : string
            path to the file. uses local path if set to None
        dict_package_name : string
            package name for dictionary lookup
        parent_package : MFPackage
            parent package

        Examples
        --------
        """
        if ftype == 'gnc':
            if fname not in self._ghost_node_files:
                # get package type from parent package
                if parent_package:
                    package_abbr = parent_package.package_abbr[0:3]
                else:
                    package_abbr = 'GWF'
                # build package name and package
                gnc_name = '{}-GNC_{}'.format(package_abbr, self._gnc_file_num)
                ghost_node_file = mfgwfgnc.ModflowGwfgnc(self, fname=fname,
                                                         pname=gnc_name,
                                                         parent_file=
                                                         parent_package,
                                                         loading_package=True)
                ghost_node_file.load(strict)
                self._ghost_node_files[fname] = ghost_node_file
                self._gnc_file_num += 1
        elif ftype == 'mvr':
            if fname not in self._mover_files:
                # Get package type from parent package
                if parent_package:
                    package_abbr = parent_package.package_abbr[0:3]
                else:
                    package_abbr = 'GWF'
                # build package name and package
                mvr_name = '{}-MVR_{}'.format(package_abbr, self._mvr_file_num)
                mover_file = mfgwfmvr.ModflowGwfmvr(self, fname=fname,
                                                    pname=mvr_name,
                                                    parent_file=parent_package,
                                                    loading_package=True)
                mover_file.load(strict)
                self._mover_files[fname] = mover_file
                self._mvr_file_num += 1
        else:
            # create package
            package_obj = self.package_factory(ftype, '')
            package = package_obj(self, fname=fname, pname=dict_package_name,
                                  add_to_package_list=False,
                                  parent_file=parent_package,
                                  loading_package=True)
            # verify that this is a utility package
            utl_struct = mfstructure.MFStructure().sim_struct.utl_struct_objs
            if package.package_type in utl_struct:
                package.load(strict)
                self._other_files[package.filename] = package
                # register child package with the simulation
                self._add_package(package, package.path)
                if parent_package is not None:
                    # register child package with the parent package
                    parent_package._add_package(package, package.path)
            else:
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: Unsupported file type {} for '
                          'simulation.'.format(package.package_type))

    def register_ims_package(self, ims_file, model_list):
        """
        registers an ims package with the simulation

        Parameters
        ----------
        ims_file : MFPackage
            ims package to register
        model_list : list of strings
            list of models using the ims package to be registered

        Examples
        --------
        """
        if isinstance(model_list, str):
            model_list = [model_list]

        if not isinstance(ims_file, mfims.ModflowIms):
            comment = 'Parameter "ims_file" is not a valid ims file.  ' \
                      'Expected type ModflowIms, but type "{}" was given' \
                      '.'.format(type(ims_file))
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(None,
                                  'ims',
                                  '',
                                  'registering ims package',
                                  '',
                                  inspect.stack()[0][3], type_,
                                  value_,
                                  traceback_, comment,
                                  self.simulation_data.debug)

        in_simulation = False
        pkg_with_same_name = None
        for index, file in self._ims_files.items():
            if file is ims_file:
                in_simulation = True
            if file.package_name == ims_file.package_name and \
                    file != ims_file:
                pkg_with_same_name = file
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: ims package with name {} already exists. '
                          'New ims package will replace old package'
                          '.'.format(file.package_name))
                self._remove_package(self._ims_files[file.filename])
                del self._ims_files[file.filename]
        # register ims package
        if not in_simulation:
            self._add_package(ims_file, self._get_package_path(ims_file))
        # do not allow an ims package to be registered twice with the
        # same simulation
        if not in_simulation:
            # create unique file/package name
            if ims_file.package_name is None:
                file_num = len(self._ims_files) - 1
                ims_file.package_name = 'ims_{}'.format(file_num)
            if ims_file.filename in self._ims_files:
                ims_file.filename = MFFileMgmt.unique_file_name(
                    ims_file.filename, self._ims_files)
            # add ims package to simulation
            self._ims_files[ims_file.filename] = ims_file

        # If ims file is being replaced, replace ims filename in solution group
        if pkg_with_same_name is not None and \
                self._is_in_solution_group(pkg_with_same_name.filename, 1):
            # change existing solution group to reflect new ims file
            self._replace_ims_in_solution_group(pkg_with_same_name.filename,
                                                1, ims_file.filename)
        # only allow an ims package to be registered to one solution group
        elif model_list is not None:
            ims_in_group = self._is_in_solution_group(ims_file.filename, 1)
            # add solution group to the simulation name file
            solution_recarray = self.name_file.solutiongroup
            solution_group_list = solution_recarray.get_active_key_list()
            if len(solution_group_list) == 0:
                solution_group_num = 0
            else:
                solution_group_num = solution_group_list[-1][0]

            if ims_in_group:
                self._append_to_ims_solution_group(ims_file.filename,
                                                   model_list)
            else:
                if self.name_file.mxiter.get_data(solution_group_num) is None:
                    self.name_file.mxiter.add_transient_key(solution_group_num)

                # associate any models in the model list to this simulation file
                version_string = mfstructure.MFStructure().get_version_string()
                ims_pkg = 'ims{}'.format(version_string)
                new_record = [ims_pkg, ims_file.filename]
                for model in model_list:
                    new_record.append(model)
                try:
                    solution_recarray.append_list_as_record(new_record,
                                                            solution_group_num)
                except MFDataException as mfde:
                    message = 'Error occurred while updating the ' \
                              'simulation name file with the ims package ' \
                              'file "{}".'.format(ims_file.filename)
                    raise MFDataException(mfdata_except=mfde,
                                          package='nam',
                                          message=message)

    def write_simulation(self,
                         ext_file_action=ExtFileAction.copy_relative_paths):
        """
        writes the simulation to files

        Parameters
        ----------
        ext_file_action : ExtFileAction
            defines what to do with external files when the simulation path
            has changed.  defaults to copy_relative_paths which copies only
            files with relative paths, leaving files defined by absolute
            paths fixed.

        Examples
        --------
        """
        # write simulation name file
        if self.simulation_data.verbosity_level.value >= \
                VerbosityLevel.normal.value:
            print('writing simulation...')
            print('  writing simulation name file...')
        self.name_file.write(ext_file_action=ext_file_action)

        # write TDIS file
        if self.simulation_data.verbosity_level.value >= \
                VerbosityLevel.normal.value:
            print('  writing simulation tdis package...')
        self._tdis_file.write(ext_file_action=ext_file_action)

        # write ims files
        for index, ims_file in self._ims_files.items():
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('  writing ims package {}...'.format(
                    ims_file._get_pname()))
            ims_file.write(ext_file_action=ext_file_action)

        # write exchange files
        for key, exchange_file in self._exchange_files.items():
            exchange_file.write()
            if hasattr(exchange_file, 'gnc_filerecord') and \
                    exchange_file.gnc_filerecord.has_data():
                try:
                    gnc_file = exchange_file.gnc_filerecord.get_data()[0][0]
                except MFDataException as mfde:
                    message = 'An error occurred while retrieving the ghost ' \
                              'node file record from exchange file ' \
                              '"{}".'.format(exchange_file.filename)
                    raise MFDataException(mfdata_except=mfde,
                                          package=exchange_file._get_pname(),
                                          message=message)
                if gnc_file in self._ghost_node_files:
                    if self.simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        print('  writing gnc package {}...'.format(
                            self._ghost_node_files[gnc_file]._get_pname()))
                    self._ghost_node_files[gnc_file].write(ext_file_action=
                                                           ext_file_action)
                else:
                    if self.simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        print('WARNING: Ghost node file {} not loaded prior to'
                              ' writing. File will not be written'
                              '.'.format(gnc_file))
            if hasattr(exchange_file, 'mvr_filerecord') and \
                    exchange_file.mvr_filerecord.has_data():
                try:
                    mvr_file = exchange_file.mvr_filerecord.get_data()[0][0]
                except MFDataException as mfde:
                    message = 'An error occurred while retrieving the mover ' \
                              'file record from exchange file ' \
                              '"{}".'.format(exchange_file.filename)
                    raise MFDataException(mfdata_except=mfde,
                                          package=exchange_file._get_pname(),
                                          message=message)

                if mvr_file in self._mover_files:
                    if self.simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        print('  writing mvr package {}...'.format(
                            self._mover_files[mvr_file]._get_pname()))
                    self._mover_files[mvr_file].write(ext_file_action=
                                                      ext_file_action)
                else:
                    if self.simulation_data.verbosity_level.value >= \
                            VerbosityLevel.normal.value:
                        print('WARNING: Mover file {} not loaded prior to '
                              'writing. File will not be '
                              'written.'.format(mvr_file))

        # write other packages
        for index, pp in self._other_files.items():
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('  writing package {}...'.format(pp._get_pname()))
            pp.write(ext_file_action=ext_file_action)

        # FIX: model working folder should be model name file folder

        # write models
        for key, model in self._models.items():
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('  writing model {}...'.format(model.name))
            model.write(ext_file_action=ext_file_action)

        if ext_file_action == ExtFileAction.copy_relative_paths:
            # move external files with relative paths
            num_files_copied = self.simulation_data.mfpath.copy_files()
        elif ext_file_action == ExtFileAction.copy_all:
            # move all external files
            num_files_copied = self.simulation_data.mfpath.copy_files(
                copy_relative_only=False)
        else:
            num_files_copied = 0
        if self.simulation_data.verbosity_level.value >= \
                VerbosityLevel.verbose.value and num_files_copied > 0:
            print('INFORMATION: {} external files copied'.format(
                num_files_copied))
        self.simulation_data.mfpath.set_last_accessed_path()

    def set_sim_path(self, path):
        self.simulation_data.mfpath.set_sim_path(path)

    def run_simulation(self, silent=None, pause=False, report=False,
                       normal_msg='normal termination',
                       use_async=False, cargs=None):
        """
        Run the simulation.
        """
        if silent is None:
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                silent = False
            else:
                silent = True
        return run_model(self.exe_name, None,
                         self.simulation_data.mfpath.get_sim_path(),
                         silent=silent, pause=pause, report=report,
                         normal_msg=normal_msg, use_async=use_async, cargs=cargs)

    def delete_output_files(self):
        """
        Delete simulation output files.
        """
        output_req = binaryfile_utils.MFOutputRequester
        output_file_keys = output_req.getkeys(self.simulation_data.mfdata,
                                              self.simulation_data.mfpath,
                                              False)
        for key, path in output_file_keys.binarypathdict.items():
            if os.path.isfile(path):
                os.remove(path)

    def remove_package(self, package_name):
        if isinstance(package_name, MFPackage):
            packages = [package_name]
        else:
            packages = self.get_package(package_name)
            if not isinstance(packages, list):
                packages = [packages]
        for package in packages:
            if self._tdis_file is not None and \
                    package.path == self._tdis_file.path:
                self._tdis_file = None
            if package.filename in self._exchange_files:
                del self._exchange_files[package.filename]
            if package.filename in self._ims_files:
                del self._ims_files[package.filename]
            if package.filename in self._ghost_node_files:
                del self._ghost_node_files[package.filename]
            if package.filename in self._mover_files:
                del self._mover_files[package.filename]
            if package.filename in self._other_files :
                del self._other_files[package.filename]

            self._remove_package(package)

    @property
    def model_dict(self):
        return self._models.copy()

    @property
    def model_names(self):
        return list(self._models.keys())

    def get_model(self, model_name=''):
        """
        Load an existing model.

        Parameters
        ----------
        model_name : string
            name of model to get

        Returns
        -------
        model : MFModel

        Examples
        --------
        """

        return self._models[model_name]

    def get_exchange_file(self, filename):
        """
        get a specified exchange file

        Parameters
        ----------
        filename : string
            name of exchange file to get

        Returns
        -------
        exchange package : MFPackage

        Examples
        --------
        """
        if filename in self._exchange_files:
            return self._exchange_files[filename]
        else:
            excpt_str = 'Exchange file "{}" can not be found' \
                        '.'.format(filename)
            raise FlopyException(excpt_str)

    def get_mvr_file(self, filename):
        """
        get a specified mover file

        Parameters
        ----------
        filename : string
            name of mover file to get

        Returns
        -------
        mover package : MFPackage

        Examples
        --------
        """
        if filename in self._mover_files:
            return self._mover_files[filename]
        else:
            excpt_str = 'MVR file "{}" can not be ' \
                        'found.'.format(filename)
            raise FlopyException(excpt_str)

    def get_gnc_file(self, filename):
        """
        get a specified gnc file

        Parameters
        ----------
        filename : string
            name of gnc file to get

        Returns
        -------
        gnc package : MFPackage

        Examples
        --------
        """
        if filename in self._ghost_node_files:
            return self._ghost_node_files[filename]
        else:
            excpt_str = 'GNC file "{}" can not be ' \
                        'found.'.format(filename)
            raise FlopyException(excpt_str)

    def register_exchange_file(self, package):
        """
        register an exchange package file with the simulation

        Parameters
        ----------
        package : MFPackage
            exchange package object to register

        Examples
        --------
        """
        if package.filename not in self._exchange_files:
            exgtype = package.exgtype
            exgmnamea = package.exgmnamea
            exgmnameb = package.exgmnameb

            if exgtype is None or exgmnamea is None or exgmnameb is None:
                excpt_str = 'Exchange packages require that exgtype, ' \
                            'exgmnamea, and exgmnameb are specified.'
                raise FlopyException(excpt_str)

            self._exchange_files[package.filename] = package
            try:
                exchange_recarray_data = self.name_file.exchanges.get_data()
            except MFDataException as mfde:
                message = 'An error occurred while retrieving exchange ' \
                          'data from the simulation name file.  The error ' \
                          'occurred while registering exchange file ' \
                          '"{}".'.format(package.filename)
                raise MFDataException(mfdata_except=mfde,
                                      package=package._get_pname(),
                                      message=message)
            if exchange_recarray_data is not None:
                for index, exchange in zip(range(0,
                                                 len(exchange_recarray_data)),
                                           exchange_recarray_data):
                    if exchange[1] == package.filename:
                        # update existing exchange
                        exchange_recarray_data[index][0] = exgtype
                        exchange_recarray_data[index][2] = exgmnamea
                        exchange_recarray_data[index][3] = exgmnameb
                        ex_recarray = self.name_file.exchanges
                        try:
                            ex_recarray.set_data(exchange_recarray_data)
                        except MFDataException as mfde:
                            message = 'An error occurred while setting ' \
                                      'exchange data in the simulation name ' \
                                      'file.  The error occurred while ' \
                                      'registering the following ' \
                                      'values (exgtype, filename, ' \
                                      'exgmnamea, exgmnameb): "{} {} {}' \
                                      '{}".'.format(exgtype, package.filename,
                                                    exgmnamea, exgmnameb)
                            raise MFDataException(mfdata_except=mfde,
                                                  package=package._get_pname(),
                                                  message=message)
                        return
            try:
                # add new exchange
                self.name_file.exchanges.append_data([(exgtype,
                                                       package.filename,
                                                       exgmnamea,
                                                       exgmnameb)])
            except MFDataException as mfde:
                message = 'An error occurred while setting exchange data ' \
                          'in the simulation name file.  The error occurred ' \
                          'while registering the following values (exgtype, ' \
                          'filename, exgmnamea, exgmnameb): "{} {} {}' \
                          '{}".'.format(exgtype, package.filename, exgmnamea,
                                        exgmnameb)
                raise MFDataException(mfdata_except=mfde,
                                      package=package._get_pname(),
                                      message=message)
            if package.dimensions is None:
                # resolve exchange package dimensions object
                package.dimensions = package.create_package_dimensions()

    def register_package(self, package, add_to_package_list=True,
                         set_package_name=True, set_package_filename=True):
        """
        register a package file with the simulation

        Parameters
        ----------
        package : MFPackage
            package to register
        add_to_package_list : bool
            add package to lookup list
        set_package_name : bool
            produce a package name for this package
        set_package_filename : bool
            produce a filename for this package

        Returns
        -------
        (path : tuple, package structure : MFPackageStructure)

        Examples
        --------
        """
        package.container_type = [PackageContainerType.simulation]
        path = self._get_package_path(package)
        if add_to_package_list and package.package_type.lower != 'nam':
            pname = None
            if package.package_name is not None:
                pname = package.package_name.lower()
            if package.package_type.lower() == 'tdis' and self._tdis_file is \
                    not None and self._tdis_file in self.packagelist:
                # tdis package already exists. there can be only one tdis
                # package.  remove existing tdis package
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: tdis package already exists. Replacing '
                          'existing tdis package.')
                self._remove_package(self._tdis_file)
            elif package.package_type.lower() == 'gnc' and \
                    package.filename in self._ghost_node_files and \
                    self._ghost_node_files[package.filename] in self.packagelist:
                # gnc package with same file name already exists.  remove old
                # gnc package
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: gnc package with name {} already exists. '
                          'Replacing existing gnc package'
                          '.'.format(pname))
                self._remove_package(self._ghost_node_files[package.filename])
                del self._ghost_node_files[package.filename]
            elif package.package_type.lower() == 'mvr' and \
                     package.filename in self._mover_files and \
                     self._mover_files[package.filename] in self.packagelist:
                # mvr package with same file name already exists.  remove old
                # mvr package
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: mvr package with name {} already exists. '
                          'Replacing existing mvr package'
                          '.'.format(pname))
                self._remove_package(self._mover_files[package.filename])
                del self._mover_files[package.filename]
            elif package.package_type.lower() != 'ims' and pname in \
                    self.package_name_dict:
                if self.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('WARNING: Package with name {} already exists.  '
                          'Replacing existing package'
                          '.'.format(package.package_name.lower()))
                self._remove_package(self.package_name_dict[pname])
            if package.package_type.lower() != 'ims':
                # all but ims packages get added here.  ims packages are
                # added during ims package registration
                self._add_package(package, path)
        if package.package_type.lower() == 'nam':
            return path, self.structure.name_file_struct_obj
        elif package.package_type.lower() == 'tdis':
            self._tdis_file = package
            struct_root = mfstructure.MFStructure()
            tdis_pkg = 'tdis{}'.format(struct_root.get_version_string())
            tdis_attr = getattr(self.name_file, tdis_pkg)
            try:
                tdis_attr.set_data(package.filename)
            except MFDataException as mfde:
                message = 'An error occurred while setting the tdis package ' \
                          'file name "{}".  The error occurred while ' \
                          'registering the tdis package with the ' \
                          'simulation'.format(package.filename)
                raise MFDataException(mfdata_except=mfde,
                                      package=package._get_pname(),
                                      message=message)
            return path, self.structure.package_struct_objs[
                package.package_type.lower()]
        elif package.package_type.lower() == 'gnc':
            if package.filename not in self._ghost_node_files:
                self._ghost_node_files[package.filename] = package
                self._gnc_file_num += 1
            elif self._ghost_node_files[package.filename] != package:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(package.filename,
                                                        self._ghost_node_files)
                package.filename = file_name
                self._ghost_node_files[file_name] = package
        elif package.package_type.lower() == 'mvr':
            if package.filename not in self._mover_files:
                self._mover_files[package.filename] = package
            else:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(package.filename,
                                                        self._mover_files)
                package.filename = file_name
                self._mover_files[file_name] = package
        elif package.package_type.lower() == 'ims':
            # default behavior is to register the ims package with the first
            # unregistered model
            unregistered_models = []
            for model in self._models:
                model_registered = self._is_in_solution_group(model, 2)
                if not model_registered:
                    unregistered_models.append(model)
            if unregistered_models:
                self.register_ims_package(package, unregistered_models)
            else:
                self.register_ims_package(package, None)
            return path, self.structure.package_struct_objs[
                package.package_type.lower()]

        if package.package_type.lower() in self.structure.package_struct_objs:
            return path, self.structure.package_struct_objs[
                package.package_type.lower()]
        elif package.package_type.lower() in self.structure.utl_struct_objs:
            return path, self.structure.utl_struct_objs[
                package.package_type.lower()]
        else:
            excpt_str = 'Invalid package type "{}".  Unable to register ' \
                        'package.'.format(package.package_type)
            print(excpt_str)
            raise FlopyException(excpt_str)

    def register_model(self, model, model_type, model_name, model_namefile):
        """
        add a model to the simulation.

        Parameters
        ----------
        model : MFModel
            model object to add to simulation
        sln_group : string
            solution group of model

        Returns
        -------
        model_structure_object : MFModelStructure

        Examples
        --------
        """

        # get model structure from model type
        if model_type not in self.structure.model_struct_objs:
            message = 'Invalid model type: "{}".'.format(model_type)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(model.name,
                                  '', model.name,
                                  'registering model', 'sim',
                                  inspect.stack()[0][3],
                                  type_, value_, traceback_, message,
                                  self.simulation_data.debug)

        # add model
        self._models[model_name] = model

        # update simulation name file
        self.name_file.models.append_list_as_record([model_type,
                                                            model_namefile,
                                                            model_name])

        if len(self._ims_files) > 0:
            # register model with first ims file found
            first_ims_key = next(iter(self._ims_files))
            self.register_ims_package(self._ims_files[first_ims_key],
                                      model_name)

        return self.structure.model_struct_objs[model_type]

    def remove_model(self, model_name):
        """
        remove a model from the simulation.

        Parameters
        ----------
        model_name : string
            model name to remove from simulation

        Examples
        --------
        """

        # Remove model
        del self._models[model_name]

        # TODO: Fully implement this
        # Update simulation name file

    def is_valid(self):
        """
        check all packages and models in the simulation to verify validity

        Returns
        ----------
        valid : boolean
            simulation validity

        Examples
        --------
        """

        # name file valid
        if not self.name_file.is_valid():
            return False

        # tdis file valid
        if not self._tdis_file.is_valid():
            return False

        # exchanges valid
        for exchange in self._exchange_files:
            if not exchange.is_valid():
                return False

        # ims files valid
        for index, imsfile in self._ims_files.items():
            if not imsfile.is_valid():
                return False

        # a model exists
        if not self._models:
            return False

        # models valid
        for key in self._models:
            if not self._models[key].is_valid():
                return False

        # each model has an imsfile

        return True

    def _get_package_path(self, package):
        if package.parent_file is not None:
            return (package.parent_file.path) + (package.package_type,)
        else:
            return (package.package_type,)

    def _append_to_ims_solution_group(self, ims_file, new_models):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = 'An error occurred while getting solution group' \
                          '"{}" from the simulation name file.  The error ' \
                          'occurred while replacing IMS file "{}" with "{}"' \
                          'at index "{}"'.format(solution_group_num[0],
                                                 item, new_item, index)
                raise MFDataException(mfdata_except=mfde,
                                      package='nam',
                                      message=message)
            new_array = []
            for index, record in enumerate(rec_array):
                new_record = []
                rec_model_dict = {}
                for index, item in enumerate(record):
                    if record[1] == ims_file or item not in new_models:
                        new_record.append(item)
                        if index > 1:
                            rec_model_dict[item] = 1

                if record[1] == ims_file:
                    for model in new_models:
                        if model not in rec_model_dict:
                            new_record.append(model)

                new_array.append(tuple(new_record))
            solution_recarray.set_data(new_array,
                                       solution_group_num[0])

    def _replace_ims_in_solution_group(self, item, index, new_item):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = 'An error occurred while getting solution group' \
                          '"{}" from the simulation name file.  The error ' \
                          'occurred while replacing IMS file "{}" with "{}"' \
                          'at index "{}"'.format(solution_group_num[0],
                                                 item, new_item, index)
                raise MFDataException(mfdata_except=mfde,
                                      package='nam',
                                      message=message)
            if rec_array is not None:
                for rec_item in rec_array:
                    if rec_item[index] == item:
                        rec_item[index] = new_item

    def _is_in_solution_group(self, item, index):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = 'An error occurred while getting solution group' \
                          '"{}" from the simulation name file.  The error ' \
                          'occurred while verifying file "{}" at index "{}" ' \
                          'is in the simulation name file' \
                          '.'.format(solution_group_num[0], item, index)
                raise MFDataException(mfdata_except=mfde,
                                      package='nam',
                                      message=message)

            if rec_array is not None:
                for rec_item in rec_array:
                    if rec_item[index] == item:
                        return True
        return False

