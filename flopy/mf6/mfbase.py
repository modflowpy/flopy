import glob
import importlib
import inspect
import os, collections, copy
from shutil import copyfile
from enum import Enum


class PackageContainerType(Enum):
    simulation = 1
    model = 2
    package = 3


class ExtFileAction(Enum):
    copy_all = 1
    copy_none = 2
    copy_relative_paths = 3


class MFFilePath(object):
    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = {model_name:0}

    def isabs(self):
        return os.path.isabs(self.file_path)


class MFFileMgmt(object):
    """
    Class containing MODFLOW path data

    Parameters
    ----------
    path : string
        path on disk to the simulation

    Attributes
    ----------
    sim_path : string
        path to the simulation
    model_relative_path : OrderedDict
        dictionary of relative paths to each model folder

    Methods
    -------
    get_model_path : (key : string) : string
        returns the model working path for the model key
    set_sim_path
        sets the simulation working path
    """
    def __init__(self, path):
        self._sim_path = ''
        self.set_sim_path(path)

        # keys:fully pathed filenames, vals:FilePath instances
        self.existing_file_dict = {}
        # keys:filenames,vals:instance name

        self.model_relative_path = collections.OrderedDict()

        self._last_loaded_sim_path = None
        self._last_loaded_model_relative_path = collections.OrderedDict()

    def copy_files(self, copy_relative_only=True):
        num_files_copied = 0
        if self._last_loaded_sim_path is not None:
            for key, mffile_path in self.existing_file_dict.items():
                for model_name in mffile_path.model_name:
                    if model_name in self._last_loaded_model_relative_path:
                        if os.path.isfile(self.resolve_path(mffile_path,
                                                            model_name,
                                                            True)) and \
                          (not mffile_path.isabs() or not copy_relative_only):
                            if not os.path.exists(
                              self.resolve_path(mffile_path, model_name)):
                                copyfile(self.resolve_path(mffile_path,
                                                           model_name, True),
                                         self.resolve_path(mffile_path,
                                                           model_name))
                                num_files_copied += 1
        print('INFORMATION: {} external files copied'.format(num_files_copied))

    def get_updated_path(self, external_file_path, model_name,
                         ext_file_action):
        external_file_path = self.string_to_file_path(external_file_path)
        if ext_file_action == ExtFileAction.copy_all:
            if os.path.isabs(external_file_path):
                # move file path to local model or simulation path
                base_path, file_name = os.path.split(external_file_path)
                if model_name:
                    return os.path.join(self.get_model_path(model_name),
                                        file_name)
                else:
                    return os.path.join(self.get_sim_path(), file_name)
            else:
                return external_file_path
        elif ext_file_action == ExtFileAction.copy_relative_paths:
            return external_file_path
        elif ext_file_action == ExtFileAction.copy_none:
            if os.path.isabs(external_file_path):
                return external_file_path
            else:
                return os.path.join(self._build_relative_path(model_name),
                                    external_file_path)
        else:
            return None

    def _build_relative_path(self, model_name):
        old_abs_path = self.resolve_path('', model_name, True)
        current_abs_path = self.resolve_path('', model_name, False)
        return os.path.relpath(old_abs_path, current_abs_path)

    def strip_model_relative_path(self, model_name, path):
        if model_name in self.model_relative_path:
            model_rel_path = self.model_relative_path[model_name]
            new_path = None
            while path:
                path, leaf = os.path.split(path)
                if leaf != model_rel_path:
                    if new_path:
                        new_path = os.path.join(leaf, new_path)
                    else:
                        new_path = leaf
            return new_path

    @staticmethod
    def unique_file_name(file_name, lookup):
        num = 0
        while MFFileMgmt._build_file(file_name, num) in lookup:
            num += 1
        return MFFileMgmt._build_file(file_name, num)

    @staticmethod
    def _build_file(file_name, num):
        file, ext = os.path.splitext(file_name)
        if ext:
            return '{}_{}{}'.format(file, num, ext)
        else:
            return '{}_{}'.format(file, num)

    @staticmethod
    def string_to_file_path(fp_string):
        file_delimitiers = ['/','\\']
        new_string = fp_string
        for delimiter in file_delimitiers:
            arr_string = new_string.split(delimiter)
            if len(arr_string) > 1:
                if os.path.isabs(fp_string):
                    new_string = '{}{}{}'.format(arr_string[0], delimiter,
                                                 arr_string[1])
                else:
                    new_string = os.path.join(arr_string[0], arr_string[1])
                if len(arr_string) > 2:
                    for path_piece in arr_string[2:]:
                        new_string = os.path.join(new_string, path_piece)
        return new_string

    def set_last_accessed_path(self):
        self._last_loaded_sim_path = self._sim_path
        for key, item in self.model_relative_path.items():
            self._last_loaded_model_relative_path[key] = copy.deepcopy(item)

    def get_model_path(self, key, last_loaded_path=False):
        if last_loaded_path:
            return os.path.join(self._last_loaded_sim_path,
                                self._last_loaded_model_relative_path[key])
        else:
            if key in self.model_relative_path:
                return os.path.join(self._sim_path, self.model_relative_path[key])
            else:
                return self._sim_path

    def get_sim_path(self, last_loaded_path=False):
        if last_loaded_path:
            return self._last_loaded_sim_path
        else:
            return self._sim_path

    def add_ext_file(self, file_path, model_name):
        if file_path in self.existing_file_dict:
            if model_name not in self.existing_file_dict[file_path].model_name:
                self.existing_file_dict[file_path].model_name[model_name] = 0
        else:
            new_file_path = MFFilePath(file_path,
                                       model_name)
            self.existing_file_dict[file_path] = new_file_path

    def set_sim_path(self, path):
        """
        set the file path to the simulation files

        Parameters
        ----------
        path : string
            full path or relative path from working directory to
            simulation folder

        Returns
        -------

        Examples
        --------
        self.simulation_data.mfdata.set_sim_path()
        """

        # recalculate paths for everything
        # resolve path type
        path = self.string_to_file_path(path)
        if os.path.isabs(path):
            self._sim_path = path
        else:
            # assume path is relative to working directory
            self._sim_path = os.path.join(os.getcwd(), path)

    def resolve_path(self, path, model_name, last_loaded_path=False,
                     move_abs_paths=False):
        if isinstance(path, MFFilePath):
            file_path = path.file_path
        else:
            file_path = path

        if os.path.isabs(file_path):
            # path is an absolute path
            if move_abs_paths:
                if model_name is None:
                    return self.get_sim_path(last_loaded_path)
                else:
                    return self.get_model_path(model_name, last_loaded_path)
            else:
                return file_path
        else:
            # path is a relative path
            if model_name is None:
                return os.path.join(self.get_sim_path(last_loaded_path),
                                    file_path)
            else:
                return os.path.join(self.get_model_path(model_name,
                                                        last_loaded_path),
                                    file_path)


class PackageContainer(object):
    """
    Base class for any class containing packages.

    Parameters
    ----------
    simulation_data : SimulationData
        the simulation's SimulationData object
    name : string
        name of the package container object

    Attributes
    ----------
    packages : list
        packages contained in the package container
    package_type_dict : dictionary
        dictionary of packages by package type
    package_name_dict : dictionary
        dictionary of packages by package name
    package_key_dict : dictionary
        dictionary of packages by package key

    Methods
    -------
    package_factory : (package_type : string, model_type : string) :
      MFPackage subclass
        Static method that returns the appropriate package type object based
        on the package_type and model_type strings
    get_package : (name : string) : MFPackage or [MfPackage]
        finds a package by package name, package key, package type, or partial
        package name. returns either a single package, a list of packages,
        or None
    register_package : (package : MFPackage) : (tuple, PackageStructure)
        base class method for package registration
    """

    def __init__(self, simulation_data, name):
        self.type = 'PackageContainer'
        self.simulation_data = simulation_data
        self.name = name
        self.packages = []
        self.package_type_dict = {}
        self.package_name_dict = {}
        self.package_key_dict = {}

    @staticmethod
    def package_factory(package_type, model_type):
        package_abbr = '{}{}'.format(model_type, package_type)
        package_utl_abbr = 'utl{}'.format(package_type)
        base_path, tail = os.path.split(os.path.realpath(__file__))
        package_path = os.path.join(base_path, 'modflow')
        package_list = []
        # iterate through python files
        package_file_paths = glob.glob(os.path.join(package_path, "*.py"))
        for package_file_path in package_file_paths:
            package_file_name = os.path.basename(package_file_path)
            module_path = os.path.splitext(package_file_name)[0]
            module_name = '{}{}{}'.format('Modflow', module_path[2].upper(),
                                          module_path[3:])
            if module_name.startswith("__"):
                continue

            # import
            module = importlib.import_module("flopy.mf6.modflow.{}".format(
              module_path))

            # iterate imported items
            for item in dir(module):
                value = getattr(module, item)
                # verify this is a class
                if not value or not inspect.isclass(value) or not \
                  hasattr(value, 'package_abbr'):
                    continue
                if package_type is None:
                    package_list.append(value)
                else:
                    # check package type
                    if value.package_abbr == package_abbr or \
                      value.package_abbr == package_utl_abbr:
                        return value
        if package_type is None:
            return package_list
        else:
            return None

    def _add_package(self, package, path):
        # put in packages list and update lookup dictionaries
        self.packages.append(package)
        if package.package_name is not None:
            self.package_name_dict[package.package_name.lower()] = package
        self.package_key_dict[path[-1].lower()] = package
        if package.package_type not in self.package_type_dict:
            self.package_type_dict[package.package_type.lower()] = []
        self.package_type_dict[package.package_type.lower()].append(package)

    def _remove_package(self, package):
        self.packages.remove(package)
        if package.package_name is not None and \
                package.package_name.lower() in self.package_name_dict:
            del self.package_name_dict[package.package_name.lower()]
        del self.package_key_dict[package.path[-1].lower()]
        package_list = self.package_type_dict[package.package_type.lower()]
        package_list.remove(package)
        if len(package_list) == 0:
            del self.package_type_dict[package.package_type.lower()]

        # collect keys of items to be removed from main dictionary
        items_to_remove = []
        for key, data in self.simulation_data.mfdata.items():
            is_subkey = True
            for pitem, ditem in zip(package.path, key):
                if pitem != ditem:
                    is_subkey = False
                    break
            if is_subkey:
                items_to_remove.append(key)

        # remove items from main dictionary
        for key in items_to_remove:
            del self.simulation_data.mfdata[key]

    def get_package(self, name=None):
        """
        Get a package.

        Parameters
        ----------
        name : str
            Name of the package, 'RIV', 'LPF', etc.

        Returns
        -------
        pp : Package object

        """
        if name is None:
            return self.packages[:]

        # search for full package name
        if name.lower() in self.package_name_dict:
            return self.package_name_dict[name.lower()]

        # search for package key
        if name.lower() in self.package_key_dict:
            return self.package_key_dict[name.lower()]

        # search for package type
        if name.lower() in self.package_type_dict:
            if len(self.package_type_dict[name.lower()]) == 0:
                return None
            elif len(self.package_type_dict[name.lower()]) == 1:
                return self.package_type_dict[name.lower()][0]
            else:
                return self.package_type_dict[name.lower()]

        # search for partial package name
        for pp in self.packages:
            # get first package of the type requested
            package_name = pp.package_name.lower()
            if len(package_name) > len(name):
                package_name = package_name[0:len(name)]
            if package_name.lower() == name.lower():
                return pp

        return None

    def register_package(self, package):
        path = (package.package_name,)
        return (path, None)