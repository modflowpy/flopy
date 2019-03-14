"""
mfmodel module.  Contains the MFModel class

"""
import os, sys, inspect, warnings
import numpy as np
from .mfbase import PackageContainer, ExtFileAction, PackageContainerType, \
                    MFDataException, ReadAsArraysException, FlopyException, \
                    VerbosityLevel
from .mfpackage import MFPackage
from .coordinates import modeldimensions
from .data import mfstructure
from ..utils import datautil
from ..discretization.structuredgrid import StructuredGrid
from ..discretization.vertexgrid import VertexGrid
from ..discretization.unstructuredgrid import UnstructuredGrid
from ..discretization.grid import Grid
from flopy.discretization.modeltime import ModelTime
from ..mbase import ModelInterface
from .utils.mfenums import DiscretizationType
from .data import mfstructure, mfdatautil


class MFModel(PackageContainer, ModelInterface):
    """
    MODFLOW Model Class.  Represents a single model in a simulation.

    Parameters
    ----------
    simulation_data : MFSimulationData
        simulation data object
    structure : MFModelStructure
        structure of this type of model
    modelname : string
        name of the model
    model_nam_file : string
        relative path to the model name file from model working folder
    version : string
        version of modflow
    exe_name : string
        model executable name
    model_ws : string
        model working folder path
    disfile : string
        relative path to dis file from model working folder
    grid_type : string
        type of grid the model will use (structured, unstructured, vertices)
    verbose : bool
        verbose setting for model operations (default False)

    Attributes
    ----------
    model_name : string
        name of the model
    exe_name : string
        model executable name
    packages : OrderedDict(MFPackage)
        dictionary of model packages
    _name_file_io : MFNameFile
        name file

    Methods
    -------
    load : (simulation : MFSimulationData, model_name : string,
      namfile : string, type : string, version : string, exe_name : string,
      model_ws : string, strict : boolean) : MFSimulation
        a class method that loads a model from files
    write
        writes the simulation to files
    remove_package : (package_name : string)
        removes package from the model.  package_name can be the
        package's name, type, or package object to be removed from
        the model
    set_model_relative_path : (path : string)
        sets the file path to the model folder and updates all model file paths
    is_valid : () : boolean
        checks the validity of the model and all of its packages

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """
    def __init__(self, simulation, model_type='gwf6', modelname='model',
                 model_nam_file=None, version='mf6',
                 exe_name='mf6.exe', add_to_simulation=True,
                 structure=None, model_rel_path='.', verbose=False, **kwargs):
        super(MFModel, self).__init__(simulation.simulation_data, modelname)
        self.simulation = simulation
        self.simulation_data = simulation.simulation_data
        self.name = modelname
        self.name_file = None
        self._version = version
        self.model_type = model_type
        self.type = 'Model'

        if model_nam_file is None:
            model_nam_file = '{}.nam'.format(modelname)

        self.set_model_relative_path(model_rel_path)
        if add_to_simulation:
            self.structure = simulation.register_model(self, model_type,
                                                       modelname,
                                                       model_nam_file)
        else:
            self.structure = structure
        self.exe_name = exe_name
        self.dimensions = modeldimensions.ModelDimensions(self.name,
                                                          self.simulation_data)
        self.simulation_data.model_dimensions[modelname] = self.dimensions
        self._ftype_num_dict = {}
        self._package_paths = {}
        self._verbose = verbose

        if model_nam_file is None:
            self.model_nam_file = '{}.nam'.format(modelname)
        else:
            self.model_nam_file = model_nam_file

        # check for spatial reference info in kwargs
        xll = kwargs.pop("xll", None)
        yll = kwargs.pop("yll", None)
        self._xul = kwargs.pop("xul", None)
        if self._xul is not None:
            warnings.warn('xul/yul have been deprecated. Use xll/yll instead.',
                          DeprecationWarning)
        self._yul = kwargs.pop("yul", None)
        if self._yul is not None:
            warnings.warn('xul/yul have been deprecated. Use xll/yll instead.',
                          DeprecationWarning)
        rotation = kwargs.pop("rotation", 0.)
        proj4 = kwargs.pop("proj4_str", None)
        # build model grid object
        self._modelgrid = Grid(proj4=proj4, xoff=xll, yoff=yll,
                               angrot=rotation)

        self.start_datetime = None
        # check for extraneous kwargs
        if len(kwargs) > 0:
            kwargs_str = ', '.join(kwargs.keys())
            excpt_str = 'Extraneous kwargs "{}" provided to ' \
                        'MFModel.'.format(kwargs_str)
            raise FlopyException(excpt_str)

        # build model name file
        # create name file based on model type - support different model types
        package_obj = self.package_factory('nam', model_type[0:3])
        if not package_obj:
            excpt_str = 'Name file could not be found for model' \
                        '{}.'.format(model_type[0:3])
            raise FlopyException(excpt_str)

        self.name_file = package_obj(self, filename=self.model_nam_file,
                                     pname=self.name)

    def __getattr__(self, item):
        """
        __getattr__ - used to allow for getting packages as if they are
                      attributes

        Parameters
        ----------
        item : str
            3 character package name (case insensitive)


        Returns
        -------
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        """
        return self.get_package(item)

    def __repr__(self):
        return self._get_data_str(True)

    def __str__(self):
        return self._get_data_str(False)

    def _get_data_str(self, formal):
        file_mgr = self.simulation_data.mfpath
        data_str = 'name = {}\nmodel_type = {}\nversion = {}\nmodel_' \
                   'relative_path = {}' \
                   '\n\n'.format(self.name, self.model_type, self.version,
                                 file_mgr.model_relative_path[self.name])

        for package in self.packagelist:
            pk_str = package._get_data_str(formal, False)
            if formal:
                if len(pk_str.strip()) > 0:
                    data_str = '{}###################\nPackage {}\n' \
                               '###################\n\n' \
                               '{}\n'.format(data_str, package._get_pname(),
                                             pk_str)
            else:
                pk_str = package._get_data_str(formal, False)
                if len(pk_str.strip()) > 0:
                    data_str = '{}###################\nPackage {}\n' \
                               '###################\n\n' \
                               '{}\n'.format(data_str, package._get_pname(),
                                             pk_str)
        return data_str

    @property
    def nper(self):
        try:
            return self.simulation.tdis.nper.array
        except AttributeError:
            return None

    @property
    def modeltime(self):
        # build model time
        tdis = self.simulation.get_package('tdis')
        itmuni = tdis.time_units.get_data()
        start_date_time = tdis.start_date_time.get_data()
        if itmuni is None:
            itmuni = 0
        if start_date_time is None:
            start_date_time = '01-01-1970'
        period_data = tdis.perioddata.get_data()
        data_frame = {'perlen': period_data['perlen'],
                      'nstp': period_data['nstp'],
                      'tsmult': period_data['tsmult']}
        self._model_time = ModelTime(data_frame, itmuni, start_date_time)
        return self._model_time

    @property
    def modelgrid(self):
        if not self._mg_resync:
            return self._modelgrid

        if self.get_grid_type() == DiscretizationType.DIS:
            dis = self.get_package('dis')
            self._modelgrid = StructuredGrid(delc=dis.delc.array, delr=dis.delr.array,
                                  top=dis.top.array, botm=dis.botm.array,
                                  idomain=dis.idomain.array,
                                  lenuni=dis.length_units.array,
                                  proj4=self._modelgrid.proj4,
                                  epsg=self._modelgrid.epsg,
                                  xoff=self._modelgrid.xoffset,
                                  yoff=self._modelgrid.yoffset,
                                  angrot=self._modelgrid.angrot)
        elif self.get_grid_type() == DiscretizationType.DISV:
            disv = self.get_package('disv')
            self._modelgrid = VertexGrid(vertices=disv.vertices.array,
                                         cell2d=disv.cell2d.array,
                                  top=disv.top.array, botm=disv.botm.array,
                                  idomain=disv.idomain.array,
                                  lenuni=disv.length_units.array,
                                  proj4=self._modelgrid.proj4,
                                  epsg=self._modelgrid.epsg,
                                  xoff=self._modelgrid.xoffset,
                                  yoff=self._modelgrid.yoffset,
                                  angrot=self._modelgrid.angrot)
        elif self.get_grid_type() == DiscretizationType.DISU:
            disu = self.get_package('disu')
            iverts = [list(i)[4:] for i in disu.cell2d.array]
            self._modelgrid = UnstructuredGrid(vertices=np.array(disu.vertices.array),
                                               iverts=iverts,
                                               xcenters = disu.cell2d.array['xc'],
                                               ycenters = disu.cell2d.array['yc'],
                                               top=disu.top.array, botm=disu.botm.array,
                                               idomain=disu.idomain.array,
                                               lenuni=disu.length_units.array,
                                               proj4=self._modelgrid.proj4,
                                               epsg=self._modelgrid.epsg,
                                               xoff=self._modelgrid.xoffset,
                                               yoff=self._modelgrid.yoffset,
                                               angrot=self._modelgrid.angrot)

        else:
            return self._modelgrid

        # resolve offsets
        xoff = self._modelgrid.xoffset
        if xoff is None:
            if self._xul is not None:
                xoff = self._modelgrid._xul_to_xll(self._xul)
            else:
                xoff = 0.0
        yoff = self._modelgrid.yoffset
        if yoff is None:
            if self._yul is not None:
                yoff = self._modelgrid._yul_to_yll(self._yul)
            else:
                yoff = 0.0
        self._modelgrid.set_coord_info(xoff, yoff, self._modelgrid.angrot,
                                       self._modelgrid.epsg,
                                       self._modelgrid.proj4)

        return self._modelgrid

    @property
    def packagelist(self):
        return self._packagelist

    @property
    def namefile(self):
        return self.model_nam_file

    @property
    def model_ws(self):
        file_mgr = self.simulation_data.mfpath
        return file_mgr.get_model_path(self.name)

    @property
    def exename(self):
        return self.exe_name

    @property
    def version(self):
        return self._version

    @property
    def solver_tols(self):
        ims = self.get_ims_package()
        if ims is not None:
            rclose = ims.rcloserecord.get_data()
            if rclose is not None:
                rclose = rclose[0][0]
            return ims.inner_hclose.get_data(), rclose
        return None

    @property
    def laytyp(self):
        try:
            return self.npf.icelltype.array
        except AttributeError:
            return None

    @property
    def hdry(self):
        return None

    @property
    def hnoflow(self):
        return None

    @property
    def laycbd(self):
        return None

    def export(self, f, **kwargs):
        from ..export import utils
        return utils.model_export(f, self, **kwargs)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @classmethod
    def load_base(cls, simulation, structure, modelname='NewModel',
                  model_nam_file='modflowtest.nam', type='gwf', version='mf6',
                  exe_name='mf6.exe', strict=True, model_rel_path='.'):
        """
        Load an existing model.

        Parameters
        ----------
        simulation : MFSimulation
            simulation object that this model is a part of
        simulation_data : MFSimulationData
            simulation data object
        structure : MFModelStructure
            structure of this type of model
        model_name : string
            name of the model
        model_nam_file : string
            relative path to the model name file from model working folder
        version : string
            version of modflow
        exe_name : string
            model executable name
        model_ws : string
            model working folder relative to simulation working folder
        strict : boolean
            strict mode when loading files
        model_rel_path : string
            relative path of model folder to simulation folder
        Returns
        -------
        model : MFModel

        Examples
        --------
        """
        instance = cls(simulation, type, modelname,
                       model_nam_file=model_nam_file,
                       version=version, exe_name=exe_name,
                       add_to_simulation=False, structure=structure,
                       model_rel_path=model_rel_path)
        # load name file
        instance.name_file.load(strict)

        # order packages
        vnum = mfstructure.MFStructure().get_version_string()
        # FIX: Transport - Priority packages maybe should not be hard coded
        priority_packages = {'dis{}'.format(vnum): 1,'disv{}'.format(vnum): 1,
                             'disu{}'.format(vnum): 1}
        packages_ordered = []
        package_recarray = instance.simulation_data.mfdata[(modelname, 'nam',
                                                            'packages',
                                                            'packages')]
        for item in package_recarray.get_data():
            if item[0] in priority_packages:
                packages_ordered.insert(0, (item[0], item[1], item[2]))
            else:
                packages_ordered.append((item[0], item[1], item[2]))

        # load packages
        sim_struct = mfstructure.MFStructure().sim_struct
        instance._ftype_num_dict = {}
        for ftype, fname, pname in packages_ordered:
            ftype = ftype[0:-1].lower()
            if ftype in structure.package_struct_objs or ftype in \
              sim_struct.utl_struct_objs:
                if model_rel_path and model_rel_path != '.':
                    # strip off model relative path from the file path
                    filemgr = simulation.simulation_data.mfpath
                    fname = filemgr.strip_model_relative_path(modelname,
                                                              fname)
                if simulation.simulation_data.verbosity_level.value >= \
                        VerbosityLevel.normal.value:
                    print('    loading package {}...'.format(ftype))
                # load package
                instance.load_package(ftype, fname, pname, strict, None)

        # load referenced packages
        if modelname in instance.simulation_data.referenced_files:
            for index, ref_file in \
              instance.simulation_data.referenced_files[modelname].items():
                if (ref_file.file_type in structure.package_struct_objs or
                  ref_file.file_type in sim_struct.utl_struct_objs) and \
                  not ref_file.loaded:
                    instance.load_package(ref_file.file_type,
                                          ref_file.file_name, None, strict,
                                          ref_file.reference_path)
                    ref_file.loaded = True

        # TODO: fix jagged lists where appropriate

        return instance

    def write(self, ext_file_action=ExtFileAction.copy_relative_paths):
        """
        write model to model files

        Parameters
        ----------
        ext_file_action : ExtFileAction
            defines what to do with external files when the simulation path has
            changed.  defaults to copy_relative_paths which copies only files
            with relative paths, leaving files defined by absolute paths fixed.

        Returns
        -------

        Examples
        --------
        """

        # write name file
        if self.simulation_data.verbosity_level.value >= \
                VerbosityLevel.normal.value:
            print('    writing model name file...')

        self.name_file.write(ext_file_action=ext_file_action)

        # write packages
        for pp in self.packagelist:
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('    writing package {}...'.format(pp._get_pname()))
            pp.write(ext_file_action=ext_file_action)

    def get_grid_type(self):
        """
        Return the type of grid used by model 'model_name' in simulation
        containing simulation data 'simulation_data'.

        Returns
        -------
        grid type : DiscritizationType
        """
        package_recarray = self.name_file.packages
        structure = mfstructure.MFStructure()
        if package_recarray.search_data(
                'dis{}'.format(structure.get_version_string()),
                0) is not None:
            return DiscretizationType.DIS
        elif package_recarray.search_data(
                'disv{}'.format(structure.get_version_string()),
                0) is not None:
            return DiscretizationType.DISV
        elif package_recarray.search_data(
                'disu{}'.format(structure.get_version_string()),
                0) is not None:
            return DiscretizationType.DISU

        return DiscretizationType.UNDEFINED

    def get_ims_package(self):
        solution_group = self.simulation.name_file.solutiongroup.get_data()
        for record in solution_group:
            for model_name in record[2:]:
                if model_name == self.name:
                    return self.simulation.get_ims_package(record[1])
        return None

    def get_steadystate_list(self):
        ss_list = []
        tdis = self.simulation.get_package('tdis')
        period_data = tdis.perioddata.get_data()
        for index in range(0, len(period_data)):
            ss_list.append(True)

        storage = self.get_package('sto')
        if storage is not None:
            tr_keys = storage.transient.get_keys(True)
            ss_keys = storage.steady_state.get_keys(True)
            for key in tr_keys:
                ss_list[key] = False
                for ss_list_key in range(key + 1, len(ss_list)):
                    for ss_key in ss_keys:
                        if ss_key == ss_list_key:
                            break
                        ss_list[key] = False
        return ss_list

    def is_valid(self):
        """
        checks the validity of the model and all of its packages

        Parameters
        ----------

        Returns
        -------
        valid : boolean

        Examples
        --------
        """

        # valid name file
        if not self.name_file.is_valid():
            return False

        # valid packages
        for pp in self.packagelist:
            if not pp.is_valid():
                return False

        # required packages exist
        for key, package_struct in self.structure.package_struct_objs.items():
            if not package_struct.optional and not package_struct.file_type \
              in self.package_type_dict:
                return False

        return True

    def set_model_relative_path(self, model_ws):
        """
        sets the file path to the model folder relative to the simulation
        folder and updates all model file paths, placing them in the model
        folder

        Parameters
        ----------
        model_ws : string
            model working folder relative to simulation working folder

        Returns
        -------

        Examples
        --------
        """
        # update path in the file manager
        file_mgr = self.simulation_data.mfpath
        file_mgr.set_last_accessed_model_path()
        path = file_mgr.string_to_file_path(model_ws)
        file_mgr.model_relative_path[self.name] = path

        if model_ws and model_ws != '.' and self.simulation.name_file is not \
                None:
            # update model name file location in simulation name file
            models = self.simulation.name_file.models
            models_data = models.get_data()
            for index, entry in enumerate(models_data):
                old_model_path, old_model_file_name = os.path.split(entry[1])
                old_model_base_name = os.path.splitext(old_model_file_name)[0]
                if old_model_base_name.lower() == self.name.lower() or \
                        self.name == entry[2]:
                    models_data[index][1] = os.path.join(path,
                                                         old_model_file_name)
                    break
            models.set_data(models_data)

            if self.name_file is not None:
                # update listing file location in model name file
                list_file = self.name_file.list.get_data()
                if list_file:
                    path, list_file_name = os.path.split(list_file)
                    try:
                        self.name_file.list.set_data(os.path.join(
                            path, list_file_name))
                    except MFDataException as mfde:
                        message = 'Error occurred while setting relative ' \
                                  'path "{}" in model '\
                                  '"{}".'.format(os.path.join(path,
                                                              list_file_name),
                                                 self.name)
                        raise MFDataException(mfdata_except=mfde,
                                              model=self.model_name,
                                              package=self.name_file.
                                              _get_pname(),
                                              message=message)
                # update package file locations in model name file
                packages = self.name_file.packages
                packages_data = packages.get_data()
                for index, entry in enumerate(packages_data):
                    old_package_path, \
                    old_package_name = os.path.split(entry[1])
                    packages_data[index][1] = os.path.join(path,
                                                           old_package_name)
                packages.set_data(packages_data)

                # update files referenced from within packages
                for package in self.packagelist:
                    package.set_model_relative_path(model_ws)

    def _remove_package_from_dictionaries(self, package):
        # remove package from local dictionaries and lists
        if package.path in self._package_paths:
            del self._package_paths[package.path]
        self._remove_package(package)

    def remove_package(self, package_name):
        """
        removes a package and all child packages from the model

        Parameters
        ----------
        package_name : str
            package name, package type, or package object to be removed from
            the model

        Returns
        -------

        Examples
        --------
        """
        if isinstance(package_name, MFPackage):
            packages = [package_name]
        else:
            packages = self.get_package(package_name)
            if not isinstance(packages, list):
                packages = [packages]
        for package in packages:
            if package._model_or_sim.name != self.name:
                except_text = 'Package can not be removed from model {} ' \
                              'since it is ' \
                              'not part of '
                raise mfstructure.FlopyException(except_text)

            self._remove_package_from_dictionaries(package)

            try:
                # remove package from name file
                package_data = self.name_file.packages.get_data()
            except MFDataException as mfde:
                message = 'Error occurred while reading package names ' \
                          'from name file in model ' \
                          '"{}".'.format(self.name)
                raise MFDataException(mfdata_except=mfde,
                                      model=self.model_name,
                                      package=self.name_file._get_pname(),
                                      message=message)
            try:
                new_rec_array = None
                for item in package_data:
                    if item[1] != package._filename:
                        if new_rec_array is None:
                            new_rec_array = np.rec.array(item, package_data.dtype)
                        else:
                            new_rec_array = np.hstack((item, new_rec_array))
            except:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(self.structure.get_model(),
                                      self.structure.get_package(),
                                      self._path,
                                      'building package recarray',
                                      self.structure.name,
                                      inspect.stack()[0][3],
                                      type_, value_, traceback_, None,
                                      self._simulation_data.debug)
            try:
                self.name_file.packages.set_data(new_rec_array)
            except MFDataException as mfde:
                message = 'Error occurred while setting package names ' \
                          'from name file in model "{}".  Package name ' \
                          'data:\n{}'.format(self.name, new_rec_array)
                raise MFDataException(mfdata_except=mfde,
                                      model=self.model_name,
                                      package=self.name_file._get_pname(),
                                      message=message)

            # build list of child packages
            child_package_list = []
            for pkg in self.packagelist:
                if pkg.parent_file is not None and pkg.parent_file.path == \
                        package.path:
                    child_package_list.append(pkg)
            # remove child packages
            for child_package in child_package_list:
                self._remove_package_from_dictionaries(child_package)

    def register_package(self, package, add_to_package_list=True,
                         set_package_name=True, set_package_filename=True):
        """
        registers a package with the model

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
        package.container_type = [PackageContainerType.model]
        if package.parent_file is not None:
            path = package.parent_file.path + (package.package_type,)
        else:
            path = (self.name, package.package_type)

        if add_to_package_list and path in self._package_paths and not \
                set_package_name and package.package_name in \
                self.package_name_dict:
            # package of this type with this name already exists, replace it
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('WARNING: Package with name {} already exists. '
                      'Replacing existing package.'.format(package.package_name))
            self.remove_package(self.package_name_dict[package.package_name])

        # make sure path is unique
        if path in self._package_paths:
            path_iter = datautil.PathIter(path)
            for new_path in path_iter:
                if new_path not in self._package_paths:
                    path = new_path
                    break
        self._package_paths[path] = 1

        if package.package_type.lower() == 'nam':
            return path, self.structure.name_file_struct_obj

        package_struct = \
          self.structure.get_package_struct(package.package_type)
        if set_package_name:
            # produce a default package name
            if package_struct is not None and \
              package_struct.multi_package_support:
                # check for other registered packages of this type
                name_iter = datautil.NameIter(package.package_type, False)
                for package_name in name_iter:
                    if package_name not in self.package_name_dict:
                        package.package_name = package_name
                        break
            else:
                package.package_name = package.package_type

        if set_package_filename:
            package._filename = '{}.{}'.format(self.name, package.package_type)

        if add_to_package_list:
            self._add_package(package, path)

            # add obs file to name file if it does not have a parent
            if package.package_type in self.structure.package_struct_objs or \
              (package.package_type == 'obs' and package.parent_file is None):
                # update model name file
                pkg_type = package.package_type.upper()
                if len(pkg_type) > 3 and pkg_type[-1] == 'A':
                    pkg_type = pkg_type[0:-1]
                # Model Assumption - assuming all name files have a package
                # recarray
                self.name_file.packages.\
                    update_record(['{}6'.format(pkg_type), package._filename,
                                   package.package_name], 0)
        if package_struct is not None:
            return (path, package_struct)
        else:
            if self.simulation_data.verbosity_level.value >= \
                    VerbosityLevel.normal.value:
                print('WARNING: Unable to register unsupported file type {} '
                      'for model {}.'.format(package.package_type, self.name))
        return None, None

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
        if ref_path is not None:
            fname = os.path.join(ref_path, fname)
        sim_struct = mfstructure.MFStructure().sim_struct
        if (ftype in self.structure.package_struct_objs and
          self.structure.package_struct_objs[ftype].multi_package_support) or \
          (ftype in sim_struct.utl_struct_objs and
          sim_struct.utl_struct_objs[ftype].multi_package_support):
            # resolve dictionary name for package
            if dict_package_name is not None:
                if parent_package is not None:
                    dict_package_name = '{}_{}'.format(parent_package.path[-1],
                                                       ftype)
                else:
                    # use dict_package_name as the base name
                    if ftype in self._ftype_num_dict:
                        self._ftype_num_dict[dict_package_name] += 1
                    else:
                        self._ftype_num_dict[dict_package_name] = 0
                    dict_package_name = '{}_{}'.format(dict_package_name,
                                                       self._ftype_num_dict[
                                                         dict_package_name])
            else:
                # use ftype as the base name
                if ftype in self._ftype_num_dict:
                    self._ftype_num_dict[ftype] += 1
                else:
                    self._ftype_num_dict[ftype] = 0
                if pname is not None:
                    dict_package_name = pname
                else:
                    dict_package_name = '{}_{}'.format(ftype,
                                                       self._ftype_num_dict[
                                                       ftype])
        else:
            dict_package_name = ftype

        # clean up model type text
        model_type = self.structure.model_type
        while datautil.DatumUtil.is_int(model_type[-1]):
            model_type = model_type[0:-1]

        # create package
        package_obj = self.package_factory(ftype, model_type)
        package = package_obj(self, filename=fname, pname=dict_package_name,
                              loading_package=True,
                              parent_file=parent_package)
        try:
            package.load(strict)
        except ReadAsArraysException:
            #  create ReadAsArrays package and load it instead
            package_obj = self.package_factory('{}a'.format(ftype), model_type)
            package = package_obj(self, filename=fname, pname=dict_package_name,
                                  loading_package=True,
                                  parent_file=parent_package)
            package.load(strict)

        # register child package with the model
        self._add_package(package, package.path)
        if parent_package is not None:
            # register child package with the parent package
            parent_package._add_package(package, package.path)

        return package

    def plot(self, SelPackList=None, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        model input data from a model instance

        Args:
            model: Flopy model instance
            SelPackList: (list) list of package names to plot, if none
                all packages will be plotted

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
                    MODFLOW zero-based stress period number to return.
                    (default is zero)
                key : str
                    MfList dictionary key. (default is None)

        Returns:
            axes : list
                Empty list is returned if filename_base is not None. Otherwise
                a list of matplotlib.pyplot.axis are returned.
        """
        from flopy.plot.plotutil import PlotUtilities

        axes = PlotUtilities._plot_model_helper(self,
                                                SelPackList=SelPackList,
                                                **kwargs)

        return axes