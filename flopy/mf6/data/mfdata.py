from operator import itemgetter
import sys
import inspect
from ..mfbase import (
    MFDataException,
    MFInvalidTransientBlockHeaderException,
    FlopyException,
    VerbosityLevel,
)
from ..data.mfstructure import DatumType
from ..coordinates.modeldimensions import DataDimensions, DiscretizationType
from ...datbase import DataInterface, DataType
from .mfdatastorage import DataStructureType
from .mfdatautil import to_string
from ...mbase import ModelInterface


class MFTransient:
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

    def add_transient_key(self, transient_key):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)

    def update_transient_key(self, old_transient_key, new_transient_key):
        if old_transient_key in self._data_storage:
            # replace dictionary key
            self._data_storage[new_transient_key] = self._data_storage[
                old_transient_key
            ]
            del self._data_storage[old_transient_key]
            if self._current_key == old_transient_key:
                # update current key
                self._current_key = new_transient_key

    def _transient_setup(self, data_storage):
        self._data_storage = data_storage

    def get_data_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)
        self._current_key = transient_key
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)

    def _set_data_prep(self, data, transient_key=0):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)
        if isinstance(transient_key, tuple):
            self._current_key = transient_key[0]
        else:
            self._current_key = transient_key
        if self._current_key not in self._data_storage:
            self.add_transient_key(self._current_key)

    def _get_file_entry_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)
        self._current_key = transient_key

    def _load_prep(self, block_header):
        # transient key is first non-keyword block variable
        transient_key = block_header.get_transient_key()
        if isinstance(transient_key, int):
            if not self._verify_sp(transient_key):
                message = 'Invalid transient key "{}" in block' ' "{}"'.format(
                    transient_key, block_header.name
                )
                raise MFInvalidTransientBlockHeaderException(message)
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)
        self._current_key = transient_key

    def _append_list_as_record_prep(self, record, transient_key=0):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)
        self._current_key = transient_key
        if transient_key not in self._data_storage:
            self.add_transient_key(transient_key)

    def _update_record_prep(self, transient_key=0):
        if isinstance(transient_key, int):
            self._verify_sp(transient_key)
        self._current_key = transient_key

    def get_active_key_list(self):
        return sorted(self._data_storage.items(), key=itemgetter(0))

    def get_active_key_dict(self):
        key_dict = {}
        for key in self._data_storage.keys():
            key_dict[key] = True
        return key_dict

    def _verify_sp(self, sp_num):
        if self._path[0].lower() == "nam":
            return True
        if not ("tdis", "dimensions", "nper") in self._simulation_data.mfdata:
            raise FlopyException(
                "Could not find number of stress periods (nper)."
            )
        nper = self._simulation_data.mfdata[("tdis", "dimensions", "nper")]
        if not (sp_num <= nper.get_data()):
            if (
                self._simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "WARNING: Stress period value {} in package {} is "
                    "greater than the number of stress periods defined "
                    "in nper.".format(sp_num + 1, self.structure.get_package())
                )
        return True


class MFData(DataInterface):
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
    *args, **kwargs : exists to support different child class parameter sets
        with extra init parameters

    Attributes
    ----------
    _current_key : str
        current key defining specific transient dataset to be accessed

    Methods
    -------
    new_simulation(sim_data)
        points data object to a new simulation
    layer_shape() : tuple
        returns the shape of the layered dimensions

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        enable=True,
        path=None,
        dimensions=None,
        *args,
        **kwargs
    ):
        # initialize
        self._current_key = None
        self._valid = True
        self._simulation_data = sim_data
        self._model_or_sim = model_or_sim
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
        self._keyword = ""
        if self._simulation_data is not None:
            self._data_dimensions = DataDimensions(dimensions, structure)
            # build a unique path in the simulation dictionary
            self._org_path = self._path
            index = 0
            while self._path in self._simulation_data.mfdata:
                self._path = self._org_path[:-1] + (
                    "{}_{}".format(self._org_path[-1], index),
                )
                index += 1
        self._structure_init()
        # tie this to the simulation dictionary
        sim_data.mfdata[self._path] = self
        # set up model grid caching
        self._cache_next_grid = False
        self._grid_cached = False
        self._cached_model_grid = None

    def __repr__(self):
        return repr(self._get_storage_obj())

    def __str__(self):
        return str(self._get_storage_obj())

    @property
    def array(self):
        kwargs = {"array": True}
        return self.get_data(apply_mult=True, **kwargs)

    @property
    def name(self):
        return self.structure.name

    @property
    def model(self):
        if (
            self._model_or_sim is not None
            and self._model_or_sim.type == "Model"
        ):
            return self._model_or_sim
        else:
            return None

    @property
    def data_type(self):
        raise NotImplementedError(
            "must define dat_type in child class to use this base class"
        )

    @property
    def dtype(self):
        raise NotImplementedError(
            "must define dtype in child class to use this base class"
        )

    @property
    def plottable(self):
        raise NotImplementedError(
            "must define plottable in child class to use this base class"
        )

    @property
    def _cache_model_grid(self):
        return self._cache_next_grid

    @_cache_model_grid.setter
    def _cache_model_grid(self, cache_model_grid):
        if cache_model_grid:
            self._cache_next_grid = True
            self._grid_cached = False
        else:
            self._cache_next_grid = False
            self._grid_cached = False
            self._cached_model_grid = None

    def _resync(self):
        model = self.model
        if model is not None:
            model._mg_resync = True

    @staticmethod
    def _tas_info(tas_str):
        if isinstance(tas_str, str):
            lst_str = tas_str.split(" ")
            if len(lst_str) >= 2 and lst_str[0].lower() == "timearrayseries":
                return lst_str[1], lst_str[0]
        return None, None

    def export(self, f, **kwargs):
        from flopy.export import utils

        if (
            self.data_type == DataType.array2d
            and len(self.array.shape) == 2
            and self.array.shape[1] > 0
        ):
            return utils.array2d_export(f, self, **kwargs)
        elif self.data_type == DataType.array3d:
            return utils.array3d_export(f, self, **kwargs)
        elif self.data_type == DataType.transient2d:
            return utils.transient2d_export(f, self, **kwargs)
        elif self.data_type == DataType.transientlist:
            return utils.mflist_export(f, self, **kwargs)
        return utils.transient2d_export(f, self, **kwargs)

    def new_simulation(self, sim_data):
        self._simulation_data = sim_data
        self._data_storage = None

    def find_dimension_size(self, dimension_name):
        parent_path = self._path[:-1]
        result = self._simulation_data.mfdata.find_in_path(
            parent_path, dimension_name
        )
        if result[0] is not None:
            return [result[0].get_data()]
        else:
            return []

    def aux_var_names(self):
        return self.find_dimension_size("auxnames")

    def layer_shape(self):
        layers = []
        layer_dims = self.structure.data_item_structures[0].layer_dims
        if len(layer_dims) == 1:
            layers.append(self._data_dimensions.get_model_grid().num_layers())
        else:
            for layer in layer_dims:
                if layer == "nlay":
                    # get the layer size from the model grid
                    try:
                        model_grid = self._data_dimensions.get_model_grid()
                    except Exception as ex:
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.structure.get_model(),
                            self.structure.get_package(),
                            self.path,
                            "getting model grid",
                            self.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            None,
                            self.sim_data.debug,
                            ex,
                        )

                    if model_grid.grid_type() == DiscretizationType.DISU:
                        layers.append(1)
                    else:
                        num_layers = model_grid.num_layers()
                        if num_layers is not None:
                            layers.append(num_layers)
                        else:
                            layers.append(1)
                else:
                    # search data dictionary for layer size
                    layer_size = self.find_dimension_size(layer)
                    if len(layer_size) == 1:
                        layers.append(layer_size[0])
                    else:
                        message = (
                            "Unable to find the size of expected layer "
                            "dimension {} ".format(layer)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.structure.get_model(),
                            self.structure.get_package(),
                            self.structure.path,
                            "resolving layer dimensions",
                            self.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
        return tuple(layers)

    def get_description(self, description=None, data_set=None):
        if data_set is None:
            data_set = self.structure
        for data_item in data_set.data_items.values():
            if data_item.type == DatumType.record:
                # record within a record, recurse
                description = self.get_description(description, data_item)
            else:
                if data_item.description:
                    if description:
                        description = "{}\n{}".format(
                            description, data_item.description
                        )
                    else:
                        description = data_item.description
        return description

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        self.enabled = True

    def is_valid(self):
        # TODO: Implement for each data type
        return self._valid

    def _get_model_grid(self):
        mg = None
        if (
            self._cache_next_grid
            or not self._grid_cached
            or self._cached_model_grid is None
        ):
            # construct a new model grid
            if isinstance(self._model_or_sim, ModelInterface) and hasattr(
                self._model_or_sim, "modelgrid"
            ):
                # get model grid info
                mg = self._model_or_sim.modelgrid
            else:
                mg = None
        if self._grid_cached and self._cached_model_grid is not None:
            # get the model grid from cache
            mg = self._cached_model_grid
        elif self._cache_next_grid:
            # cache the existing model grid
            self._cached_model_grid = mg
            self._grid_cached = mg is not None
            self._cache_next_grid = False
        return mg

    def _structure_init(self, data_set=None):
        if data_set is None:
            # Initialize variables
            data_set = self.structure
        for data_item_struct in data_set.data_item_structures:
            if data_item_struct.type == DatumType.record:
                # this is a record within a record, recurse
                self._structure_init(data_item_struct)
            else:
                if len(self.structure.data_item_structures) == 1:
                    # data item name is a keyword to look for
                    self._keyword = data_item_struct.name

    def _get_constant_formatting_string(
        self, const_val, layer, data_type, suffix="\n"
    ):
        if (
            self.structure.data_item_structures[0].numeric_index
            or self.structure.data_item_structures[0].is_cellid
        ):
            # for cellid and numeric indices convert from 0 base to 1 based
            const_val = abs(const_val) + 1

        sim_data = self._simulation_data
        const_format = list(sim_data.constant_formatting)
        const_format[1] = to_string(
            const_val,
            data_type,
            self._simulation_data,
            self._data_dimensions,
            verify_data=self._simulation_data.verify_data,
        )
        return "{}{}".format(sim_data.indent_string.join(const_format), suffix)

    def _get_aux_var_name(self, aux_var_index):
        aux_var_names = self._data_dimensions.package_dim.get_aux_variables()
        # TODO: Verify that this works for multi-dimensional layering
        return aux_var_names[0][aux_var_index[0] + 1]

    def _get_storage_obj(self):
        return self._data_storage


class MFMultiDimVar(MFData):
    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        enable=True,
        path=None,
        dimensions=None,
    ):
        super().__init__(
            sim_data, model_or_sim, structure, enable, path, dimensions
        )

    @property
    def data_type(self):
        raise NotImplementedError(
            "must define dat_type in child class to use this base class"
        )

    @property
    def plottable(self):
        raise NotImplementedError(
            "must define plottable in child class to use this base class"
        )

    def _get_internal_formatting_string(self, layer):
        storage = self._get_storage_obj()
        if layer is None:
            layer_storage = storage.layer_storage.first_item()
        else:
            layer_storage = storage.layer_storage[layer]
        int_format = ["INTERNAL"]
        data_type = self.structure.get_datum_type(return_enum_type=True)
        if storage.data_structure_type != DataStructureType.recarray:
            int_format.append("FACTOR")
            if layer_storage.factor is not None:
                if data_type == DatumType.integer:
                    int_format.append(str(int(layer_storage.factor)))
                else:
                    int_format.append(str(layer_storage.factor))
            else:
                if data_type == DatumType.double_precision:
                    int_format.append("1.0")
                else:
                    int_format.append("1")
        if layer_storage.iprn is not None:
            int_format.append("IPRN")
            int_format.append(str(layer_storage.iprn))
        return self._simulation_data.indent_string.join(int_format)

    def _get_external_formatting_string(self, layer, ext_file_action):
        storage = self._get_storage_obj()
        if layer is None:
            layer_storage = storage.layer_storage.first_item()
        else:
            layer_storage = storage.layer_storage[layer]
        # resolve external file path
        file_mgmt = self._simulation_data.mfpath
        model_name = self._data_dimensions.package_dim.model_dim[0].model_name
        ext_file_path = file_mgmt.get_updated_path(
            layer_storage.fname, model_name, ext_file_action
        )
        layer_storage.fname = ext_file_path
        ext_format = ["OPEN/CLOSE", "'{}'".format(ext_file_path)]
        if storage.data_structure_type != DataStructureType.recarray:
            if layer_storage.factor is not None:
                data_type = self.structure.get_datum_type(
                    return_enum_type=True
                )
                ext_format.append("FACTOR")
                if data_type == DatumType.integer:
                    ext_format.append(str(int(layer_storage.factor)))
                else:
                    ext_format.append(str(layer_storage.factor))
        if layer_storage.binary:
            ext_format.append("(BINARY)")
        if layer_storage.iprn is not None:
            ext_format.append("IPRN")
            ext_format.append(str(layer_storage.iprn))
        return "{}\n".format(
            self._simulation_data.indent_string.join(ext_format)
        )
