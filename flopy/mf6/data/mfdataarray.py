import sys, inspect, copy, os
import numpy as np
from collections import OrderedDict
from ..data.mfstructure import DatumType
from .mfdatastorage import DataStorage, DataStructureType, DataStorageType
from ...utils.datautil import MultiList, DatumUtil
from ..mfbase import ExtFileAction, MFDataException, VerbosityLevel
from ..utils.mfenums import DiscretizationType
from ...datbase import DataType
from .mffileaccess import MFFileAccessArray
from .mfdata import MFMultiDimVar, MFTransient
from ...mbase import ModelInterface


class MFArray(MFMultiDimVar):
    """
    Provides an interface for the user to access and update MODFLOW array data.
    MFArray objects are not designed to be directly constructed by the end
    user. When a FloPy for MODFLOW 6 package object is constructed, the
    appropriate MFArray objects are automatically built.

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

    """

    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        data=None,
        enable=True,
        path=None,
        dimensions=None,
    ):
        super().__init__(
            sim_data, model_or_sim, structure, enable, path, dimensions
        )
        if self.structure.layered:
            try:
                self._layer_shape = self.layer_shape()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "resolving layer dimensions",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
        else:
            self._layer_shape = (1,)
        if self._layer_shape[0] is None:
            self._layer_shape = (1,)
        self._data_type = structure.data_item_structures[0].type
        try:
            shp_ml = MultiList(shape=self._layer_shape)
            self._data_storage = self._new_storage(
                shp_ml.get_total_size() != 1
            )
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                structure.get_model(),
                structure.get_package(),
                path,
                "creating storage",
                structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                sim_data.debug,
                ex,
            )
        self._last_line_info = []
        if self.structure.type == DatumType.integer:
            multiplier = [1]
        else:
            multiplier = [1.0]
        if data is not None:
            try:
                self._get_storage_obj().set_data(
                    data, key=self._current_key, multiplier=multiplier
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )

    def __setattr__(self, name, value):
        if name == "__setstate__":
            raise AttributeError(name)
        elif name == "fname":
            self._get_storage_obj().layer_storage.first_item().fname = value
        elif name == "factor":
            self._get_storage_obj().layer_storage.first_item().factor = value
        elif name == "iprn":
            self._get_storage_obj().layer_storage.first_item().iprn = value
        elif name == "binary":
            self._get_storage_obj().layer_storage.first_item().binary = value
        else:
            super().__setattr__(name, value)

    def __getitem__(self, k):
        if isinstance(k, int):
            k = (k,)
        storage = self._get_storage_obj()
        if storage.layered and (isinstance(k, tuple) or isinstance(k, list)):
            if not storage.layer_storage.in_shape(k):
                comment = (
                    'Could not retrieve layer {} of "{}". There'
                    "are only {} layers available"
                    ".".format(
                        k, self.structure.name, len(storage.layer_storage)
                    )
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    self._simulation_data.debug,
                )
            # for layered data treat k as layer number(s)
            return storage.layer_storage[k]
        else:
            # for non-layered data treat k as an array/list index of the data
            if isinstance(k, int):
                try:
                    if len(self._get_data(apply_mult=True).shape) == 1:
                        return self._get_data(apply_mult=True)[k]
                    elif self._get_data(apply_mult=True).shape[0] == 1:
                        return self._get_data(apply_mult=True)[0, k]
                    elif self._get_data(apply_mult=True).shape[1] == 1:
                        return self._get_data(apply_mult=True)[k, 0]
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.structure.get_model(),
                        self.structure.get_package(),
                        self._path,
                        "setting data",
                        self.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        None,
                        self._simulation_data.debug,
                        ex,
                    )

                comment = (
                    'Unable to resolve index "{}" for '
                    "multidimensional data.".format(k)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    self._simulation_data.debug,
                )
            else:
                try:
                    if isinstance(k, tuple):
                        if len(k) == 3:
                            return self._get_data(apply_mult=True)[
                                k[0], k[1], k[2]
                            ]
                        elif len(k) == 2:
                            return self._get_data(apply_mult=True)[k[0], k[1]]
                        if len(k) == 1:
                            return self._get_data(apply_mult=True)[k]
                    else:
                        return self._get_data(apply_mult=True)[(k,)]
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.structure.get_model(),
                        self.structure.get_package(),
                        self._path,
                        "setting data",
                        self.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        None,
                        self._simulation_data.debug,
                        ex,
                    )

    def __setitem__(self, k, value):
        storage = self._get_storage_obj()
        self._resync()
        if storage.layered:
            if isinstance(k, int):
                k = (k,)
            # for layered data treat k as a layer number
            try:
                storage.layer_storage[k]._set_data(value)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )

        else:
            try:
                # for non-layered data treat k as an array/list index of the data
                a = self._get_data()
                a[k] = value
                a = a.astype(self._get_data().dtype)
                layer_storage = storage.layer_storage.first_item()
                self._get_storage_obj()._set_data(
                    a, key=self._current_key, multiplier=layer_storage.factor
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )

    @property
    def data_type(self):
        """Type of data (DataType) stored in the array"""
        if self.structure.layered:
            return DataType.array3d
        else:
            return DataType.array2d

    @property
    def dtype(self):
        """Type of data (numpy.dtype) stored in the array"""
        return self._get_data().dtype.type

    @property
    def plottable(self):
        """If the array is plottable"""
        if self.model is None:
            return False
        else:
            return True

    @property
    def data(self):
        """Returns array data.  Calls get_data with default parameters."""
        return self._get_data()

    def new_simulation(self, sim_data):
        """Initialize MFArray object for a new simulation

        Parameters
        ----------
            sim_data : MFSimulationData
                Data dictionary containing simulation data.


        """
        super().new_simulation(sim_data)
        self._data_storage = self._new_storage(False)
        self._layer_shape = (1,)

    def supports_layered(self):
        """Returns whether this MFArray supports layered data

        Returns
        -------
            layered data supported: bool
                Whether or not this data object supports layered data

        """

        try:
            model_grid = self._data_dimensions.get_model_grid()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "getting model grid",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )
        return (
            self.structure.layered
            and model_grid.grid_type() != DiscretizationType.DISU
        )

    def set_layered_data(self, layered_data):
        """Sets whether this MFArray supports layered data

        Parameters
        ----------
        layered_data : bool
            Whether data is layered or not.

        """
        if layered_data is True and self.structure.layered is False:
            if (
                self._data_dimensions.get_model_grid().grid_type()
                == DiscretizationType.DISU
            ):
                comment = (
                    "Layered option not available for unstructured "
                    "grid. {}".format(self._path)
                )
            else:
                comment = (
                    'Data "{}" does not support layered option. '
                    "{}".format(self._data_name, self._path)
                )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "setting layered data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self._simulation_data.debug,
            )
        self._get_storage_obj().layered = layered_data

    def make_layered(self):
        """Changes the data to be stored by layer instead of as a single array."""

        if self.supports_layered():
            try:
                self._get_storage_obj().make_layered()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "making data layered",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
        else:
            if (
                self._data_dimensions.get_model_grid().grid_type()
                == DiscretizationType.DISU
            ):
                comment = (
                    "Layered option not available for unstructured "
                    "grid. {}".format(self._path)
                )
            else:
                comment = (
                    'Data "{}" does not support layered option. '
                    "{}".format(self._data_name, self._path)
                )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "converting data to layered",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self._simulation_data.debug,
            )

    def store_as_external_file(
        self,
        external_file_path,
        layer=None,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Stores data from layer `layer` to an external file at
        `external_file_path`.  For unlayered data do not pass in `layer`.
        If layer is not specified all layers will be stored with each layer
        as a separate file. If replace_existing_external is set to False,
        this method will not do anything if the data is already in an
        external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            layer : int
                Which layer to store in external file, `None` value stores all
                layers.
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing
        """
        storage = self._get_storage_obj()
        if storage is None:
            self._set_storage_obj(self._new_storage(False, True))
            storage = self._get_storage_obj()
        # build list of layers
        if layer is None:
            layer_list = []
            for index in range(0, storage.layer_storage.get_total_size()):
                if (
                    replace_existing_external
                    or storage.layer_storage[index].data_storage_type
                    == DataStorageType.internal_array
                    or storage.layer_storage[index].data_storage_type
                    == DataStorageType.internal_constant
                ):
                    layer_list.append(index)
        else:
            if (
                replace_existing_external
                or storage.layer_storage[layer].data_storage_type
                == DataStorageType.internal_array
                or storage.layer_storage[layer].data_storage_type
                == DataStorageType.internal_constant
            ):
                layer_list = [layer]
            else:
                layer_list = []

        # store data from each layer in a separate file
        for current_layer in layer_list:
            # determine external file name for layer
            if len(layer_list) > 0:
                fname, ext = os.path.splitext(external_file_path)
                if len(layer_list) == 1:
                    file_path = "{}{}".format(fname, ext)
                else:
                    file_path = "{}_layer{}{}".format(
                        fname, current_layer + 1, ext
                    )
            else:
                file_path = external_file_path
            if isinstance(current_layer, int):
                current_layer = (current_layer,)
            # get the layer's data
            data = self._get_data(current_layer, True)

            if data is None:
                # do not write empty data to an external file
                continue
            if isinstance(data, str) and self._tas_info(data)[0] is not None:
                # data must not be time array series information
                continue
            if storage.get_data_dimensions(current_layer)[0] == -9999:
                # data must have well defined dimensions to make external
                continue
            try:
                # store layer's data in external file
                if (
                    self._simulation_data.verbosity_level.value
                    >= VerbosityLevel.verbose.value
                ):
                    print(
                        "Storing {} layer {} to external file {}.."
                        ".".format(
                            self.structure.name,
                            current_layer[0] + 1,
                            file_path,
                        )
                    )
                factor = storage.layer_storage[current_layer].factor
                external_data = {
                    "filename": file_path,
                    "data": self._get_data(current_layer, True),
                    "factor": factor,
                    "binary": binary,
                }
                self._set_data(
                    external_data, layer=current_layer, check_data=False
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "storing data in external file "
                    "{}".format(external_file_path),
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )

    def store_internal(
        self,
        layer=None,
        check_data=True,
    ):
        """Stores data from layer `layer` internally.  For unlayered data do
        not pass in `layer`.  If layer is not specified all layers will be
        stored internally

        Parameters
        ----------
            layer : int
                Which layer to store in external file, `None` value stores all
                layers.
            check_data : bool
                Verify data prior to storing
        """
        storage = self._get_storage_obj()
        if storage is None:
            self._set_storage_obj(self._new_storage(False, True))
            storage = self._get_storage_obj()
        # build list of layers
        if layer is None:
            layer_list = []
            for index in range(0, storage.layer_storage.get_total_size()):
                if (
                    storage.layer_storage[index].data_storage_type
                    == DataStorageType.external_file
                ):
                    layer_list.append(index)
        else:
            if (
                storage.layer_storage[layer].data_storage_type
                == DataStorageType.external_file
            ):
                layer_list = [layer]
            else:
                layer_list = []

        # store data from each layer
        for current_layer in layer_list:
            if isinstance(current_layer, int):
                current_layer = (current_layer,)
            # get the layer's data
            data = self._get_data(current_layer, True)

            if data is None:
                # do not write empty data to an internal file
                continue
            try:
                # store layer's data internally
                if (
                    self._simulation_data.verbosity_level.value
                    >= VerbosityLevel.verbose.value
                ):
                    print(
                        "Storing {} layer {} internally.."
                        ".".format(
                            self.structure.name,
                            current_layer[0] + 1,
                        )
                    )
                factor = storage.layer_storage[current_layer].factor
                internal_data = {
                    "data": self._get_data(current_layer, True),
                    "factor": factor,
                }
                self._set_data(
                    internal_data, layer=current_layer, check_data=False
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "storing data {} internally".format(self.structure.name),
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )

    def has_data(self, layer=None):
        """Returns whether layer "layer_num" has any data associated with it.

        Parameters
        ----------
            layer_num : int
                Layer number to check for data.  For unlayered data do not
                pass anything in

        Returns
        -------
            has data: bool
                Returns if there is data.

        """
        storage = self._get_storage_obj()
        if storage is None:
            return False
        if isinstance(layer, int):
            layer = (layer,)
        try:
            return storage.has_data(layer)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "checking for data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def get_data(self, layer=None, apply_mult=False, **kwargs):
        """Returns the data associated with layer "layer_num".  If "layer_num"
        is None, returns all data.

        Parameters
        ----------
            layer_num : int

        Returns
        -------
             data : ndarray
                Array data in an ndarray

        """
        return self._get_data(layer, apply_mult, **kwargs)

    def _get_data(self, layer=None, apply_mult=False, **kwargs):
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        if isinstance(layer, int):
            layer = (layer,)
        storage = self._get_storage_obj()
        if storage is not None:
            try:
                data = storage.get_data(layer, apply_mult)
                if (
                    "array" in kwargs
                    and kwargs["array"]
                    and isinstance(self, MFTransientArray)
                ):
                    data = np.expand_dims(data, 0)
                return data
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
        return None

    def set_data(self, data, multiplier=None, layer=None):
        """Sets the contents of the data at layer `layer` to `data` with
        multiplier `multiplier`. For unlayered data do not pass in
        `layer`.  Data can have the following formats:
        1) ndarray - numpy ndarray containing all of the data
        2) [data] - python list containing all of the data
        3) val - a single constant value to be used for all of the data
        4) {'filename':filename, 'factor':fct, 'iprn':print, 'data':data} -
        dictionary defining external file information
        5) {'data':data, 'factor':fct, 'iprn':print) - dictionary defining
        internal information. Data that is layered can also be set by defining
        a list with a length equal to the number of layers in the model.
        Each layer in the list contains the data as defined in the
        formats above:
            [layer_1_val, [layer_2_array_vals],
            {'filename':file_with_layer_3_data, 'factor':fct, 'iprn':print}]

        Parameters
        ----------
        data : ndarray/list
            An ndarray or nested lists containing the data to set.
        multiplier : float
            Multiplier to apply to data
        layer : int
            Data layer that is being set

        """
        self._set_data(data, multiplier, layer)

    def _set_data(self, data, multiplier=None, layer=None, check_data=True):
        self._resync()
        if self._get_storage_obj() is None:
            self._data_storage = self._new_storage(False)
        if multiplier is None:
            multiplier = [self._get_storage_obj().get_default_mult()]
        if isinstance(layer, int):
            layer = (layer,)
        if isinstance(data, str):
            # check to see if this is a time series array
            tas_name, tas_label = self._tas_info(data)
            if tas_name is not None:
                # verify and save as time series array
                self._get_storage_obj().set_tas(
                    tas_name, tas_label, self._current_key, check_data
                )
                return

        storage = self._get_storage_obj()
        if self.structure.name == "aux" and layer is None:
            if isinstance(data, dict):
                aux_data = copy.deepcopy(data["data"])
            else:
                aux_data = data
            # make a list out of a single item
            if (
                isinstance(aux_data, int)
                or isinstance(aux_data, float)
                or isinstance(aux_data, str)
            ):
                aux_data = [[aux_data]]
            # handle special case of aux variables in an array
            self.layered = True
            aux_var_names = (
                self._data_dimensions.package_dim.get_aux_variables()
            )
            if len(aux_data) == len(aux_var_names[0]) - 1:
                for layer, aux_var_data in enumerate(aux_data):
                    if (
                        layer > 0
                        and layer >= storage.layer_storage.get_total_size()
                    ):
                        storage.add_layer()
                    if isinstance(data, dict):
                        # put layer data back in dictionary
                        layer_data = data
                        layer_data["data"] = aux_var_data
                    else:
                        layer_data = aux_var_data
                    try:
                        storage.set_data(
                            layer_data, [layer], multiplier, self._current_key
                        )
                    except Exception as ex:
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.structure.get_model(),
                            self.structure.get_package(),
                            self._path,
                            "setting data",
                            self.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            None,
                            self._simulation_data.debug,
                            ex,
                        )
            else:
                message = (
                    "Unable to set data for aux variable. "
                    "Expected {} aux variables but got "
                    "{}.".format(len(aux_var_names[0]), len(data))
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self._data_dimensions.structure.get_model(),
                    self._data_dimensions.structure.get_package(),
                    self._data_dimensions.structure.path,
                    "setting aux variables",
                    self._data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
        else:
            try:
                storage.set_data(
                    data, layer, multiplier, key=self._current_key
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
        self._layer_shape = storage.layer_storage.list_shape

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        """Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read and
        the second item being the last line of text read from the file.  This
        method is for internal flopy use and is not intended for the end user.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this array.
            file_handle : file descriptor
                A file handle for the data file which points to the second
                line of data for this array
            block_header : MFBlockHeader
                Block header object that contains block header information
                for the block containing this data
            pre_data_comments : MFComment
                Comments immediately prior to the data
            external_file_info : list
                Contains information about storing files externally

        Returns
        -------
            more data : bool,
            next data line : str

        """
        super().load(
            first_line,
            file_handle,
            block_header,
            pre_data_comments=None,
            external_file_info=None,
        )
        self._resync()
        if self.structure.layered:
            try:
                model_grid = self._data_dimensions.get_model_grid()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting model grid",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
            if self._layer_shape[-1] != model_grid.num_layers():
                if model_grid.grid_type() == DiscretizationType.DISU:
                    self._layer_shape = (1,)
                else:
                    self._layer_shape = (model_grid.num_layers(),)
                    if self._layer_shape[-1] is None:
                        self._layer_shape = (1,)
                shape_ml = MultiList(shape=self._layer_shape)
                self._set_storage_obj(
                    self._new_storage(shape_ml.get_total_size() != 1, True)
                )
        file_access = MFFileAccessArray(
            self.structure,
            self._data_dimensions,
            self._simulation_data,
            self._path,
            self._current_key,
        )
        storage = self._get_storage_obj()
        self._layer_shape, return_val = file_access.load_from_package(
            first_line,
            file_handle,
            self._layer_shape,
            storage,
            self._keyword,
            pre_data_comments=None,
        )
        if external_file_info is not None:
            storage.point_to_existing_external_file(
                external_file_info, storage.layer_storage.get_total_size() - 1
            )

        return return_val

    def _is_layered_aux(self):
        # determine if this is the special aux variable case
        if (
            self.structure.name.lower() == "aux"
            and self._get_storage_obj().layered
        ):
            return True
        else:
            return False

    def get_file_entry(
        self, layer=None, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        """Returns a string containing the data in layer "layer" formatted for
        a MODFLOW 6 file.  For unlayered data do not pass in "layer".

        Parameters
        ----------
            layer : int
                The layer to return file entry for.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        return self._get_file_entry(layer, ext_file_action)

    def _get_file_entry(
        self, layer=None, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        if isinstance(layer, int):
            layer = (layer,)
        data_storage = self._get_storage_obj()
        if (
            data_storage is None
            or data_storage.layer_storage.get_total_size() == 0
            or not data_storage.has_data()
        ):
            return ""

        layered_aux = self._is_layered_aux()

        # prepare indent
        indent = self._simulation_data.indent_string
        shape_ml = MultiList(shape=self._layer_shape)
        if shape_ml.get_total_size() == 1:
            data_indent = indent
        else:
            data_indent = "{}{}".format(
                indent, self._simulation_data.indent_string
            )

        file_entry_array = []
        if data_storage.data_structure_type == DataStructureType.scalar:
            # scalar data, like in the case of a time array series gets written
            # on a single line
            try:
                data = data_storage.get_data()
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting data",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
            if (
                self.structure.data_item_structures[0].numeric_index
                or self.structure.data_item_structures[0].is_cellid
            ):
                # for cellid and numeric indices convert from 0 base to 1 based
                data = abs(data) + 1
            file_entry_array.append(
                "{}{}{}{}\n".format(indent, self.structure.name, indent, data)
            )
        elif data_storage.layered:
            if not layered_aux:
                if not self.structure.data_item_structures[0].just_data:
                    name = self.structure.name
                    file_entry_array.append(
                        "{}{}{}{}\n".format(indent, name, indent, "LAYERED")
                    )
                else:
                    file_entry_array.append("{}{}\n".format(indent, "LAYERED"))

            if layer is None:
                layer_min = shape_ml.first_index()
                layer_max = copy.deepcopy(self._layer_shape)
            else:
                # set layer range
                if not shape_ml.in_shape(layer):
                    comment = (
                        'Layer {} for variable "{}" does not exist'
                        ".".format(layer, self._data_name)
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.structure.get_model(),
                        self.structure.get_package(),
                        self._path,
                        "getting file entry",
                        self.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        comment,
                        self._simulation_data.debug,
                    )

                layer_min = layer
                layer_max = shape_ml.inc_shape_idx(layer)
            for layer in shape_ml.indexes(layer_min, layer_max):
                file_entry_array.append(
                    self._get_file_entry_layer(
                        layer,
                        data_indent,
                        data_storage.layer_storage[layer].data_storage_type,
                        ext_file_action,
                        layered_aux,
                    )
                )
        else:
            # data is not layered
            if not self.structure.data_item_structures[0].just_data:
                if self._data_name == "aux":
                    file_entry_array.append(
                        "{}{}\n".format(indent, self._get_aux_var_name([0]))
                    )
                else:
                    file_entry_array.append(
                        "{}{}\n".format(indent, self.structure.name)
                    )

            data_storage_type = data_storage.layer_storage[0].data_storage_type
            file_entry_array.append(
                self._get_file_entry_layer(
                    None, data_indent, data_storage_type, ext_file_action
                )
            )

        return "".join(file_entry_array)

    def _new_storage(
        self, set_layers=True, base_storage=False, stress_period=0
    ):
        if set_layers:
            return DataStorage(
                self._simulation_data,
                self._model_or_sim,
                self._data_dimensions,
                self._get_file_entry,
                DataStorageType.internal_array,
                DataStructureType.ndarray,
                self._layer_shape,
                stress_period=stress_period,
                data_path=self._path,
            )
        else:
            return DataStorage(
                self._simulation_data,
                self._model_or_sim,
                self._data_dimensions,
                self._get_file_entry,
                DataStorageType.internal_array,
                DataStructureType.ndarray,
                stress_period=stress_period,
                data_path=self._path,
            )

    def _get_storage_obj(self):
        return self._data_storage

    def _set_storage_obj(self, storage):
        self._data_storage = storage

    def _get_file_entry_layer(
        self,
        layer,
        data_indent,
        storage_type,
        ext_file_action,
        layered_aux=False,
    ):
        if (
            not self.structure.data_item_structures[0].just_data
            and not layered_aux
        ):
            indent_string = "{}{}".format(
                self._simulation_data.indent_string,
                self._simulation_data.indent_string,
            )
        else:
            indent_string = self._simulation_data.indent_string

        file_entry = ""
        if layered_aux:
            try:
                # display aux name
                file_entry = "{}{}\n".format(
                    indent_string, self._get_aux_var_name(layer)
                )
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting aux variables",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
            indent_string = "{}{}".format(
                indent_string, self._simulation_data.indent_string
            )

        data_storage = self._get_storage_obj()
        if storage_type == DataStorageType.internal_array:
            # internal data header + data
            format_str = self._get_internal_formatting_string(layer).upper()
            lay_str = self._get_data_layer_string(layer, data_indent).upper()
            file_entry = "{}{}{}\n{}".format(
                file_entry, indent_string, format_str, lay_str
            )
        elif storage_type == DataStorageType.internal_constant:
            #  constant data
            try:
                const_val = data_storage.get_const_val(layer)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting constant value",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                    ex,
                )
            const_str = self._get_constant_formatting_string(
                const_val, layer, self._data_type
            ).upper()
            file_entry = "{}{}{}".format(file_entry, indent_string, const_str)
        else:
            #  external data
            ext_str = self._get_external_formatting_string(
                layer, ext_file_action
            )
            file_entry = "{}{}{}".format(file_entry, indent_string, ext_str)
            #  add to active list of external files
            try:
                file_path = data_storage.get_external_file_path(layer)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                comment = (
                    "Could not get external file path for layer "
                    '"{}"'.format(layer),
                )
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "getting external file path",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    self._simulation_data.debug,
                    ex,
                )
            package_dim = self._data_dimensions.package_dim
            model_name = package_dim.model_dim[0].model_name
            self._simulation_data.mfpath.add_ext_file(file_path, model_name)
        return file_entry

    def _get_data_layer_string(self, layer, data_indent):
        # iterate through data layer
        try:
            data = self._get_storage_obj().get_data(layer, False)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment = 'Could not get data for layer "{}"'.format(layer)
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "getting data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self._simulation_data.debug,
                ex,
            )
        file_access = MFFileAccessArray(
            self.structure,
            self._data_dimensions,
            self._simulation_data,
            self._path,
            self._current_key,
        )
        return file_access.get_data_string(data, self._data_type, data_indent)

    def _resolve_layer_index(self, layer, allow_multiple_layers=False):
        # handle layered vs non-layered data
        storage = self._get_storage_obj()
        if storage.layered:
            if layer is None:
                if storage.layer_storage.get_total_size() == 1:
                    layer_index = [0]
                elif allow_multiple_layers:
                    layer_index = storage.get_active_layer_indices()
                else:
                    comment = (
                        'Data "{}" is layered but no '
                        "layer_num was specified"
                        ".".format(self._data_name)
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.structure.get_model(),
                        self.structure.get_package(),
                        self._path,
                        "resolving layer index",
                        self.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        comment,
                        self._simulation_data.debug,
                    )

            else:
                layer_index = [layer]
        else:
            layer_index = [[0]]
        return layer_index

    def _verify_data(self, data_iter, layer_num):
        # TODO: Implement
        return True

    def plot(
        self,
        filename_base=None,
        file_extension=None,
        mflay=None,
        fignum=None,
        title=None,
        **kwargs
    ):
        """
        Plot 3-D model input data

        Parameters
        ----------
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
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        ----------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.
        """
        from flopy.plot import PlotUtilities

        if not self.plottable:
            raise TypeError(
                "This MFArray is not plottable likely because modelgrid is "
                "not available."
            )

        modelgrid = self._get_model_grid()
        a = self.array
        num_plottable_layers = modelgrid.get_number_plottable_layers(a)

        if num_plottable_layers == 1:
            axes = PlotUtilities._plot_util2d_helper(
                self,
                title=title,
                filename_base=filename_base,
                file_extension=file_extension,
                fignum=fignum,
                **kwargs
            )
        elif num_plottable_layers > 1:
            axes = PlotUtilities._plot_util3d_helper(
                self,
                filename_base=filename_base,
                file_extension=file_extension,
                mflay=mflay,
                fignum=fignum,
                **kwargs
            )
        else:
            axes = None

        return axes


class MFTransientArray(MFArray, MFTransient):
    """
    Provides an interface for the user to access and update MODFLOW transient
    array data.  MFTransientArray objects are not designed to be directly
    constructed by the end user. When a FloPy for MODFLOW 6 package object is
    constructed, the appropriate MFArray objects are automatically built.

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
    ):
        super().__init__(
            sim_data=sim_data,
            model_or_sim=model_or_sim,
            structure=structure,
            data=None,
            enable=enable,
            path=path,
            dimensions=dimensions,
        )
        self._transient_setup(self._data_storage)
        self.repeating = True

    @property
    def data_type(self):
        """Type of data (DataType) stored in the array"""
        return DataType.transient2d

    def remove_transient_key(self, transient_key):
        """Removes a new transient time `transient_key` and any data stored
        at that time.  This method is intended for internal library usage only.

        Parameters
        ----------
            transient_key : int
                Zero-based stress period

        """
        if transient_key in self._data_storage:
            del self._data_storage[transient_key]

    def add_transient_key(self, transient_key):
        """Adds a new transient time allowing data for that time to be stored
        and retrieved using the key `transient_key`.  This method is intended
        for internal library usage only.

        Parameters
        ----------
            transient_key : int
                Zero-based stress period

        """
        super().add_transient_key(transient_key)
        self._data_storage[transient_key] = super()._new_storage(
            stress_period=transient_key
        )

    def store_as_external_file(
        self,
        external_file_path,
        layer=None,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Stores data from layer `layer` to an external file at
        `external_file_path`.  For unlayered data do not pass in `layer`.
        If layer is not specified all layers will be stored with each layer
        as a separate file. If replace_existing_external is set to False,
        this method will not do anything if the data is already in an
        external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            layer : int
                Which layer to store in external file, `None` value stores all
                layers.
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing
        """
        # store each stress period in separate file(s)
        for sp in self._data_storage.keys():
            self._current_key = sp
            layer_storage = self._get_storage_obj().layer_storage
            if (
                layer_storage.get_total_size() > 0
                and self._get_storage_obj().layer_storage[0].data_storage_type
                != DataStorageType.external_file
            ):
                fname, ext = os.path.splitext(external_file_path)
                if DatumUtil.is_int(sp):
                    full_name = "{}_{}{}".format(fname, sp + 1, ext)
                else:
                    full_name = "{}_{}{}".format(fname, sp, ext)
                super().store_as_external_file(
                    full_name,
                    layer,
                    binary,
                    replace_existing_external,
                    check_data,
                )

    def store_internal(
        self,
        layer=None,
        check_data=True,
    ):
        """Stores data from layer `layer` internally.  For unlayered data do
        not pass in `layer`. If layer is not specified all layers will be
        stored internally.

        Parameters
        ----------
            layer : int
                Which layer to store internally file, `None` value stores all
                layers.
            check_data : bool
                Verify data prior to storing
        """
        for sp in self._data_storage.keys():
            self._current_key = sp
            layer_storage = self._get_storage_obj().layer_storage
            if (
                layer_storage.get_total_size() > 0
                and self._get_storage_obj().layer_storage[0].data_storage_type
                == DataStorageType.external_file
            ):
                super().store_internal(
                    layer,
                    check_data,
                )

    def get_data(self, layer=None, apply_mult=True, **kwargs):
        """Returns the data associated with stress period key `layer`.
        If `layer` is None, returns all data for time `layer`.

        Parameters
        ----------
            layer : int
                Zero-based stress period of data to return
            apply_mult : bool
                Whether to apply multiplier to data prior to returning it

        """
        if self._data_storage is not None and len(self._data_storage) > 0:
            if layer is None:
                output = None
                sim_time = self._data_dimensions.package_dim.model_dim[
                    0
                ].simulation_time
                num_sp = sim_time.get_num_stress_periods()
                if "array" in kwargs:
                    data = None
                    for sp in range(0, num_sp):
                        if sp in self._data_storage:
                            self.get_data_prep(sp)
                            data = super().get_data(
                                apply_mult=apply_mult, **kwargs
                            )
                            data = np.expand_dims(data, 0)
                        else:
                            if data is None:
                                # get any data
                                self.get_data_prep(self._data_storage.key()[0])
                                data = super().get_data(
                                    apply_mult=apply_mult, **kwargs
                                )
                                data = np.expand_dims(data, 0)
                            if self.structure.type == DatumType.integer:
                                data = np.full_like(data, 0)
                            else:
                                data = np.full_like(data, 0.0)
                        if output is None:
                            output = data
                        else:
                            output = np.concatenate((output, data))
                    return output
                else:
                    for sp in range(0, num_sp):
                        data = None
                        if sp in self._data_storage:
                            self.get_data_prep(sp)
                            data = super().get_data(
                                apply_mult=apply_mult, **kwargs
                            )
                        if output is None:
                            if "array" in kwargs:
                                output = [data]
                            else:
                                output = {sp: data}
                        else:
                            if "array" in kwargs:
                                output.append(data)
                            else:
                                output[sp] = data
                    return output
            else:
                self.get_data_prep(layer)
                return super().get_data(apply_mult=apply_mult)
        else:
            return None

    def set_data(self, data, multiplier=None, layer=None, key=None):
        """Sets the contents of the data at layer `layer` and time `key` to
        `data` with multiplier `multiplier`. For unlayered data do not pass
        in `layer`.

        Parameters
        ----------
            data : dict, ndarray, list
                Data being set.  Data can be a dictionary with keys as
                zero-based stress periods and values as the data.  If data is
                an ndarray or list of lists, it will be assigned to the the
                stress period specified in `key`.  If any is set to None, that
                stress period of data will be removed.
            multiplier : int
                multiplier to apply to data
            layer : int
                Layer of data being set.  Keep default of None of data is not
                layered.
            key : int
                Zero based stress period to assign data too.  Does not apply
                if `data` is a dictionary.
        """

        if isinstance(data, dict) or isinstance(data, OrderedDict):
            # each item in the dictionary is a list for one stress period
            # the dictionary key is the stress period the list is for
            del_keys = []
            for key, list_item in data.items():
                if list_item is None:
                    self.remove_transient_key(key)
                    del_keys.append(key)
                else:
                    self._set_data_prep(list_item, key)
                    super().set_data(list_item, multiplier, layer)
            for key in del_keys:
                del data[key]
        else:
            if key is None:
                # search for a key
                new_key_index = self.structure.first_non_keyword_index()
                if (
                    new_key_index is not None
                    and hasattr(data, "__len__")
                    and len(data) > new_key_index
                ):
                    key = data[new_key_index]
                else:
                    key = 0
            if data is None:
                self.remove_transient_key(key)
            else:
                self._set_data_prep(data, key)
                super().set_data(data, multiplier, layer)

    def get_file_entry(
        self, key=0, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        """Returns a string containing the data in stress period "key".

        Parameters
        ----------
            key : int
                The stress period to return file entry for.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """

        self._get_file_entry_prep(key)
        return super().get_file_entry(ext_file_action=ext_file_action)

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        """Loads data from first_line (the first line of data) and open file
        handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
        This method is for internal flopy use and is not intended to be called
        by the end user.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this array.
            file_handle : file descriptor
                A file handle for the data file which points to the second
                line of data for this array
            block_header : MFBlockHeader
                Block header object that contains block header information
                for the block containing this data
            pre_data_comments : MFComment
                Comments immediately prior to the data
            external_file_info : list
                Contains information about storing files externally

        Returns
        -------
            more data : bool,
            next data line : str

        """
        self._load_prep(block_header)
        return super().load(
            first_line, file_handle, pre_data_comments, external_file_info
        )

    def _new_storage(
        self, set_layers=True, base_storage=False, stress_period=0
    ):
        if base_storage:
            if not isinstance(stress_period, int):
                stress_period = 1
            return super()._new_storage(
                set_layers, base_storage, stress_period
            )
        else:
            return OrderedDict()

    def _set_storage_obj(self, storage):
        self._data_storage[self._current_key] = storage

    def _get_storage_obj(self):
        if (
            self._current_key is None
            or self._current_key not in self._data_storage
        ):
            return None
        return self._data_storage[self._current_key]

    def plot(
        self,
        kper=None,
        filename_base=None,
        file_extension=None,
        mflay=None,
        fignum=None,
        **kwargs
    ):
        """
        Plot transient array model input data

        Parameters
        ----------
        transient2d : flopy.utils.util_array.Transient2D object
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.
            kper : str
                MODFLOW zero-based stress period number to return. If
                kper='all' then data for all stress period will be
                extracted. (default is zero).

        Returns
        ----------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.
        """
        from flopy.plot.plotutil import PlotUtilities

        if not self.plottable:
            raise TypeError("Simulation level packages are not plottable")

        axes = PlotUtilities._plot_transient2d_helper(
            self,
            filename_base=filename_base,
            file_extension=file_extension,
            kper=kper,
            fignum=fignum,
            **kwargs
        )
        return axes
