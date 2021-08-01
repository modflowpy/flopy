from copy import deepcopy
import sys
import os
import inspect
from shutil import copyfile
from collections import OrderedDict
from enum import Enum
import numpy as np
from ..mfbase import MFDataException, VerbosityLevel
from ..data.mfstructure import DatumType, MFDataItemStructure
from ..data import mfdatautil
from .mfdatautil import iterable
from ...utils.datautil import (
    DatumUtil,
    FileIter,
    MultiListIter,
    PyListUtil,
    ArrayIndexIter,
    MultiList,
)
from .mfdatautil import convert_data, MFComment
from .mffileaccess import MFFileAccessArray, MFFileAccessList, MFFileAccess


class DataStorageType(Enum):
    """
    Enumeration of different ways that data can be stored
    """

    internal_array = 1
    internal_constant = 2
    external_file = 3


class DataStructureType(Enum):
    """
    Enumeration of different data structures used to store data
    """

    ndarray = 1
    recarray = 2
    scalar = 3


class LayerStorage:
    """
    Stores a single layer of data.

    Parameters
    ----------
    data_storage : DataStorage
        Parent data storage object that layer is contained in
    lay_num : int
        Layer number of layered being stored
    data_storage_type : DataStorageType
        Method used to store the data

    Attributes
    ----------
    internal_data : ndarray or recarray
        data being stored, if full data is being stored internally in memory
    data_const_value : int/float
        constant value of data being stored, if data is a constant
    data_storage_type : DataStorageType
        method used to store the data
    fname : str
        file name of external file containing the data
    factor : int/float
        factor to multiply the data by
    iprn : int
        print code
    binary : bool
        whether the data is stored in a binary file

    Methods
    -------
    get_const_val(layer)
        gets the constant value of a given layer.  data storage type for layer
        must be "internal_constant".
    get_data(layer) : ndarray/recarray/string
        returns the data for the specified layer
    set_data(data, layer=None, multiplier=[1.0]
        sets the data being stored to "data" for layer "layer", replacing all
        data for that layer.  a multiplier can be specified.

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(
        self,
        data_storage,
        lay_indexes,
        data_storage_type=DataStorageType.internal_array,
        data_type=None,
    ):
        self._data_storage_parent = data_storage
        self._lay_indexes = lay_indexes
        self.internal_data = None
        self.data_const_value = None
        self.data_storage_type = data_storage_type
        self.data_type = data_type
        self.fname = None
        if self.data_type == DatumType.integer:
            self.factor = 1
        else:
            self.factor = 1.0
        self.iprn = None
        self.binary = False

    def set_internal_constant(self):
        self.data_storage_type = DataStorageType.internal_constant

    def set_internal_array(self):
        self.data_storage_type = DataStorageType.internal_array

    @property
    def name(self):
        return self._data_storage_parent.data_dimensions.structure.name

    def __repr__(self):
        if self.data_storage_type == DataStorageType.internal_constant:
            return "constant {}".format(self.get_data_const_val())
        else:
            return repr(self.get_data())

    def __str__(self):
        if self.data_storage_type == DataStorageType.internal_constant:
            return str(self.get_data_const_val())
        else:
            return str(self.get_data())

    def __getattr__(self, attr):
        if attr == "binary" or not hasattr(self, "binary"):
            raise AttributeError(attr)

        if attr == "array":
            return self._data_storage_parent.get_data(self._lay_indexes, True)
        elif attr == "__getstate__":
            raise AttributeError(attr)

    def set_data(self, data):
        self._data_storage_parent.set_data(
            data, self._lay_indexes, [self.factor]
        )

    def get_data(self):
        return self._data_storage_parent.get_data(self._lay_indexes, False)

    def get_data_const_val(self):
        if isinstance(self.data_const_value, list):
            return self.data_const_value[0]
        else:
            return self.data_const_value


class DataStorage:
    """
    Stores and retrieves data.


    Parameters
    ----------
    sim_data : simulation data class
        reference to the simulation data class
    data_dimensions : data dimensions class
        a data dimensions class for the data being stored
    get_file_entry : method reference
        method that returns the file entry for the stored data
    data_storage_type : enum
        how the data will be stored (internally, as a constant, as an external
        file)
    data_structure_type : enum
        what internal type is the data stored in (ndarray, recarray, scalar)
    layer_shape : int
        number of data layers
    layered : bool
        is the data layered
    layer_storage : MultiList<LayerStorage>
        one or more dimensional list of LayerStorage

    Attributes
    ----------
    data_storage_type : list
        list of data storage types, one for each layer
    data_const_value : list
        list of data constants, one for each layer
    external_file_path : list
        list of external file paths, one for each layer
    multiplier : list
        list of multipliers, one for each layer
    print_format : list
        list of print formats, one for each layer
    data_structure_type :
        what internal type is the data stored in (ndarray, recarray, scalar)
    layered : bool
        is the data layered
    pre_data_comments : string
        any comments before the start of the data
    comments : OrderedDict
        any comments mixed in with the data, dictionary keys are data lines
    post_data_comments : string
        any comments after the end of the data

    Methods
    -------
    override_data_type : (index, data_type)
        overrides the data type used in a recarray at index "index" with data
        type "data_type"
    get_external_file_path(layer)
        gets the path to an external file for layer "layer"
    get_const_val(layer)
        gets the constant value of a given layer.  data storage type for layer
        must be "internal_constant".
    has_data(layer) : bool
        returns true if data exists for the specified layer, false otherwise
    get_data(layer) : ndarray/recarray/string
        returns the data for the specified layer
    update_item(data, key_index)
        updates the data in a recarray at index "key_index" with data "data".
        data is a list containing all data for a single record in the
        recarray.  .  data structure type must be recarray
    append_data(data)
        appends data "data" to the end of a recarray.  data structure type
        must be recarray
    set_data(data, layer=None, multiplier=[1.0]
        sets the data being stored to "data" for layer "layer", replacing all
        data for that layer.  a multiplier can be specified.
    get_active_layer_indices() : list
        returns the indices of all layers expected to contain data
    store_internal(data, layer=None, const=False, multiplier=[1.0])
        store data "data" at layer "layer" internally
    store_external(file_path, layer=None, multiplier=[1.0], print_format=None,
        data=None, do_not_verify=False) store data "data" at layer "layer"
        externally in file "file_path"
    external_to_external(new_external_file, multiplier=None, layer=None)
        copies existing external data to the new file location and points to
        the new file
    external_to_internal(layer_num=None, store_internal=False) :
      ndarray/recarray
        loads existing external data for layer "layer_num" and returns it.  if
        store_internal is True it also storages the data internally,
        changing the storage type for "layer_num" layer to internal.
    internal_to_external(new_external_file, multiplier=None, layer=None,
                         print_format=None)
        stores existing internal data for layer "layer" to external file
        "new_external_file"
    read_data_from_file(layer, fd=None, multiplier=None) : (ndarray, int)
        reads in data from a given file "fd" as data from layer "layer".
        returns data as an ndarray along with the size of the data
    to_string(val, type, is_cellid=False, possible_cellid=False)
        converts data "val" of type "type" to a string.  is_cellid is True if
        the data type is known to be a cellid and is treated as such.  when
        possible_cellid is True the data is checked to see if it matches the
        shape/dimensions of a cellid before using it as one.
    resolve_data_size(index) : int
        resolves the size of a given data element in a recarray based on the
        names in the existing rec_array.  assumes repeating data element
        names follow the format <data_element_name>_X.  returns the number of
        times the data element repeats.
    flatten()
        converts layered data to a non-layered data
    make_layered()
        converts non-layered data to layered data

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
        data_dimensions,
        get_file_entry,
        data_storage_type=DataStorageType.internal_array,
        data_structure_type=DataStructureType.ndarray,
        layer_shape=(1,),
        layered=False,
        stress_period=0,
        data_path=(),
    ):
        self.data_dimensions = data_dimensions
        self._model_or_sim = model_or_sim
        self._simulation_data = sim_data
        self._get_file_entry = get_file_entry
        self._data_type_overrides = {}
        self._data_storage_type = data_storage_type
        self._stress_period = stress_period
        self._data_path = data_path
        if not data_structure_type == DataStructureType.recarray:
            self._data_type = self.data_dimensions.structure.get_datum_type(
                return_enum_type=True
            )
        else:
            self._data_type = None
        self.layer_storage = MultiList(
            shape=layer_shape, callback=self._create_layer
        )
        # self.layer_storage = [LayerStorage(self, x, data_storage_type)
        #                      for x in range(layer_shape)]
        self.data_structure_type = data_structure_type
        package_dim = self.data_dimensions.package_dim
        self.in_model = (
            self.data_dimensions is not None
            and len(package_dim.package_path) > 1
            and package_dim.model_dim[0].model_name.lower()
            == package_dim.package_path[0]
        )

        if data_structure_type == DataStructureType.recarray:
            self.build_type_list(resolve_data_shape=False)

        self.layered = layered

        # initialize comments
        self.pre_data_comments = None
        self.comments = OrderedDict()

    def __repr__(self):
        return self.get_data_str(True)

    def __str__(self):
        return self.get_data_str(False)

    def _create_layer(self, indexes):
        return LayerStorage(
            self, indexes, self._data_storage_type, self._data_type
        )

    def flatten(self):
        self.layered = False
        storage_type = self.layer_storage.first_item().data_storage_type
        self.layer_storage = MultiList(
            mdlist=[LayerStorage(self, 0, storage_type, self._data_type)]
        )

    def make_layered(self):
        if not self.layered:
            if self.data_structure_type != DataStructureType.ndarray:
                message = (
                    'Data structure type "{}" does not support '
                    "layered data.".format(self.data_structure_type)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "making data layered",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            if (
                self.layer_storage.first_item().data_storage_type
                == DataStorageType.external_file
            ):
                message = (
                    "Converting external file data into layered "
                    "data currently not support."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "making data layered",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )

            previous_storage = self.layer_storage.first_item()
            data = previous_storage.get_data()
            data_dim = self.get_data_dimensions(None)
            self.layer_storage = MultiList(
                shape=(data_dim[0],), callback=self._create_layer
            )
            if (
                previous_storage.data_storage_type
                == DataStorageType.internal_constant
            ):
                for storage in self.layer_storage.elements():
                    storage.data_const_value = (
                        previous_storage.data_const_value
                    )
            elif (
                previous_storage.data_storage_type
                == DataStorageType.internal_array
            ):
                data_ml = MultiList(data)
                if not (
                    data_ml.get_total_size()
                    == self.layer_storage.get_total_size()
                ):
                    message = (
                        "Size of data ({}) does not match expected "
                        "value of {}"
                        ".".format(
                            data_ml.get_total_size(),
                            self.layer_storage.get_total_size(),
                        )
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.data_dimensions.structure.get_model(),
                        self.data_dimensions.structure.get_package(),
                        self.data_dimensions.structure.path,
                        "making data layered",
                        self.data_dimensions.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        message,
                        self._simulation_data.debug,
                    )
                for data_layer, storage in zip(
                    data, self.layer_storage.elements()
                ):
                    storage.internal_data = data_layer
                    storage.factor = previous_storage.factor
                    storage.iprn = previous_storage.iprn
            self.layered = True

    def get_data_str(self, formal):
        data_str = ""
        # Assemble strings for internal array data
        for index, storage in enumerate(self.layer_storage.elements()):
            if self.layered:
                layer_str = "Layer_{}".format(str(index + 1))
            else:
                layer_str = ""
            if storage.data_storage_type == DataStorageType.internal_array:
                if storage.internal_data is not None:
                    header = self._get_layer_header_str(index)
                    if formal:
                        if self.layered:
                            data_str = "{}{}{{{}}}\n({})\n".format(
                                data_str,
                                layer_str,
                                header,
                                repr(self.get_data((index,))),
                            )
                        else:
                            data_str = "{}{}{{{}}}\n({})\n".format(
                                data_str,
                                layer_str,
                                header,
                                repr(self.get_data((index,))),
                            )
                    else:
                        data_str = "{}{}{{{}}}\n({})\n".format(
                            data_str,
                            layer_str,
                            header,
                            str(self.get_data((index,))),
                        )
            elif (
                storage.data_storage_type == DataStorageType.internal_constant
            ):
                if formal:
                    if storage.data_const_value is not None:
                        data_str = "{}{}{{{}}}\n".format(
                            data_str,
                            layer_str,
                            self._get_layer_header_str(index),
                        )
                else:
                    if storage.data_const_value is not None:
                        data_str = "{}{}{{{}}}\n".format(
                            data_str,
                            layer_str,
                            self._get_layer_header_str(index),
                        )
        return data_str

    def _get_layer_header_str(self, layer):
        header_list = []
        if (
            self.layer_storage[layer].data_storage_type
            == DataStorageType.external_file
        ):
            header_list.append(
                "open/close {}".format(self.layer_storage[layer].fname)
            )
        elif (
            self.layer_storage[layer].data_storage_type
            == DataStorageType.internal_constant
        ):
            lr = self.layer_storage[layer]
            header_list.append("constant {}".format(lr))
        else:
            header_list.append("internal")
        if (
            self.layer_storage[layer].factor != 1.0
            and self.layer_storage[layer].factor != 1
            and self.data_structure_type != DataStructureType.recarray
        ):
            header_list.append(
                "factor {}".format(self.layer_storage[layer].factor)
            )
        if self.layer_storage[layer].iprn is not None:
            header_list.append(
                "iprn {}".format(self.layer_storage[layer].iprn)
            )
        if len(header_list) > 0:
            return ", ".join(header_list)
        else:
            return ""

    def init_layers(self, dimensions):
        self.layer_storage = MultiList(
            shape=dimensions, callback=self._create_layer
        )

    def add_layer(self, dimension=2):
        self.layer_storage.increment_dimension(dimension, self._create_layer)

    def override_data_type(self, index, data_type):
        self._data_type_overrides[index] = data_type

    def get_external_file_path(self, layer):
        if layer is None:
            return self.layer_storage[0].fname
        else:
            return self.layer_storage[layer].fname

    def get_const_val(self, layer=None):
        if layer is None:
            if not self.layer_storage.get_total_size() >= 1:
                message = "Can not get constant value. No data is available."
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "getting constant value",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            first_item = self.layer_storage.first_item()
            if (
                not first_item.data_storage_type
                == DataStorageType.internal_constant
            ):
                message = (
                    "Can not get constant value. Storage type must be "
                    "internal_constant."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "getting constant value",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )

            return first_item.get_data_const_val()
        else:
            if not self.layer_storage.in_shape(layer):
                message = (
                    'Can not get constant value. Layer "{}" is not a '
                    "valid layer.".format(layer)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "getting constant value",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            if (
                not self.layer_storage[layer].data_storage_type
                == DataStorageType.internal_constant
            ):
                message = (
                    "Can not get constant value. Storage type must be "
                    "internal_constant."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "getting constant value",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            return self.layer_storage[layer].get_data_const_val()

    def has_data(self, layer=None):
        ret_val = self._access_data(layer, False)
        return ret_val is not None and ret_val != False

    def get_data(self, layer=None, apply_mult=True):
        return self._access_data(layer, True, apply_mult=apply_mult)

    def _access_data(self, layer, return_data=False, apply_mult=True):
        layer_check = self._resolve_layer(layer)
        if (
            self.layer_storage[layer_check].internal_data is None
            and self.layer_storage[layer_check].data_storage_type
            == DataStorageType.internal_array
        ) or (
            self.layer_storage[layer_check].data_const_value is None
            and self.layer_storage[layer_check].data_storage_type
            == DataStorageType.internal_constant
        ):
            return None
        if (
            layer is None
            and (
                self.data_structure_type == DataStructureType.ndarray
                or self.data_structure_type == DataStructureType.scalar
            )
            and return_data
        ):
            # return data from all layers
            data = self._build_full_data(apply_mult)
            if data is None:
                if (
                    self.layer_storage.first_item().data_storage_type
                    == DataStorageType.internal_constant
                ):
                    return self.layer_storage.first_item().get_data()[0]
            else:
                return data

        if (
            self.layer_storage[layer_check].data_storage_type
            == DataStorageType.external_file
        ):
            if return_data:
                return self.external_to_internal(layer)
            else:
                return True
        else:
            if (
                self.data_structure_type == DataStructureType.ndarray
                and self.layer_storage[layer_check].data_const_value is None
                and self.layer_storage[layer_check].internal_data is None
            ):
                return None
            if not (layer is None or self.layer_storage.in_shape(layer)):
                message = 'Layer "{}" is an invalid layer.'.format(layer)
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "accessing data",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            if layer is None:
                if (
                    self.data_structure_type == DataStructureType.ndarray
                    or self.data_structure_type == DataStructureType.scalar
                ):
                    if self.data_structure_type == DataStructureType.scalar:
                        return (
                            self.layer_storage.first_item().internal_data
                            is not None
                        )
                    check_storage = self.layer_storage[layer_check]
                    return (
                        check_storage.data_const_value is not None
                        and check_storage.data_storage_type
                        == DataStorageType.internal_constant
                    ) or (
                        check_storage.internal_data is not None
                        and check_storage.data_storage_type
                        == DataStorageType.internal_array
                    )
                else:
                    if (
                        self.layer_storage[layer_check].data_storage_type
                        == DataStorageType.internal_constant
                    ):
                        if return_data:
                            # recarray stored as a constant.  currently only
                            # support grid-based constant recarrays.  build
                            # a recarray of all cells
                            data_list = []
                            model_grid = self.data_dimensions.get_model_grid()
                            structure = self.data_dimensions.structure
                            package_dim = self.data_dimensions.package_dim
                            for cellid in model_grid.get_all_model_cells():
                                first_item = self.layer_storage.first_item()
                                data_line = (cellid,) + (
                                    first_item.data_const_value,
                                )
                                if len(structure.data_item_structures) > 2:
                                    # append None any expected optional data
                                    for (
                                        data_item_struct
                                    ) in structure.data_item_structures[2:]:
                                        if (
                                            data_item_struct.name
                                            != "boundname"
                                            or package_dim.boundnames()
                                        ):
                                            data_line = data_line + (None,)
                                data_list.append(data_line)
                            type_list = self.resolve_typelist(data_list)
                            return np.rec.array(data_list, type_list)
                        else:
                            return (
                                self.layer_storage[
                                    layer_check
                                ].data_const_value
                                is not None
                            )
                    else:
                        if return_data:
                            return (
                                self.layer_storage.first_item().internal_data
                            )
                        else:
                            return True
            elif (
                self.layer_storage[layer].data_storage_type
                == DataStorageType.internal_array
            ):
                if return_data:
                    return self.layer_storage[layer].internal_data
                else:
                    return self.layer_storage[layer].internal_data is not None
            elif (
                self.layer_storage[layer].data_storage_type
                == DataStorageType.internal_constant
            ):
                layer_storage = self.layer_storage[layer]
                if return_data:
                    data = self._fill_const_layer(layer)
                    if data is None:
                        if (
                            layer_storage.data_storage_type
                            == DataStructureType.internal_constant
                        ):
                            return layer_storage.data_const_value[0]
                    else:
                        return data
                else:
                    return layer_storage.data_const_value is not None
            else:
                if return_data:
                    return self.get_external(layer)
                else:
                    return True

    def append_data(self, data):
        # currently only support appending to recarrays
        if not (self.data_structure_type == DataStructureType.recarray):
            message = (
                'Can not append to data structure "{}". Can only '
                "append to a recarray datastructure"
                ".".format(self.data_structure_type)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "appending data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )
        internal_data = self.layer_storage.first_item().internal_data

        if internal_data is None:
            if len(data[0]) != len(self._recarray_type_list):
                # rebuild type list using existing data as a guide
                self.build_type_list(data=data)
            self.set_data(np.rec.array(data, self._recarray_type_list))
        else:
            if len(self.layer_storage.first_item().internal_data[0]) < len(
                data[0]
            ):
                # Rebuild recarray to fit larger size
                count = 0
                last_count = len(data[0]) - len(internal_data[0])
                while count < last_count:
                    self._duplicate_last_item()
                    count += 1
                internal_data_list = internal_data.tolist()
                for data_item in data:
                    internal_data_list.append(data_item)
                self._add_placeholders(internal_data_list)
                self.set_data(
                    np.rec.array(internal_data_list, self._recarray_type_list)
                )
            else:
                first_item = self.layer_storage.first_item()
                if len(first_item.internal_data[0]) > len(data[0]):
                    # Add placeholders to data
                    self._add_placeholders(data)
                self.set_data(
                    np.hstack(
                        (
                            internal_data,
                            np.rec.array(data, self._recarray_type_list),
                        )
                    )
                )

    def set_data(
        self,
        data,
        layer=None,
        multiplier=None,
        key=None,
        autofill=False,
        check_data=False,
    ):
        if multiplier is None:
            multiplier = [1.0]
        if (
            self.data_structure_type == DataStructureType.recarray
            or self.data_structure_type == DataStructureType.scalar
        ):
            self._set_list(data, layer, multiplier, key, autofill, check_data)
        else:
            self._set_array(data, layer, multiplier, key, autofill)

    def _set_list(
        self, data, layer, multiplier, key, autofill, check_data=False
    ):
        if isinstance(data, dict):
            if "filename" in data:
                if "binary" in data and data["binary"]:
                    if self.data_dimensions.package_dim.boundnames():
                        message = (
                            "Unable to store list data ({}) to a binary "
                            "file when using boundnames"
                            ".".format(self.data_dimensions.structure.name)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.data_dimensions.structure.get_model(),
                            self.data_dimensions.structure.get_package(),
                            self.data_dimensions.structure.path,
                            "writing list data to binary file",
                            self.data_dimensions.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
                self.process_open_close_line(data, layer)
                return
            elif "data" in data:
                data = data["data"]
        if isinstance(data, list):
            if (
                len(data) > 0
                and not isinstance(data[0], tuple)
                and not isinstance(data[0], list)
            ):
                # single line of data needs to be encapsulated in a tuple
                data = [tuple(data)]
        self.store_internal(
            data,
            layer,
            False,
            multiplier,
            key=key,
            autofill=autofill,
            check_data=check_data,
        )

    def _set_array(self, data, layer, multiplier, key, autofill):
        # make a list out of a single item
        if (
            isinstance(data, int)
            or isinstance(data, float)
            or isinstance(data, str)
        ):
            data = [data]

        # check for possibility of multi-layered data
        success = False
        layer_num = 0
        if (
            layer is None
            and self.data_structure_type == DataStructureType.ndarray
            and len(data) == self.layer_storage.get_total_size()
            and not isinstance(data, dict)
        ):
            # loop through list and try to store each list entry as a layer
            success = True
            for layer_num, layer_data in enumerate(data):
                if (
                    not isinstance(layer_data, list)
                    and not isinstance(layer_data, dict)
                    and not isinstance(layer_data, np.ndarray)
                ):
                    layer_data = [layer_data]
                layer_index = self.layer_storage.nth_index(layer_num)
                success = success and self._set_array_layer(
                    layer_data, layer_index, multiplier, key
                )
        if not success:
            # try to store as a single layer
            success = self._set_array_layer(data, layer, multiplier, key)
        self.layered = bool(self.layer_storage.get_total_size() > 1)
        if not success:
            message = (
                'Unable to set data "{}" layer {}.  Data is not '
                "in a valid format"
                ".".format(self.data_dimensions.structure.name, layer_num)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "setting array data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

    def _set_array_layer(self, data, layer, multiplier, key):
        # look for a single constant value
        data_type = self.data_dimensions.structure.get_datum_type(
            return_enum_type=True
        )
        if not isinstance(data, dict) and not isinstance(data, str):
            if self._calc_data_size(data, 2) == 1 and self._is_type(
                data[0], data_type
            ):
                # store data as const
                self.store_internal(data, layer, True, multiplier, key=key)
                return True

        # look for internal and open/close data
        if isinstance(data, dict):
            if "data" in data:
                if (
                    isinstance(data["data"], int)
                    or isinstance(data["data"], float)
                    or isinstance(data["data"], str)
                ):
                    # data should always in in a list/array
                    data["data"] = [data["data"]]

            if "filename" in data:
                multiplier, iprn, binary = self.process_open_close_line(
                    data, layer
                )[0:3]
                # store location to file
                self.store_external(
                    data["filename"],
                    layer,
                    [multiplier],
                    print_format=iprn,
                    binary=binary,
                    do_not_verify=True,
                )
                return True
            elif "data" in data:
                multiplier, iprn = self.process_internal_line(data)
                if len(data["data"]) == 1 and (
                    DatumUtil.is_float(data["data"][0])
                    or DatumUtil.is_int(data["data"][0])
                ):
                    # merge multiplier with single value and make constant
                    if DatumUtil.is_float(multiplier):
                        mult = 1.0
                    else:
                        mult = 1
                    self.store_internal(
                        [data["data"][0] * multiplier],
                        layer,
                        True,
                        [mult],
                        key=key,
                        print_format=iprn,
                    )
                else:
                    self.store_internal(
                        data["data"],
                        layer,
                        False,
                        [multiplier],
                        key=key,
                        print_format=iprn,
                    )
                return True
        elif isinstance(data[0], str):
            if data[0].lower() == "internal":
                multiplier, iprn = self.process_internal_line(data)
                self.store_internal(
                    data[-1],
                    layer,
                    False,
                    [multiplier],
                    key=key,
                    print_format=iprn,
                )
                return True
            elif data[0].lower() != "open/close":
                # assume open/close is just omitted
                new_data = data[:]
                new_data.insert(0, "open/close")
            else:
                new_data = data[:]
            self.process_open_close_line(new_data, layer, True)
            return True
        # try to resolve as internal array
        layer_storage = self.layer_storage[self._resolve_layer(layer)]
        if not (
            layer_storage.data_storage_type
            == DataStorageType.internal_constant
            and PyListUtil.has_one_item(data)
        ):
            # store data as is
            try:
                self.store_internal(data, layer, False, multiplier, key=key)
            except MFDataException:
                return False
            return True
        return False

    def get_active_layer_indices(self):
        layer_index = []
        for index in self.layer_storage.indexes():
            if (
                self.layer_storage[index].fname is not None
                or self.layer_storage[index].internal_data is not None
            ):
                layer_index.append(index)
        return layer_index

    def get_external(self, layer=None):
        if not (layer is None or self.layer_storage.in_shape(layer)):
            message = 'Can not get external data for layer "{}"' ".".format(
                layer
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "getting external data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

    def store_internal(
        self,
        data,
        layer=None,
        const=False,
        multiplier=None,
        key=None,
        autofill=False,
        print_format=None,
        check_data=False,
    ):
        if multiplier is None:
            multiplier = [self.get_default_mult()]
        if self.data_structure_type == DataStructureType.recarray:
            if (
                self.layer_storage.first_item().data_storage_type
                == DataStorageType.internal_constant
            ):
                self.layer_storage.first_item().data_const_value = data
            else:
                self.layer_storage.first_item().data_storage_type = (
                    DataStorageType.internal_array
                )
                if data is None or isinstance(data, np.recarray):
                    if self._simulation_data.verify_data and check_data:
                        self._verify_list(data)
                    self.layer_storage.first_item().internal_data = data
                else:
                    if data is None:
                        self.set_data(None)
                    self.build_type_list()
                    if isinstance(data, list):
                        # look for single strings in list that describe
                        # multiple items
                        new_data = []
                        for item in data:
                            if isinstance(item, str):
                                # parse possible multi-item string
                                new_data.append(
                                    self._resolve_data_line(item, key)
                                )
                            else:
                                new_data.append(item)
                        data = new_data
                    if isinstance(data, str):
                        # parse possible multi-item string
                        data = [self._resolve_data_line(data, key)]

                    if (
                        data is not None
                        and check_data
                        and self._simulation_data.verify_data
                    ):
                        # check data line length
                        self._check_list_length(data)

                    if isinstance(data, np.recarray):
                        self.layer_storage.first_item().internal_data = data
                    elif autofill and data is not None:
                        if isinstance(data, tuple) and isinstance(
                            data[0], tuple
                        ):
                            # convert to list of tuples
                            data = list(data)
                        if isinstance(data, list) and DatumUtil.is_basic_type(
                            data[0]
                        ):
                            # this is a simple list, turn it into a tuple
                            # inside a list so that it is interpreted
                            # correctly by numpy.recarray
                            tupled_data = ()
                            for data_item in data:
                                tupled_data += (data_item,)
                            data = [tupled_data]

                        if not isinstance(data, list):
                            # put data in a list format for recarray
                            data = [(data,)]
                        # auto-fill tagged keyword
                        structure = self.data_dimensions.structure
                        data_item_structs = structure.data_item_structures
                        if (
                            data_item_structs[0].tagged
                            and not data_item_structs[0].type
                            == DatumType.keyword
                        ):
                            for data_index, data_entry in enumerate(data):
                                if (
                                    isinstance(data_entry[0], str)
                                    and data_entry[0].lower()
                                    == data_item_structs[0].name.lower()
                                ):
                                    break
                                data[data_index] = (
                                    data_item_structs[0].name.lower(),
                                ) + data[data_index]
                    if data is not None:
                        new_data = self._build_recarray(data, key, autofill)
                        self.layer_storage.first_item().internal_data = (
                            new_data
                        )
        elif self.data_structure_type == DataStructureType.scalar:
            if data == [()]:
                data = [(True,)]
            self.layer_storage.first_item().internal_data = data
        else:
            layer, multiplier = self._store_prep(layer, multiplier)
            dimensions = self.get_data_dimensions(layer)
            if const:
                self.layer_storage[
                    layer
                ].data_storage_type = DataStorageType.internal_constant
                self.layer_storage[layer].data_const_value = [
                    mfdatautil.get_first_val(data)
                ]
            else:
                self.layer_storage[
                    layer
                ].data_storage_type = DataStorageType.internal_array
                try:
                    self.layer_storage[layer].internal_data = np.reshape(
                        data, dimensions
                    )
                except:
                    message = (
                        'An error occurred when reshaping data "{}" to store. '
                        "Expected data dimensions: {}".format(
                            self.data_dimensions.structure.name, dimensions
                        )
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        self.data_dimensions.structure.get_model(),
                        self.data_dimensions.structure.get_package(),
                        self.data_dimensions.structure.path,
                        "setting array data",
                        self.data_dimensions.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        message,
                        self._simulation_data.debug,
                    )
            self.layer_storage[layer].factor = multiplier
            self.layer_storage[layer].iprn = print_format

    def _resolve_data_line(self, data, key):
        if len(self._recarray_type_list) > 1:
            # add any missing leading keywords to the beginning of the string
            data_lst = data.strip().split()
            data_lst_updated = []
            struct = self.data_dimensions.structure
            for data_item_index, data_item in enumerate(
                struct.data_item_structures
            ):
                print(data_item)
                if data_item.type == DatumType.keyword:
                    if data_lst[0].lower() != data_item.name.lower():
                        data_lst_updated.append(data_item.name)
                    else:
                        data_lst_updated.append(data_lst.pop(0))
                else:
                    if (
                        struct.type == DatumType.record
                        and data_lst[0].lower() != data_item.name.lower()
                    ):
                        data_lst_updated.append(data_item.name)
                    break
            data_lst_updated += data_lst

            # parse the string as if it is being read from a package file
            file_access = MFFileAccessList(
                self.data_dimensions.structure,
                self.data_dimensions,
                self._simulation_data,
                self._data_path,
                self._stress_period,
            )
            data_loaded = []
            data_out = file_access.load_list_line(
                self,
                data_lst_updated,
                0,
                data_loaded,
                False,
                current_key=key,
                data_line=data,
                zero_based=True,
            )[1]
            return tuple(data_out)
        return data

    def _get_min_record_entries(self, data=None):
        try:
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            type_list = self.build_type_list(data=data, min_size=True)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "getting min record entries",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )
        return len(type_list)

    def _check_line_size(self, data_line, min_line_size):
        if 0 < len(data_line) < min_line_size:
            message = (
                "Data line {} only has {} entries, "
                "minimum number of entries is "
                "{}.".format(data_line, len(data_line), min_line_size)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "storing data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

    def _check_list_length(self, data_check):
        if iterable(data_check):
            # verify data length
            min_line_size = self._get_min_record_entries(data_check)
            if isinstance(data_check[0], np.record) or (
                iterable(data_check[0]) and not isinstance(data_check[0], str)
            ):
                # data contains multiple records
                for data_line in data_check:
                    self._check_line_size(data_line, min_line_size)
            else:
                self._check_line_size(data_check, min_line_size)

    def _build_recarray(self, data, key, autofill):
        if not mfdatautil.PyListUtil.is_iterable(data) or len(data) == 0:
            # set data to build empty recarray
            data = [()]
        self.build_type_list(data=data, key=key)
        if not self.tuple_cellids(data):
            # fix data so cellid is a single tuple
            data = self.make_tuple_cellids(data)

        if autofill and data is not None:
            # resolve any fields with data types that do not
            # agree with the expected type list
            self._resolve_multitype_fields(data)
        if isinstance(data, list):
            # data needs to be stored as tuples within a list.
            # if this is not the case try to fix it
            self._tupleize_data(data)
            # add placeholders to data so it agrees with
            # expected dimensions of recarray
            self._add_placeholders(data)
        try:
            new_data = np.rec.array(data, self._recarray_type_list)
        except:
            data_expected = []
            for data_type in self._recarray_type_list:
                data_expected.append("<{}>".format(data_type[0]))
            message = (
                "An error occurred when storing data "
                '"{}" in a recarray. {} data is a one '
                "or two dimensional list containing "
                'the variables "{}" (some variables '
                "may be optional, see MF6 "
                'documentation), but data "{}" was '
                "supplied.".format(
                    self.data_dimensions.structure.name,
                    self.data_dimensions.structure.name,
                    " ".join(data_expected),
                    data,
                )
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "setting array data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )
        if self._simulation_data.verify_data:
            self._verify_list(new_data)
        return new_data

    def make_tuple_cellids(self, data):
        # convert cellids from individual layer, row, column fields into
        # tuples (layer, row, column)
        data_dim = self.data_dimensions
        model_grid = data_dim.get_model_grid()
        cellid_size = model_grid.get_num_spatial_coordinates()

        new_data = []
        current_cellid = ()
        for line in data:
            new_line = []
            for item, is_cellid in zip(line, self.recarray_cellid_list_ex):
                if is_cellid:
                    current_cellid += (item,)
                    if len(current_cellid) == cellid_size:
                        new_line.append(current_cellid)
                        current_cellid = ()
                else:
                    new_line.append(item)
            new_data.append(tuple(new_line))
        return new_data

    def tuple_cellids(self, data):
        for data_entry, cellid in zip(data[0], self.recarray_cellid_list):
            if cellid:
                if isinstance(data_entry, int):
                    # cellid is stored in separate columns in the recarray
                    # (eg: one column for layer one column for row and
                    # one columne for column)
                    return False
                else:
                    # cellid is stored in a single column in the recarray
                    # as a tuple
                    return True
        return True

    def resolve_typelist(self, data):
        if self.tuple_cellids(data):
            return self._recarray_type_list
        else:
            return self._recarray_type_list_ex

    def resolve_cellidlist(self, data):
        if self.tuple_cellids(data):
            return self.recarray_cellid_list
        else:
            return self.recarray_cellid_list_ex

    def _resolve_multitype_fields(self, data):
        # find any data fields where the data is not a consistent type
        itype_len = len(self._recarray_type_list)
        for data_entry in data:
            for index, data_val in enumerate(data_entry):
                if (
                    index < itype_len
                    and self._recarray_type_list[index][1] != object
                    and not isinstance(
                        data_val, self._recarray_type_list[index][1]
                    )
                    and (
                        not isinstance(data_val, int)
                        or self._recarray_type_list[index][1] != float
                    )
                ):
                    # for inconsistent types use generic object type
                    self._recarray_type_list[index] = (
                        self._recarray_type_list[index][0],
                        object,
                    )

    def store_external(
        self,
        file_path,
        layer=None,
        multiplier=None,
        print_format=None,
        data=None,
        do_not_verify=False,
        binary=False,
    ):
        if multiplier is None:
            multiplier = [self.get_default_mult()]
        layer_new, multiplier = self._store_prep(layer, multiplier)

        # pathing to external file
        data_dim = self.data_dimensions
        model_name = data_dim.package_dim.model_dim[0].model_name
        fp_relative = file_path
        if model_name is not None:
            rel_path = self._simulation_data.mfpath.model_relative_path[
                model_name
            ]
            if rel_path is not None and len(rel_path) > 0 and rel_path != ".":
                # include model relative path in external file path
                fp_relative = os.path.join(rel_path, file_path)
        fp = self._simulation_data.mfpath.resolve_path(fp_relative, model_name)
        if data is not None:
            if self.data_structure_type == DataStructureType.recarray:
                # create external file and write file entry to the file
                # store data internally first so that a file entry
                # can be generated
                self.store_internal(
                    data,
                    layer_new,
                    False,
                    [multiplier],
                    None,
                    False,
                    print_format,
                )
                if binary:
                    file_access = MFFileAccessList(
                        self.data_dimensions.structure,
                        self.data_dimensions,
                        self._simulation_data,
                        self._data_path,
                        self._stress_period,
                    )
                    file_access.write_binary_file(
                        self.layer_storage.first_item().internal_data,
                        fp,
                        self._model_or_sim.modeldiscrit,
                        precision="double",
                    )
                else:
                    try:
                        fd = open(fp, "w")
                    except:
                        message = (
                            "Unable to open file {}.  Make sure the "
                            "file is not locked and the folder exists"
                            ".".format(fp)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.data_dimensions.structure.get_model(),
                            self.data_dimensions.structure.get_package(),
                            self.data_dimensions.structure.path,
                            "opening external file for writing",
                            data_dim.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
                    ext_file_entry = self._get_file_entry()
                    fd.write(ext_file_entry)
                    fd.close()

                # set as external data
                self.layer_storage.first_item().internal_data = None
            else:
                # store data externally in file
                data_size = self.get_data_size(layer_new)
                data_type = data_dim.structure.data_item_structures[0].type

                if self._calc_data_size(data, 2) == 1 and data_size > 1:
                    # constant data, need to expand
                    self.layer_storage[layer_new].data_const_value = data
                    self.layer_storage[
                        layer_new
                    ].data_storage_type = DataStorageType.internal_constant
                    data = self._fill_const_layer(layer)
                elif isinstance(data, list):
                    data = self._to_ndarray(data, layer)
                if binary:
                    text = self.data_dimensions.structure.name
                    file_access = MFFileAccessArray(
                        self.data_dimensions.structure,
                        self.data_dimensions,
                        self._simulation_data,
                        self._data_path,
                        self._stress_period,
                    )
                    str_layered = self.data_dimensions.structure.layered
                    file_access.write_binary_file(
                        data,
                        fp,
                        text,
                        self._model_or_sim.modeldiscrit,
                        self._model_or_sim.modeltime,
                        stress_period=self._stress_period,
                        precision="double",
                        write_multi_layer=(layer is None and str_layered),
                    )
                else:
                    file_access = MFFileAccessArray(
                        self.data_dimensions.structure,
                        self.data_dimensions,
                        self._simulation_data,
                        self._data_path,
                        self._stress_period,
                    )
                    file_access.write_text_file(
                        data,
                        fp,
                        data_type,
                        data_size,
                    )
                self.layer_storage[layer_new].factor = multiplier
                self.layer_storage[layer_new].internal_data = None
                self.layer_storage[layer_new].data_const_value = None

        else:
            if self.data_structure_type == DataStructureType.recarray:
                self.layer_storage.first_item().internal_data = None
            else:
                self.layer_storage[layer_new].factor = multiplier
                self.layer_storage[layer_new].internal_data = None
        self.set_ext_file_attributes(
            layer_new, fp_relative, print_format, binary
        )

    def set_ext_file_attributes(self, layer, file_path, print_format, binary):
        # point to the external file and set flags
        self.layer_storage[layer].fname = file_path
        self.layer_storage[layer].iprn = print_format
        self.layer_storage[layer].binary = binary
        self.layer_storage[
            layer
        ].data_storage_type = DataStorageType.external_file

    def point_to_existing_external_file(self, arr_line, layer):
        (
            multiplier,
            print_format,
            binary,
            data_file,
        ) = self.process_open_close_line(arr_line, layer, store=False)
        self.set_ext_file_attributes(layer, data_file, print_format, binary)
        self.layer_storage[layer].factor = multiplier

    def external_to_external(
        self, new_external_file, multiplier=None, layer=None, binary=None
    ):
        # currently only support files containing ndarrays
        if not (self.data_structure_type == DataStructureType.ndarray):
            message = (
                'Can not copy external file of type "{}". Only '
                "files containing ndarrays currently supported"
                ".".format(self.data_structure_type)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "copy external file",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )
        if not (
            (layer is None and self.layer_storage.get_total_size() == 1)
            or (layer is not None and self.layer_storage.in_shape(layer))
        ):
            if layer is None:
                message = (
                    "When no layer is supplied the data must contain "
                    "only one layer. Data contains {} layers"
                    ".".format(self.layer_storage.get_total_size())
                )
            else:
                message = 'layer "{}" is not a valid layer'.format(layer)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "copy external file",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )
        # get data storage
        if layer is None:
            layer = 1
        if self.layer_storage[layer].fname is None:
            message = "No file name exists for layer {}.".format(layer)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "copy external file",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

        # copy file to new location
        copyfile(self.layer_storage[layer].fname, new_external_file)

        # update
        if binary is None:
            binary = self.layer_storage[layer].binary
        self.store_external(
            new_external_file,
            layer,
            [self.layer_storage[layer].factor],
            self.layer_storage[layer].iprn,
            binary=binary,
        )

    def external_to_internal(self, layer, store_internal=False):
        if layer is None:
            layer = 0
        # load data from external file
        model_name = self.data_dimensions.package_dim.model_dim[0].model_name
        read_file = self._simulation_data.mfpath.resolve_path(
            self.layer_storage[layer].fname, model_name
        )
        # currently support files containing ndarrays or recarrays
        if self.data_structure_type == DataStructureType.ndarray:
            file_access = MFFileAccessArray(
                self.data_dimensions.structure,
                self.data_dimensions,
                self._simulation_data,
                self._data_path,
                self._stress_period,
            )
            if self.layer_storage[layer].binary:
                data_out = file_access.read_binary_data_from_file(
                    read_file,
                    self.get_data_dimensions(layer),
                    self.get_data_size(layer),
                    self._data_type,
                    self._model_or_sim.modeldiscrit,
                )[0]
            else:
                data_out = file_access.read_text_data_from_file(
                    self.get_data_size(layer),
                    self._data_type,
                    self.get_data_dimensions(layer),
                    layer,
                    read_file,
                )[0]
            if self.layer_storage[layer].factor is not None:
                data_out = data_out * self.layer_storage[layer].factor

            if store_internal:
                self.store_internal(data_out, layer)
            return data_out
        elif self.data_structure_type == DataStructureType.recarray:
            file_access = MFFileAccessList(
                self.data_dimensions.structure,
                self.data_dimensions,
                self._simulation_data,
                self._data_path,
                self._stress_period,
            )
            if self.layer_storage[layer].binary:
                data = file_access.read_binary_data_from_file(
                    read_file, self._model_or_sim.modeldiscrit
                )
                data_out = self._build_recarray(data, layer, False)
            else:
                with open(read_file, "r") as fd_read_file:
                    data_out = file_access.read_list_data_from_file(
                        fd_read_file,
                        self,
                        self._stress_period,
                        store_internal=False,
                    )
            if store_internal:
                self.store_internal(data_out, layer)
            return data_out
        else:
            path = self.data_dimensions.structure.path
            message = (
                "Can not convert {} to internal data. External to "
                "internal file operations currently only supported "
                "for ndarrays.".format(path[-1])
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "opening external file for writing",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

    def internal_to_external(
        self,
        new_external_file,
        multiplier=None,
        layer=None,
        print_format=None,
        binary=False,
    ):
        if layer is None:
            layer_item = self.layer_storage.first_item()
        else:
            layer_item = self.layer_storage[layer]
        if layer_item.data_storage_type == DataStorageType.internal_array:
            data = layer_item.internal_data
        else:
            data = self._fill_const_layer(layer)
        self.store_external(
            new_external_file,
            layer,
            multiplier,
            print_format,
            data,
            binary=binary,
        )

    def resolve_shape_list(
        self,
        data_item,
        repeat_count,
        current_key,
        data_line,
        cellid_size=None,
    ):
        struct = self.data_dimensions.structure
        try:
            resolved_shape, shape_rule = self.data_dimensions.get_data_shape(
                data_item, struct, data_line, repeating_key=current_key
            )
        except Exception as se:
            comment = (
                'Unable to resolve shape for data "{}" field "{}"'
                ".".format(struct.name, data_item.name)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                struct.get_model(),
                struct.get_package(),
                struct.path,
                "loading data list from package file",
                struct.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self._simulation_data.debug,
                se,
            )

        if cellid_size is not None:
            data_item.remove_cellid(resolved_shape, cellid_size)

        if len(resolved_shape) == 1:
            if repeat_count < resolved_shape[0]:
                return True, shape_rule is not None
            elif resolved_shape[0] == -9999:
                # repeating unknown number of times in 1-D array
                return False, True
        return False, False

    def _validate_cellid(self, arr_line, data_index):
        if not self.data_dimensions.structure.model_data:
            # not model data so this is not a cell id
            return False
        if arr_line is None:
            return False
        model_grid = self.data_dimensions.get_model_grid()
        cellid_size = model_grid.get_num_spatial_coordinates()
        if cellid_size + data_index > len(arr_line):
            return False
        for index, dim_size in zip(
            range(data_index, cellid_size + data_index),
            model_grid.get_model_dim(),
        ):
            if not DatumUtil.is_int(arr_line[index]):
                return False
            val = int(arr_line[index])
            if val <= 0 or val > dim_size:
                return False
        return True

    def add_data_line_comment(self, comment, line_num):
        if line_num in self.comments:
            self.comments[line_num].add_text("\n")
            self.comments[line_num].add_text(" ".join(comment))
        else:
            self.comments[line_num] = MFComment(
                " ".join(comment),
                self.data_dimensions.structure.path,
                self._simulation_data,
                line_num,
            )

    def process_internal_line(self, arr_line):
        multiplier = self.get_default_mult()
        print_format = None
        if isinstance(arr_line, list):
            index = 1
            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    word = arr_line[index].lower()
                    if word == "factor" and index + 1 < len(arr_line):
                        multiplier = convert_data(
                            arr_line[index + 1],
                            self.data_dimensions,
                            self._data_type,
                        )
                        index += 2
                    elif word == "iprn" and index + 1 < len(arr_line):
                        print_format = arr_line[index + 1]
                        index += 2
                    else:
                        break
                else:
                    break
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == "factor":
                    multiplier = convert_data(
                        value, self.data_dimensions, self._data_type
                    )
                if key.lower() == "iprn":
                    print_format = value
        return multiplier, print_format

    def process_open_close_line(self, arr_line, layer, store=True):
        # process open/close line
        index = 2
        if self._data_type == DatumType.integer:
            multiplier = 1
        else:
            multiplier = 1.0
        print_format = None
        binary = False
        data_file = None
        data = None

        data_dim = self.data_dimensions
        if isinstance(arr_line, list):
            if len(arr_line) < 2 and store:
                message = (
                    'Data array "{}" contains a OPEN/CLOSE '
                    "that is not followed by a file. {}".format(
                        data_dim.structure.name, data_dim.structure.path
                    )
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "processing open/close line",
                    data_dim.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    word = arr_line[index].lower()
                    if word == "factor" and index + 1 < len(arr_line):
                        try:
                            multiplier = convert_data(
                                arr_line[index + 1],
                                self.data_dimensions,
                                self._data_type,
                            )
                        except Exception as ex:
                            message = (
                                "Data array {} contains an OPEN/CLOSE "
                                "with an invalid multiplier following "
                                'the "factor" keyword.'
                                ".".format(data_dim.structure.name)
                            )
                            type_, value_, traceback_ = sys.exc_info()
                            raise MFDataException(
                                self.data_dimensions.structure.get_model(),
                                self.data_dimensions.structure.get_package(),
                                self.data_dimensions.structure.path,
                                "processing open/close line",
                                data_dim.structure.name,
                                inspect.stack()[0][3],
                                type_,
                                value_,
                                traceback_,
                                message,
                                self._simulation_data.debug,
                                ex,
                            )
                        index += 2
                    elif word == "iprn" and index + 1 < len(arr_line):
                        print_format = arr_line[index + 1]
                        index += 2
                    elif word == "data" and index + 1 < len(arr_line):
                        data = arr_line[index + 1]
                        index += 2
                    elif word == "binary" or word == "(binary)":
                        binary = True
                        index += 1
                    else:
                        break
                else:
                    break
                # save comments
            if index < len(arr_line):
                self.layer_storage[layer].comments = MFComment(
                    " ".join(arr_line[index:]),
                    self.data_dimensions.structure.path,
                    self._simulation_data,
                    layer,
                )
            if arr_line[0].lower() == "open/close":
                data_file = arr_line[1]
            else:
                data_file = arr_line[0]
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == "factor":
                    try:
                        multiplier = convert_data(
                            value, self.data_dimensions, self._data_type
                        )
                    except Exception as ex:
                        message = (
                            "Data array {} contains an OPEN/CLOSE "
                            "with an invalid factor following the "
                            '"factor" keyword.'
                            ".".format(data_dim.structure.name)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.data_dimensions.structure.get_model(),
                            self.data_dimensions.structure.get_package(),
                            self.data_dimensions.structure.path,
                            "processing open/close line",
                            data_dim.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                            ex,
                        )
                if key.lower() == "iprn":
                    print_format = value
                if key.lower() == "binary":
                    binary = bool(value)
                if key.lower() == "data":
                    data = value
            if "filename" in arr_line:
                data_file = arr_line["filename"]

        if data_file is None:
            message = (
                "Data array {} contains an OPEN/CLOSE without a "
                "fname (file name) specified"
                ".".format(data_dim.structure.name)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "processing open/close line",
                data_dim.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

        if store:
            # store external info
            self.store_external(
                data_file,
                layer,
                [multiplier],
                print_format,
                binary=binary,
                data=data,
            )

        #  add to active list of external files
        model_name = data_dim.package_dim.model_dim[0].model_name
        self._simulation_data.mfpath.add_ext_file(data_file, model_name)

        return multiplier, print_format, binary, data_file

    @staticmethod
    def _tupleize_data(data):
        for index, data_line in enumerate(data):
            if not isinstance(data_line, tuple):
                if isinstance(data_line, list):
                    data[index] = tuple(data_line)
                else:
                    data[index] = (data_line,)

    def _verify_list(self, data):
        if data is not None:
            model_grid = None
            cellid_size = None
            for data_line in data:
                data_line_len = len(data_line)
                for index in range(
                    0, min(data_line_len, len(self._recarray_type_list))
                ):
                    datadim = self.data_dimensions
                    if (
                        self._recarray_type_list[index][0] == "cellid"
                        and datadim.get_model_dim(None).model_name is not None
                        and data_line[index] is not None
                    ):
                        # this is a cell id.  verify that it contains the
                        # correct number of integers
                        if cellid_size is None:
                            model_grid = datadim.get_model_grid()
                            cellid_size = (
                                model_grid.get_num_spatial_coordinates()
                            )
                        if (
                            cellid_size != 1
                            and len(data_line[index]) != cellid_size
                            and isinstance(data_line[index], int)
                        ):
                            message = (
                                'Cellid "{}" contains {} integer(s). '
                                "Expected a cellid containing {} "
                                "integer(s) for grid type"
                                " {}.".format(
                                    data_line[index],
                                    len(data_line[index]),
                                    cellid_size,
                                    str(model_grid.grid_type()),
                                )
                            )
                            type_, value_, traceback_ = sys.exc_info()
                            raise MFDataException(
                                self.data_dimensions.structure.get_model(),
                                self.data_dimensions.structure.get_package(),
                                self.data_dimensions.structure.path,
                                "verifying cellid",
                                self.data_dimensions.structure.name,
                                inspect.stack()[0][3],
                                type_,
                                value_,
                                traceback_,
                                message,
                                self._simulation_data.debug,
                            )

    def _add_placeholders(self, data):
        for idx, data_line in enumerate(data):
            data_line_len = len(data_line)
            if data_line_len < len(self._recarray_type_list):
                for index in range(
                    data_line_len, len(self._recarray_type_list)
                ):
                    if self._recarray_type_list[index][1] == int:
                        self._recarray_type_list[index] = (
                            self._recarray_type_list[index][0],
                            object,
                        )
                        data_line += (None,)
                    elif self._recarray_type_list[index][1] == float:
                        data_line += (np.nan,)
                    else:
                        data_line += (None,)
                data[idx] = data_line
            elif data_line_len > len(self._recarray_type_list):
                for index in range(
                    len(self._recarray_type_list), data_line_len
                ):
                    if data_line[-1] is None:
                        dl = list(data_line)
                        del dl[-1]
                        data_line = tuple(dl)
                data[idx] = data_line

    def _duplicate_last_item(self):
        last_item = self._recarray_type_list[-1]
        arr_item_name = last_item[0].split("_")
        if DatumUtil.is_int(arr_item_name[-1]):
            new_item_num = int(arr_item_name[-1]) + 1
            new_item_name = "_".join(arr_item_name[0:-1])
            new_item_name = "{}_{}".format(new_item_name, new_item_num)
        else:
            new_item_name = "{}_1".format(last_item[0])
        self._recarray_type_list.append((new_item_name, last_item[1]))

    def _build_full_data(self, apply_multiplier=False):
        if self.data_structure_type == DataStructureType.scalar:
            return self.layer_storage.first_item().internal_data
        dimensions = self.get_data_dimensions(None)
        if dimensions[0] < 0:
            return None
        all_none = True
        np_data_type = self.data_dimensions.structure.get_datum_type()
        full_data = np.full(
            dimensions,
            np.nan,
            self.data_dimensions.structure.get_datum_type(True),
        )
        is_aux = self.data_dimensions.structure.name == "aux"
        if is_aux:
            aux_data = []
        if not self.layered:
            layers_to_process = [0]
        else:
            layers_to_process = self.layer_storage.indexes()
        for layer in layers_to_process:
            if (
                self.layer_storage[layer].factor is not None
                and apply_multiplier
            ):
                mult = self.layer_storage[layer].factor
            elif self._data_type == DatumType.integer:
                mult = 1
            else:
                mult = 1.0

            if (
                self.layer_storage[layer].data_storage_type
                == DataStorageType.internal_array
            ):
                if (
                    self.layer_storage[layer].internal_data is None
                    or len(self.layer_storage[layer].internal_data) > 0
                    and self.layer_storage[layer].internal_data[0] is None
                ):
                    if is_aux:
                        full_data = None
                    else:
                        return None
                elif (
                    self.layer_storage.get_total_size() == 1
                    or not self.layered
                    or not self._has_layer_dim()
                ):
                    full_data = self.layer_storage[layer].internal_data * mult
                else:
                    full_data[layer] = (
                        self.layer_storage[layer].internal_data * mult
                    )
            elif (
                self.layer_storage[layer].data_storage_type
                == DataStorageType.internal_constant
            ):
                if (
                    self.layer_storage.get_total_size() == 1
                    or not self.layered
                    or not self._has_layer_dim()
                ):
                    full_data = self._fill_const_layer(layer) * mult
                else:
                    full_data[layer] = self._fill_const_layer(layer) * mult
            else:
                file_access = MFFileAccessArray(
                    self.data_dimensions.structure,
                    self.data_dimensions,
                    self._simulation_data,
                    self._data_path,
                    self._stress_period,
                )
                model_name = self.data_dimensions.package_dim.model_dim[
                    0
                ].model_name
                read_file = self._simulation_data.mfpath.resolve_path(
                    self.layer_storage[layer].fname, ""
                )

                if self.layer_storage[layer].binary:
                    data_out = (
                        file_access.read_binary_data_from_file(
                            read_file,
                            self.get_data_dimensions(layer),
                            self.get_data_size(layer),
                            self._data_type,
                            self._model_or_sim.modeldiscrit,
                            False,
                        )[0]
                        * mult
                    )
                else:
                    data_out = (
                        file_access.read_text_data_from_file(
                            self.get_data_size(layer),
                            np_data_type,
                            self.get_data_dimensions(layer),
                            layer,
                            read_file,
                        )[0]
                        * mult
                    )
                if (
                    self.layer_storage.get_total_size() == 1
                    or not self.layered
                ):
                    full_data = data_out
                else:
                    full_data[layer] = data_out
            if is_aux:
                if full_data is not None:
                    all_none = False
                aux_data.append(full_data)
                full_data = np.full(
                    dimensions,
                    np.nan,
                    self.data_dimensions.structure.get_datum_type(True),
                )
        if is_aux:
            if all_none:
                return None
            else:
                return np.stack(aux_data, axis=0)
        else:
            return full_data

    def _resolve_layer(self, layer):
        if layer is None:
            return self.layer_storage.first_index()
        else:
            return layer

    def _to_ndarray(self, data, layer):
        data_dimensions = self.get_data_dimensions(layer)
        data_iter = MultiListIter(data)
        return self._fill_dimensions(data_iter, data_dimensions)

    def _fill_const_layer(self, layer):
        data_dimensions = self.get_data_dimensions(layer)
        if layer is None:
            ls = self.layer_storage.first_item()
        else:
            ls = self.layer_storage[layer]
        if data_dimensions[0] < 0:
            return ls.data_const_value
        else:
            data_type = self.data_dimensions.structure.get_datum_type(
                numpy_type=True
            )
            return np.full(data_dimensions, ls.data_const_value[0], data_type)

    def _is_type(self, data_item, data_type):
        if data_type == DatumType.string or data_type == DatumType.keyword:
            return True
        elif data_type == DatumType.integer:
            return DatumUtil.is_int(data_item)
        elif data_type == DatumType.double_precision:
            return DatumUtil.is_float(data_item)
        elif data_type == DatumType.keystring:
            # TODO: support keystring type
            if (
                self._simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print("Keystring type currently not supported.")
            return True
        else:
            if (
                self._simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "{} type checking currently not supported".format(
                        data_type
                    )
                )
            return True

    def _fill_dimensions(self, data_iter, dimensions):
        if self.data_structure_type == DataStructureType.ndarray:
            np_dtype = MFFileAccess.datum_to_numpy_type(self._data_type)[0]
            # initialize array
            data_array = np.ndarray(shape=dimensions, dtype=np_dtype)
            # fill array
            for index in ArrayIndexIter(dimensions):
                data_array.itemset(index, data_iter.__next__())
            return data_array
        elif self.data_structure_type == DataStructureType.scalar:
            return data_iter.__next__()
        else:
            data_array = None
            data_line = ()
            # fill array
            array_index_iter = ArrayIndexIter(dimensions)
            current_col = 0
            for index in array_index_iter:
                data_line += (index,)
                if current_col == dimensions[1] - 1:
                    try:
                        if data_array is None:
                            data_array = np.rec.array(
                                data_line, self._recarray_type_list
                            )
                        else:
                            rec_array = np.rec.array(
                                data_line, self._recarray_type_list
                            )
                            data_array = np.hstack((data_array, rec_array))
                    except:
                        message = (
                            "An error occurred when storing data "
                            '"{}" in a recarray. Data line being '
                            "stored: {}".format(
                                self.data_dimensions.structure.name, data_line
                            )
                        )

                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.data_dimensions.structure.get_model(),
                            self.data_dimensions.structure.get_package(),
                            self.data_dimensions.structure.path,
                            "processing open/close line",
                            dimensions.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
                    current_col = 0
                    data_line = ()
                data_array[index] = data_iter.next()
            return data_array

    def set_tas(self, tas_name, tas_label, current_key, check_name=True):
        if check_name:
            package_dim = self.data_dimensions.package_dim
            tas_names = package_dim.get_tasnames()
            if (
                tas_name.lower() not in tas_names
                and self._simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "WARNING: Time array series name {} not found in any "
                    "time series file".format(tas_name)
                )
        # this is a time series array with a valid tas variable
        self.data_structure_type = DataStructureType.scalar
        try:
            self.set_data(
                "{} {}".format(tas_label, tas_name), 0, key=current_key
            )
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            structure = self.data_dimensions.structure
            raise MFDataException(
                structure.get_model(),
                structure.get_package(),
                structure.path,
                "storing data",
                structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def resolve_data_size(self, index):
        # Resolves the size of a given data element based on the names in the
        # existing rec_array. Assumes repeating data element names follow the
        #  format <data_element_name>_X
        if self.data_structure_type != DataStructureType.recarray:
            message = (
                "Data structure type is {}. Data structure type must "
                "be recarray.".format(self.data_structure_type)
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "resolving data size",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

        if len(self.layer_storage.first_item().internal_data[0]) <= index:
            return 0
        label = self.layer_storage.first_item().internal_data.dtype.names[
            index
        ]
        label_list = label.split("_")
        if len(label_list) == 1:
            return 1
        internal_data = self.layer_storage.first_item().internal_data
        for forward_index in range(index + 1, len(internal_data.dtype.names)):
            forward_label = internal_data.dtype.names[forward_index]
            forward_label_list = forward_label.split("_")
            if forward_label_list[0] != label_list[0]:
                return forward_index - index
        return len(internal_data.dtype.names) - index

    def build_type_list(
        self,
        data_set=None,
        data=None,
        resolve_data_shape=True,
        key=None,
        nseg=None,
        cellid_expanded=False,
        min_size=False,
        overwrite_existing_type_list=True,
    ):
        if not overwrite_existing_type_list:
            existing_type_list = self._recarray_type_list
            existing_type_list_ex = self._recarray_type_list_ex
        if data_set is None:
            self.jagged_record = False
            self._recarray_type_list = []
            self._recarray_type_list_ex = []
            self.recarray_cellid_list = []
            self.recarray_cellid_list_ex = []
            data_set = self.data_dimensions.structure
        initial_keyword = True
        package_dim = self.data_dimensions.package_dim
        for data_item, index in zip(
            data_set.data_item_structures,
            range(0, len(data_set.data_item_structures)),
        ):
            # handle optional mnames
            if (
                not data_item.optional
                or len(data_item.name) < 5
                or data_item.name.lower()[0:5] != "mname"
                or not self.in_model
            ):
                overrides = self._data_type_overrides
                if len(self._recarray_type_list) in overrides:
                    data_type = overrides[len(self._recarray_type_list)]
                elif isinstance(data_item, MFDataItemStructure):
                    data_type = data_item.get_rec_type()
                else:
                    data_type = None
                if data_item.name.lower() == "aux" and resolve_data_shape:
                    aux_var_names = package_dim.get_aux_variables()
                    if aux_var_names is not None:
                        for aux_var_name in aux_var_names[0]:
                            if aux_var_name.lower() != "auxiliary":
                                self._append_type_lists(
                                    aux_var_name, data_type, False
                                )

                elif data_item.type == DatumType.record:
                    # record within a record, recurse
                    self.build_type_list(data_item, True, data)
                elif data_item.type == DatumType.keystring:
                    self.jagged_record = True
                    self._append_type_lists(
                        data_item.name, data_type, data_item.is_cellid
                    )
                    # add potential data after keystring to type list
                    ks_data_item = deepcopy(data_item)
                    ks_data_item.type = DatumType.string
                    ks_data_item.name = "{}_data".format(ks_data_item.name)
                    ks_rec_type = ks_data_item.get_rec_type()
                    if not min_size:
                        self._append_type_lists(
                            ks_data_item.name,
                            ks_rec_type,
                            ks_data_item.is_cellid,
                        )
                    if (
                        index == len(data_set.data_item_structures) - 1
                        and data is not None
                    ):
                        idx = 1
                        (
                            line_max_size,
                            line_min_size,
                        ) = self._get_max_min_data_line_size(data)
                        if min_size:
                            line_size = line_min_size
                        else:
                            line_size = line_max_size
                        type_list = self.resolve_typelist(data)
                        while len(type_list) < line_size:
                            # keystrings at the end of a line can contain
                            # items of variable length. assume everything at
                            # the end of the data line is related to the last
                            # keystring
                            name = "{}_{}".format(ks_data_item.name, idx)
                            self._append_type_lists(
                                name, ks_rec_type, ks_data_item.is_cellid
                            )
                            idx += 1

                elif (
                    data_item.name != "boundname"
                    or self.data_dimensions.package_dim.boundnames()
                ):
                    # don't include initial keywords
                    if (
                        data_item.type != DatumType.keyword
                        or data_set.block_variable
                    ):
                        initial_keyword = False
                        shape_rule = None
                        if data_item.tagged:
                            if (
                                data_item.type != DatumType.string
                                and data_item.type != DatumType.keyword
                            ):
                                name = "{}_label".format(data_item.name)
                                self._append_type_lists(
                                    name, object, data_item.is_cellid
                                )
                        if (
                            nseg is not None
                            and len(data_item.shape) > 0
                            and isinstance(data_item.shape[0], str)
                            and data_item.shape[0][0:4] == "nseg"
                        ):
                            # nseg explicitly specified.  resolve any formula
                            # nseg is in
                            model_dim = self.data_dimensions.get_model_dim(
                                None
                            )
                            exp_array = model_dim.build_shape_expression(
                                data_item.shape
                            )
                            if (
                                isinstance(exp_array, list)
                                and len(exp_array) == 1
                            ):
                                exp = exp_array[0]
                                resolved_shape = [
                                    model_dim.resolve_exp(exp, nseg)
                                ]
                            else:
                                resolved_shape = [1]
                        else:
                            if resolve_data_shape:
                                data_dim = self.data_dimensions
                                (
                                    resolved_shape,
                                    shape_rule,
                                ) = data_dim.get_data_shape(
                                    data_item,
                                    data_set,
                                    data,
                                    repeating_key=key,
                                    min_size=min_size,
                                )
                            else:
                                resolved_shape = [1]
                        if (
                            not resolved_shape
                            or len(resolved_shape) == 0
                            or resolved_shape[0] == -1
                        ):
                            # could not resolve shape
                            resolved_shape = [1]
                        elif (
                            resolved_shape[0] == -9999
                            or shape_rule is not None
                        ):
                            if data is not None and not min_size:
                                # shape is an indeterminate 1-d array and
                                # should consume the remainder of the data
                                max_s = PyListUtil.max_multi_dim_list_size(
                                    data
                                )
                                resolved_shape[0] = max_s - len(
                                    self._recarray_type_list
                                )
                            else:
                                # shape is indeterminate 1-d array and either
                                # no data provided to resolve or request is
                                # for minimum data size
                                resolved_shape[0] = 1
                                if not min_size:
                                    self.jagged_record = True
                        if data_item.is_cellid:
                            if (
                                data_item.shape is not None
                                and len(data_item.shape) > 0
                                and data_item.shape[0] == "ncelldim"
                            ):
                                # A cellid is a single entry (tuple) in the
                                # recarray.  Adjust dimensions accordingly.
                                data_dim = self.data_dimensions
                                grid = data_dim.get_model_grid()
                                size = grid.get_num_spatial_coordinates()
                                data_item.remove_cellid(resolved_shape, size)
                        if not data_item.optional or not min_size:
                            for index in range(0, resolved_shape[0]):
                                if resolved_shape[0] > 1:
                                    name = "{}_{}".format(
                                        data_item.name, index
                                    )
                                else:
                                    name = data_item.name
                                self._append_type_lists(
                                    name, data_type, data_item.is_cellid
                                )
        if cellid_expanded:
            new_type_list_ex = self._recarray_type_list_ex
            if not overwrite_existing_type_list:
                self._recarray_type_list = existing_type_list
                self._recarray_type_list_ex = existing_type_list_ex
            return new_type_list_ex
        else:
            new_type_list = self._recarray_type_list
            if not overwrite_existing_type_list:
                self._recarray_type_list = existing_type_list
                self._recarray_type_list_ex = existing_type_list_ex
            return new_type_list

    def get_default_mult(self):
        if self._data_type == DatumType.integer:
            return 1
        else:
            return 1.0

    def _append_type_lists(self, name, data_type, iscellid):
        # add entry(s) to type lists
        self._recarray_type_list.append((name, data_type))
        self.recarray_cellid_list.append(iscellid)
        if iscellid and self._model_or_sim.model_type is not None:
            # write each part of the cellid out as a separate entry
            # to _recarray_list_list_ex
            model_grid = self.data_dimensions.get_model_grid()
            cellid_size = model_grid.get_num_spatial_coordinates()
            # determine header for different grid types
            if cellid_size == 1:
                self._do_ex_list_append(name, int, iscellid)
            elif cellid_size == 2:
                self._do_ex_list_append("layer", int, iscellid)
                self._do_ex_list_append("cell2d_num", int, iscellid)
            else:
                self._do_ex_list_append("layer", int, iscellid)
                self._do_ex_list_append("row", int, iscellid)
                self._do_ex_list_append("column", int, iscellid)
        else:
            self._do_ex_list_append(name, data_type, iscellid)
        return iscellid

    def _do_ex_list_append(self, name, data_type, iscellid):
        self._recarray_type_list_ex.append((name, data_type))
        self.recarray_cellid_list_ex.append(iscellid)

    @staticmethod
    def _calc_data_size(data, count_to=None, current_length=None):
        if current_length is None:
            current_length = [0]
        if isinstance(data, np.ndarray):
            current_length[0] += data.size
            return data.size
        if isinstance(data, str) or isinstance(data, dict):
            return 1
        try:
            for data_item in data:
                if hasattr(data_item, "__len__"):
                    DataStorage._calc_data_size(
                        data_item, count_to, current_length
                    )
                else:
                    current_length[0] += 1
                if count_to is not None and current_length[0] >= count_to:
                    return current_length[0]
        except (ValueError, IndexError, TypeError):
            return 1
        return current_length[0]

    @staticmethod
    def _get_max_min_data_line_size(data):
        max_size = 0
        min_size = sys.maxsize
        if data is not None:
            for value in data:
                if len(value) > max_size:
                    max_size = len(value)
                if len(value) < min_size:
                    min_size = len(value)
        if min_size == sys.maxsize:
            min_size = 0
        return max_size, min_size

    def get_data_dimensions(self, layer):
        data_dimensions = self.data_dimensions.get_data_shape()[0]
        if (
            layer is not None
            and self.layer_storage.get_total_size() > 1
            and self._has_layer_dim()
        ):
            # remove all "layer" dimensions from the list
            layer_dims = self.data_dimensions.structure.data_item_structures[
                0
            ].layer_dims
            data_dimensions = data_dimensions[len(layer_dims) :]
        return data_dimensions

    def _has_layer_dim(self):
        return (
            "nlay" in self.data_dimensions.structure.shape
            or "nodes" in self.data_dimensions.structure.shape
        )

    def _store_prep(self, layer, multiplier):
        if not (layer is None or self.layer_storage.in_shape(layer)):
            message = "Layer {} is not a valid layer.".format(layer)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "storing data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )
        if layer is None:
            # layer is none means the data provided is for all layers or this
            # is not layered data
            layer = (0,)
            self.layer_storage.list_shape = (1,)
            self.layer_storage.multi_dim_list = [
                self.layer_storage.first_item()
            ]
        mult_ml = MultiList(multiplier)
        if not mult_ml.in_shape(layer):
            if multiplier[0] is None:
                multiplier = self.get_default_mult()
            else:
                multiplier = multiplier[0]
        else:
            if mult_ml.first_item() is None:
                multiplier = self.get_default_mult()
            else:
                multiplier = mult_ml.first_item()

        return layer, multiplier

    def get_data_size(self, layer):
        dimensions = self.get_data_dimensions(layer)
        data_size = 1
        for dimension in dimensions:
            data_size = data_size * dimension
        return data_size
