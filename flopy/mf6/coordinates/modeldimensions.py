"""
modeldimensions module.  Contains the model dimension information


"""

from .simulationtime import SimulationTime
from .modelgrid import UnstructuredModelGrid, ModelGrid
from ..mfbase import StructException, FlopyException, VerbosityLevel
from ..data.mfstructure import DatumType
from ..utils.mfenums import DiscretizationType
from ...utils.datautil import DatumUtil, NameIter


class DataDimensions(object):
    """
    Resolves dimension information for model data using information contained
    in the model files

    Parameters
    ----------
    package_dim : PackageDimensions
        PackageDimension object for the package that the data is contained in
    structure : MFDataStructure
        MFDataStructure object of data whose dimensions need to be resolved
        (optional)

    Methods
    ----------
    get_model_grid : ()
        returns a model grid based on the current simulation data

    def get_data_shape(data_item : MFDataItemStructure, data_set_struct :
      MFDataStructure, data_item_num : int):
        returns the shape of modflow data structure.  returns shape of entire
        data structure if no data item is specified, otherwise returns shape of
        individual data time.  user data and the dictionary path to the data
        can be passed in "data" to help resolve the data shape
    model_subspace_size : (subspace_string : string)
        returns the size of the model subspace specified in subspace_string

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, package_dim, structure):
        self.package_dim = package_dim
        self.structure = structure
        self.model_grid = None
        self.locked = False

    def lock(self):
        self.model_grid = None
        self.locked = True
        self.package_dim.lock()

    def unlock(self):
        self.locked = False
        self.package_dim.unlock()

    def get_model_grid(self, data_item_num=None):
        if self.locked:
            if self.model_grid is None:
                self.model_grid = self.get_model_dim(
                    data_item_num
                ).get_model_grid()
            return self.model_grid
        else:
            return self.get_model_dim(data_item_num).get_model_grid()

    def get_data_shape(
        self,
        data_item=None,
        data_set_struct=None,
        data=None,
        data_item_num=None,
        repeating_key=None,
    ):
        return self.get_model_dim(data_item_num).get_data_shape(
            self.structure,
            data_item,
            data_set_struct,
            data,
            self.package_dim.package_path,
            repeating_key=repeating_key,
        )

    def model_subspace_size(self, subspace_string="", data_item_num=None):
        return self.get_model_dim(data_item_num).model_subspace_size(
            subspace_string
        )

    def get_model_dim(self, data_item_num):
        if (
            self.package_dim.model_dim is None
            or data_item_num is None
            or len(self.package_dim.model_dim) == 1
        ):
            return self.package_dim.model_dim[0]
        else:
            if not (len(self.structure.data_item_structures) > data_item_num):
                raise FlopyException(
                    'Data item index "{}" requested which '
                    "is greater than the maximum index of"
                    "{}.".format(
                        data_item_num,
                        len(self.structure.data_item_structures) - 1,
                    )
                )
            model_num = self.structure.data_item_structures[data_item_num][-1]
            if DatumUtil.is_int(model_num):
                return self.package_dim.model_dim[int(model_num)]


class PackageDimensions(object):
    """
    Resolves dimension information for common parts of a package

    Parameters
    ----------
    model_dim : ModelDimensions
        ModelDimensions object for the model that the package is contained in
    structure : MFPackageStructure
        MFPackageStructure object of package
    package_path : tuple
        Tuple representing the path to this package

    Methods
    ----------
    get_aux_variables : (model_num=0)
        returns the package's aux variables
    boundnames : (model_num=0)
        returns true of the boundnames option is in the package
    get_tasnames : (model_num=0)
        returns a dictionary of all the tas names used in a tas file
    get_tsnames : (model_num=0)
        returns a dictionary of all the ts names used in a ts file

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, model_dim, structure, package_path):
        self.model_dim = model_dim
        self.package_struct = structure
        self.package_path = package_path
        self.locked = False
        self.ts_names_dict = {}
        self.tas_names_dict = {}
        self.aux_variables = {}
        self.boundnames_dict = {}

    def lock(self):
        self.locked = True
        for model_dim in self.model_dim:
            model_dim.lock()

    def unlock(self):
        self.locked = False
        self.ts_names_dict = {}
        self.tas_names_dict = {}
        self.aux_variables = {}
        self.boundnames_dict = {}
        for model_dim in self.model_dim:
            model_dim.unlock()

    def get_aux_variables(self, model_num=0):
        if self.locked and model_num in self.aux_variables:
            return self.aux_variables[model_num]
        aux_path = self.package_path + ("options", "auxiliary")
        if aux_path in self.model_dim[model_num].simulation_data.mfdata:
            ret_val = (
                self.model_dim[model_num]
                .simulation_data.mfdata[aux_path]
                .get_data()
            )
        else:
            ret_val = None
        if self.locked:
            self.aux_variables[model_num] = ret_val
        return ret_val

    def boundnames(self, model_num=0):
        if self.locked and model_num in self.boundnames_dict:
            return self.boundnames_dict[model_num]
        ret_val = False
        bound_path = self.package_path + ("options", "boundnames")
        if bound_path in self.model_dim[model_num].simulation_data.mfdata:
            if (
                self.model_dim[model_num]
                .simulation_data.mfdata[bound_path]
                .get_data()
                is not None
            ):
                ret_val = True
        if self.locked:
            self.boundnames_dict[model_num] = ret_val
        return ret_val

    def get_tasnames(self, model_num=0):
        if self.locked and model_num in self.tas_names_dict:
            return self.tas_names_dict[model_num]
        names_dict = {}
        tas_record_path = self.package_path + ("options", "tas_filerecord")
        if tas_record_path in self.model_dim[model_num].simulation_data.mfdata:
            tas_record_data = (
                self.model_dim[model_num]
                .simulation_data.mfdata[tas_record_path]
                .get_data()
            )
            if tas_record_data is not None:
                name_iter = NameIter("tas")
                for tas_name in name_iter:
                    tas_names_path = self.package_path + (
                        tas_name,
                        "attributes",
                        "time_series_namerecord",
                    )
                    if (
                        tas_names_path
                        in self.model_dim[model_num].simulation_data.mfdata
                    ):
                        tas_names_data = (
                            self.model_dim[model_num]
                            .simulation_data.mfdata[tas_names_path]
                            .get_data()
                        )
                        if tas_names_data is not None:
                            names_dict[tas_names_data[0][0]] = 0
                    else:
                        break
        if self.locked:
            self.tas_names_dict[model_num] = names_dict
        return names_dict

    def get_tsnames(self, model_num=0):
        if self.locked and model_num in self.ts_names_dict:
            return self.ts_names_dict[model_num]
        names_dict = {}
        ts_record_path = self.package_path + ("options", "ts_filerecord")
        if ts_record_path in self.model_dim[model_num].simulation_data.mfdata:
            ts_record_data = (
                self.model_dim[model_num]
                .simulation_data.mfdata[ts_record_path]
                .get_data()
            )
            if ts_record_data is not None:
                name_iter = NameIter("ts")
                for ts_name in name_iter:
                    ts_names_path = self.package_path + (
                        ts_name,
                        "attributes",
                        "time_series_namerecord",
                    )
                    if (
                        ts_names_path
                        in self.model_dim[model_num].simulation_data.mfdata
                    ):
                        ts_names_data = (
                            self.model_dim[model_num]
                            .simulation_data.mfdata[ts_names_path]
                            .get_data()
                        )
                        if ts_names_data is not None:
                            for name in ts_names_data[0]:
                                names_dict[name] = 0
                    else:
                        break
        if self.locked:
            self.ts_names_dict[model_num] = names_dict
        return names_dict


class ModelDimensions(object):
    """
    Contains model dimension information and helper methods

    Parameters
    ----------
    model_name : string
        name of the model
    simulation_data : MFSimulationData
        contains all simulation related data
    structure : MFDataStructure
        MFDataStructure object of data whose dimensions need to be resolved
        (optional)

    Attributes
    ----------
    simulation_time : SimulationTime
        object containing simulation time information

    Methods
    ----------
    get_model_grid : ()
        returns a model grid based on the current simulation data

    def get_data_shape(structure : MFDataStructure, data_item :
                       MFDataItemStructure, data_set_struct : MFDataStructure,
                       data : list, path : tuple, deconstruct_axis : bool):
        returns the shape of modflow data structure.  returns shape of entire
        data structure if no data item is specified, otherwise returns shape of
        individual data time.  user data and the dictionary path to the data
        can be passed in "data" to help resolve the data shape.  if
        deconstruct_axis is True any spatial axis will be automatically
        deconstructed into its component parts (model grid will be
        deconstructed into layer/row/col)
    data_reshape : ()
        reshapes jagged model data
    model_subspace_size : (subspace_string : string)
        returns the size of the model subspace specified in subspace_string

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, model_name, simulation_data):
        self.model_name = model_name
        self.simulation_data = simulation_data
        self._model_grid = None
        self.simulation_time = SimulationTime(simulation_data)
        self.locked = False
        self.stored_shapes = {}

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False
        self.stored_shapes = {}

    # returns model grid
    def get_model_grid(self):
        if not self.locked or self._model_grid is None:
            grid_type = ModelGrid.get_grid_type(
                self.simulation_data, self.model_name
            )
            if not self._model_grid:
                self._create_model_grid(grid_type)
            else:
                # if existing model grid is consistent with model data
                if not self._model_grid.grid_type_consistent():
                    # create new model grid and return
                    self._create_model_grid(grid_type)
                    print(
                        "WARNING: Model grid type has changed.  get_model_grid() "
                        "is returning a new model grid object of the appropriate "
                        "type.  References to the old model grid object are "
                        "invalid."
                    )
        self._model_grid.freeze_grid = True
        return self._model_grid

    def _create_model_grid(self, grid_type):
        if grid_type == DiscretizationType.DIS:
            self._model_grid = ModelGrid(
                self.model_name, self.simulation_data, DiscretizationType.DIS
            )
        elif grid_type == DiscretizationType.DISV:
            self._model_grid = ModelGrid(
                self.model_name, self.simulation_data, DiscretizationType.DISV
            )
        elif grid_type == DiscretizationType.DISU:
            self._model_grid = UnstructuredModelGrid(
                self.model_name, self.simulation_data
            )
        elif grid_type == DiscretizationType.DISL:
            self._model_grid = ModelGrid(
                self.model_name, self.simulation_data, DiscretizationType.DISL
            )
        else:
            self._model_grid = ModelGrid(
                self.model_name,
                self.simulation_data,
                DiscretizationType.UNDEFINED,
            )

    # Returns a shape for a given set of axes
    def get_data_shape(
        self,
        structure,
        data_item=None,
        data_set_struct=None,
        data=None,
        path=None,
        deconstruct_axis=True,
        repeating_key=None,
    ):
        if structure is None:
            raise FlopyException(
                "get_data_shape requires a valid structure " "object"
            )
        if self.locked:
            if data_item is not None and data_item.path in self.stored_shapes:
                return (
                    self.stored_shapes[data_item.path][0],
                    self.stored_shapes[data_item.path][1],
                )
            if structure.path in self.stored_shapes:
                return (
                    self.stored_shapes[structure.path][0],
                    self.stored_shapes[structure.path][1],
                )

        shape_dimensions = []
        shape_rule = None
        shape_consistent = True
        if data_item is None:
            if (
                structure.type == DatumType.recarray
                or structure.type == DatumType.record
            ):
                if structure.type == DatumType.record:
                    num_rows = 1
                else:
                    num_rows, consistent_shape = self._resolve_data_item_shape(
                        structure
                    )[0]
                    shape_consistent = shape_consistent and consistent_shape
                num_cols = 0
                for data_item_struct in structure.data_item_structures:
                    if data_item_struct.type != DatumType.keyword:
                        (
                            num,
                            shape_rule,
                            consistent_shape,
                        ) = self._resolve_data_item_shape(
                            data_item_struct,
                            path=path,
                            repeating_key=repeating_key,
                        )[
                            0
                        ]
                        num_cols = num_cols + num
                        shape_consistent = (
                            shape_consistent and consistent_shape
                        )
                shape_dimensions = [num_rows, num_cols]
            else:
                for data_item_struct in structure.data_item_structures:
                    if len(shape_dimensions) == 0:
                        (
                            shape_dimensions,
                            shape_rule,
                            consistent_shape,
                        ) = self._resolve_data_item_shape(
                            data_item_struct, repeating_key=repeating_key
                        )
                    else:

                        (
                            dim,
                            shape_rule,
                            consistent_shape,
                        ) = self._resolve_data_item_shape(
                            data_item_struct, repeating_key=repeating_key
                        )
                        shape_dimensions += dim
                    shape_consistent = shape_consistent and consistent_shape
            if self.locked and shape_consistent:
                self.stored_shapes[structure.path] = (
                    shape_dimensions,
                    shape_rule,
                )
        else:
            (
                shape_dimensions,
                shape_rule,
                consistent_shape,
            ) = self._resolve_data_item_shape(
                data_item,
                data_set_struct,
                data,
                path,
                deconstruct_axis,
                repeating_key=repeating_key,
            )
            if self.locked and consistent_shape:
                self.stored_shapes[data_item.path] = (
                    shape_dimensions,
                    shape_rule,
                )

        return shape_dimensions, shape_rule

    def _resolve_data_item_shape(
        self,
        data_item_struct,
        data_set_struct=None,
        data=None,
        path=None,
        deconstruct_axis=True,
        repeating_key=None,
    ):
        if isinstance(data, tuple):
            data = [data]
        shape_rule = None
        consistent_shape = True
        if path is None:
            parent_path = data_item_struct.path[:-2]
        else:
            parent_path = path
        shape_dimensions = []
        if len(data_item_struct.shape) > 0:
            shape = data_item_struct.shape[:]
            # resolve approximate shapes
            for index, shape_item in enumerate(shape):
                if shape_item[0] == "<" or shape_item[0] == ">":
                    shape_rule = shape_item[0]
                    shape[index] = shape_item[1:]

            if deconstruct_axis:
                shape = self.deconstruct_axis(shape)
            ordered_shape = self._order_shape(shape, data_item_struct)
            ordered_shape_expression = self.build_shape_expression(
                ordered_shape
            )
            for item in ordered_shape_expression:
                dim_size = self.dimension_size(item[0])
                if dim_size is not None:
                    if isinstance(dim_size, list):
                        shape_dimensions += dim_size
                    else:
                        shape_dimensions.append(
                            self.resolve_exp(item, dim_size)
                        )
                elif item[0].lower() == "nstp" and DatumUtil.is_int(
                    repeating_key
                ):
                    # repeating_key is a stress period.  get number of time
                    # steps for that stress period
                    shape_dimensions.append(
                        self.simulation_time.get_sp_time_steps(
                            int(repeating_key)
                        )
                    )
                else:
                    result = None
                    if data_set_struct is not None:
                        # try to resolve dimension in the existing data
                        # set first
                        result = self.resolve_exp(
                            item,
                            self._find_in_dataset(
                                data_set_struct, item[0], data
                            ),
                        )
                        if result:
                            consistent_shape = False
                    if result:
                        shape_dimensions.append(result)
                    else:
                        if (
                            item[0] == "any1d"
                            or item[0] == "naux"
                            or item[0] == "nconrno"
                            or item[0] == "unknown"
                            or item[0] == ":"
                        ):
                            consistent_shape = False
                            shape_dimensions.append(-9999)
                        elif item[0] == "any2d":
                            consistent_shape = False
                            shape_dimensions.append(-9999)
                            shape_dimensions.append(-9999)
                        elif DatumUtil.is_int(item[0]):
                            shape_dimensions.append(int(item[0]))
                        else:
                            # try to resolve dimension within the existing block
                            result = self.simulation_data.mfdata.find_in_path(
                                parent_path, item[0]
                            )
                            if result[0] is not None:
                                data = result[0].get_data()
                                if data is None:
                                    if (
                                        self.simulation_data.verbosity_level.value
                                        >= VerbosityLevel.normal.value
                                    ):
                                        print(
                                            "WARNING: Unable to resolve "
                                            "dimension of {} based on shape "
                                            '"{}".'.format(
                                                data_item_struct.path, item[0]
                                            )
                                        )
                                    shape_dimensions.append(-9999)
                                    consistent_shape = False
                                elif result[1] is not None:
                                    # if int return first value otherwise
                                    # return shape of data stored
                                    if DatumUtil.is_int(data[result[1]]):
                                        shape_dimensions.append(
                                            self.resolve_exp(item, int(data))
                                        )
                                    else:
                                        shape_dimensions.append(
                                            self.resolve_exp(
                                                item, len(data[result[1]])
                                            )
                                        )
                                else:
                                    if DatumUtil.is_int(data):
                                        shape_dimensions.append(
                                            self.resolve_exp(item, int(data))
                                        )
                                    else:
                                        shape_dimensions.append(
                                            self.resolve_exp(item, len(data))
                                        )
                            else:
                                if (
                                    self.simulation_data.verbosity_level.value
                                    >= VerbosityLevel.normal.value
                                ):
                                    print(
                                        "WARNING: Unable to resolve "
                                        "dimension of {} based on shape "
                                        '"{}".'.format(
                                            data_item_struct.path, item[0]
                                        )
                                    )
                                shape_dimensions.append(-9999)
                                consistent_shape = False
        else:
            if (
                data_item_struct.type == DatumType.recarray
                or data_item_struct.type == DatumType.record
            ):
                # shape is unknown
                shape_dimensions.append(-9999)
                consistent_shape = False
            else:
                # shape is assumed to be that of single entry
                shape_dimensions.append(1)
        return shape_dimensions, shape_rule, consistent_shape

    def resolve_exp(self, expression, value):
        if len(expression) == 3 and value is not None:
            if not DatumUtil.is_int(expression[1]):
                # try to resolve the 2nd term in the equation
                expression[1] = self.dimension_size(expression[1])
                if expression[1] is None:
                    except_str = (
                        'Expression "{}" contains an invalid '
                        "second term and can not be "
                        "resolved.".format(expression)
                    )
                    raise StructException(except_str, "")

            if expression[2] == "+":
                return value + int(expression[1])
            elif expression[2] == "-":
                return value - int(expression[1])
            elif expression[2] == "*":
                return value * int(expression[1])
            elif expression[2] == "/":
                return value / int(expression[1])
            else:
                except_str = (
                    'Expression "{}" contains an invalid operator '
                    "and can not be resolved.".format(expression)
                )
                raise StructException(except_str, "")
        else:
            return value

    @staticmethod
    def _find_in_dataset(data_set_struct, item, data):
        if data is not None:
            # find the current data item in data_set_struct
            for index, data_item in zip(
                range(0, len(data_set_struct.data_item_structures)),
                data_set_struct.data_item_structures,
            ):
                if (
                    data_item.name.lower() == item.lower()
                    and len(data[0]) > index
                ):
                    # always use the maximum value
                    max_val = 0
                    for data_line in data:
                        if data_line[index] > max_val:
                            max_val = data_line[index]
                    return max_val
        return None

    @staticmethod
    def build_shape_expression(shape_array):
        new_expression_array = []
        for entry in shape_array:
            entry_minus = entry.split("-")
            if len(entry_minus) > 1:
                entry_minus.append("-")
                new_expression_array.append(entry_minus)
            else:
                entry_plus = entry.split("+")
                if len(entry_plus) > 1:
                    entry_plus.append("+")
                    new_expression_array.append(entry_plus)
                else:
                    entry_mult = entry.split("*")
                    if len(entry_mult) > 1:
                        entry_mult.append("*")
                        new_expression_array.append(entry_mult)
                    else:
                        entry_div = entry.split("*")
                        if len(entry_div) > 1:
                            entry_div.append("/")
                            new_expression_array.append(entry_div)
                        else:
                            new_expression_array.append([entry])
        return new_expression_array

    def _order_shape(self, shape_array, data_item_struct):
        new_shape_array = []

        for entry in shape_array:
            if entry in data_item_struct.layer_dims:
                # "layer" dimensions get ordered first
                new_shape_array.append(entry)

        order = ["nlay", "nrow", "ncol"]
        for order_item in order:
            if order_item not in data_item_struct.layer_dims:
                for entry in shape_array:
                    if entry == order_item:
                        new_shape_array.append(entry)
        for entry in shape_array:
            if entry not in order and entry not in data_item_struct.layer_dims:
                new_shape_array.append(entry)
        return new_shape_array

    def model_subspace_size(self, subspace_string):
        axis_found = False
        subspace_size = 1
        for axis in subspace_string:
            dim_size = self.dimension_size(axis, False)
            if dim_size is not None:
                subspace_size = subspace_size * dim_size
                axis_found = True
        if axis_found:
            return subspace_size
        else:
            return -1

    def dimension_size(self, dimension_string, return_shape=True):
        if dimension_string == "nrow":
            return self.get_model_grid().num_rows()
        elif dimension_string == "ncol":
            return self.get_model_grid().num_columns()
        elif dimension_string == "nlay":
            return self.get_model_grid().num_layers()
        elif dimension_string == "ncpl":
            return self.get_model_grid().num_cells_per_layer()
        elif dimension_string == "nodes":
            if return_shape:
                return self.get_model_grid().get_model_dim()
            else:
                return self.get_model_grid().num_cells()
        elif dimension_string == "nja":
            return self.get_model_grid().num_connections()
        elif dimension_string == "ncelldim":
            return self.get_model_grid().get_num_spatial_coordinates()
        else:
            return None

    def deconstruct_axis(self, shape_array):
        deconstructed_shape_array = []
        for entry in shape_array:
            if entry == "ncpl":
                if self.get_model_grid().grid_type() == DiscretizationType.DIS:
                    deconstructed_shape_array.append("ncol")
                    deconstructed_shape_array.append("nrow")
                else:
                    deconstructed_shape_array.append(entry)
            elif entry == "nodes":
                if self.get_model_grid().grid_type() == DiscretizationType.DIS:
                    deconstructed_shape_array.append("ncol")
                    deconstructed_shape_array.append("nrow")
                    deconstructed_shape_array.append("nlay")
                elif (
                    self.get_model_grid().grid_type()
                    == DiscretizationType.DISV
                ):
                    deconstructed_shape_array.append("ncpl")
                    deconstructed_shape_array.append("nlay")
                else:
                    deconstructed_shape_array.append(entry)
            else:
                deconstructed_shape_array.append(entry)
        return deconstructed_shape_array
