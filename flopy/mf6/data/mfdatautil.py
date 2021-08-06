import sys, inspect
import numpy as np
from copy import deepcopy
from collections.abc import Iterable
from ..mfbase import MFDataException, FlopyException
from .mfstructure import DatumType
from ...utils.datautil import PyListUtil, DatumUtil
import struct


def iterable(obj):
    return isinstance(obj, Iterable)


def get_first_val(arr):
    while isinstance(arr, list) or isinstance(arr, np.ndarray):
        arr = arr[0]
    return arr


# convert_data(data, type) : type
#    converts data "data" to type "type" and returns the converted data
def convert_data(data, data_dimensions, data_type, data_item=None, sub_amt=1):
    if data_type == DatumType.double_precision:
        if data_item is not None and data_item.support_negative_index:
            val = int(PyListUtil.clean_numeric(data))
            if val == -1:
                return -0.0
            elif val == 1:
                return 0.0
            elif val < 0:
                val += 1
            else:
                val -= 1
            try:
                return float(val)
            except (ValueError, TypeError):
                message = (
                    'Data "{}" with value "{}" can '
                    "not be converted to float"
                    ".".format(data_dimensions.structure.name, data)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    data_dimensions.structure.get_model(),
                    data_dimensions.structure.get_package(),
                    data_dimensions.structure.path,
                    "converting data",
                    data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    False,
                )
        else:
            try:
                if isinstance(data, str):
                    # fix any scientific formatting that python can't handle
                    data = data.replace("d", "e")
                return float(data)
            except (ValueError, TypeError):
                try:
                    return float(PyListUtil.clean_numeric(data))
                except (ValueError, TypeError):
                    message = (
                        'Data "{}" with value "{}" can '
                        "not be converted to float"
                        ".".format(data_dimensions.structure.name, data)
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        data_dimensions.structure.get_model(),
                        data_dimensions.structure.get_package(),
                        data_dimensions.structure.path,
                        "converting data",
                        data_dimensions.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        message,
                        False,
                    )
    elif data_type == DatumType.integer:
        if data_item is not None and data_item.numeric_index:
            return int(PyListUtil.clean_numeric(data)) - sub_amt
        try:
            return int(data)
        except (ValueError, TypeError):
            try:
                return int(PyListUtil.clean_numeric(data))
            except (ValueError, TypeError):
                message = (
                    'Data "{}" with value "{}" can not be '
                    "converted to int"
                    ".".format(data_dimensions.structure.name, data)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    data_dimensions.structure.get_model(),
                    data_dimensions.structure.get_package(),
                    data_dimensions.structure.path,
                    "converting data",
                    data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    False,
                )
    elif data_type == DatumType.string and data is not None:
        if data_item is None or not data_item.preserve_case:
            # keep strings lower case
            return data.lower()
    return data


def to_string(
    val,
    data_type,
    sim_data,
    data_dim,
    is_cellid=False,
    possible_cellid=False,
    data_item=None,
    verify_data=True,
):
    if data_type == DatumType.double_precision:
        if data_item is not None and data_item.support_negative_index:
            if val > 0:
                return str(int(val + 1))
            elif val == 0.0:
                if (
                    struct.pack(">d", val)
                    == b"\x80\x00\x00\x00\x00\x00\x00\x00"
                ):
                    # value is negative zero
                    return str(int(val - 1))
                else:
                    # value is positive zero
                    return str(int(val + 1))
            else:
                return str(int(val - 1))
        else:
            try:
                abs_val = abs(val)
            except TypeError:
                return str(val)
            if (
                abs_val > sim_data._sci_note_upper_thres
                or abs_val < sim_data._sci_note_lower_thres
            ) and abs_val != 0:
                return sim_data.reg_format_str.format(val)
            else:
                return sim_data.sci_format_str.format(val)
    elif is_cellid or (possible_cellid and isinstance(val, tuple)):
        if DatumUtil.is_int(val):
            return str(val + 1)
        if len(val) == 4 and isinstance(val, str) and val.lower() == "none":
            # handle case that cellid is 'none'
            return val
        if (
            verify_data
            and is_cellid
            and data_dim.get_model_dim(None).model_name is not None
        ):
            model_grid = data_dim.get_model_grid()
            cellid_size = model_grid.get_num_spatial_coordinates()
            if len(val) != cellid_size:
                message = (
                    'Cellid "{}" contains {} integer(s). Expected a'
                    " cellid containing {} integer(s) for grid type"
                    " {}.".format(
                        val,
                        len(val),
                        cellid_size,
                        str(model_grid.grid_type()),
                    )
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    data_dim.structure.get_model(),
                    data_dim.structure.get_package(),
                    data_dim.structure.path,
                    "converting cellid to string",
                    data_dim.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    sim_data.debug,
                )

        string_val = []
        if isinstance(val, str):
            string_val.append(val)
        else:
            for item in val:
                string_val.append(str(item + 1))
        return " ".join(string_val)
    elif data_type == DatumType.integer:
        if data_item is not None and data_item.numeric_index:
            return str(int(val) + 1)
        return str(int(val))
    elif data_type == DatumType.string:
        try:
            arr_val = val.split()
        except AttributeError:
            return str(val)
        if len(arr_val) > 1:
            # quote any string with spaces
            string_val = "'{}'".format(val)
            if data_item is not None and data_item.ucase:
                return string_val.upper()
            else:
                return string_val
    if data_item is not None and data_item.ucase:
        return str(val).upper()
    else:
        return str(val)


class MFComment:
    """
    Represents a variable in a MF6 input file


    Parameters
    ----------
    comment : string or list
        comment to be displayed in output file
    path : string
        tuple representing location in the output file
    line_number : integer
        line number to display comment in output file

    Attributes
    ----------
    comment : string or list
        comment to be displayed in output file
    path : string
        tuple representing location in the output file
    line_number : integer
        line number to display comment in output file

    Methods
    -------
    write : (file)
        writes the comment to file
    add_text(additional_text)
        adds text to the comment
    get_file_entry(eoln_suffix=True)
        returns the comment text in the format to write to package files
    is_empty(include_whitespace=True)
        checks to see if comment is just an empty string ''.  if
        include_whitespace is set to false a string with only whitespace is
        considered empty
    is_comment(text, include_empty_line=False) : bool
        returns true if text is a comment.  an empty line is considered a
        comment if include_empty_line is true.

    See Also
    --------

    Notes
    -----

    Examples
    --------


    """

    def __init__(self, comment, path, sim_data, line_number=0):
        if not (
            isinstance(comment, str)
            or isinstance(comment, list)
            or comment is None
        ):
            raise FlopyException(
                'Comment "{}" not valid.  Comment must be '
                "of type str of list.".format(comment)
            )
        self.text = comment
        self.path = path
        self.line_number = line_number
        self.sim_data = sim_data

    """
    Add text to the comment string.

    Parameters
    ----------
    additional_text: string
        text to add
    """

    def add_text(self, additional_text, new_line=False):
        if additional_text:
            if isinstance(self.text, list):
                self.text.append(additional_text)
            elif new_line:
                self.text = "{}{}".format(self.text, additional_text)
            else:
                self.text = "{} {}".format(self.text, additional_text)

    """
    Get the comment text in the format to write to package files.

    Parameters
    ----------
    eoln_suffix: boolean
        have comment text end with end of line character
    Returns
    -------
    string : comment text
    """

    def get_file_entry(self, eoln_suffix=True):
        file_entry = ""
        if self.text and self.sim_data.comments_on:
            if not isinstance(self.text, str) and isinstance(self.text, list):
                file_entry = self._recursive_get(self.text)
            else:
                if self.text.strip():
                    file_entry = self.text
            if eoln_suffix:
                file_entry = "{}\n".format(file_entry)
        return file_entry

    def _recursive_get(self, base_list):
        file_entry = ""
        if base_list and self.sim_data.comments_on:
            for item in base_list:
                if not isinstance(item, str) and isinstance(item, list):
                    file_entry = "{}{}".format(
                        file_entry, self._recursive_get(item)
                    )
                else:
                    file_entry = "{} {}".format(file_entry, item)
        return file_entry

    """
    Write the comment text to a file.

    Parameters
    ----------
    fd : file
        file to write to
    eoln_suffix: boolean
        have comment text end with end of line character
    """

    def write(self, fd, eoln_suffix=True):
        if self.text and self.sim_data.comments_on:
            if not isinstance(self.text, str) and isinstance(self.text, list):
                self._recursive_write(fd, self.text)
            else:
                if self.text.strip():
                    fd.write(self.text)
            if eoln_suffix:
                fd.write("\n")

    """
    Check for comment text

    Parameters
    ----------
    include_whitespace : boolean
        include whitespace as text
    Returns
    -------
    boolean : True if comment text exists
    """

    def is_empty(self, include_whitespace=True):
        if include_whitespace:
            if self.text():
                return False
            return True
        else:
            if self.text.strip():
                return False
            return True

    """
    Check text to see if it is valid comment text

    Parameters
    ----------
    text : string
        potential comment text
    include_empty_line : boolean
        allow empty line to be valid
    Returns
    -------
    boolean : True if text is valid comment text
    """

    @staticmethod
    def is_comment(text, include_empty_line=False):
        if not text:
            return include_empty_line
        if text and isinstance(text, list):
            # look for comment mark in first item of list
            text_clean = text[0].strip()
        else:
            text_clean = text.strip()
        if include_empty_line and not text_clean:
            return True
        if text_clean and (
            text_clean[0] == "#"
            or text_clean[0] == "!"
            or text_clean[0] == "//"
        ):
            return True
        return False

    # recursively writes a nested list to a file
    def _recursive_write(self, fd, base_list):
        if base_list:
            for item in base_list:
                if not isinstance(item, str) and isinstance(item, list):
                    self._recursive_write(fd, item)
                else:
                    fd.write(" {}".format(item))


class TemplateGenerator:
    """
    Abstract base class for building a data template for different data types.
    This is a generic class that is initialized with a path that identifies
    the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)
    """

    def __init__(self, path):
        self.path = path

    def _get_data_dimensions(self, model):
        from ..data import mfstructure
        from ..coordinates import modeldimensions

        # get structure info
        sim_struct = mfstructure.MFStructure().sim_struct
        package_struct = sim_struct.get_data_structure(self.path[0:-2])

        # get dimension info
        data_struct = sim_struct.get_data_structure(self.path)
        package_dim = modeldimensions.PackageDimensions(
            [model.dimensions], package_struct, self.path[0:-1]
        )
        return (
            data_struct,
            modeldimensions.DataDimensions(package_dim, data_struct),
        )

    def build_type_header(self, ds_type, data=None):
        from ..data.mfdatastorage import DataStorageType

        if ds_type == DataStorageType.internal_array:
            if isinstance(self, ArrayTemplateGenerator):
                return {"factor": 1.0, "iprn": 1, "data": data}
            else:
                return None
        elif ds_type == DataStorageType.internal_constant:
            return data
        elif ds_type == DataStorageType.external_file:
            return {"filename": "", "factor": 1.0, "iprn": 1}
        return None


class ArrayTemplateGenerator(TemplateGenerator):
    """
    Class that builds a data template for MFArrays.  This is a generic class
    that is initialized with a path that identifies the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)

    Methods
    -------
    empty: (model: MFModel, layered: bool, data_storage_type_list: bool,
            default_value: int/float) : variable
        Builds a template for the data you need to specify for a specific data
        type (ie. "hk") in a specific model.  The data type and dimensions
        is determined by "path" during initialization of this class and the
        model is passed in to this method as the "model" parameter.  If the
        data is transient a dictionary containing a single stress period
        will be returned.  If "layered" is set to true, data will be returned
        as a list ndarrays, one for each layer.  data_storage_type_list is a
        list of DataStorageType, one type for each layer.  If "default_value"
        is specified the data template will be populated with that value,
        otherwise each ndarray in the data template will be populated with
        np.empty (0 or 0.0 if the DataStorageType is a constant).
    """

    def __init__(self, path):
        super().__init__(path)

    def empty(
        self,
        model=None,
        layered=False,
        data_storage_type_list=None,
        default_value=None,
    ):
        from ..data import mfdatastorage, mfstructure
        from ..data.mfdatastorage import DataStorageType, DataStructureType

        # get the expected dimensions of the data
        data_struct, data_dimensions = self._get_data_dimensions(model)
        datum_type = data_struct.get_datum_type()
        data_type = data_struct.get_datatype()
        # build a temporary data storage object
        data_storage = mfdatastorage.DataStorage(
            model.simulation_data,
            model,
            data_dimensions,
            None,
            DataStorageType.internal_array,
            DataStructureType.recarray,
            data_path=self.path,
        )
        dimension_list = data_storage.get_data_dimensions(None)

        # if layered data
        if layered and dimension_list[0] > 1:
            if (
                data_storage_type_list is not None
                and len(data_storage_type_list) != dimension_list[0]
            ):
                comment = (
                    "data_storage_type_list specified with the "
                    "wrong size.  Size {} but expected to be "
                    "the same as the number of layers, "
                    "{}.".format(
                        len(data_storage_type_list), dimension_list[0]
                    )
                )
                type_, value_, traceback_ = sys.exc_info()

                raise MFDataException(
                    data_struct.get_model(),
                    data_struct.get_package(),
                    data_struct.path,
                    "generating array template",
                    data_struct.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    model.simulation_data.debug,
                )
            # build each layer
            data_with_header = []
            for layer in range(0, dimension_list[0]):
                # determine storage type
                if data_storage_type_list is None:
                    data_storage_type = DataStorageType.internal_array
                else:
                    data_storage_type = data_storage_type_list[layer]
                # build data type header
                data_with_header.append(
                    self._build_layer(
                        datum_type,
                        data_storage_type,
                        default_value,
                        dimension_list,
                    )
                )
        else:
            if (
                data_storage_type_list is None
                or data_storage_type_list[0] == DataStorageType.internal_array
            ):
                data_storage_type = DataStorageType.internal_array
            else:
                data_storage_type = data_storage_type_list[0]
            # build data type header
            data_with_header = self._build_layer(
                datum_type,
                data_storage_type,
                default_value,
                dimension_list,
                True,
            )

        # if transient/multiple list
        if data_type == mfstructure.DataType.array_transient:
            # Return as dictionary
            return {0: data_with_header}
        else:
            return data_with_header

    def _build_layer(
        self,
        data_type,
        data_storage_type,
        default_value,
        dimension_list,
        all_layers=False,
    ):
        from ..data.mfdatastorage import DataStorageType

        # build data
        if data_storage_type == DataStorageType.internal_array:
            if default_value is None:
                if all_layers:
                    data = np.empty(dimension_list, data_type)
                else:
                    data = np.empty(dimension_list[1:], data_type)
            else:
                if all_layers:
                    data = np.full(dimension_list, default_value, data_type)
                else:
                    data = np.full(
                        dimension_list[1:], default_value, data_type
                    )
        elif data_storage_type == DataStorageType.internal_constant:
            if default_value is None:
                if data_type == np.int32:
                    data = 0
                else:
                    data = 0.0
            else:
                data = default_value
        else:
            data = None
        # build data type header
        return self.build_type_header(data_storage_type, data)


class ListTemplateGenerator(TemplateGenerator):
    """
    Class that builds a data template for MFLists.  This is a generic class
    that is initialized with a path that identifies the data to be built.

    Parameters
    ----------
    path : string
        tuple containing path of data is described in dfn files
        (<model>,<package>,<block>,<data name>)

    Methods
    -------
    empty: (maxbound: int, aux_vars: list, boundnames: bool, nseg: int) :
            dictionary
        Builds a template for the data you need to specify for a specific data
        type (ie. "stress_period_data") in a specific model.  The data type is
        determined by "path" during initialization of this class.  If the data
        is transient a dictionary containing a single stress period will be
        returned.  The number of entries in the recarray are determined by
        the "maxbound" parameter.  The "aux_vars" parameter is a list of aux
        var names to be used in this data list.  If boundnames is set to
        true and boundname field will be included in the recarray.  nseg is
        only used on list data that contains segments.  If timeseries is true,
        a template that is compatible with time series data is returned.
    """

    def __init__(self, path):
        super().__init__(path)

    def _build_template_data(self, type_list):
        template_data = []
        for type in type_list:
            if type[1] == int:
                template_data.append(0)
            elif type[1] == float:
                template_data.append(np.nan)
            else:
                template_data.append(None)
        return tuple(template_data)

    def dtype(
        self,
        model,
        aux_vars=None,
        boundnames=False,
        nseg=None,
        timeseries=False,
        cellid_expanded=False,
    ):
        from ..data import mfdatastorage

        # get data storage
        data_struct, data_dimensions = self._get_data_dimensions(model)
        # build a temporary data storage object
        data_storage = mfdatastorage.DataStorage(
            model.simulation_data,
            model,
            data_dimensions,
            None,
            mfdatastorage.DataStorageType.internal_array,
            mfdatastorage.DataStructureType.recarray,
        )

        # build type list
        type_list = data_storage.build_type_list(
            nseg=nseg, cellid_expanded=cellid_expanded
        )
        if data_storage.jagged_record:
            comment = (
                "Data dimensions can not be determined for  "
                "{}. Data structure may be jagged or may contain "
                "a keystring. Data type information is therefore "
                "dependant on the data and can not be retreived "
                "prior to the data being loaded"
                ".".format(data_storage.data_dimensions.structure.name)
            )
            type_, value_, traceback_ = sys.exc_info()

            raise MFDataException(
                data_struct.get_model(),
                data_struct.get_package(),
                data_struct.path,
                "generating array template",
                data_struct.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                model.simulation_data.debug,
            )
        if aux_vars is not None:
            if len(aux_vars) > 0:
                if isinstance(aux_vars[0], list) or isinstance(
                    aux_vars[0], tuple
                ):
                    aux_vars = aux_vars[0]
            for aux_var in aux_vars:
                type_list.append((aux_var, object))
        if boundnames:
            type_list.append(("boundname", object))

        if timeseries:
            # fix type list to make all types objects
            for index, d_type in enumerate(type_list):
                type_list[index] = (d_type[0], object)
        return type_list

    def empty(
        self,
        model,
        maxbound=None,
        aux_vars=None,
        boundnames=False,
        nseg=None,
        timeseries=False,
        stress_periods=None,
        cellid_expanded=False,
    ):
        from ..data import mfstructure

        # get type list
        type_list = self.dtype(
            model,
            aux_vars,
            boundnames,
            nseg,
            timeseries,
            cellid_expanded,
        )

        # get data storage
        data_struct = self._get_data_dimensions(model)[0]
        data_type = data_struct.get_datatype()

        # build recarray
        template_data = self._build_template_data(type_list)
        rec_array_data = []
        if maxbound is not None:
            for index in range(0, maxbound):
                rec_array_data.append(template_data)
        else:
            rec_array_data.append(template_data)
        rec_array = np.rec.array(rec_array_data, type_list)

        # if transient/multiple list
        if (
            data_type == mfstructure.DataType.list_transient
            or data_type == mfstructure.DataType.list_multiple
        ):
            # Return as dictionary
            if stress_periods is None:
                return {0: rec_array}
            else:
                template = {}
                for stress_period in stress_periods:
                    template[stress_period] = deepcopy(rec_array)
                return template
        else:
            return rec_array


class MFDocString:
    """
    Helps build a python class doc string

    Parameters
    ----------
    description : string
        description of the class

    Attributes
    ----------
    indent: string
        indent to use in doc string
    description : string
        description of the class
    parameter_header : string
        header for parameter section of doc string
    parameters : list
        list of docstrings for class parameters

    Methods
    -------
    add_parameter : (param_descr : string, beginning_of_list : bool)
        adds doc string for a parameter with description 'param_descr' to the
        end of the list unless beginning_of_list is True
    get_doc_string : () : string
        builds and returns the docstring for the class
    """

    def __init__(self, description):
        self.indent = "    "
        self.description = description
        self.parameter_header = "{}Parameters\n{}----------".format(
            self.indent, self.indent
        )
        self.parameters = []
        self.model_parameters = []

    def add_parameter(
        self, param_descr, beginning_of_list=False, model_parameter=False
    ):
        if beginning_of_list:
            self.parameters.insert(0, param_descr)
            if model_parameter:
                self.model_parameters.insert(0, param_descr)
        else:
            self.parameters.append(param_descr)
            if model_parameter:
                self.model_parameters.append(param_descr)

    def get_doc_string(self, model_doc_string=False):
        doc_string = '{}"""\n{}{}\n\n{}\n'.format(
            self.indent, self.indent, self.description, self.parameter_header
        )
        if model_doc_string:
            param_list = self.model_parameters
            doc_string = (
                "{}    modelname : string\n        name of the "
                "model\n    model_nam_file : string\n"
                "        relative path to the model name file from "
                "model working folder\n    version : string\n"
                "        version of modflow\n    exe_name : string\n"
                "        model executable name\n"
                "    model_ws : string\n"
                "        model working folder path"
                "\n".format(doc_string)
            )
        else:
            param_list = self.parameters
        for parameter in param_list:
            doc_string += "{}\n".format(parameter)
        if not model_doc_string:
            doc_string += '\n{}"""'.format(self.indent)
        return doc_string
