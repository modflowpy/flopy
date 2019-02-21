import os, sys, inspect
import numpy as np
from copy import deepcopy
from ..mfbase import MFDataException


def get_first_val(arr):
    while isinstance(arr, list) or isinstance(arr, np.ndarray):
        arr = arr[0]
    return arr


class TemplateGenerator(object):
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
        package_dim = modeldimensions.PackageDimensions([model.dimensions],
                                                        package_struct,
                                                        self.path[0:-1])
        return data_struct, modeldimensions.DataDimensions(package_dim,
                                                           data_struct)

    def build_type_header(self, type, data=None):
        from ..data.mfdata import DataStorageType

        if type == DataStorageType.internal_array:
            if isinstance(self, ArrayTemplateGenerator):
                return {'factor':1.0, 'iprn':1, 'data':data}
            else:
                return None
        elif type == DataStorageType.internal_constant:
            return data
        elif type == DataStorageType.external_file:
            return {'filename':'', 'factor':1.0, 'iprn':1}
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
    empty: (model: MFModel, layered: boolean, data_storage_type_list: boolean,
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
        super(ArrayTemplateGenerator, self).__init__(path)

    def empty(self, model=None, layered=False, data_storage_type_list=None,
              default_value=None):
        from ..data import mfdata, mfstructure
        from ..data.mfdata import DataStorageType

        # get the expected dimensions of the data
        data_struct, data_dimensions = self._get_data_dimensions(model)
        datum_type = data_struct.get_datum_type()
        data_type = data_struct.get_datatype()
        # build a temporary data storage object
        data_storage = mfdata.DataStorage(
                model.simulation_data, data_dimensions, None,
                mfdata.DataStorageType.internal_array,
                mfdata.DataStructureType.recarray)
        dimension_list = data_storage.get_data_dimensions(None)

        # if layered data
        if layered and dimension_list[0] > 1:
            if data_storage_type_list is not None and \
                    len(data_storage_type_list) != dimension_list[0]:
                comment = 'data_storage_type_list specified with the ' \
                          'wrong size.  Size {} but expected to be ' \
                          'the same as the number of layers, ' \
                          '{}.'.format(len(data_storage_type_list),
                                       dimension_list[0])
                type_, value_, traceback_ = sys.exc_info()

                raise MFDataException(data_struct.get_model(),
                                      data_struct.get_package(),
                                      data_struct.path,
                                      'generating array template',
                                      data_struct.name,
                                      inspect.stack()[0][3],
                                      type_, value_, traceback_, comment,
                                      model.simulation_data.debug)
            # build each layer
            data_with_header = []
            for layer in range(0, dimension_list[0]):
                # determine storage type
                if data_storage_type_list is None:
                    data_storage_type = DataStorageType.internal_array
                else:
                    data_storage_type = data_storage_type_list[layer]
                # build data type header
                data_with_header.append(self._build_layer(datum_type,
                                                          data_storage_type,
                                                          default_value,
                                                          dimension_list))
        else:
            if data_storage_type_list is None or \
                    data_storage_type_list[0] == \
                            DataStorageType.internal_array:
                data_storage_type = DataStorageType.internal_array
            else:
                data_storage_type = data_storage_type_list[0]
            # build data type header
            data_with_header = self._build_layer(datum_type,
                                                 data_storage_type,
                                                 default_value,
                                                 dimension_list, True)

        # if transient/multiple list
        if data_type == mfstructure.DataType.array_transient:
            # Return as dictionary
            return {0:data_with_header}
        else:
            return data_with_header

    def _build_layer(self, data_type, data_storage_type, default_value,
                     dimension_list, all_layers=False):
        from ..data.mfdata import DataStorageType

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
                    data = np.full(dimension_list[1:], default_value,
                                   data_type)
        elif data_storage_type == DataStorageType.internal_constant:
            if default_value is None:
                if data_type == np.int:
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
    empty: (maxbound: int, aux_vars: list, boundnames: boolean, nseg: int) :
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
        super(ListTemplateGenerator, self).__init__(path)

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

    def empty(self, model, maxbound=None, aux_vars=None, boundnames=False,
              nseg=None, timeseries=False, stress_periods=None):
        from ..data import mfdata, mfstructure

        data_struct, data_dimensions = self._get_data_dimensions(model)
        data_type = data_struct.get_datatype()
        # build a temporary data storage object
        data_storage = mfdata.DataStorage(
                model.simulation_data, data_dimensions, None,
                mfdata.DataStorageType.internal_array,
                mfdata.DataStructureType.recarray)

        # build type list
        type_list = data_storage.build_type_list(nseg=nseg)
        if aux_vars is not None:
            if len(aux_vars) > 0 and (isinstance(aux_vars[0], list) or
                    isinstance(aux_vars[0], tuple)):
                aux_vars = aux_vars[0]
            for aux_var in aux_vars:
                type_list.append((aux_var, object))
        if boundnames:
            type_list.append(('boundnames', object))

        if timeseries:
            # fix type list to make all types objects
            for index in range(0, len(type_list)):
                type_list[index] = (type_list[index][0], object)

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
        if data_type == mfstructure.DataType.list_transient or \
                data_type == mfstructure.DataType.list_multiple:
            # Return as dictionary
            if stress_periods is None:
                return {0:rec_array}
            else:
                template = {}
                for stress_period in stress_periods:
                    template[stress_period] = deepcopy(rec_array)
                return template
        else:
            return rec_array


class MFDocString(object):
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
        self.indent = '    '
        self.description = description
        self.parameter_header = '{}Parameters\n{}' \
                                '----------'.format(self.indent, self.indent)
        self.parameters = []
        self.model_parameters = []

    def add_parameter(self, param_descr, beginning_of_list=False,
                      model_parameter=False):
        if beginning_of_list:
            self.parameters.insert(0, param_descr)
            if model_parameter:
                self.model_parameters.insert(0, param_descr)
        else:
            self.parameters.append(param_descr)
            if model_parameter:
                self.model_parameters.append(param_descr)

    def get_doc_string(self, model_doc_string=False):
        doc_string = '{}"""\n{}{}\n\n{}\n'.format(self.indent, self.indent,
                                                  self.description,
                                                  self.parameter_header)
        if model_doc_string:
            param_list = self.model_parameters
            doc_string = '{}    modelname : string\n        name of the ' \
                         'model\n    model_nam_file : string\n' \
                         '        relative path to the model name file from ' \
                         'model working folder\n    version : string\n' \
                         '        version of modflow\n    exe_name : string\n'\
                         '        model executable name\n' \
                         '    model_ws : string\n' \
                         '        model working folder path' \
                         '\n'.format(doc_string)
        else:
            param_list = self.parameters
        for parameter in param_list:
            doc_string += '{}\n'.format(parameter)
        if not model_doc_string:
            doc_string += '\n{}"""'.format(self.indent)
        return doc_string
