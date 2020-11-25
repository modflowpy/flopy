import sys, inspect
import numpy as np
from ..data.mfstructure import DatumType
from ..data import mfdata
from collections import OrderedDict
from ..mfbase import ExtFileAction, MFDataException
from ...datbase import DataType
from .mfdatautil import convert_data, to_string
from .mffileaccess import MFFileAccessScalar
from .mfdatastorage import DataStorage, DataStructureType, DataStorageType


class MFScalar(mfdata.MFData):
    """
    Provides an interface for the user to access and update MODFLOW
    scalar data.

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

    Attributes
    ----------
    data_type : DataType
        type of data stored in the scalar
    plottable : bool
        if the scalar is plottable
    dtype : numpy.dtype
        the scalar's numpy data type
    data : variable
        calls get_data with default parameters

    Methods
    -------
    has_data : () : bool
        Returns whether this object has data associated with it.
    get_data : () : ndarray
        Returns the data associated with this object.
    set_data : (data : ndarray/list, multiplier : float)
        Sets the contents of the data to "data" with
        multiplier "multiplier".
    load : (first_line : string, file_handle : file descriptor,
            block_header : MFBlockHeader, pre_data_comments : MFComment) :
            tuple (bool, string)
        Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
    get_file_entry : () : string
        Returns a string containing the data.

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
        data=None,
        enable=True,
        path=None,
        dimensions=None,
    ):
        super(MFScalar, self).__init__(
            sim_data, model_or_sim, structure, enable, path, dimensions
        )
        self._data_type = self.structure.data_item_structures[0].type
        self._data_storage = self._new_storage()
        if data is not None:
            self.set_data(data)

    @property
    def data_type(self):
        return DataType.scalar

    @property
    def plottable(self):
        return False

    @property
    def dtype(self):
        if self.structure.type == DatumType.double_precision:
            return np.float64
        elif self.structure.type == DatumType.integer:
            return np.int32
        elif (
            self.structure.type == DatumType.recarray
            or self.structure.type == DatumType.record
            or self.structure.type == DatumType.repeating_record
        ):
            for data_item_struct in self.structure.data_item_structures:
                if data_item_struct.type == DatumType.double_precision:
                    return np.float64
                elif data_item_struct.type == DatumType.integer:
                    return np.int32
        return None

    def has_data(self):
        try:
            return self._get_storage_obj().has_data()
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

    @property
    def data(self):
        return self.get_data()

    def get_data(self, apply_mult=False, **kwargs):
        try:
            return self._get_storage_obj().get_data(apply_mult=apply_mult)
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

    def set_data(self, data):
        self._resync()
        if self.structure.type == DatumType.record:
            if data is not None:
                if (
                    not isinstance(data, list)
                    or isinstance(data, np.ndarray)
                    or isinstance(data, tuple)
                ):
                    data = [data]
        else:
            while (
                isinstance(data, list)
                or isinstance(data, np.ndarray)
                or isinstance(data, tuple)
            ):
                data = data[0]
                if (isinstance(data, list) or isinstance(data, tuple)) and len(
                    data
                ) > 1:
                    self._add_data_line_comment(data[1:], 0)
        storage = self._get_storage_obj()
        data_struct = self.structure.data_item_structures[0]
        try:
            converted_data = convert_data(
                data, self._data_dimensions, self._data_type, data_struct
            )
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment = 'Could not convert data "{}" to type ' '"{}".'.format(
                data, self._data_type
            )
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "converting data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self._simulation_data.debug,
                ex,
            )
        try:
            storage.set_data(converted_data, key=self._current_key)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            comment = 'Could not set data "{}" to type ' '"{}".'.format(
                data, self._data_type
            )
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
                comment,
                self._simulation_data.debug,
                ex,
            )

    def add_one(self):
        datum_type = self.structure.get_datum_type()
        if datum_type == int or datum_type == np.int32:
            if self._get_storage_obj().get_data() is None:
                try:
                    self._get_storage_obj().set_data(1)
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    comment = "Could not set data to 1"
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
                        comment,
                        self._simulation_data.debug,
                        ex,
                    )
            else:
                try:
                    current_val = self._get_storage_obj().get_data()
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
                try:
                    self._get_storage_obj().set_data(current_val + 1)
                except Exception as ex:
                    type_, value_, traceback_ = sys.exc_info()
                    comment = 'Could increment data "{}" by one' ".".format(
                        current_val
                    )
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
                        comment,
                        self._simulation_data.debug,
                        ex,
                    )
        else:
            message = (
                "{} of type {} does not support add one "
                "operation.".format(
                    self._data_name, self.structure.get_datum_type()
                )
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "adding one to scalar",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

    def get_file_entry(
        self,
        values_only=False,
        one_based=False,
        ext_file_action=ExtFileAction.copy_relative_paths,
    ):
        storage = self._get_storage_obj()
        try:
            if storage is None or self._get_storage_obj().get_data() is None:
                return ""
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
            self.structure.type == DatumType.keyword
            or self.structure.type == DatumType.record
        ):
            try:
                data = storage.get_data()
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
        if self.structure.type == DatumType.keyword:
            if data is not None and data != False:
                # keyword appears alone
                return "{}{}\n".format(
                    self._simulation_data.indent_string,
                    self.structure.name.upper(),
                )
            else:
                return ""
        elif self.structure.type == DatumType.record:
            text_line = []
            index = 0
            for data_item in self.structure.data_item_structures:
                if (
                    data_item.type == DatumType.keyword
                    and data_item.optional == False
                ):
                    if isinstance(data, list) or isinstance(data, tuple):
                        if len(data) > index and (
                            data[index] is not None and data[index] != False
                        ):
                            text_line.append(data_item.name.upper())
                            if (
                                isinstance(data[index], str)
                                and data_item.name.upper()
                                != data[index].upper()
                                and data[index] != ""
                            ):
                                # since the data does not match the keyword
                                # assume the keyword was excluded
                                index -= 1
                    else:
                        if data is not None and data != False:
                            text_line.append(data_item.name.upper())
                else:
                    if data is not None and data != "":
                        if isinstance(data, list) or isinstance(data, tuple):
                            if len(data) > index:
                                if (
                                    data[index] is not None
                                    and data[index] != False
                                ):
                                    current_data = data[index]
                                else:
                                    break
                            elif data_item.optional == True:
                                break
                            else:
                                message = (
                                    "Missing expected data. Data "
                                    "size is {}. Index {} not"
                                    "found.".format(len(data), index)
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
                                    message,
                                    self._simulation_data.debug,
                                )

                        else:
                            current_data = data
                        if data_item.type == DatumType.keyword:
                            if (
                                current_data is not None
                                and current_data != False
                            ):
                                if (
                                    isinstance(data[index], str)
                                    and data[index] == "#"
                                ):
                                    # if data has been commented out,
                                    # keep the comment
                                    text_line.append(data[index])
                                text_line.append(data_item.name.upper())
                        else:
                            try:
                                text_line.append(
                                    to_string(
                                        current_data,
                                        self._data_type,
                                        self._simulation_data,
                                        self._data_dimensions,
                                        data_item=data_item,
                                    )
                                )
                            except Exception as ex:
                                message = (
                                    'Could not convert "{}" of type '
                                    '"{}" to a string'
                                    ".".format(current_data, self._data_type)
                                )
                                type_, value_, traceback_ = sys.exc_info()
                                raise MFDataException(
                                    self.structure.get_model(),
                                    self.structure.get_package(),
                                    self._path,
                                    "converting data to string",
                                    self.structure.name,
                                    inspect.stack()[0][3],
                                    type_,
                                    value_,
                                    traceback_,
                                    message,
                                    self._simulation_data.debug,
                                )
                index += 1

            text = self._simulation_data.indent_string.join(text_line)
            return "{}{}\n".format(self._simulation_data.indent_string, text)
        else:
            data_item = self.structure.data_item_structures[0]
            try:
                if one_based:
                    if self.structure.type != DatumType.integer:
                        message = (
                            'Data scalar "{}" can not be one_based '
                            "because it is not an integer"
                            ".".format(self.structure.name)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self.structure.get_model(),
                            self.structure.get_package(),
                            self._path,
                            "storing one based integer",
                            self.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
                    data = self._get_storage_obj().get_data() + 1
                else:
                    data = self._get_storage_obj().get_data()
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
                )
            try:
                # data
                values = to_string(
                    data,
                    self._data_type,
                    self._simulation_data,
                    self._data_dimensions,
                    data_item=data_item,
                )
            except Exception as ex:
                message = (
                    'Could not convert "{}" of type "{}" '
                    "to a string.".format(data, self._data_type)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "converting data to string",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            if values_only:
                return "{}{}".format(
                    self._simulation_data.indent_string, values
                )
            else:
                # keyword + data
                return "{}{}{}{}\n".format(
                    self._simulation_data.indent_string,
                    self.structure.name.upper(),
                    self._simulation_data.indent_string,
                    values,
                )

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        super(MFScalar, self).load(
            first_line,
            file_handle,
            block_header,
            pre_data_comments=None,
            external_file_info=None,
        )
        self._resync()
        file_access = MFFileAccessScalar(
            self.structure,
            self._data_dimensions,
            self._simulation_data,
            self._path,
            self._current_key,
        )
        return file_access.load_from_package(
            first_line,
            file_handle,
            self._get_storage_obj(),
            self._data_type,
            self._keyword,
            pre_data_comments,
        )

    def _new_storage(self, stress_period=0):
        return DataStorage(
            self._simulation_data,
            self._model_or_sim,
            self._data_dimensions,
            self.get_file_entry,
            DataStorageType.internal_array,
            DataStructureType.scalar,
            stress_period=stress_period,
            data_path=self._path,
        )

    def _get_storage_obj(self):
        return self._data_storage

    def plot(self, filename_base=None, file_extension=None, **kwargs):
        """
        Helper method to plot scalar objects

        Parameters:
            scalar : flopy.mf6.data.mfscalar object
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')

        Returns:
             axes: list matplotlib.axes object
        """
        from flopy.plot.plotutil import PlotUtilities

        if not self.plottable:
            raise TypeError("Scalar values are not plottable")

        axes = PlotUtilities._plot_scalar_helper(
            self,
            filename_base=filename_base,
            file_extension=file_extension,
            **kwargs
        )
        return axes


class MFScalarTransient(MFScalar, mfdata.MFTransient):
    """
    Provides an interface for the user to access and update MODFLOW transient
    scalar data.

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

    Methods
    -------
    add_transient_key : (transient_key : int)
        Adds a new transient time allowing data for that time to be stored and
        retrieved using the key "transient_key"
    add_one :(transient_key : int)
        Adds one to the data stored at key "transient_key"
    get_data : (key : int) : ndarray
        Returns the data associated with "key".
    set_data : (data : ndarray/list, multiplier : float, key : int)
        Sets the contents of the data at time "key" to
        "data" with multiplier "multiplier".
    load : (first_line : string, file_handle : file descriptor,
            block_header : MFBlockHeader, pre_data_comments : MFComment) :
            tuple (bool, string)
        Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
    get_file_entry : (key : int) : string
        Returns a string containing the data at time "key".

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
    ):
        super(MFScalarTransient, self).__init__(
            sim_data=sim_data,
            model_or_sim=model_or_sim,
            structure=structure,
            enable=enable,
            path=path,
            dimensions=dimensions,
        )
        self._transient_setup(self._data_storage)
        self.repeating = True

    @property
    def data_type(self):
        return DataType.transientscalar

    @property
    def plottable(self):
        if self.model is None:
            return False
        else:
            return True

    def add_transient_key(self, key):
        super(MFScalarTransient, self).add_transient_key(key)
        if isinstance(key, int):
            stress_period = key
        else:
            stress_period = 1
        self._data_storage[key] = super(MFScalarTransient, self)._new_storage(
            stress_period
        )

    def add_one(self, key=0):
        self._update_record_prep(key)
        super(MFScalarTransient, self).add_one()

    def has_data(self, key=None):
        if key is None:
            data_found = False
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                data_found = (
                    data_found or super(MFScalarTransient, self).has_data()
                )
                if data_found:
                    break
        else:
            self.get_data_prep(key)
            data_found = super(MFScalarTransient, self).has_data()
        return data_found

    def get_data(self, key=0, **kwargs):
        self.get_data_prep(key)
        return super(MFScalarTransient, self).get_data()

    def set_data(self, data, key=None):
        if isinstance(data, dict) or isinstance(data, OrderedDict):
            # each item in the dictionary is a list for one stress period
            # the dictionary key is the stress period the list is for
            for key, list_item in data.items():
                self._set_data_prep(list_item, key)
                super(MFScalarTransient, self).set_data(list_item)
        else:
            self._set_data_prep(data, key)
            super(MFScalarTransient, self).set_data(data)

    def get_file_entry(
        self, key=None, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        if key is None:
            file_entry = []
            for sto_key in self._data_storage.keys():
                if self.has_data(sto_key):
                    self._get_file_entry_prep(sto_key)
                    text_entry = super(MFScalarTransient, self).get_file_entry(
                        ext_file_action=ext_file_action
                    )
                    file_entry.append(text_entry)
            if file_entry > 1:
                return "\n\n".join(file_entry)
            elif file_entry == 1:
                return file_entry[0]
            else:
                return ""
        else:
            self._get_file_entry_prep(key)
            return super(MFScalarTransient, self).get_file_entry(
                ext_file_action=ext_file_action
            )

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        self._load_prep(block_header)
        return super(MFScalarTransient, self).load(
            first_line, file_handle, pre_data_comments, external_file_info
        )

    def _new_storage(self, stress_period=0):
        return OrderedDict()

    def _get_storage_obj(self):
        if (
            self._current_key is None
            or self._current_key not in self._data_storage
        ):
            return None
        return self._data_storage[self._current_key]

    def plot(
        self,
        filename_base=None,
        file_extension=None,
        kper=0,
        fignum=None,
        **kwargs
    ):
        """
        Plot transient scalar model data

        Parameters
        ----------
        transientscalar : flopy.mf6.data.mfdatascalar.MFScalarTransient object
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
