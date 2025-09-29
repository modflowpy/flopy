import copy
import inspect
import io
import os
import sys
import warnings

import numpy as np
import pandas

from ...datbase import DataListInterface, DataType
from ...discretization.structuredgrid import StructuredGrid
from ...discretization.unstructuredgrid import UnstructuredGrid
from ...discretization.vertexgrid import VertexGrid
from ...utils import datautil
from ..data import mfdata
from ..mfbase import ExtFileAction, MFDataException, VerbosityLevel
from ..utils.mfenums import DiscretizationType
from .mfdatalist import MFList
from .mfdatastorage import DataStorageType, DataStructureType
from .mfdatautil import MFComment, list_to_array, process_open_close_line
from .mffileaccess import MFFileAccessList
from .mfstructure import DatumType, MFDataStructure


class PandasListStorage:
    """
    Contains data storage information for a single list.

    Attributes
    ----------
    internal_data : ndarray or recarray
        data being stored, if full data is being stored internally in memory
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
    modified : bool
        whether data in storage has been modified since last write
    pre_data_comments : string
        any comments before the start of the data

    Methods
    -------
    get_record : dict
        returns a dictionary with all data storage information
    set_record(rec)
        sets data storage information based on the the dictionary "rec"
    set_internal(internal_data)
        make data storage internal, using "internal_data" as the data
    set_external(fname, data)
        make data storage external, with file "fname" and external data "data"
    internal_size : int
        size of the internal data
    has_data : bool
        whether or not data exists
    """

    def __init__(self):
        self.internal_data = None
        self.fname = None
        self.iprn = None
        self.binary = False
        self.data_storage_type = None
        self.modified = False
        self.pre_data_comments = None

    def __repr__(self):
        return self.get_data_str(True)

    def __str__(self):
        return self.get_data_str(False)

    def _get_header_str(self):
        header_list = []
        if self.data_storage_type == DataStorageType.external_file:
            header_list.append(f"open/close {self.fname}")
        else:
            header_list.append("internal")
        if self.iprn is not None:
            header_list.append(f"iprn {self.iprn}")
        if len(header_list) > 0:
            return ", ".join(header_list)
        else:
            return ""

    def get_data_str(self, formal):
        data_str = ""
        layer_str = ""
        if self.data_storage_type == DataStorageType.internal_array:
            if self.internal_data is not None:
                header = self._get_header_str()
                if formal:
                    data_str = "{}{}{{{}}}\n({})\n".format(
                        data_str,
                        layer_str,
                        header,
                        repr(self.internal_data),
                    )
                else:
                    data_str = "{}{}{{{}}}\n({})\n".format(
                        data_str,
                        layer_str,
                        header,
                        str(self.internal_data),
                    )
        elif self.data_storage_type == DataStorageType.external_file:
            header = self._get_header_str()
            data_str = "{}{}{{{}}}\n({})\n".format(
                data_str,
                layer_str,
                header,
                "External data not displayed",
            )
        return data_str

    def get_record(self):
        rec = {}
        if self.internal_data is not None:
            rec["data"] = copy.deepcopy(self.internal_data)
        if self.fname is not None:
            rec["filename"] = self.fname
        if self.iprn is not None:
            rec["iprn"] = self.iprn
        if self.binary is not None:
            rec["binary"] = self.binary
        return rec

    def set_record(self, rec):
        if "data" in rec:
            self.internal_data = rec["data"]
        if "filename" in rec:
            self.fname = rec["filename"]
        if "iprn" in rec:
            self.iprn = rec["iprn"]
        if "binary" in rec:
            self.binary = rec["binary"]

    def set_internal(self, internal_data):
        self.data_storage_type = DataStorageType.internal_array
        self.internal_data = internal_data
        self.fname = None
        self.binary = False

    def set_external(self, fname, data=None):
        self.data_storage_type = DataStorageType.external_file
        self.internal_data = data
        self.fname = fname

    @property
    def internal_size(self):
        if not isinstance(self.internal_data, pandas.DataFrame):
            return 0
        else:
            return len(self.internal_data)

    def has_data(self):
        if self.data_storage_type == DataStorageType.internal_array:
            return self.internal_data is not None
        else:
            return self.fname is not None


class MFPandasList(mfdata.MFMultiDimVar, DataListInterface):
    """
    Provides an interface for the user to access and update MODFLOW
    list data using Pandas.  MFPandasList objects are not designed to be
    directly constructed by the end user. When a flopy for MODFLOW 6 package
    object is constructed, the appropriate MFList objects are automatically
    built.

    Parameters
    ----------
    sim_data : MFSimulationData
        data contained in the simulation
    model_or_sim : MFSimulation or MFModel
        parent model, or if not part of a model, parent simulation
    structure : MFDataStructure
        describes the structure of the data
    data : list or ndarray or None
        actual data
    enable : bool
        enable/disable the array
    path : tuple
        path in the data dictionary to this MFArray
    dimensions : MFDataDimensions
        dimension information related to the model, package, and array
    package : MFPackage
        parent package
    block : MFBlock
        parnet block
    """

    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        data=None,
        enable=None,
        path=None,
        dimensions=None,
        package=None,
        block=None,
    ):
        super().__init__(
            sim_data, model_or_sim, structure, enable, path, dimensions
        )
        self._data_storage = self._new_storage()
        self._package = package
        self._block = block
        self._last_line_info = []
        self._data_line = None
        self._temp_dict = {}
        self._crnt_line_num = 1
        self._data_header = None
        self._header_names = None
        self._data_types = None
        self._data_item_names = None
        self._mg = None
        self._current_key = 0
        self._max_file_size = 1000000000000000
        if self._model_or_sim.type == "Model":
            self._mg = self._model_or_sim.modelgrid

        if data is not None:
            try:
                self.set_data(data, True)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    structure.get_model(),
                    structure.get_package(),
                    path,
                    "setting data",
                    structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    sim_data.debug,
                    ex,
                )

    @property
    def data_type(self):
        """Type of data (DataType) stored in the list"""
        return DataType.list

    @property
    def package(self):
        """Package object that this data belongs to."""
        return self._package

    @property
    def dtype(self):
        """Type of data (numpy.dtype) stored in the list"""
        return self.get_dataframe().dtype

    def _append_type_list(self, data_name, data_type, include_header=False):
        if include_header:
            self._data_header[data_name] = data_type
        self._header_names.append(data_name)
        self._data_types.append(data_type)

    def _process_open_close_line(self, arr_line, store=True):
        """
        Process open/close line extracting the multiplier, print format,
        binary flag, data file path, and any comments
        """
        data_dim = self.data_dimensions
        (
            multiplier,
            print_format,
            binary,
            data_file,
            data,
            comment,
        ) = process_open_close_line(
            arr_line,
            data_dim,
            self._data_type,
            self._simulation_data.debug,
            store,
        )
        #  add to active list of external files
        model_name = data_dim.package_dim.model_dim[0].model_name
        self._simulation_data.mfpath.add_ext_file(data_file, model_name)

        return data, multiplier, print_format, binary, data_file

    def _add_cellid_fields(self, data, keep_existing=False):
        """
        Add cellid fields to a Pandas DataFrame and drop the layer,
        row, column, cell, node, fields that the cellid is based on
        """
        for data_item in self.structure.data_item_structures:
            if data_item.type == DatumType.integer:
                if data_item.name.lower() == "cellid":
                    columns = data.columns.tolist()
                    if isinstance(self._mg, StructuredGrid):
                        if (
                            "cellid_layer" in columns
                            and "cellid_row" in columns
                            and "cellid_column" in columns
                        ):
                            data["cellid"] = data[
                                ["cellid_layer", "cellid_row", "cellid_column"]
                            ].apply(tuple, axis=1)
                            if not keep_existing:
                                data = data.drop(
                                    columns=[
                                        "cellid_layer",
                                        "cellid_row",
                                        "cellid_column",
                                    ]
                                )
                    elif isinstance(self._mg, VertexGrid):
                        cell_2 = None
                        if "cellid_cell" in columns:
                            cell_2 = "cellid_cell"
                        elif "ncpl" in columns:
                            cell_2 = "cellid_ncpl"
                        if cell_2 is not None and "cellid_layer" in columns:
                            data["cellid"] = data[
                                ["cellid_layer", cell_2]
                            ].apply(tuple, axis=1)
                            if not keep_existing:
                                data = data.drop(
                                    columns=["cellid_layer", cell_2]
                                )
                    elif isinstance(self._mg, UnstructuredGrid):
                        if "cellid_node" in columns:
                            data["cellid"] = data[["cellid_node"]].apply(
                                tuple, axis=1
                            )
                            if not keep_existing:
                                data = data.drop(columns=["cellid_node"])
                    else:
                        raise MFDataException(
                            "ERROR: Unrecognized model grid "
                            "{str(self._mg)} not supported by MFBasicList"
                        )
                    # reorder columns
                    column_headers = data.columns.tolist()
                    column_headers.insert(0, column_headers.pop())
                    data = data[column_headers]

        return data

    def _remove_cellid_fields(self, data):
        """remove cellid fields from data"""
        for data_item in self.structure.data_item_structures:
            if data_item.type == DatumType.integer:
                if data_item.name.lower() == "cellid":
                    # if there is a cellid field, remove it
                    if "cellid" in data.columns:
                        return data.drop("cellid", axis=1)
        return data

    def _get_cellid_size(self, data_item_name):
        """get the number of spatial coordinates used in the cellid"""
        model_num = datautil.DatumUtil.cellid_model_num(
            data_item_name,
            self.data_dimensions.structure.model_data,
            self.data_dimensions.package_dim.model_dim,
        )
        model_grid = self.data_dimensions.get_model_grid(model_num=model_num)
        return model_grid.get_num_spatial_coordinates()

    def _build_data_header(self):
        """
        Constructs lists of data column header names and data column types
        based on data structure information, boundname and aux information,
        and model discretization type.
        """
        # initialize
        self._data_header = {}
        self._header_names = []
        self._data_types = []
        self._data_item_names = []
        s_type = pandas.StringDtype
        f_type = np.float64
        i_type = np.int64
        data_dim = self.data_dimensions
        # loop through data structure definition information
        for data_item, index in zip(
            self.structure.data_item_structures,
            range(0, len(self.structure.data_item_structures)),
        ):
            if data_item.name.lower() == "aux":
                # get all of the aux variables for this dataset
                aux_var_names = data_dim.package_dim.get_aux_variables()
                if aux_var_names is not None:
                    for aux_var_name in aux_var_names[0]:
                        if aux_var_name.lower() != "auxiliary":
                            self._append_type_list(aux_var_name, f_type)
                            self._data_item_names.append(aux_var_name)
            elif data_item.name.lower() == "boundname":
                # see if boundnames is enabled for this dataset
                if data_dim.package_dim.boundnames():
                    self._append_type_list("boundname", s_type)
                    self._data_item_names.append("boundname")
            else:
                if data_item.type == DatumType.keyword:
                    self._append_type_list(data_item.name, s_type)
                elif data_item.type == DatumType.string:
                    self._append_type_list(data_item.name, s_type)
                elif data_item.type == DatumType.integer:
                    if data_item.name.lower() == "cellid":
                        # get the appropriate cellid column headings for the
                        # model's discretization type
                        if isinstance(self._mg, StructuredGrid):
                            self._append_type_list(
                                "cellid_layer", i_type, True
                            )
                            self._append_type_list("cellid_row", i_type, True)
                            self._append_type_list(
                                "cellid_column", i_type, True
                            )
                        elif isinstance(self._mg, VertexGrid):
                            self._append_type_list(
                                "cellid_layer", i_type, True
                            )
                            self._append_type_list("cellid_cell", i_type, True)
                        elif isinstance(self._mg, UnstructuredGrid):
                            self._append_type_list("cellid_node", i_type, True)
                        else:
                            raise MFDataException(
                                "ERROR: Unrecognized model grid "
                                "{str(self._mg)} not supported by MFBasicList"
                            )
                    else:
                        self._append_type_list(data_item.name, i_type)
                elif data_item.type == DatumType.double_precision:
                    self._append_type_list(data_item.name, f_type)
                else:
                    self._data_header = None
                    self._header_names = None
                self._data_item_names.append(data_item.name)

    @staticmethod
    def _unique_column_name(data, col_base_name):
        """generate a unique column name based on "col_base_name" """
        col_name = col_base_name
        idx = 2
        while col_name in data:
            col_name = f"{col_base_name}_{idx}"
            idx += 1
        return col_name

    @staticmethod
    def _untuple_manually(pdata, loc, new_column_name, column_name, index):
        """
        Loop through pandas DataFrame removing tuples from cellid columns.
        Used when pandas "insert" method to perform the same task fails.
        """
        # build new column list
        new_column = []
        for idx, row in pdata.iterrows():
            if isinstance(row[column_name], tuple) or isinstance(
                row[column_name], list
            ):
                new_column.append(row[column_name][index])
            else:
                new_column.append(row[column_name])

        # insert list as new column
        pdata.insert(
            loc=loc,
            column=new_column_name,
            value=new_column,
        )

    def _untuple_cellids(self, pdata):
        """
        For all cellids in "pdata", convert them to layer, row, column fields and
        and then drop the cellids from "pdata".  Returns the updated "pdata".
        """
        if pdata is None or len(pdata) == 0:
            return pdata, 0
        fields_to_correct = []
        data_idx = 0
        # find cellid columns that need to be fixed
        columns = pdata.columns
        for data_item in self.structure.data_item_structures:
            if data_idx >= len(columns) + 1:
                break
            if (
                data_item.type == DatumType.integer
                and data_item.name.lower() == "cellid"
            ):
                if isinstance(pdata.iloc[0, data_idx], (list, tuple)):
                    fields_to_correct.append((data_idx, columns[data_idx]))
                    data_idx += 1
                else:
                    data_idx += self._get_cellid_size(data_item.name)
            else:
                data_idx += 1

        # fix columns
        for field_idx, column_name in fields_to_correct:
            # add individual layer/row/column/cell/node columns
            if isinstance(self._mg, StructuredGrid):
                try:
                    pdata.insert(
                        loc=field_idx,
                        column=self._unique_column_name(pdata, "cellid_layer"),
                        value=pdata.apply(lambda x: x[column_name][0], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx,
                        self._unique_column_name(pdata, "cellid_layer"),
                        column_name,
                        0,
                    )
                try:
                    pdata.insert(
                        loc=field_idx + 1,
                        column=self._unique_column_name(pdata, "cellid_row"),
                        value=pdata.apply(lambda x: x[column_name][1], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx + 1,
                        self._unique_column_name(pdata, "cellid_row"),
                        column_name,
                        1,
                    )
                try:
                    pdata.insert(
                        loc=field_idx + 2,
                        column=self._unique_column_name(
                            pdata, "cellid_column"
                        ),
                        value=pdata.apply(lambda x: x[column_name][2], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx + 2,
                        self._unique_column_name(pdata, "cellid_column"),
                        column_name,
                        2,
                    )
            elif isinstance(self._mg, VertexGrid):
                try:
                    pdata.insert(
                        loc=field_idx,
                        column=self._unique_column_name(pdata, "cellid_layer"),
                        value=pdata.apply(lambda x: x[column_name][0], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx,
                        self._unique_column_name(pdata, "cellid_layer"),
                        column_name,
                        0,
                    )
                try:
                    pdata.insert(
                        loc=field_idx + 1,
                        column=self._unique_column_name(pdata, "cellid_cell"),
                        value=pdata.apply(lambda x: x[column_name][1], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx + 1,
                        self._unique_column_name(pdata, "cellid_cell"),
                        column_name,
                        1,
                    )
            elif isinstance(self._mg, UnstructuredGrid):
                if column_name == "cellid_node":
                    # fixing a problem where node was specified as a tuple
                    # make sure new column is named properly
                    column_name = "cellid_node_2"
                    pdata = pdata.rename(columns={"cellid_node": column_name})
                try:
                    pdata.insert(
                        loc=field_idx,
                        column=self._unique_column_name(pdata, "cellid_node"),
                        value=pdata.apply(lambda x: x[column_name][0], axis=1),
                    )
                except (ValueError, TypeError):
                    self._untuple_manually(
                        pdata,
                        field_idx,
                        self._unique_column_name(pdata, "cellid_node"),
                        column_name,
                        0,
                    )
            # remove cellid tuple
            pdata = pdata.drop(column_name, axis=1)
        return pdata, len(fields_to_correct)

    def _resolve_columns(self, data):
        """resolve the column headings for a specific dataset provided"""
        if len(data) == 0:
            return self._header_names, False
        if len(data[0]) == len(self._header_names) or len(data[0]) == 0:
            return self._header_names, False

        if len(data[0]) == len(self._data_item_names):
            return self._data_item_names, True

        if (
            len(data[0]) == len(self._header_names) - 1
            and self._header_names[-1] == "boundname"
        ):
            return self._header_names[:-1], True

        if (
            len(data[0]) == len(self._data_item_names) - 1
            and self._data_item_names[-1] == "boundname"
        ):
            return self._data_item_names[:-1], True

        return None, None

    def _untuple_recarray(self, rec):
        rec_list = rec.tolist()
        for row, line in enumerate(rec_list):
            for column, data in enumerate(line):
                if isinstance(data, tuple) and len(data) == 1:
                    line_lst = list(line)
                    line_lst[column] = data[0]
                    rec_list[row] = tuple(line_lst)
        return rec_list

    def set_data(self, data, autofill=False, check_data=True, append=False):
        """Sets the contents of the data to "data".  Data can have the
        following formats:
            1) recarray - recarray containing the datalist
            2) [(line_one), (line_two), ...] - list where each line of the
               datalist is a tuple within the list
        If the data is transient, a dictionary can be used to specify each
        stress period where the dictionary key is <stress period> - 1 and
        the dictionary value is the datalist data defined above:
        {0:ndarray, 1:[(line_one), (line_two), ...], 2:{'filename':filename})

        Parameters
        ----------
            data : ndarray/list/dict
                Data to set
            autofill : bool
                Automatically correct data
            check_data : bool
                Whether to verify the data
            append : bool
                Append to existing data

        """  # (re)build data header
        self._build_data_header()
        if isinstance(data, dict) and not self.has_data():
            MFPandasList.set_record(self, data)
            return
        if isinstance(data, np.recarray):
            # verify data shape of data (recarray)
            if len(data) == 0:
                # create empty dataset
                data = pandas.DataFrame(columns=self._header_names)
            elif len(data[0]) != len(self._header_names):
                if len(data[0]) == len(self._data_item_names):
                    # data most likely being stored with cellids as tuples,
                    # create a dataframe and untuple the cellids
                    data = pandas.DataFrame(
                        data, columns=self._data_item_names
                    )
                    data = self._untuple_cellids(data)[0]
                    # make sure columns are still in correct order
                    data = pandas.DataFrame(data, columns=self._header_names)
                else:
                    raise MFDataException(
                        f"ERROR: Data list {self._data_name} supplied the "
                        f"wrong number of columns of data, expected "
                        f"{len(self._data_item_names)} got {len(data[0])}."
                    )
            else:
                # data size matches the expected header names, create a pandas
                # dataframe from the data
                data_new = pandas.DataFrame(data, columns=self._header_names)
                if not self._dataframe_check(data_new):
                    data_list = self._untuple_recarray(data)
                    data = pandas.DataFrame(
                        data_list, columns=self._header_names
                    )
                else:
                    data, count = self._untuple_cellids(data_new)
                    if count > 0:
                        # make sure columns are still in correct order
                        data = pandas.DataFrame(
                            data, columns=self._header_names
                        )
        elif isinstance(data, list) or isinstance(data, tuple):
            if not (isinstance(data[0], list) or isinstance(data[0], tuple)):
                # get data in the format of a tuple of lists (or tuples)
                data = [data]
            # resolve the data's column headings
            columns = self._resolve_columns(data)[0]
            if columns is None:
                message = (
                    f"ERROR: Data list {self._data_name} supplied the "
                    f"wrong number of columns of data, expected "
                    f"{len(self._data_item_names)} got {len(data[0])}."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "setting list data",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )

            if len(data[0]) == 0:
                # create empty dataset
                data = pandas.DataFrame(columns=columns)
            else:
                # create dataset
                data = pandas.DataFrame(data, columns=columns)
            if (
                self._data_item_names[-1] == "boundname"
                and "boundname" not in columns
            ):
                # add empty boundname column
                data["boundname"] = ""
            # get rid of tuples from cellids
            data, count = self._untuple_cellids(data)
            if count > 0:
                # make sure columns are still in correct order
                data = pandas.DataFrame(data, columns=self._header_names)
        elif isinstance(data, pandas.DataFrame):
            if len(data.columns) != len(self._header_names):
                message = (
                    f"ERROR: Data list {self._data_name} supplied the "
                    f"wrong number of columns of data, expected "
                    f"{len(self._data_item_names)} got {len(data[0])}.\n"
                    f"Data columns supplied: {data.columns}\n"
                    f"Data columns expected: {self._header_names}"
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.data_dimensions.structure.get_model(),
                    self.data_dimensions.structure.get_package(),
                    self.data_dimensions.structure.path,
                    "setting list data",
                    self.data_dimensions.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            # set correct data header names
            data = data.set_axis(self._header_names, axis=1)
        else:
            message = (
                f"ERROR: Data list {self._data_name} is an unsupported type: "
                f"{type(data)}."
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.data_dimensions.structure.get_model(),
                self.data_dimensions.structure.get_package(),
                self.data_dimensions.structure.path,
                "setting list data",
                self.data_dimensions.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self._simulation_data.debug,
            )

        data_storage = self._get_storage_obj()
        if append:
            # append data to existing dataframe
            current_data = self._get_dataframe()
            if current_data is not None:
                data = pandas.concat([current_data, data])
        if data_storage.data_storage_type == DataStorageType.external_file:
            # store external data until next write
            data_storage.internal_data = data
        else:
            # store data internally
            data_storage.set_internal(data)
            data_storage.modified = True

    def has_modified_ext_data(self):
        """check to see if external data has been modified since last read"""
        data_storage = self._get_storage_obj()
        return (
            data_storage.data_storage_type == DataStorageType.external_file
            and data_storage.internal_data is not None
        )

    def binary_ext_data(self):
        """check for binary data"""
        data_storage = self._get_storage_obj()
        return data_storage.binary

    def to_array(self, kper=0, mask=False):
        """Convert stress period boundary condition (MFDataList) data for a
        specified stress period to a 3-D numpy array.

        Parameters
        ----------
        kper : int
            MODFLOW zero-based stress period number to return (default is
            zero)
        mask : bool
            return array with np.nan instead of zero

        Returns
        -------
        out : dict of numpy.ndarrays
            Dictionary of 3-D numpy arrays containing the stress period data
            for a selected stress period. The dictionary keys are the
            MFDataList dtype names for the stress period data."""
        sarr = self.get_data(key=kper)
        model_grid = self.data_dimensions.get_model_grid()
        return list_to_array(sarr, model_grid, kper, mask)

    def set_record(self, record, autofill=False, check_data=True):
        """Sets the contents of the data and metadata to "data_record".
        Data_record is a dictionary with has the following format:
            {'filename':filename, 'binary':True/False, 'data'=data}
        To store to file include 'filename' in the dictionary.

        Parameters
        ----------
            record : ndarray/list/dict
                Data and metadata to set
            autofill : bool
                Automatically correct data
            check_data : bool
                Whether to verify the data

        """
        if isinstance(record, dict):
            data_storage = self._get_storage_obj()
            if "filename" in record:
                data_storage.set_external(record["filename"])
                if "binary" in record:
                    if (
                        record["binary"]
                        and self.data_dimensions.package_dim.boundnames()
                    ):
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
                    data_storage.binary = record["binary"]
                if "data" in record:
                    # data gets written out to file
                    MFPandasList.set_data(self, record["data"])
                    # get file path
                    fd_file_path = self._get_file_path()
                    # make sure folder exists
                    folder_path = os.path.split(fd_file_path)[0]
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    # store data
                    self._write_file_entry(fd_file_path)
            else:
                if "data" in record:
                    data_storage.modified = True
                    data_storage.set_internal(None)
                    MFPandasList.set_data(self, record["data"])
            if "iprn" in record:
                data_storage.iprn = record["iprn"]

    def append_data(self, data):
        """Appends "data" to the end of this list.  Assumes data is in a format
        that can be appended directly to a pandas dataframe.

        Parameters
        ----------
            data : list(tuple)
                Data to append.

        """
        try:
            self._resync()
            if self._get_storage_obj() is None:
                self._data_storage = self._new_storage()
            data_storage = self._get_storage_obj()
            if (
                data_storage.data_storage_type
                == DataStorageType.internal_array
            ):
                # update internal data
                MFPandasList.set_data(self, data, append=True)
            elif (
                data_storage.data_storage_type == DataStorageType.external_file
            ):
                # get external data from file
                external_data = self._get_dataframe()
                if isinstance(data, list):
                    # build dataframe
                    data = pandas.DataFrame(
                        data, columns=external_data.columns
                    )
                # concatenate
                data = pandas.concat([external_data, data])
                # store
                ext_record = self._get_record()
                ext_record["data"] = data
                MFPandasList.set_record(self, ext_record)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "appending data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def append_list_as_record(self, record):
        """Appends the list `record` as a single record in this list's
        dataframe.  Assumes "data" has the correct dimensions.

        Parameters
        ----------
            record : list
                List to be appended as a single record to the data's existing
                recarray.

        """
        self._resync()
        try:
            # store
            self.append_data([record])
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "appending data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def update_record(self, record, key_index):
        """Updates a record at index "key_index" with the contents of "record".
        If the index does not exist update_record appends the contents of
        "record" to this list's recarray.

        Parameters
        ----------
            record : list
                New record to update data with
            key_index : int
                Stress period key of record to update.  Only used in transient
                data types.
        """
        self.append_list_as_record(record)

    def store_internal(
        self,
        check_data=True,
    ):
        """Store all data internally.

        Parameters
        ----------
            check_data : bool
                Verify data prior to storing

        """
        storage = self._get_storage_obj()
        # check if data is already stored external
        if (
            storage is None
            or storage.data_storage_type == DataStorageType.external_file
        ):
            data = self._get_dataframe()
            # if not empty dataset
            if data is not None:
                if (
                    self._simulation_data.verbosity_level.value
                    >= VerbosityLevel.verbose.value
                ):
                    print(f"Storing {self.structure.name} internally...")
                internal_data = {
                    "data": data,
                }
                MFPandasList.set_record(
                    self, internal_data, check_data=check_data
                )

    def store_as_external_file(
        self,
        external_file_path,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Store all data externally in file external_file_path. the binary
        allows storage in a binary file. If replace_existing_external is set
        to False, this method will not do anything if the data is already in
        an external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing

        """
        # only store data externally (do not subpackage info)
        if self.structure.construct_package is None:
            storage = self._get_storage_obj()
            # check if data is already stored external
            if (
                replace_existing_external
                or storage is None
                or storage.data_storage_type == DataStorageType.internal_array
                or storage.data_storage_type
                == DataStorageType.internal_constant
            ):
                data = self._get_dataframe()
                # if not empty dataset
                if data is not None:
                    if (
                        self._simulation_data.verbosity_level.value
                        >= VerbosityLevel.verbose.value
                    ):
                        print(
                            "Storing {} to external file {}.." ".".format(
                                self.structure.name, external_file_path
                            )
                        )
                    external_data = {
                        "filename": external_file_path,
                        "data": data,
                        "binary": binary,
                    }
                    MFPandasList.set_record(
                        self, external_data, check_data=check_data
                    )

    def external_file_name(self):
        """Returns external file name, or None if this is not external data."""
        storage = self._get_storage_obj()
        if storage is None:
            return None
        if (
            storage.data_storage_type == DataStorageType.external_file
            and storage.fname is not None
            and storage.fname != ""
        ):
            return storage.fname
        return None

    @staticmethod
    def _file_data_to_memory(fd_data_file, first_line):
        """
        scan data file from starting point to find the extent of the data

        Parameters
        ----------
        fd_data_file : file descriptor
            File with data to scan.  File location should be at the beginning
            of the data.

        Returns
        -------
        list, str : data from file, next line in file after data
        """
        data_lines = []
        clean_first_line = first_line.strip().lower()
        if clean_first_line.startswith("end"):
            return data_lines, fd_data_file.readline()
        if len(clean_first_line) > 0 and clean_first_line[0] != "#":
            data_lines.append(clean_first_line)
        line = fd_data_file.readline()
        while line:
            line_mod = line.strip().lower()
            if line_mod.startswith("end"):
                return data_lines, line
            if len(line_mod) > 0 and line_mod[0] != "#":
                data_lines.append(line_mod)
            line = fd_data_file.readline()
        return data_lines, ""

    def _dataframe_check(self, data_frame):
        valid = data_frame.shape[0] > 0
        if valid:
            for name in self._header_names:
                if (
                    name != "boundname"
                    and data_frame[name].isnull().values.any()
                ):
                    valid = False
                    break
        return valid

    def _try_pandas_read(self, fd_data_file, file_name):
        delimiter_list = ["\\s+", ","]
        for delimiter in delimiter_list:
            try:
                with warnings.catch_warnings(record=True) as warn:
                    # read flopy formatted data, entire file
                    data_frame = pandas.read_csv(
                        fd_data_file,
                        sep=delimiter,
                        names=self._header_names,
                        dtype=self._data_header,
                        comment="#",
                        index_col=False,
                        skipinitialspace=True,
                    )
                    if (
                        self._simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        for warning in warn:
                            print(
                                "Pandas warning occurred while loading data "
                                f"{self.path}:"
                            )
                            print(f'    Data File: "{file_name}:"')
                            print(f'    Pandas Message: "{warning.message}"')
            except BaseException:
                fd_data_file.seek(0)
                continue

            # basic check for valid dataset
            if self._dataframe_check(data_frame):
                return data_frame
            else:
                fd_data_file.seek(0)
        return None

    def _read_text_data(self, fd_data_file, first_line, external_file=False):
        """
        read list data from data file

        Parameters
        ----------
        fd_data_file : file descriptor
            File with data.  File location should be at the beginning of the
            data.

        external_file : bool
            whether this is an external file

        Returns
        -------
        DataFrame : file's list data
        list : containing boolean for success of operation and the next line of
               data in the file
        """
        # initialize
        data_frame = None
        return_val = [False, None]

        # read pre data comments
        pos = fd_data_file.tell()
        datautil.PyListUtil.reset_delimiter_used()
        line_num = 0
        pre_data_comments = None
        line = fd_data_file.readline()
        while MFComment.is_comment(line, True) and line != "":
            if pre_data_comments is not None:
                pre_data_comments.add_text("\n")
                pre_data_comments.add_text(" ".join(line))
            else:
                pre_data_comments = MFComment(
                    line, self._path, self._simulation_data, line_num
                )

            line = fd_data_file.readline()
            line = datautil.PyListUtil.split_data_line(line)
            line_num += 1
        fd_data_file.seek(pos)

        # build header
        self._build_data_header()
        file_data, next_line = self._file_data_to_memory(
            fd_data_file, first_line
        )
        io_file_data = io.StringIO("\n".join(file_data))
        if external_file:
            data_frame = self._try_pandas_read(io_file_data, fd_data_file.name)
            if data_frame is not None:
                self._decrement_id_fields(data_frame)
        else:
            # get number of rows of data
            if len(file_data) > 0:
                data_frame = self._try_pandas_read(
                    io_file_data, fd_data_file.name
                )
                if data_frame is not None:
                    self._decrement_id_fields(data_frame)
                    return_val = [True, fd_data_file.readline()]

        if data_frame is None:
            # read user formatted data using MFList class
            list_data = MFList(
                self._simulation_data,
                self._model_or_sim,
                self.structure,
                None,
                True,
                self.path,
                self.data_dimensions.package_dim,
                self._package,
                self._block,
            )
            # start in original location
            io_file_data.seek(0)
            return_val = list_data.load(
                None, io_file_data, self._block.block_headers[-1]
            )
            rec_array = list_data.get_data()
            if rec_array is not None:
                data_frame = pandas.DataFrame(rec_array)
                data_frame = self._untuple_cellids(data_frame)[0]
                return_val = [True, fd_data_file.readline()]
            else:
                data_frame = None
        return data_frame, return_val, pre_data_comments

    def _save_binary_data(self, fd_data_file, data):
        # write
        file_access = MFFileAccessList(
            self.structure,
            self.data_dimensions,
            self._simulation_data,
            self._path,
            self._current_key,
        )
        file_access.write_binary_file(
            self._dataframe_to_recarray(data),
            fd_data_file,
        )
        data_storage = self._get_storage_obj()
        data_storage.internal_data = None

    def has_data(self, key=None):
        """Returns whether this MFList has any data associated with it."""
        try:
            if self._get_storage_obj() is None:
                return False
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

    def _load_external_data(self, data_storage):
        """loads external data into a panda's dataframe"""
        file_path = self._resolve_ext_file_path(data_storage)
        # parse next line in file as data header
        if data_storage.binary:
            file_access = MFFileAccessList(
                self.structure,
                self.data_dimensions,
                self._simulation_data,
                self._path,
                self._current_key,
            )
            np_data = file_access.read_binary_data_from_file(
                file_path,
                build_cellid=False,
            )
            pd_data = pandas.DataFrame(np_data)
            if "col" in pd_data:
                # keep layer/row/column names consistent
                pd_data = pd_data.rename(columns={"col": "cellid_column"})
            self._decrement_id_fields(pd_data)
        else:
            with open(file_path, "r") as fd_data_file:
                pd_data, return_val, comments = self._read_text_data(
                    fd_data_file, "", True
                )
                data_storage.pre_data_comments = comments
        return pd_data

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
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
        This method was only designed for internal FloPy use and is not
        recommended for end users.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this list.
            file_handle : file descriptor
                A file handle for the data file which points to the second
                line of data for this list
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
        data_storage = self._get_storage_obj()
        data_storage.modified = False
        # parse first line to determine if this is internal or external data
        datautil.PyListUtil.reset_delimiter_used()
        line = datautil.PyListUtil.split_data_line(first_line)
        if line and (
            len(line[0]) >= 2 and line[0][:3].upper() == "END"
        ):
            return [False, line]
        if len(line) >= 2 and line[0].upper() == "OPEN/CLOSE":
            try:
                (
                    data,
                    multiplier,
                    iprn,
                    binary,
                    data_file,
                ) = self._process_open_close_line(line)
            except Exception as ex:
                message = (
                    "An error occurred while processing the following "
                    "open/close line: {}".format(line)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "processing open/close line",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                    ex,
                )
            data_storage.set_external(data_file, data)
            data_storage.binary = binary
            data_storage.iprn = iprn
            return_val = [False, None]
        # else internal
        else:
            # read data into pandas dataframe
            pd_data, return_val, comments = self._read_text_data(
                file_handle, first_line, False
            )
            data_storage.pre_data_comments = comments
            # verify this is the end of the block?

            # store internal data
            data_storage.set_internal(pd_data)
        return return_val

    def _new_storage(self):
        return {"Data": PandasListStorage()}

    def _get_storage_obj(self, first_record=False):
        return self._data_storage["Data"]

    def _get_id_fields(self, data_frame):
        """
        assemble a list of id fields in this dataset

        Parameters
        ----------
        data_frame : DataFrame
            data for this list

        Returns
        -------
        list of column names that are id fields
        """
        id_fields = []
        # loop through the data structure
        for idx, data_item_struct in enumerate(
            self.structure.data_item_structures
        ):
            if data_item_struct.type == DatumType.keystring:
                # handle id fields for keystring
                # ***Code not necessary for this version
                ks_key = data_frame.iloc[0, idx].lower()
                if ks_key in data_item_struct.keystring_dict:
                    data_item_ks = data_item_struct.keystring_dict[ks_key]
                else:
                    ks_key = f"{ks_key}record"
                    if ks_key in data_item_struct.keystring_dict:
                        data_item_ks = data_item_struct.keystring_dict[ks_key]
                    else:
                        continue
                if isinstance(data_item_ks, MFDataStructure):
                    dis = data_item_ks.data_item_structures
                    for data_item in dis:
                        self._update_id_fields(
                            id_fields, data_item, data_frame
                        )
                else:
                    self._update_id_fields(id_fields, data_item_ks, data_frame)
            else:
                self._update_id_fields(id_fields, data_item_struct, data_frame)
        return id_fields

    def _update_id_fields(self, id_fields, data_item_struct, data_frame):
        """
        update the "id_fields" list with new field(s) based on the
        an item in the expected data structure and the data provided.
        """
        if data_item_struct.numeric_index or data_item_struct.is_cellid:
            name = data_item_struct.name.lower()
            if name.startswith("cellid"):
                if isinstance(self._mg, StructuredGrid):
                    id_fields.append(f"{name}_layer")
                    id_fields.append(f"{name}_row")
                    id_fields.append(f"{name}_column")
                elif isinstance(self._mg, VertexGrid):
                    id_fields.append(f"{name}_layer")
                    id_fields.append(f"{name}_cell")
                elif isinstance(self._mg, UnstructuredGrid):
                    id_fields.append(f"{name}_node")
                else:
                    raise MFDataException(
                        "ERROR: Unrecognized model grid "
                        "{str(self._mg)} not supported by MFBasicList"
                    )
            else:
                for col in data_frame.columns:
                    if col.startswith(data_item_struct.name):
                        data_item_len = len(data_item_struct.name)
                        if len(col) > data_item_len:
                            col_end = col[data_item_len:]
                            if (
                                len(col_end) > 1
                                and col_end[0] == "_"
                                and datautil.DatumUtil.is_int(col_end[1:])
                            ):
                                id_fields.append(col)
                            else:
                                id_fields.append(data_item_struct.name)

    def _increment_id_fields(self, data_frame):
        """increment all id fields by 1 (reverse for negative values)"""
        dtypes = data_frame.dtypes
        for id_field in self._get_id_fields(data_frame):
            if id_field in data_frame:
                if id_field in dtypes and dtypes[id_field].str != "<i8":
                    data_frame.astype({id_field: "<i8"})
                data_frame.loc[data_frame[id_field].ge(-1), id_field] += 1
                data_frame.loc[data_frame[id_field].lt(-1), id_field] -= 1

    def _decrement_id_fields(self, data_frame):
        """decrement all id fields by 1 (reverse for negative values)"""
        for id_field in self._get_id_fields(data_frame):
            if id_field in data_frame:
                data_frame.loc[data_frame[id_field].le(-1), id_field] += 1
                data_frame.loc[data_frame[id_field].gt(-1), id_field] -= 1

    def _resolve_ext_file_path(self, data_storage):
        """
        returned the resolved relative path of external file in "data_storage"
        """
        # pathing to external file
        data_dim = self.data_dimensions
        model_name = data_dim.package_dim.model_dim[0].model_name
        fp_relative = data_storage.fname
        if model_name is not None and fp_relative is not None:
            rel_path = self._simulation_data.mfpath.model_relative_path[
                model_name
            ]
            if rel_path is not None and len(rel_path) > 0 and rel_path != ".":
                # include model relative path in external file path
                # only if model relative path is not already in external
                # file path i.e. when reading!
                fp_rp_l = fp_relative.split(os.path.sep)
                rp_l_r = rel_path.split(os.path.sep)[::-1]
                for i, rp in enumerate(rp_l_r):
                    if rp != fp_rp_l[len(rp_l_r) - i - 1]:
                        fp_relative = os.path.join(rp, fp_relative)
            fp = self._simulation_data.mfpath.resolve_path(
                fp_relative, model_name
            )
        else:
            if fp_relative is not None:
                fp = os.path.join(
                    self._simulation_data.mfpath.get_sim_path(), fp_relative
                )
            else:
                fp = self._simulation_data.mfpath.get_sim_path()
        return fp

    def _dataframe_to_recarray(self, data_frame):
        # convert cellids to tuple
        df_rec = self._add_cellid_fields(data_frame, False)

        # convert to recarray
        return df_rec.to_records(index=False)

    def _get_data(self):
        dataframe = self._get_dataframe()
        if dataframe is None:
            return None
        return self._dataframe_to_recarray(dataframe)

    def _get_dataframe(self):
        """get and return dataframe for this list data"""
        data_storage = self._get_storage_obj()
        if data_storage is None or data_storage.data_storage_type is None:
            block_exists = self._block.header_exists(
                self._current_key, self.path
            )
            if block_exists:
                self._build_data_header()
                return pandas.DataFrame(columns=self._header_names)
            else:
                return None
        if data_storage.data_storage_type == DataStorageType.internal_array:
            data = copy.deepcopy(data_storage.internal_data)
        else:
            if data_storage.internal_data is not None:
                # latest data is in internal cache
                data = copy.deepcopy(data_storage.internal_data)
            else:
                # load data from file and return
                data = self._load_external_data(data_storage)
        return data

    def get_dataframe(self):
        """Returns the list's data as a dataframe.

        Returns
        -------
            data : DataFrame

        """
        return self._get_dataframe()

    def get_data(self, apply_mult=False, **kwargs):
        """Returns the list's data as a recarray.

        Parameters
        ----------
            apply_mult : bool
                Whether to apply a multiplier.

        Returns
        -------
            data : recarray

        """
        return self._get_data()

    def get_record(self, data_frame=False):
        """Returns the list's data and metadata in a dictionary.  Data is in
        key "data" and metadata in keys "filename" and "binary".

        Returns
        -------
            data_record : dict

        """
        return self._get_record(data_frame)

    def _get_record(self, data_frame=False):
        """Returns the list's data and metadata in a dictionary.  Data is in
        key "data" and metadata in keys "filename" and "binary".

        Returns
        -------
            data_record : dict

        """
        try:
            if self._get_storage_obj() is None:
                return None
            record = self._get_storage_obj().get_record()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "getting record",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )
        if not data_frame:
            if "data" not in record:
                record["data"] = self._get_data()
            elif record["data"] is not None:
                data = copy.deepcopy(record["data"])
                record["data"] = self._dataframe_to_recarray(data)
        else:
            if "data" not in record:
                record["data"] = self._get_dataframe()
        return record

    def write_file_entry(
        self,
        fd_data_file,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        """
        Writes file entry to file, or if fd_data_file is None returns file
        entry as string.

        Parameters
        ----------
        fd_data_file : file descriptor
            where data is written
        ext_file_action : ExtFileAction
            What action to perform on external files
        fd_main
            file descriptor where open/close string should be written (for
            external file data)

        Returns
        -------
            file entry : str

        """
        return self._write_file_entry(fd_data_file, ext_file_action, fd_main)

    def get_file_entry(
        self,
        ext_file_action=ExtFileAction.copy_relative_paths,
    ):
        """Returns a string containing the data formatted for a MODFLOW 6
        file.

        Parameters
        ----------
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        return self._write_file_entry(None)

    def _write_file_entry(
        self,
        fd_data_file,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        """
        Writes file entry to file, or if fd_data_file is None returns file
        entry as string.

        Parameters
        ----------
        fd_data_file : file descriptor
            Where data is written
        ext_file_action : ExtFileAction
            What action to perform on external files
        fd_main
            file descriptor where open/close string should be written (for
            external file data)
        Returns
        -------
            result of pandas to_csv call
        """
        data_storage = self._get_storage_obj()
        if data_storage is None:
            return ""
        if (
            data_storage.data_storage_type == DataStorageType.external_file
            and fd_main is not None
        ):
            indent = self._simulation_data.indent_string
            ext_string, fname = self._get_external_formatting_str(
                data_storage.fname,
                None,
                data_storage.binary,
                data_storage.iprn,
                DataStructureType.recarray,
                ext_file_action,
            )
            data_storage.fname = fname
            fd_main.write(f"{indent}{indent}{ext_string}")
        if data_storage is None or data_storage.internal_data is None:
            return ""

        # Write out pre-data comments (including headers) like MFList does
        mode = "w"
        if fd_data_file is not None and data_storage.pre_data_comments:
            if hasattr(fd_data_file, "write"):
                fd_data_file.write(data_storage.pre_data_comments.get_file_entry())
            else:
                mode = "a"
                with open(fd_data_file, "w") as f:
                    f.write(data_storage.pre_data_comments.get_file_entry())

        # Loop through data pieces
        data = self._remove_cellid_fields(data_storage.internal_data)
        if (
            data_storage.data_storage_type == DataStorageType.internal_array
            or not data_storage.binary
            or fd_data_file is None
        ):
            # add spacer column
            if "leading_space" not in data:
                data.insert(loc=0, column="leading_space", value="")
            if "leading_space_2" not in data:
                data.insert(loc=0, column="leading_space_2", value="")

        result = ""
        # if data is internal or has been modified
        if (
            data_storage.data_storage_type == DataStorageType.internal_array
            or data is not None
            or fd_data_file is None
        ):
            if (
                data_storage.data_storage_type == DataStorageType.external_file
                and data_storage.binary
                and fd_data_file is not None
            ):
                # write old way using numpy
                self._save_binary_data(fd_data_file, data)
            else:
                if data.shape[0] == 0:
                    if fd_data_file is None or not isinstance(
                        fd_data_file, io.TextIOBase
                    ):
                        result = "\n"
                    else:
                        # no data, just write empty line
                        fd_data_file.write("\n")
                else:
                    # convert data to 1-based
                    self._increment_id_fields(data)
                    # write converted data
                    float_format = (
                        f"%{self._simulation_data.reg_format_str[2:-1]}"
                    )
                    result = data.to_csv(
                        fd_data_file,
                        sep=" ",
                        header=False,
                        index=False,
                        mode=mode,
                        float_format=float_format,
                        lineterminator="\n",
                    )
                    # clean up
                    data_storage.modified = False
                    self._decrement_id_fields(data)
                if (
                    data_storage.data_storage_type
                    == DataStorageType.external_file
                ):
                    data_storage.internal_data = None

        if data_storage.internal_data is not None:
            # clean up
            if "leading_space" in data_storage.internal_data:
                data_storage.internal_data = data_storage.internal_data.drop(
                    columns="leading_space"
                )
            if "leading_space_2" in data_storage.internal_data:
                data_storage.internal_data = data_storage.internal_data.drop(
                    columns="leading_space_2"
                )
        return result

    def _get_file_path(self):
        """
        gets the file path to the data

        Returns
        -------
        file_path : file path to data

        """
        data_storage = self._get_storage_obj()
        if data_storage.fname is None:
            return None
        if self._model_or_sim.type == "model":
            rel_path = self._simulation_data.mfpath.model_relative_path[
                self._model_or_sim.name
            ]
            fp_relative = data_storage.fname
            if rel_path is not None and len(rel_path) > 0 and rel_path != ".":
                # include model relative path in external file path
                # only if model relative path is not already in external
                #  file path i.e. when reading!
                fp_rp_l = fp_relative.split(os.path.sep)
                rp_l_r = rel_path.split(os.path.sep)[::-1]
                for i, rp in enumerate(rp_l_r):
                    if rp != fp_rp_l[len(rp_l_r) - i - 1]:
                        fp_relative = os.path.join(rp, fp_relative)
            return self._simulation_data.mfpath.resolve_path(
                fp_relative, self._model_or_sim.name
            )
        else:
            return os.path.join(
                self._simulation_data.mfpath.get_sim_path(), data_storage.fname
            )

    def plot(
        self,
        key=None,
        names=None,
        filename_base=None,
        file_extension=None,
        mflay=None,
        **kwargs,
    ):
        """
        Plot boundary condition (MfList) data

        Parameters
        ----------
        key : str
            MfList dictionary key. (default is None)
        names : list
            List of names for figure titles. (default is None)
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
        -------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.
        """
        from ...plot import PlotUtilities

        if not self.plottable:
            raise TypeError("Simulation level packages are not plottable")

        if "cellid" not in self.dtype.names:
            return

        PlotUtilities._plot_mflist_helper(
            mflist=self,
            key=key,
            kper=None,
            names=names,
            filename_base=None,
            file_extension=None,
            mflay=None,
            **kwargs,
        )


class MFPandasTransientList(
    MFPandasList, mfdata.MFTransient, DataListInterface
):
    """
    Provides an interface for the user to access and update MODFLOW transient
    pandas list data.

    Parameters
    ----------
    sim_data : MFSimulationData
        data contained in the simulation
    structure : MFDataStructure
        describes the structure of the data
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
        enable=True,
        path=None,
        dimensions=None,
        package=None,
        block=None,
    ):
        super().__init__(
            sim_data=sim_data,
            model_or_sim=model_or_sim,
            structure=structure,
            data=None,
            enable=enable,
            path=path,
            dimensions=dimensions,
            package=package,
            block=block,
        )
        self.repeating = True
        self.empty_keys = {}

    @property
    def data_type(self):
        return DataType.transientlist

    @property
    def dtype(self):
        data = self.get_data()
        if len(data) > 0:
            if 0 in data:
                return data[0].dtype
            else:
                return next(iter(data.values())).dtype
        else:
            return None

    @property
    def plottable(self):
        """If this list data is plottable"""
        if self.model is None:
            return False
        else:
            return True

    @property
    def data(self):
        """Returns list data.  Calls get_data with default parameters."""
        return self.get_data()

    @property
    def dataframe(self):
        """Returns list data.  Calls get_data with default parameters."""
        return self.get_dataframe()

    def to_array(self, kper=0, mask=False):
        """Returns list data as an array."""
        return super().to_array(kper, mask)

    def remove_transient_key(self, transient_key):
        """Remove transient stress period key.  Method is used
        internally by FloPy and is not intended to the end user.

        """
        if transient_key in self._data_storage:
            del self._data_storage[transient_key]

    def add_transient_key(self, transient_key):
        """Adds a new transient time allowing data for that time to be stored
        and retrieved using the key `transient_key`.  Method is used
        internally by FloPy and is not intended to the end user.

        Parameters
        ----------
            transient_key : int
                Zero-based stress period to add

        """
        super().add_transient_key(transient_key)
        self._data_storage[transient_key] = PandasListStorage()

    def store_as_external_file(
        self,
        external_file_path,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Store all data externally in file external_file_path. the binary
        allows storage in a binary file. If replace_existing_external is set
        to False, this method will not do anything if the data is already in
        an external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing
        """
        self._cache_model_grid = True
        for sp in self._data_storage.keys():
            self._current_key = sp
            storage = self._get_storage_obj()
            if storage.internal_size == 0:
                storage.internal_data = self.get_dataframe(sp)
            if storage.internal_size > 0 and (
                self._get_storage_obj().data_storage_type
                != DataStorageType.external_file
                or replace_existing_external
            ):
                fname, ext = os.path.splitext(external_file_path)
                if datautil.DatumUtil.is_int(sp):
                    full_name = f"{fname}_{int(sp) + 1}{ext}"
                else:
                    full_name = f"{fname}_{sp}{ext}"

                super().store_as_external_file(
                    full_name,
                    binary,
                    replace_existing_external,
                    check_data,
                )
        self._cache_model_grid = False

    def store_internal(
        self,
        check_data=True,
    ):
        """Store all data internally.

        Parameters
        ----------
            check_data : bool
                Verify data prior to storing

        """
        self._cache_model_grid = True
        for sp in self._data_storage.keys():
            self._current_key = sp
            if (
                self._get_storage_obj().data_storage_type
                == DataStorageType.external_file
            ):
                super().store_internal(
                    check_data,
                )
        self._cache_model_grid = False

    def has_data(self, key=None):
        """Returns whether this MFList has any data associated with it in key
        "key"."""
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().has_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().has_data()

    def has_modified_ext_data(self, key=None):
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().has_modified_ext_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().has_modified_ext_data()

    def binary_ext_data(self, key=None):
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().binary_ext_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().binary_ext_data()

    def get_record(self, key=None, data_frame=False):
        """Returns the data for stress period `key`.  If no key is specified
        returns all records in a dictionary with zero-based stress period
        numbers as keys.  See MFList's get_record documentation for more
        information on the format of each record returned.

        Parameters
        ----------
            key : int
                Zero-based stress period to return data from.
            data_frame : bool
                whether to return a Pandas DataFrame object instead of a
                recarray
        Returns
        -------
            data_record : dict

        """
        if self._data_storage is not None and len(self._data_storage) > 0:
            if key is None:
                output = {}
                for key in self._data_storage.keys():
                    self.get_data_prep(key)
                    output[key] = super().get_record(data_frame=data_frame)
                return output
            self.get_data_prep(key)
            return super().get_record()
        else:
            return None

    def get_dataframe(self, key=None, apply_mult=False):
        return self.get_data(key, apply_mult, dataframe=True)

    def get_data(self, key=None, apply_mult=False, dataframe=False, **kwargs):
        """Returns the data for stress period `key`.

        Parameters
        ----------
            key : int
                Zero-based stress period to return data from.
            apply_mult : bool
                Apply multiplier
            dataframe : bool
                Get as pandas dataframe

        Returns
        -------
            data : recarray

        """
        if self._data_storage is not None and len(self._data_storage) > 0:
            if key is None:
                if "array" in kwargs:
                    output = []
                    sim_time = self.data_dimensions.package_dim.model_dim[
                        0
                    ].simulation_time
                    num_sp = sim_time.get_num_stress_periods()
                    data = None
                    for sp in range(0, num_sp):
                        if sp in self._data_storage:
                            self.get_data_prep(sp)
                            data = super().get_data(apply_mult=apply_mult)
                        elif self._block.header_exists(sp):
                            data = None
                        output.append(data)
                    return output
                else:
                    output = {}
                    for key in self._data_storage.keys():
                        self.get_data_prep(key)
                        if dataframe:
                            output[key] = super().get_dataframe()
                        else:
                            output[key] = super().get_data(
                                apply_mult=apply_mult
                            )
                    return output
            self.get_data_prep(key)
            if dataframe:
                return super().get_dataframe()
            else:
                return super().get_data(apply_mult=apply_mult)
        else:
            return None

    def set_record(self, record, autofill=False, check_data=True):
        """Sets the contents of the data based on the contents of
        'record`.

        Parameters
        ----------
        record : dict
            Record being set.  Record must be a dictionary with
            keys as zero-based stress periods and values as dictionaries
            containing the data and metadata.  See MFList's set_record
            documentation for more information on the format of the values.
        autofill : bool
            Automatically correct data
        check_data : bool
            Whether to verify the data
        """
        self._set_data_record(
            record,
            autofill=autofill,
            check_data=check_data,
            is_record=True,
        )

    def set_data(self, data, key=None, autofill=False):
        """Sets the contents of the data at time `key` to `data`.

        Parameters
        ----------
        data : dict, recarray, list
            Data being set.  Data can be a dictionary with keys as
            zero-based stress periods and values as the data.  If data is
            a recarray or list of tuples, it will be assigned to the
            stress period specified in `key`.  If any is set to None, that
            stress period of data will be removed.
        key : int
            Zero based stress period to assign data too.  Does not apply
            if `data` is a dictionary.
        autofill : bool
            Automatically correct data.
        """
        self._set_data_record(data, key, autofill)

    def masked_4D_arrays_itr(self):
        """Returns list data as an iterator of a masked 4D array."""
        nper = self.data_dimensions.package_dim.model_dim[
            0
        ].simulation_time.get_num_stress_periods()

        # get the first kper array to extract array shape and names
        arrays_kper_0 = self.to_array(kper=0, mask=True)
        shape_per_spd = next(iter(arrays_kper_0.values())).shape

        for name in arrays_kper_0.keys():
            ma = np.zeros((nper, *shape_per_spd))
            for kper in range(nper):
                # If new_arrays is not None, overwrite arrays
                if new_arrays := self.to_array(kper=kper, mask=True):
                    arrays = new_arrays
                ma[kper] = arrays[name]
            yield name, ma

    def _set_data_record(
        self,
        data_record,
        key=None,
        autofill=False,
        check_data=False,
        is_record=False,
    ):
        self._cache_model_grid = True
        if isinstance(data_record, dict):
            if "filename" not in data_record and "data" not in data_record:
                # each item in the dictionary is a list for one stress period
                # the dictionary key is the stress period the list is for
                del_keys = []
                for key, list_item in data_record.items():
                    list_item_record = False
                    if list_item is None:
                        self.remove_transient_key(key)
                        del_keys.append(key)
                        self.empty_keys[key] = False
                    elif isinstance(list_item, list) and len(list_item) == 0:
                        self.empty_keys[key] = True
                    else:
                        self.empty_keys[key] = False
                        if isinstance(list_item, dict):
                            list_item_record = True
                        self._set_data_prep(list_item, key)
                        if list_item_record:
                            super().set_record(list_item, autofill, check_data)
                        else:
                            super().set_data(
                                list_item,
                                autofill=autofill,
                                check_data=check_data,
                            )
                for key in del_keys:
                    del data_record[key]
            else:
                self.empty_keys[key] = False
                self._set_data_prep(data_record["data"], key)
                super().set_data(data_record, autofill)
        else:
            if is_record:
                comment = (
                    "Set record method requires that data_record is a "
                    "dictionary."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data record",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    self._simulation_data.debug,
                )
            if key is None:
                # search for a key
                new_key_index = self.structure.first_non_keyword_index()
                if (
                    new_key_index is not None
                    and len(data_record) > new_key_index
                ):
                    key = data_record[new_key_index]
                else:
                    key = 0
            if isinstance(data_record, list) and len(data_record) == 0:
                self.empty_keys[key] = True
            else:
                check = True
                if (
                    isinstance(data_record, list)
                    and len(data_record) > 0
                    and data_record[0] == "no_check"
                ):
                    # not checking data
                    check = False
                    data_record = data_record[1:]
                self.empty_keys[key] = False
                if data_record is None:
                    self.remove_transient_key(key)
                else:
                    self._set_data_prep(data_record, key)
                    super().set_data(data_record, autofill, check_data=check)
        self._cache_model_grid = False

    def external_file_name(self, key=0):
        """Returns external file name, or None if this is not external data.

        Parameters
        ----------
            key : int
                Zero based stress period to return data from.
        """
        if key in self.empty_keys and self.empty_keys[key]:
            return None
        else:
            self._get_file_entry_prep(key)
            return super().external_file_name()

    def write_file_entry(
        self,
        fd_data_file,
        key=0,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        """Returns a string containing the data at time `key` formatted for a
        MODFLOW 6 file.

        Parameters
        ----------
            fd_data_file : file
                File to write to
            key : int
                Zero based stress period to return data from.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        if key in self.empty_keys and self.empty_keys[key]:
            return ""
        else:
            self._get_file_entry_prep(key)
            return super().write_file_entry(
                fd_data_file,
                ext_file_action=ext_file_action,
                fd_main=fd_main,
            )

    def get_file_entry(
        self, key=0, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        """Returns a string containing the data at time `key` formatted for a
        MODFLOW 6 file.

        Parameters
        ----------
            key : int
                Zero based stress period to return data from.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        if key in self.empty_keys and self.empty_keys[key]:
            return ""
        else:
            self._get_file_entry_prep(key)
            return super()._write_file_entry(
                None, ext_file_action=ext_file_action
            )

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
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this list.
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

        """
        self._load_prep(block_header)
        return super().load(
            first_line,
            file_handle,
            block_header,
            pre_data_comments,
            external_file_info,
        )

    def append_list_as_record(self, record, key=0):
        """Appends the list `data` as a single record in this list's recarray
        at time `key`.  Assumes `data` has the correct dimensions.

        Parameters
        ----------
            record : list
                Data to append
            key : int
                Zero based stress period to append data too.

        """
        self._append_list_as_record_prep(record, key)
        super().append_list_as_record(record)

    def update_record(self, record, key_index, key=0):
        """Updates a record at index `key_index` and time `key` with the
        contents of `record`.  If the index does not exist update_record
        appends the contents of `record` to this list's recarray.

        Parameters
        ----------
            record : list
                Record to append
            key_index : int
                Index to update
            key : int
                Zero based stress period to append data too

        """

        self._update_record_prep(key)
        super().update_record(record, key_index)

    def _new_storage(self):
        return {}

    def _get_storage_obj(self, first_record=False):
        if first_record and isinstance(self._data_storage, dict):
            for value in self._data_storage.values():
                return value
            return None
        if (
            self._current_key is None
            or self._current_key not in self._data_storage
        ):
            return None
        return self._data_storage[self._current_key]

    def plot(
        self,
        key=None,
        names=None,
        kper=0,
        filename_base=None,
        file_extension=None,
        mflay=None,
        **kwargs,
    ):
        """
        Plot stress period boundary condition (MfList) data for a specified
        stress period

        Parameters
        ----------
        key : str
            MfList dictionary key. (default is None)
        names : list
            List of names for figure titles. (default is None)
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)
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
        -------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.
        """
        from ...plot import PlotUtilities

        if not self.plottable:
            raise TypeError("Simulation level packages are not plottable")

        # model.plot() will not work for a mf6 model oc package unless
        # this check is here
        if self.get_data() is None:
            return

        if "cellid" not in self.dtype.names:
            return

        axes = PlotUtilities._plot_mflist_helper(
            self,
            key=key,
            names=names,
            kper=kper,
            filename_base=filename_base,
            file_extension=file_extension,
            mflay=mflay,
            **kwargs,
        )
        return axes
