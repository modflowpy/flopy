import numpy as np
from ..utils.mfenums import DiscretizationType
from ..data.mfstructure import MFStructure


class MFGridException(Exception):
    """
    Model grid related exception
    """

    def __init__(self, error):
        Exception.__init__(self, "MFGridException: {}".format(error))


class ModelCell(object):
    """
    Represents a model cell

    Parameters
    ----------
    cellid : string
        id of model cell

    Methods
    ----------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, cellid):
        self._cellid = cellid


class UnstructuredModelCell(ModelCell):
    """
    Represents an unstructured model cell

    Parameters
    ----------
    cellid : string
        id of model cell
    simulation_data : object
        contains all simulation related data
    model_name : string
        name of the model

    Methods
    ----------
    get_cellid : ()
        returns the cellid
    get_top : ()
        returns the top elevation of the model cell
    get_bot : ()
        returns the bottom elevation of the model cell
    get_area: ()
        returns the area of the model cell
    get_num_connections_iac : ()
        returns the number of connections to/from the model cell
    get_connecting_cells_ja : ()
        returns the cellids of cells connected to this cell
    get_connection_direction_ihc : ()
        returns the connection directions for all connections to this cell
    get_connection_length_cl12 : ()
        returns the connection lengths for all connections to this cell
    get_connection_area_fahl : ()
        returns the connection areas for all connections to this cell
    get_connection_anglex : ()
        returns the connection angles for all connections to this cell
    set_top : (top_elv : float, update_connections : boolean)
        sets the top elevation of the model cell and updates the connection
        properties if update_connections is true
    set_bot : (bot_elv : float, update_connections : boolean)
        sets the bottom elevation of the model cell and updates the connection
        properties if update_connections is true
    set_area : (area : float)
        sets the area of the model cell
    add_connection : (to_cellid, ihc_direction, connection_length,
      connection_area, connection_angle=0)
        adds a connection from this cell to the cell with ID to_cellid
        connection properties ihc_direction, connection_length,
          connection_area, and connection_angle
        are set for the new connection
    remove_connection : (to_cellid)
        removes an existing connection between this cell and the cell with ID
        to_cellid

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, cellid, simulation_data, model_name):
        # init
        self._cellid = cellid
        self._simulation_data = simulation_data
        self._model_name = model_name

    def get_cellid(self):
        return self._cellid

    def get_top(self):
        tops = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "top")
        ]
        return tops[self._cellid - 1]

    def get_bot(self):
        bots = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "bot")
        ]
        return bots[self._cellid - 1]

    def get_area(self):
        areas = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "area")
        ]
        return areas[self._cellid - 1]

    def get_num_connections_iac(self):
        iacs = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "iac")
        ]
        return iacs[self._cellid - 1]

    def get_connecting_cells_ja(self):
        jas = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ja")
        ]
        return jas[self._cellid - 1]

    def get_connection_direction_ihc(self):
        ihc = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ihc")
        ]
        return ihc[self._cellid - 1]

    def get_connection_length_cl12(self):
        cl12 = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "cl12")
        ]
        return cl12[self._cellid - 1]

    def get_connection_area_fahl(self):
        fahl = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "fahl")
        ]
        return fahl[self._cellid - 1]

    def get_connection_anglex(self):
        anglex = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "anglex")
        ]
        return anglex[self._cellid - 1]

    def set_top(self, top_elv, update_connections=True):
        tops = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "top")
        ]
        if update_connections:
            self._update_connections(
                self.get_top(), top_elv, self.get_bot(), self.get_bot()
            )
        tops[self._cellid - 1] = top_elv

    def set_bot(self, bot_elv, update_connections=True):
        bots = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "bot")
        ]
        if update_connections:
            self._update_connections(
                self.get_top(), self.get_top(), self.get_bot(), bot_elv
            )
        bots[self._cellid - 1] = bot_elv

    def set_area(self, area):
        # TODO: Update vertical connection areas
        # TODO: Options for updating horizontal connection lengths???
        areas = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "DISDATA", "area")
        ]
        areas[self._cellid - 1] = area

    def add_connection(
        self,
        to_cellid,
        ihc_direction,
        connection_length,
        connection_area,
        connection_angle=0,
    ):
        iacs = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "iac")
        ]
        jas = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ja")
        ]
        ihc = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ihc")
        ]
        cl12 = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "cl12")
        ]
        fahl = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "fahl")
        ]
        anglex = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "anglex")
        ]

        iacs[self._cellid - 1] += 1
        iacs[to_cellid - 1] += 1
        jas[self._cellid - 1].append(to_cellid)
        jas[to_cellid - 1].append(self._cellid)
        ihc[self._cellid - 1].append(ihc_direction)
        ihc[to_cellid - 1].append(ihc_direction)
        cl12[self._cellid - 1].append(connection_length)
        cl12[to_cellid - 1].append(connection_length)
        fahl[self._cellid - 1].append(connection_area)
        fahl[to_cellid - 1].append(connection_area)
        anglex[self._cellid - 1].append(connection_angle)
        anglex[to_cellid - 1].append(connection_angle)

    def remove_connection(self, to_cellid):
        iacs = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "iac")
        ]
        jas = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ja")
        ]
        ihc = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "ihc")
        ]
        cl12 = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "cl12")
        ]
        fahl = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "fahl")
        ]
        anglex = self._simulation_data.mfdata[
            (self._model_name, "DISU8", "CONNECTIONDATA", "anglex")
        ]

        iacs[self._cellid - 1] -= 1
        iacs[to_cellid - 1] -= 1

        # find connection number
        forward_con_number = self._get_connection_number(to_cellid)
        reverse_con_number = self._get_connection_number(to_cellid, True)

        # update arrays
        del jas[self._cellid - 1][forward_con_number]
        del jas[to_cellid - 1][reverse_con_number]
        del ihc[self._cellid - 1][forward_con_number]
        del ihc[to_cellid - 1][reverse_con_number]
        del cl12[self._cellid - 1][forward_con_number]
        del cl12[to_cellid - 1][reverse_con_number]
        del fahl[self._cellid - 1][forward_con_number]
        del fahl[to_cellid - 1][reverse_con_number]
        del anglex[self._cellid - 1][forward_con_number]
        del anglex[to_cellid - 1][reverse_con_number]

    def _get_connection_number(self, cellid, reverse_connection=False):
        # init
        jas = self._simulation_data.mfdata[
            (self._model_name, "disu8", "connectiondata", "ja")
        ]
        if reverse_connection == False:
            connection_list = jas[self._cellid - 1]
            connecting_cellid = cellid
        else:
            connection_list = jas[cellid - 1]
            connecting_cellid = self._cellid

        # search
        for connection_number, list_cellid in zip(
            range(0, len(connection_list)), connection_list
        ):
            if list_cellid == connecting_cellid:
                return connection_number

    def _update_connections(
        self, old_top_elv, new_top_elv, old_bot_elv, new_bot_elv
    ):
        # TODO: Support connection angles
        # TODO: Support vertically staggered connections
        old_thickness = old_top_elv - old_bot_elv
        new_thickness = new_top_elv - new_bot_elv
        vert_con_diff = (new_thickness - old_thickness) * 0.5
        con_area_mult = new_thickness / old_thickness

        jas = self._simulation_data.mfdata[
            (self._model_name, "disu8", "connectiondata", "ja")
        ]
        ihc = self._simulation_data.mfdata[
            (self._model_name, "disu8", "connectiondata", "ihc")
        ]
        cl12 = self._simulation_data.mfdata[
            (self._model_name, "disu8", "connectiondata", "cl12")
        ]
        fahl = self._simulation_data.mfdata[
            (self._model_name, "disu8", "connectiondata", "fahl")
        ]

        # loop through connecting cells
        for con_number, connecting_cell in zip(
            range(0, len(jas[self._cellid])), jas[self._cellid - 1]
        ):
            rev_con_number = self._get_connection_number(connecting_cell, True)
            if ihc[self._cellid - 1][con_number] == 0:
                # vertical connection, update connection length
                cl12[self._cellid - 1][con_number] += vert_con_diff
                cl12[connecting_cell - 1][rev_con_number] += vert_con_diff
            elif ihc[self._cellid - 1][con_number] == 1:
                # horizontal connection, update connection area
                fahl[self._cellid - 1][con_number] *= con_area_mult
                fahl[connecting_cell - 1][rev_con_number] *= con_area_mult


class ModelGrid(object):
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    model_name : string
        name of the model
    simulation_data : object
        contains all simulation related data
    grid_type : enumeration
        type of model grid (DiscretizationType.DIS, DiscretizationType.DISV,
        DiscretizationType.DISU)

    Methods
    ----------
    grid_type : ()
        returns the grid type
    grid_type_consistent : ()
        returns True if the grid type is consistent with the current
        simulation data
    grid_connections_array : ()
        for DiscretizationType.DISU grids, returns an array containing the
        number of connections of it cell
    get_horizontal_cross_section_dim_arrays : ()
        returns a list of numpy ndarrays sized to the horizontal cross section
        of the model grid
    get_model_dim : ()
        returns the dimensions of the model
    get_model_dim_arrays : ()
        returns a list of numpy ndarrays sized to the model grid
    get_row_array : ()
        returns a numpy ndarray sized to a model row
    get_column_array : ()
        returns a numpy ndarray sized to a model column
    get_layer_array : ()
        returns a numpy ndarray sized to a model layer
    get_horizontal_cross_section_dim_names : ()
        returns the appropriate dimension axis for a horizontal cross section
        based on the model discretization type
    get_model_dim_names : ()
        returns the names of the model dimensions based on the model
        discretization type
    get_num_spatial_coordinates : ()
        returns the number of spatial coordinates based on the model
        discretization type
    num_rows
        returns the number of model rows.  model discretization type must be
        DIS
    num_columns
        returns the number of model columns.  model discretization type must
        be DIS
    num_connections
        returns the number of model connections.  model discretization type
        must be DIS
    num_cells_per_layer
        returns the number of cells per model layer.  model discretization
        type must be DIS or DISV
    num_layers
        returns the number of layers in the model
    num_cells
        returns the total number of cells in the model
    get_all_model_cells
        returns a list of all model cells, represented as a layer/row/column
        tuple, a layer/cellid tuple, or a cellid for the DIS, DISV, and DISU
        discretizations, respectively

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, model_name, simulation_data, grid_type):
        self._model_name = model_name
        self._simulation_data = simulation_data
        self._grid_type = grid_type
        self.freeze_grid = False

    @staticmethod
    def get_grid_type(simulation_data, model_name):
        """
        Return the type of grid used by model 'model_name' in simulation
        containing simulation data 'simulation_data'.

        Parameters
        ----------
        simulation_data : MFSimulationData
            object containing simulation data for a simulation
        model_name : string
            name of a model in the simulation
        Returns
        -------
        grid type : DiscretizationType
        """
        package_recarray = simulation_data.mfdata[
            (model_name, "nam", "packages", "packages")
        ]
        structure = MFStructure()
        if (
            package_recarray.search_data(
                "dis{}".format(structure.get_version_string()), 0
            )
            is not None
        ):
            return DiscretizationType.DIS
        elif (
            package_recarray.search_data(
                "disv{}".format(structure.get_version_string()), 0
            )
            is not None
        ):
            return DiscretizationType.DISV
        elif (
            package_recarray.search_data(
                "disu{}".format(structure.get_version_string()), 0
            )
            is not None
        ):
            return DiscretizationType.DISU
        elif (
            package_recarray.search_data(
                "disl{}".format(structure.get_version_string()), 0
            )
            is not None
        ):
            return DiscretizationType.DISL

        return DiscretizationType.UNDEFINED

    def get_idomain(self):
        if self._grid_type == DiscretizationType.DIS:
            return self._simulation_data.mfdata[
                (self._model_name, "dis", "griddata", "idomain")
            ].get_data()
        elif self._grid_type == DiscretizationType.DISV:
            return self._simulation_data.mfdata[
                (self._model_name, "disv", "griddata", "idomain")
            ].get_data()
        elif self._grid_type == DiscretizationType.DISL:
            return self._simulation_data.mfdata[
                (self._model_name, "disl", "griddata", "idomain")
            ].get_data()
        elif self._grid_type == DiscretizationType.DISU:
            return self._simulation_data.mfdata[
                (self._model_name, "disu", "griddata", "idomain")
            ].get_data()
        except_str = (
            "ERROR: Grid type {} for model {} not "
            "recognized.".format(self._grid_type, self._model_name)
        )
        print(except_str)
        raise MFGridException(except_str)

    def grid_type(self):
        if self.freeze_grid:
            return self._grid_type
        else:
            return self.get_grid_type(self._simulation_data, self._model_name)

    def grid_type_consistent(self):
        return self.grid_type() == self._grid_type

    def get_connections_array(self):
        if self.grid_type() == DiscretizationType.DISU:
            return np.arange(1, self.num_connections() + 1, 1, np.int32)
        else:
            except_str = (
                "ERROR: Can not get connections arrays for model "
                '"{}" Only DISU (unstructured) grids '
                "support connections.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

    def get_horizontal_cross_section_dim_arrays(self):
        if self.grid_type() == DiscretizationType.DIS:
            return [
                np.arange(1, self.num_rows() + 1, 1, np.int32),
                np.arange(1, self.num_columns() + 1, 1, np.int32),
            ]
        elif self.grid_type() == DiscretizationType.DISV:
            return [np.arange(1, self.num_cells_per_layer() + 1, 1, np.int32)]
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            except_str = (
                "ERROR: Can not get horizontal plane arrays for "
                'model "{}" grid.  DISU and DISL grids do not '
                "support individual layers.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

    def get_model_dim(self):
        if self.grid_type() == DiscretizationType.DIS:
            return [self.num_layers(), self.num_rows(), self.num_columns()]
        elif self.grid_type() == DiscretizationType.DISV:
            return [self.num_layers(), self.num_cells_per_layer()]
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            return [self.num_cells()]

    def get_model_dim_arrays(self):
        if self.grid_type() == DiscretizationType.DIS:
            return [
                np.arange(1, self.num_layers() + 1, 1, np.int32),
                np.arange(1, self.num_rows() + 1, 1, np.int32),
                np.arange(1, self.num_columns() + 1, 1, np.int32),
            ]
        elif self.grid_type() == DiscretizationType.DISV:
            return [
                np.arange(1, self.num_layers() + 1, 1, np.int32),
                np.arange(1, self.num_cells_per_layer() + 1, 1, np.int32),
            ]
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            return [np.arange(1, self.num_cells() + 1, 1, np.int32)]

    def get_row_array(self):
        return np.arange(1, self.num_rows() + 1, 1, np.int32)

    def get_column_array(self):
        return np.arange(1, self.num_columns() + 1, 1, np.int32)

    def get_layer_array(self):
        return np.arange(1, self.num_layers() + 1, 1, np.int32)

    def get_horizontal_cross_section_dim_names(self):
        if self.grid_type() == DiscretizationType.DIS:
            return ["row", "column"]
        elif self.grid_type() == DiscretizationType.DISV:
            return ["layer_cell_num"]
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            except_str = (
                "ERROR: Can not get layer dimension name for model "
                '"{}" DISU grid. DISU grids do not support '
                "layers.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

    def get_model_dim_names(self):
        if self.grid_type() == DiscretizationType.DIS:
            return ["layer", "row", "column"]
        elif self.grid_type() == DiscretizationType.DISV:
            return ["layer", "layer_cell_num"]
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            return ["node"]

    def get_num_spatial_coordinates(self):
        if self.grid_type() == DiscretizationType.DIS:
            return 3
        elif self.grid_type() == DiscretizationType.DISV:
            return 2
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            return 1

    def num_rows(self):
        if self.grid_type() != DiscretizationType.DIS:
            except_str = (
                'ERROR: Model "{}" does not have rows.  Can not '
                "return number of rows.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

        return self._simulation_data.mfdata[
            (self._model_name, "dis", "dimensions", "nrow")
        ].get_data()

    def num_columns(self):
        if self.grid_type() != DiscretizationType.DIS:
            except_str = (
                'ERROR: Model "{}" does not have columns.  Can not '
                "return number of columns.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

        return self._simulation_data.mfdata[
            (self._model_name, "dis", "dimensions", "ncol")
        ].get_data()

    def num_connections(self):
        if self.grid_type() == DiscretizationType.DISU:
            return self._simulation_data.mfdata[
                (self._model_name, "disu", "dimensions", "nja")
            ].get_data()
        else:
            except_str = (
                "ERROR: Can not get number of connections for "
                'model "{}" Only DISU (unstructured) grids support '
                "connections.".format(self._model_name)
            )
            print(except_str)
            raise MFGridException(except_str)

    def num_cells_per_layer(self):
        if self.grid_type() == DiscretizationType.DIS:
            return self.num_rows() * self.num_columns()
        elif self.grid_type() == DiscretizationType.DISV:
            return self._simulation_data.mfdata[
                (self._model_name, "disv", "dimensions", "ncpl")
            ].get_data()
        elif self.grid_type() == DiscretizationType.DISU:
            return self._simulation_data.mfdata[
                (self._model_name, "disu", "dimensions", "nodes")
            ].get_data()

    def num_layers(self):
        if self.grid_type() == DiscretizationType.DIS:
            return self._simulation_data.mfdata[
                (self._model_name, "dis", "dimensions", "nlay")
            ].get_data()
        elif self.grid_type() == DiscretizationType.DISV:
            return self._simulation_data.mfdata[
                (self._model_name, "disv", "dimensions", "nlay")
            ].get_data()
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            return None

    def num_cells(self):
        if self.grid_type() == DiscretizationType.DIS:
            return self.num_rows() * self.num_columns() * self.num_layers()
        elif self.grid_type() == DiscretizationType.DISV:
            return self.num_layers() * self.num_cells_per_layer()
        elif self.grid_type() == DiscretizationType.DISU:
            return self._simulation_data.mfdata[
                (self._model_name, "disu", "dimensions", "nodes")
            ].get_data()
        elif self.grid_type() == DiscretizationType.DISL:
            return self._simulation_data.mfdata[
                (self._model_name, "disl", "dimensions", "nodes")
            ].get_data()

    def get_all_model_cells(self):
        model_cells = []
        if self.grid_type() == DiscretizationType.DIS:
            for layer in range(0, self.num_layers()):
                for row in range(0, self.num_rows()):
                    for column in range(0, self.num_columns()):
                        model_cells.append((layer + 1, row + 1, column + 1))
            return model_cells
        elif self.grid_type() == DiscretizationType.DISV:
            for layer in range(0, self.num_layers()):
                for layer_cellid in range(0, self.num_rows()):
                    model_cells.append((layer + 1, layer_cellid + 1))
            return model_cells
        elif (
            self.grid_type() == DiscretizationType.DISU
            or self.grid_type() == DiscretizationType.DISL
        ):
            for node in range(0, self.num_cells()):
                model_cells.append(node + 1)
            return model_cells


class UnstructuredModelGrid(ModelGrid):
    """
    Class for an unstructured model grid

    Parameters
    ----------
    model_name : string
        name of the model
    simulation_data : object
        contains all simulation related data

    Methods
    ----------
    get_unstruct_jagged_array_list : {}
        returns a dictionary of jagged arrays used in the unstructured grid

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, model_name, simulation_data):
        super(UnstructuredModelGrid, self).__init__(
            model_name, simulation_data, DiscretizationType.DISU
        )

    def __getitem__(self, index):
        return UnstructuredModelCell(
            index, self._simulation_data, self._model_name
        )

    @staticmethod
    def get_unstruct_jagged_array_list():
        return {"ihc": 1, "ja": 1, "cl12": 1, "fahl": 1, "anglex": 1}
