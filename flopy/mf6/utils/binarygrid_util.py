"""
Module to read MODFLOW 6 binary grid files (*.grb) that define the model
grid binary output files. The module contains the MfGrdFile class that can
be accessed by the user.

"""

import warnings

import numpy as np

from ...utils.utils_def import FlopyBinaryData

warnings.simplefilter("always", DeprecationWarning)


class MfGrdFile(FlopyBinaryData):
    """
    The MfGrdFile class.

    Parameters
    ----------
    filename : str
        Name of the MODFLOW 6 binary grid file
    precision : string
        'single' or 'double'.  Default is 'double'.
    verbose : bool
        Write information to standard output.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The MfGrdFile class provides simple ways to retrieve data from binary
    MODFLOW 6 binary grid files (.grb). The binary grid file contains data
    that can be used for post processing MODFLOW 6 model results. For
    example, the ia and ja arrays for a model grid.

    Examples
    --------
    >>> import flopy
    >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
    """

    def __init__(self, filename, precision="double", verbose=False):
        """
        Class constructor.

        """

        # Call base class init
        super().__init__()

        # set attributes
        self.precision = precision
        self.verbose = verbose
        self._initial_len = 50
        self._recorddict = {}
        self._datadict = {}
        self._recordkeys = []
        self.filename = filename

        if self.verbose:
            print(f"\nProcessing binary grid file: {filename}")

        # open the grb file
        self.file = open(filename, "rb")

        # grid type
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._grid_type = t[1]

        # version
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._version = t[1]

        # ntxt
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._ntxt = int(t[1])

        # length of text
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._lentxt = int(t[1])

        # read text strings
        for idx in range(self._ntxt):
            line = self.read_text(self._lentxt).strip()
            if line.startswith("#"):
                continue
            t = line.split()
            key = t[0]
            dt = t[1]
            if dt == "INTEGER":
                dtype = np.int32
            elif dt == "SINGLE":
                dtype = np.float32
            elif dt == "DOUBLE":
                dtype = np.float64
            elif dt == "CHARACTER":
                dtype = str
            else:
                dtype = None
            nd = int(t[3])
            if nd > 0:
                shp = [int(v) for v in t[4:]]
                shp = tuple(shp[::-1])
            else:
                shp = (0,)
            self._recorddict[key] = (dtype, nd, shp)
            self._recordkeys.append(key)
            if self.verbose:
                s = ""
                if nd > 0:
                    s = shp
                print(f"  File contains data for {key} with shape {s}")

        if self.verbose:
            print(f"Attempting to read {len(self._recordkeys)} records from {filename}")

        for key in self._recordkeys:
            if self.verbose:
                print(f"  Reading {key}")
            dt, nd, shp = self._recorddict[key]
            # read array data
            if nd > 0:
                count = 1
                for v in shp:
                    count *= v
                if dt == str:
                    v = self.read_text(nchar=count)
                else:
                    v = self.read_record(count=count, dtype=dt)
            # read variable data
            else:
                if dt == np.int32:
                    v = self.read_integer()
                elif dt == np.float32:
                    v = self.read_real()
                elif dt == np.float64:
                    v = self.read_real()
            self._datadict[key] = v

            if self.verbose:
                if nd == 0:
                    print(f"  {key} = {v}")
                else:
                    print(f"  {key}: min = {v.min()} max = {v.max()}")

        # close the file
        self.file.close()

        # initialize the model grid to None
        self._modelgrid = None

        # set ia and ja
        self._set_iaja()

    # internal functions
    def _set_iaja(self):
        """
        Set ia and ja from _datadict.
        """
        self._ia = self._datadict["IA"] - 1
        self._ja = self._datadict["JA"] - 1

    def _set_modelgrid(self):
        """
        Define structured, vertex, or unstructured grid based on MODFLOW 6
        discretization type.

        Returns
        -------
        modelgrid : grid
        """
        from ...discretization.structuredgrid import StructuredGrid
        from ...discretization.unstructuredgrid import UnstructuredGrid
        from ...discretization.vertexgrid import VertexGrid

        modelgrid = None
        idomain = self.idomain
        xorigin = self.xorigin
        yorigin = self.yorigin
        angrot = self.angrot

        try:
            top = self.top
            botm = self.bot

            if self._grid_type == "DISV":
                nlay, ncpl = self.nlay, self.ncpl
                vertices, cell2d = self.cell2d
                top = np.ravel(top)
                botm.shape = (nlay, ncpl)
                modelgrid = VertexGrid(
                    vertices,
                    cell2d,
                    top,
                    botm,
                    idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )

            elif self._grid_type == "DIS":
                nlay, nrow, ncol = (
                    self.nlay,
                    self.nrow,
                    self.ncol,
                )
                delr, delc = self.delr, self.delc

                top.shape = (nrow, ncol)
                botm.shape = (nlay, nrow, ncol)
                modelgrid = StructuredGrid(
                    delc,
                    delr,
                    top,
                    botm,
                    idomain=idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )
            else:
                iverts, verts = self.iverts, self.verts
                vertc = self.cellcenters
                xc, yc = vertc[:, 0], vertc[:, 1]
                modelgrid = UnstructuredGrid(
                    vertices=verts,
                    iverts=iverts,
                    xcenters=xc,
                    ycenters=yc,
                    top=top,
                    botm=botm,
                    idomain=idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )

        except:
            print(f"could not set model grid for {self.file.name}")

        self._modelgrid = modelgrid

        return

    def _build_vertices_cell2d(self):
        """
        Build the mf6 vertices and cell2d array to generate a VertexGrid

        Returns:
        -------
            vertices: list
            cell2d: list
        """
        iverts, verts = self.iverts, self.verts
        vertc = self.cellcenters
        vertices = [[ix] + list(i) for ix, i in enumerate(verts)]
        cell2d = [
            [ix] + list(vertc[ix]) + [len(i) - 1] + i[:-1]
            for ix, i in enumerate(iverts)
        ]
        return vertices, cell2d

    def _get_iverts(self):
        """
        Get a list of the vertices that define each model cell.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.

        """
        iverts = None
        if "IAVERT" in self._datadict:
            iverts = []
            iavert = self.iavert
            javert = self.javert
            nsize = iavert.shape[0] - 1
            for ivert in range(nsize):
                i0 = iavert[ivert]
                i1 = iavert[ivert + 1]
                iverts.append((javert[i0:i1]).tolist())
            if self.verbose:
                print(f"returning iverts from {self.file.name}")
        return iverts

    def _get_verts(self):
        """
        Get a list of the x, y pair for each vertex from the data in the
        binary grid file.

        Returns
        -------
        verts : np.ndarray
            Array with x, y pairs for every vertex used to define the model.

        """
        verts = None
        if "VERTICES" in self._datadict:
            shpvert = self._recorddict["VERTICES"][2]
            verts = self._datadict["VERTICES"].reshape(shpvert)
            if self._grid_type == "DISU":
                # modify verts
                verts = [
                    [idx, verts[idx, 0], verts[idx, 1]] for idx in range(shpvert[0])
                ]
            if self.verbose:
                print(f"returning verts from {self.file.name}")
        return verts

    def _get_cellcenters(self):
        """
        Get the cell centers centroids for a MODFLOW 6 GWF model that uses
        the DISV or DISU discretization.

        Returns
        -------
        vertc : np.ndarray
            Array with x, y pairs of the centroid for every model cell

        """
        xycellcenters = None
        if "CELLX" in self._datadict:
            x = self._datadict["CELLX"]
            y = self._datadict["CELLY"]
            xycellcenters = np.column_stack((x, y))
            if self.verbose:
                print(f"returning cell centers from {self.file.name}")
        return xycellcenters

    # properties
    @property
    def grid_type(self):
        """
        Grid type defined in the MODFLOW 6 grid file.

        Returns
        -------
        grid_type : str
        """
        return self._grid_type

    @property
    def nlay(self):
        """
        Number of layers. None for DISU grids.

        Returns
        -------
        nlay : int
        """
        nlay = None
        if "NLAY" in self._datadict:
            nlay = self._datadict["NLAY"]
        return nlay

    @property
    def nrow(self):
        """
        Number of rows. None for DISV and DISU grids.

        Returns
        -------
        nrow : int
        """
        nrow = None
        if "NROW" in self._datadict:
            nrow = self._datadict["NROW"]
        return nrow

    @property
    def ncol(self):
        """
        Number of columns. None for DISV and DISU grids.

        Returns
        -------
        ncol : int
        """
        ncol = None
        if "NCOL" in self._datadict:
            ncol = self._datadict["NCOL"]
        return ncol

    @property
    def ncpl(self):
        """
        Number of cells per layer. None for DISU grids.

        Returns
        -------
        ncpl : int
        """
        ncpl = None
        if "NCPL" in self._datadict:
            ncpl = self._datadict["NCPL"]
        return ncpl

    @property
    def ncells(self):
        """
        Number of cells.

        Returns
        -------
        ncells : int
        """
        # disu is the only grid that has the number of cells
        # set to nodes.  All other grids use NCELLS in grb
        if "NCELLS" in self._datadict:
            ncells = self._datadict["NCELLS"]
        elif "NODES" in self._datadict:
            ncells = self._datadict["NODES"]
        else:
            ncells = None
        return ncells

    @property
    def nodes(self):
        """
        Number of nodes.

        Returns
        -------
        nodes : int
        """
        nodes = self.ncells
        return nodes

    @property
    def shape(self):
        """
        Shape of the model grid (tuple).

        Returns
        -------
        shape : tuple
        """
        if self._grid_type == "DIS":
            shape = (self.nlay, self.nrow, self.ncol)
        elif self._grid_type == "DIS2D":
            shape = (self.nrow, self.ncol)
        elif self._grid_type == "DISV":
            shape = (self.nlay, self.ncpl)
        elif self._grid_type == "DISV2D":
            shape = (self.ncells,)
        elif self._grid_type == "DISV1D":
            shape = (self.ncells,)
        elif self._grid_type == "DISU":
            shape = (self.nodes,)
        else:
            shape = None
        return shape

    @property
    def xorigin(self):
        """
        x-origin of the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        xorigin : float
        """
        if "XORIGIN" in self._datadict:
            xorigin = self._datadict["XORIGIN"]
        else:
            xorigin = None
        return xorigin

    @property
    def yorigin(self):
        """
        y-origin of the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        yorigin : float
        """
        if "YORIGIN" in self._datadict:
            yorigin = self._datadict["YORIGIN"]
        else:
            yorigin = None
        return yorigin

    @property
    def angrot(self):
        """
        Model grid rotation angle. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        angrot : float
        """
        if "ANGROT" in self._datadict:
            angrot = self._datadict["ANGROT"]
        else:
            angrot = None
        return angrot

    @property
    def idomain(self):
        """
        IDOMAIN for the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        idomain : ndarray of ints
        """
        if "IDOMAIN" in self._datadict:
            idomain = self._datadict["IDOMAIN"]
        else:
            idomain = None
        return idomain

    @property
    def delr(self):
        """
        Cell size in the row direction (y-direction). None if not
        defined in the MODFLOW 6 grid file.

        Returns
        -------
        delr : ndarray of floats
        """
        delr = None
        if "DELR" in self._datadict:
            delr = self._datadict["DELR"]
        return delr

    @property
    def delc(self):
        """
        Cell size in the column direction (x-direction). None if not
        defined in the MODFLOW 6 grid file.

        Returns
        -------
        delc : ndarray of floats
        """
        delc = None
        if "DELC" in self._datadict:
            delc = self._datadict["DELC"]
        return delc

    @property
    def top(self):
        """
        Top of the model cells in the upper model layer for DIS and
        DISV grids. Top of the model cells for DISU grids.

        Returns
        -------
        top : ndarray of floats
        """
        top = None
        if "TOP" in self._datadict:
            top = self._datadict["TOP"]
        return top

    @property
    def bot(self):
        """
        Bottom of the model cells.

        Returns
        -------
        bot : ndarray of floats
        """
        bot = None
        if "BOTM" in self._datadict:
            bot = self._datadict["BOTM"]
        elif "BOT" in self._datadict:
            bot = self._datadict["BOT"]
        return bot

    @property
    def nja(self):
        """
        Number of non-zero entries JA vector array.

        Returns
        -------
        nja : int
        """
        return self._datadict["NJA"]

    @property
    def ia(self):
        """
        index array that defines indexes for `.ja`. Each ia value is the
        starting position of data for a cell. [ia[n]:ia[n+1]] would give you
        all data for a cell. ia[n] is also the location of data for the
        diagonal position. See `.ja` property documentation
        for an example of getting a cell's number and connected cells

        Returns
        -------
        ia : ndarray of ints
        """
        return np.array(self._ia, dtype=int)

    @property
    def ja(self):
        """
        Flat jagged connection array for a model. `.ja` for a cell includes the
        cell number and the cell number for all connected cells. Indexes for
        cells are stored in the `.ia` variable.

        Returns
        -------
        ja : ndarray of ints

        Examples
        --------
        >>> from flopy.mf6.utils import MfGrdFile
        >>> grb = MfGrdFile("my_model.dis.grb")
        >>> ia = grb.ia
        >>> ja = grb.ja
        >>> # get connections for node 0
        >>> ja_node0 = ja[ia[0]:ia[1]]
        >>> node = ja_node0[0]
        >>> connections = ja_node0[1:]
        """
        return self._ja

    @property
    def iavert(self):
        """
        index array that defines indexes for `.javart`. Each ia value is the
        starting position of data for a cell. [iavert[n]:iavert[n+1]] would
        give you all data for a cell. See `.javert` property documentation for
        an example of getting cell number and it's vertex numbers.
        Alternatively, the `.iverts` property can be used to get this
        information

        Returns
        -------
        iavert : ndarray of ints or None for structured grids
        """
        if "IAVERT" in self._datadict:
            iavert = self._datadict["IAVERT"] - 1
        else:
            iavert = None
        return iavert

    @property
    def javert(self):
        """
        Flat jagged array of vertex numbers that comprise all of the cells

        Returns
        -------
        javerts : ndarray of ints or None for structured grids

        Examples
        --------
        >>> from flopy.mf6.utils import MfGrdFile
        >>> grb = MfGrdFile("my_model.dis.grb")
        >>> iavert = self.iavert
        >>> javert = self.javert
        >>> # get vertex numbers for node 0
        >>> vertnums = javert[iavert[0]:iavert[1]]
        """
        if "JAVERT" in self._datadict:
            javert = self._datadict["JAVERT"] - 1
        else:
            javert = None
        return javert

    @property
    def iverts(self):
        """
        Vertex numbers comprising each cell for every cell in model grid.

        Returns
        -------
        iverts : list of lists of ints
        """
        return self._get_iverts()

    @property
    def verts(self):
        """
        x,y location of each vertex that defines the model grid.

        Returns
        -------
        verts : ndarray of floats
        """
        return self._get_verts()

    @property
    def cellcenters(self):
        """
        Cell centers (x,y).

        Returns
        -------
        cellcenters : ndarray of floats
        """
        return self._get_cellcenters()

    @property
    def modelgrid(self):
        """
        Model grid object.

        Returns
        -------
        modelgrid : StructuredGrid, VertexGrid, UnstructuredGrid
        """
        if self._modelgrid is None:
            self._set_modelgrid()
        return self._modelgrid

    @property
    def cell2d(self):
        """
        cell2d data for a DISV grid. None for DIS and DISU grids.

        Returns
        -------
        cell2d : list of lists
        """
        if self._grid_type in ("DISV", "DISV2D", "DISV1D"):
            vertices, cell2d = self._build_vertices_cell2d()
        else:
            vertices, cell2d = None, None
        return vertices, cell2d

    def write(self, filename, precision=None, verbose=False):
        """
        Write the binary grid file to a new file.

        Parameters
        ----------
        filename : str or PathLike
            Path to output .grb file
        precision : str, optional
            'single' or 'double'. If None, uses the precision from the
            original file (default None)
        verbose : bool, optional
            Print progress messages (default False)

        Examples
        --------
        >>> from flopy.mf6.utils import MfGrdFile
        >>> grb = MfGrdFile('model.dis.grb')
        >>> grb.write('model_copy.dis.grb')
        """
        if precision is None:
            precision = self.precision

        # Extract all data from this instance
        data_dict = {
            "NCELLS": self.nodes,
            "NLAY": self.nlay,
            "NROW": self.nrow,
            "NCOL": self.ncol,
            "NJA": self.nja,
            "XORIGIN": self.xorigin,
            "YORIGIN": self.yorigin,
            "ANGROT": self.angrot,
            "DELR": self.delr,
            "DELC": self.delc,
            "TOP": self.top,
            "BOTM": self.bot,
            "IA": self._datadict["IA"],  # Use 1-based from original file
            "JA": self._datadict["JA"],  # Use 1-based from original file
            "IDOMAIN": self.idomain,
        }

        # Add ICELLTYPE if it exists
        if "ICELLTYPE" in self._datadict:
            data_dict["ICELLTYPE"] = self._datadict["ICELLTYPE"]
        else:
            # Provide default if not in original file
            data_dict["ICELLTYPE"] = np.zeros(self.nodes, dtype=np.int32)

        # Call static method
        MfGrdFile.write_grb(
            filename, self.grid_type, data_dict, precision=precision, verbose=verbose
        )

    @staticmethod
    def write_grb(
        filename,
        grid_type,
        data_dict,
        version=1,
        precision="double",
        verbose=False,
    ):
        """
        Write a MODFLOW 6 binary grid file (.grb).

        Parameters
        ----------
        filename : str or PathLike
            Path to output .grb file
        grid_type : str
            Grid type: 'DIS', 'DISV', or 'DISU'
        data_dict : dict
            Dictionary with grid data arrays. Required keys depend on grid_type.
            For DIS grids: NCELLS, NLAY, NROW, NCOL, NJA, XORIGIN, YORIGIN, ANGROT,
                          DELR, DELC, TOP, BOTM, IA, JA, IDOMAIN, ICELLTYPE
        version : int, optional
            Grid file version (default 1)
        precision : str, optional
            'single' or 'double' (default 'double')
        verbose : bool, optional
            Print progress messages (default False)

        Notes
        -----
        The binary grid file format consists of:
        1. Text header lines (50 chars each):
           - "GRID {grid_type}"
           - "VERSION {version}"
           - "NTXT {ntxt}"
           - "LENTXT {lentxt}"
        2. Variable definition lines (100 chars each):
           - "{NAME} {TYPE} NDIM {ndim} {dimensions...}"
        3. Binary data for each variable

        Arrays should be in Python (row-major) order and will be written
        in Fortran (column-major) order as required by MF6.

        Examples
        --------
        >>> import numpy as np
        >>> from flopy.mf6.utils import MfGrdFile
        >>> data = {
        ...     'NCELLS': 800,
        ...     'NLAY': 1,
        ...     'NROW': 40,
        ...     'NCOL': 20,
        ...     'NJA': 3367,
        ...     'XORIGIN': 0.0,
        ...     'YORIGIN': 0.0,
        ...     'ANGROT': 0.0,
        ...     'DELR': np.full(20, 250.0),
        ...     'DELC': np.full(40, 250.0),
        ...     'TOP': np.full(800, 35.0),
        ...     'BOTM': np.random.rand(800),
        ...     'IA': ia_array,
        ...     'JA': ja_array,
        ...     'IDOMAIN': np.ones(800, dtype=np.int32),
        ...     'ICELLTYPE': np.zeros(800, dtype=np.int32)
        ... }
        >>> MfGrdFile.write_grb('model.dis.grb', 'DIS', data)
        """
        import os

        # Create FlopyBinaryData instance for write helpers
        writer = FlopyBinaryData()
        writer.precision = precision

        # Define variable metadata based on grid type
        # Use precision parameter to determine floating point type
        float_type = "SINGLE" if precision.lower() == "single" else "DOUBLE"

        if grid_type.upper() == "DIS":
            var_list = [
                ("NCELLS", "INTEGER", 0, []),
                ("NLAY", "INTEGER", 0, []),
                ("NROW", "INTEGER", 0, []),
                ("NCOL", "INTEGER", 0, []),
                ("NJA", "INTEGER", 0, []),
                ("XORIGIN", float_type, 0, []),
                ("YORIGIN", float_type, 0, []),
                ("ANGROT", float_type, 0, []),
                ("DELR", float_type, 1, [data_dict.get("NCOL", 0)]),
                ("DELC", float_type, 1, [data_dict.get("NROW", 0)]),
                ("TOP", float_type, 1, [data_dict.get("NCELLS", 0)]),
                ("BOTM", float_type, 1, [data_dict.get("NCELLS", 0)]),
                ("IA", "INTEGER", 1, [data_dict.get("NCELLS", 0) + 1]),
                ("JA", "INTEGER", 1, [data_dict.get("NJA", 0)]),
                ("IDOMAIN", "INTEGER", 1, [data_dict.get("NCELLS", 0)]),
                ("ICELLTYPE", "INTEGER", 1, [data_dict.get("NCELLS", 0)]),
            ]
        else:
            raise NotImplementedError(
                f"Grid type {grid_type} not yet implemented. "
                "Currently only DIS grids are supported."
            )

        ntxt = len(var_list)
        lentxt = 100

        if verbose:
            print(f"Writing binary grid file: {filename}")
            print(f"  Grid type: {grid_type}")
            print(f"  Version: {version}")
            print(f"  Number of variables: {ntxt}")

        # Helper function to write text with fixed width
        def write_text(f, text, width):
            """Write text padded to fixed width."""
            text_bytes = text.encode("ascii")
            if len(text_bytes) > width:
                text_bytes = text_bytes[:width]
            else:
                text_bytes = text_bytes.ljust(width)
            f.write(text_bytes)

        with open(filename, "wb") as f:
            # Write text header lines (50 chars each, newline terminated)
            header_len = 50
            write_text(f, f"GRID {grid_type.upper()}\n", header_len)
            write_text(f, f"VERSION {version}\n", header_len)
            write_text(f, f"NTXT {ntxt}\n", header_len)
            write_text(f, f"LENTXT {lentxt}\n", header_len)

            # Write variable definition lines (100 chars each)
            for name, dtype_str, ndim, dims in var_list:
                if ndim == 0:
                    line = f"{name} {dtype_str} NDIM {ndim}\n"
                else:
                    dims_str = " ".join(
                        str(d) for d in dims[::-1]
                    )  # Reverse for Fortran order
                    line = f"{name} {dtype_str} NDIM {ndim} {dims_str}\n"
                write_text(f, line, lentxt)

            # Write binary data for each variable
            for name, dtype_str, ndim, dims in var_list:
                if name not in data_dict:
                    raise ValueError(
                        f"Required variable '{name}' not found in data_dict"
                    )

                value = data_dict[name]

                if verbose:
                    if ndim == 0:
                        print(f"  Writing {name} = {value}")
                    else:
                        if hasattr(value, "min"):
                            print(
                                f"  Writing {name}: min = {value.min()} max = {value.max()}"
                            )
                        else:
                            print(f"  Writing {name}")

                # Write scalar or array data
                if ndim == 0:
                    # Scalar value
                    if dtype_str == "INTEGER":
                        f.write(np.array(int(value), dtype=np.int32).tobytes())
                    elif dtype_str == "DOUBLE":
                        f.write(np.array(float(value), dtype=np.float64).tobytes())
                    elif dtype_str == "SINGLE":
                        f.write(np.array(float(value), dtype=np.float32).tobytes())
                else:
                    # Array data
                    arr = np.asarray(value)
                    if dtype_str == "INTEGER":
                        arr = arr.astype(np.int32)
                    elif dtype_str == "DOUBLE":
                        arr = arr.astype(np.float64)
                    elif dtype_str == "SINGLE":
                        arr = arr.astype(np.float32)

                    # Write array in column-major (Fortran) order
                    f.write(arr.flatten(order="F").tobytes())

        if verbose:
            print(f"Successfully wrote {filename}")


def build_structured_connectivity(nlay, nrow, ncol, idomain=None):
    """
    Build IA and JA connectivity arrays for a structured (DIS) grid.

    Parameters
    ----------
    nlay : int
        Number of layers
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    idomain : np.ndarray, optional
        Domain array indicating active (>0) and inactive (<=0) cells.
        Shape: (nlay, nrow, ncol). If None, all cells are active.

    Returns
    -------
    ia : np.ndarray
        Index array (CSR format), shape (ncells + 1,), dtype int32.
        ia[n] is the starting position in ja for cell n's connections.
        ia[ncells] is the total number of connections.
    ja : np.ndarray
        Connection array (CSR format), shape (nja,), dtype int32.
        Contains cell numbers for each connection (0-based).
    nja : int
        Total number of connections

    Notes
    -----
    Connectivity order for structured grids (upper triangle only):
    1. Diagonal (self connection)
    2. Right (+1 in j, same k, i)
    3. Front (+1 in i, same k, j)
    4. Lower (+1 in k, same i, j)

    The IA/JA arrays use 0-based indexing (Python convention).
    When writing to MF6 binary files, add 1 to convert to Fortran 1-based indexing.

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.mf6.utils import build_structured_connectivity
    >>> nlay, nrow, ncol = 2, 3, 3
    >>> ia, ja, nja = build_structured_connectivity(nlay, nrow, ncol)
    >>> print(f"Total cells: {nlay * nrow * ncol}, connections: {nja}")
    Total cells: 18, connections: 42
    """
    ncells = nlay * nrow * ncol

    # Default to all active cells if idomain not provided
    if idomain is None:
        idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
    else:
        idomain = np.asarray(idomain, dtype=np.int32)
        if idomain.shape != (nlay, nrow, ncol):
            raise ValueError(
                f"idomain shape {idomain.shape} does not match grid shape "
                f"({nlay}, {nrow}, {ncol})"
            )

    ia = np.zeros(ncells + 1, dtype=np.int32)
    ja_list = []
    nja = 0

    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                node = k * nrow * ncol + i * ncol + j

                # Skip inactive cells - they still get an entry in IA
                if idomain[k, i, j] <= 0:
                    ia[node + 1] = nja
                    continue

                # Add diagonal (self connection)
                ja_list.append(node)
                nja += 1

                # Add connections to neighbors (upper triangle only)
                # Right neighbor (j+1)
                if j + 1 < ncol and idomain[k, i, j + 1] > 0:
                    m = k * nrow * ncol + i * ncol + (j + 1)
                    ja_list.append(m)
                    nja += 1

                # Front neighbor (i+1)
                if i + 1 < nrow and idomain[k, i + 1, j] > 0:
                    m = k * nrow * ncol + (i + 1) * ncol + j
                    ja_list.append(m)
                    nja += 1

                # Lower neighbor (k+1)
                if k + 1 < nlay and idomain[k + 1, i, j] > 0:
                    m = (k + 1) * nrow * ncol + i * ncol + j
                    ja_list.append(m)
                    nja += 1

                ia[node + 1] = nja

    ja = np.array(ja_list, dtype=np.int32)

    return ia, ja, nja
