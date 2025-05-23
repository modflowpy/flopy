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
                    [idx, verts[idx, 0], verts[idx, 1]]
                    for idx in range(shpvert[0])
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
