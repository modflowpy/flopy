"""
Module to read MODFLOW 6 binary grid files (*.grb) that define the model
grid binary output files. The module contains the MfGrdFile class that can
be accessed by the user.

"""

import numpy as np
import collections

from ...utils.utils_def import FlopyBinaryData
import warnings

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
        self.set_float(precision=precision)
        self.verbose = verbose
        self._initial_len = 50
        self._recorddict = collections.OrderedDict()
        self._datadict = collections.OrderedDict()
        self._recordkeys = []
        self.filename = filename

        if self.verbose:
            print("\nProcessing binary grid file: {}".format(filename))

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

        # version
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
            t = line.split()
            key = t[0]
            dt = t[1]
            if dt == "INTEGER":
                dtype = np.int32
            elif dt == "SINGLE":
                dtype = np.float32
            elif dt == "DOUBLE":
                dtype = np.float64
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
                msg = "  File contains data for {} ".format(
                    key
                ) + "with shape {}".format(s)
                print(msg)

        if self.verbose:
            msg = "Attempting to read {} ".format(
                self._ntxt
            ) + "records from {}".format(filename)
            print(msg)

        for key in self._recordkeys:
            if self.verbose:
                msg = "  Reading {}".format(key)
                print(msg)
            dt, nd, shp = self._recorddict[key]
            # read array data
            if nd > 0:
                count = 1
                for v in shp:
                    count *= v
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
                    msg = "  {} = {}".format(key, v)
                    print(msg)
                else:
                    msg = "  {}: ".format(key) + "min = {} max = {}".format(
                        v.min(), v.max()
                    )
                    print(msg)

        # close the file
        self.file.close()

        # initialize the model grid to None
        self.__modelgrid = None

        # set ia and ja
        self.__set_iaja()

    # internal functions
    def __set_iaja(self):
        """
        Set ia and ja from _datadict.
        """
        self._ia = self._datadict["IA"] - 1
        self._ja = self._datadict["JA"] - 1

    def __set_modelgrid(self):
        """
        Define structured, vertex, or unstructured grid based on MODFLOW 6
        discretization type.

        Returns
        -------
        modelgrid : grid
        """
        from ...discretization.structuredgrid import StructuredGrid
        from ...discretization.vertexgrid import VertexGrid
        from ...discretization.unstructuredgrid import UnstructuredGrid

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
            print("could not set model grid for {}".format(self.file.name))

        self.__modelgrid = modelgrid

        return

    def __set_spatialreference(self):
        """
        Define structured or unstructured spatial reference based on
        MODFLOW 6 discretization type.
        Returns
        -------
        sr : SpatialReference
        """
        sr = None
        try:
            if self._grid_type in ("DISV", "DISU"):
                from flopy.utils.reference import SpatialReferenceUnstructured

                try:
                    vertc = self.xycentroids()
                    xc = vertc[:, 0]
                    yc = vertc[:, 1]
                    sr = SpatialReferenceUnstructured(
                        xc,
                        yc,
                        self.__modelgrid.verts,
                        self.__modelgrid.iverts,
                        [xc.shape[0]],
                    )
                except:
                    print(
                        "could not set spatial reference for "
                        "{} discretization defined in "
                        "{}".format(self._grid_type, self.file.name)
                    )
            elif self._grid_type == "DIS":
                from flopy.utils.reference import SpatialReference

                delr, delc = self._datadict["DELR"], self._datadict["DELC"]
                xorigin, yorigin, rot = (
                    self._datadict["XORIGIN"],
                    self._datadict["YORIGIN"],
                    self._datadict["ANGROT"],
                )
                sr = SpatialReference(
                    delr=delr,
                    delc=delc,
                    xll=xorigin,
                    yll=yorigin,
                    rotation=rot,
                )
        except:
            print(
                "could not set spatial reference for {}".format(self.file.name)
            )

        return sr

    def __build_vertices_cell2d(self):
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

    def __get_iverts(self):
        """
        Get a list of the vertices that define each model cell.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.

        """
        iverts = None
        if "IAVERT" in self._datadict:
            if self._grid_type == "DISV":
                nsize = self.ncpl
            elif self._grid_type == "DISU":
                nsize = self.nodes
            iverts = []
            iavert = self.iavert
            javert = self.javert
            for ivert in range(nsize):
                i0 = iavert[ivert]
                i1 = iavert[ivert + 1]
                iverts.append((javert[i0:i1]).tolist())
            if self.verbose:
                msg = "returning iverts from {}".format(self.file.name)
                print(msg)
        return iverts

    def __get_verts(self):
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
                msg = "returning verts from {}".format(self.file.name)
                print(msg)
        return verts

    def __get_cellcenters(self):
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
                msg = "returning cell centers from {}".format(self.file.name)
                print(msg)
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
        if self._grid_type in ("DIS", "DISV"):
            nlay = self._datadict["NLAY"]
        else:
            nlay = None
        return nlay

    @property
    def nrow(self):
        """
        Number of rows. None for DISV and DISU grids.

        Returns
        -------
        nrow : int
        """
        if self._grid_type == "DIS":
            nrow = self._datadict["NROW"]
        else:
            nrow = None
        return nrow

    @property
    def ncol(self):
        """
        Number of columns. None for DISV and DISU grids.

        Returns
        -------
        ncol : int
        """
        if self._grid_type == "DIS":
            ncol = self._datadict["NCOL"]
        else:
            ncol = None
        return ncol

    @property
    def ncpl(self):
        """
        Number of cells per layer. None for DISU grids.

        Returns
        -------
        ncpl : int
        """
        if self._grid_type == "DISV":
            ncpl = self._datadict["NCPL"]
        if self._grid_type == "DIS":
            ncpl = self.nrow * self.ncol
        else:
            None
        return ncpl

    @property
    def ncells(self):
        """
        Number of cells.

        Returns
        -------
        ncells : int
        """
        if self._grid_type in ("DIS", "DISV"):
            ncells = self._datadict["NCELLS"]
        else:
            ncells = self._datadict["NODES"]
        return ncells

    @property
    def nodes(self):
        """
        Number of nodes.

        Returns
        -------
        nodes : int
        """
        if self._grid_type in ("DIS", "DISV"):
            nodes = self.ncells
        else:
            nodes = self._datadict["NODES"]
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
        elif self._grid_type == "DISV":
            shape = (self.nlay, self.ncpl)
        else:
            shape = (self.nodes,)
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
        if self.grid_type == "DIS":
            delr = self._datadict["DELR"]
        else:
            delr = None
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
        if self.grid_type == "DIS":
            delc = self._datadict["DELC"]
        else:
            delc = None
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
        return self._datadict["TOP"]

    @property
    def bot(self):
        """
        Bottom of the model cells.

        Returns
        -------
        bot : ndarray of floats
        """
        if self.grid_type in ("DIS", "DISV"):
            bot = self._datadict["BOTM"]
        else:
            bot = self._datadict["BOT"]
        return bot

    @property
    def nja(self):
        """
        Number of non-zero entries in the CRS column pointer vector.

        Returns
        -------
        nja : int
        """
        return self._datadict["NJA"]

    @property
    def ia(self):
        """
        CRS row pointers for the model grid.

        Returns
        -------
        ia : ndarray of ints
        """
        return np.array(self._ia, dtype=int)

    @property
    def ja(self):
        """
        CRS column pointers for the model grid.

        Returns
        -------
        ja : ndarray of ints
        """
        return self._ja

    @property
    def iavert(self):
        """
        CRS cell pointers for cell vertices.

        Returns
        -------
        iavert : ndarray of ints
        """
        if "IAVERT" in self._datadict:
            iavert = self._datadict["IAVERT"] - 1
        else:
            iavert = None
        return iavert

    @property
    def javert(self):
        """
        CRS vertex numbers for the vertices comprising each cell.

        Returns
        -------
        javerts : ndarray of ints
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
        return self.__get_iverts()

    @property
    def verts(self):
        """
        x,y location of each vertex that defines the model grid.

        Returns
        -------
        verts : ndarray of floats
        """
        return self.__get_verts()

    @property
    def cellcenters(self):
        """
        Cell centers (x,y).

        Returns
        -------
        cellcenters : ndarray of floats
        """
        return self.__get_cellcenters()

    @property
    def modelgrid(self):
        """
        Model grid object.

        Returns
        -------
        modelgrid : StructuredGrid, VertexGrid, UnstructuredGrid
        """
        if self.__modelgrid is None:
            self.__set_modelgrid()
        return self.__modelgrid

    @property
    def cell2d(self):
        """
        cell2d data for a DISV grid. None for DIS and DISU grids.

        Returns
        -------
        cell2d : list of lists
        """
        if self._grid_type == "DISV":
            vertices, cell2d = self.__build_vertices_cell2d()
        else:
            vertices, cell2d = None, None
        return vertices, cell2d

    @property
    def spatialreference(self):
        """
        Spatial reference for model grid.

        Returns
        -------
        spatialreference : SpatialReference
        """
        warnings.warn(
            "SpatialReference has been deprecated and will be "
            "removed in version 3.3.5. Use get_modelgrid instead.",
            category=DeprecationWarning,
        )

        return self.__set_spatialreference()
