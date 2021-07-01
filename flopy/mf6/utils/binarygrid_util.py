"""
Module to read MODFLOW 6 binary grid files (*.grb) that define the model
grid binary output files. The module contains the MfGrdFile class that can
be accessed by the user.

"""

import numpy as np
import collections

from flopy.utils.utils_def import FlopyBinaryData
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.discretization.unstructuredgrid import UnstructuredGrid
from flopy.utils.reference import SpatialReferenceUnstructured
from flopy.utils.reference import SpatialReference
import warnings

warnings.simplefilter("always", PendingDeprecationWarning)


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
        Write information to the screen.  Default is False.

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
    that can be used for post processing MODFLOW 6 model results.

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

        if self.verbose:
            print("\nProcessing binary grid file: {}".format(filename))

        # open the grb file
        self.file = open(filename, "rb")

        # grid type
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._grid = t[1]

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

        # set the model grid
        self._modelgrid = self._set_modelgrid()

        # set iverts and verts
        self._iverts, self._verts = self._get_verts()

        self.file.close()

    def _set_modelgrid(self):
        """
        Define structured, vertex, or unstructured grid based on MODFLOW 6
        discretization type.

        Returns
        -------
        modelgrid : grid
        """
        modelgrid = None
        idomain = None
        xorigin = None
        yorigin = None
        angrot = None
        if "IDOMAIN" in self._datadict:
            idomain = self._datadict["IDOMAIN"]

        if "XORIGIN" in self._datadict:
            xorigin = self._datadict["XORIGIN"]

        if "YORIGIN" in self._datadict:
            yorigin = self._datadict["YORIGIN"]

        if "ANGROT" in self._datadict:
            angrot = self._datadict["ANGROT"]

        try:
            if self._grid in ("DIS", "DISV"):
                top, botm = self._datadict["TOP"], self._datadict["BOTM"]
            else:
                top, botm = self._datadict["TOP"], self._datadict["BOT"]

            if self._grid == "DISV":
                nlay, ncpl = self.nlay, self.ncpl
                vertices, cell2d = self._build_vertices_cell2d()
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

            elif self._grid == "DIS":
                nlay, nrow, ncol = (
                    self.nlay,
                    self.nrow,
                    self.ncol,
                )
                delr, delc = self._datadict["DELR"], self._datadict["DELC"]

                top.shape = (nrow, ncol)
                botm.shape = (nlay, nrow, ncol)
                modelgrid = StructuredGrid(
                    delc,
                    delr,
                    top,
                    botm,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )
            else:
                iverts, verts = self._get_verts()
                vertc = self._get_cellcenters()
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

        return modelgrid

    def _set_spatialreference(self):
        """
        Define structured or unstructured spatial reference based on
        MODFLOW 6 discretization type.
        Returns
        -------
        sr : SpatialReference
        """
        sr = None
        try:
            if self._grid == "DISV" or self._grid == "DISU":
                try:
                    vertc = self.get_centroids()
                    xc = vertc[:, 0]
                    yc = vertc[:, 1]
                    sr = SpatialReferenceUnstructured(
                        xc, yc, self._verts, self._iverts, [xc.shape[0]]
                    )
                except:
                    msg = (
                        "could not set spatial reference for "
                        + "{} discretization ".format(self._grid)
                        + "defined in {}".format(self.file.name)
                    )
                    print(msg)
            elif self._grid == "DIS":
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

    def _build_vertices_cell2d(self):
        """
        Build the mf6 vertices and cell2d array to generate a VertexGrid

        Returns:
        -------
            vertices: list
            cell2d: list
        """
        iverts, verts = self._get_verts()
        vertc = self._get_cellcenters()
        vertices = [[ix] + list(i) for ix, i in enumerate(verts)]
        cell2d = [
            [ix] + list(vertc[ix]) + [len(i) - 1] + i[:-1]
            for ix, i in enumerate(iverts)
        ]
        return vertices, cell2d

    def _build_structured_iverts(self, i, j):
        """
        Build list of vertex numbers for a cell in a model with a structured
        grid

        Parameters
        ----------
        i : int
            row index
        j : int
            column indes

        Returns
        -------
        iv_list : list
            list of vertex number for a cell in a structured model

        """
        iv_list = [i * (self.ncol + 1) + j]
        iv_list.append(i * (self.ncol + 1) + j + 1)
        iv_list.append((i + 1) * (self.ncol + 1) + j + 1)
        iv_list.append((i + 1) * (self.ncol + 1) + j)
        return iv_list

    def _get_verts(self):
        """
        Get a list of the vertices that define each model cell and the x, y
        pair for each vertex from the data in the binary grid file.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.
        verts : np.ndarray
            Array with x, y pairs for every vertex used to define the model.

        """
        if self._grid == "DISV":
            try:
                iverts = []
                iavert = self.iavert
                javert = self.javert
                shpvert = self._recorddict["VERTICES"][2]
                for ivert in range(self.ncpl):
                    i0 = iavert[ivert]
                    i1 = iavert[ivert + 1]
                    iverts.append((javert[i0:i1]).tolist())
                if self.verbose:
                    msg = "returning vertices from {}".format(self.file.name)
                    print(msg)
                return iverts, self._datadict["VERTICES"].reshape(shpvert)
            except:
                msg = "could not return vertices from " + "{}".format(
                    self.file.name
                )
                raise KeyError(msg)
        elif self._grid == "DISU":
            try:
                iverts = []
                iavert = self.iavert
                javert = self.javert
                shpvert = self._recorddict["VERTICES"][2]
                # create iverts
                for ivert in range(self.nodes):
                    i0 = iavert[ivert]
                    i1 = iavert[ivert + 1]
                    iverts.append((javert[i0:i1]).tolist())
                v0 = self._datadict["VERTICES"].reshape(shpvert)
                # create verts
                verts = [
                    [idx, v0[idx, 0], v0[idx, 1]] for idx in range(shpvert[0])
                ]
                if self.verbose:
                    msg = "returning vertices from {}".format(self.file.name)
                    print(msg)
                return iverts, verts
            except:
                msg = "could not get vertices from {}".format(self.file.name)
                print(msg)
                return None, None
        elif self._grid == "DIS":
            try:
                nrow, ncol = (
                    self.nrow,
                    self.ncol,
                )
                iverts = []
                for i in range(nrow):
                    for j in range(ncol):
                        iverts.append(self._build_structured_iverts(i, j))
                x = self._modelgrid.get_xvertices_for_layer(0).flatten()
                y = self._modelgrid.get_yvertices_for_layer(0).flatten()
                return iverts, np.column_stack((x, y))
            except:
                msg = "could not get vertices from {}".format(self.file.name)
                raise KeyError(msg)
        return

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
        if self._grid in ("DISV", "DISU"):
            try:
                x = self._datadict["CELLX"]
                y = self._datadict["CELLY"]
                xycellcenters = np.column_stack((x, y))
            except:
                msg = "could not get cell centers from {}".format(
                    self.file.name
                )
                print(msg)
        return xycellcenters

    @property
    def nlay(self):
        """
        Return the number of layers in a structured and vertex grid.
        None is returned for an unstructured grid.

        Returns
        -------
        nrow : int
            number of layers

        """
        if self._grid in ("DIS", "DISV"):
            nlay = self._datadict["NLAY"]
        else:
            nlay = None
        return nlay

    @property
    def nrow(self):
        """
        Return the number of rows in a structured grid. None is returned
        for vertex and unstructured grids.

        Returns
        -------
        nrow : int
            number of rows

        """
        if self._grid == "DIS":
            nrow = self._datadict["NROW"]
        else:
            nrow = None
        return nrow

    @property
    def ncol(self):
        """
        Return the number of columns in a structured grid. None is returned
        for vertex and unstructured grids.

        Returns
        -------
        ncol : int
            number of columns

        """
        if self._grid == "DIS":
            ncol = self._datadict["NCOL"]
        else:
            ncol = None
        return ncol

    @property
    def ncpl(self):
        """
        Return the number of cells per layer in a structured and vertex
        grid. None is returned for an unstructured model.

        Returns
        -------
        ncpl : int
            number of cells per layer

        """
        if self._grid == "DISV":
            ncpl = self._datadict["NCPL"]
        if self._grid == "DIS":
            ncpl = self.nrow * self.ncol
        else:
            None
        return ncpl

    @property
    def ncells(self):
        """
        Return the number of cells in a grid.

        Returns
        -------
        ncells : int
            number of cells in a grid

        """
        if self._grid in ("DIS", "DISV"):
            ncells = self._datadict["NCELLS"]
        else:
            ncells = self._datadict["NODES"]
        return ncells

    @property
    def nodes(self):
        """
        Return the number of cells in a grid.

        Returns
        -------
        nodes : int
            number of nodes in a grid

        """
        if self._grid in ("DIS", "DISV"):
            nodes = self.ncells
        else:
            nodes = self._datadict["NODES"]
        return nodes

    @property
    def nconnections(self):
        """
        Return the number of intercell connections in a grid.

        Returns
        -------
        nconnections : int
            number of cells intercell connections

        """
        return self.nja - self.nodes

    @property
    def nja(self):
        """
        Return the number of entries in the compressed row storage column
        pointer vector.

        Returns
        -------
        nja : int
            number of entries in the JA vector

        """
        return self._datadict["NJA"]

    @property
    def ia(self):
        """
        Zero-based compressed row storage row pointers

        Returns
        -------
        ia : list
            compressed row storage row pointers

        """
        return self._datadict["IA"] - 1

    @property
    def ja(self):
        """
        Zero-based compressed row storage column pointers

        Returns
        -------
        ja : list
            compressed row storage column pointers

        """
        return self._datadict["JA"] - 1

    @property
    def nvert(self):
        """
        Number of vertices that define a grid.

        Returns
        -------
        nvert : int
            number of vertices

        """
        if self._grid == "DIS":
            nvert = (self.nrow + 1) * (self.ncol + 1)
        elif self._grid == "DISV":
            nvert = self._datadict["NVERT"]
        else:
            try:
                nvert = len(self._verts)
            except:
                nvert = None
        return nvert

    @property
    def iavert(self):
        """
        Zero-based vertex row pointers

        Returns
        -------
        iavert : list
            row pointers for javert entries for each cell

        """
        return self._datadict["IAVERT"] - 1

    @property
    def javert(self):
        """
        Zero-based vertex numbers for vertices comprising a cell

        Returns
        -------
        iavert : list
            vertex numbers that define each cell

        """
        return self._datadict["JAVERT"] - 1

    @property
    def connectivity(self):
        """
        Return a list containing the zero-based node number and the
        zero-based cell numbers of cells connected to the cell.

        Returns
        -------
        connectivity : list
            node numbers for each cell and node numbers for all connected cells

        """
        ia = self.ia
        ja = self.ja
        connectivity = []
        for n in range(self.ncells):
            i0, i1 = ia[n] + 1, ia[n + 1]
            tlist = [n]
            for j in range(i0, i1):
                tlist.append(ja[j])
            connectivity.append(tlist)
        return connectivity

    @property
    def cellconnections(self):
        """
        Return a numpy recarray of all cell connections. Columns have
        zero-based n and m cell numbers for each connection. All
        connections are returned.

        Returns
        -------
        cellconnections : structured ndarray
            n and m cell numbers

        """
        cellconnections = np.zeros(
            self.nconnections, dtype=[("n", int), ("m", int)]
        )
        ia, ja = self.ia, self.ja
        idx = 0
        for n in range(self.nodes):
            i0, i1 = ia[n] + 1, ia[n + 1]
            for j in range(i0, i1):
                cellconnections["n"][idx] = n
                cellconnections["m"][idx] = ja[j]
                idx += 1
        return cellconnections

    @property
    def modelgrid(self):
        """
        Return the modelgrid based on the MODFLOW 6 discretization type

        Returns
        -------
        modelgrid : Grid object

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> modelgrid = gobj.modelgrid
        """
        return self._modelgrid

    @property
    def get_centroids(self):
        """
        Return the centroids for a MODFLOW 6 GWF model that uses the DIS,
        DISV, or DISU discretization.

        Returns
        -------
        vertc : np.ndarray
            Array with x, y pairs of the centroid for every model cell

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> vertc = gobj.get_centroids()

        """
        try:
            if self._grid == "DISU":
                x = self._modelgrid.xcellcenters.flatten()
                y = self._modelgrid.ycellcenters.flatten()
            elif self._grid in ("DIS", "DISV"):
                nlay = self.nlay
                x = np.tile(self._modelgrid.xcellcenters.flatten(), nlay)
                y = np.tile(self._modelgrid.ycellcenters.flatten(), nlay)
            return np.column_stack((x, y))
        except:
            msg = "could not return centroids" + " for {}".format(
                self.file.name
            )
            raise KeyError(msg)

    @property
    def spatialreference(self):
        """
        Get the SpatialReference based on the MODFLOW 6 discretization type

        Returns
        -------
        sr : SpatialReference

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> sr = gobj.spatialreference()
        """

        warnings.warn(
            "SpatialReference has been deprecated and will be "
            "removed in version 3.3.5. Use get_modelgrid instead.",
            category=DeprecationWarning,
        )

        return self._set_spatialreference()

    @property
    def vertices(self):
        """
        Get a list of the vertices that define each model cell and the x, y
        pair for each vertex from the data in the binary grid file.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.
        verts : np.ndarray
            Array with x, y pairs for every vertex used to define the model.

        """
        return self._iverts, self._verts

    def get_faceflows(self, flowja):
        """
        Get the face flows for the flow right face, flow front face, and
        flow lower face from the MODFLOW 6 flowja flows. This method can
        be useful for building face flow arrays for MT3DMS, MT3D-USGS, and
        RT3D. This method only works for a structured MODFLOW 6 model.

        Parameters
        ----------
        flowja : ndarray
            flowja array for a structured MODFLOW 6 model

        Returns
        -------
        frf : ndarray
            right face flows
        fff : ndarray
            front face flows
        flf : ndarray
            lower face flows

        """
        if self._grid == "DIS":
            ia = self.ia
            ja = self.ja
            if len(flowja.shape) > 0:
                flowja = flowja.flatten()
            if flowja.shape != ja.shape:
                raise ValueError(
                    "size of flowja ({}) ".format(flowja.shape)
                    + "not equal to {}".format(ja.shape)
                )
            shape = (self.nlay, self.nrow, self.ncol)
            frf = np.zeros(shape, dtype=float).flatten()
            fff = np.zeros(shape, dtype=float).flatten()
            flf = np.zeros(shape, dtype=float)
            # fill flow terms
            vmult = [-1.0, -1.0, -1.0]
            flows = [frf, fff, flf]
            for n in range(self.nodes):
                i0, i1 = ia[n] + 1, ia[n + 1]
                ipos = 0
                for j in range(i0, i1):
                    jcol = ja[j]
                    if jcol > n:
                        flows[ipos][n] = vmult[ipos] * flowja[j]
                        ipos += 1
            # reshape flow terms
            frf = frf.reshape(shape)
            fff = fff.reshape(shape)
            flf = flf.reshape(shape)
        else:
            frf, fff, flf = None, None, None
        return frf, fff, flf
