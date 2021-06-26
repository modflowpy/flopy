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
        self.modelgrid = self._set_modelgrid()

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
                xc = vertc[:, 0]
                yc = vertc[:, 1]
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
                    iverts, verts = self._get_verts()
                    vertc = self.get_centroids()
                    xc = vertc[:, 0]
                    yc = vertc[:, 1]
                    sr = SpatialReferenceUnstructured(
                        xc, yc, verts, iverts, [xc.shape[0]]
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
        Build the mf6 vertices and cell2d array
         to generate a VertexGrid

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
                    msg = "returning vertices for {}".format(self.file.name)
                    print(msg)
                return iverts, self._datadict["VERTICES"].reshape(shpvert)
            except:
                msg = "could not return vertices for " + "{}".format(
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
                    msg = "returning vertices for {}".format(self.file.name)
                    print(msg)
                return iverts, verts
            except:
                msg = "could not return vertices for {}".format(self.file.name)
                raise KeyError(msg)
        elif self._grid == "DIS":
            try:
                nlay, nrow, ncol = (
                    self.nlay,
                    self.nrow,
                    self.ncol,
                )
                iv = 0
                verts = []
                iverts = []
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            ivlist = []
                            v = self.modelgrid.get_cell_vertices(i, j)
                            for (x, y) in v:
                                verts.append((x, y))
                                ivlist.append(iv)
                                iv += 1
                            iverts.append(ivlist)
                verts = np.array(verts)
                return iverts, verts
            except:
                msg = "could not return vertices for {}".format(self.file.name)
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
        if self._grid in ("DISV", "DISU"):
            x = self._datadict["CELLX"]
            y = self._datadict["CELLY"]
            xycellcenters = np.column_stack((x, y))
        else:
            xycellcenters = None
        return xycellcenters

    @property
    def nlay(self):
        if self._grid in ("DIS", "DISV"):
            nlay = self._datadict["NLAY"]
        else:
            nlay = None
        return nlay

    @property
    def nrow(self):
        if self._grid == "DIS":
            nrow = self._datadict["NROW"]
        else:
            nrow = None
        return nrow

    @property
    def ncol(self):
        if self._grid == "DIS":
            ncol = self._datadict["NCOL"]
        else:
            ncol = None
        return ncol

    @property
    def ncpl(self):
        if self._grid == "DISV":
            ncpl = self._datadict["NCPL"]
        else:
            ncpl = None
        return ncpl

    @property
    def ncells(self):
        if self._grid in ("DIS", "DISV"):
            ncells = self._datadict["NCELLS"]
        else:
            ncells = self._datadict["NODES"]
        return ncells

    @property
    def nodes(self):
        if self._grid in ("DIS", "DISV"):
            nodes = self.ncells
        else:
            nodes = self._datadict["NODES"]
        return nodes

    @property
    def nconnections(self):
        return self.nja - self.nodes

    @property
    def nja(self):
        return self._datadict["NJA"]

    @property
    def ia(self):
        """
        Zero-based compressed row storage row pointers

        Returns
        -------
        ia : list

        """
        return self._datadict["IA"] - 1

    @property
    def ja(self):
        """
        Zero-based compressed row storage column pointers

        Returns
        -------
        ja : list

        """
        return self._datadict["JA"] - 1

    @property
    def iavert(self):
        """
        Zero-based vertex row pointers

        Returns
        -------
        iavert : list

        """
        return self._datadict["IAVERT"] - 1

    @property
    def javert(self):
        """
        Zero-based vertex numbers for vertices comprising a cell

        Returns
        -------
        iavert : list

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
    def get_modelgrid(self):
        """
        Get the modelgrid based on the MODFLOW 6 discretization type

        Returns
        -------
        modelgrid : Grid object

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> modelgrid = gobj.get_modelgrid()
        """
        return self.modelgrid

    @property
    def get_centroids(self):
        """
        Get the centroids for a MODFLOW 6 GWF model that uses the DIS,
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
                x = self.modelgrid.xcellcenters.flatten()
                y = self.modelgrid.ycellcenters.flatten()
            elif self._grid in ("DIS", "DISV"):
                nlay = self.nlay
                x = np.tile(self.modelgrid.xcellcenters.flatten(), nlay)
                y = np.tile(self.modelgrid.ycellcenters.flatten(), nlay)
            return np.column_stack((x, y))
        except:
            msg = "could not return centroids" + " for {}".format(
                self.file.name
            )
            raise KeyError(msg)

    @property
    def get_spatialreference(self):
        """
        Get the SpatialReference based on the MODFLOW 6 discretization type

        Returns
        -------
        sr : SpatialReference

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> sr = gobj.get_spatialreference()
        """

        warnings.warn(
            "SpatialReference has been deprecated and will be "
            "removed in version 3.3.5. Use get_modelgrid instead.",
            category=DeprecationWarning,
        )

        return self._set_spatialreference()

    def get_frffffflf(self, flowja):
        """
        Get

        Parameters
        ----------
        flowja

        Returns
        -------
        frf : ndarray

        fff : ndarray



        """
        if self._grid == "DIS":
            ia = self.ia
            ja = self.ja
            if flowja.shape != ja.shape:
                raise ValueError(
                    "size of flowja ({}) ".format(flowja.shape)
                    + "not equal to {}".format(ja.shape)
                )
            shape = (self.nlay, self.nrow, self.ncol)
            frf = np.zeros(shape, dtype=float).flatten()
            fff = np.zeros(shape, dtype=float).flatten()
            if self.nlay > 1:
                shapez = (self.nlay - 1, self.nrow, self.ncol)
                flf = np.zeros(shapez, dtype=float)
            else:
                shapez = None
                flf = None
            # fill flow terms
            flows = [frf, fff, flf]
            for n in range(self.nodes):
                i0, i1 = ia[n] + 1, ia[n + 1]
                ipos = 0
                for j in range(i0, i1):
                    jcol = ja[j]
                    if jcol > n:
                        flows[ipos][n] = flowja[j]
                        ipos += 1
            # reshape flow terms
            frf = frf.reshape(shape)
            fff = fff.reshape(shape)
            if shapez is not None:
                flf = flf.reshape(shapez)
        else:
            frf, fff, flf = None, None, None
        return frf, fff, flf
