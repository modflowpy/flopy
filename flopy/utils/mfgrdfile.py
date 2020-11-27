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
        super(MfGrdFile, self).__init__()

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
        self.mg = self._set_modelgrid()

        self.file.close()

    def get_modelgrid(self):
        """
        Get the modelgrid based on the MODFLOW 6 discretization type
        Returns
        -------
        sr : SpatialReference
        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> sr = gobj.get_modelgrid()
        """
        return self.mg

    def _set_modelgrid(self):
        """
        Define structured or unstructured modelgrid based on
        MODFLOW 6 discretization type.
        Returns
        -------
        mg : modelgrid
        """
        mg = None
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
            top, botm = self._datadict["TOP"], self._datadict["BOTM"]

            if self._grid == "DISV":
                nlay, ncpl = self._datadict["NLAY"], self._datadict["NCPL"]
                vertices, cell2d = self._build_vertices_cell2d()
                top = np.ravel(top)
                botm.shape = (nlay, ncpl)
                mg = VertexGrid(
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
                    self._datadict["NLAY"],
                    self._datadict["NROW"],
                    self._datadict["NCOL"],
                )
                delr, delc = self._datadict["DELR"], self._datadict["DELC"]

                top.shape = (nrow, ncol)
                botm.shape = (nlay, nrow, ncol)
                mg = StructuredGrid(
                    delc,
                    delr,
                    top,
                    botm,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )
            else:
                iverts, verts = self.get_verts()
                vertc = self.get_centroids()
                xc = vertc[:, 0]
                yc = vertc[:, 1]
                mg = UnstructuredGrid(
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

        return mg

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
            if self._grid in ["DISV", "DISU"]:
                x = self._datadict["CELLX"]
                y = self._datadict["CELLY"]
            elif self._grid == "DIS":
                nlay = self._datadict["NLAY"]
                x = np.tile(self.mg.xcellcenters.flatten(), nlay)
                y = np.tile(self.mg.ycellcenters.flatten(), nlay)
            return np.column_stack((x, y))
        except:
            msg = "could not return centroids" + " for {}".format(
                self.file.name
            )
            raise KeyError(msg)

    def _build_vertices_cell2d(self):
        """
        Build the mf6 vertices and cell2d array
         to generate a VertexGrid

        Returns:
        -------
            vertices: list
            cell2d: list
        """
        iverts, verts = self.get_verts()
        vertc = self.get_centroids()

        vertices = [[ix] + list(i) for ix, i in enumerate(verts)]
        cell2d = [
            [ix] + list(vertc[ix]) + [len(i) - 1] + i[:-1]
            for ix, i in enumerate(iverts)
        ]
        return vertices, cell2d

    def get_verts(self):
        """
        Get a list of the vertices that define each model cell and the x, y
        pair for each vertex.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.
        verts : np.ndarray
            Array with x, y pairs for every vertex used to define the model.

        Examples
        --------
        >>> import flopy
        >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
        >>> iverts, verts = gobj.get_verts()

        """
        if self._grid == "DISV":
            try:
                iverts = []
                iavert = self._datadict["IAVERT"]
                javert = self._datadict["JAVERT"]
                shpvert = self._recorddict["VERTICES"][2]
                for ivert in range(self._datadict["NCPL"]):
                    i0 = iavert[ivert] - 1
                    i1 = iavert[ivert + 1] - 1
                    iverts.append((javert[i0:i1] - 1).tolist())
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
                iavert = self._datadict["IAVERT"]
                javert = self._datadict["JAVERT"]
                shpvert = self._recorddict["VERTICES"][2]
                for ivert in range(self._datadict["NODES"]):
                    i0 = iavert[ivert] - 1
                    i1 = iavert[ivert + 1] - 1
                    iverts.append((javert[i0:i1] - 1).tolist())
                if self.verbose:
                    msg = "returning vertices for {}".format(self.file.name)
                    print(msg)
                return iverts, self._datadict["VERTICES"].reshape(shpvert)
            except:
                msg = "could not return vertices for {}".format(self.file.name)
                raise KeyError(msg)
        elif self._grid == "DIS":
            try:
                nlay, nrow, ncol = (
                    self._datadict["NLAY"],
                    self._datadict["NROW"],
                    self._datadict["NCOL"],
                )
                iv = 0
                verts = []
                iverts = []
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            ivlist = []
                            v = self.mg.get_cell_vertices(i, j)
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
                    iverts, verts = self.get_verts()
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

        err_msg = (
            "get_spatialreference will be depreciated "
            "get_modelgrid() is replacing it "
        )
        warnings.warn(err_msg, PendingDeprecationWarning)

        return self._set_spatialreference()
