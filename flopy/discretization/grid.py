import copy
import os
import re
import warnings
from collections import defaultdict

import numpy as np

try:
    import pyproj

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from ..utils import geometry
from ..utils.crs import get_crs
from ..utils.gridutil import get_lni


class CachedData:
    def __init__(self, data):
        self._data = data
        self.out_of_date = False

    @property
    def data_nocopy(self):
        return self._data

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def update_data(self, data):
        self._data = data
        self.out_of_date = False


def _get_epsg_from_crs_or_proj4(crs, proj4=None):
    """Try to get EPSG identifier from a crs object."""
    if isinstance(crs, int):
        return crs
    if isinstance(crs, str):
        if match := re.findall(r"epsg:([\d]+)", crs, re.IGNORECASE):
            return int(match[0])
    if proj4 and isinstance(proj4, str):
        if match := re.findall(r"epsg:([\d]+)", proj4, re.IGNORECASE):
            return int(match[0])


class Grid:
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex', 'unstructured')
    top : float or ndarray
        top elevations of cells in topmost layer
    botm : float or ndarray
        bottom elevations of all cells
    idomain : int or ndarray
        ibound/idomain value for each cell
    lenuni : int or ndarray
        model length units
    crs : pyproj.CRS, int, str, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    prjfile : str or pathlike, optional if `crs` is specified
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).
    xoff : float
        x coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    yoff : float
        y coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    angrot : float
        rotation angle of model grid, as it is rotated around the origin point
    **kwargs : dict, optional
        Support deprecated keyword options.

        .. deprecated:: 3.5
           The following keyword options will be removed for FloPy 3.6:

             - ``prj`` (str or pathlike): use ``prjfile`` instead.
             - ``epsg`` (int): use ``crs`` instead.
             - ``proj4`` (str): use ``crs`` instead.

    Attributes
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex', 'unstructured')
    top : float or ndarray
        top elevations of cells in topmost layer
    botm : float or ndarray
        bottom elevations of all cells
    idomain : int or ndarray
        ibound/idomain value for each cell
    lenuni : int
        modflow lenuni parameter
    xoffset : float
        x coordinate of the origin point in the spatial reference coordinate
        system
    yoffset : float
        y coordinate of the origin point in the spatial reference coordinate
        system
    angrot : float
        rotation angle of model grid, as it is rotated around the origin point
    angrot_radians : float
        rotation angle of model grid, in radians
    xcenters : ndarray
        returns x coordinate of cell centers
    ycenters : ndarray
        returns y coordinate of cell centers
    ycenters : ndarray
        returns z coordinate of cell centers
    xyzcellcenters : [ndarray, ndarray, ndarray]
        returns the cell centers of all model cells in the model grid.  if
        the model grid contains spatial reference information, the cell centers
        are in the coordinate system provided by the spatial reference
        information. otherwise the cell centers are based on a 0,0 location
        for the upper left corner of the model grid. returns a list of three
        ndarrays for the x, y, and z coordinates

    Methods
    ----------
    get_coords(x, y)
        transform point or array of points x, y from model coordinates to
        spatial coordinates
    grid_lines : (point_type=PointType.spatialxyz) : list
        returns the model grid lines in a list.  each line is returned as a
        list containing two tuples in the format [(x1,y1), (x2,y2)] where
        x1,y1 and x2,y2 are the endpoints of the line.

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """

    def __init__(
        self,
        grid_type=None,
        top=None,
        botm=None,
        idomain=None,
        lenuni=None,
        crs=None,
        prjfile=None,
        xoff=0.0,
        yoff=0.0,
        angrot=0.0,
        **kwargs,
    ):
        lenunits = {0: "undefined", 1: "feet", 2: "meters", 3: "centimeters"}
        LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}
        self.use_ref_coords = True
        self._grid_type = grid_type
        if top is not None:
            top = top.astype(float)
        self._top = top
        if botm is not None:
            botm = botm.astype(float)
        self._botm = botm
        self._idomain = idomain

        if lenuni is None:
            lenuni = 0
        elif isinstance(lenuni, str):
            lenuni = LENUNI[lenuni.lower()[0]]
        self._lenuni = lenuni

        self._units = lenunits[self._lenuni]
        # Handle deprecated projection kwargs; warnings are raised in crs.py
        self._crs = None
        get_crs_args = {"crs": crs, "prjfile": prjfile}
        if "epsg" in kwargs:
            self.epsg = get_crs_args["epsg"] = kwargs.pop("epsg")
        if "proj4" in kwargs:
            self.proj4 = get_crs_args["proj4"] = kwargs.pop("proj4")
        if "prj" in kwargs:
            self.prjfile = get_crs_args["prj"] = kwargs.pop("prj")
        if kwargs:
            raise TypeError(f"unhandled keywords: {kwargs}")
        if prjfile is not None:
            self.prjfile = prjfile
        if HAS_PYPROJ:
            self._crs = get_crs(**get_crs_args)
        elif crs is not None:
            # provide some support without pyproj
            if isinstance(crs, str) and self.proj4 is None:
                self._proj4 = crs
            if self.epsg is None:
                if epsg := _get_epsg_from_crs_or_proj4(crs, self.proj4):
                    self._epsg = epsg
        self._prjfile = prjfile
        self._xoff = xoff
        self._yoff = yoff
        if angrot is None:
            angrot = 0.0
        self._angrot = angrot
        self._polygons = None
        self._cache_dict = {}
        self._copy_cache = True

        self._iverts = None
        self._verts = None
        self._laycbd = None
        self._neighbors = None
        self._edge_set = None

    ###################################
    # access to basic grid properties
    ###################################
    def __repr__(self):
        items = []
        if (
            self.xoffset is not None
            and self.yoffset is not None
            and self.angrot is not None
        ):
            items += [
                f"xll:{self.xoffset!s}",
                f"yll:{self.yoffset!s}",
                f"rotation:{self.angrot!s}",
            ]
        if self.crs is not None:
            items.append(f"crs:{self.crs.srs}")
        elif self.epsg is not None:
            items.append(f"crs:EPSG:{self.epsg}")
        elif self.proj4 is not None:
            items.append(f"proj4_str:{self.proj4}")
        if self.units is not None:
            items.append(f"units:{self.units}")
        if self.lenuni is not None:
            items.append(f"lenuni:{self.lenuni}")
        return "; ".join(items)

    @property
    def is_valid(self):
        return True

    @property
    def is_complete(self):
        if (
            self._top is not None
            and self._botm is not None
            and self._idomain is not None
        ):
            return True
        return False

    @property
    def grid_type(self):
        return self._grid_type

    @property
    def xoffset(self):
        return self._xoff

    @property
    def yoffset(self):
        return self._yoff

    @property
    def angrot(self):
        return self._angrot

    @property
    def angrot_radians(self):
        return self._angrot * np.pi / 180.0

    @property
    def crs(self):
        """Coordinate reference system (CRS) for the model grid.

        If pyproj is not installed this property is always None;
        see :py:attr:`epsg` for an alternative CRS definition.
        """
        return self._crs

    @crs.setter
    def crs(self, crs):
        if crs is None:
            self._crs = None
            return
        if HAS_PYPROJ:
            self._crs = get_crs(crs=crs)
        else:
            warnings.warn(
                "cannot set 'crs' property without pyproj; "
                "try setting 'epsg' or 'proj4' instead",
                UserWarning,
            )
            self._crs = None

    @property
    def epsg(self):
        """EPSG integer code registered to a coordinate reference system.

        This property is derived from :py:attr:`crs` if pyproj is installed,
        otherwise it preserved from the constructor.
        """
        epsg = None
        if hasattr(self, "_epsg"):
            epsg = self._epsg
        elif self._crs is not None:
            epsg = self._crs.to_epsg()
            if epsg is not None:
                self._epsg = epsg
        return epsg

    @epsg.setter
    def epsg(self, epsg):
        if not (isinstance(epsg, int) or epsg is None):
            raise ValueError("epsg property must be an int or None")
        self._epsg = epsg
        # If crs was previously unset, use EPSG code
        if HAS_PYPROJ and self._crs is None and epsg is not None:
            self._crs = get_crs(crs=epsg)

    @property
    def proj4(self):
        """PROJ string for a coordinate reference system.

        This property is derived from :py:attr:`crs` if pyproj is installed,
        otherwise it preserved from the constructor.
        """
        proj4 = None
        if hasattr(self, "_proj4"):
            proj4 = self._proj4
        elif self._crs is not None:
            proj4 = self._crs.to_proj4()
            if proj4 is not None:
                self._proj4 = proj4
        return proj4

    @proj4.setter
    def proj4(self, proj4):
        if not (isinstance(proj4, str) or proj4 is None):
            raise ValueError("proj4 property must be a str or None")
        self._proj4 = proj4
        # If crs was previously unset, use lossy PROJ string
        if HAS_PYPROJ and self._crs is None and proj4 is not None:
            self._crs = get_crs(crs=proj4)

    @property
    def prj(self):
        warnings.warn(
            "prj property is deprecated, use prjfile instead",
            PendingDeprecationWarning,
        )
        return self.prjfile

    @prj.setter
    def prj(self, prj):
        warnings.warn(
            "prj property is deprecated, use prjfile instead",
            PendingDeprecationWarning,
        )
        self.prjfile = prj

    @property
    def prjfile(self):
        """
        Path to a .prj file containing WKT for a coordinate reference system.
        """
        return getattr(self, "_prjfile", None)

    @prjfile.setter
    def prjfile(self, prjfile):
        if prjfile is None:
            self._prjfile = None
            return
        if not isinstance(prjfile, (str, os.PathLike)):
            raise ValueError("prjfile property must be str, PathLike or None")
        self._prjfile = prjfile
        # If crs was previously unset, use .prj file input
        if HAS_PYPROJ and self._crs is None:
            try:
                self._crs = get_crs(prjfile=prjfile)
            except FileNotFoundError:
                pass

    @property
    def top(self):
        return copy.deepcopy(self._top)

    @property
    def botm(self):
        return copy.deepcopy(self._botm)

    @property
    def top_botm(self):
        raise NotImplementedError("must define top_botm in child class")

    @property
    def laycbd(self):
        if self._laycbd is None:
            return None
        else:
            return self._laycbd

    @property
    def cell_thickness(self):
        """
        Get the cell thickness for a structured, vertex, or unstructured grid.

        Returns
        -------
            thick : calculated thickness
        """
        return -np.diff(self.top_botm, axis=0).reshape(self._botm.shape)

    @property
    def thick(self):
        """Raises AttributeError, use :meth:`cell_thickness`."""
        # DEPRECATED since version 3.4.0
        raise AttributeError(
            "'thick' has been removed; use 'cell_thickness()'"
        )

    def saturated_thickness(self, array, mask=None):
        """
        Get the saturated thickness for a structured, vertex, or unstructured
        grid. If the optional array is passed then thickness is returned
        relative to array values (saturated thickness). Returned values
        ranges from zero to cell thickness if optional array is passed.

        Parameters
        ----------
        array : ndarray
            array of elevations that will be used to adjust the cell thickness
        mask: float, list, tuple, ndarray
            array values to replace with a nan value.

        Returns
        -------
            thickness : calculated saturated thickness
        """
        thickness = self.cell_thickness
        top = self.top_botm[:-1].reshape(thickness.shape)
        bot = self.top_botm[1:].reshape(thickness.shape)
        thickness = self.remove_confining_beds(thickness)
        top = self.remove_confining_beds(top)
        bot = self.remove_confining_beds(bot)
        array = self.remove_confining_beds(array)

        idx = np.where((array < top) & (array > bot))
        thickness[idx] = array[idx] - bot[idx]
        idx = np.where(array <= bot)
        thickness[idx] = 0.0
        if mask is not None:
            if isinstance(mask, (float, int)):
                mask = [float(mask)]
            for mask_value in mask:
                thickness[np.where(array == mask_value)] = np.nan
        return thickness

    def saturated_thick(self, array, mask=None):
        """Raises AttributeError, use :meth:`saturated_thickness`."""
        # DEPRECATED since version 3.4.0
        raise AttributeError(
            "'saturated_thick' has been removed; use 'saturated_thickness()'"
        )

    @property
    def units(self):
        return self._units

    @property
    def lenuni(self):
        return self._lenuni

    @property
    def idomain(self):
        return copy.deepcopy(self._idomain)

    @idomain.setter
    def idomain(self, idomain):
        self._idomain = idomain

    @property
    def nlay(self):
        raise NotImplementedError("must define nlay in child class")

    @property
    def ncpl(self):
        raise NotImplementedError("must define ncpl in child class")

    @property
    def nnodes(self):
        raise NotImplementedError("must define nnodes in child class")

    @property
    def nvert(self):
        raise NotImplementedError("must define nvert in child class")

    @property
    def iverts(self):
        raise NotImplementedError("must define iverts in child class")

    @property
    def verts(self):
        raise NotImplementedError("must define vertices in child class")

    @property
    def shape(self):
        raise NotImplementedError("must define shape in child class")

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def extent(self):
        raise NotImplementedError("must define extent in child class")

    @property
    def xyzextent(self):
        return (
            np.min(self.xyzvertices[0]),
            np.max(self.xyzvertices[0]),
            np.min(self.xyzvertices[1]),
            np.max(self.xyzvertices[1]),
            np.min(self.xyzvertices[2]),
            np.max(self.xyzvertices[2]),
        )

    @property
    def grid_lines(self):
        raise NotImplementedError("must define grid_lines in child class")

    @property
    def xcellcenters(self):
        return self.xyzcellcenters[0]

    def get_xcellcenters_for_layer(self, layer):
        # default is not layer dependent; must override for unstructured grid
        return self.xcellcenters

    @property
    def ycellcenters(self):
        return self.xyzcellcenters[1]

    def get_ycellcenters_for_layer(self, layer):
        # default is not layer dependent; must override for unstructured grid
        return self.ycellcenters

    @property
    def zcellcenters(self):
        return self.xyzcellcenters[2]

    @property
    def xyzcellcenters(self):
        raise NotImplementedError(
            "must define get_cellcenters in child "
            "class to use this base class"
        )

    @property
    def xvertices(self):
        return self.xyzvertices[0]

    def get_xvertices_for_layer(self, layer):
        # default is not layer dependent; must override for unstructured grid
        return self.xvertices

    @property
    def yvertices(self):
        return self.xyzvertices[1]

    def get_yvertices_for_layer(self, layer):
        # default is not layer dependent; must override for unstructured grid
        return self.yvertices

    @property
    def zvertices(self):
        return self.xyzvertices[2]

    @property
    def xyzvertices(self):
        raise NotImplementedError("must define xyzvertices in child class")

    # @property
    # def indices(self):
    #    raise NotImplementedError(
    #        'must define indices in child '
    #        'class to use this base class')
    @property
    def cross_section_vertices(self):
        return self.xyzvertices[0], self.xyzvertices[1]

    def geo_dataframe(self, polys):
        """
        Method returns a geopandas GeoDataFrame of the Grid

        Returns
        -------
            GeoDataFrame
        """
        from ..utils.geospatial_utils import GeoSpatialCollection

        gc = GeoSpatialCollection(
            polys, shapetype=["Polygon" for _ in range(len(polys))]
        )
        gdf = gc.geo_dataframe
        if self.crs is not None:
            gdf = gdf.set_crs(crs=self.crs)

        return gdf

    def convert_grid(self, factor):
        """
        Method to scale the model grid based on user supplied scale factors

        Parameters
        ----------
        factor

        Returns
        -------
            Grid object
        """
        raise NotImplementedError(
            "convert_grid must be defined in the child class"
        )

    def _set_neighbors(self, reset=False, method="rook"):
        """
        Method to calculate neighbors via shared edges or shared vertices

        Parameters
        ----------
        reset : bool
            flag to recalculate neighbors
        method: str
            "rook" for shared edges and "queen" for shared vertex

        Returns
        -------
            None
        """
        if self._neighbors is None or reset:
            node_num = 0
            neighbors = {i: list() for i in range(len(self.iverts))}
            edge_set = {i: list() for i in range(len(self.iverts))}
            geoms = []
            node_nums = []
            if method == "rook":
                for poly in self.iverts:
                    if poly[0] == poly[-1]:
                        poly = poly[:-1]
                    for v in range(len(poly)):
                        geoms.append(tuple(sorted([poly[v - 1], poly[v]])))
                    node_nums += [node_num] * len(poly)
                    node_num += 1
            else:
                # queen neighbors
                for poly in self.iverts:
                    if poly[0] == poly[-1]:
                        poly = poly[:-1]
                    for vert in poly:
                        geoms.append(vert)
                    node_nums += [node_num] * len(poly)
                    node_num += 1

            edge_nodes = defaultdict(set)
            for i, item in enumerate(geoms):
                edge_nodes[item].add(node_nums[i])

            shared_vertices = []
            for edge, nodes in edge_nodes.items():
                if len(nodes) > 1:
                    shared_vertices.append(nodes)
                    for n in nodes:
                        edge_set[n].append(edge)
                        neighbors[n] += list(nodes)
                        try:
                            neighbors[n].remove(n)
                        except:
                            pass

            # convert use dict to create a set that preserves insertion order
            self._neighbors = {
                i: list(dict.fromkeys(v)) for i, v in neighbors.items()
            }
            self._edge_set = edge_set

    def neighbors(self, node=None, **kwargs):
        """
        Method to get nearest neighbors of a cell

        Parameters
        ----------
        node : int
            model grid node number

        ** kwargs:
            method : str
                "rook" for shared edge neighbors and "queen" for shared vertex
                neighbors
            reset : bool
                flag to reset the neighbor calculation

        Returns
        -------
            list or dict : list of cell node numbers or dict of all cells and
                neighbors
        """
        method = kwargs.pop("method", None)
        reset = kwargs.pop("reset", False)
        if method is None:
            self._set_neighbors(reset=reset)
        else:
            self._set_neighbors(reset=reset, method=method)

        if node is not None:
            lay = 0
            if not isinstance(self.ncpl, (list, np.ndarray)):
                while node >= self.ncpl:
                    node -= self.ncpl
                    lay += 1

                neighbors = self._neighbors[node]
                if lay > 0:
                    neighbors = [i + (self.ncpl * lay) for i in neighbors]
            else:
                neighbors = self._neighbors[node]

            return neighbors

        return self._neighbors

    def remove_confining_beds(self, array):
        """
        Method to remove confining bed layers from an array

        Parameters
        ----------
        array : np.ndarray
            array to remove quasi3d confining bed data from. Shape of axis 0
            should be (self.lay + ncb) to remove beds
        Returns
        -------
            np.ndarray
        """
        if self.laycbd is not None:
            ncb = np.count_nonzero(self.laycbd)
            if ncb > 0:
                if array.shape[0] == self.shape[0] + ncb:
                    cb = 0
                    idx = []
                    for ix, i in enumerate(self.laycbd):
                        idx.append(ix + cb)
                        if i > 0:
                            cb += 1
                    array = array[idx]
        return array

    def cross_section_lay_ncpl_ncb(self, ncb):
        """
        Get PlotCrossSection compatible layers, ncpl, and ncb
        variables

        Parameters
        ----------
        ncb : int
            number of confining beds

        Returns
        -------
            tuple : (int, int, int) layers, ncpl, ncb
        """
        return self.nlay, self.ncpl, ncb

    def cross_section_nodeskip(self, nlay, xypts):
        """
        Get a nodeskip list for PlotCrossSection. This is a correction
        for UnstructuredGridPlotting

        Parameters
        ----------
        nlay : int
            nlay is nlay + ncb
        xypts : dict
            dictionary of node number and xyvertices of a cross-section

        Returns
        -------
            list : n-dimensional list of nodes to not plot for each layer
        """
        return [[] for _ in range(nlay)]

    def cross_section_adjust_indicies(self, k, cbcnt):
        """
        Method to get adjusted indices by layer and confining bed
        for PlotCrossSection plotting

        Parameters
        ----------
        k : int
            zero based layer number
        cbcnt : int
            confining bed counter

        Returns
        -------
            tuple: (int, int, int) (adjusted layer, nodeskip layer, node
            adjustment value based on number of confining beds and the layer)
        """
        adjnn = k * self.ncpl
        ncbnn = adjnn - (cbcnt * self.ncpl)
        return k + 1, k + 1, ncbnn

    def cross_section_set_contour_arrays(
        self, plotarray, xcenters, head, elev, projpts
    ):
        """
        Method to set contour array centers for rare instances where
        matplotlib contouring is preferred over trimesh plotting

        Parameters
        ----------
        plotarray : np.ndarray
            array of data for contouring
        xcenters : np.ndarray
            xcenters array
        zcenters : np.ndarray
            zcenters array
        head : np.ndarray
            head array to adjust cell centers location
        elev : np.ndarray
            cell elevation array
        projpts : dict
            dictionary of projected cross sectional vertices

        Returns
        -------
            tuple: (np.ndarray, np.ndarray, np.ndarray, bool)
            plotarray, xcenter array, ycenter array, and a boolean flag
            for contouring
        """
        if self.ncpl != self.nnodes:
            return plotarray, xcenters, None, False
        else:
            zcenters = []
            if isinstance(head, np.ndarray):
                head = head.reshape(1, self.ncpl)
                head = np.vstack((head, head))
            else:
                head = elev.reshape(2, self.ncpl)

            elev = elev.reshape(2, self.ncpl)
            for k, ev in enumerate(elev):
                if k == 0:
                    zc = [
                        ev[i] if head[k][i] > ev[i] else head[k][i]
                        for i in sorted(projpts)
                    ]
                else:
                    zc = [ev[i] for i in sorted(projpts)]
                zcenters.append(zc)

            plotarray = np.vstack((plotarray, plotarray))
            xcenters = np.vstack((xcenters, xcenters))
            zcenters = np.array(zcenters)

            return plotarray, xcenters, zcenters, True

    @property
    def map_polygons(self):
        raise NotImplementedError("must define map_polygons in child class")

    def get_lni(self, nodes):
        """
        Get the layer index and within-layer node index (both 0-based) for the given nodes

        Parameters
        ----------
        nodes : node numbers (array-like)

        Returns
        -------
            list of tuples (layer index, node index)
        """

        ncpl = (
            [self.ncpl for _ in range(self.nlay)]
            if isinstance(self.ncpl, int)
            else list(self.ncpl)
        )

        return get_lni(ncpl, nodes)

    def get_plottable_layer_array(self, plotarray, layer):
        raise NotImplementedError(
            "must define get_plottable_layer_array in child class"
        )

    def get_number_plottable_layers(self, a):
        raise NotImplementedError(
            "must define get_number_plottable_layers in child class"
        )

    def get_plottable_layer_shape(self, layer=None):
        """
        Determine the shape that is required in order to plot a 2d array for
        this grid.  For a regular MODFLOW grid, this is (nrow, ncol).  For
        a vertex grid, this is (ncpl,) and for an unstructured grid this is
        (ncpl[layer],).

        Parameters
        ----------
        layer : int
            Has no effect unless grid changes by layer

        Returns
        -------
        shape : tuple
            required shape of array to plot for a layer
        """
        return self.shape[1:]

    def get_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.astype(float, copy=True), y.astype(float, copy=True)

        x += self._xoff
        y += self._yoff
        return geometry.rotate(
            x, y, self._xoff, self._yoff, self.angrot_radians
        )

    def get_local_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from real-world coordinates to model coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        x, y = geometry.transform(
            x, y, self._xoff, self._yoff, self.angrot_radians, inverse=True
        )
        # x -= self._xoff
        # y -= self._yoff

        return x, y

    def intersect(self, x, y, local=False, forgive=False):
        if not local:
            return self.get_local_coords(x, y)
        else:
            return x, y

    def set_coord_info(
        self,
        xoff=None,
        yoff=None,
        angrot=None,
        crs=None,
        prjfile=None,
        merge_coord_info=True,
        **kwargs,
    ):
        """Set coordinate information for a grid.

        Parameters
        ----------
        xoff, yoff : float, optional
            X and Y coordinate of the origin point in the spatial reference
            coordinate system.
        angrot : float, optional
            Rotation angle of model grid, as it is rotated around the origin
            point.
        crs : pyproj.CRS, int, str, optional
            Coordinate reference system (CRS) for the model grid
            (must be projected; geographic CRS are not supported).
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:26916") or a WKT string.
        prjfile : str or pathlike, optional
            ESRI-style projection file with well-known text defining the CRS
            for the model grid (must be projected; geographic CRS are not
            supported).
        merge_coord_info : bool, default True
            If True, retaining previous properties.
            If False, overwrite properties with defaults, unless specified.
        **kwargs : dict, optional
            Support deprecated keyword options.

            .. deprecated:: 3.5
               The following keyword options will be removed for FloPy 3.6:

                 - ``epsg`` (int): use ``crs`` instead.
                 - ``proj4`` (str): use ``crs`` instead.

        """
        if crs is not None:
            # Force these to be re-evaluated, if/when needed
            if hasattr(self, "_epsg"):
                delattr(self, "_epsg")
            if hasattr(self, "_proj4"):
                delattr(self, "_proj4")
        # Handle deprecated projection kwargs; warnings are raised in crs.py
        get_crs_args = {"crs": crs, "prjfile": prjfile}
        if "epsg" in kwargs:
            self.epsg = get_crs_args["epsg"] = kwargs.pop("epsg")
        if "proj4" in kwargs:
            self.proj4 = get_crs_args["proj4"] = kwargs.pop("proj4")
        if kwargs:
            raise TypeError(f"unhandled keywords: {kwargs}")
        if HAS_PYPROJ:
            new_crs = get_crs(**get_crs_args)
        else:
            new_crs = None
            # provide some support without pyproj by retaining 'epsg' integer
            if getattr(self, "_epsg", None) is None:
                epsg = _get_epsg_from_crs_or_proj4(crs, self.proj4)
                if epsg is not None:
                    self.epsg = epsg

        if merge_coord_info:
            if xoff is None:
                xoff = self._xoff
            if yoff is None:
                yoff = self._yoff
            if angrot is None:
                angrot = self._angrot
            if new_crs is None:
                new_crs = self._crs

        if xoff is None:
            xoff = 0.0
        if yoff is None:
            yoff = 0.0
        if angrot is None:
            angrot = 0.0

        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._prjfile = prjfile
        self._crs = new_crs
        self._require_cache_updates()

    def load_coord_info(self, namefile=None, reffile="usgs.model.reference"):
        """Attempts to load spatial reference information from
        the following files (in order):
        1) usgs.model.reference
        2) NAM file (header comment)
        3) defaults
        """
        reffile = os.path.join(os.path.split(namefile)[0], reffile)
        # try to load reference file
        if not self.read_usgs_model_reference_file(reffile):
            # try to load nam file
            if not self.attribs_from_namfile_header(namefile):
                # set defaults
                self.set_coord_info()

    def attribs_from_namfile_header(self, namefile):
        # check for reference info in the nam file header
        if namefile is None:
            return False
        xul, yul = None, None
        set_coord_info_args = {}
        header = []
        with open(namefile) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.extend(line.strip().replace("#", "").split(";"))

        for item in header:
            if "xll" in item.lower():
                try:
                    xll = float(item.split(":")[1])
                    self._xoff = xll
                except:
                    pass
            elif "yll" in item.lower():
                try:
                    yll = float(item.split(":")[1])
                    self._yoff = yll
                except:
                    pass
            elif "xul" in item.lower():
                try:
                    xul = float(item.split(":")[1])
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(":")[1])
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    self._angrot = float(item.split(":")[1])
                except:
                    pass
            elif "crs" in item.lower():
                try:
                    crs = ":".join(item.split(":")[1:]).strip()
                    if crs.lower() != "none":
                        set_coord_info_args["crs"] = crs
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4 = ":".join(item.split(":")[1:]).strip()
                    if proj4.lower() != "none":
                        set_coord_info_args["proj4"] = crs
                except:
                    pass
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(":")[1].strip()
                except:
                    pass

        # we need to rotate the modelgrid first, then we can
        # calculate the xll and yll from xul and yul
        if (xul, yul) != (None, None):
            set_coord_info_args["xoff"] = self._xul_to_xll(xul)
            set_coord_info_args["yoff"] = self._yul_to_yll(yul)
            set_coord_info_args["angrot"] = self._angrot

        if set_coord_info_args:
            self.set_coord_info(**set_coord_info_args)

        return True

    def read_usgs_model_reference_file(self, reffile="usgs.model.reference"):
        """read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html"""
        xul = None
        yul = None
        if os.path.exists(reffile):
            with open(reffile) as input:
                for line in input:
                    if len(line) > 1:
                        if line.strip()[0] != "#":
                            info = line.strip().split("#")[0].split()
                            if len(info) > 1:
                                data = " ".join(info[1:]).strip("'").strip('"')
                                if info[0] == "xll":
                                    self._xoff = float(data)
                                elif info[0] == "yll":
                                    self._yoff = float(data)
                                elif info[0] == "xul":
                                    xul = float(data)
                                elif info[0] == "yul":
                                    yul = float(data)
                                elif info[0] == "rotation":
                                    self._angrot = float(data)
                                elif info[0] == "epsg":
                                    self.epsg = int(data)
                                elif info[0] == "proj4":
                                    self.crs = data
                                elif info[0] == "start_date":
                                    start_datetime = data

            # model must be rotated first, before setting xoff and yoff
            # when xul and yul are provided.
            if (xul, yul) != (None, None):
                self.set_coord_info(
                    xoff=self._xul_to_xll(xul),
                    yoff=self._yul_to_yll(yul),
                    angrot=self._angrot,
                )

            return True
        else:
            return False

    # Internal
    def _xul_to_xll(self, xul, angrot=None):
        yext = self.xyedges[1][0]
        if angrot is not None:
            return xul + (np.sin(angrot * np.pi / 180) * yext)
        else:
            return xul + (np.sin(self.angrot_radians) * yext)

    def _yul_to_yll(self, yul, angrot=None):
        yext = self.xyedges[1][0]
        if angrot is not None:
            return yul - (np.cos(angrot * np.pi / 180) * yext)
        else:
            return yul - (np.cos(self.angrot_radians) * yext)

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True

    @property
    def _has_ref_coordinates(self):
        return self._xoff != 0.0 or self._yoff != 0.0 or self._angrot != 0.0

    def _load_settings(self, d):
        self._xoff = d.xul

    def _zcoords(self):
        if self.top is not None and self.botm is not None:
            zcenters = []
            top_3d = np.expand_dims(self.top, 0)
            zbdryelevs = np.concatenate(
                (top_3d, np.atleast_2d(self.botm)), axis=0
            )

            for ix in range(1, len(zbdryelevs)):
                zcenters.append((zbdryelevs[ix - 1] + zbdryelevs[ix]) / 2.0)
        else:
            zbdryelevs = None
            zcenters = None
        return zbdryelevs, zcenters

    # Exporting
    def write_shapefile(
        self, filename="grid.shp", crs=None, prjfile=None, **kwargs
    ):
        """
        Write a shapefile of the grid with just the row and column attributes.

        """
        from ..export.shapefile_utils import write_grid_shapefile

        # Handle deprecated projection kwargs; warnings are raised in crs.py
        write_grid_shapefile_args = {}
        if "epsg" in kwargs:
            write_grid_shapefile_args["epsg"] = kwargs.pop("epsg")
        if "prj" in kwargs:
            write_grid_shapefile_args["prj"] = kwargs.pop("prj")
        if kwargs:
            raise TypeError(f"unhandled keywords: {kwargs}")
        if crs is None:
            crs = self.crs
        write_grid_shapefile(
            filename,
            self,
            array_dict={},
            nan_val=-1.0e9,
            crs=crs,
            **write_grid_shapefile_args,
        )
        return

    # initialize grid from a grb file
    @classmethod
    def from_binary_grid_file(cls, file_path, verbose=False):
        raise NotImplementedError(
            "must define from_binary_grid_file in child class"
        )
