import numpy as np
import copy, os
import warnings
from ..utils import geometry


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


class Grid:
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex', 'unstructured')
    top : ndarray(float)
        top elevations of cells in topmost layer
    botm : ndarray(float)
        bottom elevations of all cells
    idomain : ndarray(int)
        ibound/idomain value for each cell
    lenuni : ndarray(int)
        model length units
    espg : str, int
        optional espg projection code
    proj4 : str
        optional proj4 projection string code
    prj : str
        optional projection file name path
    xoff : float
        x coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    yoff : float
        y coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    angrot : float
        rotation angle of model grid, as it is rotated around the origin point

    Attributes
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex', 'unstructured')
    top : ndarray(float)
        top elevations of cells in topmost layer
    botm : ndarray(float)
        bottom elevations of all cells
    idomain : ndarray(int)
        ibound/idomain value for each cell
    proj4 : proj4 SpatialReference
        spatial reference locates the grid in a coordinate system
    epsg : epsg SpatialReference
        spatial reference locates the grid in a coordinate system
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
    xgrid : ndarray
        returns numpy meshgrid of x edges in reference frame defined by
        point_type
    ygrid : ndarray
        returns numpy meshgrid of y edges in reference frame defined by
        point_type
    zgrid : ndarray
        returns numpy meshgrid of z edges in reference frame defined by
        point_type
    xcenters : ndarray
        returns x coordinate of cell centers
    ycenters : ndarray
        returns y coordinate of cell centers
    ycenters : ndarray
        returns z coordinate of cell centers
    xyzgrid : [ndarray, ndarray, ndarray]
        returns the location of grid edges of all model cells. if the model
        grid contains spatial reference information, the grid edges are in the
        coordinate system provided by the spatial reference information.
        returns a list of three ndarrays for the x, y, and z coordinates
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
    xyvertices : (point_type) : ndarray
        1D array of x and y coordinates of cell vertices for whole grid
        (single layer) in C-style (row-major) order
        (same as np.ravel())
    intersect(x, y, local)
        returns the row and column of the grid that the x, y point is in

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
        epsg=None,
        proj4=None,
        prj=None,
        xoff=0.0,
        yoff=0.0,
        angrot=0.0,
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
        self._epsg = epsg
        self._proj4 = proj4
        self._prj = prj
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
        if self.proj4 is not None:
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
    def epsg(self):
        return self._epsg

    @epsg.setter
    def epsg(self, epsg):
        self._epsg = epsg

    @property
    def proj4(self):
        proj4 = None
        if self._proj4 is not None:
            if "epsg" in self._proj4.lower():
                proj4 = self._proj4
                # set the epsg if proj4 specifies it
                tmp = [i for i in self._proj4.split() if "epsg" in i.lower()]
                self._epsg = int(tmp[0].split(":")[1])
            else:
                proj4 = self._proj4
        elif self.epsg is not None:
            proj4 = f"epsg:{self.epsg}"
        return proj4

    @proj4.setter
    def proj4(self, proj4):
        self._proj4 = proj4

    @property
    def prj(self):
        return self._prj

    @prj.setter
    def prj(self, prj):
        self._proj4 = prj

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
    def thick(self):
        """
        Get the cell thickness for a structured, vertex, or unstructured grid.

        Returns
        -------
            thick : calculated thickness
        """
        return -np.diff(self.top_botm, axis=0).reshape(self._botm.shape)

    def saturated_thick(self, array, mask=None):
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
            thick : calculated saturated thickness
        """
        thick = self.thick
        top = self.top_botm[:-1].reshape(thick.shape)
        bot = self.top_botm[1:].reshape(thick.shape)
        thick = self.remove_confining_beds(thick)
        top = self.remove_confining_beds(top)
        bot = self.remove_confining_beds(bot)
        array = self.remove_confining_beds(array)

        idx = np.where((array < top) & (array > bot))
        thick[idx] = array[idx] - bot[idx]
        idx = np.where(array <= bot)
        thick[idx] = 0.0
        if mask is not None:
            if isinstance(mask, (float, int)):
                mask = [float(mask)]
            for mask_value in mask:
                thick[np.where(array == mask_value)] = np.nan
        return thick

    @property
    def units(self):
        return self._units

    @property
    def lenuni(self):
        return self._lenuni

    @property
    def idomain(self):
        return copy.deepcopy(self._idomain)

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
        Method to get adjusted indicies by layer and confining bed
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
        Method to set countour array centers for rare instances where
        matplotlib contouring is prefered over trimesh plotting

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
        epsg=None,
        proj4=None,
        merge_coord_info=True,
    ):
        if merge_coord_info:
            if xoff is None:
                xoff = self._xoff
            if yoff is None:
                yoff = self._yoff
            if angrot is None:
                angrot = self._angrot
            if epsg is None:
                epsg = self._epsg
            if proj4 is None:
                proj4 = self._proj4

        if xoff is None:
            xoff = 0.0
        if yoff is None:
            yoff = 0.0
        if angrot is None:
            angrot = 0.0

        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._epsg = epsg
        self._proj4 = proj4
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
        header = []
        with open(namefile, "r") as f:
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
            elif "proj4_str" in item.lower():
                try:
                    self._proj4 = ":".join(item.split(":")[1:]).strip()
                    if self._proj4.lower() == "none":
                        self._proj4 = None
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
            self.set_coord_info(
                xoff=self._xul_to_xll(xul),
                yoff=self._yul_to_yll(yul),
                angrot=self._angrot,
            )

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
                                data = " ".join(info[1:])
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
                                    self._epsg = int(data)
                                elif info[0] == "proj4":
                                    self._proj4 = data
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

    def _set_sr_coord_info(self, sr):
        self._xoff = sr.xll
        self._yoff = sr.yll
        self._angrot = sr.rotation
        self._epsg = sr.epsg
        self._proj4 = sr.proj4_str
        self._require_cache_updates()

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
    def write_shapefile(self, filename="grid.shp", epsg=None, prj=None):
        """
        Write a shapefile of the grid with just the row and column attributes.

        """
        from ..export.shapefile_utils import write_grid_shapefile

        if epsg is None and prj is None:
            epsg = self.epsg
        write_grid_shapefile(
            filename, self, array_dict={}, nan_val=-1.0e9, epsg=epsg, prj=prj
        )
        return

    # initialize grid from a grb file
    @classmethod
    def from_binary_grid_file(cls, file_path, verbose=False):
        raise NotImplementedError(
            "must define from_binary_grid_file in child class"
        )
