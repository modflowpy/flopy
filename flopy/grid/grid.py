import numpy as np
import copy, os
import warnings
from ..utils import geometry


class CachedData(object):
    def __init__(self, data):
        self._data = data
        self.out_of_date = False

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def update_data(self, data):
        self._data = data
        self.out_of_date = False


class Grid(object):
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex_layered',
        'vertex_unlayered')
    top : ndarray(np.float)
        top elevations of cells in topmost layer
    botm : ndarray(np.float)
        bottom elevations of all cells
    idomain : ndarray(np.int)
        ibound/idomain value for each cell
    sr : SpatialReference
        spatial reference locates the grid in a coordinate system
    origin_loc : str
        Corner of the model grid that is the model origin
        'ul' (upper left corner) or 'll' (lower left corner)
    origin_x : float
        x coordinate of the origin point in the spatial reference coordinate
        system
    origin_y : float
        y coordinate of the origin point in the spatial reference coordinate
        system
    rotation : float
        rotation angle of model grid, as it is rotated around the origin point

    Properties
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex_layered',
        'vertex_unlayered')
    top : ndarray(np.float)
        top elevations of cells in topmost layer
    botm : ndarray(np.float)
        bottom elevations of all cells
    idomain : ndarray(np.int)
        ibound/idomain value for each cell
    proj4 : proj4 SpatialReference
        spatial reference locates the grid in a coordinate system
    epsg : epsg SpatialReference
        spatial reference locates the grid in a coordinate system
    origin_x : float
        x coordinate of the origin point in the spatial reference coordinate
        system
    origin_y : float
        y coordinate of the origin point in the spatial reference coordinate
        system
    rotation : float
        rotation angle of model grid, as it is rotated around the origin point
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
    cell_centers : [ndarray, ndarray, ndarray]
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

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, grid_type, top=None, botm=None, idomain=None, epsg=None,
                 proj4=None, xoff=0.0, yoff=0.0, angrot=0.0):
        self.use_ref_coords = True
        self._grid_type = grid_type
        self._top = top
        self._botm = botm
        self._idomain = idomain
        self._epsg = epsg
        self._proj4 = proj4
        self._wkt = None
        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._cache_dict = {}

    ###################################
    # access to basic grid properties
    ###################################
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
        return self.angrot * np.pi / 180.

    @property
    def epsg(self):
        return self._epsg

    @property
    def proj4(self):
        return self._proj4

    @property
    def top(self):
        return self._top

    @property
    def botm(self):
        return self._botm

    @property
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        return np.concatenate((new_top, self._botm), axis=0)

    @property
    def idomain(self):
        return self._idomain

    @property
    def extent(self):
        raise NotImplementedError(
            'must define extent in child '
            'class to use this base class')

    @property
    def grid_lines(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def xcellcenters(self):
        return self.xyzcellcenters[0]

    @property
    def ycellcenters(self):
        return self.xyzcellcenters[1]

    @property
    def zcellcenters(self):
        return self.xyzcellcenters[2]

    @property
    def xyzcellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def xvertices(self):
        return self.xyzvertices[0]

    @property
    def yvertices(self):
        return self.xyzvertices[1]

    @property
    def zvertices(self):
        return self.xyzvertices[2]

    @property
    def xyzvertices(self):
        raise NotImplementedError(
            'must define xyzgrid in child '
            'class to use this base class')

    #@property
    #def indices(self):
    #    raise NotImplementedError(
    #        'must define indices in child '
    #        'class to use this base class')

    def get_cell_vertices(self, cellid):
        raise NotImplementedError(
            "must define get_cell_verices in"
            " child class to use this base class")

    def get_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        x += self._xoff
        y += self._yoff
        return geometry.rotate(x, y, self._xoff, self._yoff,
                               self.angrot_radians)

    def get_local_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        x, y = geometry.rotate(x, y, self._xoff, self._yoff,
                               -self.angrot_radians)
        x -= self._xoff
        y -= self._yoff

        return x, y

    def intersect(self, x, y, local=True):
        if not local:
            return self.get_local_coords(x, y)
        else:
            return x, y

    def set_coord_info(self, xoff=0.0, yoff=0.0, angrot=0.0, epsg=None,
                       proj4=None):
        if xoff is None:
            xoff = self._xoff
        if yoff is None:
            yoff = self._yoff
        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._epsg = epsg
        self._proj4 = proj4
        self._require_cache_updates()

    def load_coord_info(self, namefile=None, reffile='usgs.model.reference'):
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
        header = []
        with open(namefile, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                header.extend(line.strip().replace('#', '').split(';'))

        for item in header:
            if "xll" in item.lower():
                try:
                    xll = float(item.split(':')[1])
                    self._xoff = xll
                except:
                    pass
            elif "yll" in item.lower():
                try:
                    yll = float(item.split(':')[1])
                    self._yoff = yll
                except:
                    pass
            elif "xul" in item.lower():
                try:
                    xul = float(item.split(':')[1])
                    self._xoff = self._xul_to_xll(xul)
                    warnings.warn(
                        'xul/yul have been deprecated. Use xll/yll instead.',
                        DeprecationWarning)
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(':')[1])
                    self._yoff = self._yul_to_yll(yul)
                    warnings.warn(
                        'xul/yul have been deprecated. Use xll/yll instead.',
                        DeprecationWarning)
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    self._angrot = float(item.split(':')[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    self._proj4 = ':'.join(item.split(':')[1:]).strip()
                    if self._proj4.lower() == 'none':
                        self._proj4 = None
                except:
                    pass
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(':')[1].strip()
                except:
                    pass
        return True

    def read_usgs_model_reference_file(self, reffile='usgs.model.reference'):
        """read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html"""
        if os.path.exists(reffile):
            with open(reffile) as input:
                for line in input:
                    if len(line) > 1:
                        if line.strip()[0] != '#':
                            info = line.strip().split('#')[0].split()
                            if len(info) > 1:
                                data = ' '.join(info[1:])
                                if info == 'xll':
                                    self._xoff = float(data)
                                elif info == 'yul':
                                    self._yoff = float(data)
                                elif info == 'xul':
                                    self._xoff = self._xul_to_xll(
                                        float(data))
                                    warnings.warn(
                                        'xul/yul have been deprecated. Use xll/yll instead.',
                                        DeprecationWarning)
                                elif info == 'yul':
                                    self._yoff = self._yul_to_yll(
                                        float(data))
                                    warnings.warn(
                                        'xul/yul have been deprecated. Use xll/yll instead.',
                                        DeprecationWarning)
                                elif info == 'rotation':
                                    self._angrot = float(data)
                                elif info == 'epsg':
                                    self._epsg = int(data)
                                elif info == 'proj4':
                                    self._proj4 = data
                                elif info == 'start_datetime':
                                    start_datetime = data
            return True
        else:
            return False

    # Internal
    def _xul_to_xll(self, xul):
        yext = self.xyedges[1][0]
        return xul + (np.sin(self.angrot_radians) * yext)

    def _yul_to_yll(self, yul):
        yext = self.xyedges[1][0]
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
            zbdryelevs = np.concatenate((top_3d, self.botm), axis=0)

            for ix in range(1, len(zbdryelevs)):
                zcenters.append((zbdryelevs[ix - 1] + zbdryelevs[ix]) / 2.)
        else:
            zbdryelevs = None
            zcenters = None
        return zbdryelevs, zcenters