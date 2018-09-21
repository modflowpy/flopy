import abc
from enum import Enum
import os
import numpy as np
import copy
from pandas import DataFrame


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
    lenuni : int
        length unit (0 - undefined, 1 - feet, 2 - meters, 3 - centimeters)
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
    lenuni : int
        length unit (0 - undefined, 1 - feet, 2 - meters, 3 - centimeters)
    model_length_units : str
        returns length unit as a string
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
    rotate(x, y, rotation, xorigin=0., yorigin=0.)
        rotate point defined by x, y by rotation around the origin point
    transform(x, y, inverse=False)
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
    lenuni_values = {'undefined': 0,
                     'feet': 1,
                     'meters': 2,
                     'centimeters': 3}
    lenuni_text = {v: k for k, v in lenuni_values.items()}

    defaults = {"origin_x": None, "origin_y": None, "rotation": 0.,
                "proj4_str": None,
                "units": None, "lenuni": 2,
                "length_multiplier": None}

    def __init__(self, grid_type, top=None, botm=None, idomain=None, lenuni=2,
                 epsg=None, proj4=None, xoff=0.0, yoff=0.0, angrot=0.0):
        self.use_ref_coords = True
        self._grid_type = grid_type
        self._top = top
        self._botm = botm
        self._idomain = idomain
        self._epsg = epsg
        self._proj4 = proj4
        self._lenuni = lenuni
        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._cache_dict = {}

    def set_coord_info(self, sr=None, origin_x=0.0,
                       origin_y=0.0, angrot=0.0):
        self._sr = sr
        self._xoff = origin_x
        self._yoff = origin_y
        self._angrot = angrot
        self._require_cache_updates()

    ###########################
    # basic functions
    ###########################
    @property
    def grid_type(self):
        return self._grid_type

    @property
    def top(self):
        return self._top

    @property
    def botm(self):
        return self._botm

    @property
    def idomain(self):
        return self._idomain

    @property
    def epsg(self):
        return self._epsg

    @property
    def proj4(self):
        return self._proj4

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
    def lenuni(self):
        return self._lenuni

    @property
    def model_length_units(self):
        return self.lenuni_text[self._lenuni]

    def rotate(self, x, y):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.

        """
        xrot = self._xoff + np.cos(self.angrot_radians) * \
               (x - self._xoff) - np.sin(self.angrot_radians) * \
               (y - self._yoff)
        yrot = self._yoff + np.sin(self.angrot_radians) * \
               (x - self._xoff) + np.cos(self.angrot_radians) * \
               (y - self._yoff)

        return xrot, yrot

    def transform(self, x, y):
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
        return self.rotate(x, y)

    @property
    def extent(self):
        raise NotImplementedError(
            'must define extent in child '
            'class to use this base class')

    @property
    def xgridlength(self):
        raise NotImplementedError(
            'must define xgridlength in child '
            'class to use this base class')

    @property
    def ygridlength(self):
        raise NotImplementedError(
            'must define ygridlength in child '
            'class to use this base class')

    @property
    def xgrid(self):
        return self.xyzgrid[0]

    @property
    def ygrid(self):
        return self.xyzgrid[1]

    @property
    def zgrid(self):
        return self.xyzgrid[2]

    @property
    def xyzgrid(self):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @property
    def grid_lines(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def xcenters(self):
        return self.cellcenters[0]

    @property
    def ycenters(self):
        return self.cellcenters[1]

    @property
    def zcenters(self):
        return self.cellcenters[2]

    @property
    def cellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

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