import abc
from enum import Enum
import os
import numpy as np
import copy
from pandas import DataFrame


class MFGridException(Exception):
    """
    Model grid related exception
    """
    def __init__(self, error):
        Exception.__init__(self, "MFGridException: {}".format(error))


class SimulationTime():
    """
    Class for MODFLOW simulation time

    Parameters
    ----------
    stress_periods : pandas dataframe
        headings are: perlen, nstp, tsmult
    temporal_reference : TemporalReference
        contains start time and time units information
    """
    def __init__(self, period_data, time_units='days',
                 temporal_reference=None):
        if isinstance(period_data, dict):
            period_data = DataFrame(period_data)
        self.period_data = period_data
        self.time_units = time_units
        self.tr = temporal_reference

    @property
    def perlen(self):
        return self.period_data['perlen'].values

    @property
    def nper(self):
        return len(self.period_data['perlen'].values)

    @property
    def nstp(self):
        return self.period_data['nstp'].values

    @property
    def tsmult(self):
        return self.period_data['tsmult'].values


class CachedDataType(Enum):
    """
    Enumeration of types of cached data
    """
    xyvertices = 0
    edge_array = 1
    edge_grid = 2
    cell_centers = 3


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


class ModelGrid(object):
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
    sim_time : SimulationTime
        simulation time provides time information for temporal grid data
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
    top_botm : ndarray(np.float)
        returns array combining top and botm arrays
    idomain : ndarray(np.int)
        ibound/idomain value for each cell
    sr : SpatialReference
        spatial reference locates the grid in a coordinate system
    sim_time : SimulationTime
        simulation time provides time information for temporal grid data
    lenuni : int
        length unit (0 - undefined, 1 - feet, 2 - meters, 3 - centimeters)
    model_length_units : str
        returns length unit as a string
    length_multiplier : float
        returns length multiplier between model and spatial coordinates
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
    write_gridSpec(filename)
        write PEST style grid specification file
    plot(**kwargs) : matplotlib.collections.LineCollection
        plot the model grid
    plot_array(a, ax=None, **kwargs) : matplotlib.collections.QuadMesh
        create a quadmesh plot of the grid.
    contour_array(ax, a, **kwargs) : ContourSet
        ax (matplotlib.axes.Axes) are axes to add to the plot
        a (np.ndarray) is the array to contour
        Create a QuadMesh plot of the specified array using pcolormesh
    get_3d_shared_vertex_connectivity() : [verts, iverts]
        returns a list of vertices of the model grid (verts) and a list of
        vertices by model cell (iverts)

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

    defaults = {"xul": None, "yul": None, "rotation": 0.,
                "proj4_str": None,
                "units": None, "lenuni": 2,
                "length_multiplier": None,
                "source": 'defaults'}

    def __init__(self, grid_type, top=None, botm=None, idomain=None, sr=None,
                 sim_time=None, lenuni=2, origin_loc='ul', origin_x=0.0,
                 origin_y=0.0, rotation=0.0):
        self.use_ref_coords = True
        self._grid_type = grid_type
        self._top = top
        self._botm = botm
        self._idomain = idomain
        self._sr = sr
        self._lenuni = lenuni
        self._origin_loc = origin_loc
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._rotation = rotation
        self._sim_time = sim_time
        self._cache_dict = {}

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
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        return np.concatenate((new_top, self._botm), axis=0)

    @property
    def idomain(self):
        return self._idomain

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        self._require_cache_updates()

    def sim_time(self):
        return self._sim_time

    @property
    def origin_loc(self):
        return self._origin_loc

    @property
    def origin_x(self):
        return self._origin_x

    @property
    def origin_y(self):
        return self._origin_y

    @property
    def rotation(self):
        return self._rotation

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True

    @property
    def _use_ref_coordinates(self):
        return (self._origin_x is not None and self._origin_x != 0.0 and
                self._origin_y is not None and self._origin_y != 0.0 and
                (self._origin_loc != 'ul' or self._rotation != 0.0)) and \
                self.use_ref_coords == True

    def _load_settings(self, d):
        self._origin_x = d.xul

    ############################
    # from spatial reference
    ############################
    @property
    def lenuni(self):
        return self._lenuni

    @property
    def model_length_units(self):
        return self.lenuni_text[self._lenuni]

    @property
    def length_multiplier(self):
        """Attempt to identify multiplier for converting from
        model units to sr units, defaulting to 1."""
        if self._sr is None:
            return 1.0
        if self.model_length_units == 'feet':
            if self._sr.units == 'meters':
                return 0.3048
            elif self._sr.units == 'feet':
                return 1.
        elif self.model_length_units == 'meters':
            if self._sr.units == 'feet':
                lm = 1 / .3048
            elif self._sr.units == 'meters':
                lm = 1.
        elif self.model_length_units == 'centimeters':
            if self._sr.units == 'meters':
                lm = 1 / 100.
            elif self._sr.units == 'feet':
                lm = 1 / 30.48
        else:  # model units unspecified; default to 1
            lm = 1.
        return lm

    @property
    def theta(self):
        return self._rotation * np.pi / 180.

    @staticmethod
    def rotate(x, y, rotation, xorigin=0., yorigin=0.):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        # jwhite changed on Oct 11 2016 - rotation is now positive CCW
        # theta = -theta * np.pi / 180.
        theta = rotation * np.pi / 180.

        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                         (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                         (y - yorigin)
        return xrot, yrot

    def transform(self, x, y, inverse=False):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        if not inverse:
            x *= self.length_multiplier
            y *= self.length_multiplier
            x += self._origin_x
            y += self._origin_y
            x, y = self.rotate(x, y, self._rotation, xorigin=self._origin_x,
                               yorigin=self._origin_y)
        else:
            x, y = self.rotate(x, y, -self._rotation, self._origin_x,
                               self._origin_y)
            x -= self._origin_x
            y -= self._origin_y
            x /= self.length_multiplier
            y /= self.length_multiplier
        return x, y

    @staticmethod
    def load(namefile=None, reffile='usgs.model.reference'):
        """Attempts to load spatial reference information from
        the following files (in order):
        1) usgs.model.reference
        2) NAM file (header comment)
        3) SpatialReference.default dictionary
        """
        reffile = os.path.join(os.path.split(namefile)[0], reffile)
        d = ModelGrid.read_usgs_model_reference_file(reffile)
        if d is not None:
            return d
        d = ModelGrid.attribs_from_namfile_header(namefile)
        if d is not None:
            return d

    @staticmethod
    def attribs_from_namfile_header(namefile):
        # check for reference info in the nam file header
        d = ModelGrid.defaults.copy()
        d['source'] = 'namfile'
        if namefile is None:
            return None
        header = []
        with open(namefile, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                header.extend(line.strip().replace('#', '').split(';'))

        for item in header:
            if "xul" in item.lower():
                try:
                    d['xul'] = float(item.split(':')[1])
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    d['yul'] = float(item.split(':')[1])
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    d['rotation'] = float(item.split(':')[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ':'.join(item.split(':')[1:]).strip()
                    if proj4_str.lower() == 'none':
                        proj4_str = None
                    d['proj4_str'] = proj4_str

                except:
                    pass
            elif "start" in item.lower():
                try:
                    d['start_datetime'] = item.split(':')[1].strip()
                except:
                    pass
            # spatial reference length units
            elif "units" in item.lower():
                d['units'] = item.split(':')[1].strip()
            # model length units
            elif "lenuni" in item.lower():
                d['lenuni'] = int(item.split(':')[1].strip())
            # multiplier for converting from model length units to sr length units
            elif "length_multiplier" in item.lower():
                d['length_multiplier'] = float(item.split(':')[1].strip())
        return d

    @staticmethod
    def read_usgs_model_reference_file(reffile='usgs.model.reference'):
        """read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html"""

        ITMUNI = {0: "undefined", 1: "seconds", 2: "minutes", 3: "hours",
                  4: "days",
                  5: "years"}
        itmuni_values = {v: k for k, v in ITMUNI.items()}

        d = ModelGrid.defaults.copy()
        d['source'] = 'usgs.model.reference'
        d.pop(
            'proj4_str')  # discard default to avoid confusion with epsg code if entered
        if os.path.exists(reffile):
            with open(reffile) as input:
                for line in input:
                    if len(line) > 1:
                        if line.strip()[0] != '#':
                            info = line.strip().split('#')[0].split()
                            if len(info) > 1:
                                d[info[0].lower()] = ' '.join(info[1:])
            d['xul'] = float(d['xul'])
            d['yul'] = float(d['yul'])
            d['rotation'] = float(d['rotation'])

            # convert the model.reference text to a lenuni value
            # (these are the model length units)
            if 'length_units' in d.keys():
                d['lenuni'] = ModelGrid.lenuni_values[d['length_units']]
            if 'time_units' in d.keys():
                d['itmuni'] = itmuni_values[d['time_units']]
            if 'start_date' in d.keys():
                start_datetime = d.pop('start_date')
                if 'start_time' in d.keys():
                    start_datetime += ' {}'.format(d.pop('start_time'))
                d['start_datetime'] = start_datetime
            if 'epsg' in d.keys():
                try:
                    d['epsg'] = int(d['epsg'])
                except Exception as e:
                    raise Exception(
                        "error reading epsg code from file:\n" + str(e))
            # this prioritizes epsg over proj4 if both are given
            # (otherwise 'proj4' entry will be dropped below)
            elif 'proj4' in d.keys():
                d['proj4_str'] = d['proj4']

            # drop any other items that aren't used in sr class
            d = {k: v for k, v in d.items() if
                 k.lower() in ModelGrid.defaults.keys()
                 or k.lower() in {'epsg', 'start_datetime', 'itmuni',
                                  'source'}}
            return d
        else:
            return None

    @property
    def extent(self):
        raise NotImplementedError(
            'must define extent in child '
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
    def xcenters(self):
        return self.cellcenters[0]

    @property
    def ycenters(self):
        return self.cellcenters[1]

    @property
    def zcenters(self):
        return self.cellcenters[2]

    @property
    def xyzgrid(self):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @property
    def cellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def grid_lines(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

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

    @abc.abstractmethod
    def write_gridSpec(self, filename):
        raise NotImplementedError(
            'must define write_gridSpec in child '
            'class to use this base class')

    # more specific name for this?
    @abc.abstractmethod
    def interpolate(self, a, xi, method='nearest'):
        """
        Use the griddata method to interpolate values from an array onto the
        points defined in xi.  For any values outside of the grid, use
        'nearest' to find a value for them.

        Parameters
        ----------
        a : numpy.ndarray
            array to interpolate from.  It must be of size nrow, ncol
        xi : numpy.ndarray
            array containing x and y point coordinates of size (npts, 2). xi
            also works with broadcasting so that if a is a 2d array, then
            xi can be passed in as (xgrid, ygrid).
        method : {'linear', 'nearest', 'cubic'}
            method to use for interpolation (default is 'nearest')

        Returns
        -------
        b : numpy.ndarray
            array of size (npts)

        """
        raise NotImplementedError('must define interpolate in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_2d_vertex_connectivity(self):
        """
        Create the cell 2d vertices array and the iverts index array.  These
        are the same form as the ones used to instantiate an unstructured
        spatial reference.

        Returns
        -------

        verts : ndarray
            array of x and y coordinates for the grid vertices

        iverts : list
            a list with a list of vertex indices for each cell in clockwise
            order starting with the upper left corner

        """
        raise NotImplementedError('must define get_2d_vertex_connectivity in '
                                  'child class to use this base class')

    def get_rc(self, x, y):
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        r : row or sequence of rows (zero-based)
        c : column or sequence of columns (zero-based)
        """
        raise NotImplementedError('must define get_rc in child '
                                  'class to use this base class')

    def plot_grid_lines(self, **kwargs):
        """
        Get a LineCollection of the model grid in
        model coordinates

        Parameters
            **kwargs: matplotlib.pyplot keyword arguments

        Returns
            matplotlib.collections.LineCollection
        """
        from flopy.plot.plotbase import PlotMapView

        map = PlotMapView(modelgrid=self)
        lc = map.plot_grid(**kwargs)
        return lc

    def plot_array(self, a, ax=None, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        from flopy.plot.plotutil import PlotUtilities

        ax = PlotUtilities._plot_array_helper(a, sr=self.sr, axes=ax, **kwargs)
        return ax

    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        from flopy.plot import PlotMapView

        kwargs['ax'] = ax
        map = PlotMapView(sr=self.sr)
        contour_set = map.contour_array(a=a, **kwargs)

        return contour_set

    @abc.abstractmethod
    # what is the difference between this and the one without "shared" below?
    def get_3d_shared_vertex_connectivity(self):
        raise NotImplementedError('must define '
                                  'get_3d_shared_vertex_connectivity in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_3d_vertex_connectivity(self):
        raise NotImplementedError('must define get_3d_vertex_connectivity in '
                                  'child class to use this base class')