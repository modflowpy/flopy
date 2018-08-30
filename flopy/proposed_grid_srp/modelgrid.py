import abc
from enum import Enum
import os
import numpy as np
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


class CachedData():
    def __init__(self, data):
        self.data = data
        self.out_of_date = False

    def update_data(self, data):
        self.data = data
        self.out_of_date = False


class ModelGrid(object):
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid (DiscritizationType.DIS, DiscritizationType.DISV,
        DiscritizationType.DISU)
    sr : SpatialReference
        Spatial reference locates the grid in a coordinate system
    simulation_time : SimulationTime
        Simulation time provides time information for temporal grid data
    model_name : str
        Name of the model associated with this grid

    Attributes
    ----------
    xedge : ndarray
        array of column edges
    yedge : ndarray
        array of row edges

    Methods
    ----------
    xedgegrid : (point_type) : ndarray
        returns numpy meshgrid of x edges in reference frame defined by
        point_type
    yedgegrid : (point_type) : ndarray
        returns numpy meshgrid of y edges in reference frame defined by
        point_type
    tabular_data : (name_list, data, location_type) : data
        returns a pandas object with the data defined in name_list in the
        spatial representation defined by coord_type
    xcenters : (point_type) : ndarray
        returns x coordinate of cell centers
    ycenters : (point_type) : ndarray
        returns y coordinate of cell centers
    get_cell_centers : (point_type) : ndarray
        returns the cell centers of all models cells in the model grid.  if
        the model grid contains spatial reference information, the cell centers
        are in the coordinate system provided by the spatial reference
        information. otherwise the cell centers are based on a 0,0 location for
        the upper left corner of the model grid
    get_grid_lines : (point_type=PointType.spatialxyz) : list
        returns the model grid lines in a list.  each line is returned as a
        list containing two tuples in the format [(x1,y1), (x2,y2)] where
        x1,y1 and x2,y2 are the endpoints of the line.
    xyvertices : (point_type) : ndarray
        1D array of x and y coordinates of cell vertices for whole grid
        (single layer) in C-style (row-major) order
        (same as np.ravel())
    get_model_dim : () : list
        returns the dimensions of the model
    get_horizontal_cross_section_dim_names : () : list
        returns the appropriate dimension axis for a horizontal cross section
        based on the model discritization type
    get_model_dim_names : () : list
        returns the names of the model dimensions based on the model
        discritization type
    get_num_spatial_coordinates : () : int
        returns the number of spatial coordinates
    num_cells_per_layer : () : list
        returns the number of cells per model layer.  model discritization
        type must be DIS or DISV
    num_layers : () : int
        returns the number of layers in the model, if the model is layered
    num_cells : () : int
        returns the total number of cells in the model
    get_all_model_cells : () : list
        returns a list of all model cells, represented as a layer/row/column
        tuple, a layer/cellid tuple, or a cellid for the structured, layered
        vertex, and vertex grids respectively

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

    def __init__(self, grid_type, sr=None, simulation_time=None, lenuni=2,
                 origin_loc='ul', origin_x=0.0, origin_y=0.0, rotation=0.0):
        self.grid_type = grid_type
        self.use_ref_coords = True
        self._sr = sr
        self._lenuni = lenuni
        self._origin_loc = origin_loc
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._rotation = rotation
        self.sim_time = simulation_time
        self._cache_dict = {}

    ###########################
    # basic functions
    ###########################
    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        self._require_cache_updates()

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True

    @property
    def _use_ref_coordinates(self):
        return (self._origin_x != 0.0 or self._origin_y != 0.0 or
                self._origin_loc != 'ul' or self._rotation != 0.0) and \
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
    def xedges(self):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @property
    def yedges(self):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @property
    def xedgegrid(self):
        return self.xygrid[0]

    @property
    def yedgegrid(self):
        return self.xygrid[1]

    @property
    def xcenters(self):
        return self.cellcenters[0]

    @property
    def ycenters(self):
        return self.cellcenters[1]

    @property
    def xyvertices(self):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @property
    def cellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def gridlines(self):
        raise NotImplementedError(
            'must define get_grid_lines in child '
            'class to use this base class')

    @property
    def xygrid(self):
        raise NotImplementedError(
            'must define get_xygrid in child '
            'class to use this base class')

    @abc.abstractmethod
    def write_gridSpec(self, filename):
        raise NotImplementedError(
            'must define write_gridSpec in child '
            'class to use this base class')

    @abc.abstractmethod
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
        raise NotImplementedError(
            'must define plot_array in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_array(self, filename, a, nodata=-9999,
                     fieldname='value',
                     **kwargs):
        """Write a numpy array to Arc Ascii grid
        or shapefile with the model reference.

        Parameters
        ----------
        filename : str
            Path of output file. Export format is determined by
            file extention.
            '.asc'  Arc Ascii grid
            '.tif'  GeoTIFF (requries rasterio package)
            '.shp'  Shapefile
        a : 2D numpy.ndarray
            Array to export
        nodata : scalar
            Value to assign to np.nan entries (default -9999)
        fieldname : str
            Attribute field name for array values (shapefile export only).
            (default 'values')
        kwargs:
            keyword arguments to np.savetxt (ascii)
            rasterio.open (GeoTIFF)
            or flopy.export.shapefile_utils.write_grid_shapefile2

        Notes
        -----
        Rotated grids will be either be unrotated prior to export,
        using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
        included in their transform property (GeoTiff format). In either case
        the pixels will be displayed in the (unrotated) projected geographic coordinate system,
        so the pixels will no longer align exactly with the model grid
        (as displayed from a shapefile, for example). A key difference between
        Arc Ascii and GeoTiff (besides disk usage) is that the
        unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
        will have the same number of rows and pixels as the original.
        """
        raise NotImplementedError(
            'must define export_array in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_contours(self, filename, contours,
                        fieldname='level', epsg=None, prj=None,
                        **kwargs):
        """Convert matplotlib contour plot object to shapefile.

        Parameters
        ----------
        filename : str
            path of output shapefile
        contours : matplotlib.contour.QuadContourSet or list of them
            (object returned by matplotlib.pyplot.contour)
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

        Returns
        -------
        df : dataframe of shapefile contents
        """
        raise NotImplementedError(
            'must define export_contours in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move export-specific code to export
    def export_array_contours(self, filename, a,
                              fieldname='level',
                              interval=None,
                              levels=None,
                              maxlevels=1000,
                              epsg=None,
                              prj=None,
                              **kwargs):
        """Contour an array using matplotlib; write shapefile of contours.

        Parameters
        ----------
        filename : str
            Path of output file with '.shp' extention.
        a : 2D numpy array
            Array to contour
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp
        """
        raise NotImplementedError(
            'must define export_array_contours in child '
            'class to use this base class')

    @abc.abstractmethod
    # refactor and move plot-specific code
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
        raise NotImplementedError(
            'must define contour_array in child '
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

    #################################
    # from flopy for mf6 model grid
    #################################
    @abc.abstractmethod
    def get_row_array(self):
        raise NotImplementedError('must define get_row_array in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_column_array(self):
        raise NotImplementedError('must define get_column_array in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_layer_array(self):
        raise NotImplementedError('must define get_layer_array in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_model_dim(self):
        raise NotImplementedError('must define get_model_dim in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_model_dim_names(self):
        raise NotImplementedError('must define get_model_dim_names in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_horizontal_cross_section_dim_names(self):
        raise NotImplementedError('must define '
                                  'get_horizontal_cross_section_dim_names in '
                                  'child class to use this base class')

    @abc.abstractmethod
    def get_horizontal_cross_section_dim_arrays(self):
        raise NotImplementedError('must define '
                                  'get_horizontal_cross_section_dim_arrays in '
                                  'child class to use this base class')

    @abc.abstractmethod
    def get_model_dim_arrays(self):
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def num_cells_per_layer(self):
        raise NotImplementedError('must define num_cells_per_layer in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def num_cells(self, active_only=False):
        raise NotImplementedError('must define num_cells in child '
                                  'class to use this base class')

    @abc.abstractmethod
    def get_all_model_cells(self):
        raise NotImplementedError('must define get_all_model_cells in child '
                                  'class to use this base class')

    @abc.abstractmethod
    # move to export folder
    def load_shapefile_data(self, shapefile):
        # this will only support loading shapefiles with the same
        # spatial reference as the model grid
        raise NotImplementedError('must define load_shapefile_data in child '
                                  'class to use this base class')