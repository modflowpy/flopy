import abc
from enum import Enum
import numpy as np
from pandas import DataFrame


class GridType(Enum):
    """
    Enumeration of grid types
    """
    structured = 0
    layered_vertex = 1
    unlayered_vertex = 2


class LocationType(Enum):
    """
    Enumeration of location types
    """
    modelxyz = 0 # coordinates defined based on 0,0 point as upper-left most
                 # point of the model
    spatialxyz = 1 # coordinates defined by spatial reference
    cellid = 2  # cell ids
    layer_cellid = 3  # layer + cell id
    lrc = 4  # layer row column


class TimeType(Enum):
    """
    Enumeration of time types
    """
    calendar = 0 # year/month/day/hour/minute/second
    sp = 1 # stress period
    sp_ts = 2 # stress period time step


class PointType(Enum):
    """
    Enumeration of vertex types
    """
    modelxyz = 0
    spatialxyz = 1


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
    cell_centers = 2


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
    get_tabular_data : (name_list, data, location_type) : data
        returns a pandas object with the data defined in name_list in the
        spatial representation defined by coord_type
    xcell_centers : (point_type) : ndarray
        returns x coordinate of cell centers
    ycell_centers : (point_type) : ndarray
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

    def __init__(self, grid_type, sr=None, simulation_time=None,
                 model_name='', steady=False):
        self.grid_type = grid_type
        self._sr = sr
        self.sim_time = simulation_time
        self.model_name = model_name
        self._cache_dict = {}
        self._steady = steady

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

    @property
    def steady(self):
        return self._steady

    @abc.abstractmethod
    def get_tabular_data(self, data, coord_type=LocationType.spatialxyz):
        raise NotImplementedError(
            'must define get_model_dim_arrays in child '
            'class to use this base class')

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True
        self._sr.set_yedge(self.yedge)

    ############################
    # from spatial reference
    ############################
    @property
    def xedge(self):
        return self.get_edge_array()[0]

    @property
    def yedge(self):
        return self.get_edge_array()[1]

    def xedgegrid(self, point_type=PointType.spatialxyz):
        return self.get_xygrid(point_type)[0]

    def yedgegrid(self, point_type=PointType.spatialxyz):
        return self.get_xygrid(point_type)[1]

    def xcell_centers(self, point_type=PointType.spatialxyz):
        return self.get_cellcenters(point_type)[0]

    def ycell_centers(self, point_type=PointType.spatialxyz):
        return self.get_cellcenters(point_type)[1]

    @abc.abstractmethod
    def xyvertices(self, point_type=PointType.spatialxyz):
        raise NotImplementedError(
            'must define xyvertices in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_cellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_grid_lines(self, point_type=PointType.spatialxyz):
        raise NotImplementedError(
            'must define get_grid_lines in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_xygrid(self, point_type=PointType.spatialxyz):
        raise NotImplementedError(
            'must define get_xygrid in child '
            'class to use this base class')

    @abc.abstractmethod
    def get_edge_array(self):
        raise NotImplementedError(
            'must define get_edge_array in child '
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