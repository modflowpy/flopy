"""
Module for exporting and importing flopy model attributes
"""
import copy
import shutil
import json
import numpy as np
import os
import sys
import warnings
from ..datbase import DataType, DataInterface, DataListInterface
from ..utils import Util2d, Util3d, Transient2d, MfList, SpatialReference


def import_shapefile():
    try:
        import shapefile as sf
        return sf
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")


def shapefile_version(sf):
    """
    Return the shapefile major version number
    Parameters
    ----------
    sf : shapefile package

    Returns
    -------
    int
    """
    return int(sf.__version__.split('.')[0])


def write_gridlines_shapefile(filename, mg):
    """
    Write a polyline shapefile of the grid lines - a lightweight alternative
    to polygons.

    Parameters
    ----------
    filename : string
        name of the shapefile to write
    mg : model grid

    Returns
    -------
    None

    """
    shapefile = import_shapefile()
    sfv = shapefile_version(shapefile)
    if sfv < 2:
        wr = shapefile.Writer(shapeType=shapefile.POLYLINE)
    else:
        wr = shapefile.Writer(filename, shapeType=shapefile.POLYLINE)
    wr.field("number", "N", 18, 0)
    if isinstance(mg, SpatialReference):
        grid_lines = mg.get_grid_lines()
        warnings.warn("SpatialReference has been deprecated. Use StructuredGrid"
                      " instead.",
                      category=DeprecationWarning)
    else:
        grid_lines = mg.grid_lines()
    for i, line in enumerate(grid_lines):
        wr.poly([line])
        wr.record(i)
    if sfv < 2:
        wr.save(filename)
    else:
        wr.close()
    return


def write_grid_shapefile(filename, mg, array_dict, nan_val=None):#-1.0e9):
    """
    Write a grid shapefile array_dict attributes.

    Parameters
    ----------
    filename : string
        name of the shapefile to write
    mg : model grid instance
        object for model grid
    array_dict : dict
       Dictionary of name and 2D array pairs.  Additional 2D arrays to add as
       attributes to the grid shapefile.

    Returns
    -------
    None

    """

    shapefile = import_shapefile()
    sfv = shapefile_version(shapefile)
    if sfv < 2:
        wr = shapefile.Writer(shapeType=shapefile.POLYGON)
    else:
        wr = shapefile.Writer(filename, shapeType=shapefile.POLYGON)
    if isinstance(mg, SpatialReference):
        warnings.warn("SpatialReference has been deprecated. Use StructuredGrid"
                      " instead.",
                      category=DeprecationWarning)
        wr.field("row", "N", 10, 0)
        wr.field("column", "N", 10, 0)
    elif mg.grid_type == 'structured':
        wr.field("row", "N", 10, 0)
        wr.field("column", "N", 10, 0)
    elif mg.grid_type == 'vertex':
        wr.field("node", "N", 10, 0)
    else:
        raise Exception('Grid type {} not supported.'.format(mg.grid_type))

    arrays = []
    names = list(array_dict.keys())
    names.sort()
    # for name,array in array_dict.items():
    for name in names:
        array = array_dict[name]
        if array.ndim == 3:
            assert array.shape[0] == 1
            array = array[0, :, :]
        assert array.shape == (mg.nrow, mg.ncol)
        if array.dtype in [np.float, np.float32, np.float64]:
            array[np.where(np.isnan(array))] = nan_val
        else:
            j=2
        #if array.dtype in [np.int, np.int32, np.int64]:
        #    wr.field(name, "N", 18, 0)
        #else:
        #    wr.field(name, "F", 18, 12)
        wr.field(name, *get_pyshp_field_info(array.dtype.name))
        arrays.append(array)

    if isinstance(mg, SpatialReference) or mg.grid_type == 'structured':
        for i in range(mg.nrow):
            for j in range(mg.ncol):
                try:
                    pts = mg.get_cell_vertices(i, j)
                except AttributeError:
                    # support old style SR object
                    pts = mg.get_vertices(i, j)

                wr.poly([pts])
                rec = [i + 1, j + 1]
                for array in arrays:
                    rec.append(array[i, j])
                wr.record(*rec)
    elif mg.grid_type == 'vertex':
        for i in range(mg.ncpl):
            pts = mg.get_cell_vertices(i)

            wr.poly([pts])
            rec = [i + 1]
            for array in arrays:
                rec.append(array[i])
            wr.record(*rec)

    # close or write the file
    if sfv < 2:
        wr.save(filename)
    else:
        wr.close()
    print('wrote {}'.format(filename))
    return


def write_grid_shapefile2(filename, mg, array_dict, nan_val=np.nan,#-1.0e9,
                          epsg=None, prj=None):

    shapefile = import_shapefile()
    sfv = shapefile_version(shapefile)
    if sfv < 2:
        w = shapefile.Writer(shapeType=shapefile.POLYGON)
    else:
        w = shapefile.Writer(filename, shapeType=shapefile.POLYGON)
    w.autoBalance = 1

    if isinstance(mg, SpatialReference):
        verts = copy.deepcopy(mg.vertices)
        warnings.warn("SpatialReference has been deprecated. Use StructuredGrid"
                      " instead.",
                      category=DeprecationWarning)
    elif mg.grid_type == 'structured':
        verts = [mg.get_cell_vertices(i, j)
                 for i in range(mg.nrow)
                 for j in range(mg.ncol)]
    elif mg.grid_type == 'vertex':
        verts = [mg.get_cell_vertices(cellid)
                 for cellid in range(mg.ncpl)]
    else:
        raise Exception('Grid type {} not supported.'.format(mg.grid_type))


    # set up the attribute fields
    if isinstance(mg, SpatialReference) or mg.grid_type == 'structured':
        names = ['node', 'row', 'column'] + list(array_dict.keys())
        names = enforce_10ch_limit(names)
        dtypes = [('node', np.dtype('int')),
                  ('row', np.dtype('int')),
                  ('column', np.dtype('int'))] + \
                 [(enforce_10ch_limit([name])[0], arr.dtype)
                  for name, arr in array_dict.items()]
    elif mg.grid_type == 'vertex':
        names = ['node'] + list(array_dict.keys())
        names = enforce_10ch_limit(names)
        dtypes = [('node', np.dtype('int'))] + \
                 [(enforce_10ch_limit([name])[0], arr.dtype)
                  for name, arr in array_dict.items()]

    fieldinfo = {name: get_pyshp_field_info(dtype.name) for name, dtype in dtypes}
    for n in names:
        w.field(n, *fieldinfo[n])

    if isinstance(mg, SpatialReference) or mg.grid_type == 'structured':
        # set-up array of attributes of shape ncells x nattributes
        node = list(range(1, mg.ncol * mg.nrow + 1))
        col = list(range(1, mg.ncol + 1)) * mg.nrow
        row = sorted(list(range(1, mg.nrow + 1)) * mg.ncol)
        at = np.vstack(
            [node, row, col] +
            [arr.ravel() for arr in array_dict.values()]).transpose()
        if at.dtype in [np.float, np.float32, np.float64]:
            at[np.isnan(at)] = nan_val
    elif mg.grid_type == 'vertex':
        # set-up array of attributes of shape ncells x nattributes
        node = list(range(1, mg.ncpl + 1))
        at = np.vstack(
            [node] +
            [arr.ravel() for arr in array_dict.values()]).transpose()
    if at.dtype in [np.float, np.float32, np.float64]:
        at[np.isnan(at)] = nan_val

    for i, r in enumerate(at):
        w.poly([verts[i]])
        w.record(*r)

    # close
    if sfv < 2:
        w.save(filename)
    else:
        w.close()
    print('wrote {}'.format(filename))
    # write the projection file
    write_prj(filename, mg, epsg, prj)
    return


def model_attributes_to_shapefile(filename, ml, package_names=None,
                                  array_dict=None,
                                  **kwargs):
    """
    Wrapper function for writing a shapefile of model data.  If package_names is
    not None, then search through the requested packages looking for arrays that
    can be added to the shapefile as attributes

    Parameters
    ----------
    filename : string
        name of the shapefile to write
    ml : flopy.mbase
        model instance
    package_names : list of package names (e.g. ["dis","lpf"])
        Packages to export data arrays to shapefile. (default is None)
    array_dict : dict of {name:2D array} pairs
       Additional 2D arrays to add as attributes to the shapefile. (default is None)


    Returns
    -------
    None

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> flopy.utils.model_attributes_to_shapefile('model.shp', m)

    """

    if array_dict is None:
        array_dict = {}

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
    else:
        package_names = [pak.name[0] for pak in ml.packagelist]

    grid = ml.modelgrid
    horz_shape = grid.shape[1:]
    for pname in package_names:
        pak = ml.get_package(pname)
        attrs = dir(pak)
        if pak is not None:
            if 'sr' in attrs:
                attrs.remove('sr')
            if 'start_datetime' in attrs:
                attrs.remove('start_datetime')
            for attr in attrs:
                a = pak.__getattribute__(attr)
                if a is None or not hasattr(a, 'data_type') or a.name == 'thickness':
                    continue
                if a.data_type == DataType.array2d and a.array.shape == horz_shape:
                    name = shape_attr_name(a.name, keep_layer=True)
                    #name = a.name.lower()
                    array_dict[name] = a.array
                elif a.data_type == DataType.array3d: #elif isinstance(a, Util3d):
                    try: # Not sure how best to check if an object has array data
                        assert a.array is not None
                    except:
                        print('Failed to get data for {} array, {} package'.format(a.name,
                                                                                   pak.name[0]))
                        continue
                    if isinstance(a.name, list) and a.name[0] == 'thickness':
                        continue
                    for ilay in range(a.array.shape[0]):
                        try:
                            arr = a[ilay].array
                        except:
                            arr = a[ilay]

                        if isinstance(a, Util3d):
                            aname = shape_attr_name(a[ilay].name)
                        else:
                            aname = a.name

                        if arr.shape == (1,) + horz_shape:
                            # fix for mf6 case.  TODO: fix this in the mf6 code
                            arr = arr[0]
                        assert arr.shape == horz_shape
                        name = '{}_{}'.format(aname, ilay + 1)
                        array_dict[name] = arr
                elif a.data_type == DataType.transient2d:#elif isinstance(a, Transient2d):
                    try: # Not sure how best to check if an object has array data
                        assert a.array is not None
                    except:
                        print('Failed to get data for {} array, {} package'.format(a.name,
                                                                                   pak.name[0]))
                        continue
                    for kper in range(a.array.shape[0]):
                        name = '{}{}'.format(
                            shape_attr_name(a.name), kper + 1)
                        arr = a.array[kper][0]
                        assert arr.shape == horz_shape
                        array_dict[name] = arr
                elif a.data_type == DataType.transientlist: #elif isinstance(a, MfList):
                    try:
                        list(a.masked_4D_arrays_itr())
                    except:
                        continue
                    for name, array in a.masked_4D_arrays_itr():
                        for kper in range(array.shape[0]):
                            for k in range(array.shape[1]):
                                n = shape_attr_name(name, length=4)
                                aname = "{}{}{}".format(n, k + 1, kper + 1)
                                arr = array[kper][k]
                                assert arr.shape == horz_shape
                                if np.all(np.isnan(arr)):
                                    continue
                                array_dict[aname] = arr
                elif isinstance(a, list):
                    for v in a:
                        if isinstance(a, DataInterface) and \
                                v.data_type == DataType.array3d:
                            for ilay in range(a.model.modelgrid.nlay):
                                u2d = a[ilay]
                                name = '{}_{}'.format(
                                    shape_attr_name(u2d.name), ilay + 1)
                                arr = u2d.array
                                assert arr.shape == horz_shape
                                array_dict[name] = arr
    # write data arrays to a shapefile
    write_grid_shapefile2(filename, ml.modelgrid, array_dict)
    epsg = kwargs.get('epsg', None)
    prj = kwargs.get('prj', None)
    write_prj(filename, ml.modelgrid, epsg, prj)


def shape_attr_name(name, length=6, keep_layer=False):
    """
    Function for to format an array name to a maximum of 10 characters to
    conform with ESRI shapefile maximum attribute name length

    Parameters
    ----------
    name : string
        data array name
    length : int
        maximum length of string to return. Value passed to function is
        overridden and set to 10 if keep_layer=True. (default is 6)
    keep_layer : bool
        Boolean that determines if layer number in name should be retained.
        (default is False)


    Returns
    -------
    String

    Examples
    --------

    >>> import flopy
    >>> name = flopy.utils.shape_attr_name('averylongstring')
    >>> name
    >>> 'averyl'

    """
    # kludges
    if name == 'model_top':
        name = 'top'
    # replace spaces with "_"
    n = name.lower().replace(' ', '_')
    # exclude "_layer_X" portion of string
    if keep_layer:
        length = 10
        n = n.replace('_layer', '_')
    else:
        try:
            idx = n.index('_layer')
            n = n[:idx]
        except:
            pass

    if len(n) > length:
        n = n[:length]
    return n


def enforce_10ch_limit(names):
    """Enforce 10 character limit for fieldnames.
    Add suffix for duplicate names starting at 0.

    Parameters
    ----------
    names : list of strings

    Returns
    -------
    names : list of unique strings of len <= 10.
    """
    names = [n[:5] + n[-4:] + '_' if len(n) > 10 else n
             for n in names]
    dups = {x: names.count(x) for x in names}
    suffix = {n: list(range(cnt)) for n, cnt in dups.items() if cnt > 1}
    for i, n in enumerate(names):
        if dups[n] > 1:
            names[i] = n[:9] + str(suffix[n].pop(0))
    return names


def get_pyshp_field_info(dtypename):
    """Get pyshp dtype information for a given numpy dtype.
    """
    fields = {'int': ('N', 18, 0),
              '<i': ('N', 18, 0),
              'float': ('F', 20, 12),
              '<f': ('F', 20, 12),
              'bool': ('L', 1),
              'b1': ('L', 1),
              'str': ('C', 50),
              'object': ('C', 50)}
    k = [k for k in fields.keys() if k in dtypename.lower()]
    if len(k) == 1:
        return fields[k[0]]
    else:
        return fields['str']


def get_pyshp_field_dtypes(code):
    """Returns a numpy dtype for a pyshp field type."""
    dtypes = {'N': np.int,
              'F': np.float,
              'L': np.bool,
              'C': np.object}
    return dtypes.get(code, np.object)


def shp2recarray(shpname):
    """Read a shapefile into a numpy recarray.

    Parameters
    ----------
    shpname : str
        ESRI Shapefile.

    Returns
    -------
    recarray : np.recarray

    """
    try:
        import shapefile as sf
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")
    from ..utils.geometry import shape

    sfobj = sf.Reader(shpname)
    dtype = [(str(f[0]), get_pyshp_field_dtypes(f[1])) for f in sfobj.fields[1:]]

    geoms = [shape(s) for s in sfobj.iterShapes()]
    records = [tuple(r) + (geoms[i],) for i, r in
               enumerate(sfobj.iterRecords())]
    dtype += [('geometry', np.object)]
    # recfunctions.append_fields(array, names='tmp1', data=col1,
    #                                           asrecarray=True)

    recarray = np.array(records, dtype=dtype).view(np.recarray)
    return recarray


def recarray2shp(recarray, geoms, shpname='recarray.shp', mg=None,
                 epsg=None, prj=None,
                 **kwargs):
    """
    Write a numpy record array to a shapefile, using a corresponding
    list of geometries.

    Parameters
    ----------
    recarray : np.recarray
        Numpy record array with attribute information that will go in the shapefile
    geoms : list of flopy.utils.geometry objects
        The number of geometries in geoms must equal the number of records in
        recarray.
    shpname : str
        Path for the output shapefile
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.

    Notes
    -----
    Uses pyshp.
    epsg code requires an internet connection the first time to get the projection
    file text from spatialreference.org, but then stashes the text in the file
    epsgref.json (located in the user's data directory) for subsequent use. See
    flopy.reference for more details.

    """


    if len(recarray) != len(geoms):
        raise IndexError(
            'Number of geometries must equal the number of records!')

    if len(recarray) == 0:
        raise Exception("Recarray is empty")

    geomtype = None
    for g in geoms:
        try:
            geomtype = g.shapeType
        except:
            continue

    # set up for pyshp 1 or 2
    shapefile = import_shapefile()
    sfv = shapefile_version(shapefile)
    if sfv < 2:
        w = shapefile.Writer(shapeType=geomtype)
    else:
        w = shapefile.Writer(shpname, shapeType=geomtype)
    w.autoBalance = 1

    # set up the attribute fields
    names = enforce_10ch_limit(recarray.dtype.names)
    for i, npdtype in enumerate(recarray.dtype.descr):
        key = names[i]
        if not isinstance(key, str):
            key = str(key)
        w.field(key, *get_pyshp_field_info(npdtype[1]))

    # write the geometry and attributes for each record
    ralist = recarray.tolist()
    if geomtype == shapefile.POLYGON:
        for i, r in enumerate(ralist):
            w.poly(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == shapefile.POLYLINE:
        for i, r in enumerate(ralist):
            w.line(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == shapefile.POINT:
        # pyshp version 2.x w.point() method can only take x and y
        # code will need to be refactored in order to write POINTZ
        # shapes with the z attribute.  The pyshp version 1.x
        # method w.point() took a z attribute, but only wrote it if
        # the shapeType was shapefile.POINTZ, which it is not for
        # flopy, even if the point is 3D.
        if sfv < 2:
            for i, r in enumerate(ralist):
                w.point(*geoms[i].pyshp_parts)
                w.record(*r)
        else:
            for i, r in enumerate(ralist):
                w.point(*geoms[i].pyshp_parts[:2])
                w.record(*r)

    if sfv < 2:
        w.save(shpname)
    else:
        w.close()
    write_prj(shpname, mg, epsg, prj)
    print('wrote {}'.format(shpname))
    return


def write_prj(shpname, mg=None, epsg=None, prj=None,
              wkt_string=None):
    # projection file name
    prjname = shpname.replace('.shp', '.prj')

    # figure which CRS option to use
    # prioritize args over grid reference
    # no proj4 option because it is too difficult
    # to create prjfile from proj4 string without OGR
    prjtxt = wkt_string
    if epsg is not None:
        prjtxt = CRS.getprj(epsg)
    # copy a supplied prj file
    elif prj is not None:
        shutil.copy(prj, prjname)

    elif mg is not None:
        if mg.epsg is not None:
            prjtxt = CRS.getprj(mg.epsg)

    else:
        print('No CRS information for writing a .prj file.\n'
              'Supply an epsg code or .prj file path to the '
              'model spatial reference or .export() method.'
              '(writing .prj files from proj4 strings not supported)'
              )
    if prjtxt is not None:
        with open(prjname, 'w') as output:
            output.write(prjtxt)


class CRS(object):
    """
    Container to parse and store coordinate reference system parameters,
    and translate between different formats.
    """

    def __init__(self, prj=None, esri_wkt=None, epsg=None):

        self.wktstr = None
        if prj is not None:
            with open(prj) as input:
                self.wktstr = input.read()
        elif esri_wkt is not None:
            self.wktstr = esri_wkt
        elif epsg is not None:
            wktstr = CRS.getprj(epsg)
            if wktstr is not None:
                self.wktstr = wktstr
        if self.wktstr is not None:
            self.parse_wkt()

    @property
    def crs(self):
        """Dict mapping crs attibutes to proj4 parameters"""
        proj = None
        if self.projcs is not None:
            # projection
            if 'mercator' in self.projcs.lower():
                if 'transvers' in self.projcs.lower() or \
                        'tm' in self.projcs.lower():
                    proj = 'tmerc'
                else:
                    proj = 'merc'
            elif 'utm' in self.projcs.lower() and \
                    'zone' in self.projcs.lower():
                proj = 'utm'
            elif 'stateplane' in self.projcs.lower():
                proj = 'lcc'
            elif 'lambert' and 'conformal' and 'conic' in self.projcs.lower():
                proj = 'lcc'
            elif 'albers' in self.projcs.lower():
                proj = 'aea'
        elif self.projcs is None and self.geogcs is not None:
            proj = 'longlat'

        # datum
        if 'NAD' in self.datum.lower() or \
                'north' in self.datum.lower() and \
                'america' in self.datum.lower():
            datum = 'nad'
            if '83' in self.datum.lower():
                datum += '83'
            elif '27' in self.datum.lower():
                datum += '27'
        elif '84' in self.datum.lower():
            datum = 'wgs84'

        # ellipse
        if '1866' in self.spheriod_name:
            ellps = 'clrk66'
        elif 'grs' in self.spheriod_name.lower():
            ellps = 'grs80'
        elif 'wgs' in self.spheriod_name.lower():
            ellps = 'wgs84'

        # prime meridian
        pm = self.primem[0].lower()

        return {'proj': proj,
                'datum': datum,
                'ellps': ellps,
                'a': self.semi_major_axis,
                'rf': self.inverse_flattening,
                'lat_0': self.latitude_of_origin,
                'lat_1': self.standard_parallel_1,
                'lat_2': self.standard_parallel_2,
                'lon_0': self.central_meridian,
                'k_0': self.scale_factor,
                'x_0': self.false_easting,
                'y_0': self.false_northing,
                'units': self.projcs_unit,
                'zone': self.utm_zone}

    @property
    def grid_mapping_attribs(self):
        """Map parameters for CF Grid Mappings
        http://http://cfconventions.org/cf-conventions/cf-conventions.html,
        Appendix F: Grid Mappings
        """
        if self.wktstr is not None:
            sp = [p for p in [self.standard_parallel_1,
                              self.standard_parallel_2]
                  if p is not None]
            sp = sp if len(sp) > 0 else None
            proj = self.crs['proj']
            names = {'aea': 'albers_conical_equal_area',
                     'aeqd': 'azimuthal_equidistant',
                     'laea': 'lambert_azimuthal_equal_area',
                     'longlat': 'latitude_longitude',
                     'lcc': 'lambert_conformal_conic',
                     'merc': 'mercator',
                     'tmerc': 'transverse_mercator',
                     'utm': 'transverse_mercator'}
            attribs = {'grid_mapping_name': names[proj],
                       'semi_major_axis': self.crs['a'],
                       'inverse_flattening': self.crs['rf'],
                       'standard_parallel': sp,
                       'longitude_of_central_meridian': self.crs['lon_0'],
                       'latitude_of_projection_origin': self.crs['lat_0'],
                       'scale_factor_at_projection_origin': self.crs['k_0'],
                       'false_easting': self.crs['x_0'],
                       'false_northing': self.crs['y_0']}
            return {k: v for k, v in attribs.items() if v is not None}

    @property
    def proj4(self):
        """Not implemented yet"""
        return None

    def parse_wkt(self):

        self.projcs = self._gettxt('PROJCS["', '"')
        self.utm_zone = None
        if self.projcs is not None and 'utm' in self.projcs.lower():
            self.utm_zone = self.projcs[-3:].lower().strip('n').strip('s')
        self.geogcs = self._gettxt('GEOGCS["', '"')
        self.datum = self._gettxt('DATUM["', '"')
        tmp = self._getgcsparam('SPHEROID')
        self.spheriod_name = tmp.pop(0)
        self.semi_major_axis = tmp.pop(0)
        self.inverse_flattening = tmp.pop(0)
        self.primem = self._getgcsparam('PRIMEM')
        self.gcs_unit = self._getgcsparam('UNIT')
        self.projection = self._gettxt('PROJECTION["', '"')
        self.latitude_of_origin = self._getvalue('latitude_of_origin')
        self.central_meridian = self._getvalue('central_meridian')
        self.standard_parallel_1 = self._getvalue('standard_parallel_1')
        self.standard_parallel_2 = self._getvalue('standard_parallel_2')
        self.scale_factor = self._getvalue('scale_factor')
        self.false_easting = self._getvalue('false_easting')
        self.false_northing = self._getvalue('false_northing')
        self.projcs_unit = self._getprojcs_unit()

    def _gettxt(self, s1, s2):
        s = self.wktstr.lower()
        strt = s.find(s1.lower())
        if strt >= 0:  # -1 indicates not found
            strt += len(s1)
            end = s[strt:].find(s2.lower()) + strt
            return self.wktstr[strt:end]

    def _getvalue(self, k):
        s = self.wktstr.lower()
        strt = s.find(k.lower())
        if strt >= 0:
            strt += len(k)
            end = s[strt:].find(']') + strt
            try:
                return float(self.wktstr[strt:end].split(',')[1])
            except:
                pass

    def _getgcsparam(self, txt):
        nvalues = 3 if txt.lower() == 'spheroid' else 2
        tmp = self._gettxt('{}["'.format(txt), ']')
        if tmp is not None:
            tmp = tmp.replace('"', '').split(',')
            name = tmp[0:1]
            values = list(map(float, tmp[1:nvalues]))
            return name + values
        else:
            return [None] * nvalues

    def _getprojcs_unit(self):
        if self.projcs is not None:
            tmp = self.wktstr.lower().split('unit["')[-1]
            uname, ufactor = tmp.strip().strip(']').split('",')[0:2]
            ufactor = float(ufactor.split(']')[0].split()[0].split(',')[0])
            return uname, ufactor
        return None, None

    @staticmethod
    def getprj(epsg, addlocalreference=True, text='esriwkt'):
        """Gets projection file (.prj) text for given epsg code from spatialreference.org
        See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        addlocalreference : boolean
            adds the projection file text associated with epsg to a local
            database, epsgref.py, located in site-packages.
        Returns
        -------
        prj : str
            text for a projection (*.prj) file.
        """
        epsgfile = EpsgReference()
        wktstr = None
        try:
            from epsgref import prj
            wktstr = prj.get(epsg)
        except:
            epsgfile.make()
        if wktstr is None:
            wktstr = CRS.get_spatialreference(epsg, text=text)
        if addlocalreference and wktstr is not None:
            epsgfile.add(epsg, wktstr)
        return wktstr

    @staticmethod
    def get_spatialreference(epsg, text='esriwkt'):
        """Gets text for given epsg code and text format from spatialreference.org
        Fetches the reference text using the url:
            http://spatialreference.org/ref/epsg/<epsg code>/<text>/
        See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        text : str
            string added to url
        Returns
        -------
        url : str
        """
        from flopy.utils.flopy_io import get_url_text

        epsg_categories = ['epsg', 'esri']
        for cat in epsg_categories:
            url = "http://spatialreference.org/ref/{2}/{0}/{1}/".format(epsg,
                                                                        text,
                                                                        cat)
            result = get_url_text(url)
            if result is not None:
                break
        if result is not None:
            return result.replace("\n", "")
        elif result is None and text != 'epsg':
            for cat in epsg_categories:
                error_msg = 'No internet connection or epsg code {0} ' \
                            'not found at http://spatialreference.org/ref/{2}/{0}/{1}'.format(
                    epsg,
                    text,
                    cat)
                print(error_msg)
        elif text == 'epsg':  # epsg code not listed on spatialreference.org may still work with pyproj
            return '+init=epsg:{}'.format(epsg)

    @staticmethod
    def getproj4(epsg):
        """Gets projection file (.prj) text for given epsg code from
        spatialreference.org. See: https://www.epsg-registry.org/
        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        Returns
        -------
        prj : str
            text for a projection (*.prj) file.
        """
        return CRS.get_spatialreference(epsg, text='proj4')


class EpsgReference:
    """Sets up a local database of projection file text referenced by epsg code.
    The database is located in the site packages folder in epsgref.py, which
    contains a dictionary, prj, of projection file text keyed by epsg value.
    """

    def __init__(self):
        try:
            from appdirs import user_data_dir
        except ImportError:
            user_data_dir = None
        if user_data_dir:
            datadir = user_data_dir('flopy')
        else:
            # if appdirs is not installed, use user's home directory
            datadir = os.path.join(os.path.expanduser('~'), '.flopy')
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
        dbname = 'epsgref.json'
        self.location = os.path.join(datadir, dbname)

    def _remove_pyc(self):
        try:  # get rid of pyc file
            os.remove(self.location + 'c')
        except:
            pass

    def make(self):
        if not os.path.exists(self.location):
            newfile = open(self.location, 'w')
            newfile.write('prj = {}\n')
            newfile.close()

    def reset(self, verbose=True):
        if os.path.exists(self.location):
            os.remove(self.location)
        self._remove_pyc()
        self.make()
        if verbose:
            print('Resetting {}'.format(self.location))

    def add(self, epsg, prj):
        """add an epsg code to epsgref.py"""
        data = {}
        data[epsg] = prj
        with open(self.location, 'w') as epsgfile:
            json.dump(data, epsgfile, indent=0)
            epsgfile.write('\n')

    def remove(self, epsg):
        """removes an epsg entry from epsgref.py"""
        from epsgref import prj
        self.reset(verbose=False)
        if epsg in prj.keys():
            del prj[epsg]
        for epsg, prj in prj.items():
            self.add(epsg, prj)

    @staticmethod
    def show():
        try:
            from importlib import reload
        except:
            from imp import reload
        import epsgref
        from epsgref import prj
        reload(epsgref)
        for k, v in prj.items():
            print('{}:\n{}\n'.format(k, v))