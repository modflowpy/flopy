"""
Module for exporting and importing flopy model attributes
"""
import copy
import shutil
import numpy as np
import numpy.lib.recfunctions as rf
from ..datbase import DataType, DataInterface, DataListInterface
from ..utils import Util2d, Util3d, Transient2d, MfList
from ..utils.reference import getprj


def import_shapefile():
    try:
        import shapefile as sf
        return sf
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")


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
    # todo: needs update for model grid
    try:
        import shapefile
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")

    wr = shapefile.Writer(shapeType=shapefile.POLYLINE)
    wr.field("number", "N", 18, 0)
    for i, line in enumerate(mg.get_grid_lines()):
        wr.poly([line])
        wr.record(i)
    wr.save(filename)


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

    try:
        import shapefile
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")

    wr = shapefile.Writer(shapeType=shapefile.POLYGON)
    wr.field("row", "N", 10, 0)
    wr.field("column", "N", 10, 0)

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

    for i in range(mg.nrow):
        for j in range(mg.ncol):
            try:
                pts = mg.get_cell_vertices(i, j)
            except AttributeError:
                # support old style SR object
                pts = mg.get_vertices(i, j)

            wr.poly(parts=[pts])
            rec = [i + 1, j + 1]
            for array in arrays:
                rec.append(array[i, j])
            wr.record(*rec)
    wr.save(filename)
    print('wrote {}'.format(filename))

def write_grid_shapefile2(filename, mg, array_dict, nan_val=np.nan,#-1.0e9,
                          epsg=None, prj=None):
    sf = import_shapefile()
    try:
        verts = copy.deepcopy(mg.xyvertices())
    except AttributeError:
        # support old style SR for legacy!
        verts = copy.deepcopy(mg.vertices)

    w = sf.Writer(5)  # polygon
    w.autoBalance = 1
    # set up the attribute fields
    names = ['node', 'row', 'column'] + list(array_dict.keys())
    names = enforce_10ch_limit(names)
    dtypes = [('node', np.dtype('int')),
              ('row', np.dtype('int')),
              ('column', np.dtype('int'))] + \
             [(name, arr.dtype) for name, arr in array_dict.items()]
    fieldinfo = {name: get_pyshp_field_info(dtype.name) for name, dtype in dtypes}
    for n in names:
        w.field(n, *fieldinfo[n])

    # set-up array of attributes of shape ncells x nattributes
    node = list(range(1, mg.ncol * mg.nrow + 1))
    col = list(range(1, mg.ncol + 1)) * mg.nrow
    row = sorted(list(range(1, mg.nrow + 1)) * mg.ncol)
    at = np.vstack(
        [node, row, col] +
        [arr.ravel() for arr in array_dict.values()]).transpose()
    if at.dtype in [np.float, np.float32, np.float64]:
        at[np.isnan(at)] = nan_val

    for i, r in enumerate(at):
        w.poly([verts[i]])
        w.record(*r)
    w.save(filename)
    print('wrote {}'.format(filename))
    # write the projection file
    write_prj(filename, mg, epsg, prj)


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

    nrow, ncol = ml.modelgrid.nrow, ml.modelgrid.ncol
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
                if a is None or not hasattr(a, 'data_type'):
                    continue
                #if isinstance(a, Util2d) and a.shape == (ml.modelgrid.nrow, ml.modelgrid.ncol):
                if a.data_type == DataType.array2d and a.array.shape == (nrow, ncol):
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
                    for ilay in range(a.array.shape[0]):
                        if isinstance(a, Util3d):
                            try:
                                a[ilay].array
                            except:
                                j=2
                            arr = a[ilay].array
                            aname = shape_attr_name(a[ilay].name)
                        elif a.array is not None:
                            arr = a[ilay]
                            aname = shape_attr_name(a.name)
                        else:
                            continue
                        assert arr.shape == (nrow, ncol)
                        name = '{}_{:03d}'.format(aname, ilay + 1)
                        array_dict[name] = arr
                    #for i, u2d in enumerate(a):
                    #    # name = u2d.name.lower().replace(' ', '_')
                    #    name = shape_attr_name(u2d.name)
                    #    name += '_{:03d}'.format(i + 1)
                    #    array_dict[name] = u2d.array
                elif a.data_type == DataType.transient2d:#elif isinstance(a, Transient2d):
                    try: # Not sure how best to check if an object has array data
                        assert a.array is not None
                    except:
                        print('Failed to get data for {} array, {} package'.format(a.name,
                                                                                   pak.name[0]))
                        continue
                    for kper in range(a.array.shape[0]):
                        name = '{}{:03d}'.format(
                            shape_attr_name(a.name), kper + 1)
                        arr = a.array[kper][0]
                        assert arr.shape == (nrow, ncol)
                        array_dict[name] = arr

                    #kpers = list(a.transient_2ds.keys())
                    #kpers.sort()
                    #for kper in range(a.model.modelgrid.sim_time.nper):
                    #    u2d = a[kper]
                    #    name = '{}_{:03d}'.format(
                    #        shape_attr_name(u2d.name), kper + 1)
                    #    array_dict[name] = u2d.array
                    #for kper in kpers:
                    #    u2d = a.transient_2ds[kper]
                    #    # name = u2d.name.lower() + "_{0:03d}".format(kper + 1)
                    #    name = shape_attr_name(u2d.name)
                    #    name = "{}_{:03d}".format(name, kper + 1)
                    #    array_dict[name] = u2d.array
                elif a.data_type == DataType.transientlist: #elif isinstance(a, MfList):
                    try:
                        list(a.masked_4D_arrays_itr())
                    except:
                        continue
                    for name, array in a.masked_4D_arrays_itr():
                        for kper in range(array.shape[0]):
                            for k in range(array.shape[1]):
                                n = shape_attr_name(name, length=4)
                                aname = "{}{:03d}{:03d}".format(n, k + 1, kper + 1)
                                arr = array[kper][k]
                                assert arr.shape == (nrow, ncol)
                                if np.all(np.isnan(arr)):
                                    continue
                                array_dict[aname] = arr
                        #kpers = a.data.keys()
                        #for kper in kpers:
                        #    try:
                        #        arrays = a.to_array(kper)
                        #    except:
                        #        print("error exporting MfList in pak {0} to shapefile".format(pname))
                        #        continue
                        #    for name, array in arrays.items():
                        #        for k in range(array.shape[0]):
                        #            # aname = name+"{0:03d}_{1:02d}".format(kk, k)
                        #            n = shape_attr_name(name, length=4)
                        #            aname = "{}{:03d}{:03d}".format(n, k + 1, kper + 1)
                        #            array_dict[aname] = array[k]
                        #for name, array in arrays.items():
                        #    for k in range(array.shape[0]):
                        #        # aname = name + "{0:03d}{1:02d}".format(kper, k)
                        #        name = shape_attr_name(name, length=4)
                        #        aname = "{}{:03d}{:03d}".format(name, k + 1,
                        #                                        kper + 1)
                        #        array_dict[aname] = array[k].astype(np.float32)
                elif isinstance(a, list):
                    for v in a:
                        #if isinstance(v, Util3d):
                        #    for i, u2d in enumerate(v):
                        #        # name = u2d.name.lower().replace(' ', '_')
                        #        name = shape_attr_name(u2d.name)
                        #        name += '_{:03d}'.format(i + 1)
                        #        array_dict[name] = u2d.array
                        if isinstance(a, DataInterface) and \
                                v.data_type == DataType.array3d:
                            for ilay in range(a.model.modelgrid.nlay):
                                u2d = a[ilay]
                                name = '{}_{:03d}'.format(
                                    shape_attr_name(u2d.name), ilay + 1)
                                arr = u2d.array
                                assert arr.shape == (nrow, ncol)
                                array_dict[name] = arr
    # write data arrays to a shapefile
    write_grid_shapefile2(filename, ml.modelgrid, array_dict)
    epsg = kwargs.get('epsg', None)
    prj = kwargs.get('prj', None)
    write_prj(filename, ml.sr, epsg, prj)


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
    names = [n[:9] + '1' if len(n) > 10 else n
             for n in names]
    dups = {x: names.count(x) for x in names}
    suffix = {n: list(range(len(cnt))) for n, cnt in dups.items() if cnt > 1}
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
    dtype = [(f[0], get_pyshp_field_dtypes(f[1])) for f in sfobj.fields[1:]]

    geoms = [shape(s) for s in sfobj.iterShapes()]
    records = [tuple(r) + (geoms[i],) for i, r in
               enumerate(sfobj.iterRecords())]
    dtype += [('geometry', np.object)]
    # recfunctions.append_fields(array, names='tmp1', data=col1,
    #                                           asrecarray=True)

    recarray = np.array(records, dtype=dtype).view(np.recarray)
    return recarray


def recarray2shp(recarray, geoms, shpname='recarray.shp', sr=None,
                 epsg=None, prj=None,
                 **kwargs):
    """
    Write a numpy record array to a shapefile, using a corresponding
    list of geometries.

    Parameters
    ----------
    recarray : np.recarry
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
    epsgref.py (located in the site-packages folder) for subsequent use. See
    flopy.reference for more details.

    """
    try:
        import shapefile as sf
    except Exception as e:
        raise Exception("io.to_shapefile(): error " +
                        "importing shapefile - try pip install pyshp")
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
    w = sf.Writer(geomtype)
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
    if geomtype == 5:
        for i, r in enumerate(ralist):
            w.poly(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == 3:
        for i, r in enumerate(ralist):
            w.line(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == 1:
        for i, r in enumerate(ralist):
            w.point(*geoms[i].pyshp_parts)
            w.record(*r)
    w.save(shpname)
    write_prj(shpname, sr, epsg, prj)
    print('wrote {}'.format(shpname))


def write_prj(shpname, sr=None, epsg=None, prj=None,
              wkt_string=None):
    # projection file name
    prjname = shpname.replace('.shp', '.prj')

    from flopy.grid import SpatialReference
    from flopy.utils.reference import SpatialReference as OGsr

    if sr is not None and \
            not isinstance(sr, SpatialReference) and \
            not isinstance(sr, OGsr):
        try:
            sr = sr.sr
        except:
            raise TypeError('Unrecognized input type for "sr"')
    # figure which CRS option to use
    # prioritize args over SpatialReference
    # no proj4 option because it is too difficult
    # to create prjfile from proj4 string without OGR
    prjtxt = wkt_string
    if epsg is not None:
        prjtxt = getprj(epsg)
    # copy a supplied prj file
    elif prj is not None:
        shutil.copy(prj, prjname)
    elif sr is not None:
        if sr.wkt is not None:
            prjtxt = sr.wkt
    else:
        print('No CRS information for writing a .prj file.\n'
              'Supply an epsg code or .prj file path to the '
              'model spatial reference or .export() method.'
              '(writing .prj files from proj4 strings not supported)'
              )
    if prjtxt is not None:
        with open(prjname, 'w') as output:
            output.write(prjtxt)
