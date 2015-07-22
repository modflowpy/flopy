"""
Module for exporting and importing flopy model attributes
"""
import numpy as np
from flopy.utils import util_2d,util_3d,transient_2d

def results_to_netCDF(filename):
    raise NotImplementedError()


def grid_attributes_from_shapefile():
    raise NotImplementedError()


def write_grid_shapefile(filename, sr, array_dict,nan_val=-1.0e10):
    """
    Write a grid shapefile array_dict attributes.
    Parameters
    ----------
    filename : string
        name of the shapefile to write
    ml : flopy.mbase
        model instance
    package_names : (optional) list of package names (e.g. ["dis","lpf"])
        packages to scrap arrays out of for adding to shapefile
    array_dict : (optional) dict of {name:2D array} pairs
       additional 2D arrays to add as attributes to the grid shapefile


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
    #for name,array in array_dict.items():
    for name in names:
        array = array_dict[name]
        if array.ndim == 3:
            assert array.shape[0] == 1
            array = array[0,:,:]
        assert array.shape == (sr.nrow, sr.ncol)
        array[np.where(np.isnan(array))] = nan_val
        wr.field(name,"N",20,12)
        arrays.append(array)

    for i in range(sr.nrow):
        for j in range(sr.ncol):
            pts = sr.get_vertices(i,j)
            wr.poly(parts=[pts])
            rec = [i+1, j+1]
            for array in arrays:
                rec.append(array[i, j])
            wr.record(*rec)
    wr.save(filename)


def model_attributes_to_shapfile(filename,ml,package_names=None,array_dict=None):
    """
    Wrapper function for writing a shapefile for the model grid.  If package_names is not none,
    then search through the requested packages looking for arrays that can
    be added to the shapefile as attributes

    Parameters
    ----------
    filename : string
        name of the shapefile to write
    ml : flopy.mbase
        model instance
    package_names : (optional) list of package names (e.g. ["dis","lpf"])
        packages to scrap arrays out of for adding to shapefile
    array_dict : (optional) dict of {name:2D array} pairs
       additional 2D arrays to add as attributes to the grid shapefile


    Returns
    -------
    None

    """

    if array_dict is None:
        array_dict = {}

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
        for pname in package_names:
            pak = ml.get_package(pname)
            if pak is not None:
                attrs = dir(pak)
                for attr in attrs:
                    a = pak.__getattribute__(attr)
                    if isinstance(a, util_2d) and a.shape == (ml.nrow,
                                                              ml.ncol):
                        name = a.name.lower()
                        array_dict[name] = a.array
                    elif isinstance(a, util_3d):
                        for i,u2d in enumerate(a):
                            name = u2d.name.lower().replace(' ','_')
                            if "_layer" in name:
                                name = name.replace("_layer", '')
                            else:
                                name += '_{0:d}'.format(i+1)

                            array_dict[name] = u2d.array
                    elif isinstance(a,transient_2d):
                        kpers = list(a.transient_2ds.keys())
                        kpers.sort()
                        for kper in kpers:
                            u2d = a.transient_2ds[kper]
                            name = u2d.name.lower() + "_{0:d}".format(kper+1)
                            array_dict[name] = u2d.array

    write_grid_shapefile(filename,ml.dis.sr,array_dict)