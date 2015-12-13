import os
import numpy as np
from flopy.utils import Util2d, Util3d, Transient2d, MfList
from ..mbase import BaseModel
from ..pakbase import Package
from . import NetCdf
from . import shapefile_utils

NC_UNITS_FORMAT = {"hk": "{0}/{1}", "sy": "", "ss": "1/{0}", "rech": "{0}/{1}", "strt": "{0}",
                   "wel_flux": "{0}^3/{1}", "top": "{0}", "botm": "{0}", "thickness": "{0}",
                   "ghb_cond": "{0}/{1}^2", "ghb_bhead": "{0}", "transmissivity": "{0}^2/{1}",
                   "vertical_conductance": "{0}/{1}^2", "primary_storage_coefficient": "1/{1}",
                   "horizontal_hydraulic_conductivity": "{0}/{1}", "riv_cond": "1/{1}",
                   "riv_stage": "{0}", "riv_rbot": "{0}"}
NC_PRECISION_TYPE = {np.float32: "f4", np.int: "i4", np.int64: "i4"}


def datafile_helper(f, df):
    raise NotImplementedError()


def model_helper(f, ml, **kwargs):
    assert isinstance(ml,BaseModel)
    package_names = kwargs.get("package_names", None)
    if package_names is None:
        package_names = [pak.name[0] for pak in ml.packagelist]

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, ml)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(f, ml,
                                      package_names=package_names,
                                      **kwargs)

    elif isinstance(f,NetCdf):
        for pak in ml.packagelist:
            if pak.name[0] in package_names:
                f = pak.export(f)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def package_helper(f, pak, **kwargs):
    assert isinstance(pak,Package)
    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, pak.parent)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(f, pak.parent,
                                                      package_names=pak.name,
                                                      **kwargs)

    elif isinstance(f, NetCdf):
        attrs = dir(pak)
        for attr in attrs:
            if '__' in attr:
                continue
            a = pak.__getattribute__(attr)
            if isinstance(a, Util2d) and len(a.shape) == 2:
                f = util2d_helper(f, a, **kwargs)
            elif isinstance(a, Util3d):
                f = util3d_helper(f, a, **kwargs)
            elif isinstance(a, Transient2d):
                f = transient2d_helper(f, a, **kwargs)
            elif isinstance(a, MfList):
                f = mflist_helper(f, a, **kwargs)
            elif isinstance(a, list):
                for v in a:
                    if isinstance(v, Util3d):
                        f = util3d_helper(f, v, **kwargs)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))



def mflist_helper(f, mfl, **kwargs):
    """ export helper for MfList instances

    Parameters:
    -----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        mfl : MfList instance

    """
    assert isinstance(mfl, MfList) \
        , "mflist_helper only helps MfList instances"

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, mfl.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        kper = kwargs.get("kper",None)
        if mfl.sr is None:
            raise Exception("MfList.to_shapefile: SpatialReference not set")
        import flopy.utils.flopy_io as fio
        if kper is None:
            keys = mfl.data.keys()
            keys.sort()
        else:
            keys = [kper]
        array_dict = {}
        for kk in keys:
            arrays = mfl.to_array(kk)
            for name, array in arrays.items():
                for k in range(array.shape[0]):
                    #aname = name+"{0:03d}_{1:02d}".format(kk, k)
                    n = fio.shape_attr_name(name, length=4)
                    aname = "{}{:03d}{:03d}".format(n, k+1, int(kk)+1)
                    array_dict[aname] = array[k]
        shapefile_utils.write_grid_shapefile(f, mfl.sr, array_dict)

    elif isinstance(f, NetCdf):
        base_name = mfl.package.name[0].lower()
        f.log("getting 4D masked arrays for {0}".format(base_name))
        m4d = mfl.masked_4D_arrays
        f.log("getting 4D masked arrays for {0}".format(base_name))

        for name, array in m4d.items():
            array[np.isnan(array)] = f.fillvalue
            var_name = base_name + '_' + name
            units = None
            if var_name in NC_UNITS_FORMAT:
                units = NC_UNITS_FORMAT[var_name].format(f.grid_units, f.time_units)
            precision_str = NC_PRECISION_TYPE[mfl.dtype[name].type]
            attribs = {"long_name": "flopy.MfList instance of {0}".format(var_name)}
            attribs["coordinates"] = "time layer latitude longitude"
            if units is not None:
                attribs["units"] = units
            try:
                var = f.create_variable(var_name, attribs, precision_str=precision_str,
                                        dimensions=("time", "layer", "y", "x"))
            except Exception as e:
                estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
                f.logger.warn(estr)
                raise Exception(estr)
            try:
                var[:] = array
            except Exception as e:
                estr = "error setting array to variable {0}:\n{1}".format(var_name, str(e))
                f.logger.warn(estr)
                raise Exception(estr)
        return f
    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def transient2d_helper(f, t2d, **kwargs):
    """ export helper for Transient2d instances

    Parameters:
    -----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        t2d : Transient2d instance
        min_valid : minimum valid value
        max_valid : maximum valid value

    """

    assert isinstance(t2d, Transient2d)\
        , "transient2d_helper only helps Transient2d instances"

    min_valid = kwargs.get("min_valid", -1.0e+9)
    max_valid = kwargs.get("max_valid", 1.0e+9)

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, t2d.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        array_dict = {}
        for kper in range(t2d.model.nper):
            u2d = t2d[kper]
            name = '{}_{:03d}'.format(shapefile_utils.shape_attr_name(u2d.name), kper + 1)
            array_dict[name] = u2d.array
        shapefile_utils.write_grid_shapefile(f, t2d.model.dis.sr, array_dict)

    elif isinstance(f, NetCdf):
        # mask the array - assume layer 1 ibound is a good mask
        f.log("getting 4D array for {0}".format(t2d.name_base))
        array = t2d.array
        f.log("getting 4D array for {0}".format(t2d.name_base))

        if t2d.model.bas6 is not None:
            array[:, t2d.model.bas6.ibound.array[0] == 0] = f.fillvalue
        array[array <= min_valid] = f.fillvalue
        array[array >= max_valid] = f.fillvalue

        units = "unitless"
        var_name = t2d.name_base.replace('_', '')
        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units, f.time_units)
        precision_str = NC_PRECISION_TYPE[t2d.dtype]
        attribs = {"long_name": "flopy.Transient2d instance of {0}".format(var_name)}
        attribs["coordinates"] = "time latitude longitude"
        attribs["units"] = units
        try:
            var = f.create_variable(var_name, attribs, precision_str=precision_str,
                                    dimensions=("time", "y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def util3d_helper(f, u3d, **kwargs):
    """ export helper for Transient2d instances

    Parameters:
    -----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        u3d : Util3d instance
        min_valid : minimum valid value
        max_valid : maximum valid value

    """

    assert isinstance(u3d, Util3d), "util3d_helper only helps Util3d instances"
    assert len(u3d.shape) == 3, "util3d_helper only supports 3D arrays"

    min_valid = kwargs.get("min_valid", -1.0e+9)
    max_valid = kwargs.get("max_valid", 1.0e+9)

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, u3d.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        array_dict = {}
        for ilay in range(u3d.model.nlay):
            u2d = u3d[ilay]
            name = '{}_{:03d}'.format(shapefile_utils.shape_attr_name(u2d.name), ilay + 1)
            array_dict[name] = u2d.array
        shapefile_utils.write_grid_shapefile(f, u3d.model.dis.sr,
                             array_dict)

    elif isinstance(f, NetCdf):
        var_name = u3d.name[0].replace(' ', '_').lower()
        f.log("getting 3D array for {0}".format(var_name))
        array = u3d.array
        # this is for the crappy vcont in bcf6
        if array.shape != f.shape:
            f.log("broadcasting 3D array for {0}".format(var_name))
            full_array = np.empty(f.shape)
            full_array[:] = np.NaN
            full_array[:array.shape[0]] = array
            array = full_array
            f.log("broadcasting 3D array for {0}".format(var_name))
        f.log("getting 3D array for {0}".format(var_name))

        if u3d.model.bas6 is not None:
            array[u3d.model.bas6.ibound.array == 0] = f.fillvalue
        array[array <= min_valid] = f.fillvalue
        array[array >= max_valid] = f.fillvalue

        units = "unitless"
        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units, f.time_units)
        precision_str = NC_PRECISION_TYPE[u3d.dtype]
        attribs = {"long_name": "flopy.Util3d instance of {0}".format(var_name)}
        attribs["coordinates"] = "layer latitude longitude"
        attribs["units"] = units
        try:
            var = f.create_variable(var_name, attribs, precision_str=precision_str,
                                    dimensions=("layer", "y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def util2d_helper(f, u2d, **kwargs):
    """ export helper for Util2d instances

    Parameters:
    ----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        u2d : Util2d instance
        min_valid : minimum valid value
        max_valid : maximum valid value

    """
    assert isinstance(u2d, Util2d), "util2d_helper only helps Util2d instances"
    assert len(u2d.shape) == 2, "util2d_helper only supports 2D arrays"

    min_valid = kwargs.get("min_valid", -1.0e+9)
    max_valid = kwargs.get("max_valid", 1.0e+9)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        name = shapefile_utils.shape_attr_name(u2d.name, keep_layer=True)
        shapefile_utils.write_grid_shapefile(f, u2d.model.dis.sr, {name: u2d.array})

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, u2d.model)

    elif isinstance(f, NetCdf):

        # try to mask the array - assume layer 1 ibound is a good mask
        f.log("getting 2D array for {0}".format(u2d.name))
        array = u2d.array
        f.log("getting 2D array for {0}".format(u2d.name))

        if u2d.model.bas6 is not None:
            array[u2d.model.bas6.ibound.array[0, :, :] == 0] = f.fillvalue
        array[array <= min_valid] = f.fillvalue
        array[array >= max_valid] = f.fillvalue

        units = "unitless"
        var_name = u2d.name
        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units, f.time_units)
        precision_str = NC_PRECISION_TYPE[u2d.dtype]
        attribs = {"long_name": "flopy.Util2d instance of {0}".format(var_name)}
        attribs["coordinates"] = "latitude longitude"
        attribs["units"] = units
        try:
            var = f.create_variable(var_name, attribs, precision_str=precision_str,
                                    dimensions=("y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))
