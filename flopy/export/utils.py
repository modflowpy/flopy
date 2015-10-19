import os
import numpy as np
from flopy.utils import util_2d, util_3d, transient_2d,mflist

from . import NetCdf


NC_UNITS_FORMAT = {"hk":"{0}/{1}","sy":"","ss":"1/{0}","rech":"{0}/{1}"}
NC_PRECISION_TYPE = {np.float32:"f4",np.int:"i4"}


def mflist_helper(f,mfl):
    assert isinstance(mfl,mflist)\
                      ,"mflist_helper only helps mflist instances"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,mfl.model)

    if isinstance(f,NetCdf):
        base_name = mfl.package.name[0].lower()
        m4d = mfl.masked_4D_arrays
        for name,array in mfl.masked_4D_arrays.items():
            var_name = base_name + '_' + name
            units = None
            if var_name in NC_UNITS_FORMAT:
                units = NC_UNITS_FORMAT[var_name].format(f.grid_units,f.time_units)
            precision_str = NC_PRECISION_TYPE[mfl.dtype[name].type]
            attribs = {"long_name":"flopy.mflist instance of {0}".format(var_name)}
            if units is not None:
                attribs["units"] = units
            var = f.create_variable(var_name,attribs,precision_str=precision_str,dimensions=("time","layer","y","x"))
            var[:] = array
        return f
    else:
        raise NotImplementedError("transient2d_helper only for netcdf (*.nc) ")


def transient2d_helper(f,t2d,min_valid=-1.0e+9, max_valid=1.0e+9):
    assert isinstance(t2d,transient_2d)\
                      ,"transient2d_helper only helps transient_2d instances"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,t2d.model)

    if isinstance(f,NetCdf):
        # mask the array - assume layer 1 ibound is a good mask
        array = t2d.array
        if t2d.model.bas6 is not None:
            array[:,t2d.model.bas6.ibound.array[0] == 0] = f.fillvalue
        array[array<=min_valid] = f.fillvalue
        array[array>=max_valid] = f.fillvalue

        units = None
        name = t2d.name_base.replace('_','')
        if name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[name].format(f.grid_units,f.time_units)
        precision_str = NC_PRECISION_TYPE[t2d.dtype]
        attribs = {"long_name":"flopy.transient_2d instance of {0}".format(name)}
        if units is not None:
            attribs["units"] = units
        var = f.create_variable(name,attribs,precision_str=precision_str,dimensions=("layer","y","x"))
        var[:] = array
        return f

    else:
        raise NotImplementedError("transient2d_helper only for netcdf (*.nc) ")

def util3d_helper(f,u3d,min_valid=-1.0e+9, max_valid=1.0e+9):
    assert isinstance(u3d,util_3d),"util3d_helper only helps util_3d instances"
    assert len(u3d.shape) == 3,"util3d_helper only supports 3D arrays"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,u3d.model)

    if isinstance(f,NetCdf):
        # mask the array
        array = u3d.array
        if u3d.model.bas6 is not None:
            array[u3d.model.bas6.ibound.array == 0] = f.fillvalue
        array[array<=min_valid] = f.fillvalue
        array[array>=max_valid] = f.fillvalue

        units = None
        name = u3d.name[0].split()[0]
        if name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[name].format(f.grid_units,f.time_units)

        precision_str = NC_PRECISION_TYPE[u3d.dtype]

        attribs = {"long_name":"flopy.util_3d instance of {0}".format(name)}
        if units is not None:
            attribs["units"] = units
        var = f.create_variable(name,attribs,precision_str=precision_str,dimensions=("layer","y","x"))
        var[:] = array
        return f

    else:
        raise NotImplementedError("util3d_helper only for netcdf (*.nc) ")


def util2d_helper(f,u2d,min_valid=-1.0e+9, max_valid=1.0e+9):

    assert isinstance(u2d,util_2d),"util2d_helper only helps util_2d instances"

    assert len(u2d.shape) == 2,"util2d_helper only supports 2D arrays"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,u2d.model)

    if isinstance(f,NetCdf):


        # try to mask the array - assume layer 1 ibound is a good mask
        array = u2d.array
        if u2d.model.bas6 is not None:
            array[u2d.model.bas6.ibound.array[0,:,:] == 0] = f.fillvalue
        array[array<=min_valid] = f.fillvalue
        array[array>=max_valid] = f.fillvalue

        units = None
        if u2d.name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[u2d.name].format(f.grid_units,f.time_units)

        precision_str = NC_PRECISION_TYPE[u2d.dtype]

        attribs = {"long_name":"flopy.util_2d instance of {0}".format(u2d.name)}
        if units is not None:
            attribs["units"] = units
        var = f.create_variable(u2d.name,attribs,precision_str=precision_str,dimensions=("y","x"))
        var[:] = array
        return f

    else:
        raise NotImplementedError("util2d_helper only for netcdf (*.nc) ")