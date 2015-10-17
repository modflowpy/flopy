import os
import numpy as np
from flopy.utils import util_2d, util_3d
from . import NetCdf


NC_UNITS_FORMAT = {"hk":"{0}/{1}","sy":"","ss":"1/{0}","rech":"{0}/{1}"}
NC_PRECISION_TYPE = {np.float32:"f4",np.int:"i4"}

def util3d_helper(f,u3d):
    assert isinstance(u3d,util_3d),"util3d_helper only helps util_3d instances"

    assert len(u3d.shape) == 3,"util3d_helper only supports 3D arrays"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,u3d.model)

    if isinstance(f,NetCdf):
        units = ''
        name = u3d.name[0].split()[0]
        if name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[name].format(f.grid_units,f.time_units)

        precision_str = NC_PRECISION_TYPE[u3d.dtype]

        attribs = {"long_name":"flopy.util_3d instance of {0}".format(name)}
        var = f.create_variable(name,attribs,precision_str=precision_str,dimensions=("layer","y","x"))
        var[:] = u3d.array
        return f

    else:
        raise NotImplementedError("util2d_helper only for netcdf (*.nc) ")


def util2d_helper(f,u2d):

    assert isinstance(u2d,util_2d),"util2d_helper only helps util_2d instances"

    assert len(u2d.shape) == 2,"util2d_helper only supports 2D arrays"

    if isinstance(f,str) and f.lower().endswith(".nc"):
        f = NetCdf(f,u2d.model)

    if isinstance(f,NetCdf):
        units = ''
        if u2d.name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[u2d.name].format(f.grid_units,f.time_units)

        precision_str = NC_PRECISION_TYPE[u2d.dtype]

        attribs = {"long_name":"flopy.util_2d instance of {0}".format(u2d.name)}
        var = f.create_variable(u2d.name,attribs,precision_str=precision_str,dimensions=("y","x"))
        var[:] = u2d.array
        return f

    else:
        raise NotImplementedError("util2d_helper only for netcdf (*.nc) ")