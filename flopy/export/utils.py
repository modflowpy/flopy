from __future__ import print_function
import json
import os
import numpy as np
from ..utils import Util2d, Util3d, Transient2d, MfList, \
    HeadFile, CellBudgetFile, UcnFile, FormattedHeadFile
from ..mbase import BaseModel
from ..pakbase import Package
from . import NetCdf, netcdf
from . import shapefile_utils


NC_PRECISION_TYPE = {np.float32: "f4", np.int: "i4", np.int64: "i4",
                     np.int32: "i4"}

path = os.path.split(netcdf.__file__)[0]
with open(path + '/longnames.json') as f:
    NC_LONG_NAMES = json.load(f)
with open(path + '/unitsformat.json') as f:
    NC_UNITS_FORMAT = json.load(f)


def get_var_array_dict(m):
    vdict = {}
    # for vname in f.var_attr_dict.keys():
    #    vdict[vname] = f.nc.variables[vname][:]
    for attr in m:
        if hasattr(attr, "stress_period_data"):
            array_dict = attr.stress_period_data.array

    return vdict


def ensemble_helper(inputs_filename, outputs_filename, models, add_reals=True,
                    **kwargs):
    """ helper to export an ensemble of model instances.  Assumes
    all models have same dis and sr, only difference is properties and
    boundary conditions.  Assumes model.nam.split('_')[-1] is the
    realization suffix to use in the netcdf variable names
    """
    f_in, f_out = None, None
    for m in models[1:]:
        assert m.get_nrow_ncol_nlay_nper() == models[
            0].get_nrow_ncol_nlay_nper()
    if inputs_filename is not None:
        f_in = models[0].export(inputs_filename, **kwargs)
        vdict = {}
        vdicts = [models[0].export(vdict, **kwargs)]
        i = 1
        for m in models[1:]:
            suffix = m.name.split('.')[0].split('_')[-1]
            vdict = {}
            m.export(vdict, **kwargs)
            vdicts.append(vdict)
            if add_reals:
                f_in.append(vdict, suffix=suffix)
            i += 1
        mean, stdev = {}, {}
        for vname in vdict.keys():
            alist = []
            for vd in vdicts:
                alist.append(vd[vname])
            alist = np.array(alist)
            mean[vname] = alist.mean(axis=0)
            stdev[vname] = alist.std(axis=0)
            mean[vname][vdict[vname] == netcdf.FILLVALUE] = netcdf.FILLVALUE
            stdev[vname][vdict[vname] == netcdf.FILLVALUE] = netcdf.FILLVALUE
            mean[vname][np.isnan(vdict[vname])] = netcdf.FILLVALUE
            stdev[vname][np.isnan(vdict[vname])] = netcdf.FILLVALUE

        if i >= 2:
            if not add_reals:
                f_in.write()
                f_in = NetCdf.empty_like(mean, output_filename=inputs_filename)
                f_in.append(mean, suffix="**mean**")
                f_in.append(stdev, suffix="**stdev**")
            else:
                f_in.append(mean, suffix="**mean**")
                f_in.append(stdev, suffix="**stdev**")
        f_in.add_global_attributes({"namefile": ''})

    if outputs_filename is not None:
        f_out = output_helper(outputs_filename, models[0], models[0]. \
                              load_results(as_dict=True), **kwargs)
        vdict = {}
        vdicts = [output_helper(vdict, models[0], models[0]. \
                                load_results(as_dict=True), **kwargs)]
        i = 1
        for m in models[1:]:
            suffix = m.name.split('.')[0].split('_')[-1]
            oudic = m.load_results(as_dict=True)
            vdict = {}
            output_helper(vdict, m, oudic, **kwargs)
            vdicts.append(vdict)
            if add_reals:
                f_out.append(vdict, suffix=suffix)
            i += 1

        mean, stdev = {}, {}
        for vname in vdict.keys():
            alist = []
            for vd in vdicts:
                alist.append(vd[vname])
            alist = np.array(alist)
            mean[vname] = alist.mean(axis=0)
            stdev[vname] = alist.std(axis=0)
            mean[vname][np.isnan(vdict[vname])] = netcdf.FILLVALUE
            stdev[vname][np.isnan(vdict[vname])] = netcdf.FILLVALUE
            mean[vname][vdict[vname] == netcdf.FILLVALUE] = netcdf.FILLVALUE
            stdev[vname][vdict[vname] == netcdf.FILLVALUE] = netcdf.FILLVALUE
        if i >= 2:
            if not add_reals:
                f_out.write()
                f_out = NetCdf.empty_like(mean,
                                          output_filename=outputs_filename)
                f_out.append(mean, suffix="**mean**")
                f_out.append(stdev, suffix="**stdev**")

            else:
                f_out.append(mean, suffix="**mean**")
                f_out.append(stdev, suffix="**stdev**")
        f_out.add_global_attributes({"namefile": ''})
    return f_in, f_out


def _add_output_nc_variable(f, times, shape3d, out_obj, var_name, logger=None,
                            text='',
                            mask_vals=[], mask_array3d=None):
    if logger:
        logger.log("creating array for {0}".format(
            var_name))

    array = np.zeros((len(times), shape3d[0], shape3d[1], shape3d[2]),
                     dtype=np.float32)
    array[:] = np.NaN
    for i, t in enumerate(times):
        if t in out_obj.recordarray["totim"]:
            try:
                if text:
                    a = out_obj.get_data(totim=t, full3D=True, text=text)
                    if isinstance(a, list):
                        a = a[0]
                else:
                    a = out_obj.get_data(totim=t)
            except Exception as e:
                estr = "error getting data for {0} at time {1}:{2}".format(
                    var_name + text.decode().strip().lower(), t, str(e))
                if logger:
                    logger.warn(estr)
                else:
                    print(estr)
                continue
            if mask_array3d is not None and a.shape == mask_array3d.shape:
                a[mask_array3d] = np.NaN
            try:
                array[i, :, :, :] = a.astype(np.float32)
            except Exception as e:
                estr = "error assigning {0} data to array for time {1}:{2}".format(
                    var_name + text.decode().strip().lower(), t, str(e))
                if logger:
                    logger.warn(estr)
                else:
                    print(estr)
                continue

    if logger:
        logger.log("creating array for {0}".format(
            var_name))

    for mask_val in mask_vals:
        array[np.where(array == mask_val)] = np.NaN
    mx, mn = np.nanmax(array), np.nanmin(array)
    array[np.isnan(array)] = netcdf.FILLVALUE

    if isinstance(f, dict):
        if text:
            var_name = text.decode().strip().lower()
        f[var_name] = array
        return f

    units = None
    if var_name in NC_UNITS_FORMAT:
        units = NC_UNITS_FORMAT[var_name].format(
            f.grid_units, f.time_units)
    precision_str = "f4"

    if text:
        var_name = text.decode().strip().lower()
    attribs = {"long_name": var_name}
    attribs["coordinates"] = "time layer latitude longitude"
    attribs["min"] = mn
    attribs["max"] = mx
    if units is not None:
        attribs["units"] = units
    try:
        var = f.create_variable(var_name, attribs,
                                precision_str=precision_str,
                                dimensions=("time", "layer", "y", "x"))
    except Exception as e:
        estr = "error creating variable {0}:\n{1}".format(
            var_name, str(e))
        if logger:
            logger.lraise(estr)
        else:
            raise Exception(estr)

    try:
        var[:] = array
    except Exception as e:
        estr = "error setting array to variable {0}:\n{1}".format(
            var_name, str(e))
        if logger:
            logger.lraise(estr)
        else:
            raise Exception(estr)


def output_helper(f, ml, oudic, **kwargs):
    """export model outputs using the model spatial reference
    info.
    Parameters
    ----------
        f : filename for output - must have .shp or .nc extension
        ml : BaseModel derived type
        oudic : dict {output_filename,flopy datafile/cellbudgetfile instance}
    Returns
    -------
        None
    Note:
    ----
        casts down double precision to single precision for netCDF files

    """
    assert isinstance(ml, BaseModel)
    assert len(oudic.keys()) > 0
    logger = kwargs.pop("logger", None)
    stride = kwargs.pop("stride", 1)
    suffix = kwargs.pop("suffix", None)
    forgive = kwargs.pop("forgive", False)
    if len(kwargs) > 0 and logger is not None:
        str_args = ','.join(kwargs)
        logger.warn("unused kwargs: " + str_args)
    # this sucks!  need to round the totims in each output file instance so
    # that they will line up
    for key, out in oudic.items():
        times = [float("{0:15.6f}".format(t)) for t in
                 out.recordarray["totim"]]
        out.recordarray["totim"] = times

    times = []
    for filename, df in oudic.items():
        [times.append(t) for t in df.recordarray["totim"] if t not in times]
    assert len(times) > 0
    times.sort()

    # rectify times - only use times that are common to every output file
    common_times = []
    skipped_times = []
    for t in times:
        keep = True
        for filename, df in oudic.items():
            if t not in df.recordarray["totim"]:
                keep = False
                break
        if keep:
            common_times.append(t)
        else:
            skipped_times.append(t)
    assert len(common_times) > 0
    if len(skipped_times) > 0:
        if logger:
            logger.warn("the following output times are not common to all" + \
                        " output files and are being skipped:\n" + \
                        "{0}".format(skipped_times))
        else:
            print("the following output times are not common to all" + \
                  " output files and are being skipped:\n" + \
                  "{0}".format(skipped_times))
    times = [t for t in common_times[::stride]]
    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, ml, time_values=times, logger=logger,
                   forgive=forgive)
    elif isinstance(f, NetCdf):
        otimes = list(f.nc.variables["time"][:])
        assert otimes == times
    if isinstance(f, NetCdf) or isinstance(f, dict):
        shape3d = (ml.nlay, ml.nrow, ml.ncol)
        mask_vals = []
        mask_array3d = None
        if ml.bas6:
            mask_vals.append(ml.bas6.hnoflo)
            mask_array3d = ml.bas6.ibound.array == 0
        if ml.bcf:
            mask_vals.append(ml.bcf.hdry)
        if ml.lpf:
            mask_vals.append(ml.lpf.hdry)

        for filename, out_obj in oudic.items():
            filename = filename.lower()

            if isinstance(out_obj, UcnFile):
                _add_output_nc_variable(f, times, shape3d, out_obj,
                                        "concentration", logger=logger,
                                        mask_vals=mask_vals,
                                        mask_array3d=mask_array3d)

            elif isinstance(out_obj, HeadFile):
                _add_output_nc_variable(f, times, shape3d, out_obj,
                                        out_obj.text.decode(), logger=logger,
                                        mask_vals=mask_vals,
                                        mask_array3d=mask_array3d)

            elif isinstance(out_obj, FormattedHeadFile):
                _add_output_nc_variable(f, times, shape3d, out_obj,
                                        out_obj.text, logger=logger,
                                        mask_vals=mask_vals,
                                        mask_array3d=mask_array3d)

            elif isinstance(out_obj, CellBudgetFile):
                var_name = "cell_by_cell_flow"
                for text in out_obj.textlist:
                    _add_output_nc_variable(f, times, shape3d, out_obj,
                                            var_name, logger=logger, text=text,
                                            mask_vals=mask_vals,
                                            mask_array3d=mask_array3d)

            else:
                estr = "unrecognized file extention:{0}".format(filename)
                if logger:
                    logger.lraise(estr)
                else:
                    raise Exception(estr)

    else:
        if logger:
            logger.lraise("unrecognized export argument:{0}".format(f))
        else:
            raise NotImplementedError("unrecognized export argument" + \
                                      ":{0}".format(f))
    return f


def model_helper(f, ml, **kwargs):
    assert isinstance(ml, BaseModel)
    package_names = kwargs.get("package_names", None)
    if package_names is None:
        package_names = [pak.name[0] for pak in ml.packagelist]

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, ml, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(f, ml,
                                                      package_names=package_names,
                                                      **kwargs)

    elif isinstance(f, NetCdf):

        for pak in ml.packagelist:
            if pak.name[0] in package_names:
                f = pak.export(f, **kwargs)
                assert f is not None
        return f

    elif isinstance(f, dict):
        for pak in ml.packagelist:
            f = pak.export(f, **kwargs)

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))

    return f


def package_helper(f, pak, **kwargs):
    assert isinstance(pak, Package)
    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, pak.parent)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(f, pak.parent,
                                                      package_names=pak.name,
                                                      **kwargs)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        attrs = dir(pak)
        if 'sr' in attrs:
            attrs.remove('sr')
        if 'start_datetime' in attrs:
            attrs.remove('start_datetime')
        for attr in attrs:
            if '__' in attr:
                continue
            a = pak.__getattribute__(attr)
            if isinstance(a, Util2d) and len(a.shape) == 2 and a.shape[1] > 0:
                try:
                    f = util2d_helper(f, a, **kwargs)
                except:
                    f.logger.warn(
                        "error adding {0} as variable".format(a.name))
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


def generic_array_helper(f, array, var_name="generic_array",
                         dimensions=("time", "layer", "y", "x"),
                         precision_str="f4", units="unitless", **kwargs):
    # assert isinstance(f,NetCdf),"generic_array_helper() can only be used " +\
    #                            "with instantiated netCDfs"
    if isinstance(f, str) and f.lower().endswith(".nc"):
        assert "model" in kwargs.keys(), "creating a new netCDF using generic_array_helper requires a 'model' kwarg"
        assert isinstance(kwargs["model"], BaseModel)
        f = NetCdf(f, kwargs.pop("model"))

    assert array.ndim == len(dimensions), "generic_array_helper() " + \
                                          "array.ndim != dimensions"
    coords_dims = {"time": "time", "layer": "layer", "y": "latitude",
                   "x": "longitude"}
    coords = ' '.join([coords_dims[d] for d in dimensions])
    mn = kwargs.pop("min", -1.0e+9)
    mx = kwargs.pop("max", 1.0e+9)
    long_name = kwargs.pop("long_name", var_name)
    if len(kwargs) > 0:
        f.logger.warn("generic_array_helper(): unrecognized kwargs:" + \
                      ",".join(kwargs.keys()))
    attribs = {"long_name": long_name}
    attribs["coordinates"] = coords
    attribs["units"] = units
    attribs["min"] = mn
    attribs["max"] = mx
    if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
        raise Exception("error processing {0}: all NaNs".format(var_name))
    try:
        var = f.create_variable(var_name, attribs, precision_str=precision_str,
                                dimensions=dimensions)
    except Exception as e:
        estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
        f.logger.warn(estr)
        raise Exception(estr)
    try:
        var[:] = array
    except Exception as e:
        estr = "error setting array to variable {0}:\n{1}".format(var_name,
                                                                  str(e))
        f.logger.warn(estr)
        raise Exception(estr)
    return f


def mflist_helper(f, mfl, **kwargs):
    """ export helper for MfList instances

    Parameters
    -----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        mfl : MfList instance

    """
    assert isinstance(mfl, MfList) \
        , "mflist_helper only helps MfList instances"

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, mfl.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        sparse = kwargs.get("sparse", False)
        kper = kwargs.get("kper", 0)
        squeeze = kwargs.get("squeeze", True)
        if mfl.sr is None:
            raise Exception("MfList.to_shapefile: SpatialReference not set")
        import flopy.utils.flopy_io as fio
        if kper is None:
            keys = mfl.data.keys()
            keys.sort()
        else:
            keys = [kper]
        if not sparse:
            array_dict = {}
            for kk in keys:
                arrays = mfl.to_array(kk)
                for name, array in arrays.items():
                    for k in range(array.shape[0]):
                        # aname = name+"{0:03d}_{1:02d}".format(kk, k)
                        n = shapefile_utils.shape_attr_name(name, length=4)
                        aname = "{}{:03d}{:03d}".format(n, k + 1, int(kk) + 1)
                        array_dict[aname] = array[k]
            shapefile_utils.write_grid_shapefile(f, mfl.sr, array_dict)
        else:
            from ..export.shapefile_utils import recarray2shp
            from ..utils.geometry import Polygon
            #if np.isscalar(kper):
            #    kper = [kper]
            sr = mfl.model.sr
            df = mfl.get_dataframe(squeeze=squeeze)
            if 'kper' in kwargs or df is None:
                ra = mfl[kper]
                verts = np.array(sr.get_vertices(ra.i, ra.j))
            elif df is not None:
                verts = sr.get_vertices(df.i.values, df.j.values)
                ra = df.to_records(index=False)
            # write the projection file
            if sr.epsg is None:
                epsg = kwargs.get('epsg', None)
            else:
                epsg = sr.epsg
            prj = kwargs.get('prj', None)
            polys = np.array([Polygon(v) for v in verts])
            recarray2shp(ra, geoms=polys, shpname=f, epsg=epsg, prj=prj)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        base_name = mfl.package.name[0].lower()
        # f.log("getting 4D masked arrays for {0}".format(base_name))
        # m4d = mfl.masked_4D_arrays
        # f.log("getting 4D masked arrays for {0}".format(base_name))

        # for name, array in m4d.items():
        for name, array in mfl.masked_4D_arrays_itr():
            var_name = base_name + '_' + name
            if isinstance(f, dict):
                f[var_name] = array
                continue
            f.log("processing {0} attribute".format(name))

            units = None
            if var_name in NC_UNITS_FORMAT:
                units = NC_UNITS_FORMAT[var_name].format(f.grid_units,
                                                         f.time_units)
            precision_str = NC_PRECISION_TYPE[mfl.dtype[name].type]
            if var_name in NC_LONG_NAMES:
                attribs = {"long_name": NC_LONG_NAMES[var_name]}
            else:
                attribs = {"long_name": var_name}
            attribs["coordinates"] = "time layer latitude longitude"
            attribs["min"] = np.nanmin(array)
            attribs["max"] = np.nanmax(array)
            if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
                raise Exception(
                    "error processing {0}: all NaNs".format(var_name))

            if units is not None:
                attribs["units"] = units
            try:
                var = f.create_variable(var_name, attribs,
                                        precision_str=precision_str,
                                        dimensions=("time", "layer", "y", "x"))
            except Exception as e:
                estr = "error creating variable {0}:\n{1}".format(var_name,
                                                                  str(e))
                f.logger.warn(estr)
                raise Exception(estr)

            array[np.isnan(array)] = f.fillvalue
            try:
                var[:] = array
            except Exception as e:
                estr = "error setting array to variable {0}:\n{1}".format(
                    var_name, str(e))
                f.logger.warn(estr)
                raise Exception(estr)
            f.log("processing {0} attribute".format(name))

        return f
    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def transient2d_helper(f, t2d, **kwargs):
    """ export helper for Transient2d instances

    Parameters
    -----------
        f : string (filename) or existing export instance type (NetCdf only for now)
        t2d : Transient2d instance
        min_valid : minimum valid value
        max_valid : maximum valid value

    """

    assert isinstance(t2d, Transient2d) \
        , "transient2d_helper only helps Transient2d instances"

    min_valid = kwargs.get("min_valid", -1.0e+9)
    max_valid = kwargs.get("max_valid", 1.0e+9)

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, t2d.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        array_dict = {}
        for kper in range(t2d.model.nper):
            u2d = t2d[kper]
            name = '{}_{:03d}'.format(
                shapefile_utils.shape_attr_name(u2d.name), kper + 1)
            array_dict[name] = u2d.array
        shapefile_utils.write_grid_shapefile(f, t2d.model.sr, array_dict)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        # mask the array is defined by any row col with at lease
        # one active cell
        mask = None
        if t2d.model.bas6 is not None:
            ibnd = np.abs(t2d.model.bas6.ibound.array).sum(axis=0)
            mask = ibnd == 0
        elif t2d.model.btn is not None:
            ibnd = np.abs(t2d.model.btn.icbund.array).sum(axis=0)
            mask = ibnd == 0

        # f.log("getting 4D array for {0}".format(t2d.name_base))
        array = t2d.array
        # f.log("getting 4D array for {0}".format(t2d.name_base))
        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int, np.int32, np.int64]:
                if mask is not None:
                    array[:, 0, mask] = np.NaN
                array[array <= min_valid] = np.NaN
                array[array >= max_valid] = np.NaN
                mx, mn = np.nanmax(array), np.nanmin(array)
            else:
                mx, mn = np.nanmax(array), np.nanmin(array)
                array[array <= min_valid] = netcdf.FILLVALUE
                array[array >= max_valid] = netcdf.FILLVALUE
                # if t2d.model.bas6 is not None:
                #    array[:, 0, t2d.model.bas6.ibound.array[0] == 0] = \
                #        f.fillvalue
                # elif t2d.model.btn is not None:
                #    array[:, 0, t2d.model.btn.icbund.array[0] == 0] = \
                #        f.fillvalue

        var_name = t2d.name_base.replace('_', '')
        if isinstance(f, dict):
            array[array == netcdf.FILLVALUE] = np.NaN
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"

        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units,
                                                     f.time_units)
        try:
            precision_str = NC_PRECISION_TYPE[t2d.dtype]
        except:
            precision_str = NC_PRECISION_TYPE[t2d.dtype.type]
        if var_name in NC_LONG_NAMES:
            attribs = {"long_name": NC_LONG_NAMES[var_name]}
        else:
            attribs = {"long_name": var_name}
        attribs["coordinates"] = "time layer latitude longitude"
        attribs["units"] = units
        attribs["min"] = mn
        attribs["max"] = mx
        if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
            raise Exception("error processing {0}: all NaNs".format(var_name))
        try:
            var = f.create_variable(var_name, attribs,
                                    precision_str=precision_str,
                                    dimensions=("time", "layer", "y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:, 0] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name,
                                                                      str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def util3d_helper(f, u3d, **kwargs):
    """ export helper for Transient2d instances

    Parameters
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
            name = '{}_{:03d}'.format(
                shapefile_utils.shape_attr_name(u2d.name), ilay + 1)
            array_dict[name] = u2d.array
        shapefile_utils.write_grid_shapefile(f, u3d.model.sr,
                                             array_dict)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        var_name = u3d.name[0].replace(' ', '_').lower()
        # f.log("getting 3D array for {0}".format(var_name))
        array = u3d.array

        # this is for the crappy vcont in bcf6
        # if isinstance(f,NetCdf) and array.shape != f.shape:
        #     f.log("broadcasting 3D array for {0}".format(var_name))
        #     full_array = np.empty(f.shape)
        #     full_array[:] = np.NaN
        #     full_array[:array.shape[0]] = array
        #     array = full_array
        #     f.log("broadcasting 3D array for {0}".format(var_name))
        # f.log("getting 3D array for {0}".format(var_name))
        #
        mask = None
        if u3d.model.bas6 is not None and "ibound" not in var_name:
            mask = u3d.model.bas6.ibound.array == 0
        elif u3d.model.btn is not None and 'icbund' not in var_name:
            mask = u3d.model.btn.icbund.array == 0

        if mask is not None and array.shape != mask.shape:
            # f.log("broadcasting 3D array for {0}".format(var_name))
            full_array = np.empty(mask.shape)
            full_array[:] = np.NaN
            full_array[:array.shape[0]] = array
            array = full_array
            # f.log("broadcasting 3D array for {0}".format(var_name))

        # runtime warning issued in some cases - need to track down cause
        # happens when NaN is already in array
        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int, np.int32, np.int64]:
                # if u3d.model.bas6 is not None and "ibound" not in var_name:
                #    array[u3d.model.bas6.ibound.array == 0] = np.NaN
                # elif u3d.model.btn is not None and 'icbund' not in var_name:
                #    array[u3d.model.btn.icbund.array == 0] = np.NaN
                if mask is not None:
                    array[mask] = np.NaN
                array[array <= min_valid] = np.NaN
                array[array >= max_valid] = np.NaN
                mx, mn = np.nanmax(array), np.nanmin(array)
            else:
                mx, mn = np.nanmax(array), np.nanmin(array)
                if mask is not None:
                    array[mask] = netcdf.FILLVALUE
                array[array <= min_valid] = netcdf.FILLVALUE
                array[array >= max_valid] = netcdf.FILLVALUE
                if u3d.model.bas6 is not None and "ibound" not in var_name:
                    array[u3d.model.bas6.ibound.array == 0] = netcdf.FILLVALUE
                elif u3d.model.btn is not None and 'icbund' not in var_name:
                    array[u3d.model.btn.icbund.array == 0] = netcdf.FILLVALUE

        if isinstance(f, dict):
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"
        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units,
                                                     f.time_units)
        precision_str = NC_PRECISION_TYPE[u3d.dtype]
        if var_name in NC_LONG_NAMES:
            attribs = {"long_name": NC_LONG_NAMES[var_name]}
        else:
            attribs = {"long_name": var_name}
        attribs["coordinates"] = "layer latitude longitude"
        attribs["units"] = units
        attribs["min"] = mn
        attribs["max"] = mx
        if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
            raise Exception("error processing {0}: all NaNs".format(var_name))
        try:
            var = f.create_variable(var_name, attribs,
                                    precision_str=precision_str,
                                    dimensions=("layer", "y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name,
                                                                      str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))


def util2d_helper(f, u2d, **kwargs):
    """ export helper for Util2d instances

    Parameters
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

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, u2d.model)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        name = shapefile_utils.shape_attr_name(u2d.name, keep_layer=True)
        shapefile_utils.write_grid_shapefile(f, u2d.model.sr,
                                             {name: u2d.array})
        return

    elif isinstance(f, str) and f.lower().endswith(".asc"):
        u2d.model.sr.export_array(f, u2d.array, **kwargs)
        return

    elif isinstance(f, NetCdf) or isinstance(f, dict):

        # try to mask the array - assume layer 1 ibound is a good mask
        # f.log("getting 2D array for {0}".format(u2d.name))
        array = u2d.array
        # f.log("getting 2D array for {0}".format(u2d.name))

        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int, np.int32, np.int64]:
                if u2d.model.bas6 is not None and \
                                "ibound" not in u2d.name.lower():
                    array[u2d.model.bas6.ibound.array[0, :, :] == 0] = np.NaN
                elif u2d.model.btn is not None and \
                                "icbund" not in u2d.name.lower():
                    array[u2d.model.btn.icbund.array[0, :, :] == 0] = np.NaN
                array[array <= min_valid] = np.NaN
                array[array >= max_valid] = np.NaN
                mx, mn = np.nanmax(array), np.nanmin(array)
            else:
                mx, mn = np.nanmax(array), np.nanmin(array)
                array[array <= min_valid] = netcdf.FILLVALUE
                array[array >= max_valid] = netcdf.FILLVALUE
                if u2d.model.bas6 is not None and \
                                "ibound" not in u2d.name.lower():
                    array[u2d.model.bas6.ibound.array[0, :, :] == 0] = \
                        netcdf.FILLVALUE
                elif u2d.model.btn is not None and \
                                "icbund" not in u2d.name.lower():
                    array[u2d.model.btn.icbund.array[0, :, :] == 0] = \
                        netcdf.FILLVALUE
        var_name = u2d.name
        if isinstance(f, dict):
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"

        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(f.grid_units,
                                                     f.time_units)
        precision_str = NC_PRECISION_TYPE[u2d.dtype]
        if var_name in NC_LONG_NAMES:
            attribs = {"long_name": NC_LONG_NAMES[var_name]}
        else:
            attribs = {"long_name": var_name}
        attribs["coordinates"] = "latitude longitude"
        attribs["units"] = units
        attribs["min"] = mn
        attribs["max"] = mx
        if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
            raise Exception("error processing {0}: all NaNs".format(var_name))
        try:
            var = f.create_variable(var_name, attribs,
                                    precision_str=precision_str,
                                    dimensions=("y", "x"))
        except Exception as e:
            estr = "error creating variable {0}:\n{1}".format(var_name, str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = "error setting array to variable {0}:\n{1}".format(var_name,
                                                                      str(e))
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    else:
        raise NotImplementedError("unrecognized export argument:{0}".format(f))

