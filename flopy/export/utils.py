import os

import numpy as np

from ..datbase import DataInterface, DataListInterface, DataType
from ..mbase import BaseModel, ModelInterface
from ..pakbase import PackageInterface
from ..utils import (
    CellBudgetFile,
    FormattedHeadFile,
    HeadFile,
    UcnFile,
    ZBNetOutput,
    import_optional_dependency,
)
from . import NetCdf, netcdf, shapefile_utils, vtk
from .longnames import NC_LONG_NAMES
from .unitsformat import NC_UNITS_FORMAT

NC_PRECISION_TYPE = {
    np.float64: "f8",
    np.float32: "f4",
    int: "i4",
    np.int64: "i4",
    np.int32: "i4",
}


def ensemble_helper(
    inputs_filename, outputs_filename, models, add_reals=True, **kwargs
):
    """
    Helper to export an ensemble of model instances.  Assumes
    all models have same dis and reference information, only difference is
    properties and boundary conditions.  Assumes model.nam.split('_')[-1] is
    the realization suffix to use in the netcdf variable names
    """
    f_in, f_out = None, None
    for m in models[1:]:
        assert (
            m.get_nrow_ncol_nlay_nper() == models[0].get_nrow_ncol_nlay_nper()
        )
    if inputs_filename is not None:
        f_in = models[0].export(inputs_filename, **kwargs)
        vdict = {}
        vdicts = [models[0].export(vdict, **kwargs)]
        i = 1
        for m in models[1:]:
            suffix = m.name.split(".")[0].split("_")[-1]
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
        f_in.add_global_attributes({"namefile": ""})

    if outputs_filename is not None:
        f_out = output_helper(
            outputs_filename,
            models[0],
            models[0].load_results(as_dict=True),
            **kwargs,
        )
        vdict = {}
        vdicts = [
            output_helper(
                vdict,
                models[0],
                models[0].load_results(as_dict=True),
                **kwargs,
            )
        ]
        i = 1
        for m in models[1:]:
            suffix = m.name.split(".")[0].split("_")[-1]
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
                f_out = NetCdf.empty_like(
                    mean, output_filename=outputs_filename
                )
                f_out.append(mean, suffix="**mean**")
                f_out.append(stdev, suffix="**stdev**")

            else:
                f_out.append(mean, suffix="**mean**")
                f_out.append(stdev, suffix="**stdev**")
        f_out.add_global_attributes({"namefile": ""})
    return f_in, f_out


def _add_output_nc_variable(
    f,
    times,
    shape3d,
    out_obj,
    var_name,
    logger=None,
    text="",
    mask_vals=(),
    mask_array3d=None,
):
    if logger:
        logger.log(f"creating array for {var_name}")

    array = np.zeros(
        (len(times), shape3d[0], shape3d[1], shape3d[2]), dtype=np.float32
    )
    array[:] = np.NaN

    if isinstance(out_obj, ZBNetOutput):
        a = np.asarray(out_obj.zone_array, dtype=np.float32)
        if mask_array3d is not None:
            a[mask_array3d] = np.NaN
        for i, _ in enumerate(times):
            array[i, :, :, :] = a

    else:
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
                    estr = (
                        "error getting data for {0} at time"
                        " {1}:{2}".format(
                            var_name + text.decode().strip().lower(), t, str(e)
                        )
                    )
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
                    estr = (
                        "error assigning {0} data to array for time"
                        " {1}:{2}".format(
                            var_name + text.decode().strip().lower(), t, str(e)
                        )
                    )
                    if logger:
                        logger.warn(estr)
                    else:
                        print(estr)
                    continue

    if logger:
        logger.log(f"creating array for {var_name}")

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
        units = NC_UNITS_FORMAT[var_name].format(f.grid_units, f.time_units)
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
        dim_tuple = ("time",) + f.dimension_names
        var = f.create_variable(
            var_name,
            attribs,
            precision_str=precision_str,
            dimensions=dim_tuple,
        )
    except Exception as e:
        estr = f"error creating variable {var_name}:\n{e!s}"
        if logger:
            logger.lraise(estr)
        else:
            raise Exception(estr)

    try:
        var[:] = array
    except Exception as e:
        estr = f"error setting array to variable {var_name}:\n{e!s}"
        if logger:
            logger.lraise(estr)
        else:
            raise Exception(estr)


def _add_output_nc_zonebudget_variable(f, array, var_name, flux, logger=None):
    """
    Method to add zonebudget output data to netcdf file

    Parameters
    ----------
    f : NetCdf object
    array : np.ndarray
        zonebudget output budget group array
    var_name : str
        variable name
    flux : bool
        flag for flux data or volumetric data
    logger : None or Logger
        logger instance

    """
    if logger:
        logger.log(f"creating array for {var_name}")

    mn = np.min(array)
    mx = np.max(array)

    precision_str = "f4"
    if flux:
        units = f"{f.grid_units}^3/{f.time_units}"
    else:
        units = f"{f.grid_units}^3"
    attribs = {"long_name": var_name}
    attribs["coordinates"] = "time zone"
    attribs["min"] = mn
    attribs["max"] = mx
    attribs["units"] = units
    dim_tuple = ("time", "zone")

    var = f.create_group_variable(
        "zonebudget", var_name, attribs, precision_str, dim_tuple
    )

    var[:] = array


def output_helper(f, ml, oudic, **kwargs):
    """
    Export model outputs using the model spatial reference info.

    Parameters
    ----------
    f : filepath to write output to (must have .shp or .nc extension)
        or NetCDF object or dict
    ml : flopy.mbase.ModelInterface derived type
    oudic : dict
        output_filename,flopy datafile/cellbudgetfile instance
    **kwargs : keyword arguments
        modelgrid : flopy.discretizaiton.Grid
            user supplied model grid instance that will be used for export
            in lieu of the models model grid instance
        mflay : int
            zero based model layer which can be used in shapefile exporting
        kper : int
            zero based stress period which can be used for shapefile exporting

    Returns
    -------
        None
    Note:
    ----
        casts down double precision to single precision for netCDF files

    """
    assert isinstance(ml, (BaseModel, ModelInterface))
    assert len(oudic.keys()) > 0
    logger = kwargs.pop("logger", None)
    stride = kwargs.pop("stride", 1)
    forgive = kwargs.pop("forgive", False)
    kwargs.pop("suffix", None)
    mask_vals = []
    mflay = kwargs.pop("mflay", None)
    kper = kwargs.pop("kper", None)
    if "masked_vals" in kwargs:
        mask_vals = kwargs.pop("masked_vals")
    if len(kwargs) > 0 and logger is not None:
        str_args = ",".join(kwargs)
        logger.warn(f"unused kwargs: {str_args}")

    zonebud = None
    zbkey = None
    for key, value in oudic.items():
        if isinstance(value, ZBNetOutput):
            zbkey = key
            break

    if zbkey is not None:
        zonebud = oudic.pop(zbkey)

    # ISSUE - need to round the totims in each output file instance so
    # that they will line up
    for key in oudic.keys():
        out = oudic[key]
        times = [float(f"{t:15.6f}") for t in out.recordarray["totim"]]
        out.recordarray["totim"] = times

    times = []
    for filename, df in oudic.items():
        for t in df.recordarray["totim"]:
            if t not in times:
                times.append(t)

    if zonebud is not None and not oudic:
        if isinstance(f, NetCdf):
            times = f.time_values_arg
        else:
            times = zonebud.time

    assert len(times) > 0
    times.sort()

    # rectify times - only use times that are common to every output file
    common_times = []
    skipped_times = []
    for t in times:
        keep = True
        for filename, df in oudic.items():
            if isinstance(df, ZBNetOutput):
                continue
            if t not in df.recordarray["totim"]:
                keep = False
                break
        if keep:
            common_times.append(t)
        else:
            skipped_times.append(t)

    assert len(common_times) > 0
    if len(skipped_times) > 0:
        msg = (
            "the following output times are not common to all "
            f"output files and are being skipped:\n{skipped_times}"
        )
        if logger:
            logger.warn(msg)
        else:
            print(msg)
    times = [t for t in common_times[::stride]]
    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(
            f, ml, time_values=times, logger=logger, forgive=forgive, **kwargs
        )
    elif isinstance(f, NetCdf):
        otimes = list(f.nc.variables["time"][:])
        assert otimes == times
    if isinstance(f, NetCdf) or isinstance(f, dict):
        shape3d = (ml.modelgrid.nlay, ml.modelgrid.nrow, ml.modelgrid.ncol)
        mask_array3d = None
        if ml.hdry is not None:
            mask_vals.append(ml.hdry)
        if ml.hnoflo is not None:
            mask_vals.append(ml.hnoflo)

        if ml.modelgrid.idomain is not None:
            mask_array3d = ml.modelgrid.idomain == 0

        for filename, out_obj in oudic.items():
            filename = filename.lower()

            if isinstance(out_obj, UcnFile):
                _add_output_nc_variable(
                    f,
                    times,
                    shape3d,
                    out_obj,
                    "concentration",
                    logger=logger,
                    mask_vals=mask_vals,
                    mask_array3d=mask_array3d,
                )

            elif isinstance(out_obj, HeadFile):
                _add_output_nc_variable(
                    f,
                    times,
                    shape3d,
                    out_obj,
                    out_obj.text.decode(),
                    logger=logger,
                    mask_vals=mask_vals,
                    mask_array3d=mask_array3d,
                )

            elif isinstance(out_obj, FormattedHeadFile):
                _add_output_nc_variable(
                    f,
                    times,
                    shape3d,
                    out_obj,
                    out_obj.text,
                    logger=logger,
                    mask_vals=mask_vals,
                    mask_array3d=mask_array3d,
                )

            elif isinstance(out_obj, CellBudgetFile):
                var_name = "cell_by_cell_flow"
                for text in out_obj.textlist:
                    _add_output_nc_variable(
                        f,
                        times,
                        shape3d,
                        out_obj,
                        var_name,
                        logger=logger,
                        text=text,
                        mask_vals=mask_vals,
                        mask_array3d=mask_array3d,
                    )

            else:
                estr = f"unrecognized file extension:{filename}"
                if logger:
                    logger.lraise(estr)
                else:
                    raise Exception(estr)

        if zonebud is not None:
            try:
                f.initialize_group(
                    "zonebudget",
                    dimensions=("time", "zone"),
                    dimension_data={
                        "time": zonebud.time,
                        "zone": zonebud.zones,
                    },
                )
            except AttributeError:
                pass

            for text, array in zonebud.arrays.items():
                _add_output_nc_zonebudget_variable(
                    f, array, text, zonebud.flux, logger
                )

            # write the zone array to standard output
            _add_output_nc_variable(
                f,
                times,
                shape3d,
                zonebud,
                "budget_zones",
                logger=logger,
                mask_vals=mask_vals,
                mask_array3d=mask_array3d,
            )

    elif isinstance(f, str) and f.endswith(".shp"):
        attrib_dict = {}
        for _, out_obj in oudic.items():

            if (
                isinstance(out_obj, HeadFile)
                or isinstance(out_obj, FormattedHeadFile)
                or isinstance(out_obj, UcnFile)
            ):
                if isinstance(out_obj, UcnFile):
                    attrib_name = "conc"
                else:
                    attrib_name = "head"
                plotarray = np.atleast_3d(
                    out_obj.get_alldata().transpose()
                ).transpose()

                for per in range(plotarray.shape[0]):
                    for k in range(plotarray.shape[1]):
                        if kper is not None and per != kper:
                            continue
                        if mflay is not None and k != mflay:
                            continue
                        name = f"{attrib_name}{per}_{k}"
                        attrib_dict[name] = plotarray[per][k]

            elif isinstance(out_obj, CellBudgetFile):
                names = out_obj.get_unique_record_names(decode=True)

                for attrib_name in names:
                    plotarray = np.atleast_3d(
                        out_obj.get_data(text=attrib_name, full3D=True)
                    )

                    attrib_name = attrib_name.strip()
                    if attrib_name == "FLOW RIGHT FACE":
                        attrib_name = "FRF"
                    elif attrib_name == "FLOW FRONT FACE":
                        attrib_name = "FFF"
                    elif attrib_name == "FLOW LOWER FACE":
                        attrib_name = "FLF"
                    else:
                        pass
                    for per in range(plotarray.shape[0]):
                        for k in range(plotarray.shape[1]):
                            if kper is not None and per != kper:
                                continue
                            if mflay is not None and k != mflay:
                                continue
                            name = f"{attrib_name}{per}_{k}"
                            attrib_dict[name] = plotarray[per][k]

        if attrib_dict:
            shapefile_utils.write_grid_shapefile(f, ml.modelgrid, attrib_dict)

    else:
        msg = f"unrecognized export argument:{f}"
        if logger:
            logger.lraise(msg)
        else:
            raise NotImplementedError(msg)
    return f


def model_export(f, ml, fmt=None, **kwargs):
    """
    Method to export a model to a shapefile or netcdf file

    Parameters
    ----------
    f : str
        file path (".nc" for netcdf or ".shp" for shapefile)
        or NetCDF object or dict
    ml : flopy.modflow.mbase.ModelInterface object
        flopy model object
    fmt : str
        output format flag. 'vtk' will export to vtk
    **kwargs : keyword arguments
        modelgrid: flopy.discretization.Grid
            user supplied modelgrid object which will supercede the built
            in modelgrid object
        epsg : int
            epsg projection code
        prj : str
            prj file name
        if fmt is set to 'vtk', parameters of vtk.export_model

    """
    assert isinstance(ml, ModelInterface)
    package_names = kwargs.get("package_names", None)
    if package_names is None:
        package_names = [pak.name[0] for pak in ml.packagelist]

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, ml, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(
            f, ml, package_names=package_names, **kwargs
        )

    elif isinstance(f, NetCdf):

        for pak in ml.packagelist:
            if pak.name[0] in package_names:
                f = package_export(f, pak, **kwargs)
                assert f is not None
        return f

    elif isinstance(f, dict):
        for pak in ml.packagelist:
            f = package_export(f, pak, **kwargs)

    elif fmt == "vtk":
        # call vtk model export
        name = kwargs.get("name", ml.name)
        xml = kwargs.get("xml", False)
        masked_values = kwargs.get("masked_values", None)
        pvd = kwargs.get("pvd", False)
        smooth = kwargs.get("smooth", False)
        point_scalars = kwargs.get("point_scalars", False)
        binary = kwargs.get("binary", True)
        vertical_exageration = kwargs.get("vertical_exageration", 1)
        kpers = kwargs.get("kpers", None)
        package_names = kwargs.get("package_names", None)

        vtkobj = vtk.Vtk(
            ml,
            vertical_exageration=vertical_exageration,
            binary=binary,
            xml=xml,
            pvd=pvd,
            smooth=smooth,
            point_scalars=point_scalars,
        )
        vtkobj.add_model(
            ml, masked_values=masked_values, selpaklist=package_names
        )
        vtkobj.write(os.path.join(f, name), kpers)

    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")

    return f


def package_export(f, pak, fmt=None, **kwargs):
    """
    Method to export a package to shapefile or netcdf

    Parameters
    ----------
    f : str
        output file path (extension .shp for shapefile or .nc for netcdf)
        or NetCDF object or dict
    pak : flopy.pakbase.Package object
        package to export
    fmt : str
        output format flag. 'vtk' will export to vtk
    ** kwargs : keword arguments
        modelgrid: flopy.discretization.Grid
            user supplied modelgrid object which will supercede the built
            in modelgrid object
        epsg : int
            epsg projection code
        prj : str
            prj file name
        if fmt is set to 'vtk', parameters of vtk.export_package

    Returns
    -------
        f : NetCdf object or None

    """
    assert isinstance(pak, PackageInterface)

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, pak.parent, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        shapefile_utils.model_attributes_to_shapefile(
            f, pak.parent, package_names=pak.name, **kwargs
        )

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        for a in pak.data_list:
            if isinstance(a, DataInterface):
                if a.array is not None:
                    if (
                        a.data_type == DataType.array2d
                        and len(a.array.shape) == 2
                        and a.array.shape[1] > 0
                    ):
                        try:
                            f = array2d_export(f, a, **kwargs)
                        except:
                            f.logger.warn(f"error adding {a.name} as variable")
                    elif a.data_type == DataType.array3d:
                        f = array3d_export(f, a, **kwargs)
                    elif a.data_type == DataType.transient2d:
                        f = transient2d_export(f, a, **kwargs)
                    elif a.data_type == DataType.transientlist:
                        f = mflist_export(f, a, **kwargs)
                    elif isinstance(a, list):
                        for v in a:
                            if (
                                isinstance(a, DataInterface)
                                and v.data_type == DataType.array3d
                            ):
                                f = array3d_export(f, v, **kwargs)
        return f

    elif fmt == "vtk":
        # call vtk array export to folder
        name = kwargs.get("name", pak.name[0])
        xml = kwargs.get("xml", False)
        masked_values = kwargs.get("masked_values", None)
        pvd = kwargs.get("pvd", False)
        smooth = kwargs.get("smooth", False)
        point_scalars = kwargs.get("point_scalars", False)
        binary = kwargs.get("binary", True)
        vertical_exageration = kwargs.get("vertical_exageration", 1)
        kpers = kwargs.get("kpers", None)

        vtkobj = vtk.Vtk(
            pak.parent,
            vertical_exageration=vertical_exageration,
            binary=binary,
            xml=xml,
            pvd=pvd,
            smooth=smooth,
            point_scalars=point_scalars,
        )

        vtkobj.add_package(pak, masked_values=masked_values)
        vtkobj.write(os.path.join(f, name), kper=kpers)

        """
        vtk.export_package(
            pak.parent,
            pak.name,
            f,
            nanval=nanval,
            smooth=smooth,
            point_scalars=point_scalars,
            vtk_grid_type=vtk_grid_type,
            true2d=true2d,
            binary=binary,
            kpers=kpers,
        )
        """

    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")


def generic_array_export(
    f,
    array,
    var_name="generic_array",
    dimensions=("time", "layer", "y", "x"),
    precision_str="f4",
    units="unitless",
    **kwargs,
):
    """
    Method to export a generic array to NetCdf

    Parameters
    ----------
    f : str
        filename or existing export instance type (NetCdf only for now)
    array : np.ndarray
    var_name : str
        variable name
    dimensions : tuple
        netcdf dimensions
    precision_str : str
        binary precision string, default "f4"
    units : string
        units of array data
    **kwargs : keyword arguments
        model : flopy.modflow.mbase
            flopy model object

    """
    if isinstance(f, str) and f.lower().endswith(".nc"):
        assert "model" in kwargs.keys(), (
            "creating a new netCDF using generic_array_helper requires a "
            "'model' kwarg"
        )
        assert isinstance(kwargs["model"], BaseModel)
        f = NetCdf(f, kwargs.pop("model"), **kwargs)

    assert array.ndim == len(
        dimensions
    ), "generic_array_helper() array.ndim != dimensions"
    coords_dims = {
        "time": "time",
        "layer": "layer",
        "y": "latitude",
        "x": "longitude",
    }
    coords = " ".join([coords_dims[d] for d in dimensions])
    mn = kwargs.pop("min", -1.0e9)
    mx = kwargs.pop("max", 1.0e9)
    long_name = kwargs.pop("long_name", var_name)
    if len(kwargs) > 0:
        f.logger.warn(
            "generic_array_helper(): unrecognized kwargs:"
            + ",".join(kwargs.keys())
        )
    attribs = {"long_name": long_name}
    attribs["coordinates"] = coords
    attribs["units"] = units
    attribs["min"] = mn
    attribs["max"] = mx
    if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
        raise Exception(f"error processing {var_name}: all NaNs")
    try:
        var = f.create_variable(
            var_name,
            attribs,
            precision_str=precision_str,
            dimensions=dimensions,
        )
    except Exception as e:
        estr = f"error creating variable {var_name}:\n{e!s}"
        f.logger.warn(estr)
        raise Exception(estr)
    try:
        var[:] = array
    except Exception as e:
        estr = f"error setting array to variable {var_name}:\n{e!s}"
        f.logger.warn(estr)
        raise Exception(estr)
    return f


def mflist_export(f, mfl, **kwargs):
    """
    export helper for MfList instances

    Parameters
    -----------
    f : str
        file path or existing export instance type (NetCdf only for now)
    mfl : MfList instance
    **kwargs : keyword arguments
        modelgrid : flopy.discretization.Grid
            model grid instance which will supercede the flopy.model.modelgrid

    """
    if not isinstance(mfl, (DataListInterface, DataInterface)):
        err = (
            "mflist_helper only helps instances that support "
            "DataListInterface"
        )
        raise AssertionError(err)

    modelgrid = mfl.model.modelgrid
    if "modelgrid" in kwargs:
        modelgrid = kwargs.pop("modelgrid")

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, mfl.model, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        sparse = kwargs.get("sparse", False)
        kper = kwargs.get("kper", 0)
        squeeze = kwargs.get("squeeze", True)

        if modelgrid is None:
            raise Exception("MfList.to_shapefile: modelgrid is not set")
        elif modelgrid.grid_type == "USG-Unstructured":
            raise Exception(
                "Flopy does not support exporting to shapefile "
                "from a MODFLOW-USG unstructured grid."
            )

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
                        aname = f"{n}{k + 1}{int(kk) + 1}"
                        array_dict[aname] = array[k]
            shapefile_utils.write_grid_shapefile(f, modelgrid, array_dict)
        else:
            from ..export.shapefile_utils import recarray2shp
            from ..utils.geometry import Polygon

            df = mfl.get_dataframe(squeeze=squeeze)
            if "kper" in kwargs or df is None:
                ra = mfl[kper]
                verts = np.array(modelgrid.get_cell_vertices(ra.i, ra.j))
            elif df is not None:
                verts = np.array(
                    [
                        modelgrid.get_cell_vertices(i, df.j.values[ix])
                        for ix, i in enumerate(df.i.values)
                    ]
                )
                ra = df.to_records(index=False)
            epsg = kwargs.get("epsg", None)
            prj = kwargs.get("prj", None)
            polys = np.array([Polygon(v) for v in verts])
            recarray2shp(
                ra, geoms=polys, shpname=f, mg=modelgrid, epsg=epsg, prj=prj
            )

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        base_name = mfl.package.name[0].lower()
        # f.log("getting 4D masked arrays for {0}".format(base_name))
        # m4d = mfl.masked_4D_arrays
        # f.log("getting 4D masked arrays for {0}".format(base_name))

        # for name, array in m4d.items():
        for name, array in mfl.masked_4D_arrays_itr():
            var_name = f"{base_name}_{name}"
            if isinstance(f, dict):
                f[var_name] = array
                continue
            f.log(f"processing {name} attribute")

            units = None
            if var_name in NC_UNITS_FORMAT:
                units = NC_UNITS_FORMAT[var_name].format(
                    f.grid_units, f.time_units
                )
            precision_str = NC_PRECISION_TYPE[mfl.dtype[name].type]
            if var_name in NC_LONG_NAMES:
                attribs = {"long_name": NC_LONG_NAMES[var_name]}
            else:
                attribs = {"long_name": var_name}
            attribs["coordinates"] = "time layer latitude longitude"
            attribs["min"] = np.nanmin(array)
            attribs["max"] = np.nanmax(array)
            if np.isnan(attribs["min"]) or np.isnan(attribs["max"]):
                raise Exception(f"error processing {var_name}: all NaNs")

            if units is not None:
                attribs["units"] = units
            try:
                dim_tuple = ("time",) + f.dimension_names
                var = f.create_variable(
                    var_name,
                    attribs,
                    precision_str=precision_str,
                    dimensions=dim_tuple,
                )
            except Exception as e:
                estr = f"error creating variable {var_name}:\n{e!s}"
                f.logger.warn(estr)
                raise Exception(estr)

            array[np.isnan(array)] = f.fillvalue
            try:
                var[:] = array
            except Exception as e:
                estr = f"error setting array to variable {var_name}:\n{e!s}"
                f.logger.warn(estr)
                raise Exception(estr)
            f.log(f"processing {name} attribute")

        return f
    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")


def transient2d_export(f, t2d, fmt=None, **kwargs):
    """
    export helper for Transient2d instances

    Parameters
    -----------
    f : str
        filename or existing export instance type (NetCdf only for now)
    t2d : Transient2d instance
    fmt : str
        output format flag. 'vtk' will export to vtk
    **kwargs : keyword arguments
        min_valid : minimum valid value
        max_valid : maximum valid value
        modelgrid : flopy.discretization.Grid
            model grid instance which will supercede the flopy.model.modelgrid
        if fmt is set to 'vtk', parameters of vtk.export_transient

    """

    if not isinstance(t2d, DataInterface):
        err = (
            "transient2d_helper only helps instances that support "
            "DataInterface"
        )
        raise AssertionError(err)

    min_valid = kwargs.get("min_valid", -1.0e9)
    max_valid = kwargs.get("max_valid", 1.0e9)

    modelgrid = t2d.model.modelgrid
    if "modelgrid" in kwargs:
        modelgrid = kwargs.pop("modelgrid")

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, t2d.model, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        array_dict = {}
        for kper in range(t2d.model.modeltime.nper):
            u2d = t2d[kper]
            name = f"{shapefile_utils.shape_attr_name(u2d.name)}_{kper + 1}"
            array_dict[name] = u2d.array
        shapefile_utils.write_grid_shapefile(f, modelgrid, array_dict)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        # mask the array is defined by any row col with at lease
        # one active cell
        mask = None
        if modelgrid.idomain is not None:
            ibnd = np.abs(modelgrid.idomain).sum(axis=0)
            mask = ibnd == 0

        # f.log("getting 4D array for {0}".format(t2d.name_base))
        array = t2d.array
        # f.log("getting 4D array for {0}".format(t2d.name_base))
        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int32, np.int64]:
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

        var_name = t2d.name.replace("_", "")
        if isinstance(f, dict):
            array[array == netcdf.FILLVALUE] = np.NaN
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"

        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(
                f.grid_units, f.time_units
            )
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
            raise Exception(f"error processing {var_name}: all NaNs")
        try:
            dim_tuple = ("time",) + f.dimension_names
            var = f.create_variable(
                var_name,
                attribs,
                precision_str=precision_str,
                dimensions=dim_tuple,
            )
        except Exception as e:
            estr = f"error creating variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:, 0] = array
        except Exception as e:
            estr = f"error setting array to variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    elif fmt == "vtk":
        name = kwargs.get("name", t2d.name)
        xml = kwargs.get("xml", False)
        masked_values = kwargs.get("masked_values", None)
        pvd = kwargs.get("pvd", False)
        smooth = kwargs.get("smooth", False)
        point_scalars = kwargs.get("point_scalars", False)
        binary = kwargs.get("binary", True)
        vertical_exageration = kwargs.get("vertical_exageration", 1)
        kpers = kwargs.get("kpers", None)

        if t2d.array is not None:
            if hasattr(t2d, "transient_2ds"):
                d = t2d.transient_2ds
            else:
                d = {ix: i for ix, i in enumerate(t2d.array)}
        else:
            raise AssertionError("No data available to export")

        vtkobj = vtk.Vtk(
            t2d.model,
            vertical_exageration=vertical_exageration,
            binary=binary,
            xml=xml,
            pvd=pvd,
            smooth=smooth,
            point_scalars=point_scalars,
        )
        vtkobj.add_transient_array(d, name=name, masked_values=masked_values)
        vtkobj.write(os.path.join(f, name), kper=kpers)

    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")


def array3d_export(f, u3d, fmt=None, **kwargs):
    """
    export helper for Transient2d instances

    Parameters
    -----------
    f : str
        filename or existing export instance type (NetCdf only for now)
    u3d : Util3d instance
    fmt : str
        output format flag. 'vtk' will export to vtk
    **kwargs : keyword arguments
        min_valid : minimum valid value
        max_valid : maximum valid value
        modelgrid : flopy.discretization.Grid
            model grid instance which will supercede the flopy.model.modelgrid
        if fmt is set to 'vtk', parameters of vtk.export_array

    """

    assert isinstance(
        u3d, DataInterface
    ), "array3d_export only helps instances that support DataInterface"

    min_valid = kwargs.get("min_valid", -1.0e9)
    max_valid = kwargs.get("max_valid", 1.0e9)

    modelgrid = u3d.model.modelgrid
    if "modelgrid" in kwargs:
        modelgrid = kwargs.pop("modelgrid")

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, u3d.model, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        array_dict = {}
        for ilay in range(modelgrid.nlay):
            u2d = u3d[ilay]
            if isinstance(u2d, np.ndarray):
                dname = u3d.name
                array = u2d
            else:
                dname = u2d.name
                array = u2d.array
            name = f"{shapefile_utils.shape_attr_name(dname)}_{ilay + 1}"
            array_dict[name] = array
        shapefile_utils.write_grid_shapefile(f, modelgrid, array_dict)

    elif isinstance(f, NetCdf) or isinstance(f, dict):
        var_name = u3d.name
        if isinstance(var_name, list) or isinstance(var_name, tuple):
            var_name = var_name[0]
        var_name = var_name.replace(" ", "_").lower()
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
        if modelgrid.idomain is not None and "ibound" not in var_name:
            mask = modelgrid.idomain == 0

        if mask is not None and array.shape != mask.shape:
            # f.log("broadcasting 3D array for {0}".format(var_name))
            full_array = np.empty(mask.shape)
            full_array[:] = np.NaN
            full_array[: array.shape[0]] = array
            array = full_array
            # f.log("broadcasting 3D array for {0}".format(var_name))

        # runtime warning issued in some cases - need to track down cause
        # happens when NaN is already in array
        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int32, np.int64]:
                # if u3d.model.modelgrid.bas6 is not None and "ibound" not
                # in var_name:
                #    array[u3d.model.modelgrid.bas6.ibound.array == 0] =
                # np.NaN
                # elif u3d.model.btn is not None and 'icbund' not in var_name:
                #    array[u3d.model.modelgrid.btn.icbund.array == 0] = np.NaN
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
                if modelgrid.idomain is not None and "ibound" not in var_name:
                    array[modelgrid.idomain == 0] = netcdf.FILLVALUE

        if isinstance(f, dict):
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"
        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(
                f.grid_units, f.time_units
            )
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
            raise Exception(f"error processing {var_name}: all NaNs")
        try:
            var = f.create_variable(
                var_name,
                attribs,
                precision_str=precision_str,
                dimensions=f.dimension_names,
            )
        except Exception as e:
            estr = f"error creating variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = f"error setting array to variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    elif fmt == "vtk":
        # call vtk array export to folder
        name = kwargs.get("name", u3d.name)
        xml = kwargs.get("xml", False)
        masked_values = kwargs.get("masked_values", None)
        smooth = kwargs.get("smooth", False)
        point_scalars = kwargs.get("point_scalars", False)
        binary = kwargs.get("binary", True)
        vertical_exageration = kwargs.get("vertical_exageration", 1)

        if isinstance(name, list) or isinstance(name, tuple):
            name = name[0]

        vtkobj = vtk.Vtk(
            u3d.model,
            smooth=smooth,
            point_scalars=point_scalars,
            binary=binary,
            xml=xml,
            vertical_exageration=vertical_exageration,
        )
        vtkobj.add_array(u3d.array, name, masked_values=masked_values)
        vtkobj.write(os.path.join(f, name))

    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")


def array2d_export(f, u2d, fmt=None, **kwargs):
    """
    export helper for Util2d instances

    Parameters
    ----------
    f : str
        filename or existing export instance type (NetCdf only for now)
    u2d : Util2d instance
    fmt : str
        output format flag. 'vtk' will export to vtk
    **kwargs : keyword arguments
        min_valid : minimum valid value
        max_valid : maximum valid value
        modelgrid : flopy.discretization.Grid
            model grid instance which will supercede the flopy.model.modelgrid
        if fmt is set to 'vtk', parameters of vtk.export_array

    """
    assert isinstance(
        u2d, DataInterface
    ), "util2d_helper only helps instances that support DataInterface"
    assert len(u2d.array.shape) == 2, "util2d_helper only supports 2D arrays"

    min_valid = kwargs.get("min_valid", -1.0e9)
    max_valid = kwargs.get("max_valid", 1.0e9)

    modelgrid = u2d.model.modelgrid
    if "modelgrid" in kwargs:
        modelgrid = kwargs.pop("modelgrid")

    if isinstance(f, str) and f.lower().endswith(".nc"):
        f = NetCdf(f, u2d.model, **kwargs)

    if isinstance(f, str) and f.lower().endswith(".shp"):
        name = shapefile_utils.shape_attr_name(u2d.name, keep_layer=True)
        shapefile_utils.write_grid_shapefile(f, modelgrid, {name: u2d.array})
        return

    elif isinstance(f, str) and f.lower().endswith(".asc"):
        export_array(modelgrid, f, u2d.array, **kwargs)
        return

    elif isinstance(f, NetCdf) or isinstance(f, dict):

        # try to mask the array - assume layer 1 ibound is a good mask
        # f.log("getting 2D array for {0}".format(u2d.name))
        array = u2d.array
        # f.log("getting 2D array for {0}".format(u2d.name))

        with np.errstate(invalid="ignore"):
            if array.dtype not in [int, np.int32, np.int64]:
                if (
                    modelgrid.idomain is not None
                    and "ibound" not in u2d.name.lower()
                    and "idomain" not in u2d.name.lower()
                ):
                    array[modelgrid.idomain[0, :, :] == 0] = np.NaN
                array[array <= min_valid] = np.NaN
                array[array >= max_valid] = np.NaN
                mx, mn = np.nanmax(array), np.nanmin(array)
            else:
                mx, mn = np.nanmax(array), np.nanmin(array)
                array[array <= min_valid] = netcdf.FILLVALUE
                array[array >= max_valid] = netcdf.FILLVALUE
                if (
                    modelgrid.idomain is not None
                    and "ibound" not in u2d.name.lower()
                    and "idomain" not in u2d.name.lower()
                    and "icbund" not in u2d.name.lower()
                ):
                    array[modelgrid.idomain[0, :, :] == 0] = netcdf.FILLVALUE
        var_name = u2d.name
        if isinstance(f, dict):
            f[var_name] = array
            return f

        array[np.isnan(array)] = f.fillvalue
        units = "unitless"

        if var_name in NC_UNITS_FORMAT:
            units = NC_UNITS_FORMAT[var_name].format(
                f.grid_units, f.time_units
            )
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
            raise Exception(f"error processing {var_name}: all NaNs")
        try:
            var = f.create_variable(
                var_name,
                attribs,
                precision_str=precision_str,
                dimensions=f.dimension_names[1:],
            )
        except Exception as e:
            estr = f"error creating variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        try:
            var[:] = array
        except Exception as e:
            estr = f"error setting array to variable {var_name}:\n{e!s}"
            f.logger.warn(estr)
            raise Exception(estr)
        return f

    elif fmt == "vtk":

        name = kwargs.get("name", u2d.name)
        xml = kwargs.get("xml", False)
        masked_values = kwargs.get("masked_values", None)
        smooth = kwargs.get("smooth", False)
        point_scalars = kwargs.get("point_scalars", False)
        binary = kwargs.get("binary", True)
        vertical_exageration = kwargs.get("vertical_exageration", 1)

        if isinstance(name, list) or isinstance(name, tuple):
            name = name[0]

        vtkobj = vtk.Vtk(
            u2d.model,
            smooth=smooth,
            point_scalars=point_scalars,
            binary=binary,
            xml=xml,
            vertical_exageration=vertical_exageration,
        )
        array = np.ones((modelgrid.nnodes,)) * np.nan
        array[0 : u2d.array.size] = np.ravel(u2d.array)
        vtkobj.add_array(array, name, masked_values=masked_values)
        vtkobj.write(os.path.join(f, name))
    else:
        raise NotImplementedError(f"unrecognized export argument:{f}")


def export_array(
    modelgrid, filename, a, nodata=-9999, fieldname="value", **kwargs
):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    modelgrid : flopy.discretization.StructuredGrid object
        model grid
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
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """

    if filename.lower().endswith(".asc"):
        if (
            len(np.unique(modelgrid.delr))
            != len(np.unique(modelgrid.delc))
            != 1
            or modelgrid.delr[0] != modelgrid.delc[0]
        ):
            raise ValueError("Arc ascii arrays require a uniform grid.")

        xoffset, yoffset = modelgrid.xoffset, modelgrid.yoffset
        cellsize = modelgrid.delr[0]
        fmt = kwargs.get("fmt", "%.18e")
        a = a.copy()
        a[np.isnan(a)] = nodata
        if modelgrid.angrot != 0:
            ndimage = import_optional_dependency(
                "scipy.ndimage",
                error_message="exporting rotated grids requires SciPy.",
            )

            a = ndimage.rotate(a, modelgrid.angrot, cval=nodata)
            height_rot, width_rot = a.shape
            xmin, xmax, ymin, ymax = modelgrid.extent
            dx = (xmax - xmin) / width_rot
            dy = (ymax - ymin) / height_rot
            cellsize = np.max((dx, dy))
            xoffset, yoffset = xmin, ymin

        filename = (
            ".".join(filename.split(".")[:-1]) + ".asc"
        )  # enforce .asc ending
        nrow, ncol = a.shape
        a[np.isnan(a)] = nodata
        txt = f"ncols  {ncol}\n"
        txt += f"nrows  {nrow}\n"
        txt += f"xllcorner  {xoffset:f}\n"
        txt += f"yllcorner  {yoffset:f}\n"
        txt += f"cellsize  {cellsize}\n"
        # ensure that nodata fmt consistent w values
        txt += f"NODATA_value  {fmt}\n" % (nodata)
        with open(filename, "w") as output:
            output.write(txt)
        with open(filename, "ab") as output:
            np.savetxt(output, a, **kwargs)
        print(f"wrote {filename}")

    elif filename.lower().endswith(".tif"):
        if (
            len(np.unique(modelgrid.delr))
            != len(np.unique(modelgrid.delc))
            != 1
            or modelgrid.delr[0] != modelgrid.delc[0]
        ):
            raise ValueError("GeoTIFF export require a uniform grid.")
        rasterio = import_optional_dependency(
            "rasterio",
            error_message="GeoTIFF export requires the rasterio.",
        )
        dxdy = modelgrid.delc[0]
        # because this is only implemented for a structured grid,
        # we can get the xul and yul coordinate from modelgrid.xvertices(0, 0)
        verts = modelgrid.get_cell_vertices(0, 0)
        xul, yul = verts[0]
        trans = (
            rasterio.Affine.translation(xul, yul)
            * rasterio.Affine.rotation(modelgrid.angrot)
            * rasterio.Affine.scale(dxdy, -dxdy)
        )

        # third dimension is the number of bands
        a = a.copy()
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))
        if a.dtype.name == "int64":
            a = a.astype("int32")
            dtype = rasterio.int32
        elif a.dtype.name == "int32":
            dtype = rasterio.int32
        elif a.dtype.name == "float64":
            dtype = rasterio.float64
        elif a.dtype.name == "float32":
            dtype = rasterio.float32
        else:
            msg = f'ERROR: invalid dtype "{a.dtype.name}"'
            raise TypeError(msg)

        meta = {
            "count": a.shape[0],
            "width": a.shape[2],
            "height": a.shape[1],
            "nodata": nodata,
            "dtype": dtype,
            "driver": "GTiff",
            "crs": modelgrid.proj4,
            "transform": trans,
        }
        meta.update(kwargs)
        with rasterio.open(filename, "w", **meta) as dst:
            dst.write(a)
        print(f"wrote {filename}")

    elif filename.lower().endswith(".shp"):
        from ..export.shapefile_utils import write_grid_shapefile

        epsg = kwargs.get("epsg", None)
        prj = kwargs.get("prj", None)
        if epsg is None and prj is None:
            epsg = modelgrid.epsg
        write_grid_shapefile(
            filename,
            modelgrid,
            array_dict={fieldname: a},
            nan_val=nodata,
            epsg=epsg,
            prj=prj,
        )


def export_contours(
    filename,
    contours,
    fieldname="level",
    epsg=None,
    prj=None,
    **kwargs,
):
    """
    Convert matplotlib contour plot object to shapefile.

    Parameters
    ----------
    filename : str
        path of output shapefile
    contours : matplotlib.contour.QuadContourSet or list of them
        (object returned by matplotlib.pyplot.contour)
    fieldname : str
        gis attribute table field name
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.
    **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

    Returns
    -------
    df : dataframe of shapefile contents

    """
    from ..utils.geometry import LineString
    from .shapefile_utils import recarray2shp

    if not isinstance(contours, list):
        contours = [contours]

    geoms = []
    level = []
    for ctr in contours:
        levels = ctr.levels
        for i, c in enumerate(ctr.collections):
            paths = c.get_paths()
            geoms += [LineString(p.vertices) for p in paths]
            level += list(np.ones(len(paths)) * levels[i])

    # convert the dictionary to a recarray
    ra = np.array(level, dtype=[(fieldname, float)]).view(np.recarray)

    recarray2shp(ra, geoms, filename, epsg=epsg, prj=prj, **kwargs)
    return


def export_contourf(
    filename, contours, fieldname="level", epsg=None, prj=None, **kwargs
):
    """
    Write matplotlib filled contours to shapefile.

    Parameters
    ----------
    filename : str
        name of output shapefile (e.g. myshp.shp)
    contours : matplotlib.contour.QuadContourSet or list of them
        (object returned by matplotlib.pyplot.contourf)
    fieldname : str
        Name of shapefile attribute field to contain the contour level.  The
        fieldname column in the attribute table will contain the lower end of
        the range represented by the polygon.  Default is 'level'.
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.

    **kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

    Returns
    -------
    None

    Examples
    --------
    >>> import flopy
    >>> import matplotlib.pyplot as plt
    >>> from flopy.export.utils import export_contourf
    >>> a = np.random.random((10, 10))
    >>> cs = plt.contourf(a)
    >>> export_contourf('myfilledcontours.shp', cs)

    """
    from ..utils.geometry import Polygon, is_clockwise
    from .shapefile_utils import recarray2shp

    geoms = []
    level = []

    if not isinstance(contours, list):
        contours = [contours]

    for c in contours:
        levels = c.levels
        for idx, col in enumerate(c.collections):
            # Loop through all polygons that have the same intensity level
            for contour_path in col.get_paths():
                # Create the polygon(s) for this intensity level
                poly = None
                for ncp, cp in enumerate(contour_path.to_polygons()):
                    x = cp[:, 0]
                    y = cp[:, 1]
                    verts = [(i[0], i[1]) for i in zip(x, y)]
                    new_shape = Polygon(verts)

                    if ncp == 0:
                        poly = new_shape
                    else:
                        # check if this is a multipolygon by checking vertex
                        # order.
                        if is_clockwise(verts):
                            # Clockwise is a hole, set to interiors
                            if not poly.interiors:
                                poly.interiors = [new_shape.exterior]
                            else:
                                poly.interiors.append(new_shape.exterior)
                        else:
                            geoms.append(poly)
                            level.append(levels[idx])
                            poly = new_shape

                if poly is not None:
                    # store geometry object
                    geoms.append(poly)
                    level.append(levels[idx])

    print(f"Writing {len(level)} polygons")

    # Create recarray
    ra = np.array(level, dtype=[(fieldname, float)]).view(np.recarray)

    recarray2shp(ra, geoms, filename, epsg=epsg, prj=prj, **kwargs)
    return


def export_array_contours(
    modelgrid,
    filename,
    a,
    fieldname="level",
    interval=None,
    levels=None,
    maxlevels=1000,
    epsg=None,
    prj=None,
    **kwargs,
):
    """
    Contour an array using matplotlib; write shapefile of contours.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object
        model grid object
    filename : str
        Path of output file with '.shp' extention.
    a : 2D numpy array
        Array to contour
    fieldname : str
        gis field name
    interval : float
        interval to calculate levels from
    levels : list
        list of contour levels
    maxlevels : int
        maximum number of contour levels
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.
    **kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

    """
    import matplotlib.pyplot as plt

    if epsg is None:
        epsg = modelgrid.epsg
    if prj is None:
        prj = modelgrid.proj4

    if interval is not None:
        imin = np.nanmin(a)
        imax = np.nanmax(a)
        nlevels = np.round(np.abs(imax - imin) / interval, 2)
        msg = f"{nlevels:.0f} levels at interval of {interval} > maxlevels={maxlevels}"
        assert nlevels < maxlevels, msg
        levels = np.arange(imin, imax, interval)
    ax = plt.subplots()[-1]
    ctr = contour_array(modelgrid, ax, a, levels=levels)
    export_contours(filename, ctr, fieldname, epsg, prj, **kwargs)
    plt.close()


def contour_array(modelgrid, ax, a, **kwargs):
    """
    Create a QuadMesh plot of the specified array using pcolormesh

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object
        modelgrid object
    ax : matplotlib.axes.Axes
        ax to add the contours

    a : np.ndarray
        array to contour

    Returns
    -------
    contour_set : ContourSet

    """
    from ..plot import PlotMapView

    kwargs["ax"] = ax
    pmv = PlotMapView(modelgrid=modelgrid)
    contour_set = pmv.contour_array(a=a, **kwargs)

    return contour_set
