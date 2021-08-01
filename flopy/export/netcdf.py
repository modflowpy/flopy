import os
import platform
import socket
import copy
import json
import numpy as np
from datetime import datetime
import time
from .metadata import acdd
import flopy

# globals
FILLVALUE = -99999.9
ITMUNI = {
    0: "undefined",
    1: "seconds",
    2: "minutes",
    3: "hours",
    4: "days",
    5: "years",
}
PRECISION_STRS = ["f4", "f8", "i4"]

STANDARD_VARS = ["longitude", "latitude", "layer", "elevation", "time"]

path = os.path.split(__file__)[0]
with open(path + "/longnames.json") as f:
    NC_LONG_NAMES = json.load(f)


class Logger:
    """
    Basic class for logging events during the linear analysis calculations
    if filename is passed, then an file handle is opened

    Parameters
    ----------
    filename : bool or string
        if string, it is the log file to write.  If a bool, then log is
        written to the screen. echo (bool): a flag to force screen output

    Attributes
    ----------
    items : dict
        tracks when something is started.  If a log entry is
        not in items, then it is treated as a new entry with the string
        being the key and the datetime as the value.  If a log entry is
        in items, then the end time and delta time are written and
        the item is popped from the keys

    """

    def __init__(self, filename, echo=False):
        self.items = {}
        self.echo = bool(echo)
        if filename == True:
            self.echo = True
            self.filename = None
        elif filename:
            self.f = open(filename, "w", 0)  # unbuffered
            self.t = datetime.now()
            self.log("opening " + str(filename) + " for logging")
        else:
            self.filename = None

    def log(self, phrase):
        """
        log something that happened

        Parameters
        ----------
        phrase : str
            the thing that happened

        """
        pass
        t = datetime.now()
        if phrase in self.items.keys():
            s = "{} finished: {}, took: {}\n".format(
                t, phrase, t - self.items[phrase]
            )
            if self.echo:
                print(s)
            if self.filename:
                self.f.write(s)
            self.items.pop(phrase)
        else:
            s = str(t) + " starting: " + str(phrase) + "\n"
            if self.echo:
                print(
                    s,
                )
            if self.filename:
                self.f.write(s)
            self.items[phrase] = copy.deepcopy(t)

    def warn(self, message):
        """
        Write a warning to the log file

        Parameters
        ----------
        message : str
            the warning text

        """
        s = str(datetime.now()) + " WARNING: " + message + "\n"
        if self.echo:
            print(
                s,
            )
        if self.filename:
            self.f.write(s)
        return


class NetCdf:
    """
    Support for writing a netCDF4 compliant file from a flopy model

    Parameters
    ----------
    output_filename : str
        Name of the .nc file to write
    model : flopy model instance
    time_values : the entries for the time dimension
        if not None, the constructor will initialize
        the file.  If None, the perlen array of ModflowDis
        will be used
    z_positive : str ('up' or 'down')
        Positive direction of vertical coordinates written to NetCDF file.
        (default 'down')
    verbose : if True, stdout is verbose.  If str, then a log file
        is written to the verbose file
    forgive : what to do if a duplicate variable name is being created.  If
        True, then the newly requested var is skipped.  If False, then
        an exception is raised.
    **kwargs : keyword arguments
        modelgrid : flopy.discretization.Grid instance
            user supplied model grid which will be used in lieu of the model
            object modelgrid for netcdf production

    Notes
    -----
    This class relies heavily on the grid and modeltime objects,
    including these attributes: lenuni, itmuni, start_datetime, and proj4.
    Make sure these attributes have meaningful values.

    """

    def __init__(
        self,
        output_filename,
        model,
        time_values=None,
        z_positive="up",
        verbose=None,
        prj=None,
        logger=None,
        forgive=False,
        **kwargs
    ):

        assert output_filename.lower().endswith(".nc")
        if verbose is None:
            verbose = model.verbose
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(verbose)
        self.var_attr_dict = {}
        self.log = self.logger.log
        if os.path.exists(output_filename):
            self.logger.warn("removing existing nc file: " + output_filename)
            os.remove(output_filename)
        self.output_filename = output_filename

        self.forgive = bool(forgive)

        self.model = model
        self.model_grid = model.modelgrid
        if "modelgrid" in kwargs:
            self.model_grid = kwargs.pop("modelgrid")
        self.model_time = model.modeltime
        if prj is not None:
            self.model_grid.proj4 = prj
        if self.model_grid.grid_type == "structured":
            self.dimension_names = ("layer", "y", "x")
            STANDARD_VARS.extend(["delc", "delr"])
        # elif self.model_grid.grid_type == 'vertex':
        #    self.dimension_names = ('layer', 'ncpl')
        else:
            raise Exception(
                "Grid type {} not supported.".format(self.model_grid.grid_type)
            )
        self.shape = self.model_grid.shape

        try:
            import dateutil.parser
        except:
            print(
                "python-dateutil is not installed\n"
                "try pip install python-dateutil"
            )
            return

        self.start_datetime = self._dt_str(
            dateutil.parser.parse(self.model_time.start_datetime)
        )
        self.logger.warn("start datetime:{0}".format(str(self.start_datetime)))

        proj4_str = self.model_grid.proj4
        if proj4_str is None:
            proj4_str = "epsg:4326"
            self.log(
                "Warning: model has no coordinate reference system specified. "
                "Using default proj4 string: {}".format(proj4_str)
            )
        self.proj4_str = proj4_str
        self.grid_units = self.model_grid.units
        self.z_positive = z_positive
        if self.grid_units is None:
            self.grid_units = "undefined"
        assert self.grid_units in ["feet", "meters", "undefined"], (
            "unsupported length units: " + self.grid_units
        )

        self.time_units = self.model_time.time_units

        # this gives us confidence that every NetCdf instance
        # has the same attributes
        self.log("initializing attributes")
        self._initialize_attributes()
        self.log("initializing attributes")

        self.time_values_arg = time_values

        self.log("initializing file")
        self.initialize_file(time_values=self.time_values_arg)
        self.log("initializing file")

    def __add__(self, other):
        new_net = NetCdf.zeros_like(self)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] + other
                )
        elif isinstance(other, NetCdf):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] + other.nc.variables[vname][:]
                )
        else:
            raise Exception(
                "NetCdf.__add__(): unrecognized other:{0}".format(
                    str(type(other))
                )
            )
        return new_net

    def __sub__(self, other):
        new_net = NetCdf.zeros_like(self)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] - other
                )
        elif isinstance(other, NetCdf):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] - other.nc.variables[vname][:]
                )
        else:
            raise Exception(
                "NetCdf.__sub__(): unrecognized other:{0}".format(
                    str(type(other))
                )
            )
        return new_net

    def __mul__(self, other):
        new_net = NetCdf.zeros_like(self)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] * other
                )
        elif isinstance(other, NetCdf):
            for vname in self.var_attr_dict.keys():
                new_net.nc.variables[vname][:] = (
                    self.nc.variables[vname][:] * other.nc.variables[vname][:]
                )
        else:
            raise Exception(
                "NetCdf.__mul__(): unrecognized other:{0}".format(
                    str(type(other))
                )
            )
        return new_net

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        new_net = NetCdf.zeros_like(self)
        with np.errstate(invalid="ignore"):
            if np.isscalar(other) or isinstance(other, np.ndarray):
                for vname in self.var_attr_dict.keys():
                    new_net.nc.variables[vname][:] = (
                        self.nc.variables[vname][:] / other
                    )
            elif isinstance(other, NetCdf):
                for vname in self.var_attr_dict.keys():
                    new_net.nc.variables[vname][:] = (
                        self.nc.variables[vname][:]
                        / other.nc.variables[vname][:]
                    )
            else:
                raise Exception(
                    "NetCdf.__sub__(): unrecognized other:{0}".format(
                        str(type(other))
                    )
                )
            return new_net

    def append(self, other, suffix="_1"):
        assert isinstance(other, NetCdf) or isinstance(other, dict)
        if isinstance(other, NetCdf):
            for vname in other.var_attr_dict.keys():
                attrs = other.var_attr_dict[vname].copy()
                var = other.nc.variables[vname]
                new_vname = vname

                if vname in self.nc.variables.keys():
                    if vname not in STANDARD_VARS:
                        new_vname = vname + suffix
                        if "long_name" in attrs:
                            attrs["long_name"] += " " + suffix
                    else:
                        continue
                assert (
                    new_vname not in self.nc.variables.keys()
                ), "var already exists:{0} in {1}".format(
                    new_vname, ",".join(self.nc.variables.keys())
                )
                attrs["max"] = var[:].max()
                attrs["min"] = var[:].min()
                new_var = self.create_variable(
                    new_vname, attrs, var.dtype, dimensions=var.dimensions
                )
                new_var[:] = var[:]
        else:
            for vname, array in other.items():
                vname_norm = self.normalize_name(vname)
                assert (
                    vname_norm in self.nc.variables.keys()
                ), "dict var not in self.vars:{0}-->".format(vname) + ",".join(
                    self.nc.variables.keys()
                )

                new_vname = vname_norm + suffix
                assert new_vname not in self.nc.variables.keys()
                attrs = self.var_attr_dict[vname_norm].copy()
                attrs["max"] = np.nanmax(array)
                attrs["min"] = np.nanmin(array)
                attrs["name"] = new_vname
                attrs["long_name"] = attrs["long_name"] + " " + suffix
                var = self.nc.variables[vname_norm]
                # assert var.shape == array.shape,\
                #    "{0} shape ({1}) doesn't make array shape ({2})".\
                #        format(new_vname,str(var.shape),str(array.shape))
                new_var = self.create_variable(
                    new_vname, attrs, var.dtype, dimensions=var.dimensions
                )
                try:
                    new_var[:] = array
                except:
                    new_var[:, 0] = array

        return

    def copy(self, output_filename):
        new_net = NetCdf.zeros_like(self, output_filename=output_filename)
        for vname in self.var_attr_dict.keys():
            new_net.nc.variables[vname][:] = self.nc.variables[vname][:]
        return new_net

    @classmethod
    def zeros_like(
        cls, other, output_filename=None, verbose=None, logger=None
    ):
        new_net = NetCdf.empty_like(
            other,
            output_filename=output_filename,
            verbose=verbose,
            logger=logger,
        )
        # add the vars to the instance
        for vname in other.var_attr_dict.keys():
            if new_net.nc.variables.get(vname) is not None:
                new_net.logger.warn(
                    "variable {0} already defined, skipping".format(vname)
                )
                continue
            new_net.log("adding variable {0}".format(vname))
            var = other.nc.variables[vname]
            data = var[:]
            try:
                mask = data.mask
                data = np.array(data)
            except:
                mask = None
            new_data = np.zeros_like(data)
            new_data[mask] = FILLVALUE
            new_var = new_net.create_variable(
                vname,
                other.var_attr_dict[vname],
                var.dtype,
                dimensions=var.dimensions,
            )
            new_var[:] = new_data
            new_net.log("adding variable {0}".format(vname))
        global_attrs = {}
        for attr in other.nc.ncattrs():
            if attr not in new_net.nc.ncattrs():
                global_attrs[attr] = other.nc[attr]
        new_net.add_global_attributes(global_attrs)
        return new_net

    @classmethod
    def empty_like(
        cls, other, output_filename=None, verbose=None, logger=None
    ):
        if output_filename is None:
            output_filename = (
                str(time.mktime(datetime.now().timetuple())) + ".nc"
            )

        while os.path.exists(output_filename):
            print("{}...already exists".format(output_filename))
            output_filename = (
                str(time.mktime(datetime.now().timetuple())) + ".nc"
            )
            print("creating temporary netcdf file..." + output_filename)

        new_net = cls(
            output_filename,
            other.model,
            time_values=other.time_values_arg,
            verbose=verbose,
            logger=logger,
        )
        return new_net

    def difference(
        self, other, minuend="self", mask_zero_diff=True, onlydiff=True
    ):
        """
        make a new NetCDF instance that is the difference with another
        netcdf file

        Parameters
        ----------
        other : either an str filename of a netcdf file or
            a netCDF4 instance

        minuend : (optional) the order of the difference operation.
            Default is self (e.g. self - other).  Can be "self" or "other"

        mask_zero_diff : bool flag to mask differences that are zero.  If
            True, positions in the difference array that are zero will be set
            to self.fillvalue

        only_diff : bool flag to only add non-zero diffs to output file

        Returns
        -------
        net NetCDF instance

        Notes
        -----
        assumes the current NetCDF instance has been populated.  The
        variable names and dimensions between the two files must match
        exactly. The name of the new .nc file is
        <self.output_filename>.diff.nc.  The masks from both self and
        other are carried through to the new instance

        """

        assert (
            self.nc is not None
        ), "can't call difference() if nc hasn't been populated"
        try:
            import netCDF4
        except Exception as e:
            mess = "error import netCDF4: {0}".format(str(e))
            self.logger.warn(mess)
            raise Exception(mess)

        if isinstance(other, str):
            assert os.path.exists(
                other
            ), "filename 'other' not found:{0}".format(other)
            other = netCDF4.Dataset(other, "r")

        assert isinstance(other, netCDF4.Dataset)

        # check for similar variables
        self_vars = set(self.nc.variables.keys())
        other_vars = set(other.variables)
        diff = self_vars.symmetric_difference(other_vars)
        if len(diff) > 0:
            self.logger.warn(
                "variables are not the same between the two nc files: "
                + ",".join(diff)
            )
            return

        # check for similar dimensions
        self_dimens = self.nc.dimensions
        other_dimens = other.dimensions
        for d in self_dimens.keys():
            if d not in other_dimens:
                self.logger.warn("missing dimension in other:{0}".format(d))
                return
            if len(self_dimens[d]) != len(other_dimens[d]):
                self.logger.warn(
                    "dimension not consistent: "
                    "{0}:{1}".format(self_dimens[d], other_dimens[d])
                )
                return
        # should be good to go
        time_values = self.nc.variables.get("time")[:]
        new_net = NetCdf(
            self.output_filename.replace(".nc", ".diff.nc"),
            self.model,
            time_values=time_values,
        )
        # add the vars to the instance
        for vname in self_vars:
            if (
                vname not in self.var_attr_dict
                or new_net.nc.variables.get(vname) is not None
            ):
                self.logger.warn("skipping variable: {0}".format(vname))
                continue
            self.log("processing variable {0}".format(vname))
            s_var = self.nc.variables[vname]
            o_var = other.variables[vname]
            s_data = s_var[:]
            o_data = o_var[:]
            o_mask, s_mask = None, None

            # keep the masks to apply later
            if isinstance(s_data, np.ma.MaskedArray):
                self.logger.warn("masked array for {0}".format(vname))
                s_mask = s_data.mask
                s_data = np.array(s_data)
                s_data[s_mask] = 0.0
            else:
                np.nan_to_num(s_data)

            if isinstance(o_data, np.ma.MaskedArray):
                o_mask = o_data.mask
                o_data = np.array(o_data)
                o_data[o_mask] = 0.0
            else:
                np.nan_to_num(o_data)

            # difference with self
            if minuend.lower() == "self":
                d_data = s_data - o_data
            elif minuend.lower() == "other":
                d_data = o_data - s_data
            else:
                mess = "unrecognized minuend {0}".format(minuend)
                self.logger.warn(mess)
                raise Exception(mess)

            # check for non-zero diffs
            if onlydiff and d_data.sum() == 0.0:
                self.logger.warn(
                    "var {0} has zero differences, skipping...".format(vname)
                )
                continue

            self.logger.warn(
                "resetting diff attrs max,min:{0},{1}".format(
                    d_data.min(), d_data.max()
                )
            )
            attrs = self.var_attr_dict[vname].copy()
            attrs["max"] = np.nanmax(d_data)
            attrs["min"] = np.nanmin(d_data)
            # reapply masks
            if s_mask is not None:
                self.log("applying self mask")
                s_mask[d_data != 0.0] = False
                d_data[s_mask] = FILLVALUE
                self.log("applying self mask")
            if o_mask is not None:
                self.log("applying other mask")
                o_mask[d_data != 0.0] = False
                d_data[o_mask] = FILLVALUE
                self.log("applying other mask")

            d_data[np.isnan(d_data)] = FILLVALUE
            if mask_zero_diff:
                d_data[np.where(d_data == 0.0)] = FILLVALUE

            var = new_net.create_variable(
                vname, attrs, s_var.dtype, dimensions=s_var.dimensions
            )

            var[:] = d_data
            self.log("processing variable {0}".format(vname))

    def _dt_str(self, dt):
        """for datetime to string for year < 1900"""
        dt_str = "{0:04d}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:02}Z".format(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
        )
        return dt_str

    def write(self):
        """write the nc object to disk"""
        self.log("writing nc file")
        assert (
            self.nc is not None
        ), "netcdf.write() error: nc file not initialized"

        # write any new attributes that have been set since
        # initializing the file
        for k, v in self.global_attributes.items():
            try:
                if self.nc.attributes.get(k) is not None:
                    self.nc.setncattr(k, v)
            except Exception:
                self.logger.warn(
                    "error setting global attribute {0}".format(k)
                )

        self.nc.sync()
        self.nc.close()
        self.log("writing nc file")

    def _initialize_attributes(self):
        """private method to initial the attributes
        of the NetCdf instance
        """
        assert (
            "nc" not in self.__dict__.keys()
        ), "NetCdf._initialize_attributes() error: nc attribute already set"

        self.nc_epsg_str = "epsg:4326"
        self.nc_crs_longname = "http://www.opengis.net/def/crs/EPSG/0/4326"
        self.nc_semi_major = float(6378137.0)
        self.nc_inverse_flat = float(298.257223563)

        self.global_attributes = {}
        self.global_attributes["namefile"] = self.model.namefile
        self.global_attributes["model_ws"] = self.model.model_ws
        self.global_attributes["exe_name"] = self.model.exe_name
        self.global_attributes["modflow_version"] = self.model.version

        self.global_attributes["create_hostname"] = socket.gethostname()
        self.global_attributes["create_platform"] = platform.system()
        self.global_attributes["create_directory"] = os.getcwd()

        htol, rtol = -999, -999
        try:
            htol, rtol = self.model.solver_tols()
        except Exception as e:
            self.logger.warn(
                "unable to get solver tolerances:{0}".format(str(e))
            )
        self.global_attributes["solver_head_tolerance"] = htol
        self.global_attributes["solver_flux_tolerance"] = rtol
        spatial_attribs = {
            "xll": self.model_grid.xoffset,
            "yll": self.model_grid.yoffset,
            "rotation": self.model_grid.angrot,
            "proj4_str": self.model_grid.proj4,
        }
        for n, v in spatial_attribs.items():
            self.global_attributes["flopy_sr_" + n] = v
        self.global_attributes[
            "start_datetime"
        ] = self.model_time.start_datetime

        self.fillvalue = FILLVALUE

        # initialize attributes
        self.grid_crs = None
        self.zs = None
        self.ys = None
        self.xs = None
        self.nc = None

    def initialize_geometry(self):
        """initialize the geometric information
        needed for the netcdf file
        """
        try:
            import pyproj
        except ImportError as e:
            raise ImportError(
                "NetCdf error importing pyproj module:\n" + str(e)
            )
        from distutils.version import LooseVersion

        # Check if using newer pyproj version conventions
        pyproj220 = LooseVersion(pyproj.__version__) >= LooseVersion("2.2.0")

        proj4_str = self.proj4_str
        print("initialize_geometry::proj4_str = {}".format(proj4_str))

        self.log("building grid crs using proj4 string: {}".format(proj4_str))
        if pyproj220:
            self.grid_crs = pyproj.CRS(proj4_str)
        else:
            if proj4_str.lower().startswith("epsg:"):
                proj4_str = "+init=" + proj4_str
            self.grid_crs = pyproj.Proj(proj4_str, preserve_units=True)

        print("initialize_geometry::self.grid_crs = {}".format(self.grid_crs))

        vmin, vmax = self.model_grid.botm.min(), self.model_grid.top.max()
        if self.z_positive == "down":
            vmin, vmax = vmax, vmin
        else:
            self.zs = self.model_grid.xyzcellcenters[2].copy()

        ys = self.model_grid.xyzcellcenters[1].copy()
        xs = self.model_grid.xyzcellcenters[0].copy()

        # Transform to a known CRS
        if pyproj220:
            nc_crs = pyproj.CRS(self.nc_epsg_str)
            self.transformer = pyproj.Transformer.from_crs(
                self.grid_crs, nc_crs, always_xy=True
            )
        else:
            nc_epsg_str = self.nc_epsg_str
            if nc_epsg_str.lower().startswith("epsg:"):
                nc_epsg_str = "+init=" + nc_epsg_str
            nc_crs = pyproj.Proj(nc_epsg_str)
            self.transformer = None

        print("initialize_geometry::nc_crs = {}".format(nc_crs))

        if pyproj220:
            print(
                "transforming coordinates using = {}".format(self.transformer)
            )

        self.log("projecting grid cell center arrays")
        if pyproj220:
            self.xs, self.ys = self.transformer.transform(xs, ys)
        else:
            self.xs, self.ys = pyproj.transform(self.grid_crs, nc_crs, xs, ys)

        # get transformed bounds and record to check against ScienceBase later
        xmin, xmax, ymin, ymax = self.model_grid.extent
        bbox = np.array(
            [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
        )
        if pyproj220:
            x, y = self.transformer.transform(*bbox.transpose())
        else:
            x, y = pyproj.transform(self.grid_crs, nc_crs, *bbox.transpose())
        self.bounds = x.min(), y.min(), x.max(), y.max()
        self.vbounds = vmin, vmax

    def initialize_file(self, time_values=None):
        """
        initialize the netcdf instance, including global attributes,
        dimensions, and grid information

        Parameters
        ----------

            time_values : list of times to use as time dimension
                entries.  If none, then use the times in
                self.model.dis.perlen and self.start_datetime

        """
        if self.nc is not None:
            raise Exception("nc file already initialized")

        if self.grid_crs is None:
            self.log("initializing geometry")
            self.initialize_geometry()
            self.log("initializing geometry")
        try:
            import netCDF4
        except Exception as e:
            self.logger.warn("error importing netCDF module")
            msg = "NetCdf error importing netCDF4 module:\n" + str(e)
            raise Exception(msg)

        # open the file for writing
        try:
            self.nc = netCDF4.Dataset(self.output_filename, "w")
        except Exception as e:
            msg = "error creating netcdf dataset:\n{}".format(str(e))
            raise Exception(msg)

        # write some attributes
        self.log("setting standard attributes")

        self.nc.setncattr(
            "Conventions",
            "CF-1.6, ACDD-1.3, flopy {}".format(flopy.__version__),
        )
        self.nc.setncattr(
            "date_created", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:00Z")
        )
        self.nc.setncattr("geospatial_vertical_positive", str(self.z_positive))
        min_vertical = np.min(self.zs)
        max_vertical = np.max(self.zs)
        self.nc.setncattr("geospatial_vertical_min", min_vertical)
        self.nc.setncattr("geospatial_vertical_max", max_vertical)
        self.nc.setncattr("geospatial_vertical_resolution", "variable")
        self.nc.setncattr("featureType", "Grid")
        for k, v in self.global_attributes.items():
            try:
                self.nc.setncattr(k, v)
            except:
                self.logger.warn(
                    "error setting global attribute {0}".format(k)
                )
        self.global_attributes = {}
        self.log("setting standard attributes")

        # spatial dimensions
        self.log("creating dimensions")
        # time
        if time_values is None:
            time_values = np.cumsum(self.model_time.perlen)
        self.nc.createDimension("time", len(time_values))
        for name, length in zip(self.dimension_names, self.shape):
            self.nc.createDimension(name, length)
        self.log("creating dimensions")

        self.log("setting CRS info")
        # Metadata variables
        crs = self.nc.createVariable("crs", "i4")
        crs.long_name = self.nc_crs_longname
        crs.epsg_code = self.nc_epsg_str
        crs.semi_major_axis = self.nc_semi_major
        crs.inverse_flattening = self.nc_inverse_flat
        self.log("setting CRS info")

        attribs = {
            "units": "{} since {}".format(
                self.time_units, self.start_datetime
            ),
            "standard_name": "time",
            "long_name": NC_LONG_NAMES.get("time", "time"),
            "calendar": "gregorian",
            "_CoordinateAxisType": "Time",
        }
        time = self.create_variable(
            "time", attribs, precision_str="f8", dimensions=("time",)
        )
        self.logger.warn("time_values:{0}".format(str(time_values)))
        time[:] = np.asarray(time_values)

        # Elevation
        attribs = {
            "units": self.model_grid.units,
            "standard_name": "elevation",
            "long_name": NC_LONG_NAMES.get("elevation", "elevation"),
            "axis": "Z",
            "valid_min": min_vertical,
            "valid_max": max_vertical,
            "positive": self.z_positive,
        }
        elev = self.create_variable(
            "elevation",
            attribs,
            precision_str="f8",
            dimensions=self.dimension_names,
        )
        elev[:] = self.zs

        # Longitude
        attribs = {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": NC_LONG_NAMES.get("longitude", "longitude"),
            "axis": "X",
            "_CoordinateAxisType": "Lon",
        }
        lon = self.create_variable(
            "longitude",
            attribs,
            precision_str="f8",
            dimensions=self.dimension_names[1:],
        )
        lon[:] = self.xs
        self.log("creating longitude var")

        # Latitude
        self.log("creating latitude var")
        attribs = {
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": NC_LONG_NAMES.get("latitude", "latitude"),
            "axis": "Y",
            "_CoordinateAxisType": "Lat",
        }
        lat = self.create_variable(
            "latitude",
            attribs,
            precision_str="f8",
            dimensions=self.dimension_names[1:],
        )
        lat[:] = self.ys

        # x
        self.log("creating x var")
        attribs = {
            "units": self.model_grid.units,
            "standard_name": "projection_x_coordinate",
            "long_name": NC_LONG_NAMES.get("x", "x coordinate of projection"),
            "axis": "X",
        }
        x = self.create_variable(
            "x_proj",
            attribs,
            precision_str="f8",
            dimensions=self.dimension_names[1:],
        )
        x[:] = self.model_grid.xyzcellcenters[0]

        # y
        self.log("creating y var")
        attribs = {
            "units": self.model_grid.units,
            "standard_name": "projection_y_coordinate",
            "long_name": NC_LONG_NAMES.get("y", "y coordinate of projection"),
            "axis": "Y",
        }
        y = self.create_variable(
            "y_proj",
            attribs,
            precision_str="f8",
            dimensions=self.dimension_names[1:],
        )
        y[:] = self.model_grid.xyzcellcenters[1]

        # grid mapping variable
        crs = flopy.utils.reference.crs(
            prj=self.model_grid.prj, epsg=self.model_grid.epsg
        )
        attribs = crs.grid_mapping_attribs
        if attribs is not None:
            self.log("creating grid mapping variable")
            self.create_variable(
                attribs["grid_mapping_name"], attribs, precision_str="f8"
            )

        # layer
        self.log("creating layer var")
        attribs = {
            "units": "",
            "standard_name": "layer",
            "long_name": NC_LONG_NAMES.get("layer", "layer"),
            "positive": "down",
            "axis": "Z",
        }
        lay = self.create_variable("layer", attribs, dimensions=("layer",))
        lay[:] = np.arange(0, self.shape[0])
        self.log("creating layer var")

        if self.model_grid.grid_type == "structured":
            # delc
            attribs = {
                "units": self.model_grid.units.strip("s"),
                "long_name": NC_LONG_NAMES.get(
                    "delc", "Model grid cell spacing along a column"
                ),
            }
            delc = self.create_variable("delc", attribs, dimensions=("y",))
            delc[:] = self.model_grid.delc[::-1]
            if self.model_grid.angrot != 0:
                delc.comments = (
                    "This is the row spacing that applied to the UNROTATED grid. "
                    "This grid HAS been rotated before being saved to NetCDF. "
                    "To compute the unrotated grid, use the origin point and this array."
                )

            # delr
            attribs = {
                "units": self.model_grid.units.strip("s"),
                "long_name": NC_LONG_NAMES.get(
                    "delr", "Model grid cell spacing along a row"
                ),
            }
            delr = self.create_variable("delr", attribs, dimensions=("x",))
            delr[:] = self.model_grid.delr[::-1]
            if self.model_grid.angrot != 0:
                delr.comments = (
                    "This is the col spacing that applied to the UNROTATED grid. "
                    "This grid HAS been rotated before being saved to NetCDF. "
                    "To compute the unrotated grid, use the origin point and this array."
                )
        # else:
        # vertices
        # attribs = {"units": self.model_grid.lenuni.strip('s'),
        #           "long_name": NC_LONG_NAMES.get("vertices",
        #                                          "List of vertices used in the model by cell"),
        #           }
        # vertices = self.create_variable('vertices', attribs, dimensions=('ncpl',))
        # vertices[:] = self.model_grid.vertices

        # Workaround for CF/CDM.
        # http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/
        # reference/StandardCoordinateTransforms.html
        # "explicit_field"
        exp = self.nc.createVariable("VerticalTransform", "S1")
        exp.transform_name = "explicit_field"
        exp.existingDataField = "elevation"
        exp._CoordinateTransformType = "vertical"
        exp._CoordinateAxes = "layer"
        return

    def initialize_group(
        self,
        group="timeseries",
        dimensions=("time",),
        attributes=None,
        dimension_data=None,
    ):
        """
        Method to initialize a new group within a netcdf file. This group
        can have independent dimensions from the global dimensions

        Parameters:
        ----------
        name : str
            name of the netcdf group
        dimensions : tuple
            data dimension names for group
        dimension_shape : tuple
            tuple of data dimension lengths
        attributes : dict
            nested dictionary of {dimension : {attributes}} for each netcdf
            group dimension
        dimension_data : dict
            dictionary of {dimension : [data]} for each netcdf group dimension

        """
        if attributes is None:
            attributes = {}

        if dimension_data is None:
            dimension_data = {}

        if self.nc is None:
            self.initialize_file()

        if group in self.nc.groups:
            raise AttributeError("{} group already initialized".format(group))

        self.log("creating netcdf group {}".format(group))
        self.nc.createGroup(group)
        self.log("{} group created".format(group))

        self.log("creating {} group dimensions".format(group))
        for dim in dimensions:
            if dim == "time":
                if "time" not in dimension_data:
                    time_values = np.cumsum(self.model_time.perlen)
                else:
                    time_values = dimension_data["time"]

                self.nc.groups[group].createDimension(dim, len(time_values))

            else:
                if dim not in dimension_data:
                    raise AssertionError(
                        "{} information must be supplied "
                        "to dimension data".format(dim)
                    )
                else:

                    self.nc.groups[group].createDimension(
                        dim, len(dimension_data[dim])
                    )

        self.log("created {} group dimensions".format(group))

        dim_names = tuple([i for i in dimensions if i != "time"])
        for dim in dimensions:
            if dim.lower() == "time":
                if "time" not in attributes:
                    unit_value = "{} since {}".format(
                        self.time_units, self.start_datetime
                    )
                    attribs = {
                        "units": unit_value,
                        "standard_name": "time",
                        "long_name": NC_LONG_NAMES.get("time", "time"),
                        "calendar": "gregorian",
                        "Axis": "Y",
                        "_CoordinateAxisType": "Time",
                    }
                else:
                    attribs = attributes["time"]

                time = self.create_group_variable(
                    group,
                    "time",
                    attribs,
                    precision_str="f8",
                    dimensions=("time",),
                )

                time[:] = np.asarray(time_values)

            elif dim.lower() == "zone":
                if "zone" not in attributes:
                    attribs = {
                        "units": "N/A",
                        "standard_name": "zone",
                        "long_name": "zonebudget zone",
                        "Axis": "X",
                        "_CoordinateAxisType": "Zone",
                    }

                else:
                    attribs = attributes["zone"]

                zone = self.create_group_variable(
                    group,
                    "zone",
                    attribs,
                    precision_str="i4",
                    dimensions=("zone",),
                )
                zone[:] = np.asarray(dimension_data["zone"])

            else:
                attribs = attributes[dim]
                var = self.create_group_variable(
                    group,
                    dim,
                    attribs,
                    precision_str="f8",
                    dimensions=dim_names,
                )
                var[:] = np.asarray(dimension_data[dim])

    @staticmethod
    def normalize_name(name):
        return name.replace(".", "_").replace(" ", "_").replace("-", "_")

    def create_group_variable(
        self, group, name, attributes, precision_str, dimensions=("time",)
    ):
        """
        Create a new group variable in the netcdf object

        Parameters
        ----------
        name : str
            the name of the variable
        attributes : dict
            attributes to add to the new variable
        precision_str : str
            netcdf-compliant string. e.g. f4
        dimensions : tuple
            which dimensions the variable applies to
            default : ("time","layer","x","y")
        group : str
            which netcdf group the variable goes in
            default : None which creates the variable in root

        Returns
        -------
        nc variable

        Raises
        ------
        AssertionError if precision_str not right
        AssertionError if variable name already in netcdf object
        AssertionError if one of more dimensions do not exist

        """
        name = self.normalize_name(name)

        if (
            name in STANDARD_VARS
            and name in self.nc.groups[group].variables.keys()
        ):
            return

        if name in self.nc.groups[group].variables.keys():
            if self.forgive:
                self.logger.warn(
                    "skipping duplicate {} group variable: {}".format(
                        group, name
                    )
                )
                return
            else:
                raise Exception(
                    "duplicate {} group variable name: {}".format(group, name)
                )

        self.log("creating group {} variable: {}".format(group, name))

        if precision_str not in PRECISION_STRS:
            raise AssertionError(
                "netcdf.create_variable() error: precision "
                "string {} not in {}".format(precision_str, PRECISION_STRS)
            )

        if group not in self.nc.groups:
            raise AssertionError(
                "netcdf group `{}` must be created before "
                "variables can be added to it".format(group)
            )

        self.var_attr_dict["{}/{}".format(group, name)] = attributes

        var = self.nc.groups[group].createVariable(
            name,
            precision_str,
            dimensions,
            fill_value=self.fillvalue,
            zlib=True,
        )

        for k, v in attributes.items():
            try:
                var.setncattr(k, v)
            except:
                self.logger.warn(
                    "error setting attribute"
                    "{} for group {} variable {}".format(k, group, name)
                )
        self.log("creating group {} variable: {}".format(group, name))

        return var

    def create_variable(
        self,
        name,
        attributes,
        precision_str="f4",
        dimensions=("time", "layer"),
        group=None,
    ):
        """
        Create a new variable in the netcdf object

        Parameters
        ----------
        name : str
            the name of the variable
        attributes : dict
            attributes to add to the new variable
        precision_str : str
            netcdf-compliant string. e.g. f4
        dimensions : tuple
            which dimensions the variable applies to
            default : ("time","layer","x","y")
        group : str
            which netcdf group the variable goes in
            default : None which creates the variable in root

        Returns
        -------
        nc variable

        Raises
        ------
        AssertionError if precision_str not right
        AssertionError if variable name already in netcdf object
        AssertionError if one of more dimensions do not exist

        """
        # Normalize variable name
        name = self.normalize_name(name)
        # if this is a core var like a dimension...
        # long_name = attributes.pop("long_name",name)
        if name in STANDARD_VARS and name in self.nc.variables.keys():
            return
        if (
            name not in self.var_attr_dict.keys()
            and name in self.nc.variables.keys()
        ):
            if self.forgive:
                self.logger.warn(
                    "skipping duplicate variable: {0}".format(name)
                )
                return
            else:
                raise Exception("duplicate variable name: {0}".format(name))
        if name in self.nc.variables.keys():
            raise Exception("duplicate variable name: {0}".format(name))

        self.log("creating variable: " + str(name))
        assert (
            precision_str in PRECISION_STRS
        ), "netcdf.create_variable() error: precision string {0} not in {1}".format(
            precision_str, PRECISION_STRS
        )

        if self.nc is None:
            self.initialize_file()

        # check that the requested dimension exists and
        # build up the chuck sizes
        # chunks = []
        # for dimension in dimensions:
        #    assert self.nc.dimensions.get(dimension) is not None, \
        #        "netcdf.create_variable() dimension not found:" + dimension
        #    chunk = self.chunks[dimension]
        #    assert chunk is not None, \
        #        "netcdf.create_variable() chunk size of {0} is None in self.chunks". \
        #            format(dimension)
        #    chunks.append(chunk)

        self.var_attr_dict[name] = attributes

        var = self.nc.createVariable(
            name,
            precision_str,
            dimensions,
            fill_value=self.fillvalue,
            zlib=True,
        )  # ,
        # chunksizes=tuple(chunks))
        for k, v in attributes.items():
            try:
                var.setncattr(k, v)
            except:
                self.logger.warn(
                    "error setting attribute"
                    "{0} for variable {1}".format(k, name)
                )
        self.log("creating variable: " + str(name))
        return var

    def add_global_attributes(self, attr_dict):
        """add global attribute to an initialized file

        Parameters
        ----------
        attr_dict : dict(attribute name, attribute value)

        Returns
        -------
        None

        Raises
        ------
        Exception of self.nc is None (initialize_file()
        has not been called)

        """
        if self.nc is None:
            # self.initialize_file()
            mess = (
                "NetCDF.add_global_attributes() should only "
                "be called after the file has been initialized"
            )
            self.logger.warn(mess)
            raise Exception(mess)

        self.log("setting global attributes")
        self.nc.setncatts(attr_dict)
        self.log("setting global attributes")

    def add_sciencebase_metadata(self, id, check=True):
        """Add metadata from ScienceBase using the
        flopy.export.metadata.acdd class.

        Returns
        -------
        metadata : flopy.export.metadata.acdd object
        """
        md = acdd(id, model=self.model)
        if md.sb is not None:
            if check:
                self._check_vs_sciencebase(md)
            # get set of public attributes
            attr = {n for n in dir(md) if "_" not in n[0]}
            # skip some convenience attributes
            skip = {
                "bounds",
                "creator",
                "sb",
                "xmlroot",
                "time_coverage",
                "get_sciencebase_xml_metadata",
                "get_sciencebase_metadata",
            }
            towrite = sorted(list(attr.difference(skip)))
            for k in towrite:
                v = md.__getattribute__(k)
                if v is not None:
                    # convert everything to strings
                    if not isinstance(v, str):
                        if isinstance(v, list):
                            v = ",".join(v)
                        else:
                            v = str(v)
                    self.global_attributes[k] = v
                    self.nc.setncattr(k, v)
            self.write()
            return md

    def _check_vs_sciencebase(self, md):
        """Check that model bounds read from flopy are consistent with those in ScienceBase."""
        xmin, ymin, xmax, ymax = self.bounds
        tol = 1e-5
        assert md.geospatial_lon_min - xmin < tol
        assert md.geospatial_lon_max - xmax < tol
        assert md.geospatial_lat_min - ymin < tol
        assert md.geospatial_lat_max - ymax < tol
        assert md.geospatial_vertical_min - self.vbounds[0] < tol
        assert md.geospatial_vertical_max - self.vbounds[1] < tol

    def get_longnames_from_docstrings(self, outfile="longnames.json"):
        """
        This is experimental.

        Scrape Flopy module docstrings and return docstrings for parameters
        included in the list of variables added to NetCdf object. Create
        a dictionary of longnames keyed by the NetCdf variable names; make each
        longname from the first sentence of the docstring for that parameter.

        One major limitation is that variables from mflists often aren't described
        in the docstrings.
        """

        def startstop(ds):
            """Get just the Parameters section of the docstring."""
            start, stop = 0, -1
            for i, l in enumerate(ds):
                if "Parameters" in l and "----" in ds[i + 1]:
                    start = i + 2
                if l.strip() in ["Attributes", "Methods", "Returns", "Notes"]:
                    stop = i - 1
                    break
                if i >= start and "----" in l:
                    stop = i - 2
                    break
            return start, stop

        def get_entries(ds):
            """Parse docstring entries into dictionary."""
            stuff = {}
            k = None
            for line in ds:
                if (
                    len(line) >= 5
                    and line[:4] == " " * 4
                    and line[4] != " "
                    and ":" in line
                ):
                    k = line.split(":")[0].strip()
                    stuff[k] = ""
                # lines with parameter descriptions
                elif k is not None and len(line) > 10:  # avoid orphans
                    stuff[k] += line.strip() + " "
            return stuff

        # get a list of the flopy classes
        # packages = inspect.getmembers(flopy.modflow, inspect.isclass)
        packages = [(pp.name[0], pp) for pp in self.model.packagelist]
        # get a list of the NetCDF variables
        attr = [v.split("_")[-1] for v in self.nc.variables]

        # parse docstrings to get long names
        longnames = {}
        for pkg in packages:
            # parse the docstring
            obj = pkg[-1]
            ds = obj.__doc__.split("\n")
            start, stop = startstop(ds)
            txt = ds[start:stop]
            if stop - start > 0:
                params = get_entries(txt)
                for k, v in params.items():
                    if k in attr:
                        longnames[k] = v.split(". ")[0]

        # add in any variables that weren't found
        for var in attr:
            if var not in longnames.keys():
                longnames[var] = ""
        with open(outfile, "w") as output:
            json.dump(longnames, output, sort_keys=True, indent=2)
        return longnames
