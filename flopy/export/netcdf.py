from __future__ import print_function
import os
import platform
import socket
import copy
import numpy as np
from datetime import datetime

# globals
FILLVALUE = -99999.9
ITMUNI = {0: "undefined", 1: "seconds", 2: "minutes", 3: "hours", 4: "days",
          5: "years"}
LENUNI = {0: "undefined", 1: "feet", 2: "meters", 3: "centimeters"}
PRECISION_STRS = ["f4", "f8", "i4"]


class Logger(object):
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
            self.f = open(filename, 'w', 0)  # unbuffered
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
            s = str(t) + ' finished: ' + str(phrase) + ", took: " + \
                str(t - self.items[phrase]) + '\n'
            if self.echo:
                print(s,)
            if self.filename:
                self.f.write(s)
            self.items.pop(phrase)
        else:
            s = str(t) + ' starting: ' + str(phrase) + '\n'
            if self.echo:
                print(s,)
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
        s = str(datetime.now()) + " WARNING: " + message + '\n'
        if self.echo:
            print(s,)
        if self.filename:
            self.f.write(s)
        return


class NetCdf(object):
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
    verbose : if True, stdout is verbose.  If str, then a log file
        is written to the verbose file

    Notes
    -----
    This class relies heavily on the ModflowDis object,
    including these attributes: lenuni, itmuni, start_datetime, sr
    (SpatialReference).  Make sure these attributes have meaningful values.

    """

    def __init__(self, output_filename, model, time_values=None, verbose=None,
                 logger=None):

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

        assert model.dis is not None
        self.model = model
        self.shape = (self.model.nlay, self.model.nrow, self.model.ncol)

        import dateutil.parser
        self.start_datetime = self._dt_str(dateutil.parser.parse(
            self.model.start_datetime))
        self.logger.warn("start datetime:{0}".format(str(self.start_datetime)))
        self.grid_units = LENUNI[self.model.sr.lenuni]
        assert self.grid_units in ["feet", "meters"], \
            "unsupported length units: " + self.grid_units

        self.time_units = ITMUNI[self.model.dis.itmuni]

        # this gives us confidence that every NetCdf instance
        # has the same attributes
        self.log("initializing attributes")
        self._initialize_attributes()
        self.log("initializing attributes")

        self.log("initializing file")
        self.initialize_file(time_values=time_values)
        self.log("initializing file")

    def difference(self, other, minuend="self", mask_zero_diff=True):
        """make a new NetCDF instance that is the difference with another
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

        assert self.nc is not None,"can't call difference() if nc " +\
                                   "hasn't been populated"
        try:
            import netCDF4
        except Exception as e:
            mess = "error import netCDF4: {0}".format(str(e))
            self.logger.warn(mess)
            raise Exception(mess)

        if isinstance(other,str):
            assert os.path.exists(other),"filename 'other' not found:" + \
                                         "{0}".format(other)
            other = netCDF4.Dataset(other,'r')

        assert isinstance(other,netCDF4.Dataset)

        # check for similar variables
        self_vars = set(self.nc.variables.keys())
        other_vars = set(other.variables)
        diff = self_vars.symmetric_difference(other_vars)
        if len(diff) > 0:
            self.logger.warn("variables are not the same between the two " +\
                             "nc files: " + ','.join(diff))
            return

        # check for similar dimensions
        self_dimens = self.nc.dimensions
        other_dimens = other.dimensions
        for d in self_dimens.keys():
            if d not in other_dimens:
                self.logger.warn("missing dimension in other:{0}".format(d))
                return
            if len(self_dimens[d]) != len(other_dimens[d]):
                self.logger.warn("dimension not consistent: "+\
                                 "{0}:{1}".format(self_dimens[d],
                                                  other_dimens[d]))
                return
        # should be good to go
        time_values = self.nc.variables.get("time")[:]
        new_net = NetCdf(self.output_filename.replace(".nc",".diff.nc"),
                         self.model,time_values=time_values)
        # add the vars to the instance
        for vname in self_vars:
            if vname not in self.var_attr_dict or \
                            new_net.nc.variables.get(vname) is not None:
                self.logger.warn("skipping variable: {0}".format(vname))
                continue
            self.log("processing variable {0}".format(vname))
            s_var = self.nc.variables[vname]
            o_var = other.variables[vname]
            s_data = s_var[:]
            o_data = o_var[:]
            o_mask, s_mask = None, None

            # keep the masks to apply later
            if isinstance(s_data,np.ma.MaskedArray):
                self.logger.warn("masked array for {0}".format(vname))
                s_mask = s_data.mask
                o_mask = o_data.mask
                s_data = np.array(s_data)
                o_data = np.array(o_data)

            # difference with self
            if minuend.lower() == "self":
                d_data = s_data - o_data
            elif minuend.lower() == "other":
                d_data = o_data - s_data
            else:
                mess = "unrecognized minuend {0}".format(minuend)
                self.logger.warn(mess)
                raise Exception(mess)

            # reapply masks
            if s_mask is not None:
                self.log("applying self mask")
                d_data[s_mask] = FILLVALUE
                self.log("applying self mask")
            if o_mask is not None:
                self.log("applying other mask")
                d_data[o_mask] = FILLVALUE
                self.log("applying other mask")
            var = new_net.create_variable(vname,self.var_attr_dict[vname],
                                          s_var.dtype,
                                          dimensions=s_var.dimensions)
            d_data[np.isnan(d_data)] = FILLVALUE
            if mask_zero_diff:
                d_data[np.where(d_data==0.0)] = FILLVALUE
            var[:] = d_data
            self.log("processing variable {0}".format(vname))

    def _dt_str(self,dt):
        """ for datetime to string for year < 1900
        """
        dt_str = '{0:04d}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:02}Z'.format(
                dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
        return dt_str

    def write(self):
        """write the nc object to disk"""
        self.log("writing nc file")
        assert self.nc is not None, \
            "netcdf.write() error: nc file not initialized"

        # write any new attributes that have been set since
        # initializing the file
        for k, v in self.global_attributes.items():
            try:
                if self.nc.attributes.get(k) is not None:
                    self.nc.setncattr(k, v)
            except Exception as e:
                self.logger.warn(
                    'error setting global attribute {0}'.format(k))

        self.nc.sync()
        self.nc.close()
        self.log("writing nc file")

    def _initialize_attributes(self):
        """private method to initial the attributes
           of the NetCdf instance
        """
        assert "nc" not in self.__dict__.keys(), \
            "NetCdf._initialize_attributes() error: nc attribute already set"

        self.nc_epsg_str = 'epsg:4326'
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

        htol,rtol = -999,-999
        try:
            htol,rtol = NetCdf.get_solver_H_R_tols(self.model)
        except Exception as e:
            self.logger.warn("unable to get solver tolerances:" +\
                             "{0}".format(str(e)))
        self.global_attributes["solver_head_tolerance"] = htol
        self.global_attributes["solver_flux_tolerance"] = rtol
        for n,v in self.model.sr.attribute_dict.items():
            self.global_attributes["flopy_sr_"+n] = v
        self.global_attributes["start_datetime"] = self.model.start_datetime

        self.fillvalue = FILLVALUE

        # initialize attributes
        self.grid_crs = None
        self.zs = None
        self.ys = None
        self.xs = None

        self.chunks = {"time": None}
        self.chunks["x"] = int(self.shape[2] / 4) + 1
        self.chunks["y"] = int(self.shape[1] / 4) + 1
        self.chunks["z"] = self.shape[0]
        self.chunks["layer"] = self.shape[0]

        self.nc = None

        self.origin_x = None
        self.origin_y = None

    def initialize_geometry(self):
        """ initialize the geometric information
            needed for the netcdf file
        """
        try:
            from pyproj import Proj, transform
        except Exception as e:
            raise Exception("NetCdf error importing pyproj module:\n" + str(e))

        proj4_str = self.model.sr.proj4_str
        if "epsg" in proj4_str.lower() and "init" not in proj4_str.lower():
            proj4_str = "+init=" + proj4_str
        self.log("building grid crs using proj4 string: {0}".format(proj4_str))
        try:
            self.grid_crs = Proj(proj4_str,preseve_units=True,errcheck=True)

        except Exception as e:
            self.log("error building grid crs:\n{0}".format(str(e)))
            raise Exception("error building grid crs:\n{0}".format(str(e)))
        self.log("building grid crs using proj4 string: {0}".format(proj4_str))

        # self.zs = -1.0 * self.model.dis.zcentroids[:,:,::-1]
        self.zs = -1.0 * self.model.dis.zcentroids

        if self.grid_units.lower().startswith('f'): #and \
               #not self.model.sr.units.startswith("f"):
            self.log("converting feet to meters")
            sr = copy.deepcopy(self.model.sr)
            sr.delr /= 3.281
            sr.delc /= 3.281
            ys = sr.ycentergrid.copy()
            xs = sr.xcentergrid.copy()
            self.log("converting feet to meters")
        else:
            ys = self.model.sr.ycentergrid.copy()
            xs = self.model.sr.xcentergrid.copy()


        # Transform to a known CRS
        nc_crs = Proj(init=self.nc_epsg_str)
        self.log("projecting grid cell center arrays " + \
                 "from {0} to {1}".format(str(self.grid_crs.srs),
                                          str(nc_crs.srs)))
        try:
            self.xs, self.ys = transform(self.grid_crs, nc_crs, xs, ys)
        except Exception as e:
            self.log("error projecting:\n{0}".format(str(e)))
            raise Exception("error projecting:\n{0}".format(str(e)))

        self.log("projecting grid cell center arrays " + \
                 "from {0} to {1}".format(str(self.grid_crs),
                                          str(nc_crs)))

        base_x = self.model.sr.xgrid[0, 0]
        base_y = self.model.sr.ygrid[0, 0]
        self.origin_x, self.origin_y = transform(self.grid_crs, nc_crs, base_x,
                                                 base_y)
        pass

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
            raise Exception(
                "NetCdf error importing netCDF4 module:\n" + str(e))

        # open the file for writing
        try:
            self.nc = netCDF4.Dataset(self.output_filename, "w")
        except Exception as e:
            raise Exception(
                "error creating netcdf dataset:\n{0}".format(str(e)))

        # write some attributes
        self.log("setting standard attributes")
        self.nc.setncattr("Conventions", "CF-1.6")
        self.nc.setncattr("date_created",
                          datetime.utcnow().strftime("%Y-%m-%dT%H:%M:00Z"))
        self.nc.setncattr("geospatial_vertical_positive", "up")
        min_vertical = np.min(self.zs)
        max_vertical = np.max(self.zs)
        self.nc.setncattr("geospatial_vertical_min", min_vertical)
        self.nc.setncattr("geospatial_vertical_max", max_vertical)
        self.nc.setncattr("geospatial_vertical_resolution", "variable")
        self.nc.setncattr("featureType", "Grid")
        self.nc.setncattr("origin_x", self.model.sr.xul)
        self.nc.setncattr("origin_y", self.model.sr.yul)
        self.nc.setncattr("origin_crs", self.model.sr.proj4_str)
        self.nc.setncattr("grid_rotation_from_origin",
                          self.model.sr.rotation)
        for k, v in self.global_attributes.items():
            try:
                self.nc.setncattr(k, v)
            except:
                self.logger.warn(
                    "error setting global attribute {0}".format(k))
        self.global_attributes = {}
        self.log("setting standard attributes")
        # spatial dimensions
        self.log("creating dimensions")
        # time
        if time_values is None:
            time_values = np.cumsum(self.model.dis.perlen)
        self.chunks["time"] = min(len(time_values), 100)
        self.nc.createDimension("time", len(time_values))
        self.nc.createDimension('layer', self.shape[0])
        self.nc.createDimension('y', self.shape[1])
        self.nc.createDimension('x', self.shape[2])
        self.log("creating dimensions")

        self.log("setting CRS info")
        # Metadata variables
        crs = self.nc.createVariable("crs", "i4")
        crs.long_name = self.nc_crs_longname
        crs.epsg_code = self.nc_epsg_str
        crs.semi_major_axis = self.nc_semi_major
        crs.inverse_flattening = self.nc_inverse_flat
        self.log("setting CRS info")

        attribs = {"units": "{0} since {1}".format(self.time_units,
                                                   self.start_datetime),
                   "standard_name": "time", "long_name": "time",
                   "calendar": "gregorian",
                   "_CoordinateAxisType": "Time"}
        time = self.create_variable("time", attribs, precision_str="f8",
                                    dimensions=("time",))
        self.logger.warn("time_values:{0}".format(str(time_values)))
        time[:] = np.asarray(time_values)

        # Elevation
        attribs = {"units": "meters", "standard_name": "elevation",
                   "long_name": "elevation", "axis": "Z",
                   "valid_min": min_vertical, "valid_max": max_vertical,
                   "positive": "down"}
        elev = self.create_variable("elevation", attribs, precision_str="f8",
                                    dimensions=("layer", "y", "x"))
        elev[:] = self.zs

        # Longitude
        attribs = {"units": "degrees_east", "standard_name": "longitude",
                   "long_name": "longitude", "axis": "X",
                   "_CoordinateAxisType": "Lon"}
        lon = self.create_variable("longitude", attribs, precision_str="f8",
                                   dimensions=("y", "x"))
        lon[:] = self.xs
        self.log("creating longitude var")

        # Latitude
        self.log("creating latitude var")
        attribs = {"units": "degrees_north", "standard_name": "latitude",
                   "long_name": "latitude", "axis": "Y",
                   "_CoordinateAxisType": "Lat"}
        lat = self.create_variable("latitude", attribs, precision_str="f8",
                                   dimensions=("y", "x"))
        lat[:] = self.ys

        # layer
        self.log("creating layer var")
        attribs = {"units": "", "standard_name": "layer", "long_name": "layer",
                   "positive": "down", "axis": "Z"}
        lay = self.create_variable("layer", attribs, dimensions=("layer",))
        lay[:] = np.arange(0, self.shape[0])
        self.log("creating layer var")

        # delc
        attribs = {"units": "meters", "long_names": "row spacing",
                   "origin_x": self.model.sr.xul,
                   "origin_y": self.model.sr.yul,
                   "origin_crs": self.nc_epsg_str}
        delc = self.create_variable('delc', attribs, dimensions=('y',))
        if self.grid_units.lower().startswith('f'):
            delc[:] = self.model.sr.delc[::-1] * 0.3048
        else:
            delc[:] = self.model.sr.delc[::-1]
        if self.model.sr.rotation != 0:
            delc.comments = "This is the row spacing that applied to the UNROTATED grid. " + \
                            "This grid HAS been rotated before being saved to NetCDF. " + \
                            "To compute the unrotated grid, use the origin point and this array."
        # delr
        attribs = {"units": "meters", "long_names": "col spacing",
                   "origin_x": self.model.sr.xul,
                   "origin_y": self.model.sr.yul,
                   "origin_crs": self.nc_epsg_str}
        delr = self.create_variable('delr', attribs, dimensions=('x',))
        if self.grid_units.lower().startswith('f'):
            delr[:] = self.model.sr.delr[::-1] * 0.3048
        else:
            delr[:] = self.model.sr.delr[::-1]
        if self.model.sr.rotation != 0:
            delr.comments = "This is the col spacing that applied to the UNROTATED grid. " + \
                            "This grid HAS been rotated before being saved to NetCDF. " + \
                            "To compute the unrotated grid, use the origin point and this array."

        # Workaround for CF/CDM.
        # http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/
        # reference/StandardCoordinateTransforms.html
        # "explicit_field"
        exp = self.nc.createVariable('VerticalTransform', 'S1')
        exp.transform_name = "explicit_field"
        exp.existingDataField = "elevation"
        exp._CoordinateTransformType = "vertical"
        exp._CoordinateAxes = "layer"
        return

    def create_variable(self, name, attributes, precision_str='f4',
                        dimensions=("time", "layer", "y", "x")):
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
        name = name.replace('.', '_').replace(' ', '_').replace('-', '_')
        self.log("creating variable: " + str(name))
        assert precision_str in PRECISION_STRS, \
            "netcdf.create_variable() error: precision string {0} not in {1}". \
                format(precision_str, PRECISION_STRS)

        if self.nc is None:
            self.initialize_file()

        # check that the requested dimension exists and
        # build up the chuck sizes
        chunks = []
        for dimension in dimensions:
            assert self.nc.dimensions.get(dimension) is not None, \
                "netcdf.create_variable() dimension not found:" + dimension
            chunk = self.chunks[dimension]
            assert chunk is not None, \
                "netcdf.create_variable() chunk size of {0} is None in self.chunks". \
                    format(dimension)
            chunks.append(chunk)

        # Normalize variable name
        name = name.replace('.', '_').replace(' ', '_').replace('-', '_')
        #if self.nc.variables.get(name) is not None:
        #    self.logger.warn("skipping duplicate var {0}".format(name))
        #    return
        #while self.nc.variables.get(name) is not None:
        #    name = name + '_1'
        assert self.nc.variables.get(name) is None, \
            "netcdf.create_variable error: variable already exists:" + name

        self.var_attr_dict[name] = attributes

        var = self.nc.createVariable(name, precision_str, dimensions,
                                     fill_value=self.fillvalue, zlib=True)#,
                                     #chunksizes=tuple(chunks))

        for k, v in attributes.items():
            try:
                var.setncattr(k, v)
            except:
                self.logger.warn("error setting attribute" + \
                                 "{0} for variable {1}".format(k, name))
        self.log("creating variable: " + str(name))
        return var


    def add_global_attributes(self,attr_dict):
        """ add global attribute to an initialized file

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
            #self.initialize_file()
            mess = "NetCDF.add_global_attributes() should only "+\
                   "be called after the file has been initialized"
            self.logger.warn(mess)
            raise Exception(mess)

        self.log("setting global attributes")
        self.nc.setncatts(attr_dict)
        self.log("setting global attributes")


    @staticmethod
    def get_solver_H_R_tols(model):
        if model.pcg is not None:
            return model.pcg.hclose,model.pcg.rclose
        elif model.nwt is not None:
            return model.nwt.headtol,model.nwt.fluxtol
        elif model.sip is not None:
            return model.sip.hclose,-999
        elif model.gmg is not None:
            return model.gmg.hclose,model.gmg.rclose