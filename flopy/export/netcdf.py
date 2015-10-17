import os
import numpy as np
from datetime import datetime

# globals
FILLVALUE = -99999.9
ITMUNI = {0:"undefined",1:"seconds",2:"minutes",3:"hours",4:"days",5:"years"}
LENUNI = {0:"undefined",1:"feet",2:"meters",3:"centimeters"}
PRECISION_STRS = ["f4","f8","i4"]

class NetCdf(object):

    def __init__(self,output_filename,ml):

        assert output_filename.lower().endswith(".nc")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        self.output_filename = output_filename

        assert ml.dis != None
        self.ml = ml
        self.shape = (self.ml.nlay,self.ml.nrow,self.ml.ncol)

        import pandas as pd
        self.start_datetime = pd.to_datetime(self.ml.dis.start_datetime).\
                                  isoformat().split('.')[0].split('+')[0] + "Z"

        self.grid_units = LENUNI[self.ml.dis.lenuni]
        assert self.grid_units in ["feet","meters"],\
            "unsupported length units: " + self.grid_units

        self.time_units = ITMUNI[self.ml.dis.itmuni]

        self._initialize_attributes()

    def write(self):
        assert self.nc is not None,"netcdf.write() error: nc file not initialized"

        # write any new attributes that have been set since initializing the file
        for k,v in self.global_attributes.items():
            try:
                if self.nc.attributes.get(k) is not None:
                    self.nc.setncattr(k,v)
            except Exception as e:
                print"warning - error setting attribute: " + k

        self.nc.sync()
        self.nc.close()


    def _initialize_attributes(self):

        assert "nc" not in self.__dict__.keys(),\
            "NetCdf._initialize_attributes() error: nc attribute already set"

        self.nc_epsg_str = 'epsg:4326'
        self.nc_semi_major = float(6378137.0)
        self.nc_inverse_flat = float(298.257223563)

        self.global_attributes = {}

        self.fillvalue = FILLVALUE

        # initialize attributes
        self.grid_crs = None
        self.zs = None
        self.ys = None
        self.xs = None

        self.chunks = {"time": None}
        self.chunks["x"] = int(self.shape[1] / 4) + 1
        self.chunks["y"] = int(self.shape[2] / 4) + 1
        self.chunks["z"] = self.shape[0]
        self.chunks["layer"] = self.shape[0]

        self.nc = None



    def initialize_geometry(self):

        try:
            from pyproj import Proj, transform
        except Exception as e:
            raise Exception("NetCdf error importing pyproj module:\n" + str(e))

        self.grid_crs = Proj(init=self.ml.dis.sr.epsg_str)

        self.zs = -1.0 * self.ml.dis.zcentroids[:,:,::-1]

        ys = np.flipud(self.ml.dis.sr.ycentergrid)
        xs = np.fliplr(self.ml.dis.sr.xcentergrid)

        if self.grid_units.lower().startswith("f"):
            ys /= 3.281
            xs /= 3.281

        # Transform to a known CRS
        nc_crs = Proj(init=self.nc_epsg_str)

        self.xs, self.ys = transform(self.grid_crs,nc_crs,xs,ys)

    def initialize_file(self,time_values=None):

        if self.nc is not None:
            raise Exception("nc file already initialized")

        if self.grid_crs is None:
            self.initialize_geometry()

        try:
            import netCDF4
        except Exception as e:
            raise Exception("NetCdf error importing netCDF4 module:\n" + str(e))

        # open the file for writing
        self.nc = netCDF4.Dataset(self.output_filename, "w")

        # write some attributes
        self.nc.setncattr("Conventions", "CF-1.6")
        self.nc.setncattr("date_created", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:00Z"))
        self.nc.setncattr("geospatial_vertical_positive",   "up")
        min_vertical = np.min(self.zs)
        max_vertical = np.max(self.zs)
        self.nc.setncattr("geospatial_vertical_min", min_vertical)
        self.nc.setncattr("geospatial_vertical_max", max_vertical)
        self.nc.setncattr("geospatial_vertical_resolution", "variable")
        self.nc.setncattr("featureType", "Grid")
        self.nc.setncattr("origin_x", self.ml.dis.sr.xul)
        self.nc.setncattr("origin_y", self.ml.dis.sr.yul)
        self.nc.setncattr("origin_crs", self.ml.dis.sr.epsg_str)
        self.nc.setncattr("grid_rotation_from_origin", self.ml.dis.sr.rotation)
        for k, v in self.global_attributes.items():
            try:
                self.nc.setself.ncattr(k, v)
            except:
                pass
        self.global_attributes = {}

        # spatial dimensions
        self.nc.createDimension('x', self.shape[2])
        self.nc.createDimension('y', self.shape[1])
        self.nc.createDimension('layer', self.shape[0])

        # Metadata variables
        crs = self.nc.createVariable("crs", "i4")
        crs.long_name = "see http://www.opengis.net for more info"
        crs.epsg_code = self.nc_epsg_str
        crs.semi_major_axis = self.nc_semi_major
        crs.inverse_flattening = self.nc_inverse_flat


        # time
        if time_values is None:
            time_values = np.cumsum(self.ml.dis.perlen)
        self.chunks["time"] = min(len(time_values), 100)
        self.nc.createDimension("time", len(time_values))

        attribs = {"units":"{0} since {1}".format(self.time_units, self.start_datetime),
                   "standard_name": "time", "long_name": "time", "calendar": "gregorian"}
        time = self.create_variable("time",attribs,precision_str="f8",dimensions=("time",))
        time[:] = np.asarray(time_values)

        # Latitude
        attribs = {"units":"degrees_north","standard_name":"latitude",
                   "long_name":"latitude","axis":"Y"}
        lat = self.create_variable("latitude",attribs,dimensions=("x","y"))
        lat[:] = self.ys

        # Longitude
        attribs = {"units":"degrees_east","standard_name":"longitude",
                   "long_name":"longitude","axis":"X"}
        lon = self.create_variable("longitude",attribs,dimensions=("x","y"))
        lon[:] = self.xs

        # Elevation

        attribs = {"units":"meters","standard_name":"elevation",
                   "long_name":"elevation","axis":"Z",
                   "valid_min":min_vertical,"valid_max":max_vertical,
                   "positive":"down"}
        elev = self.create_variable("elevation",attribs,dimensions=("x","y"))
        elev[:] = self.zs

        # layer
        attribs = {"units":"","standard_name":"layer","long_name":"layer",
                   "positive":"down","axis":"Z"}
        lay = self.create_variable("layer",attribs,dimensions=("layer",))
        lay[:] = np.arange(0, self.shape[0])

        # delc
        attribs = {"units":"meters","long_names":"row spacing",
                   "origin_x":self.ml.dis.sr.xul,
                   "origin_y":self.ml.dis.sr.yul,
                   "origin_crs":self.nc_epsg_str}
        delc = self.create_variable('delc', attribs, dimensions=('y',))
        if self.grid_units.lower().startswith('f'):
            delc[:] = self.ml.dis.delc.array[::-1] * 0.3048
        else:
            delc[:] = self.ml.dis.delc.array[::-1]
        if self.ml.dis.sr.rotation != 0:
            delc.comments = "This is the column spacing that applied to the UNROTATED grid. " +\
                            "This grid HAS been rotated before being saved to NetCDF. " +\
                            "To compute the unrotated grid, use the origin point and this array."

        # delr
        attribs = {"units":"meters","long_names":"col spacing",
                   "origin_x":self.ml.dis.sr.xul,
                   "origin_y":self.ml.dis.sr.yul,
                   "origin_crs":self.nc_epsg_str}
        delr = self.create_variable('delr', attribs, dimensions=('x',))
        if self.grid_units.lower().startswith('f'):
            delr[:] = self.ml.dis.delr.array[::-1] * 0.3048
        else:
            delr[:] = self.ml.dis.delr.array[::-1]
        if self.ml.dis.sr.rotation != 0:
            delr.comments = "This is the row spacing that applied to the UNROTATED grid. " +\
                            "This grid HAS been rotated before being saved to NetCDF. " +\
                            "To compute the unrotated grid, use the origin point and this array."

        # Workaround for CF/CDM.
        # http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html
        # "explicit_field"
        exp = self.nc.createVariable('VerticalTransform', 'S1')
        exp.transform_name = "explicit_field"
        exp.existingDataField = "elevation"
        exp._CoordinateTransformType = "vertical"
        exp._CoordinateAxes = "layer"

    def create_variable(self, name, attributes, precision_str='f4',
                        dimensions=("time", "layer", "x", "y")):

        assert precision_str in PRECISION_STRS,\
            "netcdf.create_variable() error: precision string {0} not in {1}".\
                format(precision_str,PRECISION_STRS)

        if self.nc is None:
            self.initialize_file()

        # check that the requested dimension exists and
        # build up the chuck sizes
        chunks = []
        for dimension in dimensions:
            assert self.nc.dimensions.get(dimension) is not None,\
                "netcdf.create_variable() dimension not found:" + dimension
            chunk = self.chunks[dimension]
            assert chunk is not None,\
                "netcdf.create_variable() chunk size of {0} is None in self.chunks".\
                format(dimension)
            chunks.append(chunk)

        # Normalize variable name
        name = name.replace('.', '_').replace(' ', '_').replace('-', '_')
        assert self.nc.variables.get(name) is None,\
            "netcdf.create_variable error: variable already exists:" + name

        var = self.nc.createVariable(name, precision_str, dimensions,
                                     fill_value=self.fillvalue, zlib=True,
                                     chunksizes=tuple(chunks))

        for k, v in attributes.items():
            try:
                var.setncattr(k, v)
            except:
                pass
        return var




