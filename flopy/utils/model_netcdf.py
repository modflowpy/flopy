import re
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from pyproj import CRS, Proj

import flopy

from ..discretization.grid import Grid
from ..discretization.structuredgrid import StructuredGrid
from ..discretization.unstructuredgrid import UnstructuredGrid
from ..discretization.vertexgrid import VertexGrid

FILLNA_INT32 = np.int32(-2147483647)
FILLNA_DBL = 9.96920996838687e36
lenunits = {0: "m", 1: "ft", 2: "m", 3: "m"}


class ModelNetCDFDataset:
    """
    These objects are intended to support loading, creating and
    updating model input NetCDF files.

    Newly created datasets define coordinate or mesh variables
    corresponding to the type of NetCDF file specified. When
    the discretization crs attribute is valid, a projection
    variable is also added to the dataset.

    Data will be associated with the grid and projection when
    the relevant interfaces, e.g. create_array() and write(),
    are used. Additional interfaces are provided to modify
    and read existing data.

    Additionally, these files can can be used as MODFLOW 6
    model inputs for variables that define internal attributes
    designated for that purpose, specifically "modflow6_input"
    and "modflow6_layer". These attributes are managed internally
    for MODFLOW 6 models when the supported data interfaces are
    used.
    """

    def __init__(self):
        self._modelname = None
        self._modeltype = None
        self._dataset = None
        self._modelgrid = None
        self._tags = None
        self._fname = None
        self._nc_type = None

    @property
    def gridtype(self):
        if self._nc_type == "mesh2d":
            return "LAYERED MESH"
        elif self._nc_type == "structured":
            return "STRUCTURED"

        return ""

    @property
    def modelname(self):
        return self._modelname

    @property
    def dataset(self):
        return self._dataset

    @property
    def nc_type(self):
        return self._nc_type

    @property
    def layered(self):
        res = False
        if self.nc_type == "mesh2d":
            res = True

        return res

    def open(self, nc_fpth: str) -> None:
        """
        Open an existing dataset. Assumes the dataset has been annotated
        with the necessary attributes to read and update, including global
        attributes modflow_model and modflow_grid.

        Args:
            nc_fpth (str): Path to an existing NetCDF file.
        """
        fpth = Path(nc_fpth).resolve()
        self._fname = fpth.name

        self._dataset = xr.open_dataset(fpth, engine="netcdf4")
        self._set_mapping()

        self._dataset.attrs["source"] = f"flopy v{flopy.__version__}"
        history = self._dataset.attrs["history"]
        self._dataset.attrs["history"] = f"{history}; updated {time.ctime(time.time())}"

    def create(
        self, modeltype: str, modelname: str, nc_type: str, fname: str, modelgrid: Grid
    ) -> None:
        """
        Create a new dataset.

        Args:
            modeltype (str): A model type, e.g. GWF6.
            modelname (str): The model name.
            nc_type (str): A supported NetCDF file type: mesh2d or structured.
            fname (str): The generated NetCDF file name.
            modelgrid (Grid): A FloPy derived discretization object.
        """
        self._modelname = modelname.lower()
        self._modeltype = modeltype.lower()
        self._nc_type = nc_type.lower()
        self._modelgrid = modelgrid
        self._fname = fname
        self._tags = {}

        if self._nc_type != "mesh2d" and self._nc_type != "structured":
            raise Exception('Supported NetCDF file types are "mesh2d" and "structured"')
        if isinstance(modelgrid, VertexGrid) and self._nc_type != "mesh2d":
            raise Exception("VertexGrid object must use mesh2d netcdf file type")

        self._dataset = xr.Dataset()
        self._set_global_attrs(modeltype, modelname)
        self._set_grid(modelgrid)
        # print(self._dataset.info())

    def close(self):
        self._dataset.close()
        self.__init__()

    def create_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        shape: list,
        longname: Optional[str],
    ):
        """
        Create a new array. Override this function in a derived class.

        Args:
            package (str): The name of a data grouping in the file, typically
                a package. Must be distinct for each grouping. If this dataset
                is associated with a modflow 6 model and this is a base package
                (dis, disv, npf, ic, etc.), use that name. If this is a stress
                package, use the same package name that is defined in the model
                name file, e.g. chd_0, chd_1 or the user defined name.
            param (str): The parameter name associated with the data. If this
                is a modflow 6 model this should be the same name used in a
                modflow input file for the data, e.g. strt, k, k33, idomain,
                icelltype, etc.
            data (ArrayLike): The data.
            shape (list): The named dimensions of the grid that the data is
                associated with, e.g. nlay, ncol, nrow, ncpl.
            longname (str, None): An optional longname for the parameter that
                the data is associated with.
        """
        raise NotImplementedError("create_array not implemented in base class")

    def set_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        layer: Optional[int],
    ):
        """
        Set data in an existing array. Override this function in a derived class.
        Do not use to update variables that establish the vertices and cell
        structure of the grid, e.g. delr/delc or vertex/cell parameters, only
        model data associated with the grid.

        Args:
            package (str): A package name provided as an argument to create_array().
            param (str): A parameter name provided as an argument to create_array().
            data (ArrayLike): The data.
            layer (int, None): The layer that the data applies to. If the data
                applies to the entire array or grid do not define.
        """
        raise NotImplementedError("set_array not implemented in base class")

    def array(self, package: str, param: str, layer=None):
        """
        Read data in an existing array. Override this function in a derived class.

        Args:
            package (str): A package name provided as an argument to create_array().
            param (str): A parameter name provided as an argument to create_array().
            layer (int, None): The layer that the data applies to.  If the data
                applies to the entire array or grid do not define.

        Returns:
            The data.
        """
        raise NotImplementedError("array not implemented in base class")

    def write(self, path: str, **kwargs) -> None:
        """
        Write the data set to a NetCDF file.

        Args:
            path (str): A directory in which to write the file.
            kwargs (dict): A dictionary of supported encodings to
                apply to managed grid associated arrays.
        """
        nc_fpath = Path(path) / self._fname
        self._set_projection()
        self._set_modflow_attrs()
        encoding = self._set_encoding(**kwargs)
        if encoding is not None:
            self._dataset.to_netcdf(
                nc_fpath, format="NETCDF4", engine="netcdf4", encoding=encoding
            )
        else:
            self._dataset.to_netcdf(nc_fpath, format="NETCDF4", engine="netcdf4")

    def path(self, package: str, param: str, verify=False):
        path = f"{self._modelname.lower()}/{package.lower()}/{param.lower()}"

        if verify and path not in self._tags:
            raise Exception(f"Managed variable path not found in dataset: {path}")

        return path

    def _set_grid(self, dis):
        """
        Define grid or coordinate variables associated with the NetCDF
        file type. Override this function in a derived class.

        Args:
            dis (Grid): A derived FloPy discretization object.
        """
        raise NotImplementedError("_set_grid not implemented in base class")

    def _set_coords(self, crs):
        """
        Define coordinate variables associated with the NetCDF
        file type. Override this function in a derived class.

        Args:
            dis (Grid): A derived FloPy discretization object.
        """
        raise NotImplementedError("_set_coords not implemented in base class")

    def _set_encoding(self, **kwargs):
        if self._modelgrid is None:
            return None
        # encodings: {
        #   'szip_coding',
        #   'shuffle',
        #   'fletcher32',
        #   'quantize_mode',
        #   'least_significant_digit',
        #   'endian',
        #   'szip_pixels_per_block',
        #   '_FillValue',
        #   'compression',
        #   'blosc_shuffle',
        #   'zlib',
        #   'significant_digits',
        #   'complevel',
        #   'dtype',
        #   'contiguous',
        #   'chunksizes'}
        encoding = {}
        encodes = {}

        deflate = kwargs.pop("deflate", None)
        shuffle = kwargs.pop("shuffle", None)
        chunk_time = kwargs.pop("chunk_time", None)
        chunk_face = kwargs.pop("chunk_face", None)
        chunk_x = kwargs.pop("chunk_x", None)
        chunk_y = kwargs.pop("chunk_y", None)
        chunk_z = kwargs.pop("chunk_z", None)

        if deflate:
            encodes["zlib"] = True
            encodes["complevel"] = deflate
        if shuffle:
            encodes["shuffle"] = True

        for path in self._tags:
            for l in self._tags[path]:
                if chunk_face and self._nc_type == "mesh2d":
                    codes = dict(encodes)
                    dims = self._dataset[self._tags[path][l]].dims
                    if "nmesh_face" in dims:
                        codes["chunksizes"] = [chunk_face]
                        encoding[self._tags[path][l]] = codes
                elif self._nc_type == "structured" and chunk_x and chunk_y and chunk_z:
                    codes = dict(encodes)
                    dims = self._dataset[self._tags[path][l]].dims
                    if "x" in dims and "y" in dims:
                        if "z" in dims:
                            codes["chunksizes"] = [chunk_z, chunk_y, chunk_x]
                        else:
                            codes["chunksizes"] = [chunk_y, chunk_x]
                        encoding[self._tags[path][l]] = codes

                else:
                    encoding[self._tags[path][l]] = encodes

        return encoding

    def _set_projection(self):
        if not self._modelgrid:
            return

        crs = None
        wkt = None
        projection = False
        if self._modelgrid.crs:
            try:
                crs = CRS.from_user_input(self._modelgrid.crs)
                wkt = crs.to_wkt()
                projection = True
            except Exception as e:
                warnings.warn(
                    f"Cannot generate CRS object from user input: {e}",
                    UserWarning,
                )

        # update coords based on crs
        coords = self._set_coords(crs)

        # Don't define projection and grid mapping if using
        # geographic coordinates in the structured type
        if self._nc_type == "structured" and coords == "lon lat":
            projection = False

        if projection:
            # add projection variable
            self._dataset = self._dataset.assign({"projection": ([], np.int32(1))})
            if self._nc_type == "structured":
                self._dataset["projection"].attrs["crs_wkt"] = wkt
                self._dataset["x"].attrs["grid_mapping"] = "projection"
                self._dataset["y"].attrs["grid_mapping"] = "projection"
            elif self._nc_type == "mesh2d":
                self._dataset["projection"].attrs["wkt"] = wkt
                self._dataset["mesh_node_x"].attrs["grid_mapping"] = "projection"
                self._dataset["mesh_node_y"].attrs["grid_mapping"] = "projection"
                self._dataset["mesh_face_x"].attrs["grid_mapping"] = "projection"
                self._dataset["mesh_face_y"].attrs["grid_mapping"] = "projection"

        # set grid_mapping and coordinates attributes
        for p in self._tags:
            for l in self._tags[p]:
                dims = self._dataset[self._tags[p][l]].dims
                if (self._nc_type == "structured" and len(dims) > 1) or (
                    self._nc_type == "mesh2d"
                    and ("nmesh_face" in dims or "nmesh_node" in dims)
                ):
                    if projection:
                        self._dataset[self._tags[p][l]].attrs["grid_mapping"] = (
                            "projection"
                        )
                    if coords is not None:
                        self._dataset[self._tags[p][l]].attrs["coordinates"] = coords

    def _set_modflow_attrs(self):
        if self._modeltype.endswith("6"):
            # MODFLOW 6 attributes
            for path in self._tags:
                for l in self._tags[path]:
                    vname = self._tags[path][l]
                    self._dataset[vname].attrs["modflow6_input"] = path.upper()
                    if l > -1:
                        self._dataset[vname].attrs["modflow6_layer"] = np.int32(l + 1)

    def _set_mapping(self):
        var_d = {}
        if ("modflow_grid" and "modflow_model") not in self._dataset.attrs:
            raise Exception("Invalid MODFLOW 6 netcdf dataset")
        else:
            self._modelname = self._dataset.attrs["modflow_model"].split(":")[0].lower()
            mtype_str = self._dataset.attrs["modflow_model"].lower()
            if "modflow 6" in mtype_str:
                mtype = re.findall(r"\((.*?)\)", mtype_str)
                if len(mtype) == 1:
                    self._modeltype = f"{mtype}6"
            gridtype = self._dataset.attrs["modflow_grid"].lower()
            if gridtype == "layered mesh":
                self._nc_type = "mesh2d"
            elif gridtype == "structured":
                self._nc_type = "structured"

        for varname, da in self._dataset.data_vars.items():
            if "modflow6_input" in da.attrs:
                path = da.attrs["modflow6_input"].lower()

                if "modflow6_layer" in da.attrs:
                    layer = da.attrs["modflow6_layer"]
                    # convert indexing to 0-based
                    layer = layer - 1
                else:
                    layer = -1

                if path not in var_d:
                    var_d[path] = {}
                var_d[path][layer] = varname

        self._tags = dict(var_d)

    def _set_global_attrs(self, model_type, model_name):
        if model_type.lower() == "gwf6":
            dep_var = "hydraulic head"
            model = "MODFLOW 6 Groundwater Flow (GWF)"
        elif model_type.lower() == "gwt6":
            dep_var = "concentration"
            model = "MODFLOW 6 Groundwater Transport (GWT)"
        elif model_type.lower() == "gwe6":
            dep_var = "temperature"
            model = "MODFLOW 6 Groundwater Energy (GWE)"
        else:
            dep_var = "model"
            if model_type.endswith("6"):
                mtype = re.sub(r"\d+$", "", model_type.upper())
                model = f"MODFLOW 6 {mtype}"
            else:
                model = model_type.upper()

        if self._nc_type == "structured":
            grid = self._nc_type.upper()
            conventions = "CF-1.11"
        elif self._nc_type == "mesh2d":
            grid = "LAYERED MESH"
            conventions = "CF-1.11 UGRID-1.0"

        self._dataset.attrs["title"] = f"{model_name.upper()} {dep_var} input"
        self._dataset.attrs["source"] = f"flopy v{flopy.__version__}"
        self._dataset.attrs["modflow_grid"] = grid
        self._dataset.attrs["modflow_model"] = f"{model_name.upper()}: {model} model"
        self._dataset.attrs["history"] = "first created " + time.ctime(time.time())
        self._dataset.attrs["Conventions"] = conventions

    def _create_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        nc_shape: list,
        longname: Optional[str],
    ):
        layer = -1
        if data.dtype == np.float64:
            fillna = FILLNA_DBL
        elif data.dtype == np.int32:
            fillna = FILLNA_INT32
        elif data.dtype == np.int64:
            # TODO
            fillna = FILLNA_INT32

        # set name and path
        varname = f"{package.lower()}_{param.lower()}"
        path = self.path(package, param)

        # create variable
        var_d = {varname: (nc_shape, data)}
        self._dataset = self._dataset.assign(var_d)

        # update var attrs
        self._dataset[varname].attrs["_FillValue"] = fillna
        if longname:
            self._dataset[varname].attrs["long_name"] = longname

        # update mapping
        if path not in self._tags:
            self._tags[path] = {}
        if layer in self._tags[path]:
            raise Exception(f"Array variable path already exists: {path}")
        self._tags[path][layer] = varname

    def _create_layered_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        nc_shape: list,
        longname: Optional[str],
    ):
        if data.dtype == np.float64:
            fillna = FILLNA_DBL
        elif data.dtype == np.int32:
            fillna = FILLNA_INT32
        elif data.dtype == np.int64:
            # TODO
            fillna = FILLNA_INT32

        # set basename and path
        basename = f"{package.lower()}_{param.lower()}"
        path = self.path(package, param)
        if path not in self._tags:
            self._tags[path] = {}

        for layer in range(data.shape[0]):
            # set varname
            mf6_layer = layer + 1
            layer_vname = f"{basename}_l{mf6_layer}"

            # create variable
            var_d = {layer_vname: (nc_shape, data[layer].flatten())}
            self._dataset = self._dataset.assign(var_d)

            # update var attrs
            self._dataset[layer_vname].attrs["_FillValue"] = fillna
            if longname:
                self._dataset[layer_vname].attrs["long_name"] = (
                    f"{longname} layer={mf6_layer}"
                )

            # update mapping
            if layer in self._tags[path]:
                raise Exception(
                    f"Array variable path already exists: {path}, layer={layer}"
                )
            self._tags[path][layer] = layer_vname


class DisNetCDFStructured(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        shape: list,
        longname: Optional[str],
    ):
        data = np.array(data)
        nc_shape = None
        if len(data.shape) == 3:
            nc_shape = ["z", "y", "x"]
        elif len(data.shape) == 2:
            nc_shape = ["y", "x"]
        elif len(data.shape) == 1:
            if shape[0].lower() == "nrow":
                nc_shape = ["y"]
            elif shape[0].lower() == "ncol":
                nc_shape = ["x"]

        self._create_array(package, param, data, nc_shape, longname)

    def set_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        layer: Optional[int],
    ):
        data = np.array(data)
        path = self.path(package, param, verify=True)
        vname = self._tags[path][-1]
        if len(self._dataset[vname].values.shape) == 1:
            self._dataset[vname].values = data.flatten()
        else:
            if layer is not None and layer > -1:
                self._dataset[vname].values[layer] = data
            else:
                self._dataset[vname].values = data

    def array(self, package: str, param: str, layer=None):
        path = self.path(package, param, verify=True)
        if len(self._dataset[self._tags[path][-1]].data.shape) == 3:
            if layer > -1:
                return self._dataset[self._tags[path][-1]].data[layer]
            else:
                return self._dataset[self._tags[path][-1]].data
        else:
            return self._dataset[self._tags[path][-1]].data

    def _set_grid(self, dis):
        if dis.angrot != 0.0:
            xoff = 0.0
            yoff = 0.0
        else:
            xoff = dis.xoffset
            yoff = dis.yoffset

        # set coordinate var bounds
        x_bnds = []
        xv = xoff + dis.xyedges[0]
        for idx, val in enumerate(xv):
            if idx + 1 < len(xv):
                bnd = []
                bnd.append(xv[idx])
                bnd.append(xv[idx + 1])
                x_bnds.append(bnd)

        y_bnds = []
        yv = yoff + dis.xyedges[1]
        for idx, val in enumerate(yv):
            if idx + 1 < len(yv):
                bnd = []
                bnd.append(yv[idx + 1])
                bnd.append(yv[idx])
                y_bnds.append(bnd)

        # set coordinate vars
        x = xoff + dis.xycenters[0]
        y = yoff + dis.xycenters[1]
        z = [float(x) for x in range(1, dis.nlay + 1)]

        # create coordinate vars
        var_d = {"z": (["z"], z), "y": (["y"], y), "x": (["x"], x)}
        self._dataset = self._dataset.assign(var_d)

        # create bound vars
        var_d = {"x_bnds": (["x", "bnd"], x_bnds), "y_bnds": (["y", "bnd"], y_bnds)}
        self._dataset = self._dataset.assign(var_d)

        # set coordinate variable attributes
        self._dataset["z"].attrs["units"] = "layer"
        self._dataset["z"].attrs["long_name"] = "layer number"
        self._dataset["y"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["y"].attrs["axis"] = "Y"
        self._dataset["y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["y"].attrs["long_name"] = "Northing"
        self._dataset["y"].attrs["bounds"] = "y_bnds"
        self._dataset["x"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["x"].attrs["axis"] = "X"
        self._dataset["x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["x"].attrs["long_name"] = "Easting"
        self._dataset["x"].attrs["bounds"] = "x_bnds"

    def _set_coords(self, crs):
        if crs is None or self._modelgrid is None:
            return "x y"

        lats = []
        lons = []
        xdim = self._dataset.sizes["x"]
        ydim = self._dataset.sizes["y"]

        try:
            epsg_code = crs.to_epsg(min_confidence=90)
            proj = Proj(
                f"EPSG:{epsg_code}",
            )
        except Exception as e:
            warnings.warn(
                f"Cannot create coordinates from CRS: {e}",
                UserWarning,
            )
            return "x y"

        x_local = []
        y_local = []
        xycenters = self._modelgrid.xycenters
        for y in xycenters[1]:
            for x in xycenters[0]:
                x_local.append(x)
                y_local.append(y)

        x_global, y_global = self._modelgrid.get_coords(x_local, y_local)

        for i, x in enumerate(x_global):
            lon, lat = proj(x, y_global[i], inverse=True)
            lats.append(lat)
            lons.append(lon)

        lats = np.array(lats)
        lons = np.array(lons)

        # create coordinate vars
        var_d = {
            "lat": (["y", "x"], lats.reshape(ydim, xdim)),
            "lon": (["y", "x"], lons.reshape(ydim, xdim)),
        }
        self._dataset = self._dataset.assign(var_d)

        # set coordinate attributes
        self._dataset["lat"].attrs["units"] = "degrees_north"
        self._dataset["lat"].attrs["standard_name"] = "latitude"
        self._dataset["lat"].attrs["long_name"] = "latitude"
        self._dataset["lon"].attrs["units"] = "degrees_east"
        self._dataset["lon"].attrs["standard_name"] = "longitude"
        self._dataset["lon"].attrs["long_name"] = "longitude"

        return "lon lat"


class DisNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        shape: list,
        longname: Optional[str],
    ):
        data = np.array(data)
        nc_shape = None
        if len(data.shape) == 1:
            if shape[0].lower() == "nrow":
                nc_shape = ["y"]
            elif shape[0].lower() == "ncol":
                nc_shape = ["x"]
        else:
            nc_shape = ["nmesh_face"]

        if len(data.shape) == 3:
            self._create_layered_array(package, param, data, nc_shape, longname)
        else:
            self._create_array(package, param, data.flatten(), nc_shape, longname)

    def set_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        layer: Optional[int],
    ):
        data = np.array(data)
        path = self.path(package, param, verify=True)
        if layer is not None and layer in self._tags[path]:
            vname = self._tags[path][layer]
        else:
            vname = self._tags[path][-1]
        self._dataset[vname].values = data.flatten()

    def array(self, package: str, param: str, layer=None):
        path = self.path(package, param, verify=True)
        if path in self._tags:
            if layer is None or layer == -1:
                if layer == -1 and layer in self._tags[path]:
                    return self._dataset[self._tags[path][layer]].data
                else:
                    data = []
                    for l in self._tags[path]:
                        data.append(self._dataset[self._tags[path][l]].data)
                    return np.array(data)
            elif layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, dis):
        # mesh container variable
        self._dataset = self._dataset.assign({"mesh": ([], np.int32(1))})
        self._dataset["mesh"].attrs["cf_role"] = "mesh_topology"
        self._dataset["mesh"].attrs["long_name"] = "2D mesh topology"
        self._dataset["mesh"].attrs["topology_dimension"] = np.int32(2)
        self._dataset["mesh"].attrs["face_dimension"] = "nmesh_face"
        self._dataset["mesh"].attrs["node_coordinates"] = "mesh_node_x mesh_node_y"
        self._dataset["mesh"].attrs["face_coordinates"] = "mesh_face_x mesh_face_y"
        self._dataset["mesh"].attrs["face_node_connectivity"] = "mesh_face_nodes"

        # mesh node x and y
        var_d = {
            "mesh_node_x": (["nmesh_node"], dis.verts[:, 0]),
            "mesh_node_y": (["nmesh_node"], dis.verts[:, 1]),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_node_x"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["mesh_node_x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["mesh_node_x"].attrs["long_name"] = "Easting"
        self._dataset["mesh_node_y"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["mesh_node_y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["mesh_node_y"].attrs["long_name"] = "Northing"

        # mesh face x and y
        x_bnds = []
        x_verts = dis.verts[:, 0].reshape(dis.nrow + 1, dis.ncol + 1)
        for i in range(dis.nrow):
            if i + 1 > dis.nrow:
                break
            for j in range(dis.ncol):
                if j + 1 <= dis.ncol:
                    bnd = []
                    bnd.append(x_verts[i + 1][j])
                    bnd.append(x_verts[i + 1][j + 1])
                    bnd.append(x_verts[i][j + 1])
                    bnd.append(x_verts[i][j])
                    x_bnds.append(bnd)

        y_bnds = []
        y_verts = dis.verts[:, 1].reshape(dis.nrow + 1, dis.ncol + 1)
        for i in range(dis.nrow):
            if i + 1 > dis.nrow:
                break
            for j in range(dis.ncol):
                if j + 1 <= dis.ncol:
                    bnd = []
                    bnd.append(y_verts[i + 1][j])
                    bnd.append(y_verts[i + 1][j + 1])
                    bnd.append(y_verts[i][j + 1])
                    bnd.append(y_verts[i][j])
                    y_bnds.append(bnd)

        var_d = {
            "mesh_face_x": (["nmesh_face"], dis.xcellcenters.flatten()),
            "mesh_face_xbnds": (["nmesh_face", "max_nmesh_face_nodes"], x_bnds),
            "mesh_face_y": (["nmesh_face"], dis.ycellcenters.flatten()),
            "mesh_face_ybnds": (["nmesh_face", "max_nmesh_face_nodes"], y_bnds),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_face_x"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["mesh_face_x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["mesh_face_x"].attrs["long_name"] = "Easting"
        self._dataset["mesh_face_x"].attrs["bounds"] = "mesh_face_xbnds"
        self._dataset["mesh_face_y"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["mesh_face_y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["mesh_face_y"].attrs["long_name"] = "Northing"
        self._dataset["mesh_face_y"].attrs["bounds"] = "mesh_face_ybnds"

        # mesh face nodes
        max_face_nodes = 4
        face_nodes = []
        for r in dis.iverts:
            nodes = [np.int32(x + 1) for x in r]
            nodes.reverse()
            face_nodes.append(nodes)

        var_d = {
            "mesh_face_nodes": (["nmesh_face", "max_nmesh_face_nodes"], face_nodes),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_face_nodes"].attrs["cf_role"] = "face_node_connectivity"
        self._dataset["mesh_face_nodes"].attrs["long_name"] = (
            "Vertices bounding cell (counterclockwise)"
        )
        self._dataset["mesh_face_nodes"].attrs["_FillValue"] = FILLNA_INT32
        self._dataset["mesh_face_nodes"].attrs["start_index"] = np.int32(1)

    def _set_coords(self, crs):
        return "mesh_face_x mesh_face_y"


class DisvNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        shape: list,
        longname: Optional[str],
    ):
        data = np.array(data)
        nc_shape = ["nmesh_face"]

        if len(data.shape) == 2:
            self._create_layered_array(package, param, data, nc_shape, longname)
        else:
            self._create_array(package, param, data.flatten(), nc_shape, longname)

    def set_array(
        self,
        package: str,
        param: str,
        data: np.typing.ArrayLike,
        layer: Optional[int],
    ):
        data = np.array(data)
        path = self.path(package, param, verify=True)
        if layer is not None and layer in self._tags[path]:
            vname = self._tags[path][layer]
        else:
            vname = self._tags[path][-1]
        self._dataset[vname].values = data.flatten()

    def array(self, package: str, param: str, layer=None):
        path = self.path(package, param, verify=True)
        if path in self._tags:
            if layer is None or layer == -1:
                if layer == -1 and layer in self._tags[path]:
                    return self._dataset[self._tags[path][layer]].data
                else:
                    data = []
                    for l in self._tags[path]:
                        data.append(self._dataset[self._tags[path][l]].data)
                    return np.array(data)
            elif layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, disv):
        # mesh container variable
        self._dataset = self._dataset.assign({"mesh": ([], np.int32(1))})
        self._dataset["mesh"].attrs["cf_role"] = "mesh_topology"
        self._dataset["mesh"].attrs["long_name"] = "2D mesh topology"
        self._dataset["mesh"].attrs["topology_dimension"] = np.int32(2)
        self._dataset["mesh"].attrs["face_dimension"] = "nmesh_face"
        self._dataset["mesh"].attrs["node_coordinates"] = "mesh_node_x mesh_node_y"
        self._dataset["mesh"].attrs["face_coordinates"] = "mesh_face_x mesh_face_y"
        self._dataset["mesh"].attrs["face_node_connectivity"] = "mesh_face_nodes"

        # mesh node x and y
        var_d = {
            "mesh_node_x": (["nmesh_node"], disv.verts[:, 0]),
            "mesh_node_y": (["nmesh_node"], disv.verts[:, 1]),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_node_x"].attrs["units"] = lenunits[disv.lenuni]
        self._dataset["mesh_node_x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["mesh_node_x"].attrs["long_name"] = "Easting"
        self._dataset["mesh_node_y"].attrs["units"] = lenunits[disv.lenuni]
        self._dataset["mesh_node_y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["mesh_node_y"].attrs["long_name"] = "Northing"

        # determine max number of cell vertices
        cell_nverts = [cell2d[3] for cell2d in disv.cell2d]
        max_face_nodes = max(cell_nverts)

        # mesh face x and y
        x_bnds = []
        for x in disv.xvertices:
            x = x[::-1]
            if len(x) < max_face_nodes:
                # TODO: set fill value?
                x.extend([FILLNA_INT32] * (max_face_nodes - len(x)))
            x_bnds.append(x)

        y_bnds = []
        for y in disv.yvertices:
            y = y[::-1]
            if len(y) < max_face_nodes:
                # TODO: set fill value?
                y.extend([FILLNA_INT32] * (max_face_nodes - len(y)))
            y_bnds.append(y)

        var_d = {
            "mesh_face_x": (["nmesh_face"], disv.xcellcenters),
            "mesh_face_xbnds": (["nmesh_face", "max_nmesh_face_nodes"], x_bnds),
            "mesh_face_y": (["nmesh_face"], disv.ycellcenters),
            "mesh_face_ybnds": (["nmesh_face", "max_nmesh_face_nodes"], y_bnds),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_face_x"].attrs["units"] = lenunits[disv.lenuni]
        self._dataset["mesh_face_x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["mesh_face_x"].attrs["long_name"] = "Easting"
        self._dataset["mesh_face_x"].attrs["bounds"] = "mesh_face_xbnds"
        self._dataset["mesh_face_y"].attrs["units"] = lenunits[disv.lenuni]
        self._dataset["mesh_face_y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["mesh_face_y"].attrs["long_name"] = "Northing"
        self._dataset["mesh_face_y"].attrs["bounds"] = "mesh_face_ybnds"

        # mesh face nodes
        face_nodes = []
        for idx, r in enumerate(disv.cell2d):
            nodes = disv.cell2d[idx][4 : 4 + r[3]]
            nodes = [np.int32(x + 1) for x in nodes]
            nodes.reverse()
            if len(nodes) < max_face_nodes:
                # TODO set fill value?
                nodes.extend([FILLNA_INT32] * (max_face_nodes - len(nodes)))
            face_nodes.append(nodes)

        var_d = {
            "mesh_face_nodes": (["nmesh_face", "max_nmesh_face_nodes"], face_nodes),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_face_nodes"].attrs["cf_role"] = "face_node_connectivity"
        self._dataset["mesh_face_nodes"].attrs["long_name"] = (
            "Vertices bounding cell (counterclockwise)"
        )
        self._dataset["mesh_face_nodes"].attrs["_FillValue"] = FILLNA_INT32
        self._dataset["mesh_face_nodes"].attrs["start_index"] = np.int32(1)

    def _set_coords(self, crs):
        return "mesh_face_x mesh_face_y"


def open_dataset(nc_fpth: str, grid_type: str) -> ModelNetCDFDataset:
    """
    Open an existing dataset.

    Args:
        nc_fpth (str): The path of the existing NetCDF file.
        grid_type (str): The FloPy discretizaton type corresponding
            to the model associated with the file: vertex or structured.

    Returns:
        ModelNetCDFDataset: A dataset derived from the base class.
    """
    nc_dataset = None
    grid_t = grid_type.lower()

    # grid_type corresponds to a flopy.discretization type
    if grid_t != "vertex" and grid_t != "structured":
        raise Exception(
            "Supported NetCDF discretication types "
            'are "vertex" (DISV) and "structured" '
            "(DIS)"
        )

    fpth = Path(nc_fpth).resolve()
    dataset = xr.open_dataset(fpth, engine="netcdf4")

    if ("modflow_grid" and "modflow_model") not in dataset.attrs:
        modelname = None
        gridtype = None
    else:
        modelname = dataset.attrs["modflow_model"].split(":")[0].lower()
        gridtype = dataset.attrs["modflow_grid"].lower()
        if grid_t == "vertex":
            if gridtype == "layered mesh":
                nc_dataset = DisvNetCDFMesh2d()
        elif grid_t == "structured":
            if gridtype == "layered mesh":
                nc_dataset = DisNetCDFMesh2d()
            elif gridtype == "structured":
                nc_dataset = DisNetCDFStructured()

    dataset.close()

    if nc_dataset:
        nc_dataset.open(fpth)
    else:
        raise Exception(
            f"Unable to load netcdf dataset for file grid "
            f'type "{gridtype}" and discretization grid '
            f'type "{grid_t}"'
        )

    return nc_dataset


def create_dataset(
    modeltype: str, modelname: str, nc_type: str, nc_fname: str, modelgrid: Grid
) -> ModelNetCDFDataset:
    """
    Create a new dataset.

    Args:
        modeltype (str): A model type, e.g. GWF6.
        modelname (str): The model name.
        nc_type (str): A supported NetCDF file type: mesh2d or structured.
        nc_fname (str): The generated NetCDF file name.
        modelgrid (Grid): A FloPy derived discretization object.

    Returns:
        ModelNetCDFDataset: A dataset derived from the base class.
    """
    nc_dataset = None
    if isinstance(modelgrid, VertexGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisvNetCDFMesh2d()
    elif isinstance(modelgrid, StructuredGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisNetCDFMesh2d()
        elif nc_type.lower() == "structured":
            nc_dataset = DisNetCDFStructured()

    if nc_dataset:
        nc_dataset.create(modeltype, modelname, nc_type, nc_fname, modelgrid)
    else:
        raise Exception(
            f"Unable to generate netcdf dataset for file type "
            f'"{nc_type.lower()}" and discretization grid type '
            f'"{modelgrid.grid_type}"'
        )

    return nc_dataset
