import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import flopy

from ..discretization.grid import Grid
from ..discretization.structuredgrid import StructuredGrid
from ..discretization.unstructuredgrid import UnstructuredGrid
from ..discretization.vertexgrid import VertexGrid


class ModelNetCDFDataset:
    """
    MODFLOW 6 Model NetCDF input reader-writer
    """

    def __init__(self):
        self._modelname = None
        self._gridtype = None
        self._dataset = None
        self._dis = None
        self._tags = None
        self._dpth = None
        self._name = None
        self._type = None

    @property
    def gridtype(self):
        return self._gridtype

    @property
    def modelname(self):
        return self._modelname

    @property
    def dataset(self):
        return self._dataset

    def open(self, nc_fpth: str) -> None:
        fpth = Path(nc_fpth).resolve()
        self._dpth = fpth.parent
        self._name = fpth.name

        self._dataset = xr.open_dataset(fpth, engine="netcdf4")
        self._set_mapping()

    def create(
        self, model_type: str, model_name: str, nc_type: str, fname: str, dis: Grid
    ) -> None:
        self._type = nc_type.lower()
        self._dis = dis
        self._name = fname
        self._tags = {}

        if self._type != "mesh2d" and self._type != "structured":
            raise Exception('Supported NetCDF file types are "mesh2d" and "structured"')
        if isinstance(dis, VertexGrid) and self._type != "mesh2d":
            raise Exception("VertexGrid object must use mesh2d netcdf file type")

        self._dataset = xr.Dataset()
        self._set_global_attrs(model_type, model_name)
        self._set_grid(dis)
        # print(self._dataset.info())

    def create_array(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
        raise NotImplementedError("create_array not implemented in base class")

    def write(self, path: str) -> None:
        nc_fpath = Path(path) / self._name
        self._dataset.to_netcdf(nc_fpath, format="NETCDF4", engine="netcdf4")

    def layered(self) -> bool:
        res = False
        if self._gridtype == "LAYERED MESH":
            res = True

        return res

    def array(self, path, layer=None):
        raise NotImplementedError("array not implemented in base class")

    def _set_mapping(self):
        var_d = {}
        if ("modflow6_grid" and "modflow6_model") not in self._dataset.attrs:
            raise Exception("Invalid MODFLOW 6 netcdf dataset")
        else:
            self._modelname = (
                self._dataset.attrs["modflow6_model"].split(":")[0].lower()
            )
            self._gridtype = self._dataset.attrs["modflow6_grid"]
        for varname, da in self._dataset.data_vars.items():
            if "modflow6_input" in da.attrs:
                path = da.attrs["modflow6_input"].lower()

                tokens = path.split("/")
                assert len(tokens) == 3
                assert tokens[0] == self._modelname

                if "modflow6_layer" in da.attrs:
                    layer = da.attrs["modflow6_layer"]
                else:
                    layer = 0

                if path not in var_d:
                    var_d[path] = {}
                var_d[path][layer] = varname

        self._tags = dict(var_d)

    def _set_global_attrs(self, model_type, model_name):
        if model_type.lower() == "gwf6":
            dep_var = "hydraulic head"
            model = "Groundwater Flow (GWF)"
        elif model_type.lower() == "gwt6":
            dep_var = "concentration"
            model = "Groundwater Transport (GWT)"
        elif model_type.lower() == "gwe6":
            dep_var = "temperature"
            model = "Groundwater Energy (GWE)"
        else:
            raise Exception("NetCDF supported for GWF, GWT and GWE models")

        if self._type == "structured":
            grid = self._type.upper()
            conventions = "CF-1.11"
        elif self._type == "mesh2d":
            grid = "LAYERED MESH"
            conventions = "CF-1.11 UGRID-1.0"

        self._dataset.attrs["title"] = f"{model_name.upper()} {dep_var} input"
        self._dataset.attrs["source"] = f"flopy v{flopy.__version__}"
        self._dataset.attrs["modflow6_grid"] = grid
        self._dataset.attrs["modflow6_model"] = (
            f"{model_name.upper()}: MODFLOW 6 {model} model"
        )
        self._dataset.attrs["history"] = "first created " + time.ctime(time.time())
        self._dataset.attrs["Conventions"] = conventions

    def _set_grid(self, dis):
        raise NotImplementedError("_set_grid not implemented in base class")

    def _create_array(
        self, varname: str, tag: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        layer = 0
        var_d = {varname: (nc_shape, data)}
        self._dataset = self._dataset.assign(var_d)
        self._dataset[varname].attrs["modflow6_input"] = tag
        if tag not in self._tags:
            self._tags[tag] = {}
        if layer in self._tags[tag]:
            raise Exception(f"Array variable tag already exists: {tag}")
        self._tags[tag][layer] = varname

    def _create_layered_array(
        self, varname: str, tag: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        for layer in range(data.shape[0]):
            mf6_layer = layer + 1
            layer_vname = f"{varname}_l{mf6_layer}"
            var_d = {layer_vname: (nc_shape, data[layer].flatten())}
            self._dataset = self._dataset.assign(var_d)
            self._dataset[layer_vname].attrs["modflow6_input"] = tag
            self._dataset[layer_vname].attrs["modflow6_layer"] = layer + 1
            if tag not in self._tags:
                self._tags[tag] = {}
            if mf6_layer in self._tags[tag]:
                raise Exception(
                    f"Array variable tag already exists: {tag}, layer={layer}"
                )
            self._tags[tag][mf6_layer] = layer_vname


class DisNetCDFStructured(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
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

        self._create_array(varname, tag, data, nc_shape)

    def array(self, path, layer=None):
        if len(self._dataset[self._tags[path][0]].data.shape) == 3:
            return self._dataset[self._tags[path][0]].data[layer - 1]
        else:
            return self._dataset[self._tags[path][0]].data

    def _set_grid(self, dis):
        # lenunits = {0: "undefined", 1: "feet", 2: "meters", 3: "centimeters"}
        lenunits = {0: "m", 1: "feet", 2: "m", 3: "cm"}

        x_bnds = []
        for idx, val in enumerate(dis.xyedges[0]):
            if idx + 1 < len(dis.xyedges[0]):
                bnd = []
                bnd.append(dis.xyedges[0][idx])
                bnd.append(dis.xyedges[0][idx + 1])
                x_bnds.append(bnd)

        y_bnds = []
        for idx, val in enumerate(dis.xyedges[1]):
            if idx + 1 < len(dis.xyedges[1]):
                bnd = []
                bnd.append(dis.xyedges[1][idx + 1])
                bnd.append(dis.xyedges[1][idx])
                y_bnds.append(bnd)

        x = dis.xcellcenters[0]
        y = dis.ycellcenters[:, 0]
        z = list(range(1, dis.nlay + 1))

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
        # self._dataset["y"].attrs["grid_mapping"] = "projection"  # TODO
        self._dataset["y"].attrs["bounds"] = "y_bnds"
        self._dataset["x"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["x"].attrs["axis"] = "X"
        self._dataset["x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["x"].attrs["long_name"] = "Easting"
        # self._dataset["x"].attrs["grid_mapping"] = "projection"  # TODO
        self._dataset["x"].attrs["bounds"] = "x_bnds"


class DisNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
        nc_shape = None
        if len(data.shape) == 1:
            if shape[0].lower() == "nrow":
                nc_shape = ["y"]
            elif shape[0].lower() == "ncol":
                nc_shape = ["x"]
        else:
            nc_shape = ["nmesh_face"]

        if len(data.shape) == 3:
            self._create_layered_array(varname, tag, data, nc_shape)
        else:
            self._create_array(varname, tag, data.flatten(), nc_shape)

    def array(self, path, layer=None):
        if not layer:
            layer = 0
        if path in self._tags:
            if layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, dis):
        print(dir(dis))
        print(dis.ncpl)
        print(dis.nvert)
        # print(dis.get_cell_vertices())
        # nmesh_node = dis.nvert
        # nmesh_face = dis.ncpl
        # max_nmesh_face_nodes = 4 ; # assume 4 for dis?
        # nlay = dis.nlay


class DisvNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
        nc_shape = ["nmesh_face"]

        if len(data.shape) == 2:
            self._create_layered_array(varname, tag, data, nc_shape)
        else:
            self._create_array(varname, tag, data.flatten(), nc_shape)

    def array(self, path, layer=None):
        if not layer:
            layer = 0
        if path in self._tags:
            if layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, dis):
        pass


def open_netcdf_dataset(nc_fpth: str, dis_type: str) -> ModelNetCDFDataset:
    # dis_type corresponds to a flopy.discretization derived object type
    nc_dataset = None
    dis_str = dis_type.lower()
    if dis_str != "vertex" and dis_str != "structured":
        raise Exception(
            "Supported NetCDF discretication types "
            'are "vertex" (DISV) and "structured" '
            "(DIS)"
        )

    fpth = Path(nc_fpth).resolve()
    dataset = xr.open_dataset(fpth, engine="netcdf4")
    if ("modflow6_grid" and "modflow6_model") not in dataset.attrs:
        modelname = None
        gridtype = None
    else:
        modelname = dataset.attrs["modflow6_model"].split(":")[0].lower()
        gridtype = dataset.attrs["modflow6_grid"].lower()
        if dis_str == "vertex":
            if gridtype == "layered mesh":
                nc_dataset = DisvNetCDFMesh2d()
        elif dis_str == "structured":
            if gridtype == "layered mesh":
                nc_dataset = DisNetCDFMesh2d()
            elif gridtype == "structured":
                nc_dataset = DisNetCDFStructured()

    dataset.close()

    if nc_dataset:
        fpth = Path(nc_fpth).resolve()
        nc_dataset.open(fpth)
    else:
        raise Exception(
            f"Unable to load netcdf dataset for file grid "
            f'type "{gridtype}" and discretization grid '
            f'type "{dis_str}"'
        )

    return nc_dataset


def create_netcdf_dataset(
    model_type, name, nc_type, nc_fname, dis
) -> ModelNetCDFDataset:
    assert (
        model_type.lower() == "gwf6"
        or model_type.lower() == "gwt6"
        or model_type.lower() == "gwe6"
    )
    nc_dataset = None
    if isinstance(dis, VertexGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisvNetCDFMesh2d()
    elif isinstance(dis, StructuredGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisNetCDFMesh2d()
        elif nc_type.lower() == "structured":
            nc_dataset = DisNetCDFStructured()

    if nc_dataset:
        nc_dataset.create(model_type, name, nc_type, nc_fname, dis)
    else:
        raise Exception(
            f"Unable to generate netcdf dataset for file type "
            f'"{nc_type.lower()}" and discretization grid type '
            f'"{dis.grid_type}"'
        )

    return nc_dataset
