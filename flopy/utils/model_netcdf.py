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

        try:
            self._dataset = xr.open_dataset(fpth, engine="netcdf4")
            self._set_mapping()
        except Exception as e:
            print(f"Exception: {e}")

    def create(
        self, model_type: str, model_name: str, nc_type: str, fname: str, dis: Grid
    ) -> None:
        self._type = nc_type.lower()
        self._dis = dis
        self._name = fname
        self._tags = {}

        assert self._type == "mesh2d" or self._type == "structured"
        if isinstance(dis, VertexGrid) and self._type != "mesh2d":
            # TODO error
            pass

        try:
            self._dataset = xr.Dataset()
            self._set_global_attrs(model_type, model_name)
            self._set_grid(dis)
        except Exception as e:
            print(f"Exception: {e}")

        # print(self._dataset.info())

    def create_var(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
        raise NotImplementedError("create_var not implemented in base class")

    def write(self, path: str) -> None:
        nc_fpath = Path(path) / self._name
        self._dataset.to_netcdf(nc_fpath, format="NETCDF4", engine="netcdf4")

    def layered(self) -> bool:
        res = False
        if self._gridtype == "LAYERED MESH":
            res = True

        return res

    def data(self, path, layer=None):
        raise NotImplementedError("data not implemented in base class")

    def _set_mapping(self):
        var_d = {}
        if ("modflow6_grid" and "modflow6_model") not in self._dataset.attrs:
            # TODO error
            pass
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
            # TODO error?
            pass

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

    def _create_var(
        self, varname: str, tag: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        try:
            layer = 0
            var_d = {varname: (nc_shape, data)}
            self._dataset = self._dataset.assign(var_d)
            self._dataset[varname].attrs["modflow6_input"] = tag
            if tag not in self._tags:
                self._tags[tag] = {}
            if layer in self._tags[tag]:
                # TODO error?
                pass
            self._tags[tag][layer] = varname
        except Exception as e:
            print(f"Exception: {e}")

    def _create_layered_var(
        self, varname: str, tag: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        try:
            for layer in range(data.shape[0]):
                layer_vname = f"{varname}_l{layer+1}"
                var_d = {layer_vname: (nc_shape, data[layer].flatten())}
                self._dataset = self._dataset.assign(var_d)
                self._dataset[layer_vname].attrs["modflow6_input"] = tag
                self._dataset[layer_vname].attrs["modflow6_layer"] = layer + 1
                if tag not in self._tags:
                    self._tags[tag] = {}
                if layer in self._tags[tag]:
                    # TODO error?
                    pass
                self._tags[tag][layer + 1] = layer_vname
        except Exception as e:
            print(f"Exception: {e}")


class DisNetCDFStructured(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_var(
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
            else:
                # TODO error?
                pass

        self._create_var(varname, tag, data, nc_shape)

    def data(self, path, layer=None):
        if len(self._dataset[self._tags[path][0]].data.shape) == 3:
            return self._dataset[self._tags[path][0]].data[layer - 1]
        else:
            return self._dataset[self._tags[path][0]].data

    def _set_grid(self, dis):
        x = dis.xcellcenters[0]
        y = dis.ycellcenters[:, 0]
        z = list(range(1, dis.nlay + 1))

        x_bnds = []
        for idx, x in enumerate(dis.xyedges[0]):
            if idx + 1 < len(dis.xyedges[0]):
                bnd = []
                bnd.append(dis.xyedges[0][idx])
                bnd.append(dis.xyedges[0][idx + 1])
                x_bnds.append(bnd)

        y_bnds = []
        for idx, y in enumerate(dis.xyedges[1]):
            if idx + 1 < len(dis.xyedges[1]):
                bnd = []
                bnd.append(dis.xyedges[1][idx + 1])
                bnd.append(dis.xyedges[1][idx])
                y_bnds.append(bnd)

        # create coordinate vars
        var_d = {"z": (["z"], z), "y": (["y"], y), "x": (["x"], x)}
        self._dataset = self._dataset.assign(var_d)

        # create bound vars
        var_d = {"x_bnds": (["x", "bnd"], x_bnds), "y_bnds": (["y", "bnd"], y_bnds)}
        self._dataset = self._dataset.assign(var_d)

        # set coordinate variable attributes
        self._dataset["z"].attrs["units"] = "layer"
        self._dataset["z"].attrs["long_name"] = "layer number"
        self._dataset["y"].attrs["units"] = "m"  # TODO in dis?
        self._dataset["y"].attrs["axis"] = "Y"
        self._dataset["y"].attrs["standard_name"] = "projection_y_coordinate"
        self._dataset["y"].attrs["long_name"] = "Northing"
        self._dataset["y"].attrs["grid_mapping"] = "projection"  # TODO
        self._dataset["y"].attrs["bounds"] = "y_bnds"
        self._dataset["x"].attrs["units"] = "m"  # TODO in dis?
        self._dataset["x"].attrs["axis"] = "X"
        self._dataset["x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["x"].attrs["long_name"] = "Easting"
        self._dataset["x"].attrs["grid_mapping"] = "projection"  # TODO
        self._dataset["x"].attrs["bounds"] = "x_bnds"


# TODO: base class ModelNetCDFMesh2d(ModelNetCDFDataset) ?
#       for common routines e.g. data()


class DisNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_var(
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
            self._create_layered_var(varname, tag, data, nc_shape)
        else:
            self._create_var(varname, tag, data.flatten(), nc_shape)

    def data(self, path, layer=None):
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
        print(dis.get_cell_vertices())
        # nmesh_node = dis.nvert
        # nmesh_face = dis.ncpl
        # max_nmesh_face_nodes = 4 ; # assume 4 for dis?
        # nlay = dis.nlay


class DisvNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_var(
        self, varname: str, tag: str, layer: int, data: np.typing.ArrayLike, shape: list
    ):
        pass

    def data(self, path, layer=None):
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
    assert dis_str == "structured" or dis_str == "vertex"

    fpth = Path(nc_fpth).resolve()
    try:
        dataset = xr.open_dataset(fpth, engine="netcdf4")
        if ("modflow6_grid" and "modflow6_model") not in dataset.attrs:
            # TODO error
            pass
        else:
            modelname = dataset.attrs["modflow6_model"].split(":")[0].lower()
            gridtype = dataset.attrs["modflow6_grid"].lower()
            if dis_str == "vertex":
                if gridtype == "layered mesh":
                    nc_dataset = DisvNetCDFMesh2d()
                else:
                    # TODO error?
                    pass
            elif dis_str == "structured":
                if gridtype == "layered mesh":
                    nc_dataset = DisNetCDFMesh2d()
                elif gridtype == "structured":
                    nc_dataset = DisNetCDFStructured()
                else:
                    # TODO error?
                    pass
        dataset.close()
    except Exception as e:
        print(f"Exception: {e}")

    if nc_dataset:
        fpth = Path(nc_fpth).resolve()
        nc_dataset.open(fpth)

    return nc_dataset


def create_netcdf_dataset(model_type, name, nc_type, nc_fname, dis):
    assert (
        model_type.lower() == "gwf6"
        or model_type.lower() == "gwt6"
        or model_type.lower() == "gwe6"
    )
    nc_dataset = None
    if isinstance(dis, VertexGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisvNetCDFMesh2d()
        else:
            # TODO error?
            pass
    elif isinstance(dis, StructuredGrid):
        if nc_type.lower() == "mesh2d":
            nc_dataset = DisNetCDFMesh2d()
        elif nc_type.lower() == "structured":
            nc_dataset = DisNetCDFStructured()
        else:
            # TODO error?
            pass

    if nc_dataset:
        nc_dataset.create(model_type, name, nc_type, nc_fname, dis)

    return nc_dataset
