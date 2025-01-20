import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

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
        self._modelname = model_name.lower()
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
        self, package: str, param: str, data: np.typing.ArrayLike, shape: list
    ):
        raise NotImplementedError("create_array not implemented in base class")

    def write(self, path: str) -> None:
        self._set_projection()
        nc_fpath = Path(path) / self._name
        self._dataset.to_netcdf(nc_fpath, format="NETCDF4", engine="netcdf4")

    def layered(self) -> bool:
        res = False
        if self._gridtype == "LAYERED MESH":
            res = True

        return res

    def array(self, path, layer=None):
        raise NotImplementedError("array not implemented in base class")

    def _set_projection(self):
        # TODO remove, testing
        # if self._dis:
        #    self._dis.crs = "EPSG:26918"
        if self._dis and self._dis.crs:
            proj = CRS.from_user_input(self._dis.crs)
            wkt = proj.to_wkt()
            self._dataset = self._dataset.assign({"projection": ([], np.int32(1))})
            if self._type == "structured":
                self._dataset["projection"].attrs["crs_wkt"] = wkt
            elif self._type == "mesh2d":
                self._dataset["projection"].attrs["wkt"] = wkt

            # TODO excluded? delr/delc in mesh?
            for p in self._tags:
                for l in self._tags[p]:
                    self._dataset[self._tags[p][l]].attrs["grid_mapping"] = "projection"

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
        self, package: str, param: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        layer = 0
        varname = f"{package.lower()}_{param.lower()}"
        tag = f"{self._modelname.upper()}/{package.upper()}/{param.upper()}"
        var_d = {varname: (nc_shape, data)}
        self._dataset = self._dataset.assign(var_d)
        self._dataset[varname].attrs["modflow6_input"] = tag.upper()
        if tag not in self._tags:
            self._tags[tag] = {}
        if layer in self._tags[tag]:
            raise Exception(f"Array variable tag already exists: {tag}")
        self._tags[tag][layer] = varname

    def _create_layered_array(
        self, package: str, param: str, data: np.typing.ArrayLike, nc_shape: list
    ):
        varname = f"{package.lower()}_{param.lower()}"
        tag = f"{self._modelname.upper()}/{package.upper()}/{param.upper()}"
        for layer in range(data.shape[0]):
            mf6_layer = layer + 1
            layer_vname = f"{varname}_l{mf6_layer}"
            var_d = {layer_vname: (nc_shape, data[layer].flatten())}
            self._dataset = self._dataset.assign(var_d)
            # self._dataset[layer_vname].attrs["_FillValue"] = 9.96920996838687e+36
            self._dataset[layer_vname].attrs["coordinates"] = "mesh_face_x mesh_face_y"
            self._dataset[layer_vname].attrs["modflow6_input"] = tag.upper()
            self._dataset[layer_vname].attrs["modflow6_layer"] = np.int32(layer + 1)
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
        self, package: str, param: str, data: np.typing.ArrayLike, shape: list
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

        self._create_array(package, param, data, nc_shape)

    def array(self, path, layer=None):
        # TODO update to take package/param instead of path
        if len(self._dataset[self._tags[path][0]].data.shape) == 3:
            return self._dataset[self._tags[path][0]].data[layer - 1]
        else:
            return self._dataset[self._tags[path][0]].data

    def _set_grid(self, dis):
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
        self._dataset["y"].attrs["bounds"] = "y_bnds"
        self._dataset["x"].attrs["units"] = lenunits[dis.lenuni]
        self._dataset["x"].attrs["axis"] = "X"
        self._dataset["x"].attrs["standard_name"] = "projection_x_coordinate"
        self._dataset["x"].attrs["long_name"] = "Easting"
        self._dataset["x"].attrs["bounds"] = "x_bnds"


class DisNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self, package: str, param: str, data: np.typing.ArrayLike, shape: list
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
            self._create_layered_array(package, param, data, nc_shape)
        else:
            self._create_array(package, param, data.flatten(), nc_shape)

    def array(self, path, layer=None):
        if not layer:
            layer = 0
        if path in self._tags:
            if layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, dis):
        lenunits = {0: "m", 1: "feet", 2: "m", 3: "cm"}

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
        # TODO bounds
        var_d = {
            # TODO modflow6 and flopy results differ for mesh_face_x gwf_sto01
            "mesh_face_x": (["nmesh_face"], dis.xcellcenters.flatten()),
            # "mesh_face_xbnds": ([], ),
            "mesh_face_y": (["nmesh_face"], dis.ycellcenters.flatten()),
            # "mesh_face_ybnds": ([], ),
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
            # nodes = list(map(lambda x: np.int32(x + 1), r))
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
        self._dataset["mesh_face_nodes"].attrs["_FillValue"] = np.int32(-2147483647)
        self._dataset["mesh_face_nodes"].attrs["start_index"] = np.int32(1)


class DisvNetCDFMesh2d(ModelNetCDFDataset):
    def __init__(self):
        super().__init__()

    def create_array(
        self, package: str, param: str, data: np.typing.ArrayLike, shape: list
    ):
        nc_shape = ["nmesh_face"]

        if len(data.shape) == 2:
            self._create_layered_array(package, param, data, nc_shape)
        else:
            self._create_array(package, param, data.flatten(), nc_shape)

    def array(self, path, layer=None):
        if not layer:
            layer = 0
        if path in self._tags:
            if layer in self._tags[path]:
                return self._dataset[self._tags[path][layer]].data

        return None

    def _set_grid(self, disv):
        # default metric "m" when undefined
        lenunits = {0: "m", 1: "feet", 2: "m", 3: "cm"}

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

        # mesh face x and y
        # TODO bounds
        var_d = {
            "mesh_face_x": (["nmesh_face"], disv.xcellcenters),
            # "mesh_face_xbnds": ([], ),
            "mesh_face_y": (["nmesh_face"], disv.ycellcenters),
            # "mesh_face_ybnds": ([], ),
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
        cell_nverts = [cell2d[3] for cell2d in disv.cell2d]
        max_face_nodes = max(cell_nverts)
        face_nodes = []
        for idx, r in enumerate(disv.cell2d):
            nodes = disv.cell2d[idx][4 : 4 + r[3]]
            # nodes = list(map(lambda x: np.int32(x + 1), nodes))
            nodes = [np.int32(x + 1) for x in nodes]
            nodes.reverse()
            if len(nodes) < max_face_nodes:
                nodes.extend([np.int32(-2147483647)] * (max_face_nodes - len(nodes)))
            face_nodes.append(nodes)

        var_d = {
            "mesh_face_nodes": (["nmesh_face", "max_nmesh_face_nodes"], face_nodes),
        }
        self._dataset = self._dataset.assign(var_d)
        self._dataset["mesh_face_nodes"].attrs["cf_role"] = "face_node_connectivity"
        self._dataset["mesh_face_nodes"].attrs["long_name"] = (
            "Vertices bounding cell (counterclockwise)"
        )
        self._dataset["mesh_face_nodes"].attrs["_FillValue"] = np.int32(-2147483647)
        self._dataset["mesh_face_nodes"].attrs["start_index"] = np.int32(1)


def open_dataset(nc_fpth: str, dis_type: str) -> ModelNetCDFDataset:
    nc_dataset = None
    dis_str = dis_type.lower()

    # dis_type corresponds to a flopy.discretization type
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


def create_dataset(
    model_type: str, name: str, nc_type: str, nc_fname: str, dis: Grid
) -> ModelNetCDFDataset:
    # TODO: aren't model_type and name in dis?
    if not (
        model_type.lower() == "gwf6"
        or model_type.lower() == "gwt6"
        or model_type.lower() == "gwe6"
    ):
        raise Exception("NetCDF supported model types are GWF, GWT and GWE")

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
        # TODO: remove model_type/name?
        nc_dataset.create(model_type, name, nc_type, nc_fname, dis)
    else:
        raise Exception(
            f"Unable to generate netcdf dataset for file type "
            f'"{nc_type.lower()}" and discretization grid type '
            f'"{dis.grid_type}"'
        )

    return nc_dataset
