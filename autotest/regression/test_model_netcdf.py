import os
import shutil
from pprint import pformat, pprint

import numpy as np
import pytest
import xarray as xr
from modflow_devtools.markers import requires_exe, requires_pkg
from pyproj import CRS

import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridutil import get_disv_kwargs


def check_netcdf(path, mesh=None):
    """Check for functional equivalence"""
    ds = xr.open_dataset(path, engine="netcdf4")
    packages = [
        "dis",
        "npf",
        "ic",
        "sto",
        "ghbg_0",
    ]

    # global attributes
    assert "modflow_grid" in ds.attrs
    assert "modflow_model" in ds.attrs
    if mesh is None:
        assert "mesh" not in ds.attrs
    else:
        assert "mesh" in ds.attrs
    for a in ds.attrs:
        pass

    # coordinates
    for coordname, da in ds.coords.items():
        pass

    # variables
    for varname, da in ds.data_vars.items():
        if mesh is None:
            p = varname.rsplit("_", 1)[0]
        else:
            p = varname.rsplit("_", 2)[0]

        if p in packages:
            assert "modflow_input" in da.attrs
            if mesh is None:
                assert "layer" not in da.attrs
            else:
                lstr = varname.rsplit("_", 1)[1]
                if lstr[0] == "l":
                    assert "layer" in da.attrs


def update_dataset(dataset, pobj):
    nc_info = pobj.netcdf_info()
    for v in nc_info:
        name = nc_info[v]["attrs"]["modflow_input"].rsplit("/", 1)[1].lower()
        d = getattr(pobj, name)
        dataset[nc_info[v]["varname"]].values = d.get_data()


@pytest.mark.regression
def test_uzf01_model_scope_nofile(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "nofile"
    fname = f"{sim_name}.structured.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    assert not (ws / fname).exists()


@pytest.mark.regression
def test_uzf02_model_scope_nofile(function_tmpdir, example_data_path):
    sim_name = "uzf02"
    netcdf = "nofile"
    fname = f"{sim_name}.input.nc"  # default
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    assert not (ws / fname).exists()


@pytest.mark.regression
def test_uzf01_sim_scope_nomesh(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "structured"
    fname = f"{sim_name}.input.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname)


@pytest.mark.regression
def test_uzf01_sim_scope_mesh(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "layered"
    fname = f"{sim_name}.input.nc"  # default
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname, mesh=netcdf)


@pytest.mark.regression
def test_uzf01_sim_scope_fname(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "structured"
    fname = f"{sim_name}.layered.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # update write fname
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname)


@pytest.mark.regression
def test_uzf02_sim_scope(function_tmpdir, example_data_path):
    sim_name = "uzf02"
    netcdf = "layered"
    fname = f"{sim_name}.input.nc"  # default
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname, mesh=netcdf)


@pytest.mark.regression
def test_uzf02_sim_scope_fname(function_tmpdir, example_data_path):
    sim_name = "uzf02"
    netcdf = "layered"
    fname = f"{sim_name}.layered.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # update write fname
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname, mesh=netcdf)


@pytest.mark.regression
def test_uzf01_model_scope_nomesh(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "nofile"
    fname = f"{sim_name}.structured.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    # create dataset
    ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime)
    ds = gwf.update_dataset(ds)

    # write dataset to netcdf
    ds.to_netcdf(ws / fname, format="NETCDF4", engine="netcdf4")

    check_netcdf(ws / fname)

@pytest.mark.regression
def test_uzf01_model_scope_mesh(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "nofile"
    mesh = "layered"
    fname = f"{sim_name}.layered.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    # create dataset
    ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime, mesh=mesh)
    ds = gwf.update_dataset(ds, mesh=mesh)

    # write dataset to netcdf
    ds.to_netcdf(ws / fname, format="NETCDF4", engine="netcdf4")

    check_netcdf(ws / fname, mesh=mesh)


@pytest.mark.regression
def test_uzf02_model_scope(function_tmpdir, example_data_path):
    sim_name = "uzf02"
    netcdf = "nofile"
    mesh = "layered"
    fname = f"{sim_name}.layered.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    # create dataset
    ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime, mesh=mesh)
    ds = gwf.update_dataset(ds, mesh=mesh)

    # write dataset to netcdf
    ds.to_netcdf(ws / fname, format="NETCDF4", engine="netcdf4")

    check_netcdf(ws / fname, mesh=mesh)


@pytest.mark.regression
def test_uzf01_pkg_scope(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    fname = f"{sim_name}.structured.nc"
    netcdf = "structured"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    # create dataset
    ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime)

    # get model netcdf info
    nc_info = gwf.netcdf_info()

    # update dataset directly with required attributes
    for a in nc_info["attrs"]:
        ds.attrs[a] = nc_info["attrs"][a]

    # add all packages and update data
    for p in gwf.packagelist:
        ds = p.update_dataset(ds)
        nc_info = p.netcdf_info()
        for v in nc_info:
            name = nc_info[v]["attrs"]["modflow_input"].rsplit("/", 1)[1].lower()
            d = getattr(p, name)
            if d.repeating:
                for per in d.get_data():
                    istp = sum(gwf.modeltime.nstp[0:per])
                    ds[nc_info[v]["varname"]].values[istp] = d.get_data()[per]
            else:
                ds[nc_info[v]["varname"]].values = d.get_data()

    # write dataset to netcdf
    ds.to_netcdf(ws / fname, format="NETCDF4", engine="netcdf4")

    check_netcdf(ws / fname)


@pytest.mark.regression
def test_uzf01_pkg_scope_modify(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "structured"
    fname = f"{sim_name}.structured.nc"
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    gwf = sim.get_model(sim_name)
    gwf.name_file.nc_filerecord = fname
    sim.write_simulation(netcdf=netcdf)

    # create dataset
    ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime)

    # get model netcdf info
    nc_info = gwf.netcdf_info()

    # update dataset directly with required attributes
    for a in nc_info["attrs"]:
        ds.attrs[a] = nc_info["attrs"][a]

    # update dataset with `DIS` arrays
    dis = gwf.get_package("dis")
    ds = dis.update_dataset(ds)
    update_dataset(ds, dis)

    # get npf package netcdf info
    npf = gwf.get_package("npf")
    nc_info = npf.netcdf_info()

    # update dataset with `NPF` arrays
    # change k varname and add attribute
    nc_info["k"]["varname"] = "npf_k_updated"
    nc_info["k"]["attrs"]["standard_name"] = "soil_hydraulic_conductivity_at_saturation"
    ds = npf.update_dataset(ds, netcdf_info=nc_info)

    # update dataset from npf arrays
    ds["npf_icelltype"].values = npf.icelltype.get_data()
    ds["npf_k_updated"].values = npf.k.get_data()

    # ic
    ic = gwf.get_package("ic")
    ds = ic.update_dataset(ds)
    update_dataset(ds, ic)

    # storage
    sto = gwf.get_package("sto")
    ds = sto.update_dataset(ds)
    update_dataset(ds, sto)

    # update dataset with 'GHBG' arrays
    ghbg = gwf.get_package("ghbg_0")
    ds = ghbg.update_dataset(ds)

    # update bhead netcdf array from flopy perioddata
    # timeseries step index is first of stress period
    for p in ghbg.bhead.get_data():
        istp = sum(gwf.modeltime.nstp[0:p])
        ds["ghbg_0_bhead"].values[istp] = ghbg.bhead.get_data()[p]

    # update cond netcdf array from flopy perioddata
    # timeseries step index is first of stress period
    for p in ghbg.cond.get_data():
        istp = sum(gwf.modeltime.nstp[0:p])
        ds["ghbg_0_cond"].values[istp] = ghbg.cond.get_data()[p]

    # write dataset to netcdf
    ds.to_netcdf(ws / fname, format="NETCDF4", engine="netcdf4")

    check_netcdf(ws / fname)
    assert (
        ds["npf_k_updated"].attrs["standard_name"]
        == "soil_hydraulic_conductivity_at_saturation"
    )


@pytest.mark.regression
def test_uzf01_cycle(function_tmpdir, example_data_path):
    sim_name = "uzf01"
    netcdf = "structured"
    fname = f"{sim_name}.input.nc"  # default
    data_path_base = example_data_path / "mf6" / "netcdf"
    ws = function_tmpdir / sim_name
    base_path = data_path_base / sim_name

    # load example
    sim = flopy.mf6.MFSimulation.load(sim_ws=base_path)

    # set simulation path and write simulation
    sim.set_sim_path(ws)
    sim.write_simulation(netcdf=netcdf)

    check_netcdf(ws / fname)

    # set simulation path and rewrite base simulation
    sim.set_sim_path(ws / "mf6")
    # gwf = sim.get_model(sim_name)
    # gwf.name_file.nc_filerecord = None
    sim.write_simulation()

    assert not (ws / "mf6" / fname).exists()

    success, buff = sim.run_simulation(silent=True, report=True)
    assert success, pformat(buff)
