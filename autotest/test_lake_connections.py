import os

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.discretization import StructuredGrid, VertexGrid
from flopy.discretization.grid import Grid
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisv,
    ModflowGwfevta,
    ModflowGwfic,
    ModflowGwflak,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrcha,
    ModflowIms,
    ModflowTdis,
)
from flopy.mf6.utils import get_lak_connections
from flopy.modflow import Modflow
from flopy.utils import Raster

pytestmark = pytest.mark.mf6


def export_ascii_grid(modelgrid, file_path, v, nodata=0.0):
    shape = v.shape
    xcenters = modelgrid.xcellcenters[0, :]
    cellsize = xcenters[1] - xcenters[0]
    with open(file_path, "w") as f:
        f.write(f"NCOLS {shape[1]}\n")
        f.write(f"NROWS {shape[0]}\n")
        f.write(f"XLLCENTER {modelgrid.xoffset + 0.5 * cellsize}\n")
        f.write(f"YLLCENTER {modelgrid.yoffset + 0.5 * cellsize}\n")
        f.write(f"CELLSIZE {cellsize}\n")
        f.write(f"NODATA_VALUE {nodata}\n")
        np.savetxt(f, v, fmt="%.4f")


def get_lake_connection_data(nrow, ncol, delr, delc, lakibd, idomain, lakebed_leakance):
    # derived from original modflow6-examples function in ex-gwt-prudic2004t2
    lakeconnectiondata = []
    nlakecon = [0, 0]
    lak_leakance = lakebed_leakance
    for i in range(nrow):
        for j in range(ncol):
            if lakibd[i, j] == 0:
                continue
            else:
                ilak = lakibd[i, j] - 1
                # back
                if i > 0:
                    ci2d, ci = (i - 1, j), (0, i - 1, j)
                    if lakibd[ci2d] == 0 and idomain[ci] > 0:
                        h = [
                            ilak,
                            nlakecon[ilak],
                            ci,
                            "horizontal",
                            lak_leakance,
                            0.0,
                            0.0,
                            0.5 * delc,
                            delr,
                        ]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # left
                if j > 0:
                    ci2d, ci = (i, j - 1), (0, i, j - 1)
                    if lakibd[ci2d] == 0 and idomain[ci] > 0:
                        h = [
                            ilak,
                            nlakecon[ilak],
                            ci,
                            "horizontal",
                            lak_leakance,
                            0.0,
                            0.0,
                            0.5 * delr,
                            delc,
                        ]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # right
                if j < ncol - 1:
                    ci2d, ci = (i, j + 1), (0, i, j + 1)
                    if lakibd[ci2d] == 0 and idomain[ci] > 0:
                        h = [
                            ilak,
                            nlakecon[ilak],
                            ci,
                            "horizontal",
                            lak_leakance,
                            0.0,
                            0.0,
                            0.5 * delr,
                            delc,
                        ]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # front
                if i < nrow - 1:
                    ci2d, ci = (i + 1, j), (0, i + 1, j)
                    if lakibd[ci2d] == 0 and idomain[ci] > 0:
                        h = [
                            ilak,
                            nlakecon[ilak],
                            ci,
                            "horizontal",
                            lak_leakance,
                            0.0,
                            0.0,
                            0.5 * delc,
                            delr,
                        ]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # vertical
                v = [
                    ilak,
                    nlakecon[ilak],
                    (1, i, j),
                    "vertical",
                    lak_leakance,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                nlakecon[ilak] += 1
                lakeconnectiondata.append(v)
    return lakeconnectiondata, nlakecon


@requires_exe("mf6")
def test_base_run(function_tmpdir, example_data_path):
    mpath = example_data_path / "mf6-freyberg"
    sim = MFSimulation().load(
        sim_name="freyberg",
        sim_ws=mpath,
        exe_name="mf6",
        verbosity_level=0,
    )
    sim.set_sim_path(function_tmpdir)

    # remove the well package
    gwf = sim.get_model("freyberg")
    gwf.remove_package("wel_0")

    # write the simulation files and run the model
    sim.write_simulation()
    sim.run_simulation()

    # export bottom, water levels, and k11 as ascii raster files
    # for interpolation in test_lake()
    bot = gwf.dis.botm.array.squeeze()
    export_ascii_grid(gwf.modelgrid, function_tmpdir / "bot.asc", bot)
    top = gwf.output.head().get_data().squeeze() + 2.0
    top = np.where(gwf.dis.idomain.array.squeeze() < 1.0, 0.0, top)
    export_ascii_grid(gwf.modelgrid, function_tmpdir / "top.asc", top)
    k11 = gwf.npf.k.array.squeeze()
    export_ascii_grid(gwf.modelgrid, function_tmpdir / "k11.asc", k11)


@requires_exe("mf6")
@requires_pkg("rasterio", "rasterstats")
def test_lake(function_tmpdir, example_data_path):
    mpath = example_data_path / "mf6-freyberg"
    top = Raster.load(mpath / "top.asc")
    bot = Raster.load(mpath / "bot.asc")
    k11 = Raster.load(mpath / "k11.asc")

    sim = MFSimulation().load(
        sim_name="freyberg",
        sim_ws=mpath,
        exe_name="mf6",
        verbosity_level=0,
    )

    # change the workspace
    sim.set_sim_path(function_tmpdir)

    # get groundwater flow model
    gwf = sim.get_model("freyberg")

    # define extent of lake
    lakes = gwf.dis.idomain.array.squeeze() * -1
    lakes[32:, :] = -1

    # fill bottom
    bot_tm = bot.resample_to_grid(
        gwf.modelgrid,
        band=bot.bands[0],
        method="linear",
        extrapolate_edges=True,
    )
    # mm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid)
    # mm.plot_array(bot_tm)

    # determine a reasonable lake bottom
    idx = np.asarray(lakes > -1).nonzero()
    lak_bot = bot_tm[idx].max() + 2.0

    # interpolate top elevations
    top_tm = top.resample_to_grid(
        gwf.modelgrid,
        band=top.bands[0],
        method="linear",
        extrapolate_edges=True,
    )

    # set the elevation to the lake bottom in the area of the lake
    top_tm[idx] = lak_bot

    # mm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid)
    # v = mm.plot_array(top_tm)
    # cs = mm.contour_array(
    #     top_tm, colors="white", linewidths=0.5, levels=np.arange(0, 25, 2)
    # )
    # plt.clabel(cs, fmt="%.1f", colors="white", fontsize=7)
    # plt.colorbar(v, shrink=0.5)

    gwf.dis.top = top_tm
    gwf.dis.botm = bot_tm.reshape(gwf.modelgrid.shape)

    k11_tm = k11.resample_to_grid(
        gwf.modelgrid,
        band=k11.bands[0],
        method="linear",
        extrapolate_edges=True,
    )
    gwf.npf.k = k11_tm

    # mm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid)
    # mm.plot_array(k11_tm)

    idomain, pakdata_dict, connectiondata = get_lak_connections(
        gwf.modelgrid,
        lakes,
        bedleak=5e-9,
    )

    assert pakdata_dict[0] == 54, (
        f"number of lake connections ({pakdata_dict[0]}) not equal to 54."
    )

    assert len(connectiondata) == 54, (
        "number of lake connectiondata entries ({}) not equal to 54.".format(
            len(connectiondata)
        )
    )

    lak_pak_data = []
    for key, value in pakdata_dict.items():
        lak_pak_data.append([key, 35.0, value])
    lak_spd = {0: [[0, "rainfall", 3.2e-9]]}
    lak = ModflowGwflak(
        gwf,
        print_stage=True,
        nlakes=1,
        packagedata=lak_pak_data,
        connectiondata=connectiondata,
        perioddata=lak_spd,
        pname="LAK-1",
        filename="freyberg.lak",
    )

    idomain = gwf.dis.idomain.array
    lakes.shape = idomain.shape
    gwf.dis.idomain = np.where(lakes > -1, 1, idomain)

    # convert to Newton-Raphson formulation and update the linear accelerator
    gwf.name_file.newtonoptions = "NEWTON UNDER_RELAXATION"
    sim.ims.linear_acceleration = "BICGSTAB"

    # write the revised simulation files and run the model
    sim.write_simulation()
    success = sim.run_simulation(silent=False)

    assert success, f"could not run {sim.name} with lake"


@requires_exe("mf6")
def test_embedded_lak_ex01(function_tmpdir, example_data_path):
    nper = 1
    nlay, nrow, ncol = 5, 17, 17
    shape3d = (nlay, nrow, ncol)
    delr = (
        250.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        250.0,
    )
    delc = delr
    top = 500.0
    botm = (107.0, 97.0, 87.0, 77.0, 67.0)
    lake_map = np.ones(shape3d, dtype=np.int32) * -1
    lake_map[0, 6:11, 6:11] = 0
    lake_map[1, 7:10, 7:10] = 0
    lake_map = np.ma.masked_where(lake_map < 0, lake_map)

    strt = 115.0

    k11 = 30
    k33 = (1179.0, 30.0, 30.0, 30.0, 30.0)

    mpath = example_data_path / "mf2005_test"
    ml = Modflow.load(
        "l1a2k.nam",
        model_ws=mpath,
        load_only=["EVT"],
        check=False,
    )
    rch_rate = 0.116e-1
    evt_rate = 0.141e-1
    evt_depth = 15.0
    evt_surf = ml.evt.surf[0].array

    chd_top_bottom = (
        160.0,
        158.85,
        157.31,
        155.77,
        154.23,
        152.69,
        151.54,
        150.77,
        150.0,
        149.23,
        148.46,
        147.31,
        145.77,
        144.23,
        142.69,
        141.15,
        140.0,
    )
    chd_spd = []
    for k in range(nlay):
        for i in range(nrow):
            if 0 < i < nrow - 1:
                chd_spd.append([k, i, 0, chd_top_bottom[0]])
                chd_spd.append([k, i, ncol - 1, chd_top_bottom[-1]])
            else:
                for jdx, v in enumerate(chd_top_bottom):
                    chd_spd.append([k, i, jdx, v])
    chd_spd = {0: chd_spd}

    name = "lak_ex01"
    sim = MFSimulation(
        sim_name=name,
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    tdis = ModflowTdis(
        sim,
        nper=nper,
    )
    ims = ModflowIms(
        sim,
        print_option="summary",
        linear_acceleration="BICGSTAB",
        outer_maximum=1000,
        inner_maximum=100,
        outer_dvclose=1e-8,
        inner_dvclose=1e-9,
    )
    gwf = ModflowGwf(
        sim,
        modelname=name,
        newtonoptions="newton under_relaxation",
        print_input=True,
    )
    dis = ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    ic = ModflowGwfic(
        gwf,
        strt=strt,
    )
    npf = ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=k11,
        k33=k33,
    )
    chd = ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
    )
    rch = ModflowGwfrcha(
        gwf,
        recharge=rch_rate,
    )
    evt = ModflowGwfevta(
        gwf,
        surface=evt_surf,
        depth=evt_depth,
        rate=evt_rate,
    )
    oc = ModflowGwfoc(
        gwf,
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    idomain, pakdata_dict, connectiondata = get_lak_connections(
        gwf.modelgrid,
        lake_map,
        bedleak=0.1,
    )

    assert pakdata_dict[0] == 57, (
        f"number of lake connections ({pakdata_dict[0]}) not equal to 57."
    )

    assert len(connectiondata) == 57, (
        "number of lake connectiondata entries ({}) not equal to 57.".format(
            len(connectiondata)
        )
    )

    lak_pak_data = []
    for key, value in pakdata_dict.items():
        lak_pak_data.append([key, 110.0, value])
    lak_spd = {
        0: [
            [0, "rainfall", rch_rate],
            [0, "evaporation", 0.0103],
        ]
    }
    lak = ModflowGwflak(
        gwf,
        print_stage=True,
        print_flows=True,
        nlakes=1,
        packagedata=lak_pak_data,
        connectiondata=connectiondata,
        perioddata=lak_spd,
        pname="LAK-1",
    )

    # reset idomain
    gwf.dis.idomain = idomain

    # write the simulation files and run the model
    sim.write_simulation()
    success = sim.run_simulation(silent=False)

    assert success, f"could not run {sim.name}"


@requires_exe("mf6")
def test_embedded_lak_prudic(example_data_path):
    lakebed_leakance = 1.0  # Lakebed leakance ($ft^{-1}$)
    nlay = 8  # Number of layers
    nrow = 36  # Number of rows
    ncol = 23  # Number of columns
    delr = 405.665  # Column width ($ft$)
    delc = 403.717  # Row width ($ft$)
    delv = 15.0  # Layer thickness ($ft$)
    top = 100.0  # Top of the model ($ft$)

    shape2d = (nrow, ncol)
    shape3d = (nlay, nrow, ncol)

    # load data from text files
    data_ws = example_data_path / "mf6_test"
    fname = data_ws / "prudic2004t2_bot1.dat"
    bot0 = np.loadtxt(fname)
    botm = np.array(
        [bot0]
        + [np.ones(shape2d, dtype=float) * (bot0 - (delv * k)) for k in range(1, nlay)]
    )
    fname = data_ws / "prudic2004t2_idomain1.dat"
    idomain0 = np.loadtxt(fname, dtype=np.int32)
    idomain = np.array(nlay * [idomain0], dtype=np.int32)
    fname = data_ws / "prudic2004t2_lakibd.dat"
    lakibd = np.loadtxt(fname, dtype=int)
    lake_map = np.ones(shape3d, dtype=np.int32) * -1
    lake_map[0, :, :] = lakibd[:, :] - 1

    # build StructuredGrid
    model_grid = StructuredGrid(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=np.ones(ncol, dtype=float) * delr,
        delc=np.ones(nrow, dtype=float) * delc,
        top=np.ones(shape2d, dtype=float) * top,
        botm=botm,
        idomain=idomain,
    )

    # base case
    cdata, lakconn = get_lake_connection_data(
        nrow, ncol, delr, delc, lakibd, idomain, lakebed_leakance
    )

    # flopy test
    idomain_rev, pakdata_dict, connectiondata = get_lak_connections(
        model_grid,
        lake_map,
        idomain=idomain,
        bedleak=lakebed_leakance,
    )

    # evaluate the number of connections
    for idx, nconn in enumerate(lakconn):
        assert pakdata_dict[idx] == nconn, (
            "number of connections calculated by get_lak_connections ({}) "
            "not equal to {} for lake {}.".format(pakdata_dict[idx], nconn, idx + 1)
        )

    # compare connectiondata
    for idx, (cd, cdbase) in enumerate(zip(connectiondata, cdata)):
        for jdx in (0, 1, 2, 3, 7, 8):
            match = True
            if jdx not in {7, 8}:
                if cd[jdx] != cdbase[jdx]:
                    match = False
            else:
                match = np.allclose(cd[jdx], cdbase[jdx])
            if not match:
                print(f"connection data do match for connection {idx} for lake {cd[0]}")
                break
        assert match, f"connection data do not match for connection {jdx}"

    # evaluate the revised idomain, only layer 1 has been adjusted
    idomain0_test = idomain[0, :, :].copy()
    idomain0_test[lakibd > 0] = 0
    idomain_test = idomain.copy()
    idomain[0, :, :] = idomain0_test
    assert np.array_equal(idomain_rev, idomain_test), (
        "idomain not updated correctly with lakibd"
    )


@requires_exe("mf6")
def test_embedded_lak_prudic_mixed(example_data_path):
    lakebed_leakance = 1.0  # Lakebed leakance ($ft^{-1}$)
    nlay = 8  # Number of layers
    nrow = 36  # Number of rows
    ncol = 23  # Number of columns
    delr = 405.665  # Column width ($ft$)
    delc = 403.717  # Row width ($ft$)
    delv = 15.0  # Layer thickness ($ft$)
    top = 100.0  # Top of the model ($ft$)

    shape2d = (nrow, ncol)
    shape3d = (nlay, nrow, ncol)

    # load data from text files
    data_ws = example_data_path / "mf6_test"
    fname = data_ws / "prudic2004t2_bot1.dat"
    bot0 = np.loadtxt(fname)
    botm = np.array(
        [bot0]
        + [np.ones(shape2d, dtype=float) * (bot0 - (delv * k)) for k in range(1, nlay)]
    )
    fname = data_ws / "prudic2004t2_idomain1.dat"
    idomain0 = np.loadtxt(fname, dtype=np.int32)
    idomain = np.array(nlay * [idomain0], dtype=np.int32)
    fname = data_ws / "prudic2004t2_lakibd.dat"
    lakibd = np.loadtxt(fname, dtype=int)
    lake_map = np.ones(shape3d, dtype=np.int32) * -1
    lake_map[0, :, :] = lakibd[:, :] - 1

    lakebed_leakance = np.zeros(shape2d, dtype=object)
    idx = np.asarray(lake_map[0, :, :] == 0).nonzero()
    lakebed_leakance[idx] = "none"
    idx = np.asarray(lake_map[0, :, :] == 1).nonzero()
    lakebed_leakance[idx] = 1.0
    lakebed_leakance = lakebed_leakance.tolist()

    # build StructuredGrid
    model_grid = StructuredGrid(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=np.ones(ncol, dtype=float) * delr,
        delc=np.ones(nrow, dtype=float) * delc,
        top=np.ones(shape2d, dtype=float) * top,
        botm=botm,
        idomain=idomain,
    )

    # test mixed lakebed leakance list
    _, _, connectiondata = get_lak_connections(
        model_grid,
        lake_map,
        idomain=idomain,
        bedleak=lakebed_leakance,
    )

    # test the connections
    for data in connectiondata:
        lakeno, bedleak = data[0], data[4]
        if lakeno == 0:
            assert bedleak == "none", f"bedleak for lake 0 is not 'none' ({bedleak})"
        else:
            assert bedleak == 1.0, f"bedleak for lake 1 is not 1.0 ({bedleak})"


def build_simple_disv_grid(nlay=1, return_data=False):
    vertices = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 2.0, 0.0),
        (3, 3.0, 0.0),
        (4, 0.0, 1.0),
        (5, 1.0, 1.0),
        (6, 2.0, 1.0),
        (7, 3.0, 1.0),
        (8, 0.0, 2.0),
        (9, 1.0, 2.0),
        (10, 2.0, 2.0),
        (11, 3.0, 2.0),
        (12, 1.0, 3.0),
        (13, 2.0, 3.0),
    ]

    cell2d = [
        (0, 1.5, 1.5, 4, 5, 6, 10, 9),
        (1, 1.5, 2.5, 4, 9, 10, 13, 12),
        (2, 0.5, 1.5, 4, 4, 5, 9, 8),
        (3, 2.5, 1.5, 4, 6, 7, 11, 10),
        (4, 1.5, 0.5, 4, 1, 2, 6, 5),
    ]

    ncpl = len(cell2d)

    top = np.full(ncpl, 1.0)
    botm = np.vstack([np.full(ncpl, -(k + 1), dtype=float) for k in range(nlay)])

    idomain = np.ones((nlay, ncpl), dtype=int)

    grid = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=top,
        botm=botm,
        idomain=idomain,
        nlay=nlay,
    )

    if return_data:
        return grid, vertices, cell2d

    return grid


def build_asymmetric_disv_grid():
    vertices = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 4.0, 0.0),
        (3, 0.0, 1.0),
        (4, 1.0, 1.0),
        (5, 4.0, 1.0),
    ]
    cell2d = [
        (0, 0.5, 0.5, 4, 0, 1, 4, 3),
        (1, 2.5, 0.5, 4, 1, 2, 5, 4),
    ]

    return VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(2),
        botm=-np.ones((1, 2)),
        idomain=np.ones((1, 2), dtype=int),
        nlay=1,
    )


def test_disv_horizontal_connections():
    modelgrid = build_simple_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    idomain = np.ones((1, modelgrid.ncpl), dtype=int)

    _, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        idomain=idomain,
        bedleak=1.0,
    )

    assert pakdata[0] == 4

    expected = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    }

    returned = {c[2] for c in connectiondata}

    assert returned == expected

    assert all(c[3] == "horizontal" for c in connectiondata)


def test_disv_connection_widths():
    modelgrid = build_simple_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    _, _, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        bedleak=1.0,
    )

    horizontal = [c for c in connectiondata if c[3] == "horizontal"]

    assert len(horizontal) == 4

    widths = sorted(conn[8] for conn in horizontal)
    assert widths == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_disv_connection_lengths():
    modelgrid = build_simple_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    _, _, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        bedleak=1.0,
    )

    horizontal = [c for c in connectiondata if c[3] == "horizontal"]

    assert len(horizontal) == 4

    lengths = sorted(conn[7] for conn in horizontal)

    assert lengths == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_disv_connection_length_uses_aquifer_cell_center():
    modelgrid = build_asymmetric_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    _, _, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        bedleak=1.0,
    )

    assert len(connectiondata) == 1
    assert connectiondata[0][2] == (0, 1)
    assert connectiondata[0][7] == pytest.approx(1.5)


def test_disv_vertical_connection():
    modelgrid = build_simple_disv_grid(nlay=2)

    lake_map = np.full((2, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    _, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        bedleak=1.0,
    )

    assert pakdata[0] == 5

    horizontal = [c for c in connectiondata if c[3] == "horizontal"]
    vertical = [c for c in connectiondata if c[3] == "vertical"]

    assert len(horizontal) == 4
    assert len(vertical) == 1

    vconn = vertical[0]

    assert vconn[2] == (1, 0)
    assert vconn[5:] == [0.0, 0.0, 0.0, 0.0]


def test_disv_horizontal_connections_in_nonzero_layer():
    modelgrid = build_simple_disv_grid(nlay=2)

    lake_map = np.full((2, modelgrid.ncpl), -1, dtype=int)
    lake_map[1, 0] = 0

    _, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        bedleak=1.0,
    )

    assert pakdata[0] == 4
    assert {conn[2] for conn in connectiondata} == {
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    }
    assert all(conn[3] == "horizontal" for conn in connectiondata)


def test_disv_inactive_neighbor():
    modelgrid = build_simple_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    idomain = np.ones((1, modelgrid.ncpl), dtype=int)

    # Make the east neighbour inactive
    idomain[0, 3] = 0

    _, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        lake_map,
        idomain=idomain,
        bedleak=1.0,
    )

    assert pakdata[0] == 3

    expected = {
        (0, 1),  # north
        (0, 2),  # west
        (0, 4),  # south
    }

    returned = {c[2] for c in connectiondata}

    assert returned == expected
    assert all(c[3] == "horizontal" for c in connectiondata)


def test_disv_idomain_update():
    modelgrid = build_simple_disv_grid()

    lake_map = np.full((1, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, 0] = 0

    idomain = np.ones((1, modelgrid.ncpl), dtype=int)

    idomain_out, _, _ = get_lak_connections(
        modelgrid,
        lake_map,
        idomain=idomain,
        bedleak=1.0,
    )

    expected = np.ones((1, modelgrid.ncpl), dtype=int)
    expected[0, 0] = 0

    assert np.array_equal(idomain_out, expected)


def test_disv_grid_geometry_accessed_once(monkeypatch):
    modelgrid = build_simple_disv_grid(nlay=2)
    lake_map = np.full((2, modelgrid.ncpl), -1, dtype=int)
    lake_map[0, :2] = 0

    # Build neighbor topology before counting geometry accesses. This is cached
    # independently by Grid.neighbors().
    modelgrid.neighbors(method="rook")

    properties = {
        "top_botm": VertexGrid,
        "verts": VertexGrid,
        "iverts": VertexGrid,
        "xcellcenters": Grid,
        "ycellcenters": Grid,
    }
    originals = {name: getattr(owner, name) for name, owner in properties.items()}
    access_count = dict.fromkeys(properties, 0)

    def counted_property(name):
        def getter(grid):
            access_count[name] += 1
            return originals[name].fget(grid)

        return property(getter)

    for name, owner in properties.items():
        monkeypatch.setattr(owner, name, counted_property(name))

    get_lak_connections(modelgrid, lake_map, bedleak=1.0)

    assert all(count <= 1 for count in access_count.values())


def test_disv_shared_boundary_split_by_vertex():
    # vertex 2 splits the common boundary into two shared edges
    vertices = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 1.0, 1.0),
        (3, 1.0, 2.0),
        (4, 0.0, 2.0),
        (5, 2.0, 0.0),
        (6, 2.0, 2.0),
    ]
    cell2d = [
        (0, 0.5, 1.0, 5, 0, 1, 2, 3, 4),
        (1, 1.5, 1.0, 5, 1, 5, 6, 3, 2),
    ]
    modelgrid = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(2),
        botm=-np.ones((1, 2)),
        idomain=np.ones((1, 2), dtype=int),
        nlay=1,
    )

    idomain, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        np.array([[0, -1]], dtype=int),
        bedleak=1.0,
    )

    assert pakdata[0] == 1
    assert len(connectiondata) == 1
    conn = connectiondata[0]
    assert conn[2] == (0, 1)
    assert conn[3] == "horizontal"
    assert conn[7] == pytest.approx(0.5)
    assert conn[8] == pytest.approx(2.0)
    assert np.array_equal(idomain, np.array([[0, 1]]))


def test_disv_lake_without_connections_warns():
    # Only the lake cell carries the hanging vertex, so rook adjacency is absent.
    vertices = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 1.0, 1.0),
        (3, 1.0, 2.0),
        (4, 0.0, 2.0),
        (5, 2.0, 0.0),
        (6, 2.0, 2.0),
    ]
    cell2d = [
        (0, 0.5, 1.0, 5, 0, 1, 2, 3, 4),
        (1, 1.5, 1.0, 4, 1, 5, 6, 3),
    ]
    modelgrid = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(2),
        botm=-np.ones((1, 2)),
        idomain=np.ones((1, 2), dtype=int),
        nlay=1,
    )

    with pytest.warns(UserWarning, match="Embedded lake 0 has no connections"):
        idomain, pakdata, connectiondata = get_lak_connections(
            modelgrid,
            np.array([[0, -1]], dtype=int),
            bedleak=1.0,
        )

    assert pakdata[0] == 0
    assert connectiondata == []
    assert np.array_equal(idomain, np.ones((1, 2), dtype=int))


@pytest.mark.parametrize("closed", (False, True))
def test_disv_nonrectangular_cells(closed):
    vertices = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 1.0, 1.0),
        (3, 2.0, 0.0),
    ]
    if closed:
        cell2d = [
            (0, 2.0 / 3.0, 1.0 / 3.0, 4, 0, 1, 2, 0),
            (1, 4.0 / 3.0, 1.0 / 3.0, 4, 1, 3, 2, 1),
        ]
    else:
        cell2d = [
            (0, 2.0 / 3.0, 1.0 / 3.0, 3, 0, 1, 2),
            (1, 4.0 / 3.0, 1.0 / 3.0, 3, 1, 3, 2),
        ]
    modelgrid = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(2),
        botm=-np.ones((1, 2)),
        idomain=np.ones((1, 2), dtype=int),
        nlay=1,
    )

    _, pakdata, connectiondata = get_lak_connections(
        modelgrid,
        np.array([[0, -1]], dtype=int),
        bedleak=1.0,
    )

    assert pakdata[0] == 1
    assert connectiondata[0][7] == pytest.approx(1.0 / 3.0)
    assert connectiondata[0][8] == pytest.approx(1.0)


def build_dis_and_equivalent_disv(nlay, nrow, ncol, delr, delc, top, botm):
    structured = StructuredGrid(
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=np.ones((nlay, nrow, ncol), dtype=int),
        nlay=nlay,
    )

    xv = np.concatenate(([0.0], np.cumsum(delr)))
    yv = delc.sum() - np.concatenate(([0.0], np.cumsum(delc)))
    vertices = []
    ivert = {}
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            ivert[(i, j)] = len(vertices)
            vertices.append((len(vertices), float(xv[j]), float(yv[i])))

    cell2d = []
    for i in range(nrow):
        for j in range(ncol):
            cell2d.append(
                (
                    i * ncol + j,
                    0.5 * (xv[j] + xv[j + 1]),
                    0.5 * (yv[i] + yv[i + 1]),
                    4,
                    ivert[(i, j)],
                    ivert[(i, j + 1)],
                    ivert[(i + 1, j + 1)],
                    ivert[(i + 1, j)],
                )
            )

    ncpl = nrow * ncol
    vertex = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=top.flatten(),
        botm=botm.reshape(nlay, ncpl),
        idomain=np.ones((nlay, ncpl), dtype=int),
        nlay=nlay,
    )
    return structured, vertex, vertices, cell2d


@pytest.mark.parametrize(
    "lakes, inactive",
    (
        ([[(0, 2, 2)]], []),
        ([[(0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2)]], []),
        ([[(0, 0, 0)]], []),
        ([[(0, 0, 2)]], []),
        ([[(1, 2, 2)]], []),
        ([[(0, 2, 2), (1, 2, 2)]], []),
        ([[(0, 2, 2)]], [(0, 2, 3)]),
        ([[(0, 2, 1)], [(0, 2, 2)]], []),
        ([[(0, 1, 1)], [(0, 3, 3)]], []),
        ([[(0, 2, 2), (0, 2, 3)]], [(0, 2, 2)]),
    ),
)
def test_disv_matches_dis_embedded_lake(lakes, inactive):
    nlay, nrow, ncol = 2, 5, 5
    delr = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
    delc = np.array([5.0, 15.0, 25.0, 15.0, 5.0])
    top = np.full((nrow, ncol), 10.0)
    botm = np.array([np.zeros((nrow, ncol)), np.full((nrow, ncol), -10.0)])
    ncpl = nrow * ncol
    structured, vertex, _, _ = build_dis_and_equivalent_disv(
        nlay, nrow, ncol, delr, delc, top, botm
    )

    lake_map = np.full((nlay, nrow, ncol), -1, dtype=int)
    for lake_number, cells in enumerate(lakes):
        for cell in cells:
            lake_map[cell] = lake_number
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    for cell in inactive:
        idomain[cell] = 0

    dis_idomain, dis_pakdata, dis_conn = get_lak_connections(
        structured, lake_map.copy(), idomain=idomain.copy(), bedleak=1.0
    )
    disv_idomain, disv_pakdata, disv_conn = get_lak_connections(
        vertex,
        lake_map.reshape(nlay, ncpl).copy(),
        idomain=idomain.reshape(nlay, ncpl).copy(),
        bedleak=1.0,
    )

    def normalize(conn):
        lake_number, _, cellid, claktype, _, _, _, connlen, connwidth = conn
        if len(cellid) == 3:
            k, i, j = cellid
            cellid = (k, i * ncol + j)
        return (
            lake_number,
            *cellid,
            claktype,
            round(connlen, 6),
            round(connwidth, 6),
        )

    assert dis_pakdata == disv_pakdata
    assert sorted(map(normalize, dis_conn)) == sorted(map(normalize, disv_conn))
    assert np.array_equal(dis_idomain.reshape(nlay, ncpl), disv_idomain)


@requires_exe("mf6")
def test_disv_lake_matches_dis_run(function_tmpdir):
    nlay, nrow, ncol = 1, 3, 3
    delr = np.array([1.0, 1.5, 2.0])
    delc = np.array([2.0, 1.5, 1.0])
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))
    structured, vertex, vertices, cell2d = build_dis_and_equivalent_disv(
        nlay, nrow, ncol, delr, delc, top, botm
    )

    def build_and_run(grid_type):
        name = f"{grid_type}_lake_test"
        modelgrid = structured if grid_type == "dis" else vertex
        lake_map = np.full(modelgrid.shape, -1, dtype=int)
        lake_map[(0, 1, 1) if grid_type == "dis" else (0, 4)] = 0
        idomain, pakdata_dict, connectiondata = get_lak_connections(
            modelgrid,
            lake_map,
            bedleak=1.0,
        )
        assert pakdata_dict[0] == 4

        sim = MFSimulation(
            sim_name=name,
            sim_ws=function_tmpdir / grid_type,
            exe_name="mf6",
        )
        ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
        ModflowIms(
            sim,
            print_option="summary",
            linear_acceleration="BICGSTAB",
            outer_dvclose=1e-9,
            inner_dvclose=1e-10,
        )
        gwf = ModflowGwf(
            sim,
            modelname=name,
            newtonoptions="newton under_relaxation",
            save_flows=True,
        )
        if grid_type == "dis":
            ModflowGwfdis(
                gwf,
                nlay=nlay,
                nrow=nrow,
                ncol=ncol,
                delr=delr,
                delc=delc,
                top=top,
                botm=botm,
                idomain=idomain,
            )
        else:
            ModflowGwfdisv(
                gwf,
                nlay=nlay,
                ncpl=nrow * ncol,
                top=top.flatten(),
                botm=botm.reshape(nlay, nrow * ncol),
                vertices=vertices,
                cell2d=cell2d,
                idomain=idomain,
            )
        ModflowGwfic(gwf, strt=0.5)
        ModflowGwfnpf(gwf, icelltype=1, k=1.0)

        chd_spd = []
        for i in range(nrow):
            if grid_type == "dis":
                chd_spd.extend([((0, i, 0), 0.75), ((0, i, 2), 0.25)])
            else:
                chd_spd.extend([((0, i * ncol), 0.75), ((0, i * ncol + 2), 0.25)])
        ModflowGwfchd(gwf, stress_period_data=chd_spd)

        lak = ModflowGwflak(
            gwf,
            print_stage=True,
            stage_filerecord=f"{name}.lak.stage.bin",
            nlakes=1,
            packagedata=[[0, 0.5, pakdata_dict[0]]],
            connectiondata=connectiondata,
            perioddata={0: [[0, "rainfall", 0.01]]},
            pname="LAK-1",
        )
        ModflowGwfoc(
            gwf,
            head_filerecord=f"{name}.hds",
            saverecord=[("HEAD", "ALL")],
            printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        sim.write_simulation()
        success, _ = sim.run_simulation(silent=False)
        assert success, f"could not run {sim.name}"

        heads = gwf.output.head().get_data().reshape(nlay, nrow, ncol)
        stage = lak.output.stage().get_data()
        return heads, stage

    dis_heads, dis_stage = build_and_run("dis")
    disv_heads, disv_stage = build_and_run("disv")

    assert np.allclose(dis_heads, disv_heads, rtol=1e-6, atol=1e-6)
    assert np.allclose(dis_stage, disv_stage, rtol=1e-6, atol=1e-6)
