import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import flopy

pth = os.path.join("..", "examples", "data", "mf6-freyberg")

name = "freyberg"
tpth = os.path.join("temp", "t078")
# delete the directory if it exists
if os.path.isdir(tpth):
    shutil.rmtree(tpth)
# make the directory
os.makedirs(tpth)


def export_ascii_grid(modelgrid, file_path, v, nodata=0.0):
    shape = v.shape
    xcenters = modelgrid.xcellcenters[0, :]
    cellsize = xcenters[1] - xcenters[0]
    with open(file_path, "w") as f:
        f.write("NCOLS {}\n".format(shape[1]))
        f.write("NROWS {}\n".format(shape[0]))
        f.write("XLLCENTER {}\n".format(modelgrid.xoffset + 0.5 * cellsize))
        f.write("YLLCENTER {}\n".format(modelgrid.yoffset + 0.5 * cellsize))
        f.write("CELLSIZE {}\n".format(cellsize))
        f.write("NODATA_VALUE {}\n".format(nodata))
        np.savetxt(f, v, fmt="%.4f")
    return


def test_base_run():
    sim = flopy.mf6.MFSimulation().load(
        sim_name=name,
        sim_ws=pth,
        exe_name="mf6",
        verbosity_level=0,
    )
    ws = os.path.join(tpth, "freyberg")
    sim.set_sim_path(ws)

    # remove the well package
    gwf = sim.get_model("freyberg")
    gwf.remove_package("wel_0")

    # write the simulation files and run the model
    sim.write_simulation()
    sim.run_simulation()

    # export bottom, water levels, and k11 as ascii raster files
    # for interpolation in test_lake()
    bot = gwf.dis.botm.array.squeeze()
    export_ascii_grid(
        gwf.modelgrid,
        os.path.join(ws, "bot.asc"),
        bot,
    )
    top = gwf.output.head().get_data().squeeze() + 2.0
    top = np.where(gwf.dis.idomain.array.squeeze() < 1.0, 0.0, top)
    export_ascii_grid(
        gwf.modelgrid,
        os.path.join(ws, "top.asc"),
        top,
    )
    k11 = gwf.npf.k.array.squeeze()
    export_ascii_grid(
        gwf.modelgrid,
        os.path.join(ws, "k11.asc"),
        k11,
    )

    return


def test_lake():
    ws = os.path.join(tpth, "freyberg")
    top = flopy.utils.Raster.load(os.path.join(ws, "top.asc"))
    bot = flopy.utils.Raster.load(os.path.join(ws, "bot.asc"))
    k11 = flopy.utils.Raster.load(os.path.join(ws, "k11.asc"))

    sim = flopy.mf6.MFSimulation().load(
        sim_name=name,
        sim_ws=ws,
        exe_name="mf6",
        verbosity_level=0,
    )

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
    idx = np.where(lakes > -1)
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

    # v = gwf.dis.top.array
    # v = gwf.dis.botm.array

    k11_tm = k11.resample_to_grid(
        gwf.modelgrid,
        band=k11.bands[0],
        method="linear",
        extrapolate_edges=True,
    )
    gwf.npf.k = k11_tm

    # mm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid)
    # mm.plot_array(k11_tm)

    pakdata_dict, connectiondata = flopy.mf6.utils.get_lak_connections(
        gwf.modelgrid,
        lakes,
        bedleak=5e-9,
    )
    lak_pak_data = []
    for key, value in pakdata_dict.items():
        lak_pak_data.append([key, 35.0, value])
    lak_spd = {0: [[0, "rainfall", 3.2e-9]]}
    lak = flopy.mf6.ModflowGwflak(
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

    # convert to Newton-Raphson fomulation and update the linear accelerator
    gwf.name_file.newtonoptions = "NEWTON UNDER_RELAXATION"
    sim.ims.linear_acceleration = "BICGSTAB"

    # write the revised simulation files and run the model
    sim.write_simulation()
    sim.run_simulation(silent=False)

    return


def test_embedded_lak_ex01():
    nper = 1
    perioddata = [(1.0, 1, 1.0)]
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
    botm = (
        107.0,
        97.0,
        87.0,
        77.0,
        67.0,
    )
    idomain = np.ones(shape3d, dtype=np.int32)
    idomain[0, 6:11, 6:11] = 0
    idomain[1, 7:10, 7:10] = 0

    lake_map = idomain.copy()
    lake_map[idomain == 1] = -1

    strt = 115.0

    k11 = 30
    k33 = (
        1179.0,
        30.0,
        30.0,
        30.0,
        30.0,
    )

    load_pth = os.path.join("..", "examples", "data", "mf2005_test")
    ml = flopy.modflow.Modflow.load(
        "l1a2k.nam",
        model_ws=load_pth,
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
    ws = os.path.join(tpth, "lak_ex01")
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name="mf6",
        sim_ws=ws,
    )
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=perioddata,
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        linear_acceleration="BICGSTAB",
        outer_maximum=1000,
        inner_maximum=100,
        outer_dvclose=1e-8,
        inner_dvclose=1e-9,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        newtonoptions="newton under_relaxation",
        print_input=True,
    )
    dis = flopy.mf6.ModflowGwfdis(
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
    ic = flopy.mf6.ModflowGwfic(
        gwf,
        strt=strt,
    )
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=k11,
        k33=k33,
    )
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
    )
    rch = flopy.mf6.ModflowGwfrcha(
        gwf,
        recharge=rch_rate,
    )
    evt = flopy.mf6.ModflowGwfevta(
        gwf,
        surface=evt_surf,
        depth=evt_depth,
        rate=evt_rate,
    )
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    pakdata_dict, connectiondata = flopy.mf6.utils.get_lak_connections(
        gwf.modelgrid,
        lake_map,
        bedleak=0.1,
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
    lak = flopy.mf6.ModflowGwflak(
        gwf,
        print_stage=True,
        print_flows=True,
        nlakes=1,
        packagedata=lak_pak_data,
        connectiondata=connectiondata,
        perioddata=lak_spd,
        pname="LAK-1",
    )

    # write the simulation files and run the model
    sim.write_simulation()
    sim.run_simulation(silent=False)


if __name__ == "__main__":
    # test_base_run()
    # test_lake()
    test_embedded_lak_ex01()
