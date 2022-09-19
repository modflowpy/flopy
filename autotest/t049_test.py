# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os

import matplotlib.pyplot as plt
import numpy as np

import flopy
from autotest.conftest import requires_exe


@requires_exe('mf2005', 'mp6')
def test_modpath(tmpdir, example_data_path):
    pth = example_data_path / "freyberg"
    mfnam = "freyberg.nam"

    m = flopy.modflow.Modflow.load(
        mfnam,
        model_ws=str(pth),
        verbose=True,
        exe_name='mf2005',
        check=False,
    )
    assert m.load_fail is False

    m.change_model_ws(str(tmpdir))
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "modflow model run did not terminate successfully"

    # create the forward modpath file
    mpnam = "freybergmp"
    mp = flopy.modpath.Modpath6(
        mpnam,
        exe_name='mp6',
        modflowmodel=m,
        model_ws=str(tmpdir),
    )
    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mp.create_mpsim(
        trackdir="forward", simtype="endpoint", packages="RCH"
    )

    # write forward particle track files
    mp.write_input()

    if success:
        success, buff = mp.run_model(silent=False)
        assert (
            success
        ), "forward modpath model run did not terminate successfully"

    mpnam = "freybergmpp"
    mpp = flopy.modpath.Modpath6(
        mpnam,
        exe_name='mp6',
        modflowmodel=m,
        model_ws=str(tmpdir),
    )
    mpbas = flopy.modpath.Modpath6Bas(
        mpp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mpp.create_mpsim(
        trackdir="backward", simtype="pathline", packages="WEL"
    )

    # write backward particle track files
    mpp.write_input()

    if success:
        success, buff = mpp.run_model(silent=False)
        assert (
            success
        ), "backward modpath model run did not terminate successfully"

    # load modpath output files
    if success:
        endfile = str(tmpdir / mp.sim.endpoint_file)
        pthfile = str(tmpdir / mpp.sim.pathline_file)

        # load the endpoint data
        try:
            endobj = flopy.utils.EndpointFile(endfile)
        except:
            assert False, "could not load endpoint file"
        ept = endobj.get_alldata()
        assert ept.shape == (695,), "shape of endpoint file is not (695,)"

        # load the pathline data
        try:
            pthobj = flopy.utils.PathlineFile(pthfile)
        except:
            assert False, "could not load pathline file"
        plines = pthobj.get_alldata()
        assert (
            len(plines) == 576
        ), "there are not 576 particle pathlines in file"

        eval_pathline_plot(str(tmpdir))


def eval_pathline_plot(lpth):
    # load the modflow files for model map
    mfnam = "freyberg.nam"
    m = flopy.modflow.Modflow.load(
        mfnam,
        model_ws=lpth,
        verbose=True,
        forgive=False,
        exe_name='mf2005',
    )

    # load modpath output files
    pthfile = os.path.join(lpth, "freybergmpp.mppth")

    # load the pathline data
    pthobj = flopy.utils.PathlineFile(pthfile)

    # determine version
    ver = pthobj.version
    assert ver == 6, f"{pthfile} is not a MODPATH version 6 pathline file"

    # get all pathline data
    plines = pthobj.get_alldata()

    mm = flopy.plot.PlotMapView(model=m)
    mm.plot_pathline(plines, colors="blue", layer="all")

    # plot the grid and ibound array
    mm.plot_grid()
    mm.plot_ibound()
    fpth = os.path.join(lpth, "pathline.png")
    plt.savefig(fpth)
    plt.close()

    mm = flopy.plot.PlotMapView(model=m)
    mm.plot_pathline(plines, colors="green", layer=0)

    # plot the grid and ibound array
    mm.plot_grid()
    mm.plot_ibound()

    fpth = os.path.join(lpth, "pathline2.png")
    plt.savefig(fpth)
    plt.close()

    mm = flopy.plot.PlotMapView(model=m)
    mm.plot_pathline(plines, colors="red")

    # plot the grid and ibound array
    mm.plot_grid()
    mm.plot_ibound()

    fpth = os.path.join(lpth, "pathline3.png")
    plt.savefig(fpth)
    plt.close()


@requires_exe('mf2005', 'mp6')
def test_pathline_plot_xc(tmpdir, example_data_path):
    from matplotlib.collections import LineCollection

    # test with multi-layer example
    load_ws = example_data_path / "mp6"

    ml = flopy.modflow.Modflow.load(
        "EXAMPLE.nam", model_ws=load_ws, exe_name='mf2005'
    )
    ml.change_model_ws(str(tmpdir))
    ml.write_input()
    ml.run_model()

    mp = flopy.modpath.Modpath6(
        modelname="ex6",
        exe_name='mp6',
        modflowmodel=ml,
        model_ws=str(tmpdir),
        dis_file=None,
        head_file=None,
        budget_file=None,
    )

    mpb = flopy.modpath.Modpath6Bas(
        mp, hdry=ml.lpf.hdry, laytyp=ml.lpf.laytyp, ibound=1, prsity=0.1
    )

    sim = mp.create_mpsim(
        trackdir="forward",
        simtype="pathline",
        packages="RCH",
        start_time=(2, 0, 1.0),
    )
    mp.write_input()

    mp.run_model(silent=False)

    pthobj = flopy.utils.PathlineFile(str(tmpdir / "ex6.mppth"))
    well_pathlines = pthobj.get_destination_pathline_data(
        dest_cells=[(4, 12, 12)]
    )

    mx = flopy.plot.PlotCrossSection(model=ml, line={"row": 4})
    mx.plot_bc("WEL", kper=2, color="blue")
    pth = mx.plot_pathline(well_pathlines, method="cell", colors="red")

    if not isinstance(pth, LineCollection):
        raise AssertionError()

    if len(pth._paths) != 6:
        raise AssertionError()

    plt.close()


@requires_exe('mf2005', 'mp5')
def test_mp5_load(tmpdir, example_data_path):
    # load the base freyberg model
    load_ws = example_data_path / "freyberg"
    # load the modflow files for model map
    m = flopy.modflow.Modflow.load(
        "freyberg.nam",
        model_ws=str(load_ws),
        check=False,
        verbose=True,
        forgive=False,
    )

    # load the pathline data
    fpth = example_data_path / "mp5", "m.ptl"
    try:
        pthobj = flopy.utils.PathlineFile(str(fpth))
    except:
        assert False, "could not load pathline file"

    # load endpoint data
    fpth = example_data_path / "mp5" / "m.ept"
    try:
        endobj = flopy.utils.EndpointFile(str(fpth), verbose=True)
    except:
        assert False, "could not load endpoint file"

    # determine version
    ver = pthobj.version
    assert ver == 5, f"{fpth} is not a MODPATH version 5 pathline file"

    # read all of the pathline and endpoint data
    plines = pthobj.get_alldata()
    epts = endobj.get_alldata()

    # determine the number of particles in the pathline file
    nptl = pthobj.nid.shape[0]
    assert nptl == 64, "number of MODPATH 5 particles does not equal 64"

    hsv = plt.get_cmap("hsv")
    colors = hsv(np.linspace(0, 1.0, nptl))

    # plot the pathlines one pathline at a time
    mm = flopy.plot.PlotMapView(model=m)
    for n in pthobj.nid:
        p = pthobj.get_data(partid=n)
        e = endobj.get_data(partid=n)
        try:
            mm.plot_pathline(p, colors=colors[n], layer="all")
        except:
            assert False, f'could not plot pathline {n + 1} with layer="all"'
        try:
            mm.plot_endpoint(e)
        except:
            assert False, f'could not plot endpoint {n + 1} with layer="all"'

    # plot the grid and ibound array
    try:
        mm.plot_grid(lw=0.5)
        mm.plot_ibound()
    except:
        assert False, "could not plot grid and ibound"

    try:
        fpth = tmpdir / "mp5.pathline.png"
        plt.savefig(str(fpth), dpi=300)
        plt.close()
    except:
        assert False, f"could not save plot as {fpth}"


@requires_exe('mf2005', 'mp5')
def test_mp5_timeseries_load(example_data_path):
    pth = example_data_path / "mp5"
    files = [
        str(pth / name)
        for name in sorted(os.listdir(str(pth)))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)


@requires_exe('mf2005', 'mp6')
def test_mp6_timeseries_load(example_data_path):
    pth = example_data_path / "mp6"
    files = [
        str(pth / name)
        for name in sorted(os.listdir(str(pth)))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)


def eval_timeseries(file):
    ts = flopy.utils.TimeseriesFile(file)
    msg = (
        f"{os.path.basename(file)} "
        "is not an instance of flopy.utils.TimeseriesFile"
    )
    assert isinstance(ts, flopy.utils.TimeseriesFile), msg

    # get the all of the data
    try:
        tsd = ts.get_alldata()
    except:
        pass
    msg = f"could not load data using get_alldata() from {os.path.basename(file)}."
    assert len(tsd) > 0, msg

    # get the data for the last particleid
    partid = None
    try:
        partid = ts.get_maxid()
    except:
        pass
    msg = (
        "could not get maximum particleid using get_maxid() from "
        f"{os.path.basename(file)}."
    )
    assert partid is not None, msg

    try:
        tsd = ts.get_data(partid=partid)
    except:
        pass
    msg = (
        f"could not load data for particleid {partid} using get_data() from "
        f"{os.path.basename(file)}. Maximum partid = {ts.get_maxid()}."
    )
    assert tsd.shape[0] > 0, msg

    timemax = None
    try:
        timemax = ts.get_maxtime() / 2.0
    except:
        pass
    msg = (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )
    assert timemax is not None, msg

    try:
        tsd = ts.get_alldata(totim=timemax)
    except:
        pass
    msg = (
        f"could not load data for totim>={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )
    assert len(tsd) > 0, msg

    timemax = None
    try:
        timemax = ts.get_maxtime()
    except:
        pass
    msg = (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )
    assert timemax is not None, msg

    try:
        tsd = ts.get_alldata(totim=timemax, ge=False)
    except:
        pass
    msg = (
        f"could not load data for totim<={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )
    assert len(tsd) > 0, msg
