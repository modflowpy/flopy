# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import flopy
import numpy as np
import matplotlib.pyplot as plt

try:
    import pymake
except ImportError:
    print("could not import pymake")
    pymake = False

cpth = os.path.join("temp", "t049")
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth, exist_ok=True)

mf2005_exe = "mf2005"
v = flopy.which(mf2005_exe)

mpth_exe = "mp6"
v2 = flopy.which(mpth_exe)

rung = True
if v is None or v2 is None:
    rung = False


def test_modpath():
    pth = os.path.join("..", "examples", "data", "freyberg")
    mfnam = "freyberg.nam"

    if pymake:
        run = rung
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
        pymake.setup(os.path.join(pth, mfnam), lpth)
    else:
        run = False
        lpth = pth

    m = flopy.modflow.Modflow.load(
        mfnam, model_ws=lpth, verbose=True, exe_name=mf2005_exe
    )
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            success = False
        assert success, "modflow model run did not terminate successfully"

    # create the forward modpath file
    mpnam = "freybergmp"
    mp = flopy.modpath.Modpath6(
        mpnam, exe_name=mpth_exe, modflowmodel=m, model_ws=lpth
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

    if run:
        try:
            success, buff = mp.run_model(silent=False)
        except:
            success = False
        assert (
            success
        ), "forward modpath model run did not terminate successfully"

    mpnam = "freybergmpp"
    mpp = flopy.modpath.Modpath6(
        mpnam, exe_name=mpth_exe, modflowmodel=m, model_ws=lpth
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

    if run:
        try:
            success, buff = mpp.run_model(silent=False)
        except:
            success = False
        assert (
            success
        ), "backward modpath model run did not terminate successfully"

    # load modpath output files
    if run:
        endfile = os.path.join(lpth, mp.sim.endpoint_file)
        pthfile = os.path.join(lpth, mpp.sim.pathline_file)
    else:
        endfile = os.path.join(
            "..", "examples", "data", "mp6_examples", "freybergmp.gitmpend"
        )
        pthfile = os.path.join(
            "..", "examples", "data", "mp6_examples", "freybergmpp.gitmppth"
        )

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
    assert len(plines) == 576, "there are not 576 particle pathlines in file"

    return


def test_pathline_plot():
    pth = os.path.join("..", "examples", "data", "freyberg")
    mfnam = "freyberg.nam"

    run = rung
    try:
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
    except:
        run = False
        lpth = pth

    nampath = os.path.join(lpth, mfnam)
    assert os.path.exists(nampath), f"namefile {nampath} doesn't exist."
    # load the modflow files for model map
    m = flopy.modflow.Modflow.load(
        mfnam, model_ws=lpth, verbose=True, forgive=False, exe_name=mf2005_exe
    )

    # load modpath output files
    if run:
        pthfile = os.path.join(lpth, "freybergmpp.mppth")
    else:
        pthfile = os.path.join(
            "..", "examples", "data", "mp6_examples", "freybergmpp.gitmppth"
        )

    # load the pathline data
    try:
        pthobj = flopy.utils.PathlineFile(pthfile)
    except:
        assert False, "could not load pathline file"

    # determine version
    ver = pthobj.version
    assert ver == 6, f"{pthfile} is not a MODPATH version 6 pathline file"

    # get all pathline data
    plines = pthobj.get_alldata()

    mm = flopy.plot.PlotMapView(model=m)
    try:
        mm.plot_pathline(plines, colors="blue", layer="all")
    except:
        assert False, 'could not plot pathline with layer="all"'

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, "could not plot grid and ibound"

    try:
        fpth = os.path.join(lpth, "pathline.png")
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, f"could not save plot as {fpth}"

    mm = flopy.plot.PlotMapView(model=m)
    try:
        mm.plot_pathline(plines, colors="green", layer=0)
    except:
        assert False, "could not plot pathline with layer=0"

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, "could not plot grid and ibound"

    try:
        fpth = os.path.join(lpth, "pathline2.png")
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, f"could not save plot as {fpth}"

    mm = flopy.plot.PlotMapView(model=m)
    try:
        mm.plot_pathline(plines, colors="red")
    except:
        assert False, "could not plot pathline"

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, "could not plot grid and ibound"

    try:
        fpth = os.path.join(lpth, "pathline3.png")
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, f"could not save plot as {fpth}"

    return


def test_pathline_plot_xc():
    from matplotlib.collections import LineCollection

    # test with multi-layer example
    model_ws = os.path.join("..", "examples", "data", "mp6")

    ml = flopy.modflow.Modflow.load(
        "EXAMPLE.nam", model_ws=model_ws, exe_name=mf2005_exe
    )
    ml.change_model_ws(os.path.join(".", "temp"))
    ml.write_input()
    ml.run_model()

    mp = flopy.modpath.Modpath6(
        modelname="ex6",
        exe_name=mpth_exe,
        modflowmodel=ml,
        model_ws=os.path.join(".", "temp"),
        dis_file=f"{ml.name}.DIS",
        head_file=f"{ml.name}.hed",
        budget_file=f"{ml.name}.bud",
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

    pthobj = flopy.utils.PathlineFile(os.path.join("temp", "ex6.mppth"))
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


def test_mp5_load():
    # load the base freyberg model
    pth = os.path.join("..", "examples", "data", "freyberg")
    # load the modflow files for model map
    m = flopy.modflow.Modflow.load(
        "freyberg.nam", model_ws=pth, check=False, verbose=True, forgive=False
    )

    # load the pathline data
    fpth = os.path.join("..", "examples", "data", "mp5", "m.ptl")
    try:
        pthobj = flopy.utils.PathlineFile(fpth)
    except:
        assert False, "could not load pathline file"

    # load endpoint data
    fpth = os.path.join("..", "examples", "data", "mp5", "m.ept")
    try:
        endobj = flopy.utils.EndpointFile(fpth, verbose=True)
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
        fpth = os.path.join(cpth, "mp5.pathline.png")
        plt.savefig(fpth, dpi=300)
        plt.close()
    except:
        assert False, f"could not save plot as {fpth}"

    return


def test_mp5_timeseries_load():
    pth = os.path.join("..", "examples", "data", "mp5")
    files = [
        os.path.join(pth, name)
        for name in sorted(os.listdir(pth))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)
    return


def test_mp6_timeseries_load():
    pth = os.path.join("..", "examples", "data", "mp6")
    files = [
        os.path.join(pth, name)
        for name in sorted(os.listdir(pth))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)
    return


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

    return


if __name__ == "__main__":
    test_modpath()
    test_pathline_plot()
    test_pathline_plot_xc()
    test_mp5_load()
    test_mp5_timeseries_load()
    test_mp6_timeseries_load()
