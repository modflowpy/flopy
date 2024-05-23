import os

import numpy as np
from matplotlib import pyplot as plt

from autotest.test_mp6 import eval_timeseries
from flopy.modflow import Modflow
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile


def test_mp5_load(function_tmpdir, example_data_path):
    # load the base freyberg model
    freyberg_ws = example_data_path / "freyberg"
    # load the modflow files for model map
    m = Modflow.load(
        "freyberg.nam",
        model_ws=freyberg_ws,
        check=False,
        verbose=True,
        forgive=False,
    )

    # load the pathline data
    pthobj = PathlineFile(str(example_data_path / "mp5" / "m.ptl"))

    # load endpoint data
    fpth = str(example_data_path / "mp5" / "m.ept")
    endobj = EndpointFile(fpth, verbose=True)

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
    mm = PlotMapView(model=m)
    for n in pthobj.nid:
        p = pthobj.get_data(partid=n)
        e = endobj.get_data(partid=n)
        mm.plot_pathline(p, colors=colors[n], layer="all")
        mm.plot_endpoint(e)

    # plot the grid and ibound array
    mm.plot_grid(lw=0.5)
    mm.plot_ibound()

    fpth = function_tmpdir / "mp5.pathline.png"
    plt.savefig(fpth, dpi=300)
    plt.close()


def test_mp5_timeseries_load(example_data_path):
    pth = str(example_data_path / "mp5")
    files = [
        os.path.join(pth, name)
        for name in sorted(os.listdir(pth))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)
