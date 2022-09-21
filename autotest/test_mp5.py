import os

import numpy as np
from autotest.conftest import requires_pkg
from autotest.test_mp6 import eval_timeseries
from matplotlib import pyplot as plt

from flopy.modflow import Modflow
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile


@requires_pkg("pandas")
def test_mp5_load(tmpdir, example_data_path):
    # load the base freyberg model
    freyberg_ws = example_data_path / "freyberg"
    # load the modflow files for model map
    m = Modflow.load(
        "freyberg.nam",
        model_ws=str(freyberg_ws),
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
        try:
            mm.plot_pathline(p, colors=colors[n], layer="all")
        except:
            assert False, f'could not plot pathline {n + 1} with layer="all"'
        try:
            mm.plot_endpoint(e)
        except:
            assert False, f'could not plot endpoint {n + 1} with layer="all"'

    # plot the grid and ibound array
    mm.plot_grid(lw=0.5)
    mm.plot_ibound()

    fpth = os.path.join(str(tmpdir), "mp5.pathline.png")
    plt.savefig(fpth, dpi=300)
    plt.close()


@requires_pkg("pandas")
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
