import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from modflow_devtools.markers import requires_exe

from autotest.conftest import get_example_data_path
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowGhb,
    ModflowNwt,
    ModflowOc,
    ModflowRch,
    ModflowUpw,
)
from flopy.utils import HeadFile


def analytical_water_table_solution(h1, h2, z, R, K, L, x):
    h = np.zeros((x.shape[0]), float)
    b1 = h1 - z
    b2 = h2 - z
    h = np.sqrt(b1**2 - (x / L) * (b1**2 - b2**2) + (R * x / K) * (L - x)) + z
    return h


def fnwt_model_files(pattern):
    path = get_example_data_path() / "nwt_test"
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(pattern)]


@pytest.mark.parametrize("nwtfile", fnwt_model_files(".nwt"))
def test_nwt_pack_load(function_tmpdir, nwtfile):
    ws = os.path.dirname(nwtfile)
    ml = Modflow(model_ws=ws, version="mfnwt")
    if "fmt." in nwtfile.lower():
        ml.array_free_format = False
    else:
        ml.array_free_format = True

    nwt = ModflowNwt.load(nwtfile, ml)
    msg = f"{os.path.basename(nwtfile)} load unsuccessful"
    assert isinstance(nwt, ModflowNwt), msg

    # write the new file in the working directory
    ml.change_model_ws(function_tmpdir)
    nwt.write_file()

    fn = function_tmpdir / (ml.name + ".nwt")
    msg = f"{os.path.basename(nwtfile)} write unsuccessful"
    assert os.path.isfile(fn), msg

    ml2 = Modflow(model_ws=function_tmpdir, version="mfnwt")
    nwt2 = ModflowNwt.load(fn, ml2)
    lst = [
        a for a in dir(nwt) if not a.startswith("__") and not callable(getattr(nwt, a))
    ]
    for l in lst:
        msg = (
            "{} data instantiated from {} load  is not the same as written to "
            "{}".format(l, nwtfile, os.path.basename(fn))
        )
        assert nwt2[l] == nwt[l], msg


@pytest.mark.parametrize("namfile", fnwt_model_files(".nam"))
def test_nwt_model_load(function_tmpdir, namfile):
    f = os.path.basename(namfile)
    model_ws = os.path.dirname(namfile)
    ml = Modflow.load(f, model_ws=model_ws)
    msg = "Error: flopy model instance was not created"
    assert isinstance(ml, Modflow), msg

    # change the model work space and rewrite the files
    ml.change_model_ws(function_tmpdir)
    ml.write_input()

    # reload the model that was just written
    ml2 = Modflow.load(f, model_ws=function_tmpdir)

    # check that the data are the same
    for pn in ml.get_package_list():
        p = ml.get_package(pn)
        p2 = ml2.get_package(pn)
        lst = [
            a for a in dir(p) if not a.startswith("__") and not callable(getattr(p, a))
        ]
        for l in lst:
            msg = (
                "{}.{} data instantiated from {} load  is not the same as "
                "written to {}".format(pn, l, model_ws, function_tmpdir)
            )
            assert p[l] == p2[l], msg


@requires_exe("mfnwt")
def test_mfnwt_run(function_tmpdir):
    modelname = "watertable"

    # model dimensions
    nlay, nrow, ncol = 1, 1, 100

    # cell spacing
    delr = 50.0
    delc = 1.0

    # domain length
    L = 5000.0

    # boundary heads
    h1 = 20.0
    h2 = 11.0

    # ibound
    ibound = np.ones((nlay, nrow, ncol), dtype=int)

    # starting heads
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    strt[0, 0, 0] = h1
    strt[0, 0, -1] = h2

    # top of the aquifer
    top = 25.0

    # bottom of the aquifer
    botm = 0.0

    # hydraulic conductivity
    hk = 50.0

    # location of cell centroids
    x = np.arange(0.0, L, delr) + (delr / 2.0)

    # location of cell edges
    xa = np.arange(0, L + delr, delr)

    # recharge rate
    rchrate = 0.001

    # calculate the head at the cell centroids using the analytical solution function
    hac = analytical_water_table_solution(h1, h2, botm, rchrate, hk, L, x)

    # calculate the head at the cell edges using the analytical solution function
    ha = analytical_water_table_solution(h1, h2, botm, rchrate, hk, L, xa)

    # ghbs
    # ghb conductance
    b1, b2 = 0.5 * (h1 + hac[0]), 0.5 * (h2 + hac[-1])
    c1, c2 = hk * b1 * delc / (0.5 * delr), hk * b2 * delc / (0.5 * delr)
    # dtype
    ghb_dtype = ModflowGhb.get_default_dtype()

    # build ghb recarray
    stress_period_data = np.zeros((2), dtype=ghb_dtype)
    stress_period_data = stress_period_data.view(np.recarray)

    # fill ghb recarray
    stress_period_data[0] = (0, 0, 0, h1, c1)
    stress_period_data[1] = (0, 0, ncol - 1, h2, c2)

    mf = Modflow(
        modelname=modelname,
        exe_name="mfnwt",
        model_ws=function_tmpdir,
        version="mfnwt",
    )
    dis = ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    bas = ModflowBas(mf, ibound=ibound, strt=strt)
    lpf = ModflowUpw(mf, hk=hk, laytyp=1)
    ghb = ModflowGhb(mf, stress_period_data=stress_period_data)
    rch = ModflowRch(mf, rech=rchrate, nrchop=1)
    oc = ModflowOc(mf)
    nwt = ModflowNwt(mf)
    mf.write_input()

    # remove existing heads results, if necessary
    try:
        Path(function_tmpdir / f"{modelname}.hds").unlink()
    except:
        pass

    # run existing model
    mf.run_model()

    # Read the simulated MODFLOW-2005 model results
    # Create the headfile object
    headfile = function_tmpdir / f"{modelname}.hds"
    headobj = HeadFile(headfile, precision="single")
    times = headobj.get_times()
    head = headobj.get_data(totim=times[-1])

    # Plot the results
    if plt is not None:
        fig = plt.figure(figsize=(16, 6))

        ax = fig.add_subplot(1, 3, 1)
        ax.plot(xa, ha, linewidth=8, color="0.5", label="analytical solution")
        ax.plot(x, head[0, 0, :], color="red", label="MODFLOW-NWT")
        leg = ax.legend(loc="lower left")
        leg.draw_frame(False)
        ax.set_xlabel("Horizontal distance, in m")
        ax.set_ylabel("Head, in m")

        ax = fig.add_subplot(1, 3, 2)
        ax.plot(x, head[0, 0, :] - hac, linewidth=1, color="blue")
        ax.set_xlabel("Horizontal distance, in m")
        ax.set_ylabel("Error, in m")

        ax = fig.add_subplot(1, 3, 3)
        ax.plot(x, 100.0 * (head[0, 0, :] - hac) / hac, linewidth=1, color="blue")
        ax.set_xlabel("Horizontal distance, in m")
        ax.set_ylabel("Percent Error")

        fig.savefig(function_tmpdir / f"{modelname}.png")
