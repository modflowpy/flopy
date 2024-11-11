import os

import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import (
    LineCollection,
    PatchCollection,
    PathCollection,
    QuadMesh,
)
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.discretization import StructuredGrid
from flopy.mf6 import MFSimulation
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowWel,
)
from flopy.modpath import Modpath6, Modpath6Bas
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import CellBudgetFile, EndpointFile, HeadFile, PathlineFile


@pytest.fixture
def quasi3d_model(function_tmpdir):
    mf = Modflow("model_mf", model_ws=function_tmpdir, exe_name="mf2005")

    # Model domain and grid definition
    Lx = 1000.0
    Ly = 1000.0
    ztop = 0.0
    zbot = -30.0
    nlay = 3
    nrow = 10
    ncol = 10
    delr = Lx / ncol
    delc = Ly / nrow
    laycbd = [0] * (nlay)
    laycbd[0] = 1
    botm = np.linspace(ztop, zbot, nlay + np.sum(laycbd) + 1)[1:]

    # Create the discretization object
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=ztop,
        botm=botm,
        laycbd=laycbd,
    )

    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.0
    strt[:, :, -1] = 0.0
    ModflowBas(mf, ibound=ibound, strt=strt)

    # Add LPF package to the MODFLOW model
    ModflowLpf(mf, hk=10.0, vka=10.0, ipakcb=53, vkcb=10)

    # add a well
    row = int((nrow - 1) / 2)
    col = int((ncol - 1) / 2)
    spd = {0: [[1, row, col, -1000]]}
    ModflowWel(mf, stress_period_data=spd)

    # Add OC package to the MODFLOW model
    spd = {(0, 0): ["save head", "save budget"]}
    ModflowOc(mf, stress_period_data=spd, compact=True)

    # Add PCG package to the MODFLOW model
    ModflowPcg(mf)

    # Write the MODFLOW model input files
    mf.write_input()

    # Run the MODFLOW model
    success, buff = mf.run_model()

    assert success, "test_plotting_with_quasi3d_layers() failed"

    return mf


@requires_exe("mf2005")
def test_map_plot_with_quasi3d_layers(quasi3d_model):
    # read output
    hf = HeadFile(os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.hds"))
    head = hf.get_data(totim=1.0)
    cbb = CellBudgetFile(
        os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.cbc")
    )
    frf = cbb.get_data(text="FLOW RIGHT FACE", totim=1.0)[0]
    fff = cbb.get_data(text="FLOW FRONT FACE", totim=1.0)[0]
    flf = cbb.get_data(text="FLOW LOWER FACE", totim=1.0)[0]

    # plot a map
    plt.figure()
    mv = PlotMapView(model=quasi3d_model, layer=1)
    mv.plot_grid()
    mv.plot_array(head)
    mv.contour_array(head)
    mv.plot_ibound()
    mv.plot_bc("wel")
    mv.plot_vector(frf, fff)
    plt.savefig(os.path.join(quasi3d_model.model_ws, "plt01.png"))


@requires_exe("mf2005")
def test_cross_section_with_quasi3d_layers(quasi3d_model):
    # read output
    hf = HeadFile(os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.hds"))
    head = hf.get_data(totim=1.0)
    cbb = CellBudgetFile(
        os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.cbc")
    )
    frf = cbb.get_data(text="FLOW RIGHT FACE", totim=1.0)[0]
    fff = cbb.get_data(text="FLOW FRONT FACE", totim=1.0)[0]
    flf = cbb.get_data(text="FLOW LOWER FACE", totim=1.0)[0]

    # plot a cross-section
    plt.figure()
    cs = PlotCrossSection(
        model=quasi3d_model,
        line={"row": int((quasi3d_model.modelgrid.nrow - 1) / 2)},
    )
    cs.plot_grid()
    cs.plot_array(head)
    cs.contour_array(head)
    cs.plot_ibound()
    cs.plot_bc("wel")
    cs.plot_vector(frf, fff, flf, head=head)
    plt.savefig(os.path.join(quasi3d_model.model_ws, "plt02.png"))
    plt.close()
