from os.path import join

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.modflow import Modflow
from flopy.modpath import Modpath6, Modpath6Bas
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import CellBudgetFile, EndpointFile, HeadFile, PathlineFile


@pytest.fixture
def modpath_model(function_tmpdir, example_data_path):
    # test with multi-layer example
    load_ws = example_data_path / "mp6"

    ml = Modflow.load("EXAMPLE.nam", model_ws=load_ws, exe_name="mf2005")
    ml.change_model_ws(function_tmpdir)
    ml.write_input()
    ml.run_model()

    mp = Modpath6(
        modelname="ex6",
        exe_name="mp6",
        modflowmodel=ml,
        model_ws=function_tmpdir,
    )

    mpb = Modpath6Bas(
        mp, hdry=ml.lpf.hdry, laytyp=ml.lpf.laytyp, ibound=1, prsity=0.1
    )

    sim = mp.create_mpsim(
        trackdir="forward",
        simtype="pathline",
        packages="RCH",
        start_time=(2, 0, 1.0),
    )
    return ml, mp, sim


@requires_exe("mf2005", "mp6")
def test_plot_map_view_mp6_plot_pathline(modpath_model):
    ml, mp, sim = modpath_model
    mp.write_input()
    mp.run_model(silent=False)

    pthobj = PathlineFile(join(mp.model_ws, "ex6.mppth"))
    well_pathlines = pthobj.get_destination_pathline_data(
        dest_cells=[(4, 12, 12)]
    )

    def test_plot(pl):
        mx = PlotMapView(model=ml)
        mx.plot_grid()
        mx.plot_bc("WEL", kper=2, color="blue")
        pth = mx.plot_pathline(pl, colors="red")
        # plt.show()
        assert isinstance(pth, LineCollection)
        assert len(pth._paths) == 114

    # support pathlines as list of recarrays
    test_plot(well_pathlines)

    # support pathlines as list of dataframes
    test_plot([pd.DataFrame(pl) for pl in well_pathlines])

    # support pathlines as single recarray
    test_plot(np.concatenate(well_pathlines))

    # support pathlines as single dataframe
    test_plot(pd.DataFrame(np.concatenate(well_pathlines)))


@pytest.mark.slow
@requires_exe("mf2005", "mp6")
def test_plot_cross_section_mp6_plot_pathline(modpath_model):
    ml, mp, sim = modpath_model
    mp.write_input()
    mp.run_model(silent=False)

    pthobj = PathlineFile(join(mp.model_ws, "ex6.mppth"))
    well_pathlines = pthobj.get_destination_pathline_data(
        dest_cells=[(4, 12, 12)]
    )

    def test_plot(pl):
        mx = PlotCrossSection(model=ml, line={"row": 4})
        mx.plot_bc("WEL", kper=2, color="blue")
        pth = mx.plot_pathline(pl, method="cell", colors="red")
        assert isinstance(pth, LineCollection)
        assert len(pth._paths) == 6

    # support pathlines as list of recarrays
    test_plot(well_pathlines)

    # support pathlines as list of dataframes
    test_plot([pd.DataFrame(pl) for pl in well_pathlines])

    # support pathlines as single recarray
    test_plot(np.concatenate(well_pathlines))

    # support pathlines as single dataframe
    test_plot(pd.DataFrame(np.concatenate(well_pathlines)))


@requires_exe("mf2005", "mp6")
def test_plot_map_view_mp6_endpoint(modpath_model):
    ml, mp, sim = modpath_model
    mp.write_input()
    mp.run_model(silent=False)

    pthobj = EndpointFile(join(mp.model_ws, "ex6.mpend"))
    endpts = pthobj.get_alldata()

    # support endpoints as recarray
    assert isinstance(endpts, np.recarray)
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(endpts, direction="ending")
    # plt.show()
    assert isinstance(ep, PathCollection)

    # support endpoints as dataframe
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(pd.DataFrame(endpts), direction="ending")
    # plt.show()
    assert isinstance(ep, PathCollection)

    # test various possibilities for endpoint color configuration.
    # first, color kwarg as scalar
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(endpts, direction="ending", color="red")
    # plt.show()
    assert isinstance(ep, PathCollection)

    # c kwarg as array
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(
        endpts,
        direction="ending",
        c=np.random.rand(625) * -1000,
        cmap="viridis",
    )
    # plt.show()
    assert isinstance(ep, PathCollection)

    # colorbar: color by time to termination
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(
        endpts, direction="ending", shrink=0.5, colorbar=True
    )
    # plt.show()
    assert isinstance(ep, PathCollection)

    # if both color and c are provided, c takes precedence
    mv = PlotMapView(model=ml)
    mv.plot_bc("WEL", kper=2, color="blue")
    ep = mv.plot_endpoint(
        endpts,
        direction="ending",
        color="red",
        c=np.random.rand(625) * -1000,
        cmap="viridis",
    )
    # plt.show()
    assert isinstance(ep, PathCollection)
