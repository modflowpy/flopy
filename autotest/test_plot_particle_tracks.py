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


# MP7


simname = "test_plot"
nlay = 1
nrow = 10
ncol = 10
top = 1.0
botm = [0.0]
nper = 1
perlen = 1.0
nstp = 1
tsmult = 1.0
porosity = 0.1


@pytest.fixture
def mf6_gwf_sim(module_tmpdir):
    gwfname = f"{simname}_gwf"

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=simname,
        exe_name="mf6",
        version="mf6",
        sim_ws=module_tmpdir,
    )

    # create tdis package
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)],
    )

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)

    # create gwf discretization
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # create gwf initial conditions package
    flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic")

    # create gwf node property flow package
    flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_saturation=True,
        save_specific_discharge=True,
    )

    # create gwf chd package
    spd = {
        0: [[(0, 0, 0), 1.0, 1.0], [(0, 9, 9), 0.0, 0.0]],
        1: [[(0, 0, 0), 0.0, 0.0], [(0, 9, 9), 1.0, 2.0]],
    }
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-1",
        stress_period_data=spd,
        auxiliary=["concentration"],
    )

    # create gwf output control package
    gwf_budget_file = f"{gwfname}.bud"
    gwf_head_file = f"{gwfname}.hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=gwf_budget_file,
        head_filerecord=gwf_head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # create iterative model solution for gwf model
    ims = flopy.mf6.ModflowIms(sim)

    return sim


@pytest.fixture
def mp7_sim(function_tmpdir, mf6_gwf_sim):
    pass


@pytest.mark.skip(reason="todo")
@requires_exe("mf6", "mp7")
def test_plot_map_view_mp7_pathline(mp7_sim):
    pass


@pytest.mark.skip(reason="todo")
@requires_exe("mf6", "mp7")
def test_plot_cross_section_mp7_pathline(mp7_sim):
    pass


# MF6 PRT


# @pytest.fixture
# def mf6_prt_sim(function_tmpdir, mf6_gwf_sim):
#     prtname = f"{simname}_prt"
#
#     # create prt model
#     prt = flopy.mf6.ModflowPrt(mf6_gwf_sim, modelname=prtname)
#
#     # create prt discretization
#     flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
#         prt,
#         pname="dis",
#         nlay=nlay,
#         nrow=nrow,
#         ncol=ncol,
#     )
#
#     # create mip package
#     flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)
#
#     # create prp package
#     flopy.mf6.ModflowPrtprp(
#         prt,
#         pname="prp1",
#         filename=f"{prtname}_1.prp",
#         nreleasepts=len(releasepts),
#         packagedata=releasepts,
#         perioddata={0: ["FIRST"]},
#     )
#
#     # create output control package
#     flopy.mf6.ModflowPrtoc(
#         prt,
#         pname="oc",
#         track_filerecord=[prt_track_file],
#         trackcsv_filerecord=[prt_track_csv_file],
#     )
#
#     # create a flow model interface
#     # todo Fienen's report (crash when FMI created but not needed)
#     # flopy.mf6.ModflowPrtfmi(
#     #     prt,
#     #     packagedata=[
#     #         ("GWFHEAD", gwf_head_file),
#     #         ("GWFBUDGET", gwf_budget_file),
#     #     ],
#     # )
#
#     # create exchange
#     flopy.mf6.ModflowGwfprt(
#         sim,
#         exgtype="GWF6-PRT6",
#         exgmnamea=gwfname,
#         exgmnameb=prtname,
#         filename=f"{gwfname}.gwfprt",
#     )
#
#     # add explicit model solution
#     ems = flopy.mf6.ModflowEms(
#         sim,
#         pname="ems",
#         filename=f"{prtname}.ems",
#     )
#     sim.register_solution_package(ems, [prt.name])
#
#     return sim
#
#
# @requires_exe("mf6")
# def test_plot_map_view_prt_pathline(mf6_prt_sim):
#     pass
#
#
# @requires_exe("mf6")
# def test_plot_cross_section_prt_pathline(mf6_prt_sim):
#     pass
