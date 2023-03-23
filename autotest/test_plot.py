import os

import numpy as np
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


def test_map_view():
    m = flopy.modflow.Modflow(rotation=20.0)
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=40, ncol=20, delr=250.0, delc=250.0, top=10, botm=0
    )
    # transformation assigned by arguments
    xll, yll, rotation = 500000.0, 2934000.0, 45.0

    def check_vertices():
        xllp, yllp = pc._paths[0].vertices[0]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6

    m.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)
    modelmap = flopy.plot.PlotMapView(model=m)
    pc = modelmap.plot_grid()
    check_vertices()

    modelmap = flopy.plot.PlotMapView(modelgrid=m.modelgrid)
    pc = modelmap.plot_grid()
    check_vertices()

    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay=1,
        nrow=10,
        ncol=20,
        delr=1.0,
        delc=1.0,
    )
    xul, yul = 100.0, 210.0
    mg = mf.modelgrid
    mf.modelgrid.set_coord_info(
        xoff=mg._xul_to_xll(xul, 0.0), yoff=mg._yul_to_yll(yul, 0.0)
    )
    verts = [[101.0, 201.0], [119.0, 209.0]]
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={"line": verts})
    patchcollection = modelxsect.plot_grid()


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_gwfs_disv(example_data_path):
    mpath = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")
    ml6.modelgrid.set_coord_info(angrot=-14)
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("CHD")
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_lake2tr(example_data_path):
    mpath = example_data_path / "mf6" / "test045_lake2tr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("lakeex2a")
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("LAK")
    mapview.plot_bc("SFR")

    ax = mapview.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_2models_mvr(example_data_path):
    mpath = example_data_path / "mf6" / "test006_2models_mvr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("parent")
    ml6c = sim.get_model("child")
    ml6c.modelgrid.set_coord_info(xoff=700, yoff=0, angrot=0)

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("MAW")
    ax = mapview.ax

    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    mapview2 = flopy.plot.PlotMapView(model=ml6c, ax=mapview.ax)
    mapview2.plot_bc("MAW")
    ax = mapview2.ax

    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_UZF_3lay(example_data_path):
    mpath = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("UZF")
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(
    reason="sometimes get LineCollections instead of PatchCollections"
)
def test_cross_section_bc_gwfs_disv(example_data_path):
    mpath = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")
    xc = flopy.plot.PlotCrossSection(ml6, line={"line": ([0, 5.5], [10, 5.5])})
    xc.plot_bc("CHD")
    ax = xc.ax

    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(
            col, PatchCollection
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(
    reason="sometimes get LineCollections instead of PatchCollections"
)
def test_cross_section_bc_lake2tr(example_data_path):
    mpath = example_data_path / "mf6" / "test045_lake2tr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("lakeex2a")
    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 10})
    xc.plot_bc("LAK")
    xc.plot_bc("SFR")

    ax = xc.ax
    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(
            col, PatchCollection
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(
    reason="sometimes get LineCollections instead of PatchCollections"
)
def test_cross_section_bc_2models_mvr(example_data_path):
    mpath = example_data_path / "mf6" / "test006_2models_mvr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("parent")
    xc = flopy.plot.PlotCrossSection(ml6, line={"column": 1})
    xc.plot_bc("MAW")

    ax = xc.ax
    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(
            col, PatchCollection
        ), f"Unexpected collection type: {type(col)}"


@pytest.mark.mf6
@pytest.mark.xfail(
    reason="sometimes get LineCollections instead of PatchCollections"
)
def test_cross_section_bc_UZF_3lay(example_data_path):
    mpath = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")

    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 0})
    xc.plot_bc("UZF")

    ax = xc.ax
    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(
            col, PatchCollection
        ), f"Unexpected collection type: {type(col)}"


def test_map_view_contour(function_tmpdir):
    arr = np.random.rand(10, 10) * 100
    arr[-1, :] = np.nan
    delc = np.array([10] * 10, dtype=float)
    delr = np.array([8] * 10, dtype=float)
    top = np.ones((10, 10), dtype=float)
    botm = np.ones((3, 10, 10), dtype=float)
    botm[0] = 0.75
    botm[1] = 0.5
    botm[2] = 0.25
    idomain = np.ones((3, 10, 10))
    idomain[0, 0, :] = 0
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    levels = np.linspace(vmin, vmax, 7)

    grid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
        lenuni=1,
        nlay=3,
        nrow=10,
        ncol=10,
    )

    pmv = PlotMapView(modelgrid=grid, layer=0)
    contours = pmv.contour_array(a=arr)
    plt.savefig(function_tmpdir / "map_view_contour.png")

    # if we ever revert from standard contours to tricontours, restore this nan check
    # for ix, lev in enumerate(contours.levels):
    #     if not np.allclose(lev, levels[ix]):
    #         raise AssertionError("TriContour NaN catch Failed")


@pytest.mark.mf6
def test_vertex_model_dot_plot(example_data_path):
    rcParams["figure.max_open_warning"] = 36

    # load up the vertex example problem
    sim = MFSimulation.load(
        sim_ws=example_data_path / "mf6" / "test003_gwftri_disv"
    )
    disv_ml = sim.get_model("gwf_1")
    ax = disv_ml.plot()
    assert isinstance(ax, list)
    assert len(ax) == 36


# occasional _tkinter.TclError: Can't find a usable tk.tcl (or init.tcl)
# similar: https://github.com/microsoft/azure-pipelines-tasks/issues/16426
@flaky
def test_model_dot_plot(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = flopy.modflow.Modflow.load(
        "ibs2k.nam", "mf2k", model_ws=loadpth, check=False
    )
    ax = ml.plot()
    assert isinstance(ax, list), "ml.plot() ax is is not a list"
    assert len(ax) == 18, f"number of axes ({len(ax)}) is not equal to 18"


def test_dataset_dot_plot(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = flopy.modflow.Modflow.load(
        "ibs2k.nam", "mf2k", model_ws=loadpth, check=False
    )

    # plot specific dataset
    ax = ml.bcf6.hy.plot()
    assert isinstance(ax, list), "ml.bcf6.hy.plot() ax is is not a list"
    assert len(ax) == 2, f"number of hy axes ({len(ax)}) is not equal to 2"


def test_dataset_dot_plot_nlay_ne_plottable(
    function_tmpdir, example_data_path
):
    import matplotlib.pyplot as plt

    loadpth = example_data_path / "mf2005_test"
    ml = flopy.modflow.Modflow.load(
        "ibs2k.nam", "mf2k", model_ws=loadpth, check=False
    )
    # special case where nlay != plottable
    ax = ml.bcf6.vcont.plot()
    assert isinstance(
        ax, plt.Axes
    ), "ml.bcf6.vcont.plot() ax is is not of type plt.Axes"


def test_model_dot_plot_export(function_tmpdir, example_data_path):
    loadpth = example_data_path / "mf2005_test"
    ml = flopy.modflow.Modflow.load(
        "ibs2k.nam", "mf2k", model_ws=loadpth, check=False
    )

    fh = os.path.join(function_tmpdir, "ibs2k")
    ml.plot(mflay=0, filename_base=fh, file_extension="png")
    files = [f for f in os.listdir(function_tmpdir) if f.endswith(".png")]
    if len(files) < 10:
        raise AssertionError(
            "ml.plot did not properly export all supported data types"
        )

    for f in files:
        t = f.split("_")
        if len(t) < 3:
            raise AssertionError("Plot filenames not written correctly")


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


@requires_pkg("pandas")
@requires_exe("mf2005", "mp6")
def test_xc_plot_particle_pathlines(modpath_model):
    ml, mp, sim = modpath_model

    mp.write_input()
    mp.run_model(silent=False)

    pthobj = PathlineFile(os.path.join(mp.model_ws, "ex6.mppth"))
    well_pathlines = pthobj.get_destination_pathline_data(
        dest_cells=[(4, 12, 12)]
    )

    mx = PlotCrossSection(model=ml, line={"row": 4})
    mx.plot_bc("WEL", kper=2, color="blue")
    pth = mx.plot_pathline(well_pathlines, method="cell", colors="red")

    assert isinstance(pth, LineCollection)
    assert len(pth._paths) == 6


@requires_pkg("pandas")
@requires_exe("mf2005", "mp6")
def test_map_plot_particle_endpoints(modpath_model):
    ml, mp, sim = modpath_model
    mp.write_input()
    mp.run_model(silent=False)

    pthobj = EndpointFile(os.path.join(mp.model_ws, "ex6.mpend"))
    endpts = pthobj.get_alldata()

    # color kwarg as scalar
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
    hf = HeadFile(
        os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.hds")
    )
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
    hf = HeadFile(
        os.path.join(quasi3d_model.model_ws, f"{quasi3d_model.name}.hds")
    )
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


def structured_square_grid(side: int = 10, thick: int = 10):
    """
    Creates a basic 1-layer structured grid with the given thickness and number of cells per side
    Parameters
    ----------
    side : The number of cells per side
    thick : The thickness of the grid's single layer
    Returns
    -------
    A single-layer StructuredGrid of the given size and thickness
    """

    from flopy.discretization.structuredgrid import StructuredGrid

    delr = np.ones(side)
    delc = np.ones(side)
    top = np.ones((side, side)) * thick
    botm = np.ones((side, side)) * (top - thick).reshape(1, side, side)
    return StructuredGrid(delr=delr, delc=delc, top=top, botm=botm)


@pytest.mark.parametrize(
    "line",
    [(), [], (()), [[]], (0, 0), [0, 0], [[0, 0]]],
)
def test_cross_section_invalid_lines_raise_error(line):
    grid = structured_square_grid(side=10)
    with pytest.raises(ValueError):
        flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "line",
    [
        # diagonal
        [(0, 0), (10, 10)],
        ([0, 0], [10, 10]),
        # horizontal
        ([0, 5.5], [10, 5.5]),
        [(0, 5.5), (10, 5.5)],
        # vertical
        [(5.5, 0), (5.5, 10)],
        ([5.5, 0], [5.5, 10]),
        # multiple segments
        [(0, 0), (4, 6), (10, 10)],
        ([0, 0], [4, 6], [10, 10]),
    ],
)
def test_cross_section_valid_line_representations(line):
    from shapely.geometry import LineString as SLS

    from flopy.utils.geometry import LineString as FLS

    grid = structured_square_grid(side=10)

    fls = FLS(line)
    sls = SLS(line)

    # use raw, flopy.utils.geometry and shapely.geometry representations
    lxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})
    fxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": fls})
    sxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": sls})

    # make sure parsed points are identical for all line representations
    assert np.allclose(lxc.pts, fxc.pts) and np.allclose(lxc.pts, sxc.pts)
    assert (
        set(lxc.xypts.keys()) == set(fxc.xypts.keys()) == set(sxc.xypts.keys())
    )
    for k in lxc.xypts.keys():
        assert np.allclose(lxc.xypts[k], fxc.xypts[k]) and np.allclose(
            lxc.xypts[k], sxc.xypts[k]
        )


@pytest.mark.parametrize(
    "line",
    [
        0,
        [0],
        [0, 0],
        (0, 0),
        [(0, 0)],
        ([0, 0]),
    ],
)
@requires_pkg("shapely", "geojson")
def test_cross_section_invalid_line_representations_fail(line):
    grid = structured_square_grid(side=10)
    with pytest.raises(ValueError):
        flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})
