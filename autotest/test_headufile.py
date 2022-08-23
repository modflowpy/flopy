from pathlib import Path

import pytest
from autotest.conftest import requires_exe, requires_pkg

from flopy.discretization import UnstructuredGrid
from flopy.mfusg import MfUsg, MfUsgDisU, MfUsgLpf, MfUsgSms
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowChd,
    ModflowDis,
    ModflowOc,
)
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen
from flopy.utils.gridutil import get_lni


@pytest.fixture(scope="module")
def mfusg_model(module_tmpdir):
    from shapely.geometry import Polygon

    name = "dummy"
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.0
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # create dummy model and dis package for gridgen
    m = Modflow(modelname=name, model_ws=str(module_tmpdir))
    dis = ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(dis, model_ws=str(module_tmpdir))

    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, "polygon", 3, layers=[0])
    g.build()

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", 0)
        ic = ra["nodenumber"][0]
        chdspd.append([ic, head, head])

    # gridprops = g.get_gridprops()
    gridprops = g.get_gridprops_disu5()

    # create the mfusg modoel
    name = "mymodel"
    m = MfUsg(
        modelname=name,
        model_ws=str(module_tmpdir),
        exe_name="mfusg",
        structured=False,
    )
    disu = MfUsgDisU(m, **gridprops)
    bas = ModflowBas(m)
    lpf = MfUsgLpf(m)
    chd = ModflowChd(m, stress_period_data=chdspd)
    sms = MfUsgSms(m)
    oc = ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
    m.write_input()

    # MODFLOW-USG does not have vertices, so we need to create
    # and unstructured grid and then assign it to the model. This
    # will allow plotting and other features to work properly.
    gridprops_ug = g.get_gridprops_unstructuredgrid()
    ugrid = UnstructuredGrid(**gridprops_ug, angrot=-15)
    m.modelgrid = ugrid

    m.run_model()

    # head contains a list of head arrays for each layer
    head_file_path = Path(module_tmpdir / f"{name}.hds")
    return m, HeadUFile(str(head_file_path))


@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_get_ts_single_node(mfusg_model):
    model, head_file = mfusg_model
    head = head_file.get_data()

    # test if single node idx works
    one_hds = head_file.get_ts(idx=300)
    assert (
        one_hds[0, 1] == head[0][300]
    ), "head from 'get_ts' != head from 'get_data'"


@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_get_ts_multiple_nodes(mfusg_model):
    model, head_file = mfusg_model
    grid = model.modelgrid
    head = head_file.get_data()

    # test if list of nodes for idx works
    nodes = [500, 300, 182, 65]
    multi_hds = head_file.get_ts(idx=nodes)
    for i, node in enumerate(nodes):
        li, ni = get_lni(grid.ncpl, node)
        assert (
            multi_hds[0, i + 1] == head[li][ni]
        ), "head from 'get_ts' != head from 'get_data'"


@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_get_ts_all_nodes(mfusg_model):
    model, head_file = mfusg_model
    grid = model.modelgrid
    head = head_file.get_data()

    # test if list of nodes for idx works
    nodes = list(range(0, grid.nnodes))
    multi_hds = head_file.get_ts(idx=nodes)
    for node in nodes:
        li, ni = get_lni(grid.ncpl, node)
        assert (
            multi_hds[0, node + 1] == head[li][ni]
        ), "head from 'get_ts' != head from 'get_data'"


@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_get_lni(mfusg_model):
    # added to help reproduce https://github.com/modflowpy/flopy/issues/1503

    model, head_file = mfusg_model
    grid = model.modelgrid
    head = head_file.get_data()

    def get_expected():
        exp = dict()
        for l, ncpl in enumerate(list(grid.ncpl)):
            exp[l] = dict()
            for nn in range(ncpl):
                exp[l][nn] = head[l][nn]
        return exp

    nodes = list(range(0, grid.nnodes))
    expected = get_expected()
    for node in nodes:
        layer, nn = get_lni(grid.ncpl, node)
        assert expected[layer][nn] == head[layer][nn]
