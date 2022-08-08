"""
HeadUFile get_ts tests using t505_test.py
"""

import os

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


@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_mfusg(tmpdir):
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
    m = Modflow(modelname=name, model_ws=str(tmpdir))
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
    g = Gridgen(dis, model_ws=str(tmpdir))

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
        model_ws=str(tmpdir),
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

    # head is returned as a list of head arrays for each layer
    head_file = os.path.join(str(tmpdir), f"{name}.hds")
    head = HeadUFile(head_file).get_data()

    # test if single node idx works
    one_hds = HeadUFile(head_file).get_ts(idx=300)
    if one_hds[0, 1] != head[0][300]:
        raise AssertionError(
            "Error head from 'get_ts' != head from 'get_data'"
        )

    # test if list of nodes for idx works
    nodes = [300, 182, 65]

    multi_hds = HeadUFile(head_file).get_ts(idx=nodes)
    for i, node in enumerate(nodes):
        if multi_hds[0, i + 1] != head[0][node]:
            raise AssertionError(
                "Error head from 'get_ts' != head from 'get_data'"
            )


def test_usg_iverts():
    iverts = [
        [4, 3, 2, 1, 0, None],
        [7, 0, 1, 6, 5, None],
        [11, 10, 9, 8, 2, 3],
        [1, 6, 13, 12, 8, 2],
        [15, 14, 13, 6, 5, None],
        [10, 9, 18, 17, 16, None],
        [8, 12, 20, 19, 18, 9],
        [22, 14, 13, 12, 20, 21],
        [24, 17, 18, 19, 23, None],
        [21, 20, 19, 23, 25, None],
    ]
    verts = [
        [0.0, 22.5],
        [5.1072, 22.5],
        [7.5, 24.0324],
        [7.5, 30.0],
        [0.0, 30.0],
        [0.0, 7.5],
        [4.684, 7.5],
        [0.0, 15.0],
        [14.6582, 21.588],
        [22.5, 24.3766],
        [22.5, 30.0],
        [15.0, 30.0],
        [15.3597, 8.4135],
        [7.5, 5.6289],
        [7.5, 0.0],
        [0.0, 0.0],
        [30.0, 30.0],
        [30.0, 22.5],
        [25.3285, 22.5],
        [24.8977, 7.5],
        [22.5, 5.9676],
        [22.5, 0.0],
        [15.0, 0.0],
        [30.0, 7.5],
        [30.0, 15.0],
        [30.0, 0.0],
    ]

    grid = UnstructuredGrid(verts, iverts, ncpl=[len(iverts)])

    iverts = grid.iverts
    if any(None in l for l in iverts):
        raise ValueError("None type should not be returned in iverts list")
