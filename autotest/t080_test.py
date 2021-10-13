"""
HeadUFile get_ts tests using t505_test.py
"""

import os
import sys
import platform
import shutil
import numpy as np

try:
    from shapely.geometry import Polygon
except ImportWarning as e:
    print("Shapely not installed, tests cannot be run.")
    Polygon = None


import flopy
from flopy.utils.gridgen import Gridgen

# Set gridgen executable
gridgen_exe = "gridgen"
if platform.system() in "Windows":
    gridgen_exe += ".exe"
gridgen_exe = flopy.which(gridgen_exe)

# set mfusg executable
mfusg_exe = "mfusg"
if platform.system() in "Windows":
    mfusg_exe += ".exe"
mfusg_exe = flopy.which(mfusg_exe)

# set up the example folder
tpth = os.path.join("temp", "t420")
if not os.path.isdir(tpth):
    os.makedirs(tpth)

# set up a gridgen workspace
gridgen_ws = os.path.join(tpth, "gridgen_t420")
if not os.path.exists(gridgen_ws):
    os.makedirs(gridgen_ws)


def test_mfusg():

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
    m = flopy.modflow.Modflow(modelname=name, model_ws=gridgen_ws)
    dis = flopy.modflow.ModflowDis(
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
    g = Gridgen(dis, model_ws=gridgen_ws)
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
    ws = os.path.join(tpth, "gridgen_mfusg")
    name = "mymodel"
    m = flopy.mfusg.MfUsg(
        modelname=name,
        model_ws=ws,
        exe_name=mfusg_exe,
        structured=False,
    )
    disu = flopy.mfusg.MfUsgDisU(m, **gridprops)
    bas = flopy.modflow.ModflowBas(m)
    lpf = flopy.mfusg.MfUsgLpf(m)
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
    sms = flopy.mfusg.MfUsgSms(m)
    oc = flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
    m.write_input()

    # MODFLOW-USG does not have vertices, so we need to create
    # and unstructured grid and then assign it to the model. This
    # will allow plotting and other features to work properly.
    gridprops_ug = g.get_gridprops_unstructuredgrid()
    ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug, angrot=-15)
    m.modelgrid = ugrid

    if mfusg_exe is not None:
        m.run_model()

        # head is returned as a list of head arrays for each layer
        head_file = os.path.join(ws, f"{name}.hds")
        head = flopy.utils.HeadUFile(head_file).get_data()

        # test if single node idx works
        one_hds = flopy.utils.HeadUFile(head_file).get_ts(idx=300)
        if one_hds[0, 1] != head[0][300]:
            raise AssertionError(
                "Error head from 'get_ts' != head from 'get_data'"
            )

        # test if list of nodes for idx works
        nodes = [300, 182, 65]

        multi_hds = flopy.utils.HeadUFile(head_file).get_ts(idx=nodes)
        for i, node in enumerate(nodes):
            if multi_hds[0, i + 1] != head[0][node]:
                raise AssertionError(
                    "Error head from 'get_ts' != head from 'get_data'"
                )

    return


if __name__ == "__main__":
    test_mfusg()
