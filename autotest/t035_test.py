"""
Test the lgr model
"""
import os
import shutil
import sys

import numpy as np
import pymake
from ci_framework import FlopyTestSetup, base_test_dir

import flopy

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

exe_name = "mflgr"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False
# fix for intermittent CI failure on windows
else:
    if sys.platform.lower() in ("win32", "darwin"):
        run = False


def test_simplelgr_load_and_write(silent=True):
    # Test load and write of distributed MODFLOW-LGR example problem
    model_ws = f"{base_dir}_test_simplelgr_load_and_write"
    test_setup = FlopyTestSetup(
        verbose=True,
        test_dirs=model_ws,
    )

    pth = os.path.join("..", "examples", "data", "mflgr_v2", "ex3")

    # copy the original files
    files = os.listdir(pth)
    for file in files:
        src = os.path.join(pth, file)
        dst = os.path.join(model_ws, file)
        shutil.copyfile(src, dst)

    # load the lgr model
    lgr = flopy.modflowlgr.ModflowLgr.load(
        "ex3.lgr", verbose=True, model_ws=model_ws, exe_name=exe_name
    )

    # get the namefiles of the parent and child
    namefiles = lgr.get_namefiles()
    msg = f"get_namefiles returned {len(namefiles)} items instead of 2"
    assert len(namefiles) == 2, msg

    tpth = os.path.dirname(namefiles[0])
    assert tpth == model_ws, f"dir path is {tpth} not {model_ws}"

    # run the lgr model
    if run:
        success, buff = lgr.run_model(silent=silent)
        assert success, "could not run original modflow-lgr model"

    # check that a parent and child were read
    msg = "modflow-lgr ex3 does not have 2 grids"
    assert lgr.ngrids == 2, msg

    model_ws2 = os.path.join(model_ws, "new")
    lgr.change_model_ws(new_pth=model_ws2, reset_external=True)

    # get the namefiles of the parent and child
    namefiles = lgr.get_namefiles()
    assert (
        len(namefiles) == 2
    ), f"get_namefiles returned {len(namefiles)} items instead of 2"

    tpth = os.path.dirname(namefiles[0])
    assert tpth == model_ws2, f"dir path is {tpth} not {model_ws2}"

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    if run:
        success, buff = lgr.run_model(silent=silent)
        assert success, "could not run new modflow-lgr model"

        # compare parent results
        print("compare parent results")
        pth0 = os.path.join(model_ws, "ex3_parent.nam")
        pth1 = os.path.join(model_ws2, "ex3_parent.nam")
        success = pymake.compare_heads(pth0, pth1)
        assert success, "parent heads do not match"

        # compare child results
        print("compare child results")
        pth0 = os.path.join(model_ws, "ex3_child.nam")
        pth1 = os.path.join(model_ws2, "ex3_child.nam")
        success = pymake.compare_heads(pth0, pth1)
        assert success, "child heads do not match"


def singleModel(
    iChild,
    modelname,
    Lx,
    Ly,
    nlay,
    nrow,
    ncol,
    delr,
    delc,
    botm,
    hkPerLayer,
    vkaPerLayer,
    laytyp,
    ssPerLayer,
    nper,
    perlen,
    tsmult,
    nstp,
    steady,
    xul,
    yul,
    proj4_str,
    mfExe,
    rundir=".",
    welInfo=[],
    startingHead=0.0,
    lRunSingle=False,
):

    if iChild > 0:
        print(f"child model {modelname}")
        iLUoffset = 100 * int(iChild)
        print(f"increase Unit Numbers by {iLUoffset}")
    else:
        print(f"parent model {modelname}")
        iLUoffset = 0
    if steady:
        nper = 1
        perlen = 1
        nstp = [1]

    # Assign name and create modflow model object
    mf = flopy.modflow.Modflow(
        modelname, exe_name=mfExe, listunit=2 + iLUoffset, model_ws=rundir
    )

    # Create the discretization object
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=botm[0],
        botm=botm[1:],
        nper=nper,
        perlen=perlen,
        tsmult=1.07,
        nstp=nstp,
        steady=steady,
        itmuni=4,
        lenuni=2,
        unitnumber=11 + iLUoffset,
        xul=xul,
        yul=yul,
        proj4_str=proj4_str,
        start_datetime="28/2/2019",
    )

    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    if iChild > 0:
        iBndBnd = 59  # code for child cell to be linked to parent; value assigned to ibflg in the LGR-data
    else:
        iBndBnd = -1
    ibound[:, 0, :] = iBndBnd
    ibound[:, -1, :] = iBndBnd
    ibound[:, :, 0] = iBndBnd
    ibound[:, :, -1] = iBndBnd

    strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * startingHead

    bas = flopy.modflow.ModflowBas(
        mf, ibound=ibound, strt=strt, unitnumber=13 + iLUoffset
    )

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(
        mf,
        hk=hkPerLayer,
        vka=vkaPerLayer,
        ss=ssPerLayer,
        ipakcb=53 + iLUoffset,
        unitnumber=15 + iLUoffset,
    )

    # add WEL package to the MODFLOW model
    if len(welInfo) > 0:
        wel_sp = []
        for welData in welInfo:
            # get data for current well
            welLay = welData[0]
            welX = welData[1]
            welY = welData[2]
            welQ = welData[3]
            # calculate row and column for current well in grid
            welRow = int((yul - welY) / delc)  # check this calculation !!!
            welCol = int((welX - xul) / delr)  # check this calculation !!!
            if welRow < nrow and welRow >= 0 and welCol < ncol and welCol >= 0:
                # add well package data for well
                wel_sp.append([welLay, welRow, welCol, welQ])
        if len(wel_sp) > 0:
            stress_period_data = {0: wel_sp}
            wel = flopy.modflow.ModflowWel(
                mf,
                stress_period_data=stress_period_data,
                unitnumber=20 + iLUoffset,
            )

    # Add OC package to the MODFLOW model
    spd = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            spd[(kper, kstp)] = ["save head", "save budget"]
    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data=spd,
        compact=True,
        extension=["oc", "hds", "cbc"],
        unitnumber=[14 + iLUoffset, 51 + iLUoffset, 53 + iLUoffset],
    )

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf, unitnumber=27 + iLUoffset)

    if lRunSingle:
        # Write the MODFLOW model input files
        mf.write_input()

        # Run the MODFLOW model
        if run:
            success, buff = mf.run_model()
            if success:
                print(modelname, " ran successfully")
            else:
                print("problem running ", modelname)

    return mf


def test_simple_lgrmodel_from_scratch(silent=True):
    model_ws = f"{base_dir}_test_simple_lgrmodel_from_scratch"
    test_setup = FlopyTestSetup(
        verbose=True,
        test_dirs=model_ws,
    )

    # coordinates and extend Mother
    Lx_m = 1500.0
    Ly_m = 2500.0
    nrow_m = 25
    ncol_m = 15
    delr_m = Lx_m / ncol_m
    delc_m = Ly_m / nrow_m
    xul_m = 50550
    yul_m = 418266

    # Child Model domain and grid definition
    modelname = "child0"  # steady steate version of 'T_PW_50cm'
    Lx = 300.0
    Ly = 300.0
    ncpp = 10  # number of child cells per parent cell
    nrow = int(Ly * float(ncpp) / float(delc_m))
    ncol = int(Lx * float(ncpp) / float(delr_m))
    delr = Lx / ncol
    delc = Ly / nrow
    botm = [0.0, -15.0, -20.0, -40.0]
    hkPerLayer = [1.0, 0.0015, 15.0]
    ssPerLayer = [0.1, 0.001, 0.001]
    nlay = len(hkPerLayer)
    ilayW = 2
    laytyp = 0
    xul_c = 50985.00
    yul_c = 416791.06
    proj4_str = "EPSG:28992"
    nper = 1
    at = 42
    perlen = [at]
    ats = 100
    nstp = [ats]
    tsmult = 1.07
    steady = True
    rundir = model_ws
    lgrExe = exe_name

    # wel data
    pumping_rate = -720
    infiltration_rate = 360
    welInfo = [
        [ilayW, 51135.0, 416641.0, pumping_rate],
        [ilayW, 51059.0, 416750.0, infiltration_rate],
        [ilayW, 51170.0, 416560.0, 0.0],
        [ilayW, 51012.0, 416693.0, infiltration_rate],
        [ilayW, 51220.0, 416628.0, 0.0],
    ]

    child = singleModel(
        1,
        modelname,
        Lx,
        Ly,
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        botm,
        hkPerLayer,
        hkPerLayer,
        laytyp,
        ssPerLayer,
        nper,
        perlen,
        tsmult,
        nstp,
        steady,
        xul_c,
        yul_c,
        proj4_str,
        exe_name,
        rundir=model_ws,
        welInfo=welInfo,
        startingHead=-2.0,
    )

    modelname = "mother0"
    mother = singleModel(
        0,
        modelname,
        Lx_m,
        Ly_m,
        nlay,
        nrow_m,
        ncol_m,
        delr_m,
        delc_m,
        botm,
        hkPerLayer,
        hkPerLayer,
        laytyp,
        ssPerLayer,
        nper,
        perlen,
        tsmult,
        nstp,
        steady,
        xul_m,
        yul_m,
        proj4_str,
        exe_name,
        rundir=model_ws,
        welInfo=welInfo,
        startingHead=-2.0,
    )

    # setup LGR
    nprbeg = int((yul_m - yul_c) / delc_m)
    npcbeg = int((xul_c - xul_m) / delr_m)
    nprend = int(nrow / ncpp + nprbeg - 1)
    npcend = int(ncol / ncpp + npcbeg - 1)

    childData = [
        flopy.modflowlgr.mflgr.LgrChild(
            ishflg=1,
            ibflg=59,
            iucbhsv=80,
            iucbfsv=81,
            mxlgriter=20,
            ioutlgr=1,
            relaxh=0.4,
            relaxf=0.4,
            hcloselgr=5e-3,
            fcloselgr=5e-2,
            nplbeg=0,
            nprbeg=nprbeg,
            npcbeg=npcbeg,
            nplend=nlay - 1,
            nprend=nprend,
            npcend=npcend,
            ncpp=ncpp,
            ncppl=1,
        )
    ]

    lgrModel = flopy.modflowlgr.mflgr.ModflowLgr(
        modelname="PS1",
        exe_name=lgrExe,
        iupbhsv=82,
        iupbfsv=83,
        parent=mother,
        children=[child],
        children_data=childData,
        model_ws=model_ws,
        external_path=None,
        verbose=False,
    )

    # write LGR-files
    lgrModel.write_input()

    # run LGR
    if run:
        success, buff = lgrModel.run_model(silent=silent)
        assert success

    return


if __name__ == "__main__":
    test_simplelgr_load_and_write(silent=False)
    test_simple_lgrmodel_from_scratch(silent=False)
