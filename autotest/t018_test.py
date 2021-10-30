import os
import numpy as np
import flopy
from ci_framework import baseTestDir, flopyTest

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def test_tpl_constant():
    model_ws = f"{baseDir}_test_tpl_constant"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    # Define the model dimensions
    nlay = 3
    nrow = 20
    ncol = 20

    # Create the flopy model object and add the dis and lpf packages
    m = flopy.modflow.Modflow(modelname="tpl1", model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol)
    lpf = flopy.modflow.ModflowLpf(m, hk=10.0)

    mfpackage = "lpf"
    partype = "hk"
    parname = "HK_LAYER_1"
    idx = np.empty((nlay, nrow, ncol), dtype=bool)
    idx[0] = True
    idx[1:] = False

    # The span variable defines how the parameter spans the package
    span = {"idx": idx}

    # These parameters have not affect yet, but may in the future
    startvalue = 10.0
    lbound = 0.001
    ubound = 1000.0
    transform = "log"

    p = flopy.pest.Params(
        mfpackage, partype, parname, startvalue, lbound, ubound, span
    )

    tw = flopy.pest.templatewriter.TemplateWriter(m, [p])
    tw.write_template()

    tplfile = os.path.join(model_ws, "tpl1.lpf.tpl")
    assert os.path.isfile(tplfile)

    return


def test_tpl_layered():
    model_ws = f"{baseDir}_test_tpl_layered"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    nlay = 3
    nrow = 20
    ncol = 20

    # Create the flopy model object and add the dis and lpf packages
    m = flopy.modflow.Modflow(modelname="tpl2", model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol)
    lpf = flopy.modflow.ModflowLpf(m, hk=10.0)

    mfpackage = "lpf"
    partype = "hk"
    parname = "HK_LAYER_1-3"

    # Span indicates that the hk parameter applies as a multiplier to layers 0 and 2 (MODFLOW layers 1 and 3)
    span = {"layers": [0, 2]}

    # These parameters have not affect yet, but may in the future
    startvalue = 10.0
    lbound = 0.001
    ubound = 1000.0
    transform = "log"

    p = flopy.pest.Params(
        mfpackage, partype, parname, startvalue, lbound, ubound, span
    )
    tw = flopy.pest.templatewriter.TemplateWriter(m, [p])
    tw.write_template()

    tplfile = os.path.join(model_ws, "tpl2.lpf.tpl")
    assert os.path.isfile(tplfile)

    return


def test_tpl_zoned():
    model_ws = f"{baseDir}_test_tpl_zoned"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    nlay = 3
    nrow = 20
    ncol = 20

    # Create the flopy model object and add the dis and lpf packages
    m = flopy.modflow.Modflow(modelname="tpl3", model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol)
    lpf = flopy.modflow.ModflowLpf(m, hk=10.0)

    # Create a zone array
    zonearray = np.ones((nlay, nrow, ncol), dtype=int)
    zonearray[0, 10:, 7:] = 2
    zonearray[0, 15:, 9:] = 3
    zonearray[1] = 4

    # Create a list of parameters for HK
    mfpackage = "lpf"
    parzones = [2, 3, 4]
    parvals = [56.777, 78.999, 99.0]
    lbound = 5
    ubound = 500
    transform = "log"
    plisthk = flopy.pest.zonearray2params(
        mfpackage,
        "hk",
        parzones,
        lbound,
        ubound,
        parvals,
        transform,
        zonearray,
    )

    # Create a list of parameters for VKA
    parzones = [1, 2]
    parvals = [0.001, 0.0005]
    zonearray = np.ones((nlay, nrow, ncol), dtype=int)
    zonearray[1] = 2
    plistvk = flopy.pest.zonearray2params(
        mfpackage,
        "vka",
        parzones,
        lbound,
        ubound,
        parvals,
        transform,
        zonearray,
    )

    # Combine the HK and VKA parameters together
    plist = plisthk + plistvk

    # Write the template file
    tw = flopy.pest.templatewriter.TemplateWriter(m, plist)
    tw.write_template()

    tplfile = os.path.join(model_ws, "tpl3.lpf.tpl")
    assert os.path.isfile(tplfile)

    return


if __name__ == "__main__":
    test_tpl_constant()
    test_tpl_layered()
    test_tpl_zoned()
