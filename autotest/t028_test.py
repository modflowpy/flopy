import pytest
import os
import shutil
import flopy
from ci_framework import baseTestDir, flopyTest

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)

pthtest = os.path.join("..", "examples", "data", "swtv4_test")

swtv4_exe = "swtv4"
isswtv4 = flopy.which(swtv4_exe)
runmodel = False
verbose = False

swtdir = [
    "1_box",
    "1_box",
    "2_henry",
    "2_henry",
    "2_henry",
    "2_henry",
    "2_henry",
    "2_henry",
    "3_elder",
    "4_hydrocoin",
    "5_saltlake",
    "6_rotation",
    "6_rotation",
    "7_swtv4_ex",
    "7_swtv4_ex",
    "7_swtv4_ex",
    "7_swtv4_ex",
    "7_swtv4_ex",
    "7_swtv4_ex",
    "7_swtv4_ex",
]

subds = [
    "case1",
    "case2",
    "1_classic_case1",
    "2_classic_case2",
    "3_VDF_no_Trans",
    "4_VDF_uncpl_Trans",
    "5_VDF_DualD_Trans",
    "6_age_simulation",
    "",
    "",
    "",
    "1_symmetric",
    "2_asymmetric",
    "case1",
    "case2",
    "case3",
    "case4",
    "case5",
    "case6",
    "case7",
]


def test_seawat_array_format():
    testFramework = flopyTest(verbose=True)

    d = "2_henry"
    subds = ["1_classic_case1"]
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)

        model_ws = os.path.join(
            f"{baseDir}_test_seawat_array_format_{d}-{subd}"
        )
        testFramework.addTestDir(model_ws)

        namfile = "seawat.nam"
        if subd == "6_age_simulation":
            namfile = "henry_mod.nam"
        m = flopy.seawat.Seawat.load(namfile, model_ws=pth, verbose=verbose)
        m.change_model_ws(model_ws, reset_external=True)

        m.bcf6.hy[0].fmtin = "(BINARY)"
        m.btn.prsity[0].fmtin = "(BINARY)"
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, f"{m.name} did not run"
    return


@pytest.mark.parametrize(
    "d, subd",
    zip(swtdir, subds),
)
def test_swtv4(d, subd):
    run_swtv4(d, subd)
    return


def run_swtv4(d, subd):
    testFramework = flopyTest(verbose=True)
    model_ws = os.path.join(f"{baseDir}_test_swtv4_{d}-{subd}")
    testFramework.addTestDir(model_ws)

    # set up paths
    pth = os.path.join(pthtest, d, subd)

    namfile = "seawat.nam"
    if subd == "6_age_simulation":
        namfile = "henry_mod.nam"

    # load the existing model
    m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth, verbose=verbose)

    # change working directory
    m.change_model_ws(model_ws)

    # write input files
    m.write_input()

    # run the model
    if isswtv4 is not None and runmodel:
        success, buff = m.run_model(silent=False)
        assert success, f"{m.name} did not run"


if __name__ == "__main__":
    for d, subd in zip(swtdir, subds):
        run_swtv4(d, subd)

    test_seawat_array_format()
