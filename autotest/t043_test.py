"""
Test the observation process load and write
"""
import os
import filecmp
import flopy
import pymake
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

exe_name = "mf2005"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_gage_load_and_write():
    """
    test043 load and write of MODFLOW-2005 GAGE example problem
    """
    model_ws = f"{base_dir}_test_gage_load_and_write"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    pth = os.path.join("..", "examples", "data", "mf2005_test")

    # copy the original files
    fpth = os.path.join(pth, "testsfr2_tab.nam")
    pymake.setup(fpth, model_ws)

    # load the modflow model
    mf = flopy.modflow.Modflow.load(
        "testsfr2_tab.nam", verbose=True, model_ws=model_ws, exe_name=exe_name
    )

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, "could not run original MODFLOW-2005 model"

        try:
            files = mf.gage.files
        except:
            raise ValueError("could not load original GAGE output files")

    model_ws2 = os.path.join(model_ws, "flopy")
    mf.change_model_ws(new_pth=model_ws2, reset_external=True)

    # write the modflow model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, "could not run new MODFLOW-2005 model"

        # compare the two results
        try:
            for f in files:
                pth0 = os.path.join(model_ws, f)
                pth1 = os.path.join(model_ws2, f)
                msg = f'new and original gage file "{f}" are not binary equal.'
                assert filecmp.cmp(pth0, pth1), msg
        except:
            raise ValueError("could not load new GAGE output files")


if __name__ == "__main__":
    test_gage_load_and_write()
