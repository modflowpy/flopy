import os
import flopy
import platform
import shutil
import numpy as np


mpth = os.path.join('..', 'examples', 'data', 'ag_test')
opth = os.path.join('temp', 't070')
if not os.path.exists(opth):
    os.makedirs(opth)

tabfiles_high = ['seg1_high.tab', 'seg9.tab']

nwt_exe_name = 'mfnwt'
if platform.system().lower() == "windows":
    nwt_exe_name = "mfnwt.exe"

nwt_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       nwt_exe_name)


def test_empty_ag_package():
    ml = flopy.modflow.Modflow("agtest", version='mfnwt')
    ag = flopy.modflow.ModflowAg(ml)

    if not isinstance(ag, flopy.pakbase.Package):
        raise Exception


def test_load_run_agwater():
    nam = "Agwater1_high.nam"
    ml = flopy.modflow.Modflow.load(nam, exe_name=nwt_exe, model_ws=mpth)

    loaded = False
    for pak in ml.packagelist:
        if isinstance(pak, flopy.modflow.ModflowAg):
            loaded = True
            break

    if not loaded:
        raise AssertionError("ModflowAg package not loaded")

    # remove this after new release of MODFLOW-NWT
    if platform.system().lower() == "windows":
        ml.change_model_ws(opth)
        ext_fnames = []
        for ef in ml.external_fnames:
            ext_fnames.append(os.path.split(ef)[-1])
        ml.external_fnames = ext_fnames
        ml.write_input()

        for f in tabfiles_high:
            shutil.copyfile(os.path.join(mpth, "input", f),
                            os.path.join(opth, f))

        ml2 = flopy.modflow.Modflow.load(nam, exe_name=nwt_exe, model_ws=opth)
        success, _ = ml2.run_model()

        if not success:
            raise AssertionError("Model did not run properly")

        fhd1 = flopy.utils.FormattedHeadFile(os.path.join(mpth, "output",
                                                          "Agwater1a_high.hed"))
        head1 = fhd1.get_alldata()

        fhd2 = flopy.utils.FormattedHeadFile(os.path.join(opth,
                                                          "Agwater1a_high.hed"))
        head2 = fhd2.get_alldata()

        success = np.allclose(head1, head2)
        if not success:
            raise AssertionError("Head arrays are out of defined tolerance")


if __name__ == "__main__":
    test_empty_ag_package()
    test_load_run_agwater()
