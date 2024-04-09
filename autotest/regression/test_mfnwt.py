import os

import pytest
from modflow_devtools.markers import requires_exe

from autotest.conftest import get_example_data_path
from flopy.modflow import Modflow, ModflowNwt, ModflowUpw
from flopy.utils import parsenamefile
from flopy.utils.compare import compare_budget, compare_heads


def get_nfnwt_namfiles():
    # build list of name files to try and load
    nwtpth = get_example_data_path() / "mf2005_test"
    namfiles = []
    m = Modflow("test", version="mfnwt")
    for namfile in nwtpth.rglob("*.nam"):
        nf = parsenamefile(namfile, m.mfnam_packages)
        lpf = False
        wel = False
        for key, value in nf.items():
            if "LPF" in value.filetype:
                lpf = True
            if "WEL" in value.filetype:
                wel = True
        if lpf and wel:
            namfiles.append(namfile)
    return namfiles


@requires_exe("mfnwt")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("namfile", get_nfnwt_namfiles())
def test_run_mfnwt_model(function_tmpdir, namfile):
    # load a MODFLOW-2005 model, convert to a MFNWT model,
    # write it back out, run the MFNWT model, load the MFNWT model,
    # and compare the results.

    load_ws, namfile = os.path.split(namfile)
    base_name = os.path.splitext(namfile)[0]

    # load MODFLOW-2005 models as MODFLOW-NWT models
    m = Modflow.load(
        namfile,
        model_ws=load_ws,
        version="mfnwt",
        verbose=True,
        check=False,
        exe_name="mfnwt",
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False

    # convert to MODFLOW-NWT model
    m.set_version("mfnwt")

    # extract data from existing flow package
    flowpaks = ["LPF"]
    for pak in m.get_package_list():
        if pak == "LPF":
            lpf = m.get_package(pak)
            layavg = lpf.layavg
            laytyp = lpf.laytyp
            layvka = lpf.layvka
            ss = lpf.ss
            sy = lpf.sy
            hk = lpf.hk
            vka = lpf.vka
            hani = lpf.hani
            chani = lpf.chani
            ipakcb = lpf.ipakcb
            unitnumber = lpf.unit_number[0]
            # remove existing package
            m.remove_package(pak)
            break

    # create UPW file from existing flow package
    upw = ModflowUpw(
        m,
        layavg=layavg,
        laytyp=laytyp,
        ipakcb=ipakcb,
        unitnumber=unitnumber,
        layvka=layvka,
        hani=hani,
        chani=chani,
        hk=hk,
        vka=vka,
        ss=ss,
        sy=sy,
    )

    # remove the existing solver
    solvers = ["SIP", "PCG", "PCGN", "GMG", "DE4"]
    for pak in m.get_package_list():
        solv = m.get_package(pak)
        if pak in solvers:
            unitnumber = solv.unit_number[0]
            m.remove_package(pak)
    nwt = ModflowNwt(m, unitnumber=unitnumber)

    # add specify option to the well package
    wel = m.get_package("WEL")
    wel.specify = True
    wel.phiramp = 1.0e-5
    wel.iunitramp = 2

    # change workspace and write MODFLOW-NWT model
    m.change_model_ws(function_tmpdir)
    m.write_input()
    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = function_tmpdir / namfile

    # reload the model just written
    m = Modflow.load(
        namfile,
        model_ws=function_tmpdir,
        version="mfnwt",
        verbose=True,
        check=False,
        exe_name="mfnwt",
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False

    # change workspace and write MODFLOW-NWT model
    pthf = function_tmpdir / "flopy"
    m.change_model_ws(pthf)
    m.write_input()
    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn1 = os.path.join(pthf, namfile)

    fsum = function_tmpdir / f"{base_name}.head.out"
    assert compare_heads(fn0, fn1, outfile=fsum), "head comparison failure"

    fsum = function_tmpdir / f"{base_name}.budget.out"
    assert compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    ), "budget comparison failure"
