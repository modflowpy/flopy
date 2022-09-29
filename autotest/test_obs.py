import os
import shutil

import numpy as np
import pytest
from autotest.conftest import requires_exe

from flopy.modflow import (
    HeadObservation,
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowDrn,
    ModflowFlwob,
    ModflowHob,
    ModflowLpf,
    ModflowPcg,
)

from flopy.utils.observationfile import Mf6Obs


@requires_exe("mf2005")
def test_hob_simple(tmpdir):
    """
    test041 create and run a simple MODFLOW-2005 OBS example
    """
    modelname = "hob_simple"
    ws = str(tmpdir)
    nlay, nrow, ncol = 1, 11, 11
    shape3d = (nlay, nrow, ncol)
    shape2d = (nrow, ncol)
    ib = np.ones(shape3d, dtype=int)
    ib[0, 0, 0] = -1
    m = Modflow(
        modelname=modelname,
        model_ws=ws,
        verbose=False,
        exe_name="mf2005",
    )
    dis = ModflowDis(m, nlay=1, nrow=11, ncol=11, nper=2, perlen=[1, 1])

    bas = ModflowBas(m, ibound=ib, strt=10.0)
    lpf = ModflowLpf(m)
    pcg = ModflowPcg(m)
    obs = HeadObservation(
        m,
        layer=0,
        row=5,
        column=5,
        time_series_data=[[1.0, 54.4], [2.0, 55.2]],
    )
    hob = ModflowHob(m, iuhobsv=51, hobdry=-9999.0, obs_data=[obs])

    # Write the model input files
    m.write_input()

    # run the modflow-2005 model
    success, buff = m.run_model(silent=False)
    assert success, "could not run simple MODFLOW-2005 model"

    print(
        "test041 load and run a simple MODFLOW-2005 OBS example with"
        " specified filenames"
    )
    modelname = "hob_simple"
    pkglst = ["dis", "bas6", "pcg", "lpf"]
    m = Modflow.load(
        f"{modelname}.nam",
        model_ws=ws,
        check=False,
        load_only=pkglst,
        verbose=False,
        exe_name="mf2005",
        forgive=False,
    )

    obs = HeadObservation(
        m,
        layer=0,
        row=5,
        column=5,
        time_series_data=[[1.0, 54.4], [2.0, 55.2]],
    )
    f_in = f"{modelname}_custom_fname.hob"
    f_out = f"{modelname}_custom_fname.hob.out"
    filenames = [f_in, f_out]
    hob = ModflowHob(
        m,
        iuhobsv=51,
        hobdry=-9999.0,
        obs_data=[obs],
        options=["NOPRINT"],
        filenames=filenames,
    )

    # Write the model input files
    m.write_input()

    s = f"output filename ({m.get_output(unit=51)}) does not match specified name"
    assert m.get_output(unit=51) == f_out, s
    s = "specified HOB input file not found"
    assert os.path.isfile(os.path.join(ws, f_in)), s

    # run the modflow-2005 model
    success, buff = m.run_model(silent=False)
    assert success, "could not run simple MODFLOW-2005 model"


@requires_exe("mf2005")
def test_obs_load_and_write(tmpdir, example_data_path):
    """
    test041 load and write of MODFLOW-2005 OBS example problem
    """

    pth = str(example_data_path / "mf2005_obs")
    ws = str(tmpdir)

    # copy the original files
    files = os.listdir(pth)
    for file in files:
        src = os.path.join(pth, file)
        dst = os.path.join(ws, file)
        shutil.copyfile(src, dst)

    # load the modflow model
    mf = Modflow.load(
        "tc1-true.nam", verbose=True, model_ws=ws, exe_name="mf2005"
    )

    # run the modflow-2005 model
    success, buff = mf.run_model(silent=False)
    assert success, "could not run original MODFLOW-2005 model"

    try:
        iu = mf.hob.iuhobsv
        fpth = mf.get_output(unit=iu)
        pth0 = os.path.join(ws, fpth)
        obs0 = np.genfromtxt(pth0, skip_header=1)
    except:
        raise ValueError("could not load original HOB output file")

    model_ws2 = os.path.join(ws, "flopy")
    mf.change_model_ws(new_pth=model_ws2, reset_external=True)

    # write the lgr model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    success, buff = mf.run_model(silent=False)
    assert success, "could not run new MODFLOW-2005 model"

    # compare parent results
    try:
        pth1 = os.path.join(model_ws2, fpth)
        obs1 = np.genfromtxt(pth1, skip_header=1)

        msg = "new simulated heads are not approximately equal"
        assert np.allclose(obs0[:, 0], obs1[:, 0], atol=1e-4), msg

        msg = "new observed heads are not approximately equal"
        assert np.allclose(obs0[:, 1], obs1[:, 1], atol=1e-4), msg
    except:
        raise ValueError("could not load new HOB output file")

    # load the modflow model
    m = Modflow.load(
        "tc1-true.nam", verbose=True, model_ws=ws, exe_name="mf2005"
    )

    model_ws2 = os.path.join(ws, "flwob")
    m.change_model_ws(new_pth=model_ws2, reset_external=True)

    # write the lgr model in to the new path
    m.write_input()

    # add DRN package
    spd = {0: [[0, 5, 5, 0.5, 8e6], [0, 8, 8, 0.7, 8e6]]}
    drn = ModflowDrn(m, 53, stress_period_data=spd)

    # flow observation

    # Lists of length nqfb
    nqobfb = [1, 1]
    nqclfb = [1, 1]

    # Lists of length nqtfb
    obsnam = ["drob_1", "drob_2"]
    irefsp = [0, 0]
    toffset = [0, 0]
    flwobs = [-5.678, -6.874]

    # Lists of length (nqfb, nqclfb)
    layer = [[0], [0]]
    row = [[5], [8]]
    column = [[5], [8]]
    factor = [[1.0], [1.0]]

    drob = ModflowFlwob(
        m,
        nqfb=len(nqclfb),
        nqcfb=np.sum(nqclfb),
        nqtfb=np.sum(nqobfb),
        nqobfb=nqobfb,
        nqclfb=nqclfb,
        obsnam=obsnam,
        irefsp=irefsp,
        toffset=toffset,
        flwobs=flwobs,
        layer=layer,
        row=row,
        column=column,
        factor=factor,
        flowtype="drn",
        options=["NOPRINT"],
    )
    # Write the model input files
    m.write_input(check=False)

    # Load the DROB package and compare it to the original
    pkglst = ["drob"]
    m = Modflow.load(
        "tc1-true.nam",
        model_ws=model_ws2,
        check=False,
        load_only=pkglst,
        verbose=False,
        exe_name="mf2005",
        forgive=False,
    )

    # check variables were read properly
    s = f"nqfb loaded from {m.drob.fn_path} read incorrectly"
    assert drob.nqfb == m.drob.nqfb, s
    s = f"nqcfb loaded from {m.drob.fn_path} read incorrectly"
    assert drob.nqcfb == m.drob.nqcfb, s
    s = f"nqtfb loaded from {m.drob.fn_path} read incorrectly"
    assert drob.nqtfb == m.drob.nqtfb, s
    s = f"obsnam loaded from {m.drob.fn_path} read incorrectly"
    assert list([n for n in drob.obsnam]) == list(
        [n for n in m.drob.obsnam]
    ), s
    s = f"flwobs loaded from {m.drob.fn_path} read incorrectly"
    assert np.array_equal(drob.flwobs, m.drob.flwobs), s
    s = f"layer loaded from {m.drob.fn_path} read incorrectly"
    assert np.array_equal(drob.layer, m.drob.layer), s
    s = f"row loaded from {m.drob.fn_path} read incorrectly"
    assert np.array_equal(drob.row, m.drob.row), s
    s = f"column loaded from {m.drob.fn_path} read incorrectly"
    assert np.array_equal(drob.column, m.drob.column), s


def test_obs_single_time(tmpdir):
    """
    test reading a mf6 observation file with a single time
    """

    pth = str(tmpdir / "single.csv")
    with open(pth, "w") as file:
        file.write("time,obs01,obs02\n1.0,10.0,20.0\n")

    obs = Mf6Obs(pth)
    data = obs.get_data()

    assert data.shape[0] == 1, "data shape != 1"
    assert data["totim"][0] == 1.0, "totim[0] != 1.0"
    assert data["obs01"][0] == 10.0, "obs01[0] != 10.0"
    assert data["obs02"][0] == 20.0, "obs02[0] != 20.0"


def test_obs_create_and_write(tmpdir, example_data_path):
    """
    test041 create and write of MODFLOW-2005 OBS example problem
    """

    pth = str(example_data_path / "mf2005_obs")
    ws = str(tmpdir)

    # copy the original files
    files = os.listdir(pth)
    for file in files:
        src = os.path.join(pth, file)
        dst = os.path.join(ws, file)
        shutil.copyfile(src, dst)

    # load the modflow model
    mf = Modflow.load(
        "tc1-true.nam",
        verbose=True,
        model_ws=ws,
        exe_name="mf2005",
        forgive=False,
    )
    # remove the existing hob package
    iuhob = mf.hob.unit_number[0]
    mf.remove_package("HOB")

    # create a new hob object
    obs_data = []

    # observation location 1
    tsd = [
        [1.0, 1.0],
        [87163.0, 2.0],
        [348649.0, 3.0],
        [871621.0, 4.0],
        [24439070.0, 5.0],
        [24439072.0, 6.0],
    ]
    names = ["o1.1", "o1.2", "o1.3", "o1.4", "o1.5", "o1.6"]
    obs_data.append(
        HeadObservation(
            mf,
            layer=0,
            row=2,
            column=0,
            time_series_data=tsd,
            names=names,
            obsname="o1",
        )
    )
    # observation location 2
    tsd = [
        [0.0, 126.938],
        [87163.0, 126.904],
        [871621.0, 126.382],
        [871718.5943, 115.357],
        [871893.7713, 112.782],
    ]
    names = ["o2.1", "o2.2", "o2.3", "o2.4", "o2.5"]
    obs_data.append(
        HeadObservation(
            mf,
            layer=0,
            row=3,
            column=3,
            time_series_data=tsd,
            names=names,
            obsname="o2",
        )
    )
    hob = ModflowHob(mf, iuhobsv=51, obs_data=obs_data, unitnumber=iuhob)
    # write the hob file
    hob.write_file()

    # run the modflow-2005 model
    success, buff = mf.run_model(silent=False)
    assert success, "could not run original MODFLOW-2005 model"

    try:
        iu = mf.hob.iuhobsv
        fpth = mf.get_output(unit=iu)
        pth0 = os.path.join(ws, fpth)
        obs0 = np.genfromtxt(pth0, skip_header=1)
    except:
        raise ValueError("could not load original HOB output file")

    model_ws2 = os.path.join(ws, "flopy")
    mf.change_model_ws(new_pth=model_ws2, reset_external=True)

    # write the model at the new path
    mf.write_input()

    # run the modflow-2005 model
    success, buff = mf.run_model(silent=False)
    assert success, "could not run new MODFLOW-2005 model"

    # compare parent results
    try:
        pth1 = os.path.join(model_ws2, fpth)
        obs1 = np.genfromtxt(pth1, skip_header=1)

        msg = "new simulated heads are not approximately equal"
        assert np.allclose(obs0[:, 0], obs1[:, 0], atol=1e-4), msg

        msg = "new observed heads are not approximately equal"
        assert np.allclose(obs0[:, 1], obs1[:, 1], atol=1e-4), msg
    except:
        raise ValueError("could not load new HOB output file")


def test_multilayerhob_pr():
    """
    test041 test multilayer obs PR == 1 criteria with problematic PRs
    """
    ml = Modflow()
    dis = ModflowDis(ml, nlay=3, nrow=1, ncol=1, nper=1, perlen=[1])
    HeadObservation(
        ml,
        layer=-3,
        row=0,
        column=0,
        time_series_data=[[1.0, 0]],
        mlay={0: 0.19, 1: 0.69, 2: 0.12},
    )


def test_multilayerhob_prfail():
    """
    test041 failure of multilayer obs PR == 1 criteria
    """
    ml = Modflow()
    dis = ModflowDis(ml, nlay=3, nrow=1, ncol=1, nper=1, perlen=[1])
    with pytest.raises(ValueError):
        HeadObservation(
            ml,
            layer=-3,
            row=0,
            column=0,
            time_series_data=[[1.0, 0]],
            mlay={0: 0.50, 1: 0.50, 2: 0.01},
        )


def test_multilayerhob_pr_multiline():
    from io import StringIO

    problem_hob = [
        "2 4 7",
        "1 1",
        "A19E1_1 -2 140 91 1 1 -0.28321 -0.05389"
        " 69 1 1 1  # A19E1 8/13/1975",
        "3 0.954",
        "4 0.046",
        "A19E1_2 -2 140 91 1 1 -0.28321 -0.05389"
        " 72 1 1 1  # A19E1 10/9/1975",
        "3 0.954",
        "4 0.046",
    ]

    problem_hob = "\n".join(problem_hob)
    ml = Modflow("hobtest")
    dis = ModflowDis(
        ml,
        nlay=4,
        nrow=200,
        ncol=200,
        nper=100,
        perlen=10,
        nstp=4,
        tsmult=1.0,
        steady=False,
    )
    hob = ModflowHob.load(StringIO(problem_hob), ml)

    assert len(hob.obs_data) == 2, "pr, mlay... load error"
