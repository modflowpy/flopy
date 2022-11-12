import os
import re
from pathlib import Path

import numpy as np

import flopy


def test_import_and_version_string():
    import flopy

    # matches any 1-3 component, dot-separated version string
    # https://stackoverflow.com/a/82205/6514033
    pattern = r"^(\d+\.)?(\d+\.)?(\*|\d+)$"
    assert re.match(pattern, flopy.__version__)


def test_modflow():
    import flopy

    mf = flopy.modflow.Modflow()
    assert isinstance(mf, flopy.modflow.Modflow)
    assert not mf.has_package("DIS")  # not yet
    dis = flopy.modflow.ModflowDis(mf)
    assert mf.has_package("DIS")
    assert mf.has_package("dis")  # case-insensitive
    assert not mf.has_package("DISU")  # not here
    assert isinstance(dis, flopy.modflow.ModflowDis)
    bas = flopy.modflow.ModflowBas(mf)
    assert isinstance(bas, flopy.modflow.ModflowBas)
    lpf = flopy.modflow.ModflowLpf(mf)
    assert isinstance(lpf, flopy.modflow.ModflowLpf)
    wel = flopy.modflow.ModflowWel(mf)
    assert isinstance(wel, flopy.modflow.ModflowWel)
    oc = flopy.modflow.ModflowOc(mf)
    assert isinstance(oc, flopy.modflow.ModflowOc)
    pcg = flopy.modflow.ModflowPcg(mf)
    assert isinstance(pcg, flopy.modflow.ModflowPcg)


def test_modflow_unstructured(function_tmpdir):
    import flopy

    mf = flopy.mfusg.MfUsg(structured=False, model_ws=str(function_tmpdir))
    assert isinstance(mf, flopy.mfusg.MfUsg)

    disu = flopy.mfusg.MfUsgDisU(
        mf, nodes=1, iac=[1], njag=1, ja=np.array([0]), cl12=[1.0], fahl=[1.0]
    )
    assert isinstance(disu, flopy.mfusg.MfUsgDisU)

    bas = flopy.modflow.ModflowBas(mf)
    assert isinstance(bas, flopy.modflow.ModflowBas)

    lpf = flopy.mfusg.MfUsgLpf(mf)
    assert isinstance(lpf, flopy.mfusg.MfUsgLpf)

    wel = flopy.mfusg.MfUsgWel(mf, stress_period_data={0: [[0, -100]]})
    assert isinstance(wel, flopy.mfusg.MfUsgWel)

    ghb = flopy.modflow.ModflowGhb(
        mf, stress_period_data={0: [[1, 5.9, 1000.0]]}
    )
    assert isinstance(ghb, flopy.modflow.ModflowGhb)

    oc = flopy.modflow.ModflowOc(mf)
    assert isinstance(oc, flopy.modflow.ModflowOc)

    sms = flopy.mfusg.MfUsgSms(mf)
    assert isinstance(sms, flopy.mfusg.MfUsgSms)

    # write well file
    wel.write_file()
    wel_path = Path(function_tmpdir / f"{mf.name}.wel")
    assert wel_path.is_file()
    wel2 = flopy.mfusg.MfUsgWel.load(str(wel_path), mf)
    assert wel2.stress_period_data[0] == wel.stress_period_data[0]

    # write ghb file
    ghb.write_file(check=False)
    ghb_path = Path(function_tmpdir / f"{mf.name}.ghb")
    assert ghb_path.is_file() is True
    ghb2 = flopy.modflow.ModflowGhb.load(str(ghb_path), mf)


def test_mflist_reference(function_tmpdir):
    # make the model
    ml = flopy.modflow.Modflow()
    assert isinstance(ml, flopy.modflow.Modflow)
    perlen = np.arange(1, 20, 1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow, ncol = 50, 40
    botm = np.arange(0, -100, -10)
    hk = np.random.random((nrow, ncol))
    dis = flopy.modflow.ModflowDis(
        ml,
        delr=100.0,
        delc=100.0,
        nrow=nrow,
        ncol=ncol,
        nlay=nlay,
        nper=perlen.shape[0],
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        top=10,
        botm=botm,
        steady=False,
    )
    assert isinstance(dis, flopy.modflow.ModflowDis)
    lpf = flopy.modflow.ModflowLpf(ml, hk=hk, vka=10.0, laytyp=1)
    assert isinstance(lpf, flopy.modflow.ModflowLpf)
    pcg = flopy.modflow.ModflowPcg(ml)
    assert isinstance(pcg, flopy.modflow.ModflowPcg)
    oc = flopy.modflow.ModflowOc(ml)
    assert isinstance(oc, flopy.modflow.ModflowOc)
    ibound = np.ones((nrow, ncol))
    ibound[:, 0] = -1
    ibound[25:30, 30:39] = 0
    bas = flopy.modflow.ModflowBas(ml, strt=5.0, ibound=ibound)
    assert isinstance(bas, flopy.modflow.ModflowBas)
    rch = flopy.modflow.ModflowRch(ml, rech={0: 0.00001, 5: 0.0001, 6: 0.0})
    assert isinstance(rch, flopy.modflow.ModflowRch)
    wel_dict = {}
    wel_data = [[9, 25, 20, -200], [0, 0, 0, -400], [5, 20, 32, 500]]
    wel_dict[0] = wel_data
    wel_data2 = [[45, 20, 200], [9, 49, 39, 400], [5, 20, 32, 500]]
    wel_dict[10] = wel_data2
    wel = flopy.modflow.ModflowWel(ml, stress_period_data={0: wel_data})
    assert isinstance(wel, flopy.modflow.ModflowWel)
    ghb_dict = {0: [1, 10, 10, 400, 300]}
    ghb = flopy.modflow.ModflowGhb(ml, stress_period_data=ghb_dict)
    assert isinstance(ghb, flopy.modflow.ModflowGhb)

    # TODO: test separately
    # shp_path = str(function_tmpdir / "test3.shp")
    # ml.export(shp_path, kper=0)
    # shp = shapefile.Reader(shp_path)
    # assert shp.numRecords == nrow * ncol


def test_pyinstaller_flopy_runs_without_dfn_folder(
    flopy_data_path, example_data_path
):
    """
    Test to ensure that flopy can load a modflow 6 simulation without dfn
    files being present.
    """

    dfn_path = flopy_data_path / "mf6" / "data" / "dfn"
    rename_path = flopy_data_path / "mf6" / "data" / "no-dfn"

    exists = dfn_path.exists()
    if exists:
        if rename_path.exists():
            os.rmdir(rename_path)
        os.rename(dfn_path, rename_path)
    try:
        # run built executable
        sim_path = example_data_path / "mf6" / "test006_gwf3"

        flopy.mf6.MFSimulation.load(sim_ws=str(sim_path))
    finally:
        if exists and rename_path.exists():
            os.rename(rename_path, dfn_path)
