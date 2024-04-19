import os
import shutil
from io import StringIO

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe

from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowNwt,
    ModflowOc,
    ModflowSfr2,
    ModflowUpw,
    ModflowUzf1,
    ModflowWel,
)
from flopy.utils.optionblock import OptionBlock
from flopy.utils.util_array import Util2d


@pytest.fixture
def mf2005_test_path(example_data_path):
    return example_data_path / "mf2005_test"


@pytest.fixture
def uzf_test_path(example_data_path):
    return example_data_path / "uzf_examples"


@pytest.fixture
def options_path(example_data_path):
    return example_data_path / "options"


def test_create_uzf(function_tmpdir, mf2005_test_path, uzf_test_path):
    # copy the test files
    ws = function_tmpdir
    for f in mf2005_test_path.glob("UZFtest2.*"):
        shutil.copy(f, ws)

    # load the model
    m = Modflow.load(
        "UZFtest2.nam",
        version="mf2005",
        exe_name="mf2005",
        model_ws=ws,
        load_only=["ghb", "dis", "bas6", "oc", "sip", "lpf", "sfr"],
        verbose=True,
    )
    rm = [True if ".uz" in f else False for f in m.external_fnames]
    m.external_fnames = [
        f for i, f in enumerate(m.external_fnames) if not rm[i]
    ]
    m.external_binflag = [
        f for i, f in enumerate(m.external_binflag) if not rm[i]
    ]
    m.external_output = [
        f for i, f in enumerate(m.external_output) if not rm[i]
    ]
    m.external_units = [
        f for i, f in enumerate(m.external_output) if not rm[i]
    ]

    datpth = uzf_test_path
    irnbndpth = os.path.join(datpth, "irunbnd.dat")
    irunbnd = np.loadtxt(irnbndpth)

    vksbndpth = os.path.join(datpth, "vks.dat")
    vks = np.loadtxt(vksbndpth)

    finf = np.loadtxt(os.path.join(datpth, "finf.dat"))
    finf = np.reshape(finf, (m.nper, m.nrow, m.ncol))
    finf = {i: finf[i] for i in range(finf.shape[0])}

    extwc = np.loadtxt(os.path.join(datpth, "extwc.dat"))

    uzgag = {-68: [], 65: [3, 6, 1], 66: [6, 3, 2], 67: [10, 5, 3]}
    uzf = ModflowUzf1(
        m,
        nuztop=1,
        iuzfopt=1,
        irunflg=1,
        ietflg=1,
        ipakcb=0,
        iuzfcb2=61,
        # binary output of recharge and groundwater discharge
        ntrail2=25,
        nsets=20,
        surfdep=1.0,
        uzgag=uzgag,
        iuzfbnd=m.bas6.ibound.array,
        irunbnd=irunbnd,
        vks=vks,
        finf=finf,
        eps=3.5,
        thts=0.3,
        pet=5.000000e-08,
        extdp=15.0,
        extwc=extwc,
    )
    assert uzf.uzgag == uzgag
    uzgag2 = {
        -68: [-68],
        65: [3, 6, 65, 1],
        66: [6, 3, 66, 2],
        67: [10, 5, 67, 3],
    }

    uzf = ModflowUzf1(
        m,
        nuztop=1,
        iuzfopt=1,
        irunflg=1,
        ietflg=1,
        ipakcb=0,
        iuzfcb2=61,
        # binary output of recharge and groundwater discharge
        ntrail2=25,
        nsets=20,
        surfdep=1.0,
        uzgag=uzgag2,
        iuzfbnd=m.bas6.ibound.array,
        irunbnd=irunbnd,
        vks=vks,
        finf=finf,
        eps=3.5,
        thts=0.3,
        pet=5.000000e-08,
        extdp=15.0,
        extwc=extwc,
    )
    assert uzf.uzgag == uzgag
    uzgaglist = [[-68], [3, 6, 65, 1], [6, 3, 66, 2], [10, 5, 67, 3]]
    uzf = ModflowUzf1(
        m,
        nuztop=1,
        iuzfopt=1,
        irunflg=1,
        ietflg=1,
        ipakcb=0,
        iuzfcb2=61,
        # binary output of recharge and groundwater discharge
        ntrail2=25,
        nsets=20,
        surfdep=1.0,
        uzgag=uzgaglist,
        iuzfbnd=m.bas6.ibound.array,
        irunbnd=irunbnd,
        vks=vks,
        finf=finf,
        eps=3.5,
        thts=0.3,
        pet=5.000000e-08,
        extdp=15.0,
        extwc=extwc,
    )
    assert uzf.uzgag == uzgag
    m.write_input()
    uzf2 = ModflowUzf1.load(os.path.join(ws, uzf.file_name[0]), m)
    assert uzf2.uzgag == uzgag
    m2 = Modflow.load(
        "UZFtest2.nam",
        version="mf2005",
        exe_name="mf2005",
        verbose=True,
        model_ws=mf2005_test_path,
        forgive=False,
    )
    # verify that all of the arrays in the created UZF package are the same
    # as those in the loaded example
    attrs = [
        attr
        for attr in dir(uzf)
        if not callable(getattr(uzf, attr)) and not attr.startswith("__")
    ]
    for attr in attrs:
        a1 = uzf.__getattribute__(attr)
        if isinstance(a1, Util2d):
            a2 = m2.uzf.__getattribute__(attr)
            assert np.array_equal(a1.array, a2.array)
        elif attr in ["finf", "extwc", "pet", "extdp"]:
            if isinstance(a1, list):
                l2 = m2.uzf.__getattribute__(attr)
                for i, a in enumerate(a1):
                    # the created finf arrays all have a mult of 1
                    assert np.array_equal(a.array, l2[i].array)

    # load in the test problem
    m = Modflow("UZFtest2", model_ws=ws, verbose=True)
    dis = ModflowDis.load(mf2005_test_path / "UZFtest2.dis", m)
    uzf = ModflowUzf1.load(mf2005_test_path / "UZFtest2.uzf", m)
    assert np.sum(uzf.iuzfbnd.array) == 116
    assert np.array_equal(np.unique(uzf.irunbnd.array), np.arange(9))
    assert np.abs(np.sum(uzf.vks.array) / uzf.vks.cnstnt - 116.0) < 1e-5
    assert uzf.eps._Util2d__value == 3.5
    assert np.abs(uzf.thts._Util2d__value - 0.30) < 1e-5
    assert (
        np.abs(np.sum(uzf.extwc[0].array) / uzf.extwc[0].cnstnt - 176.0) < 1e4
    )
    for per in [0, 1]:
        assert np.abs(uzf.pet[per]._Util2d__value - 5e-8) < 1e-10
    for per in range(m.nper):
        assert (
            np.abs(np.sum(uzf.finf[per].array) / uzf.finf[per].cnstnt - 339.0)
            < 1e4
        )
        assert True
    uzf.write_file()
    m2 = Modflow("UZFtest2_2", model_ws=ws)
    dis = ModflowDis(nrow=m.nrow, ncol=m.ncol, nper=12, model=m2)
    uzf2 = ModflowUzf1.load(os.path.join(ws, "UZFtest2.uzf"), m2)
    attrs = dir(uzf)
    for attr in attrs:
        a1 = uzf.__getattribute__(attr)
        if isinstance(a1, Util2d):
            a2 = uzf2.__getattribute__(attr)
            assert a1 == a2
        # some parameters such as finf are stored as lists of util2d arrays
        elif attr in ["finf", "extwc", "pet", "extdp"]:
            if isinstance(a1, list):
                l2 = uzf2.__getattribute__(attr)
                for i, a in enumerate(a1):
                    assert a == l2[i]

    # load uzf test problem for nwt model with 'nwt_11_fmt'-style options and 'open/close' array types
    tpth = uzf_test_path / "load_uzf_for_nwt"
    [
        shutil.copy(os.path.join(tpth, f), os.path.join(ws, f))
        for f in os.listdir(tpth)
    ]
    m3 = Modflow("UZFtest3", version="mfnwt", verbose=True)
    m3.model_ws = ws
    dis = ModflowDis.load(os.path.join(tpth, "UZFtest3.dis"), m3)
    uzf = ModflowUzf1.load(os.path.join(tpth, "UZFtest3.uzf"), m3)
    assert np.sum(uzf.iuzfbnd.array) == 28800
    assert np.isclose(
        np.sum(uzf.finf.array) / uzf.finf[per].cnstnt, 13.7061, atol=1e-4
    )


@requires_exe("mfnwt")
def test_uzf_surfk(function_tmpdir, uzf_test_path):
    ws = uzf_test_path
    uzf_name = "UZFtest4.uzf"
    dis_name = "UZFtest2.dis"
    ml = Modflow(modelname="UZFtest4", version="mfnwt")
    dis = ModflowDis.load(os.path.join(ws, dis_name), ml, ext_unit_dict={})
    uzf = ModflowUzf1.load(os.path.join(ws, uzf_name), ml, ext_unit_dict={})

    assert uzf.options.seepsurfk
    assert abs(np.unique(uzf.surfk.array)[0] - 0.099) < 1e-06

    ml.change_model_ws(function_tmpdir)
    dis.write_file()
    uzf.write_file()

    ml2 = Modflow(version="mfnwt")
    dis2 = ModflowDis.load(
        os.path.join(function_tmpdir, "UZFtest4.dis"),
        ml2,
        ext_unit_dict={},
    )
    uzf2 = ModflowUzf1.load(
        os.path.join(function_tmpdir, uzf_name), ml2, ext_unit_dict={}
    )

    assert uzf2.options.seepsurfk
    assert np.allclose(uzf.surfk.array, uzf2.surfk.array)


def test_read_write_nwt_options(function_tmpdir):
    welstr = "OPTIONS\nSPECIFY 0.5 10\nTABFILES 2 28\nEND\n"
    welstr2 = "OPTIONS\nSPECIFY 0.3\nTABFILES 2 28\nEND\n"
    uzfstr = (
        "OPTIONS\nSPECIFYTHTR\nSPECIFYTHTI\nNOSURFLEAK\n"
        "SPECIFYSURFK\nSEEPSURFK\nETSQUARE 0.7\nNETFLUX 10 20\n"
        "SAVEFINF\nEND\n"
    )
    sfrstr = (
        "OPTIONS\nREACHINPUT\nTRANSROUTE\nTABFILES 10 21\n"
        "LOSSFACTOR 0.5\nSTRHC1KH 0.1\nSTRHC1KV 0.2\nEND\n"
    )

    welopt = OptionBlock.load_options(StringIO(welstr), ModflowWel)
    welopt2 = OptionBlock.load_options(StringIO(welstr2), ModflowWel)
    uzfopt = OptionBlock.load_options(StringIO(uzfstr), ModflowUzf1)
    sfropt = OptionBlock.load_options(StringIO(sfrstr), ModflowSfr2)

    assert repr(welopt) == welstr
    assert repr(welopt2) == welstr2
    assert repr(uzfopt) == uzfstr
    assert repr(sfropt) == sfrstr

    ws = function_tmpdir
    welopt.write_options(os.path.join(ws, "welopt.txt"))
    welopt2.write_options(os.path.join(ws, "welopt2.txt"))
    uzfopt.write_options(os.path.join(ws, "uzfopt.txt"))
    sfropt.write_options(os.path.join(ws, "sfropt.txt"))

    welopt = OptionBlock.load_options(
        os.path.join(ws, "welopt.txt"), ModflowWel
    )
    welopt2 = OptionBlock.load_options(
        os.path.join(ws, "welopt2.txt"), ModflowWel
    )
    uzfopt = OptionBlock.load_options(
        os.path.join(ws, "uzfopt.txt"), ModflowUzf1
    )
    sfropt = OptionBlock.load_options(
        os.path.join(ws, "sfropt.txt"), ModflowSfr2
    )

    assert repr(welopt) == welstr
    assert repr(welopt2) == welstr2
    assert repr(uzfopt) == uzfstr
    assert repr(sfropt) == sfrstr


def test_load_write_sfr_option_block(function_tmpdir, options_path):
    data_path = options_path
    sfr_name = "sagehen_ob.sfr"

    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    dis = ModflowDis.load(
        os.path.join(data_path, "sagehen.dis"),
        model=ml,
        ext_unit_dict={},
        check=False,
    )

    sfr = ModflowSfr2.load(
        os.path.join(data_path, sfr_name), ml, nper=2, ext_unit_dict={}
    )

    sfr_name2 = "sagehen_ob2.sfr"
    sfr.write_file(filename=os.path.join(function_tmpdir, sfr_name2))
    ml.remove_package("SFR")

    sfr2 = ModflowSfr2.load(
        os.path.join(function_tmpdir, sfr_name2),
        ml,
        nper=2,
        ext_unit_dict={},
    )

    assert sfr.options.reachinput == sfr2.options.reachinput
    assert sfr.options.strhc1kh == sfr2.options.strhc1kh
    assert sfr.options.factorkh == sfr.options.factorkh
    assert sfr.options.strhc1kv == sfr2.options.strhc1kv
    assert sfr.options.factorkv == sfr2.options.factorkv
    assert sfr2.options.factorkv == 0.4
    assert sfr2.options.factorkh == 0.2
    assert sfr2.options.block

    sfr2.options.strhc1kh = False
    sfr2.options.strhc1kv = False
    sfr2.write_file(os.path.join(function_tmpdir, sfr_name2))
    ml.remove_package("SFR")

    sfr3 = ModflowSfr2.load(
        os.path.join(function_tmpdir, sfr_name2),
        ml,
        nper=2,
        ext_unit_dict={},
    )

    assert sfr3.options.strhc1kh is False
    assert sfr3.options.strhc1kv is False


def test_load_write_sfr_option_line(function_tmpdir, options_path):
    data_path = options_path
    sfr_name = "sagehen.sfr"

    # test with modflow-nwt
    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    dis = ModflowDis.load(
        os.path.join(data_path, "sagehen.dis"),
        model=ml,
        ext_unit_dict={},
        check=False,
    )

    sfr = ModflowSfr2.load(
        os.path.join(data_path, sfr_name), ml, nper=2, ext_unit_dict={}
    )

    sfr_name2 = "sagehen2.sfr"
    sfr.write_file(os.path.join(function_tmpdir, sfr_name2))
    ml.remove_package("SFR")

    sfr2 = ModflowSfr2.load(
        os.path.join(function_tmpdir, sfr_name2),
        ml,
        nper=2,
        ext_unit_dict={},
    )

    assert sfr2.reachinput
    assert sfr2.options.factorkv == 0.4
    assert sfr2.options.factorkh == 0.2
    assert not sfr2.options.block

    # test with modflow-2005
    ml = Modflow(modelname="optionblock", verbose=False)

    dis = ModflowDis.load(
        os.path.join(data_path, "sagehen.dis"),
        model=ml,
        ext_unit_dict={},
        check=False,
    )

    sfr = ModflowSfr2.load(
        os.path.join(data_path, sfr_name), ml, nper=2, ext_unit_dict={}
    )

    sfr_name2 = "sagehen2.sfr"
    sfr.write_file(os.path.join(function_tmpdir, sfr_name2))
    ml.remove_package("SFR")

    sfr2 = ModflowSfr2.load(
        os.path.join(function_tmpdir, sfr_name2),
        ml,
        nper=2,
        ext_unit_dict={},
    )

    assert sfr2.reachinput


def test_load_write_uzf_option_block(function_tmpdir, options_path):
    data_path = options_path
    uzf_name = "sagehen_ob.uzf"

    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    dis = ModflowDis.load(
        os.path.join(data_path, "sagehen.dis"),
        model=ml,
        ext_unit_dict={},
        check=False,
    )

    uzf = ModflowUzf1.load(
        os.path.join(data_path, uzf_name), ml, ext_unit_dict=None, check=False
    )

    uzf_name2 = "sagehen_ob2.uzf"
    uzf.write_file(os.path.join(function_tmpdir, uzf_name2))
    ml.remove_package("UZF")

    uzf2 = ModflowUzf1.load(
        os.path.join(function_tmpdir, uzf_name2),
        ml,
        ext_unit_dict=None,
        check=False,
    )

    assert uzf.options.nosurfleak == uzf2.options.nosurfleak
    assert uzf.options.etsquare == uzf2.options.etsquare
    assert uzf.options.savefinf == uzf2.options.savefinf
    assert uzf2.options.block

    uzf2.smoothfact = 0.4

    uzf2.write_file(os.path.join(function_tmpdir, uzf_name2))
    ml.remove_package("UZF")

    uzf3 = ModflowUzf1.load(
        os.path.join(function_tmpdir, uzf_name2), ml, check=False
    )

    assert uzf3.options.smoothfact == 0.4
    assert uzf3.smoothfact == 0.4
    ml.remove_package("UZF")


def test_load_write_uzf_option_line(function_tmpdir, options_path):
    data_path = options_path
    uzf_name = "sagehen.uzf"

    # test with modflow-nwt
    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    dis = ModflowDis.load(
        os.path.join(data_path, "sagehen.dis"),
        model=ml,
        ext_unit_dict={},
        check=False,
    )

    uzf = ModflowUzf1.load(os.path.join(data_path, uzf_name), ml, check=False)

    assert uzf.nosurfleak
    assert uzf.etsquare
    assert uzf.smoothfact == 0.2
    assert uzf.options.savefinf

    uzf_name2 = "sagehen2.uzf"
    uzf.write_file(os.path.join(function_tmpdir, uzf_name2))
    ml.remove_package("UZF")

    uzf2 = ModflowUzf1.load(
        os.path.join(function_tmpdir, uzf_name2), ml, check=False
    )

    assert uzf2.nosurfleak
    assert uzf2.etsquare
    assert uzf2.smoothfact == 0.2
    assert uzf2.options.savefinf
    assert not uzf2.options.block


def test_load_write_wel_option_block(function_tmpdir, options_path):
    ws = options_path
    wel_name = "sagehen_ob.wel"
    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    wel = ModflowWel.load(
        os.path.join(ws, wel_name), ml, nper=2, ext_unit_dict={}, check=False
    )

    wel_name2 = "sagehen_ob2.wel"
    wel.write_file(os.path.join(function_tmpdir, wel_name2))
    ml.remove_package("WEL")

    wel2 = ModflowWel.load(
        os.path.join(function_tmpdir, wel_name2),
        ml,
        nper=2,
        ext_unit_dict={},
        check=False,
    )

    assert wel2.options.tabfiles == wel.options.tabfiles
    assert wel2.options.specify == wel.options.specify
    assert wel2.options.noprint == wel.options.noprint

    wel2.options.tabfiles = False
    wel2.phiramp = 0.4

    wel2.write_file(os.path.join(function_tmpdir, wel_name2))
    ml.remove_package("WEL")

    wel3 = ModflowWel.load(
        os.path.join(function_tmpdir, wel_name2),
        ml,
        nper=2,
        ext_unit_dict={},
        check=False,
    )

    assert not wel3.options.tabfiles
    assert wel3.options.phiramp == 0.4
    assert wel3.options.noprint


def test_load_write_wel_option_line(function_tmpdir, options_path):
    ws = options_path
    wel_name = "sagehen.wel"

    # test with modflow-nwt
    ml = Modflow(modelname="optionblock", version="mfnwt", verbose=False)

    wel = ModflowWel.load(
        os.path.join(ws, wel_name), ml, nper=2, ext_unit_dict={}, check=False
    )

    assert wel.options.noprint
    assert wel.specify
    assert wel.phiramp - 0.1 < 0.0001
    assert wel.iunitramp == 10

    wel.iunitramp = 20
    wel_name2 = "sagehen2.wel"
    wel.write_file(os.path.join(function_tmpdir, wel_name2))
    ml.remove_package("WEL")

    wel2 = ModflowWel.load(
        os.path.join(function_tmpdir, wel_name2),
        ml,
        nper=2,
        ext_unit_dict={},
        check=False,
    )

    assert wel.options.noprint
    assert wel.specify
    assert wel.phiramp - 0.1 < 0.0001
    assert wel.iunitramp == 20


@requires_exe("mfnwt")
def test_uzf_negative_iuzfopt(function_tmpdir):
    ml = Modflow(
        modelname="uzf_neg",
        version="mfnwt",
        exe_name="mfnwt",
        model_ws=function_tmpdir,
    )
    dis = ModflowDis(
        ml,
        nper=2,
        nlay=1,
        nrow=10,
        ncol=10,
        top=10,
        perlen=[1, 1],
        nstp=[5, 5],
        tsmult=1,
        steady=[False, False],
    )
    bas = ModflowBas(ml, strt=9, ibound=1)
    upw = ModflowUpw(
        ml,
        vka=0.1,
    )
    oc = ModflowOc(ml)
    nwt = ModflowNwt(ml, options="SIMPLE")

    uzf = ModflowUzf1(
        ml,
        nuztop=3,
        iuzfopt=-1,
        irunflg=1,
        ietflg=1,
        irunbnd=1,
        finf=1e-06,
        pet=0.1,
        extdp=0.2,
        specifysurfk=True,
        seepsurfk=True,
    )

    # uzf.write_file(os.path.join(model_ws, "uzf_neg.uzf"))

    ml.write_input()
    success, buff = ml.run_model()
    assert success, "UZF model with -1 iuzfopt failed to run"

    ml2 = Modflow.load(
        "uzf_neg.nam", version="mfnwt", model_ws=function_tmpdir
    )

    pet = ml2.uzf.pet.array
    extpd = ml2.uzf.pet.array

    assert (
        np.max(pet) == np.min(pet) and np.max(pet) != 0.1
    ), "Read error for iuzfopt less than 0"
    assert (
        np.max(extpd) == np.min(extpd) and np.max(extpd) != 0.2
    ), "Read error for iuzfopt less than 0"


def test_optionsblock_auxillary_typo():
    # Incorrect: auxillary
    #   Correct: auxiliary
    options = OptionBlock("", ModflowWel, block=True)
    assert options.auxiliary == []
    with pytest.deprecated_call():
        assert options.auxillary == []
    with pytest.deprecated_call():
        options.auxillary = ["aux", "iface"]
    assert options.auxiliary == ["aux", "iface"]
    options.auxiliary = []
    with pytest.deprecated_call():
        assert options.auxillary == []
