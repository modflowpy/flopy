import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import flopy
from flopy.modflow import Modflow
from flopy.utils import (
    BinaryHeader,
    CellBudgetFile,
    HeadFile,
    HeadUFile,
    Util2d,
)
from flopy.utils.binaryfile import get_headfile_precision


@pytest.fixture
def nwt_model_path(example_data_path):
    return example_data_path / "nwt_test"


def test_binaryfile_writeread(tmpdir, nwt_model_path):
    model = "Pr3_MFNWT_lower.nam"
    ml = Modflow.load(model, version="mfnwt", model_ws=str(nwt_model_path))
    # change the model work space
    ml.change_model_ws(str(tmpdir))
    #
    ncol = ml.dis.ncol
    nrow = ml.dis.nrow
    text = "head"
    # write a double precision head file
    precision = "double"
    pertim = ml.dis.perlen.array[0].astype(np.float64)
    header = BinaryHeader.create(
        bintype=text,
        precision=precision,
        text=text,
        nrow=nrow,
        ncol=ncol,
        ilay=1,
        pertim=pertim,
        totim=pertim,
        kstp=1,
        kper=1,
    )
    b = ml.dis.botm.array[0, :, :].astype(np.float64)
    pth = str(tmpdir / "bottom.hds")
    Util2d.write_bin(b.shape, pth, b, header_data=header)

    bo = HeadFile(pth, precision=precision)
    times = bo.get_times()
    errmsg = "double precision binary totim read is not equal to totim written"
    assert times[0] == pertim, errmsg
    kstpkper = bo.get_kstpkper()
    errmsg = "kstp, kper read is not equal to kstp, kper written"
    assert kstpkper[0] == (0, 0), errmsg
    br = bo.get_data()
    errmsg = "double precision binary data read is not equal to data written"
    assert np.allclose(b, br), errmsg

    # write a single precision head file
    precision = "single"
    pertim = ml.dis.perlen.array[0].astype(np.float32)
    header = BinaryHeader.create(
        bintype=text,
        precision=precision,
        text=text,
        nrow=nrow,
        ncol=ncol,
        ilay=1,
        pertim=pertim,
        totim=pertim,
        kstp=1,
        kper=1,
    )
    b = ml.dis.botm.array[0, :, :].astype(np.float32)
    pth = str(tmpdir / "bottom_single.hds")
    Util2d.write_bin(b.shape, pth, b, header_data=header)

    bo = HeadFile(pth, precision=precision)
    times = bo.get_times()
    errmsg = "single precision binary totim read is not equal to totim written"
    assert times[0] == pertim, errmsg
    kstpkper = bo.get_kstpkper()
    errmsg = "kstp, kper read is not equal to kstp, kper written"
    assert kstpkper[0] == (0, 0), errmsg
    br = bo.get_data()
    errmsg = "singleprecision binary data read is not equal to data written"
    assert np.allclose(b, br), errmsg


def test_load_cell_budget_file_timeseries(example_data_path):
    cbf = CellBudgetFile(
        str(example_data_path / "mf2005_test" / "swiex1.gitzta"),
        precision="single",
    )
    ts = cbf.get_ts(text="ZETASRF  1", idx=(0, 0, 24))
    assert ts.shape == (
        4,
        2,
    ), f"shape of zeta timeseries is {ts.shape} not (4, 2)"


def test_load_binary_head_file(example_data_path):
    mpath = example_data_path / "freyberg"
    hf = HeadFile(str(mpath / "freyberg.githds"))
    assert isinstance(hf, HeadFile)


def test_plot_binary_head_file(example_data_path):
    hf = HeadFile(str(example_data_path / "freyberg" / "freyberg.githds"))
    hf.mg.set_coord_info(xoff=1000.0, yoff=200.0, angrot=15.0)

    assert isinstance(hf.plot(), Axes)
    plt.close()


def test_headu_file(tmpdir, example_data_path):
    fname = str(example_data_path / "unstructured" / "headu.githds")
    headobj = HeadUFile(fname)
    assert isinstance(headobj, HeadUFile)
    assert headobj.nlay == 3

    # ensure recordarray is has correct data
    ra = headobj.recordarray
    assert ra["kstp"].min() == 1
    assert ra["kstp"].max() == 1
    assert ra["kper"].min() == 1
    assert ra["kper"].max() == 5
    assert ra["ncol"].min() == 1
    assert ra["ncol"].max() == 14001
    assert ra["nrow"].min() == 7801
    assert ra["nrow"].max() == 19479

    # read the heads for the last time and make sure they are correct
    data = headobj.get_data()
    assert len(data) == 3
    minmaxtrue = [
        np.array([-1.4783, -1.0]),
        np.array([-2.0, -1.0]),
        np.array([-2.0, -1.01616]),
    ]
    for i, d in enumerate(data):
        t1 = np.array([d.min(), d.max()])
        assert np.allclose(t1, minmaxtrue[i])

    return


def test_get_headfile_precision(example_data_path):
    precision = get_headfile_precision(
        str(example_data_path / "freyberg" / "freyberg.githds")
    )
    assert precision == "single"

    precision = get_headfile_precision(
        str(
            example_data_path
            / "mf6"
            / "create_tests"
            / "test005_advgw_tidal"
            / "expected_output"
            / "AdvGW_tidal.hds"
        )
    )
    assert precision == "double"


_example_data_path = get_example_data_path()


@pytest.mark.parametrize(
    "path",
    [
        str(p)
        for p in [
            _example_data_path / "mf2005_test" / "swiex1.gitzta",
            _example_data_path / "mp6" / "EXAMPLE.BUD",
            _example_data_path
            / "mfusg_test"
            / "01A_nestedgrid_nognc"
            / "output"
            / "flow.cbc",
        ]
    ],
)
def test_budgetfile_detect_precision_single(path):
    file = CellBudgetFile(path, precision="auto")
    assert file.realtype == np.float32


@pytest.mark.parametrize(
    "path",
    [
        str(p)
        for p in [
            _example_data_path
            / "mf6"
            / "test006_gwf3"
            / "expected_output"
            / "flow_adj.cbc",
        ]
    ],
)
def test_budgetfile_detect_precision_double(path):
    file = CellBudgetFile(path, precision="auto")
    assert file.realtype == np.float64
