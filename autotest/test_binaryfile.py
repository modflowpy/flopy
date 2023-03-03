import os

import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from flopy.modflow import Modflow
from flopy.utils import (
    BinaryHeader,
    CellBudgetFile,
    HeadFile,
    HeadUFile,
    Util2d,
)
from flopy.utils.binaryfile import (
    get_headfile_precision,
    write_budget,
    write_head,
)
from flopy.utils.gridutil import uniform_flow_field


@pytest.fixture
def freyberg_model_path(example_data_path):
    return example_data_path / "freyberg"


@pytest.fixture
def nwt_model_path(example_data_path):
    return example_data_path / "nwt_test"


@pytest.fixture
def zonbud_model_path(example_data_path):
    return example_data_path / "zonbud_examples"


def test_binaryfile_writeread(function_tmpdir, nwt_model_path):
    model = "Pr3_MFNWT_lower.nam"
    ml = Modflow.load(model, version="mfnwt", model_ws=nwt_model_path)
    # change the model work space
    ml.change_model_ws(function_tmpdir)
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
    pth = function_tmpdir / "bottom.hds"
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
    pth = function_tmpdir / "bottom_single.hds"
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
        example_data_path / "mf2005_test" / "swiex1.gitzta",
        precision="single",
    )
    ts = cbf.get_ts(text="ZETASRF  1", idx=(0, 0, 24))
    assert ts.shape == (
        4,
        2,
    ), f"shape of zeta timeseries is {ts.shape} not (4, 2)"


def test_load_binary_head_file(example_data_path):
    mpath = example_data_path / "freyberg"
    hf = HeadFile(mpath / "freyberg.githds")
    assert isinstance(hf, HeadFile)


def test_plot_binary_head_file(example_data_path):
    hf = HeadFile(example_data_path / "freyberg" / "freyberg.githds")
    hf.mg.set_coord_info(xoff=1000.0, yoff=200.0, angrot=15.0)

    assert isinstance(hf.plot(), Axes)
    plt.close()


def test_headu_file_data(function_tmpdir, example_data_path):
    fname = example_data_path / "unstructured" / "headu.githds"
    headobj = HeadUFile(fname)
    assert isinstance(headobj, HeadUFile)
    assert headobj.nlay == 3

    # ensure recordarray is has correct data
    ra = headobj.recordarray
    nnodes = 19479
    assert ra["kstp"].min() == 1
    assert ra["kstp"].max() == 1
    assert ra["kper"].min() == 1
    assert ra["kper"].max() == 5
    assert ra["ncol"].min() == 1
    assert ra["ncol"].max() == 14001
    assert ra["nrow"].min() == 7801
    assert ra["nrow"].max() == nnodes

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


@pytest.mark.slow
def test_headufile_get_ts(example_data_path):
    heads = HeadUFile(example_data_path / "unstructured" / "headu.githds")
    nnodes = 19479

    # make sure timeseries can be retrieved for each node
    for i in range(0, nnodes, 100):
        heads.get_ts(idx=i)
    with pytest.raises(IndexError):
        heads.get_ts(idx=i + 100)

    # ...and retrieved in groups
    for i in range(10):
        heads.get_ts([i, i + 1, i + 2])

    heads = HeadUFile(
        example_data_path
        / "mfusg_test"
        / "01A_nestedgrid_nognc"
        / "output"
        / "flow.hds"
    )
    nnodes = 121
    for i in range(nnodes):
        heads.get_ts(idx=i)
    with pytest.raises(IndexError):
        heads.get_ts(idx=i + 1)

    # ...and retrieved in groups
    for i in range(10):
        heads.get_ts([i, i + 1, i + 2])


def test_get_headfile_precision(example_data_path):
    precision = get_headfile_precision(
        example_data_path / "freyberg" / "freyberg.githds"
    )
    assert precision == "single"

    precision = get_headfile_precision(
        example_data_path
        / "mf6"
        / "create_tests"
        / "test005_advgw_tidal"
        / "expected_output"
        / "AdvGW_tidal.hds"
    )
    assert precision == "double"


_example_data_path = get_example_data_path()


@pytest.mark.parametrize(
    "path",
    [
        _example_data_path / "mf2005_test" / "swiex1.gitzta",
        _example_data_path / "mp6" / "EXAMPLE.BUD",
        _example_data_path
        / "mfusg_test"
        / "01A_nestedgrid_nognc"
        / "output"
        / "flow.cbc",
    ],
)
def test_budgetfile_detect_precision_single(path):
    file = CellBudgetFile(path, precision="auto")
    assert file.realtype == np.float32


@pytest.mark.parametrize(
    "path",
    [
        _example_data_path
        / "mf6"
        / "test006_gwf3"
        / "expected_output"
        / "flow_adj.cbc",
    ],
)
def test_budgetfile_detect_precision_double(path):
    file = CellBudgetFile(path, precision="auto")
    assert file.realtype == np.float64


def test_write_head(function_tmpdir):
    file_path = function_tmpdir / "headfile"
    head_data = np.random.random((10, 10))

    write_head(file_path, head_data)

    assert file_path.is_file()
    content = np.fromfile(file_path)
    assert np.array_equal(head_data.ravel(), content)

    # TODO: what else needs to be checked here?


def test_write_budget(function_tmpdir):
    file_path = function_tmpdir / "budgetfile"

    nlay = 3
    nrow = 3
    ncol = 3
    qx = 1.0
    qy = 0.0
    qz = 0.0
    shape = (nlay, nrow, ncol)
    spdis, flowja = uniform_flow_field(qx, qy, qz, shape)

    write_budget(file_path, flowja, kstp=0)
    assert file_path.is_file()
    content1 = np.fromfile(file_path)

    write_budget(file_path, flowja, kstp=1, kper=1, text="text")
    assert file_path.is_file()
    content2 = np.fromfile(file_path)

    # TODO: why are these the same?
    assert np.array_equal(content1, content2)


def test_binaryfile_read(function_tmpdir, freyberg_model_path):
    h = HeadFile(freyberg_model_path / "freyberg.githds")
    assert isinstance(h, HeadFile)

    times = h.get_times()
    assert np.isclose(times[0], 10.0), f"times[0] != {times[0]}"

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (0, 0), "kstpkper[0] != (0, 0)"

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(
        h0, h1
    ), "binary head read using totim != head read using kstpkper"
    assert np.array_equal(
        h0, h2
    ), "binary head read using totim != head read using idx"

    ts = h.get_ts((0, 7, 5))
    expected = 26.00697135925293
    assert np.isclose(
        ts[0, 1], expected
    ), f"time series value ({ts[0, 1]}) != {expected}"
    h.close()

    # Check error when reading empty file
    fname = function_tmpdir / "empty.githds"
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        HeadFile(fname)
    with pytest.raises(ValueError):
        HeadFile(fname, "head", "single")


def test_binaryfile_read_context(freyberg_model_path):
    hds_path = freyberg_model_path / "freyberg.githds"
    with HeadFile(hds_path) as h:
        data = h.get_data()
        assert data.max() > 0, data.max()
        assert not h.file.closed
    assert h.file.closed

    with pytest.raises(ValueError) as e:
        h.get_data()
    assert str(e.value) == "seek of closed file", str(e.value)


def test_cellbudgetfile_read_context(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    cbc_path = mf2005_model_path / "mnw1.gitcbc"
    with CellBudgetFile(cbc_path) as v:
        data = v.get_data(text="DRAINS")[0]
        assert data.min() < 0, data.min()
        assert not v.file.closed
    assert v.file.closed

    with pytest.raises(ValueError) as e:
        v.get_data(text="DRAINS")
    assert str(e.value) == "seek of closed file", str(e.value)


def test_cellbudgetfile_read(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    v = CellBudgetFile(mf2005_model_path / "mnw1.gitcbc")
    assert isinstance(v, CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 5, "length of kstpkper != 5"

    records = v.get_unique_record_names()
    idx = 0
    for t in kstpkper:
        for record in records:
            t0 = v.get_data(kstpkper=t, text=record, full3D=True)[0]
            t1 = v.get_data(idx=idx, text=record, full3D=True)[0]
            assert np.array_equal(t0, t1), (
                f"binary budget item {record} read using kstpkper != binary "
                f"budget item {record} read using idx"
            )
            idx += 1
    v.close()


def test_cellbudgetfile_position(function_tmpdir, zonbud_model_path):
    fpth = zonbud_model_path / "freyberg.gitcbc"
    v = CellBudgetFile(fpth)
    assert isinstance(v, CellBudgetFile)

    # starting position of data
    idx = 8767
    ipos = v.get_position(idx)
    ival = 50235424
    assert ipos == ival, f"position of index 8767 != {ival}"

    ipos = v.get_position(idx, header=True)
    ival = 50235372
    assert ipos == ival, f"position of index 8767 header != {ival}"

    cbcd = []
    for i in range(idx, v.get_nrecords()):
        cbcd.append(v.get_data(i)[0])
    v.close()

    # write the last entry as a new binary file
    fin = open(fpth, "rb")
    fin.seek(ipos)
    length = os.path.getsize(fpth) - ipos

    buffsize = 32
    opth = str(function_tmpdir / "end.cbc")
    with open(opth, "wb") as fout:
        while length:
            chunk = min(buffsize, length)
            data = fin.read(chunk)
            fout.write(data)
            length -= chunk
    fin.close()

    v2 = CellBudgetFile(opth, verbose=True)

    try:
        v2.list_records()
    except:
        assert False, f"could not list records on {opth}"

    names = v2.get_unique_record_names(decode=True)

    cbcd2 = []
    for i in range(0, v2.get_nrecords()):
        cbcd2.append(v2.get_data(i)[0])
    v2.close()

    for i, (d1, d2) in enumerate(zip(cbcd, cbcd2)):
        msg = f"{names[i].rstrip()} data from slice is not identical"
        assert np.array_equal(d1, d2), msg

    # Check error when reading empty file
    fname = function_tmpdir / "empty.gitcbc"
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        CellBudgetFile(fname)


def test_cellbudgetfile_readrecord(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    cbc_fname = mf2005_model_path / "test1tr.gitcbc"
    v = CellBudgetFile(cbc_fname)
    assert isinstance(v, CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 30, "length of kstpkper != 30"

    with pytest.raises(TypeError) as e:
        v.get_data()
    assert str(e.value).startswith(
        "get_data() missing 1 required argument"
    ), str(e.exception)

    t = v.get_data(text="STREAM LEAKAGE")
    assert len(t) == 30, "length of stream leakage data != 30"
    assert (
        t[0].shape[0] == 36
    ), "sfr budget data does not have 36 reach entries"

    t = v.get_data(text="STREAM LEAKAGE", full3D=True)
    assert t[0].shape == (1, 15, 10), (
        "3D sfr budget data does not have correct shape (1, 15,10) - "
        "returned shape {}".format(t[0].shape)
    )

    for kk in kstpkper:
        t = v.get_data(kstpkper=kk, text="STREAM LEAKAGE", full3D=True)[0]
        assert t.shape == (1, 15, 10), (
            "3D sfr budget data for kstpkper {} "
            "does not have correct shape (1, 15,10) - "
            "returned shape {}".format(kk, t[0].shape)
        )

    idx = v.get_indices()
    assert idx is None, "get_indices() without record did not return None"

    records = v.get_unique_record_names()
    for record in records:
        indices = v.get_indices(text=record.strip())
        for idx, kk in enumerate(kstpkper):
            t0 = v.get_data(kstpkper=kk, text=record.strip())[0]
            t1 = v.get_data(idx=indices[idx], text=record)[0]
            assert np.array_equal(
                t0, t1
            ), "binary budget item {0} read using kstpkper != binary budget item {0} read using idx".format(
                record
            )

    # idx can be either an int or a list of ints
    s9 = v.get_data(idx=9)
    assert len(s9) == 1
    s09 = v.get_data(idx=[0, 9])
    assert len(s09) == 2
    assert (s09[1] == s9).all()

    v.close()


def test_cellbudgetfile_readrecord_waux(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    cbc_fname = mf2005_model_path / "test1tr.gitcbc"
    v = CellBudgetFile(cbc_fname)
    assert isinstance(v, CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 30, "length of kstpkper != 30"

    t = v.get_data(text="WELLS")
    assert len(t) == 30, "length of well data != 30"
    assert t[0].shape[0] == 10, "wel budget data does not have 10 well entries"
    assert t[0].dtype.names == ("node", "q", "IFACE")
    np.testing.assert_array_equal(
        t[0]["node"],
        [54, 55, 64, 65, 74, 75, 84, 85, 94, 95],
    )
    np.testing.assert_array_equal(t[0]["q"], np.repeat(np.float32(-10.0), 10))
    np.testing.assert_array_equal(
        t[0]["IFACE"],
        np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 0], np.float32),
    )

    t = v.get_data(text="WELLS", full3D=True)
    assert t[0].shape == (1, 15, 10), (
        "3D wel budget data does not have correct shape (1, 15,10) - "
        "returned shape {}".format(t[0].shape)
    )

    for kk in kstpkper:
        t = v.get_data(kstpkper=kk, text="wells", full3D=True)[0]
        assert t.shape == (1, 15, 10), (
            "3D wel budget data for kstpkper {} "
            "does not have correct shape (1, 15,10) - "
            "returned shape {}".format(kk, t[0].shape)
        )

    idx = v.get_indices()
    assert idx is None, "get_indices() without record did not return None"

    records = v.get_unique_record_names()
    for record in records:
        indices = v.get_indices(text=record.strip())
        for idx, kk in enumerate(kstpkper):
            t0 = v.get_data(kstpkper=kk, text=record.strip())[0]
            t1 = v.get_data(idx=indices[idx], text=record)[0]
            assert np.array_equal(
                t0, t1
            ), "binary budget item {0} read using kstpkper != binary budget item {0} read using idx".format(
                record
            )
    v.close()
