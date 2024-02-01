from itertools import repeat

import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from modflow_devtools.markers import requires_exe

import flopy
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
    ml = flopy.modflow.Modflow.load(
        model, version="mfnwt", model_ws=nwt_model_path
    )
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


def test_headfile_reverse_mf6(example_data_path, function_tmpdir):
    # load simulation and extract tdis
    sim_name = "test006_gwf3"
    sim = flopy.mf6.MFSimulation.load(
        sim_name=sim_name, sim_ws=example_data_path / "mf6" / sim_name
    )
    tdis = sim.get_package("tdis")

    # load cell budget file, providing tdis as kwarg
    model_path = example_data_path / "mf6" / sim_name
    file_stem = "flow_adj"
    file_path = model_path / "expected_output" / f"{file_stem}.hds"
    f = HeadFile(file_path, tdis=tdis)
    assert isinstance(f, HeadFile)

    # reverse the file
    rf_name = f"{file_stem}_rev.hds"
    f.reverse(filename=function_tmpdir / rf_name)
    rf = HeadFile(function_tmpdir / rf_name)
    assert isinstance(rf, HeadFile)

    # check that data from both files have the same shape
    f_data = f.get_alldata()
    f_shape = f_data.shape
    rf_data = rf.get_alldata()
    rf_shape = rf_data.shape
    assert f_shape == rf_shape

    # check number of records
    nrecords = f.get_nrecords()
    assert nrecords == rf.get_nrecords()

    # check that the data are reversed
    for idx in range(nrecords - 1, -1, -1):
        # check headers
        f_header = list(f.recordarray[nrecords - idx - 1])
        rf_header = list(rf.recordarray[idx])
        f_totim = f_header.pop(9)  # todo check totim
        rf_totim = rf_header.pop(9)
        assert f_header == rf_header
        assert f_header == rf_header

        # check data
        f_data = f.get_data(idx=idx)[0]
        rf_data = rf.get_data(idx=nrecords - idx - 1)[0]
        assert f_data.shape == rf_data.shape
        if f_data.ndim == 1:
            for row in range(len(f_data)):
                f_datum = f_data[row]
                rf_datum = rf_data[row]
                assert f_datum == rf_datum
        else:
            assert np.array_equal(f_data[0][0], rf_data[0][0])


@pytest.fixture
@pytest.mark.mf6
@requires_exe("mf6")
def mf6_gwf_2sp_st_tr(function_tmpdir):
    """
    A basic flow model with 2 stress periods,
    first steady-state, the second transient.
    """

    name = "mf6_gwf_2sp"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )

    tdis = flopy.mf6.ModflowTdis(
        simulation=sim,
        nper=2,
        perioddata=[(0, 1, 1), (10, 10, 1)],
    )

    ims = flopy.mf6.ModflowIms(
        simulation=sim,
        complexity="SIMPLE",
    )

    gwf = flopy.mf6.ModflowGwf(
        simulation=sim,
        modelname=name,
        save_flows=True,
    )

    dis = flopy.mf6.ModflowGwfdis(
        model=gwf, nlay=1, nrow=1, ncol=10, delr=1, delc=10, top=10, botm=0
    )

    npf = flopy.mf6.ModflowGwfnpf(
        model=gwf,
        icelltype=[0],
        k=10,
    )

    ic = flopy.mf6.ModflowGwfic(
        model=gwf,
        strt=0,
    )

    wel = flopy.mf6.ModflowGwfwel(
        model=gwf,
        stress_period_data={0: None, 1: [[(0, 0, 0), -1]]},
    )

    sto = flopy.mf6.ModflowGwfsto(
        model=gwf,
        ss=1e-4,
        steady_state={0: True},
        transient={1: True},
    )

    chd = flopy.mf6.ModflowGwfchd(
        model=gwf,
        stress_period_data={0: [[(0, 0, 9), 0]]},
    )

    oc = flopy.mf6.ModflowGwfoc(
        model=gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    return sim


def test_read_mf6_2sp(mf6_gwf_2sp_st_tr):
    sim = mf6_gwf_2sp_st_tr
    gwf = sim.get_model()
    sim.write_simulation(silent=False)
    success, _ = sim.run_simulation(silent=False)
    assert success

    # load heads and flows
    hds = gwf.output.head()
    cbb = gwf.output.budget()

    # check times
    exp_times = [float(t) for t in range(11)]
    assert hds.get_times() == exp_times
    assert cbb.get_times() == exp_times

    # check stress periods and time steps
    exp_kstpkper = [(0, 0)] + [(i, 1) for i in range(10)]
    assert hds.get_kstpkper() == exp_kstpkper
    assert cbb.get_kstpkper() == exp_kstpkper

    # check head data access by time
    exp_hds_data = np.array([[list(repeat(0.0, 10))]])
    hds_data = hds.get_data(totim=0)
    assert np.array_equal(hds_data, exp_hds_data)

    # check budget file data by time
    cbb_data = cbb.get_data(totim=0)
    assert len(cbb_data) > 0

    # check head data access by kstp and kper
    hds_data = hds.get_data(kstpkper=(0, 0))
    assert np.array_equal(hds_data, exp_hds_data)

    # check budget file data by kstp and kper
    cbb_data_kstpkper = cbb.get_data(kstpkper=(0, 0))
    assert len(cbb_data) == len(cbb_data_kstpkper)
    for i in range(len(cbb_data)):
        assert np.array_equal(cbb_data[i], cbb_data_kstpkper[i])


@pytest.mark.parametrize("compact", [True, False])
def test_read_mf2005_freyberg(example_data_path, function_tmpdir, compact):
    m = flopy.modflow.Modflow.load(
        example_data_path / "freyberg" / "freyberg.nam",
    )
    m.change_model_ws(function_tmpdir)
    oc = m.get_package("OC")
    oc.compact = compact

    m.write_input()
    success, buff = m.run_model(silent=False)
    assert success

    # load heads and flows
    hds_file = function_tmpdir / "freyberg.hds"
    cbb_file = function_tmpdir / "freyberg.cbc"
    assert hds_file.is_file()
    assert cbb_file.is_file()
    hds = HeadFile(hds_file)
    cbb = CellBudgetFile(cbb_file, model=m)  # failing to specify a model...

    # check times
    exp_times = [10.0]
    assert hds.get_times() == exp_times
    assert cbb.get_times() == exp_times  # ...causes get_times() to be empty

    # check stress periods and time steps
    exp_kstpkper = [(0, 0)]
    assert hds.get_kstpkper() == exp_kstpkper
    assert cbb.get_kstpkper() == exp_kstpkper

    # check head data access by time
    hds_data_totim = hds.get_data(totim=exp_times[0])
    assert hds_data_totim.shape == (1, 40, 20)

    # check budget file data by time
    cbb_data = cbb.get_data(totim=exp_times[0])
    assert len(cbb_data) > 0

    # check head data access by kstp and kper
    hds_data_kstpkper = hds.get_data(kstpkper=(0, 0))
    assert np.array_equal(hds_data_kstpkper, hds_data_totim)

    # check budget file data by kstp and kper
    cbb_data_kstpkper = cbb.get_data(kstpkper=(0, 0))
    assert len(cbb_data) == len(cbb_data_kstpkper)
    for i in range(len(cbb_data)):
        assert np.array_equal(cbb_data[i], cbb_data_kstpkper[i])
