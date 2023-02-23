# Test SWR binary read functionality
import pytest
from modflow_devtools.misc import has_pkg

from flopy.utils import (
    SwrBudget,
    SwrExchange,
    SwrFlow,
    SwrObs,
    SwrStage,
    SwrStructure,
)


@pytest.fixture
def swr_test_path(example_data_path):
    return example_data_path / "swr_test"


files = (
    "SWR004.stg",
    "SWR004.flow",
    "SWR004.vel",
    "swr005.qaq",
    "SWR004.str",
    "SWR004.obs",
)


@pytest.mark.parametrize("ipos", [0])
def test_swr_binary_stage(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrStage(fswr_test_path)
    assert isinstance(sobj, SwrStage), "SwrStage object not created"

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), "SwrStage records does not equal (18, 0)"

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, "SwrStage ntimes does not equal 336"

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert (
            r is not None
        ), "SwrStage could not read data with get_data(idx=)"
        assert r.shape == (
            18,
        ), "SwrStage stage data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 2
        ), "SwrStage stage data dtype does not have 2 entries"

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        336,
        3,
    ), "SwrStage kswrkstpkper shape does not equal (336, 3)"

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert (
            r is not None
        ), "SwrStage could not read data with get_data(kswrkstpkper=)"
        assert r.shape == (
            18,
        ), "SwrStage stage data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 2
        ), "SwrStage stage data dtype does not have 2 entries"

    times = sobj.get_times()
    assert len(times) == 336, "SwrStage times length does not equal 336"

    for time in times:
        r = sobj.get_data(totim=time)
        assert (
            r is not None
        ), "SwrStage could not read data with get_data(tottim=)"
        assert r.shape == (
            18,
        ), "SwrStage stage data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 2
        ), "SwrStage stage data dtype does not have 2 entries"

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (
        336,
    ), "SwrStage stage timeseries shape does not equal (336,)"
    assert (
        len(ts.dtype.names) == 2
    ), "SwrStage stage time series stage data dtype does not have 2 entries"

    # plt.plot(ts['totim'], ts['stage'])
    # plt.show()


@pytest.mark.parametrize("ipos", [1])
def test_swr_binary_budget(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrBudget(fswr_test_path)
    assert isinstance(sobj, SwrBudget), "SwrBudget object not created"

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), "SwrBudget records does not equal (18, 0)"

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, "SwrBudget ntimes does not equal 336"

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert (
            r is not None
        ), "SwrBudget could not read data with get_data(idx=)"
        assert r.shape == (
            18,
        ), "SwrBudget budget data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 15
        ), "SwrBudget data dtype does not have 15 entries"

    # plt.bar(range(18), r['inf-out'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        336,
        3,
    ), "SwrBudget kswrkstpkper shape does not equal (336, 3)"

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert (
            r is not None
        ), "SwrBudget could not read data with get_data(kswrkstpkper=)"
        assert r.shape == (
            18,
        ), "SwrBudget budget data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 15
        ), "SwrBudget budget data dtype does not have 15 entries"

    times = sobj.get_times()
    assert len(times) == 336, "SwrBudget times length does not equal 336"

    for time in times:
        r = sobj.get_data(totim=time)
        assert (
            r is not None
        ), "SwrBudget could not read data with get_data(tottim=)"
        assert r.shape == (
            18,
        ), "SwrBudget budget data shape does not equal (18,)"
        assert (
            len(r.dtype.names) == 15
        ), "SwrBudget budget data dtype does not have 15 entries"

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (
        336,
    ), "SwrBudget budget timeseries shape does not equal (336,)"
    assert (
        len(ts.dtype.names) == 15
    ), "SwrBudget time series budget data dtype does not have 15 entries"

    # plt.plot(ts['totim'], ts['qbcflow'])
    # plt.show()


@pytest.mark.parametrize("ipos", [2])
def test_swr_binary_qm(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrFlow(fswr_test_path)
    assert isinstance(sobj, SwrFlow), "SwrFlow object not created"

    nrecords = sobj.get_nrecords()
    assert nrecords == (40, 18), "SwrFlow records does not equal (40, 18)"

    connect = sobj.get_connectivity()
    assert connect.shape == (
        40,
        3,
    ), "SwrFlow connectivity shape does not equal (40, 3)"

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, "SwrFlow ntimes does not equal 336"

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, "SwrFlow could not read data with get_data(idx=)"
        assert r.shape == (40,), "SwrFlow qm data shape does not equal (40,)"
        assert (
            len(r.dtype.names) == 3
        ), "SwrFlow qm data dtype does not have 3 entries"

    # plt.bar(range(40), r['flow'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        336,
        3,
    ), "SwrFlow kswrkstpkper shape does not equal (336, 3)"

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert (
            r is not None
        ), "SwrFlow could not read data with get_data(kswrkstpkper=)"
        assert r.shape == (40,), "SwrFlow qm data shape does not equal (40,)"
        assert (
            len(r.dtype.names) == 3
        ), "SwrFlow qm data dtype does not have 3 entries"

    times = sobj.get_times()
    assert len(times) == 336, "SwrFlow times length does not equal 336"

    for time in times:
        r = sobj.get_data(totim=time)
        assert (
            r is not None
        ), "SwrFlow could not read data with get_data(tottim=)"
        assert r.shape == (40,), "SwrFlow qm data shape does not equal (40,)"
        assert (
            len(r.dtype.names) == 3
        ), "SwrFlow qm data dtype does not have 3 entries"

    ts = sobj.get_ts(irec=17, iconn=16)
    assert ts.shape == (
        336,
    ), "SwrFlow qm timeseries shape does not equal (336,)"
    assert (
        len(ts.dtype.names) == 3
    ), "SwrFlow time series qm data dtype does not have 3 entries"

    ts2 = sobj.get_ts(irec=16, iconn=17)
    assert ts2.shape == (
        336,
    ), "SwrFlow qm timeseries shape does not equal (336,)"
    assert (
        len(ts2.dtype.names) == 3
    ), "SwrFlow time series qm data dtype does not have 3 entries"

    # plt.plot(ts['totim'], ts['velocity'])
    # plt.plot(ts2['totim'], ts2['velocity'])
    # plt.show()


@pytest.mark.parametrize("ipos", [3])
def test_swr_binary_qaq(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrExchange(fswr_test_path, verbose=True)
    assert isinstance(sobj, SwrExchange), "SwrExchange object not created"

    nrecords = sobj.get_nrecords()
    assert nrecords == (19, 0), "SwrExchange records does not equal (19, 0)"

    ntimes = sobj.get_ntimes()
    assert ntimes == 350, "SwrExchange ntimes does not equal 350"

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert (
            r is not None
        ), "SwrExchange could not read data with get_data(idx=)"
        assert r.shape == (
            21,
        ), "SwrExchange qaq data shape does not equal (21,)"
        assert (
            len(r.dtype.names) == 11
        ), "SwrExchange qaq data dtype does not have 11 entries"

    # plt.bar(range(21), r['qaq'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        350,
        3,
    ), "SwrExchange kswrkstpkper shape does not equal (350, 3)"

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert (
            r is not None
        ), "SwrExchange could not read data with get_data(kswrkstpkper=)"
        assert r.shape == (
            21,
        ), "SwrExchange qaq data shape does not equal (21,)"
        assert (
            len(r.dtype.names) == 11
        ), "SwrExchange qaq data dtype does not have 11 entries"

    times = sobj.get_times()
    assert len(times) == 350, "SwrExchange times length does not equal 350"

    for time in times:
        r = sobj.get_data(totim=time)
        assert (
            r is not None
        ), "SwrExchange could not read data with get_data(tottim=)"
        assert r.shape == (
            21,
        ), "SwrExchange qaq data shape does not equal (21,)"
        assert (
            len(r.dtype.names) == 11
        ), "SwrExchange qaq data dtype does not have 11 entries"

    ts = sobj.get_ts(irec=17, klay=0)
    assert ts.shape == (
        350,
    ), "SwrExchange timeseries shape does not equal (350,)"
    assert (
        len(ts.dtype.names) == 11
    ), "SwrExchange time series qaq data dtype does not have 11 entries"

    # plt.plot(ts['totim'], ts['qaq'])
    # plt.show()


@pytest.mark.parametrize("ipos", [4])
def test_swr_binary_structure(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrStructure(fswr_test_path, verbose=True)
    assert isinstance(sobj, SwrStructure), "SwrStructure object not created"

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), "SwrStructure records does not equal (18, 0)"

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, "SwrStructure ntimes does not equal 336"

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert (
            r is not None
        ), "SwrStructure could not read data with get_data(idx=)"
        assert r.shape == (
            2,
        ), "SwrStructure structure data shape does not equal (2,)"
        assert (
            len(r.dtype.names) == 8
        ), "SwrStructure structure data dtype does not have 8 entries"

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        336,
        3,
    ), "SwrStructure kswrkstpkper shape does not equal (336, 3)"

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert (
            r is not None
        ), "SwrStructure could not read data with get_data(kswrkstpkper=)"
        assert r.shape == (
            2,
        ), "SwrStructure structure data shape does not equal (2,)"
        assert (
            len(r.dtype.names) == 8
        ), "SwrStructure structure data dtype does not have 8 entries"

    times = sobj.get_times()
    assert len(times) == 336, "SwrStructure times length does not equal 336"

    for time in times:
        r = sobj.get_data(totim=time)
        assert (
            r is not None
        ), "SwrStructure could not read data with get_data(tottim=)"
        assert r.shape == (
            2,
        ), "SwrStructure structure data shape does not equal (2,)"
        assert (
            len(r.dtype.names) == 8
        ), "SwrStructure structure data dtype does not have 8 entries"

    ts = sobj.get_ts(irec=17, istr=0)
    assert ts.shape == (
        336,
    ), "SwrStructure timeseries shape does not equal (336,)"
    assert (
        len(ts.dtype.names) == 8
    ), "SwrStructure time series structure data dtype does not have 8 entries"

    # plt.plot(ts['totim'], ts['strflow'])
    # plt.show()

    obs3 = sobj.get_ts(irec=17, istr=0)


@pytest.mark.parametrize("ipos", [5])
def test_swr_binary_obs(swr_test_path, ipos):
    fswr_test_path = swr_test_path / files[ipos]
    sobj = SwrObs(fswr_test_path)
    assert isinstance(sobj, SwrObs), "SwrObs object not created"

    nobs = sobj.get_nobs()
    assert nobs == 9, "SwrObs numobs does not equal 9"

    obsnames = sobj.get_obsnames()
    assert len(obsnames) == 9, "SwrObs number of obsnames does not equal 9"

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, "SwrObs numtimes does not equal 336"

    times = sobj.get_times()
    assert len(times) == 336, "SwrFile times length does not equal 336"

    ts = sobj.get_data()
    assert ts.shape == (
        336,
    ), "SwrObs length of data array does not equal (336,)"
    assert (
        len(ts.dtype.names) == 10
    ), "SwrObs data does not have totim + 9 observations"

    ts = sobj.get_data(obsname="OBS5")
    assert ts.shape == (
        336,
    ), "SwrObs length of data array does not equal (336,)"
    assert (
        len(ts.dtype.names) == 2
    ), "SwrObs data does not have totim + 1 observation"

    # plt.plot(ts['totim'], ts['OBS5'])
    # plt.show()

    for idx in range(ntimes):
        d = sobj.get_data(idx=idx)
        assert d.shape == (
            1,
        ), "SwrObs length of data array does not equal (1,)"
        assert (
            len(d.dtype.names) == nobs + 1
        ), "SwrObs data does not have nobs + 1"

    for time in times:
        d = sobj.get_data(totim=time)
        assert d.shape == (
            1,
        ), "SwrObs length of data array does not equal (1,)"
        assert (
            len(d.dtype.names) == nobs + 1
        ), "SwrObs data does not have nobs + 1"

    # test get_dataframes()
    if has_pkg("pandas"):
        import pandas as pd

        for idx in range(ntimes):
            df = sobj.get_dataframe(idx=idx, timeunit="S")
            assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
            assert df.shape == (1, nobs + 1), "data shape is not (1, 10)"

        for time in times:
            df = sobj.get_dataframe(totim=time, timeunit="S")
            assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
            assert df.shape == (1, nobs + 1), "data shape is not (1, 10)"

        df = sobj.get_dataframe(timeunit="S")
        assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
        assert df.shape == (336, nobs + 1), "data shape is not (336, 10)"
    else:
        print("pandas not available...")
