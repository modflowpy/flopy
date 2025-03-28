import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flopy
from flopy.discretization.modeltime import ModelTime
from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.utils.binaryfile import CellBudgetFile, HeadFile


@pytest.mark.parametrize(
    ["dt_rep"],
    [
        (datetime.datetime(2024, 11, 12),),
        (np.datetime64("2024-11-12"),),
        (pd.Timestamp("2024-11-12"),),
        ("2024-11-12",),
        ("2024/11/12",),
        ("11-12-2024",),
        ("11/12/2024",),
    ],
)
def test_date_userinput_parsing(dt_rep):
    valid = datetime.datetime(2024, 11, 12)
    dt_obj = ModelTime.parse_datetime(dt_rep)
    if dt_obj != valid:
        raise AssertionError("datetime not properly determined from user input")


@pytest.mark.parametrize(
    ["dt_rep"],
    [
        (datetime.datetime(2024, 11, 12, 14, 31, 29),),
        (np.datetime64("2024-11-12T14:31:29"),),
        (pd.Timestamp("2024-11-12T14:31:29"),),
        ("2024-11-12T14:31:29",),
        ("2024/11/12T14:31:29",),
        ("11-12-2024 14:31:29",),
        ("11/12/2024 14:31:29",),
    ],
)
def test_datetime_userinput_parsing(dt_rep):
    valid = datetime.datetime(2024, 11, 12, 14, 31, 29)
    dt_obj = ModelTime.parse_datetime(dt_rep)
    if dt_obj != valid:
        raise AssertionError("datetime not properly determined from user input")


@pytest.mark.parametrize(
    "unit_name, user_inputs",
    (
        ("years", ["years", "YeaR", "yaEr", "ayer", "y", "yr", 5]),
        ("days", ["days", "Day", "dyAs", "dysa", "d", 4]),
        ("hours", ["hours", "Hour", "huors", "h", "hrs", 3]),
        ("minutes", ["minutes", "MinUte", "minte", "m", "min", 2]),
        ("seconds", ["seconds", "Second", "sedcon", "s", "sec", 1]),
        ("unknown", ["unkonwn", "undefined", "u", 0]),
    ),
)
def test_timeunits_user_input_parsing(unit_name, user_inputs):
    for user_input in user_inputs:
        mt_unit = ModelTime.parse_timeunits(user_input)
        if mt_unit != unit_name:
            raise AssertionError("Units are unable to be determined from user input")


def test_set_datetime_and_units():
    nrec = 2
    perlen = np.full((nrec,), 10)
    nstp = np.full((nrec,), 2, dtype=int)

    unix_t0 = datetime.datetime(1970, 1, 1)
    new_dt = datetime.datetime(2024, 11, 12)

    init_units = "unknown"
    new_units = "days"

    mt = ModelTime(perlen=perlen, nstp=nstp)

    if mt.time_units != init_units:
        raise AssertionError("time_units None condition not being set to unknown")

    if mt.start_datetime != unix_t0:
        raise AssertionError("start_datetime None condition not being set to 1/1/1970")

    mt.time_units = new_units
    mt.start_datetime = new_dt

    if mt.time_units != new_units:
        raise AssertionError("time_units setting not behaving properly")

    if mt.start_datetime != new_dt:
        raise AssertionError("start_datetime setting not behaving properly")


@pytest.mark.parametrize(
    "kperkstp,totim0",
    [((0, None), 30.25), ((1, 3), 60.5), ((4, 0), 126.246), ((11, None), 363.00)],
)
def test_get_elapsed_time_from_kper_kstp(kperkstp, totim0):
    nrec = 12
    perlen = np.full((nrec,), 30.25)
    nstp = np.full((nrec,), 4, dtype=int)
    tslen = np.full((nrec,), 1.25)
    start_datetime = "2023-12-31t23:59:59"
    time_unit = "days"

    mt = ModelTime(
        perlen, nstp, tslen, time_units=time_unit, start_datetime=start_datetime
    )

    kper, kstp = kperkstp
    totim = mt.get_elapsed_time(kper, kstp=kstp)
    if np.abs(totim - totim0) > 0.01:
        raise AssertionError("Incorrect totim calculation from get_elapsed_time()")


@pytest.mark.parametrize(
    "kperkstp, dt0",
    [
        ((0, None), datetime.datetime(2024, 1, 31, 5, 59, 59)),
        ((1, 3), datetime.datetime(2024, 3, 1, 11, 59, 59)),
        ((4, 0), datetime.datetime(2024, 5, 6, 5, 55, 6)),
        ((11, None), datetime.datetime(2024, 12, 28, 23, 59, 59)),
    ],
)
def test_get_datetime_from_kper_kstp(kperkstp, dt0):
    nrec = 12
    perlen = np.full((nrec,), 30.25)
    nstp = np.full((nrec,), 4, dtype=int)
    tslen = np.full((nrec,), 1.25)
    start_datetime = "2023-12-31t23:59:59"
    time_unit = "days"

    mt = ModelTime(
        perlen, nstp, tslen, time_units=time_unit, start_datetime=start_datetime
    )

    kper, kstp = kperkstp
    dt = mt.get_datetime(kper, kstp=kstp)
    td = dt - dt0
    if np.abs(td.seconds) > 2:
        raise AssertionError("Datetime calculation incorrect for get_datetime()")


@pytest.mark.parametrize(
    "kperkstp, dt",
    [
        ((0, 3), datetime.datetime(2024, 1, 31, 5, 59, 58)),
        ((1, 3), datetime.datetime(2024, 3, 1, 11, 59, 58)),
        ((4, 0), datetime.datetime(2024, 5, 6, 5, 55, 5)),
        ((11, 3), datetime.datetime(2024, 12, 28, 23, 59, 58)),
    ],
)
def test_datetime_intersect(kperkstp, dt):
    nrec = 12
    perlen = np.full((nrec,), 30.25)
    nstp = np.full((nrec,), 4, dtype=int)
    tslen = np.full((nrec,), 1.25)
    start_datetime = "2023-12-31t23:59:59"
    time_unit = "days"

    mt = ModelTime(
        perlen, nstp, tslen, time_units=time_unit, start_datetime=start_datetime
    )

    kper0, kstp0 = kperkstp
    kper, kstp = mt.intersect(dt)
    if kper != kper0 or kstp != kstp0:
        raise AssertionError(
            "intersect() not returning correct stress-period and timestep"
        )


@pytest.mark.parametrize(
    "kperkstp,totim",
    [((0, 3), 30.2), ((1, 3), 60.4), ((4, 0), 126.23), ((11, 3), 362.9)],
)
def test_totim_intersect(kperkstp, totim):
    nrec = 12
    perlen = np.full((nrec,), 30.25)
    nstp = np.full((nrec,), 4, dtype=int)
    tslen = np.full((nrec,), 1.25)
    start_datetime = "2023-12-31t23:59:59"
    time_unit = "days"

    mt = ModelTime(
        perlen, nstp, tslen, time_units=time_unit, start_datetime=start_datetime
    )
    kper0, kstp0 = kperkstp
    kper, kstp = mt.intersect(totim=totim)
    if kper != kper0 or kstp != kstp0:
        raise AssertionError(
            "intersect() not returning correct stress-period and timestep"
        )


def test_mf2005_modeltime():
    nlay = 1
    nrow = 9
    ncol = 9
    delc = 10
    delr = 10
    top = np.full((nrow, ncol), 100)
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)
    strt = np.full(botm.shape, np.max(top) - 5)
    nper = 5
    nstp = [5, 4, 5, 5, 5]
    perlen = [31, 28, 31, 30, 31]
    start_datetime = datetime.datetime(2024, 1, 1)
    start_datetime_str = "1/1/2024"

    ml = flopy.modflow.Modflow(modelname="dev_time", model_ws="dev_time")

    dis = flopy.modflow.ModflowDis(
        ml,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        perlen=perlen,
        nstp=nstp,
        steady=False,
        itmuni=4,
        lenuni=2,
        start_datetime=start_datetime_str,
    )
    bas = flopy.modflow.ModflowBas(ml, ibound=idomain, strt=strt)

    modeltime = ml.modeltime
    if modeltime.start_datetime != start_datetime:
        raise AssertionError("start_datetime improperly stored")

    result = modeltime.intersect("3/06/2024 23:59:59")
    if result != (2, 0):
        raise AssertionError("ModelTime intersect not working correctly")


def test_mf6_modeltime():
    nlay = 1
    nrow = 9
    ncol = 9
    delc = 10
    delr = 10
    top = np.full((nrow, ncol), 100)
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)
    nper = 5
    nstp = [5, 4, 5, 5, 5]
    perlen = [31, 28, 31, 30, 31]
    period_data = [(p, nstp[ix], 1) for ix, p in enumerate(perlen)]
    start_datetime = datetime.datetime(2024, 1, 1)
    start_datetime_str = "2024-1-1t00:00:00"

    sim = flopy.mf6.MFSimulation()
    tdis = flopy.mf6.ModflowTdis(
        sim,
        time_units="days",
        start_date_time=start_datetime_str,
        nper=nper,
        perioddata=period_data,
    )
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
    )

    modeltime = gwf.modeltime
    if modeltime.start_datetime != start_datetime:
        raise AssertionError("start_datetime improperly stored")

    result = modeltime.intersect("3/06/2024 23:59:59")
    if result != (2, 0):
        raise AssertionError("ModelTime intersect not working correctly")


def test_from_headers_test006_gwf3(example_data_path):
    ws = example_data_path / "mf6" / "test006_gwf3" / "expected_output"
    sim = MFSimulation.load(sim_ws=ws.parent)
    tdis = sim.tdis
    hf = HeadFile(ws / "flow_adj.hds")
    mt = ModelTime.from_headers(hf.recordarray)

    assert np.isclose(hf.recordarray[-1]["totim"], sum(mt.perlen))
    assert np.allclose(mt.perlen, tdis.perioddata.get_data()["perlen"])
    assert np.allclose(mt.nstp, tdis.perioddata.get_data()["nstp"])
    assert np.allclose(mt.tsmult, tdis.perioddata.get_data()["tsmult"])


def test_from_headers_test027_TimeseriesTest(example_data_path):
    ws = example_data_path / "mf6" / "test027_TimeseriesTest" / "expected_output"
    sim = MFSimulation.load(sim_ws=ws.parent)
    tdis = sim.tdis
    hf = HeadFile(ws / "timeseriestest_adj.hds")
    mt = ModelTime.from_headers(hf.recordarray)

    assert np.isclose(hf.recordarray[-1]["totim"], sum(mt.perlen))
    assert np.allclose(mt.perlen, tdis.perioddata.get_data()["perlen"])
    assert np.allclose(mt.nstp, tdis.perioddata.get_data()["nstp"])
    assert np.allclose(mt.tsmult, tdis.perioddata.get_data()["tsmult"])


def test_from_headers_test005_advgw_tidal(example_data_path):
    ws = example_data_path / "mf6" / "test005_advgw_tidal" / "expected_output"
    sim = MFSimulation.load(sim_ws=ws.parent)
    tdis = sim.tdis
    hf = HeadFile(ws / "AdvGW_tidal.hds")
    mt = ModelTime.from_headers(hf.recordarray)

    assert np.isclose(hf.recordarray[-1]["totim"], sum(mt.perlen))
    assert np.allclose(mt.perlen, tdis.perioddata.get_data()["perlen"])
    assert np.allclose(mt.nstp, tdis.perioddata.get_data()["nstp"])
    assert np.allclose(mt.tsmult, tdis.perioddata.get_data()["tsmult"])


def test_reverse(example_data_path):
    ws = example_data_path / "mf6" / "test005_advgw_tidal" / "expected_output"
    sim = MFSimulation.load(sim_ws=ws.parent)
    tdis = sim.tdis
    hf = HeadFile(ws / "AdvGW_tidal.hds")
    mt = ModelTime.from_headers(hf.recordarray)
    rev = mt.reverse()

    assert np.isclose(hf.recordarray[-1]["totim"], sum(rev.perlen))
    assert np.allclose(rev.perlen[::-1], tdis.perioddata.get_data()["perlen"])
    assert np.allclose(rev.nstp[::-1], tdis.perioddata.get_data()["nstp"])
