import os

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_pkg
from modflow_devtools.misc import has_pkg

from flopy.modflow import Modflow, ModflowHyd
from flopy.utils import HydmodObs, Mf6Obs


@pytest.fixture
def mf6_obs_model_path(example_data_path):
    return example_data_path / "mf6_obs"


@pytest.fixture
def hydmod_model_path(example_data_path):
    return example_data_path / "hydmod_test"


def test_hydmodfile_create(function_tmpdir):
    m = Modflow("test", model_ws=function_tmpdir)
    hyd = ModflowHyd(m)
    m.hyd.write_file()
    pth = function_tmpdir / "test.hyd"
    hydload = ModflowHyd.load(pth, m)
    assert np.array_equal(
        hyd.obsdata, hydload.obsdata
    ), "Written hydmod data not equal to loaded hydmod data"

    # test obsdata as recarray
    obsdata = np.array(
        [(3208, "BAS", "HD", "I", 4, 630486.19, 5124733.18, "well1")],
        dtype=[
            ("index", "<i8"),
            ("pckg", "O"),
            ("arr", "O"),
            ("intyp", "O"),
            ("klay", "<i8"),
            ("xl", "<f8"),
            ("yl", "<f8"),
            ("hydlbl", "O"),
        ],
    ).view(np.recarray)
    hyd = ModflowHyd(m, obsdata=obsdata)

    # test obsdata as object array
    obsdata = np.array(
        [("BAS", "HD", "I", 4, 630486.19, 5124733.18, "well1")], dtype=object
    )
    hyd = ModflowHyd(m, obsdata=obsdata)


def test_hydmodfile_load(function_tmpdir, hydmod_model_path):
    model = "test1tr.nam"
    m = Modflow.load(
        model, version="mf2005", model_ws=hydmod_model_path, verbose=True
    )
    hydref = m.hyd
    assert isinstance(
        hydref, ModflowHyd
    ), "Did not load hydmod package...test1tr.hyd"

    m.change_model_ws(function_tmpdir)
    m.hyd.write_file()

    pth = hydmod_model_path / "test1tr.hyd"
    hydload = ModflowHyd.load(pth, m)
    assert np.array_equal(
        hydref.obsdata, hydload.obsdata
    ), "Written hydmod data not equal to loaded hydmod data"


def test_hydmodfile_read(hydmod_model_path):
    pth = hydmod_model_path / "test1tr.hyd.gitbin"
    h = HydmodObs(pth)
    assert isinstance(h, HydmodObs)

    ntimes = h.get_ntimes()
    assert ntimes == 101, "Not enough times in hydmod file ()...".format()

    times = h.get_times()
    assert len(times) == 101, "Not enough times in hydmod file ()...".format()

    nitems = h.get_nobs()
    assert nitems == 8, "Not enough records in hydmod file ()...".format()

    labels = h.get_obsnames()
    assert len(labels) == 8, "Not enough labels in hydmod file ()...".format()
    print(labels)

    for idx in range(ntimes):
        data = h.get_data(idx=idx)
        assert data.shape == (1,), "data shape is not (1,)"

    for time in times:
        data = h.get_data(totim=time)
        assert data.shape == (1,), "data shape is not (1,)"

    for label in labels:
        data = h.get_data(obsname=label)
        assert data.shape == (
            len(times),
        ), f"data shape is not ({len(times)},)"

    data = h.get_data()
    assert data.shape == (len(times),), f"data shape is not ({len(times)},)"
    assert (
        len(data.dtype.names) == nitems + 1
    ), f"data column length is not {len(nitems + 1)}"

    for idx in range(ntimes):
        df = h.get_dataframe(idx=idx, timeunit="S")
        assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
        assert df.shape == (1, 9), "data shape is not (1, 9)"

    for time in times:
        df = h.get_dataframe(totim=time, timeunit="S")
        assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
        assert df.shape == (1, 9), "data shape is not (1, 9)"

    df = h.get_dataframe(timeunit="S")
    assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
    assert df.shape == (101, 9), "data shape is not (101, 9)"


def test_mf6obsfile_read(mf6_obs_model_path):
    txt = "binary mf6 obs"
    files = ["maw_obs.gitbin", "maw_obs.gitcsv"]
    binfile = [True, False]

    for idx in range(len(files)):
        pth = mf6_obs_model_path / files[idx]
        h = Mf6Obs(pth, isBinary=binfile[idx])
        assert isinstance(h, Mf6Obs)

        ntimes = h.get_ntimes()
        assert (
            ntimes == 3
        ), f"Not enough times in {txt} file...{os.path.basename(pth)}"

        times = h.get_times()
        assert len(times) == 3, "Not enough times in {} file...{}".format(
            txt, os.path.basename(pth)
        )

        nitems = h.get_nobs()
        assert nitems == 1, "Not enough records in {} file...{}".format(
            txt, os.path.basename(pth)
        )

        labels = h.get_obsnames()
        assert len(labels) == 1, "Not enough labels in {} file...{}".format(
            txt, os.path.basename(pth)
        )
        print(labels)

        for idx in range(ntimes):
            data = h.get_data(idx=idx)
            assert data.shape == (1,), "data shape is not (1,)"

        for time in times:
            data = h.get_data(totim=time)
            assert data.shape == (1,), "data shape is not (1,)"

        for label in labels:
            data = h.get_data(obsname=label)
            assert data.shape == (
                len(times),
            ), f"data shape is not ({len(times)},)"

        data = h.get_data()
        assert data.shape == (
            len(times),
        ), f"data shape is not ({len(times)},)"
        assert (
            len(data.dtype.names) == nitems + 1
        ), f"data column length is not {len(nitems + 1)}"

        for idx in range(ntimes):
            df = h.get_dataframe(idx=idx, timeunit="S")
            assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
            assert df.shape == (1, 2), "data shape is not (1, 2)"

        for time in times:
            df = h.get_dataframe(totim=time, timeunit="S")
            assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
            assert df.shape == (1, 2), "data shape is not (1, 2)"

        df = h.get_dataframe(timeunit="S")
        assert isinstance(df, pd.DataFrame), "A DataFrame was not returned"
        assert df.shape == (3, 2), "data shape is not (3, 2)"
