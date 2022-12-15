import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from flopy.utils import CellBudgetFile, FormattedHeadFile, HeadFile


@pytest.fixture
def freyberg_model_path(example_data_path):
    return example_data_path / "freyberg"


@pytest.fixture
def zonbud_model_path(example_data_path):
    return example_data_path / "zonbud_examples"


def test_formattedfile_reference(example_data_path):
    h = FormattedHeadFile(
        str(example_data_path / "mf2005_test" / "test1tr.githds")
    )
    assert isinstance(h, FormattedHeadFile)
    h.mg.set_coord_info(xoff=1000.0, yoff=200.0, angrot=15.0)

    assert isinstance(h.plot(masked_values=[6999.000]), Axes)
    plt.close()


def test_formattedfile_read(tmpdir, example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    h = FormattedHeadFile(str(mf2005_model_path / "test1tr.githds"))
    assert isinstance(h, FormattedHeadFile)

    times = h.get_times()
    assert np.isclose(times[0], 1577880064.0)

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (49, 0), "kstpkper[0] != (49, 0)"

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(
        h0, h1
    ), "formatted head read using totim != head read using kstpkper"
    assert np.array_equal(
        h0, h2
    ), "formatted head read using totim != head read using idx"

    ts = h.get_ts((0, 7, 5))
    expected = 944.487
    assert np.isclose(
        ts[0, 1], expected, 1e-6
    ), f"time series value ({ts[0, 1]}) != {expected}"
    h.close()

    # Check error when reading empty file
    fname = str(tmpdir / "empty.githds")
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        FormattedHeadFile(fname)


def test_binaryfile_read(tmpdir, freyberg_model_path):
    h = HeadFile(str(freyberg_model_path / "freyberg.githds"))
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
    fname = str(tmpdir / "empty.githds")
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        HeadFile(fname)
    with pytest.raises(ValueError):
        HeadFile(fname, "head", "single")


def test_binaryfile_read_context(freyberg_model_path):
    hds_path = str(freyberg_model_path / "freyberg.githds")
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
    cbc_path = str(mf2005_model_path / "mnw1.gitcbc")
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
    v = CellBudgetFile(str(mf2005_model_path / "mnw1.gitcbc"))
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


def test_cellbudgetfile_position(tmpdir, zonbud_model_path):
    fpth = str(zonbud_model_path / "freyberg.gitcbc")
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
    opth = str(tmpdir / "end.cbc")
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
    fname = str(tmpdir / "empty.gitcbc")
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        CellBudgetFile(fname)


def test_cellbudgetfile_readrecord(example_data_path):
    mf2005_model_path = example_data_path / "mf2005_test"
    cbc_fname = str(mf2005_model_path / "test1tr.gitcbc")
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
    cbc_fname = str(mf2005_model_path / "test1tr.gitcbc")
    v = CellBudgetFile(cbc_fname)
    assert isinstance(v, CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 30, "length of kstpkper != 30"

    t = v.get_data(text="WELLS")
    assert len(t) == 30, "length of well data != 30"
    assert t[0].shape[0] == 10, "wel budget data does not have 10 well entries"

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
