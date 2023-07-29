import os

import numpy as np
import pytest

from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.utils.binaryfile import CellBudgetFile


@pytest.fixture
def zonbud_model_path(example_data_path):
    return example_data_path / "zonbud_examples"


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


# read context


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


# read


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


# readrecord


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


# reverse


@pytest.mark.skip(
    reason="failing, need to modify CellBudgetFile.reverse to support mf2005?"
)
def test_cellbudgetfile_reverse_mf2005(example_data_path, function_tmpdir):
    sim_name = "test1tr"

    # load simulation and extract tdis
    sim = MFSimulation.load(
        sim_name=sim_name, sim_ws=example_data_path / "mf2005_test"
    )
    tdis = sim.get_package("tdis")

    mf2005_model_path = example_data_path / sim_name
    cbc_fname = mf2005_model_path / f"{sim_name}.gitcbc"
    f = CellBudgetFile(cbc_fname, tdis=tdis)
    assert isinstance(f, CellBudgetFile)

    rf_name = "test1tr_rev.gitcbc"
    f.reverse(function_tmpdir / rf_name)
    rf = CellBudgetFile(function_tmpdir / rf_name)
    assert isinstance(rf, CellBudgetFile)


def test_cellbudgetfile_reverse_mf6(example_data_path, function_tmpdir):
    # load simulation and extract tdis
    sim_name = "test006_gwf3"
    sim = MFSimulation.load(
        sim_name=sim_name, sim_ws=example_data_path / "mf6" / sim_name
    )
    tdis = sim.get_package("tdis")

    # load cell budget file, providing tdis as kwarg
    model_path = example_data_path / "mf6" / sim_name
    file_stem = "flow_adj"
    file_path = model_path / "expected_output" / f"{file_stem}.cbc"
    f = CellBudgetFile(file_path, tdis=tdis)
    assert isinstance(f, CellBudgetFile)

    # reverse the file
    rf_name = f"{file_stem}_rev.cbc"
    f.reverse(filename=function_tmpdir / rf_name)
    rf = CellBudgetFile(function_tmpdir / rf_name)
    assert isinstance(rf, CellBudgetFile)

    # check that both files have the same number of records
    nrecords = f.get_nrecords()
    assert nrecords == rf.get_nrecords()

    # check data were reversed
    for idx in range(nrecords - 1, -1, -1):
        # check headers
        f_header = list(f.recordarray[nrecords - idx - 1])
        rf_header = list(rf.recordarray[idx])
        f_totim = f_header.pop(9)  # todo check totim
        rf_totim = rf_header.pop(9)
        assert f_header == rf_header

        # check data
        f_data = f.get_data(idx=idx)[0]
        rf_data = rf.get_data(idx=nrecords - idx - 1)[0]
        assert f_data.shape == rf_data.shape
        if f_data.ndim == 1:
            for row in range(len(f_data)):
                f_datum = f_data[row]
                rf_datum = rf_data[row]
                # flows should be negated
                rf_datum[2] = -rf_datum[2]
                assert f_datum == rf_datum
        else:
            # flows should be negated
            assert np.array_equal(f_data[0][0], -rf_data[0][0])
