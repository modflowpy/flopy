import os

import numpy as np
import pandas as pd
import pytest

from autotest.conftest import get_example_data_path
from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.utils.binaryfile import CellBudgetFile

# test low-level CellBudgetFile._build_index() method


def test_cellbudgetfile_build_index_classic(example_data_path):
    """Test reading "classic" budget file, without "COMPACT BUDGET" option."""
    pth = example_data_path / "mt3d_test/mf2kmt3d/mnw/t5.cbc"
    with CellBudgetFile(pth) as cbc:
        pass
    assert cbc.nrow == 101
    assert cbc.ncol == 101
    assert cbc.nlay == 3
    assert cbc.nper == 1
    assert cbc.totalbytes == 122_448
    assert len(cbc.recordarray) == 1
    assert type(cbc.recordarray) == np.ndarray
    assert cbc.recordarray.dtype == np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("text", "S16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("nlay", "i4"),
            ("imeth", "i4"),
            ("delt", "f4"),
            ("pertim", "f4"),
            ("totim", "f4"),
            ("modelnam", "S16"),
            ("paknam", "S16"),
            ("modelnam2", "S16"),
            ("paknam2", "S16"),
        ]
    )
    assert len(cbc.recorddict) == 1
    list_recorddict = list(cbc.recorddict.items())
    # fmt: off
    assert list_recorddict == [(
        (1, 1, b"             MNW", 101, 101, 3, 0, 0.0, 0.0, -1.0, b"", b"", b"", b""),
        36)
    ]
    # fmt: on
    assert cbc.times == []
    assert cbc.kstpkper == [(1, 1)]
    np.testing.assert_array_equal(cbc.iposheader, np.array([0]))
    assert cbc.iposheader.dtype == np.int64
    np.testing.assert_array_equal(cbc.iposarray, np.array([36]))
    assert cbc.iposarray.dtype == np.int64
    assert cbc.textlist == [b"             MNW"]
    assert cbc.imethlist == [0]
    assert cbc.paknamlist_from == [b""]
    assert cbc.paknamlist_to == [b""]
    pd.testing.assert_frame_equal(
        cbc.headers,
        pd.DataFrame(
            {
                "kstp": np.array([1], np.int32),
                "kper": np.array([1], np.int32),
                "text": ["MNW"],
                "ncol": np.array([101], np.int32),
                "nrow": np.array([101], np.int32),
                "nlay": np.array([3], np.int32),
            },
            index=[36],
        ),
    )


def test_cellbudgetfile_build_index_compact(example_data_path):
    """Test reading mfntw budget file, with "COMPACT BUDGET" option."""
    pth = example_data_path / "freyberg_multilayer_transient" / "freyberg.cbc"
    with CellBudgetFile(pth) as cbc:
        pass
    assert cbc.nrow == 40
    assert cbc.ncol == 20
    assert cbc.nlay == 3
    assert cbc.nper == 1097
    assert cbc.totalbytes == 42_658_384
    assert len(cbc.recordarray) == 5483
    assert type(cbc.recordarray) == np.ndarray
    assert cbc.recordarray.dtype == np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("text", "S16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("nlay", "i4"),
            ("imeth", "i4"),
            ("delt", "f4"),
            ("pertim", "f4"),
            ("totim", "f4"),
            ("modelnam", "S16"),
            ("paknam", "S16"),
            ("modelnam2", "S16"),
            ("paknam2", "S16"),
        ]
    )
    assert len(cbc.recorddict) == 5483
    # check first and last recorddict
    list_recorddict = list(cbc.recorddict.items())
    # fmt: off
    assert list_recorddict[0] == (
        (1, 1, b"   CONSTANT HEAD", 20, 40, -3, 2, 1.0, 1.0, 1.0, b"", b"", b"", b""),
        52,
    )
    assert list_recorddict[-1] == (
        (1, 1097, b"FLOW LOWER FACE ", 20, 40, -3, 1, 1.0, 1.0, 1097.0, b"", b"", b"", b""),  # noqa
        42648784,
    )
    # fmt: on
    assert cbc.times == list((np.arange(1097) + 1).astype(np.float32))
    assert cbc.kstpkper == [(1, kper + 1) for kper in range(1097)]
    # fmt: off
    expected_iposheader = np.cumsum([0]
            + ([296] + [9652] * 4) * 1095
            + [296] + [9652] * 3
            + [296] + [9652] * 2)
    # fmt: on
    np.testing.assert_array_equal(cbc.iposheader, expected_iposheader)
    assert cbc.iposheader.dtype == np.int64
    np.testing.assert_array_equal(cbc.iposarray, expected_iposheader + 52)
    assert cbc.iposarray.dtype == np.int64
    assert cbc.textlist == [
        b"   CONSTANT HEAD",
        b"FLOW RIGHT FACE ",
        b"FLOW FRONT FACE ",
        b"FLOW LOWER FACE ",
        b"         STORAGE",
    ]
    assert cbc.imethlist == [2, 1, 1, 1, 1]
    assert cbc.paknamlist_from == [b""]
    assert cbc.paknamlist_to == [b""]
    # check first and last row of data frame
    pd.testing.assert_frame_equal(
        cbc.headers.iloc[[0, -1]],
        pd.DataFrame(
            {
                "kstp": np.array([1, 1], np.int32),
                "kper": np.array([1, 1097], np.int32),
                "text": ["CONSTANT HEAD", "FLOW LOWER FACE"],
                "ncol": np.array([20, 20], np.int32),
                "nrow": np.array([40, 40], np.int32),
                "nlay": np.array([-3, -3], np.int32),
                "imeth": np.array([2, 1], np.int32),
                "delt": np.array([1.0, 1.0], np.float32),
                "pertim": np.array([1.0, 1.0], np.float32),
                "totim": np.array([1.0, 1097.0], np.float32),
            },
            index=[52, 42648784],
        ),
    )


def test_cellbudgetfile_build_index_mf6(example_data_path):
    cbb_file = (
        example_data_path
        / "mf6"
        / "test005_advgw_tidal"
        / "expected_output"
        / "AdvGW_tidal.cbc"
    )
    with CellBudgetFile(cbb_file) as cbb:
        pass
    assert cbb.nrow == 15
    assert cbb.ncol == 10
    assert cbb.nlay == 3
    assert cbb.nper == 4
    assert cbb.totalbytes == 13_416_552
    assert len(cbb.recordarray) == 3610
    assert type(cbb.recordarray) == np.ndarray
    assert cbb.recordarray.dtype == np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("text", "S16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("nlay", "i4"),
            ("imeth", "i4"),
            ("delt", "f8"),
            ("pertim", "f8"),
            ("totim", "f8"),
            ("modelnam", "S16"),
            ("paknam", "S16"),
            ("modelnam2", "S16"),
            ("paknam2", "S16"),
        ]
    )
    assert len(cbb.recorddict) == 3610
    # check first and last recorddict
    list_recorddict = list(cbb.recorddict.items())
    # fmt: off
    assert list_recorddict[0] == (
        (1, 1, b"          STO-SS", 10, 15, -3, 1,
         1.0, 1.0, 1.0,
         b"", b"", b"", b""),
        64,
    )
    assert list_recorddict[-1] == (
        (120, 4, b"             EVT", 10, 15, -3, 6,
         0.08333333333333333, 10.000000000000002, 30.99999999999983,
         b"GWF_1           ", b"GWF_1           ", b"GWF_1           ", b"EVT             "),  # noqa
        13414144,
    )
    # fmt: on
    assert isinstance(cbb.times, list)
    np.testing.assert_allclose(cbb.times, np.linspace(1.0, 31, 361))
    # fmt: off
    assert cbb.kstpkper == (
        [(1, 1)]
        + [(kstp + 1, 2) for kstp in range(120)]
        + [(kstp + 1, 3) for kstp in range(120)]
        + [(kstp + 1, 4) for kstp in range(120)]
    )
    # fmt: on
    # this file has a complex structure, so just look at unique ipos spacings
    assert set(np.diff(cbb.iposheader)) == (
        {184, 264, 304, 384, 456, 616, 632, 1448, 2168, 2536, 3664, 21664}
    )
    assert cbb.iposheader[0] == 0
    assert cbb.iposheader.dtype == np.int64
    assert set(np.diff(cbb.iposarray)) == (
        {184, 264, 304, 384, 456, 616, 632, 1448, 2168, 2472, 3664, 21728}
    )
    assert cbb.iposarray[0] == 64
    assert cbb.iposarray.dtype == np.int64
    # variable size headers depending on imeth
    header_sizes = np.full(3610, 64)
    header_sizes[cbb.recordarray["imeth"] == 6] = 128
    np.testing.assert_array_equal(cbb.iposheader + header_sizes, cbb.iposarray)
    assert cbb.textlist == [
        b"          STO-SS",
        b"          STO-SY",
        b"    FLOW-JA-FACE",
        b"             WEL",
        b"             RIV",
        b"             GHB",
        b"             RCH",
        b"             EVT",
    ]
    assert cbb.imethlist == [1, 1, 1, 6, 6, 6, 6, 6]
    assert cbb.paknamlist_from == [b"", b"GWF_1           "]
    assert cbb.paknamlist_to == [
        b"",
        b"WEL             ",
        b"RIV             ",
        b"GHB-TIDAL       ",
        b"RCH-ZONE_1      ",
        b"RCH-ZONE_2      ",
        b"RCH-ZONE_3      ",
        b"EVT             ",
    ]
    # check first and last row of data frame
    pd.testing.assert_frame_equal(
        cbb.headers.iloc[[0, -1]],
        pd.DataFrame(
            {
                "kstp": np.array([1, 120], np.int32),
                "kper": np.array([1, 4], np.int32),
                "text": ["STO-SS", "EVT"],
                "ncol": np.array([10, 10], np.int32),
                "nrow": np.array([15, 15], np.int32),
                "nlay": np.array([-3, -3], np.int32),
                "imeth": np.array([1, 6], np.int32),
                "delt": [1.0, 0.08333333333333333],
                "pertim": [1.0, 10.0],
                "totim": [1.0, 31.0],
                "modelnam": ["", "GWF_1"],
                "paknam": ["", "GWF_1"],
                "modelnam2": ["", "GWF_1"],
                "paknam2": ["", "EVT"],
            },
            index=[64, 13414144],
        ),
    )


def test_cellbudgetfile_imeth_5(example_data_path):
    pth = example_data_path / "preserve_unitnums/testsfr2.ghb.cbc"
    with CellBudgetFile(pth) as cbc:
        pass
    # check a few components
    pd.testing.assert_index_equal(
        cbc.headers.index, pd.Index(np.arange(12, dtype=np.int64) * 156 + 64)
    )
    assert cbc.headers.text.unique().tolist() == ["HEAD DEP BOUNDS"]
    assert cbc.headers.imeth.unique().tolist() == [5]


@pytest.fixture
def zonbud_model_path(example_data_path):
    return example_data_path / "zonbud_examples"


def test_cellbudgetfile_get_indices_nrecords(example_data_path):
    pth = example_data_path / "freyberg_multilayer_transient" / "freyberg.cbc"
    with CellBudgetFile(pth) as cbc:
        pass
    assert cbc.get_indices() is None
    idxs = cbc.get_indices("constant head")
    assert type(idxs) == np.ndarray
    assert idxs.dtype == np.int64
    np.testing.assert_array_equal(idxs, list(range(0, 5476, 5)) + [5479])
    idxs = cbc.get_indices(b"         STORAGE")
    np.testing.assert_array_equal(idxs, list(range(4, 5475, 5)))

    assert len(cbc) == 5483
    with pytest.deprecated_call():
        assert cbc.nrecords == 5483
    with pytest.deprecated_call():
        assert cbc.get_nrecords() == 5483


def test_load_cell_budget_file_timeseries(example_data_path):
    pth = example_data_path / "mf2005_test" / "swiex1.gitzta"
    cbf = CellBudgetFile(pth, precision="single")
    ts = cbf.get_ts(text="ZETASRF  1", idx=(0, 0, 24))
    assert ts.shape == (4, 2)


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
    for i in range(idx, len(v)):
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

    with pytest.deprecated_call(match="use headers instead"):
        assert v2.list_records() is None
    with pytest.deprecated_call(match=r"drop_duplicates\(\) instead"):
        assert v2.list_unique_records() is None
    with pytest.deprecated_call(match=r"drop_duplicates\(\) instead"):
        assert v2.list_unique_packages(True) is None
    with pytest.deprecated_call(match=r"drop_duplicates\(\) instead"):
        assert v2.list_unique_packages(False) is None

    names = v2.get_unique_record_names(decode=True)

    cbcd2 = []
    for i in range(len(v2)):
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
    assert str(e.value).startswith("get_data() missing 1 required argument"), str(
        e.exception
    )

    t = v.get_data(text="STREAM LEAKAGE")
    assert len(t) == 30, "length of stream leakage data != 30"
    assert t[0].shape[0] == 36, "sfr budget data does not have 36 reach entries"

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
            assert np.array_equal(t0, t1), (
                "binary budget item {0} read using kstpkper != "
                "binary budget item {0} read using idx"
            ).format(record)

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
            assert np.array_equal(t0, t1), (
                "binary budget item {0} read using kstpkper != "
                "binary budget item {0} read using idx"
            ).format(record)
    v.close()


# reverse


@pytest.mark.skip(
    reason="failing, need to modify CellBudgetFile.reverse to support mf2005?"
)
def test_cellbudgetfile_reverse_mf2005(example_data_path, function_tmpdir):
    sim_name = "test1tr"

    # load simulation and extract tdis
    sim = MFSimulation.load(sim_name=sim_name, sim_ws=example_data_path / "mf2005_test")
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
    assert len(f) == 2
    assert len(rf) == 2

    # check data were reversed
    nrecords = len(f)
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


def test_read_mf6_budgetfile(example_data_path):
    cbb_file = (
        example_data_path
        / "mf6"
        / "test005_advgw_tidal"
        / "expected_output"
        / "AdvGW_tidal.cbc"
    )
    cbb = CellBudgetFile(cbb_file)
    rch_zone_1 = cbb.get_data(paknam2="rch-zone_1".upper())
    rch_zone_2 = cbb.get_data(paknam2="rch-zone_2".upper())
    rch_zone_3 = cbb.get_data(paknam2="rch-zone_3".upper())

    # ensure there is a record for each time step
    assert len(rch_zone_1) == 120 * 3 + 1
    assert len(rch_zone_2) == 120 * 3 + 1
    assert len(rch_zone_3) == 120 * 3 + 1
