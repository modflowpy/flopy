"""Tests for the MfUsgTib (Transient IBOUND) package."""

import numpy as np
import pytest

import flopy
from flopy.mfusg import MfUsg, MfUsgDis, MfUsgDisU, MfUsgTib


def _make_model(nper=3):
    """Create a minimal MfUsg model with DIS for testing."""
    m = MfUsg(modelname="tib_test", model_ws=".")
    MfUsgDis(
        m,
        nlay=1,
        nrow=10,
        ncol=10,
        nper=nper,
        perlen=[1.0] * nper,
        nstp=[1] * nper,
        tsmult=[1.0] * nper,
        steady=[True] * nper,
    )
    return m


def test_tib_construction():
    """Create model + TIB, verify package is attached."""
    m = _make_model()
    spd = {0: {"ib0": np.array([5, 10])}}
    tib = MfUsgTib(m, stress_period_data=spd)

    assert tib is not None
    assert m.get_package("TIB") is tib
    assert tib._ftype() == "TIB"
    assert tib._defaultunit() == 160


def test_tib_write_and_load(tmp_path):
    """Round-trip: write -> load -> compare all 6 categories."""
    m = _make_model(nper=2)
    m.model_ws = str(tmp_path)

    spd = {
        0: {
            "ib0": np.array([5, 10, 15]),
            "ib1": [(20, "HEAD", 10.5), (25, "AVHEAD")],
            "ibm1": [(30, "HEAD", 5.0)],
            "icb0": np.array([5]),
            "icb1": [(20, "CONC", [0.5, 0.2])],
            "icbm1": [(30, "AVCONC")],
        },
        1: {
            "ib0": np.array([40]),
        },
    }
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    # Load
    m2 = _make_model(nper=2)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=2)

    # Compare stress period 0
    d0 = tib2.stress_period_data[0]
    np.testing.assert_array_equal(d0["ib0"], np.array([5, 10, 15]))
    assert d0["ib1"][0] == (20, "HEAD", 10.5)
    assert d0["ib1"][1] == (25, "AVHEAD")
    assert d0["ibm1"][0] == (30, "HEAD", 5.0)
    np.testing.assert_array_equal(d0["icb0"], np.array([5]))
    assert d0["icb1"][0][0] == 20
    assert d0["icb1"][0][1] == "CONC"
    np.testing.assert_allclose(d0["icb1"][0][2], [0.5, 0.2])
    assert d0["icbm1"][0] == (30, "AVCONC")

    # Compare stress period 1
    d1 = tib2.stress_period_data[1]
    np.testing.assert_array_equal(d1["ib0"], np.array([40]))
    assert "ib1" not in d1
    assert "ibm1" not in d1


def test_tib_with_transport(tmp_path):
    """Include ICBUND data with CONC values for multiple species."""
    m = _make_model(nper=1)
    m.model_ws = str(tmp_path)

    spd = {
        0: {
            "icb0": np.array([10, 20, 30]),
            "icb1": [
                (40, "CONC", [1.0, 2.0, 3.0]),
                (50, "AVCONC"),
            ],
            "icbm1": [(60, "CONC", [0.1])],
        },
    }
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    m2 = _make_model(nper=1)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=1)

    d0 = tib2.stress_period_data[0]
    np.testing.assert_array_equal(d0["icb0"], np.array([10, 20, 30]))
    assert d0["icb1"][0][0] == 40
    assert d0["icb1"][0][1] == "CONC"
    np.testing.assert_allclose(d0["icb1"][0][2], [1.0, 2.0, 3.0])
    assert d0["icb1"][1] == (50, "AVCONC")
    assert d0["icbm1"][0][0] == 60
    np.testing.assert_allclose(d0["icbm1"][0][2], [0.1])


def test_tib_empty_stress_periods(tmp_path):
    """Stress periods with no changes write all-zero counts."""
    m = _make_model(nper=3)
    m.model_ws = str(tmp_path)

    # Only stress period 1 has data; 0 and 2 are empty
    spd = {1: {"ib0": np.array([5])}}
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    m2 = _make_model(nper=3)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=3)

    assert 0 not in tib2.stress_period_data
    assert 1 in tib2.stress_period_data
    np.testing.assert_array_equal(tib2.stress_period_data[1]["ib0"], np.array([5]))
    assert 2 not in tib2.stress_period_data


def test_tib_get_empty():
    """Verify template dict structure from get_empty."""
    data = MfUsgTib.get_empty(nib0=3, nib1=2, nicb1=1)

    assert "ib0" in data
    assert data["ib0"].shape == (3,)
    assert data["ib0"].dtype == np.int32

    assert "ib1" in data
    assert len(data["ib1"]) == 2

    assert "icb1" in data
    assert len(data["icb1"]) == 1

    # Keys not requested should be absent
    assert "ibm1" not in data
    assert "icb0" not in data
    assert "icbm1" not in data


def test_tib_no_options(tmp_path):
    """ib1/ibm1 entries with no HEAD/AVHEAD keyword (node-only tuples)."""
    m = _make_model(nper=1)
    m.model_ws = str(tmp_path)

    spd = {
        0: {
            "ib1": [(10,), (20,)],
            "ibm1": [(30,)],
        },
    }
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    m2 = _make_model(nper=1)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=1)

    d0 = tib2.stress_period_data[0]
    assert d0["ib1"][0] == (10,)
    assert d0["ib1"][1] == (20,)
    assert d0["ibm1"][0] == (30,)


def test_tib_model_type_validation():
    """Non-MfUsg model raises AssertionError."""
    m = flopy.modflow.Modflow()
    with pytest.raises(AssertionError, match=r"flopy\.mfusg\.MfUsg"):
        MfUsgTib(m)


def test_tib_bare_integer_nodes(tmp_path):
    """ib1/ibm1 entries as bare integers (not wrapped in tuples)."""
    m = _make_model(nper=1)
    m.model_ws = str(tmp_path)

    spd = {
        0: {
            "ib1": [10, 20],
            "ibm1": [30],
        },
    }
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    m2 = _make_model(nper=1)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=1)

    d0 = tib2.stress_period_data[0]
    assert d0["ib1"][0] == (10,)
    assert d0["ib1"][1] == (20,)
    assert d0["ibm1"][0] == (30,)


def _make_disu_model(nper=1):
    """Create a minimal MfUsg model with DISU (4 nodes, 1 layer)."""
    m = MfUsg(modelname="tib_disu_test", structured=False, model_ws=".")
    nodes = 4
    # Simple 4-node line: 0-1-2-3, iac includes self-connection
    iac = np.array([2, 3, 3, 2])  # sum = 10 = njag
    njag = int(iac.sum())
    ja = np.array([0, 1, 1, 0, 2, 2, 1, 3, 3, 2])
    cl12 = np.ones(njag, dtype=np.float32) * 0.5
    fahl = np.ones(njag, dtype=np.float32)
    MfUsgDisU(
        m,
        nodes=nodes,
        nlay=1,
        njag=njag,
        iac=iac,
        ja=ja,
        cl12=cl12,
        fahl=fahl,
        nper=nper,
        perlen=[1.0] * nper,
        nstp=[1] * nper,
        tsmult=[1.0] * nper,
        steady=[True] * nper,
    )
    return m


def test_tib_disu_write_and_load(tmp_path):
    """Round-trip with unstructured grid (DISU)."""
    m = _make_disu_model(nper=2)
    m.model_ws = str(tmp_path)

    spd = {
        0: {
            "ib0": np.array([0, 3]),
            "ib1": [(1, "HEAD", 5.0)],
            "ibm1": [(2, "AVHEAD")],
        },
        1: {
            "ib1": [(0,), (3,)],
        },
    }
    tib = MfUsgTib(m, stress_period_data=spd)
    tib.write_file()

    m2 = _make_disu_model(nper=2)
    m2.model_ws = str(tmp_path)
    tib2 = MfUsgTib.load(tmp_path / f"{m.name}.tib", m2, nper=2)

    d0 = tib2.stress_period_data[0]
    np.testing.assert_array_equal(d0["ib0"], np.array([0, 3]))
    assert d0["ib1"][0] == (1, "HEAD", 5.0)
    assert d0["ibm1"][0] == (2, "AVHEAD")

    d1 = tib2.stress_period_data[1]
    assert d1["ib1"][0] == (0,)
    assert d1["ib1"][1] == (3,)


def test_tib_invalid_key():
    """Invalid key in stress_period_data raises ValueError."""
    m = _make_model()
    with pytest.raises(ValueError, match="Unknown key"):
        MfUsgTib(m, stress_period_data={0: {"bad_key": [1, 2]}})
