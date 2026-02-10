"""
Tests for MODFLOW-USG TVM (Time-Varying Materials) and TIB (Transient IBOUND) packages.

These tests verify the functionality of the MfUsgTvm and MfUsgTib classes
for creating and writing MODFLOW-USG input files.
"""

import os
from unittest.mock import MagicMock, PropertyMock

import pytest

import flopy
from flopy.mfusg import MfUsg, MfUsgTib, MfUsgTvm


@pytest.fixture
def temp_model_ws(tmp_path):
    """Create temporary directory for model workspace."""
    return tmp_path


@pytest.fixture
def mock_mfusg_model(temp_model_ws):
    """
    Create a mock MODFLOW-USG model for testing.

    This avoids the complexity of setting up a full DISU package.
    """
    m = MfUsg(modelname="test_model", model_ws=str(temp_model_ws))

    # Mock nper property to return 5
    type(m).nper = PropertyMock(return_value=5)

    return m


class TestMfUsgTvm:
    """Tests for Time-Varying Materials (TVM) package."""

    def test_tvm_import(self):
        """Test that MfUsgTvm can be imported."""
        from flopy.mfusg import MfUsgTvm

        assert MfUsgTvm is not None

    def test_tvm_ftype(self):
        """Test that TVM file type is correct."""
        assert MfUsgTvm._ftype() == "TVM"

    def test_tvm_defaultunit(self):
        """Test that TVM default unit number is set."""
        assert MfUsgTvm._defaultunit() == 71

    def test_tvm_init_basic(self, mock_mfusg_model):
        """Test basic TVM initialization."""
        m = mock_mfusg_model

        tvm = MfUsgTvm(m)

        assert tvm.itvmprint == 0
        assert tvm.tvmlogbasehk == 0.0
        assert tvm.stress_period_data == {}
        assert "TVM" in m.get_package_list()

    def test_tvm_init_with_parameters(self, mock_mfusg_model):
        """Test TVM initialization with custom parameters."""
        m = mock_mfusg_model

        tvm = MfUsgTvm(
            m,
            itvmprint=2,
            tvmlogbasehk=10.0,
            tvmlogbasevka=-1.0,  # step function
            tvmlogbasess=0.0,
        )

        assert tvm.itvmprint == 2
        assert tvm.tvmlogbasehk == 10.0
        assert tvm.tvmlogbasevka == -1.0

    def test_tvm_init_with_stress_period_data(self, mock_mfusg_model):
        """Test TVM initialization with stress period data."""
        m = mock_mfusg_model

        spd = {
            0: {"hk": [(1, 1e-4), (2, 1e-4)]},
            3: {"hk": [(1, 1e-6)], "ss": [(1, 1e-5)]},
        }

        tvm = MfUsgTvm(m, stress_period_data=spd)

        assert len(tvm.stress_period_data) == 2
        assert 0 in tvm.stress_period_data
        assert 3 in tvm.stress_period_data
        assert len(tvm.stress_period_data[0]["hk"]) == 2

    def test_tvm_write_file(self, mock_mfusg_model):
        """Test TVM write_file creates output."""
        m = mock_mfusg_model

        spd = {
            0: {"hk": [(1, 1e-4), (2, 1e-4)]},
            2: {"hk": [(1, 1e-5)]},
        }

        tvm = MfUsgTvm(
            m,
            itvmprint=1,
            tvmlogbasehk=10.0,
            stress_period_data=spd,
        )

        tvm.write_file()

        # Check file exists
        tvm_path = os.path.join(m.model_ws, f"{m.name}.tvm")
        assert os.path.exists(tvm_path)

        # Check file content
        with open(tvm_path) as f:
            content = f.read()

        # Verify header line with interpolation options
        assert "1" in content  # itvmprint
        assert "10.000" in content  # tvmlogbasehk

        # Verify HK values are written
        assert "1.000000E-04" in content
        assert "1.000000E-05" in content

    def test_tvm_write_all_properties(self, mock_mfusg_model):
        """Test TVM write_file with all property types."""
        m = mock_mfusg_model

        spd = {
            0: {
                "hk": [(1, 1e-4)],
                "vka": [(1, 1e-5)],
                "ss": [(1, 1e-6)],
                "sy": [(1, 0.2)],
            },
        }

        tvm = MfUsgTvm(m, stress_period_data=spd)
        tvm.write_file()

        tvm_path = os.path.join(m.model_ws, f"{m.name}.tvm")
        with open(tvm_path) as f:
            content = f.read()

        # All values should be present
        assert "1.000000E-04" in content  # HK
        assert "1.000000E-05" in content  # VKA
        assert "1.000000E-06" in content  # SS
        assert "2.000000E-01" in content  # SY

    def test_tvm_model_type_validation(self, temp_model_ws):
        """Test that TVM requires MfUsg model type."""
        # Create a regular MODFLOW model (not USG)
        m = flopy.modflow.Modflow(modelname="test", model_ws=str(temp_model_ws))

        with pytest.raises(AssertionError, match="MfUsg"):
            MfUsgTvm(m)

    def test_tvm_load_not_implemented(self, mock_mfusg_model):
        """Test that TVM load raises NotImplementedError."""
        m = mock_mfusg_model

        with pytest.raises(NotImplementedError):
            MfUsgTvm.load("dummy.tvm", m)


class TestMfUsgTib:
    """Tests for Transient IBOUND (TIB) package."""

    def test_tib_import(self):
        """Test that MfUsgTib can be imported."""
        from flopy.mfusg import MfUsgTib

        assert MfUsgTib is not None

    def test_tib_ftype(self):
        """Test that TIB file type is correct."""
        assert MfUsgTib._ftype() == "TIB"

    def test_tib_defaultunit(self):
        """Test that TIB default unit number is set."""
        assert MfUsgTib._defaultunit() == 72

    def test_tib_init_basic(self, mock_mfusg_model):
        """Test basic TIB initialization."""
        m = mock_mfusg_model

        tib = MfUsgTib(m)

        assert tib.stress_period_data == {}
        assert "TIB" in m.get_package_list()

    def test_tib_init_with_ib0(self, mock_mfusg_model):
        """Test TIB with nodes to deactivate (IBOUND=0)."""
        m = mock_mfusg_model

        spd = {
            0: {"ib0": [5, 6, 7]},
        }

        tib = MfUsgTib(m, stress_period_data=spd)

        assert len(tib.stress_period_data[0]["ib0"]) == 3

    def test_tib_init_with_ib1(self, mock_mfusg_model):
        """Test TIB with nodes to activate (IBOUND=1)."""
        m = mock_mfusg_model

        spd = {
            2: {
                "ib1": [
                    {"node": 1, "head": 50.0},
                    {"node": 2, "avhead": True},
                ],
            },
        }

        tib = MfUsgTib(m, stress_period_data=spd)

        assert len(tib.stress_period_data[2]["ib1"]) == 2

    def test_tib_init_with_ibm1(self, mock_mfusg_model):
        """Test TIB with nodes to set as constant head (IBOUND=-1)."""
        m = mock_mfusg_model

        spd = {
            1: {
                "ibm1": [
                    {"node": 3, "head": 75.0},
                    {"node": 4, "head": 75.0},
                ],
            },
        }

        tib = MfUsgTib(m, stress_period_data=spd)

        assert len(tib.stress_period_data[1]["ibm1"]) == 2

    def test_tib_write_file(self, mock_mfusg_model):
        """Test TIB write_file creates output."""
        m = mock_mfusg_model

        spd = {
            0: {"ib0": [5, 6]},
            2: {"ibm1": [{"node": 1, "head": 50.0}]},
        }

        tib = MfUsgTib(m, stress_period_data=spd)
        tib.write_file()

        # Check file exists
        tib_path = os.path.join(m.model_ws, f"{m.name}.tib")
        assert os.path.exists(tib_path)

        # Check file content
        with open(tib_path) as f:
            content = f.read()

        # Verify deactivated nodes
        assert "5" in content
        assert "6" in content

        # Verify constant head
        assert "HEAD" in content
        assert "5.000000E+01" in content

    def test_tib_write_avhead_option(self, mock_mfusg_model):
        """Test TIB write_file with AVHEAD option."""
        m = mock_mfusg_model

        spd = {
            1: {
                "ib1": [{"node": 3, "avhead": True}],
            },
        }

        tib = MfUsgTib(m, stress_period_data=spd)
        tib.write_file()

        tib_path = os.path.join(m.model_ws, f"{m.name}.tib")
        with open(tib_path) as f:
            content = f.read()

        assert "AVHEAD" in content

    def test_tib_ibm1_requires_head_option(self, mock_mfusg_model):
        """Test that IBM1 nodes require head or avhead option."""
        m = mock_mfusg_model

        # IBM1 without head option should raise error on write
        spd = {
            0: {
                "ibm1": [{"node": 1}],  # Missing head or avhead
            },
        }

        tib = MfUsgTib(m, stress_period_data=spd)

        with pytest.raises(ValueError, match="head"):
            tib.write_file()

    def test_tib_model_type_validation(self, temp_model_ws):
        """Test that TIB requires MfUsg model type."""
        m = flopy.modflow.Modflow(modelname="test", model_ws=str(temp_model_ws))

        with pytest.raises(AssertionError, match="MfUsg"):
            MfUsgTib(m)

    def test_tib_load_not_implemented(self, mock_mfusg_model):
        """Test that TIB load raises NotImplementedError."""
        m = mock_mfusg_model

        with pytest.raises(NotImplementedError):
            MfUsgTib.load("dummy.tib", m)

    def test_tib_multiple_stress_periods(self, mock_mfusg_model):
        """Test TIB with changes across multiple stress periods."""
        m = mock_mfusg_model

        spd = {
            0: {"ib0": [8, 9, 10]},  # Deactivate initially
            2: {"ibm1": [{"node": 1, "head": 90.0}]},  # Add lake
            4: {"ib1": [{"node": 1, "avhead": True}]},  # Remove lake
        }

        tib = MfUsgTib(m, stress_period_data=spd)
        tib.write_file()

        tib_path = os.path.join(m.model_ws, f"{m.name}.tib")
        assert os.path.exists(tib_path)

        with open(tib_path) as f:
            lines = f.readlines()

        # Should have data for all 5 stress periods
        # Count lines with data (excluding header comment)
        data_lines = [line for line in lines if not line.startswith("#")]
        assert len(data_lines) >= 5  # At least one line per SP


class TestTvmTibIntegration:
    """Integration tests for TVM and TIB packages together."""

    def test_tvm_and_tib_together(self, mock_mfusg_model):
        """Test using both TVM and TIB in the same model."""
        m = mock_mfusg_model

        # Add TVM
        tvm_spd = {
            0: {"hk": [(1, 1e-4)]},
            2: {"hk": [(1, 1e-5)]},
        }
        tvm = MfUsgTvm(m, stress_period_data=tvm_spd)

        # Add TIB
        tib_spd = {
            1: {"ibm1": [{"node": 5, "head": 50.0}]},
        }
        tib = MfUsgTib(m, stress_period_data=tib_spd)

        # Both packages should be in model
        pkg_list = m.get_package_list()
        assert "TVM" in pkg_list
        assert "TIB" in pkg_list

        # Write both files
        tvm.write_file()
        tib.write_file()

        # Both files should exist
        assert os.path.exists(os.path.join(m.model_ws, f"{m.name}.tvm"))
        assert os.path.exists(os.path.join(m.model_ws, f"{m.name}.tib"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
