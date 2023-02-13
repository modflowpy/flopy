import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from flopy.utils import FormattedHeadFile


@pytest.fixture
def freyberg_model_path(example_data_path):
    return example_data_path / "freyberg"


def test_formattedfile_reference(example_data_path):
    h = FormattedHeadFile(
        str(example_data_path / "mf2005_test" / "test1tr.githds")
    )
    assert isinstance(h, FormattedHeadFile)
    h.mg.set_coord_info(xoff=1000.0, yoff=200.0, angrot=15.0)

    assert isinstance(h.plot(masked_values=[6999.000]), Axes)
    plt.close()


def test_formattedfile_read(function_tmpdir, example_data_path):
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
    fname = str(function_tmpdir / "empty.githds")
    with open(fname, "w"):
        pass
    with pytest.raises(ValueError):
        FormattedHeadFile(fname)
