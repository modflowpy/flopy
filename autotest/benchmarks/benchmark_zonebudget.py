from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from flopy.modflow.mf import Modflow
from flopy.utils.zonbud import ZoneBudget


def create_zone_array(nlay, nrow, ncol, n_zones=2) -> NDArray:
    zones = np.zeros((nlay, nrow, ncol), dtype=np.int32)
    zone_width = ncol // n_zones  # roughly equal zones

    for i in range(n_zones):
        start_col = i * zone_width
        end_col = (i + 1) * zone_width if i < n_zones - 1 else ncol
        zones[:, :, start_col:end_col] = i + 1

    return zones


@pytest.fixture(scope="module", params=[2, 5])
def case(request, example_data_path) -> tuple[Path, NDArray]:
    model_path = example_data_path / "zonbud_examples"
    model = Modflow.load(
        example_data_path / "freyberg" / "freyberg.nam", version="mf2005"
    )
    cbc_path = model_path / "freyberg.gitcbc"
    zones = create_zone_array(model.nlay, model.nrow, model.ncol, n_zones=request.param)
    return cbc_path, zones


@pytest.mark.slow
@pytest.mark.benchmark
def test_zonebudget_load(benchmark, case):
    cbc_path, zones = case
    benchmark(lambda: ZoneBudget(str(cbc_path), z=zones))


@pytest.mark.slow
@pytest.mark.benchmark
def test_zonebudget_get_budget(benchmark, case):
    cbc_path, zones = case
    zb = ZoneBudget(str(cbc_path), z=zones)
    benchmark(zb.get_budget)


@pytest.mark.slow
@pytest.mark.benchmark
def test_zonebudget_get_dataframes(benchmark, case):
    cbc_path, zones = case
    zb = ZoneBudget(str(cbc_path), z=zones)
    benchmark(zb.get_dataframes)
