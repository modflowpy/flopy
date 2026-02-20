from pathlib import Path

import pytest

from flopy.utils import CellBudgetFile, HeadFile
from flopy.utils.postprocessing import (
    get_gradients,
    get_specific_discharge,
    get_transmissivities,
    get_water_table,
)

from .conftest import load_mf6_sim, load_mf2005_model


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "rcxy",
    [
        lambda m: (m.dis.nrow.array // 2, m.dis.ncol.array // 2, None, None),
        lambda m: (
            None,
            None,
            sum(m.modelgrid.extent[:2]) / 2,
            sum(m.modelgrid.extent[2:4]) / 2,
        ),
    ],
    ids=["r c", "x y"],
)
def test_get_transmissivities(benchmark, function_tmpdir, rcxy):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()
    hds_path = Path(function_tmpdir) / "freyberg.hds"
    hds = HeadFile(hds_path)
    heads = hds.get_data(totim=hds.get_times()[-1])
    r, c, x, y = rcxy(gwf)
    benchmark(lambda: get_transmissivities(heads, gwf, r=r, c=c, x=x, y=y))


@pytest.mark.benchmark
def test_get_water_table(benchmark, function_tmpdir):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    hds_path = Path(function_tmpdir) / "freyberg.hds"
    hds = HeadFile(hds_path)
    heads = hds.get_data(totim=hds.get_times()[-1])
    benchmark(lambda: get_water_table(heads))


@pytest.mark.benchmark
def test_get_gradients(benchmark, function_tmpdir):
    model = load_mf2005_model(function_tmpdir, model_key="freyberg")
    hds_path = Path(function_tmpdir) / "freyberg.hds"
    hds = HeadFile(hds_path)
    heads = hds.get_data(totim=hds.get_times()[-1])
    benchmark(lambda: get_gradients(heads, model, nodata=-999))


@pytest.mark.benchmark
def test_get_specific_discharge(benchmark, function_tmpdir):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()
    hds_path = Path(function_tmpdir) / "freyberg.hds"
    bud_path = Path(function_tmpdir) / "freyberg.cbc"
    hds = HeadFile(hds_path)
    bud = CellBudgetFile(bud_path)
    heads = hds.get_data(totim=hds.get_times()[-1])
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    benchmark(lambda: get_specific_discharge(spdis, gwf, heads))
