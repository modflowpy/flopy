import inspect

import numpy as np
import pytest

from flopy.modflow import Modflow, ModflowDis, ModflowRch, ModflowWel, ModflowSfr2


def _build_model(ws, name):
    m = Modflow(name, model_ws=ws)

    size = 100
    nlay = 10
    nper = 10
    nsfr = int((size ** 2) / 5)

    dis = ModflowDis(
        m,
        nper=nper,
        nlay=nlay,
        nrow=size,
        ncol=size,
        top=nlay,
        botm=list(range(nlay)),
    )

    rch = ModflowRch(
        m, rech={k: 0.001 - np.cos(k) * 0.001 for k in range(nper)}
    )

    ra = ModflowWel.get_empty(size ** 2)
    well_spd = {}
    for kper in range(nper):
        ra_per = ra.copy()
        ra_per["k"] = 1
        ra_per["i"] = (
            (np.ones((size, size)) * np.arange(size))
                .transpose()
                .ravel()
                .astype(int)
        )
        ra_per["j"] = list(range(size)) * size
        well_spd[kper] = ra
    wel = ModflowWel(m, stress_period_data=well_spd)

    # SFR package
    rd = ModflowSfr2.get_empty_reach_data(nsfr)
    rd["iseg"] = range(len(rd))
    rd["ireach"] = 1
    sd = ModflowSfr2.get_empty_segment_data(nsfr)
    sd["nseg"] = range(len(sd))
    sfr = ModflowSfr2(reach_data=rd, segment_data=sd, model=m)

    return m


@pytest.mark.slow
def test_model_init_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    benchmark(lambda: _build_model(ws=str(tmpdir), name=name))


@pytest.mark.slow
def test_model_write_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = _build_model(ws=str(tmpdir), name=name)
    benchmark(lambda: model.write_input())


@pytest.mark.slow
def test_model_load_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = _build_model(ws=str(tmpdir), name=name)
    model.write_input()
    benchmark(lambda: Modflow.load(f"{name}.nam", model_ws=str(tmpdir), check=False))