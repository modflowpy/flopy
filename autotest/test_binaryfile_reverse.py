from dataclasses import dataclass, replace
from itertools import repeat
from pprint import pformat
from typing import Literal, get_args

import numpy as np
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.utils import CellBudgetFile, HeadFile
from flopy.utils.gridutil import get_disv_kwargs

DisType = Literal["dis", "disv", "disu"]


@dataclass
class Case:
    name: str
    tdis: list  # of tuples (perlen, nstp, tsmult)
    # expected results
    forward: list  # of tuples (kstp, kper, time)
    reverse: list  # of tuples (kstp, kper, time)


BUDTXT = "FLOW-JA-FACE"
CASES = [
    Case(
        name="sym",
        tdis=[(1, 1, 1.0), (1, 1, 1.0), (1, 1, 1.0)],
        forward=[(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)],
        reverse=[(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)],
    ),
    Case(
        name="asym",
        tdis=[(1.0, 2, 1.0), (1.0, 1, 1.0), (1.0, 1, 1.0)],
        forward=[(0, 0, 0.5), (1, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)],
        reverse=[(0, 0, 1.0), (0, 1, 2.0), (0, 2, 2.5), (1, 2, 3.0)],
    ),
    Case(
        name="asym_tsm",
        tdis=[(1.0, 1, 1.0), (1.0, 2, 1.5), (1.0, 1, 1.0)],
        forward=[(0, 0, 1.0), (0, 1, 1.4), (1, 1, 2.0), (0, 2, 3.0)],
        reverse=[(0, 0, 1.0), (0, 1, 1.6), (1, 1, 2.0), (0, 2, 3.0)],
    ),
]


def dis_sim(case, ws) -> flopy.mf6.MFSimulation:
    sim = flopy.mf6.MFSimulation(sim_name=case.name, sim_ws=ws, exe_name="mf6")
    nper = len(case.tdis)
    tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=case.tdis)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=case.name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
    dis = gwf.get_package("DIS")
    nlay = 2
    botm = [1 - (k + 1) for k in range(nlay)]
    botm_data = np.array([list(repeat(b, 10 * 10)) for b in botm]).reshape(
        (nlay, 10, 10)
    )
    dis.nlay = nlay
    dis.botm.set_data(botm_data)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 9, 9), 0.0]]
    )
    budget_file = case.name + ".bud"
    head_file = case.name + ".hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def disv_sim(case, ws) -> flopy.mf6.MFSimulation:
    sim = flopy.mf6.MFSimulation(sim_name=case.name, sim_ws=ws, exe_name="mf6")
    nper = len(case.tdis)
    tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=case.tdis)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=case.name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdisv(
        gwf, **get_disv_kwargs(2, 10, 10, 1.0, 1.0, 25.0, [20.0, 15.0])
    )
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 9, 9), 0.0]]
    )
    budget_file = case.name + ".bud"
    head_file = case.name + ".hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def disu_sim(case, ws) -> flopy.mf6.MFSimulation:
    from autotest.test_export import disu_sim as _disu_sim

    sim = _disu_sim(case.name, ws)
    tdis = flopy.mf6.ModflowTdis(sim, nper=len(case.tdis), perioddata=case.tdis)
    return sim


def pytest_generate_tests(metafunc):
    tmp_path_factory = metafunc.config._tmp_path_factory
    cases = []
    sims = []
    names = []
    for case in CASES:
        for dis_type in get_args(DisType):
            name = f"{case.name}_{dis_type}"
            case_ = replace(case, name=name)
            ws = tmp_path_factory.mktemp(name)
            if dis_type == "dis":
                sim = dis_sim(case_, ws)
            elif dis_type == "disv":
                sim = disv_sim(case_, ws)
            elif dis_type == "disu":
                sim = disu_sim(case_, ws)
            cases.append(case_)
            sims.append(sim)
            names.append(name)
    metafunc.parametrize("case, sim", zip(cases, sims), ids=names)


@requires_exe("mf6")
@requires_pkg("shapely")
def test_reverse(case, sim):
    gwf = sim.get_model(case.name)
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True, report=True)
    assert success, pformat(buff)

    # reverse and compare head file headers
    head_file_path = sim.sim_path / gwf.oc.head_filerecord.get_data()[0][0]
    head_file_rev_path = sim.sim_path / f"{head_file_path.name}_rev.hds"
    head_file = HeadFile(head_file_path)
    head_file.reverse(filename=head_file_rev_path)
    head_file_rev = HeadFile(head_file_rev_path)
    fwd_heads = head_file.get_alldata()
    fwd_times = head_file.get_times()
    fwd_kstpkper = head_file.get_kstpkper()
    rev_heads = head_file_rev.get_alldata()
    rev_times = head_file_rev.get_times()
    rev_kstpkper = head_file_rev.get_kstpkper()
    fwd_result = [
        (int(kper), int(kstp), float(time))
        for (kper, kstp), time in zip(fwd_kstpkper, fwd_times)
    ]
    rev_result = [
        (int(kper), int(kstp), float(time))
        for (kper, kstp), time in zip(rev_kstpkper, rev_times)
    ]
    assert len(fwd_kstpkper) == len(rev_kstpkper)
    assert len(fwd_times) == len(rev_times)
    assert fwd_result == case.forward
    assert rev_result == case.reverse

    # reverse and compare budget file headers
    budget_file_path = sim.sim_path / gwf.oc.budget_filerecord.get_data()[0][0]
    budget_file_rev_path = sim.sim_path / f"{budget_file_path.name}_rev.cbb"
    budget_file = CellBudgetFile(budget_file_path)
    budget_file.reverse(budget_file_rev_path)
    budget_file_rev = CellBudgetFile(budget_file_rev_path)
    rev_times = head_file_rev.get_times()
    rev_kstpkper = head_file_rev.get_kstpkper()
    nuniq = len(budget_file.get_unique_record_names())
    ntimes = len(fwd_times)
    assert len(budget_file_rev) == ntimes * nuniq
    assert len(fwd_kstpkper) == len(rev_kstpkper)
    assert len(fwd_times) == len(rev_times)
    assert rev_result == case.reverse

    for i, t in enumerate(fwd_times):
        # compare head
        assert np.allclose(fwd_heads[i], rev_heads[-(i + 1)])

        # compare budget
        rt = rev_times[-(i + 1)]
        bud = budget_file.get_data(text=BUDTXT, totim=t)[0]
        bud_rev = budget_file_rev.get_data(text=BUDTXT, totim=rt)[0]
        assert np.allclose(bud, -bud_rev)
