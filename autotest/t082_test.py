"""
created matt_dumont 
on: 16/02/22
"""
import os.path

import flopy
import numpy as np
from ci_framework import FlopyTestSetup, base_test_dir
import platform

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)
nrow = 3
ncol = 4
nlay = 2
nper = 1
l1_ibound = np.array([[[-1, -1, -1, -1],
                       [-1, 1, 1, -1],
                       [-1, -1, -1, -1]]])
l2_ibound = np.ones((1, nrow, ncol))
l2_ibound_alt = np.ones((1, nrow, ncol))
l2_ibound_alt[0, 0, 0] = 0
ibound = {
    'mf1': np.concatenate((l1_ibound, l2_ibound), axis=0),  # constant heads around model on top row
}
laytype = {
    'mf1': [0, 1],
}
hnoflow = -888
hdry = -777
top = np.zeros((1, nrow, ncol)) + 10
bt1 = np.ones((1, nrow, ncol)) + 5
bt2 = np.ones((1, nrow, ncol)) + 3
botm = np.concatenate((bt1, bt2), axis=0)
ipakcb = 740

names = ['mf1']

mf2005_exe = "mf2005"
if platform.system() in "Windows":
    mf2005_exe += ".exe"
mf2005_exe = flopy.which(mf2005_exe)

mp6_exe = "mp6"
if platform.system() in "Windows":
    mp6_exe += ".exe"
mp6_exe = flopy.which(mp6_exe)


def make_test_modflow_model(nm, ws):
    m = flopy.modflow.Modflow(
        modelname=f"modflowtest_{nm}",
        namefile_ext="nam",
        version="mf2005",
        exe_name=mf2005_exe,
        model_ws=ws,
    )

    # dis
    dis = flopy.modflow.ModflowDis(
        model=m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delr=1.0,
        delc=1.0,
        laycbd=0,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        tsmult=1,
        steady=True,
    )

    # bas
    bas = flopy.modflow.ModflowBas(
        model=m,
        ibound=ibound[nm],
        strt=10,
        ifrefm=True,
        ixsec=False,
        ichflg=False,
        stoper=None,
        hnoflo=hnoflow,
        extension="bas",
        unitnumber=None,
        filenames=None,

    )
    # lpf
    lpf = flopy.modflow.ModflowLpf(
        model=m,
        ipakcb=ipakcb,
        laytyp=laytype[nm],
        hk=10,
        vka=10,
        hdry=hdry
    )

    # well
    wel = flopy.modflow.ModflowWel(
        model=m,
        ipakcb=ipakcb,
        stress_period_data={0: [[1, 1, 1, -5.]]},

    )

    flopy.modflow.ModflowPcg(m, hclose=0.001, rclose=0.001,
                             mxiter=150, iter1=30,
                             )

    ocspd = {}
    for p in range(nper):
        ocspd[(p, 0)] = ['save head', 'save budget']
    ocspd[(0, 0)] = ['save head', 'save budget']  # pretty sure it just uses the last for everything
    flopy.modflow.ModflowOc(m, stress_period_data=ocspd)

    m.write_input()
    success, buff = m.run_model()
    assert success

    return m


def make_mp_model(nm, m, ws, use_pandas):
    mp = flopy.modpath.Modpath6(
        modelname=nm,
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name=mp6_exe,
        modflowmodel=m,
        dis_file=None,
        head_file=None,
        budget_file=None,
        model_ws=ws,
        external_path=None,
        verbose=False,
        load=True,
        listunit=7,
    )

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=hnoflow,
        hdry=hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=laytype['mf1'],
        ibound=ibound['mf1'],
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,

    )

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp, use_pandas=use_pandas)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    return mp


def test_mp_wpandas_wo_pandas():
    """
    test that user can pass and create a mp model without an accompanying modflow model
    Returns
    -------

    """
    ws = f"{base_dir}_test_mp_wpandas_wo_pandas"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=ws)

    m1 = make_test_modflow_model('mf1', ws)
    mp_pandas = make_mp_model('pandas', m1, ws, use_pandas=True)
    mp_no_pandas = make_mp_model('no_pandas', m1, ws, use_pandas=False)

    mp_no_pandas.write_input()
    success, buff = mp_no_pandas.run_model()
    assert success

    mp_pandas.write_input()
    success, buff = mp_pandas.run_model()
    assert success

    # read the two files and ensure they are identical
    with open(mp_pandas.get_package('loc').fn_path, 'r') as f:
        particles_pandas = f.readlines()
    with open(mp_no_pandas.get_package('loc').fn_path, 'r') as f:
        particles_no_pandas = f.readlines()
    assert particles_pandas == particles_no_pandas
