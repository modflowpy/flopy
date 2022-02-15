"""
created matt_dumont 
on: 15/02/22
"""
import flopy
import numpy as np

from autotest.conftest import requires_exe

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
    'mf2': np.concatenate((l1_ibound, l2_ibound_alt), axis=0),  # constant heads around model on top row
}
laytype = {
    'mf1': [0, 1],
    'mf2': [0, 0]
}
hnoflow = -888
hdry = -777
top = np.zeros((1, nrow, ncol)) + 10
bt1 = np.ones((1, nrow, ncol)) + 5
bt2 = np.ones((1, nrow, ncol)) + 3
botm = np.concatenate((bt1, bt2), axis=0)
ipakcb = 740

names = ['mf1', 'mf2']


def _setup_modflow_model(nm, ws):
    m = flopy.modflow.Modflow(
        modelname=f"modflowtest_{nm}",
        namefile_ext="nam",
        version="mf2005",
        exe_name='mf2005',
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


@requires_exe('mf2005', 'mp6')
def test_data_pass_no_modflow(tmpdir):
    """
    test that user can pass and create a mp model without an accompanying modflow model
    Returns
    -------

    """
    dis_file = f"modflowtest_mf1.dis"
    bud_file = f"modflowtest_mf1.cbc"
    hd_file = f"modflowtest_mf1.hds"

    m1 = _setup_modflow_model('mf1', str(tmpdir))
    mp = flopy.modpath.Modpath6(
        modelname="modpathtest",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name='mp6',
        modflowmodel=None,  # do not pass modflow model

        dis_file=dis_file,
        head_file=hd_file,
        budget_file=bud_file,
        model_ws=ws,
        external_path=None,
        verbose=False,
        load=True,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (nrow, ncol, nlay, nper)

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
    # test layertype is created correctly
    assert np.isclose(mpbas.laytyp.array, laytype['mf1']).all()
    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ibound['mf1']).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success


@requires_exe('mf2005', 'mp6')
def test_data_pass_with_modflow(tmpdir):
    """
    test that user specified head files etc. are preferred over files from the modflow model
    Returns
    -------

    """
    dis_file = f"modflowtest_mf1.dis"
    bud_file = f"modflowtest_mf1.cbc"
    hd_file = f"modflowtest_mf1.hds"

    m1 = _setup_modflow_model('mf1', str(tmpdir))
    m2 = _setup_modflow_model('mf2', str(tmpdir))
    mp = flopy.modpath.Modpath6(
        modelname="modpathtest",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name='mp6',
        modflowmodel=m2,  # do not pass modflow model

        dis_file=dis_file,
        head_file=hd_file,
        budget_file=bud_file,
        model_ws=str(tmpdir),
        external_path=None,
        verbose=False,
        load=False,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (nrow, ncol, nlay, nper)

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

    # test layertype is created correctly!
    assert np.isclose(mpbas.laytyp.array, laytype['mf1']).all()
    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ibound['mf1']).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success


@requires_exe('mf2005', 'mp6')
def test_just_from_model(tmpdir):
    """
    test that user specified head files etc. are preferred over files from the modflow model
    Returns
    -------

    """
    dis_file = f"modflowtest_mf2.dis"
    bud_file = f"modflowtest_mf2.cbc"
    hd_file = f"modflowtest_mf2.hds"

    m1 = _setup_modflow_model('mf1', str(tmpdir))
    m2 = _setup_modflow_model('mf2', str(tmpdir))
    mp = flopy.modpath.Modpath6(
        modelname="modpathtest",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name='mp6',
        modflowmodel=m2,  # do not pass modflow model

        dis_file=None,
        head_file=None,
        budget_file=None,
        model_ws=str(tmpdir),
        external_path=None,
        verbose=False,
        load=False,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (nrow, ncol, nlay, nper)

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=hnoflow,
        hdry=hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=None,
        ibound=None,
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,

    )
    # test layertype is created correctly!
    assert np.isclose(mpbas.laytyp.array, laytype['mf2']).all()

    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ibound['mf2']).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success
