# Test instantiation of mf6 classes
import os
import shutil
import flopy


def test_mf6():

    out_dir = os.path.join('temp', 't501')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    sim = flopy.mf6.MFSimulation(sim_ws=out_dir)
    assert isinstance(sim, flopy.mf6.MFSimulation)

    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim)
    assert isinstance(tdis, flopy.mf6.modflow.mftdis.ModflowTdis)

    gwfgwf = flopy.mf6.modflow.mfgwfgwf.ModflowGwfgwf(sim,
                                                      exgtype='gwf6-gwf6',
                                                      exgmnamea='gwf1',
                                                      exgmnameb='gwf2')
    assert isinstance(gwfgwf, flopy.mf6.modflow.mfgwfgwf.ModflowGwfgwf)

    gwf = flopy.mf6.ModflowGwf(sim)
    assert isinstance(gwf, flopy.mf6.ModflowGwf)

    ims = flopy.mf6.modflow.mfims.ModflowIms(sim)
    assert isinstance(ims, flopy.mf6.modflow.mfims.ModflowIms)
    sim.register_ims_package(ims, [])

    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(gwf)
    assert isinstance(dis, flopy.mf6.modflow.mfgwfdis.ModflowGwfdis)

    disu = flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(gwf)
    assert isinstance(disu, flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu)

    disv = flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(gwf)
    assert isinstance(disv, flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv)

    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf)
    assert isinstance(npf, flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf)

    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf)
    assert isinstance(ic, flopy.mf6.modflow.mfgwfic.ModflowGwfic)

    sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf)
    assert isinstance(sto, flopy.mf6.modflow.mfgwfsto.ModflowGwfsto)

    hfb = flopy.mf6.modflow.mfgwfhfb.ModflowGwfhfb(gwf)
    assert isinstance(hfb, flopy.mf6.modflow.mfgwfhfb.ModflowGwfhfb)

    gnc = flopy.mf6.modflow.mfgwfgnc.ModflowGwfgnc(gwf)
    assert isinstance(gnc, flopy.mf6.modflow.mfgwfgnc.ModflowGwfgnc)

    chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf)
    assert isinstance(chd, flopy.mf6.modflow.mfgwfchd.ModflowGwfchd)

    wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf)
    assert isinstance(wel, flopy.mf6.modflow.mfgwfwel.ModflowGwfwel)

    drn = flopy.mf6.modflow.mfgwfdrn.ModflowGwfdrn(gwf)
    assert isinstance(drn, flopy.mf6.modflow.mfgwfdrn.ModflowGwfdrn)

    riv = flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(gwf)
    assert isinstance(riv, flopy.mf6.modflow.mfgwfriv.ModflowGwfriv)

    ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(gwf)
    assert isinstance(ghb, flopy.mf6.modflow.mfgwfghb.ModflowGwfghb)

    rch = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(gwf)
    assert isinstance(rch, flopy.mf6.modflow.mfgwfrch.ModflowGwfrch)

    rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(gwf)
    assert isinstance(rcha, flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha)

    evt = flopy.mf6.modflow.mfgwfevt.ModflowGwfevt(gwf)
    assert isinstance(evt, flopy.mf6.modflow.mfgwfevt.ModflowGwfevt)

    evta = flopy.mf6.modflow.mfgwfevta.ModflowGwfevta(gwf)
    assert isinstance(evta, flopy.mf6.modflow.mfgwfevta.ModflowGwfevta)

    maw = flopy.mf6.modflow.mfgwfmaw.ModflowGwfmaw(gwf)
    assert isinstance(maw, flopy.mf6.modflow.mfgwfmaw.ModflowGwfmaw)

    sfr = flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr(gwf)
    assert isinstance(sfr, flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr)

    lak = flopy.mf6.modflow.mfgwflak.ModflowGwflak(gwf)
    assert isinstance(lak, flopy.mf6.modflow.mfgwflak.ModflowGwflak)

    uzf = flopy.mf6.modflow.mfgwfuzf.ModflowGwfuzf(gwf)
    assert isinstance(uzf, flopy.mf6.modflow.mfgwfuzf.ModflowGwfuzf)

    mvr = flopy.mf6.modflow.mfgwfmvr.ModflowGwfmvr(gwf)
    assert isinstance(mvr, flopy.mf6.modflow.mfgwfmvr.ModflowGwfmvr)

    # Write files
    sim.write_simulation()

    # Verify files were written
    assert os.path.isfile(os.path.join(out_dir, 'mfsim.nam'))
    exts_model = ['nam', 'dis', 'disu', 'disv', 'npf', 'ic',
            'sto', 'hfb', 'gnc', 'chd', 'wel', 'drn', 'riv', 'ghb', 'rch',
            'rcha', 'evt', 'evta', 'maw', 'sfr', 'lak', 'mvr']
    exts_sim = ['gwfgwf', 'ims', 'tdis']
    for ext in exts_model:
        fname = os.path.join(out_dir, 'model.{}'.format(ext))
        assert os.path.isfile(fname), fname + ' not found'
    for ext in exts_sim:
        fname = os.path.join(out_dir, 'sim.{}'.format(ext))
        assert os.path.isfile(fname), fname + ' not found'


    return

if __name__ == '__main__':
    test_mf6()
