# Test instantiation of mf6 classes
import os
import flopy

from ci_framework import baseTestDir, flopyTest


def test_mf6():
    baseDir = baseTestDir(__file__, relPath="temp", verbose=True)
    testFramework = flopyTest(verbose=True, testDirs=baseDir)

    sim = flopy.mf6.MFSimulation(sim_ws=baseDir)
    assert isinstance(sim, flopy.mf6.MFSimulation)

    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim)
    assert isinstance(tdis, flopy.mf6.modflow.mftdis.ModflowTdis)

    gwfgwf = flopy.mf6.modflow.mfgwfgwf.ModflowGwfgwf(
        sim, exgtype="gwf6-gwf6", exgmnamea="gwf1", exgmnameb="gwf2"
    )
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
    assert os.path.isfile(os.path.join(baseDir, "mfsim.nam"))
    exts_model = [
        "nam",
        "dis",
        "disu",
        "disv",
        "npf",
        "ic",
        "sto",
        "hfb",
        "gnc",
        "chd",
        "wel",
        "drn",
        "riv",
        "ghb",
        "rch",
        "rcha",
        "evt",
        "evta",
        "maw",
        "sfr",
        "lak",
        "mvr",
    ]
    exts_sim = ["gwfgwf", "ims", "tdis"]
    for ext in exts_model:
        fname = os.path.join(baseDir, f"model.{ext}")
        assert os.path.isfile(fname), f"{fname} not found"
    for ext in exts_sim:
        fname = os.path.join(baseDir, f"sim.{ext}")
        assert os.path.isfile(fname), f"{fname} not found"

    return


def test_mf6_string_to_file_path():
    from flopy.mf6.mfbase import MFFileMgmt
    import platform

    if platform.system().lower() == "windows":
        unc_path = r"\\server\path\path"
        new_path = MFFileMgmt.string_to_file_path(unc_path)
        if not unc_path == new_path:
            raise AssertionError("UNC path error")

        abs_path = r"C:\Users\some_user\path"
        new_path = MFFileMgmt.string_to_file_path(abs_path)
        if not abs_path == new_path:
            raise AssertionError("Absolute path error")

        rel_path = r"..\path\some_path"
        new_path = MFFileMgmt.string_to_file_path(rel_path)
        if not rel_path == new_path:
            raise AssertionError("Relative path error")

    else:
        abs_path = "/mnt/c/some_user/path"
        new_path = MFFileMgmt.string_to_file_path(abs_path)
        if not abs_path == new_path:
            raise AssertionError("Absolute path error")

        rel_path = "../path/some_path"
        new_path = MFFileMgmt.string_to_file_path(rel_path)
        if not rel_path == new_path:
            raise AssertionError("Relative path error")


if __name__ == "__main__":
    test_mf6()
    test_mf6_string_to_file_path()
