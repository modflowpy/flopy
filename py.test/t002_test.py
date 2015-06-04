# Test instantiation of flopy classes

def test_modflow():
    import flopy
    mf = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(mf)
    bas = flopy.modflow.ModflowBas(mf)
    lpf = flopy.modflow.ModflowLpf(mf)
    wel = flopy.modflow.ModflowWel(mf)
    oc = flopy.modflow.ModflowOc(mf)
    pcg = flopy.modflow.ModflowPcg(mf)
    return
