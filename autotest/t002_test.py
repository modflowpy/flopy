# Test instantiation of flopy classes

def test_modflow():
    import flopy
    mf = flopy.modflow.Modflow()
    assert isinstance(mf, flopy.modflow.Modflow)
    assert not mf.has_package('DIS')  # not yet
    dis = flopy.modflow.ModflowDis(mf)
    assert mf.has_package('DIS')
    assert mf.has_package('dis')  # case-insensitive
    assert not mf.has_package('DISU')  # not here
    assert isinstance(dis, flopy.modflow.ModflowDis)
    bas = flopy.modflow.ModflowBas(mf)
    assert isinstance(bas, flopy.modflow.ModflowBas)
    lpf = flopy.modflow.ModflowLpf(mf)
    assert isinstance(lpf, flopy.modflow.ModflowLpf)
    wel = flopy.modflow.ModflowWel(mf)
    assert isinstance(wel, flopy.modflow.ModflowWel)
    oc = flopy.modflow.ModflowOc(mf)
    assert isinstance(oc, flopy.modflow.ModflowOc)
    pcg = flopy.modflow.ModflowPcg(mf)
    assert isinstance(pcg, flopy.modflow.ModflowPcg)
    return

if __name__ == '__main__':
    test_modflow()
