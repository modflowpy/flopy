# Test instantiation of flopy classes

def test_modflow():
    import flopy
    mf = flopy.modflow.Modflow(version='mfusg', structured=False,
                               model_ws='data/')
    dis = flopy.modflow.ModflowDis(mf)
    bas = flopy.modflow.ModflowBas(mf)
    lpf = flopy.modflow.ModflowLpf(mf)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data={0:[[0, -100]]})
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data={0:[[1,5.9,1000.]]})
    oc = flopy.modflow.ModflowOc(mf)
    sms = flopy.modflow.ModflowPcg(mf)
    wel.write_file()
    ghb.write_file()
    return
