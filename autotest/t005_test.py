# Test instantiation of flopy classes
import os

cpth = os.path.join('temp', 't005')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)


def test_modflow_unstructured():
    import flopy
    mf = flopy.modflow.Modflow(version='mfusg', structured=False,
                               model_ws=cpth)
    assert isinstance(mf, flopy.modflow.Modflow)
    dis = flopy.modflow.ModflowDis(mf)
    assert isinstance(dis, flopy.modflow.ModflowDis)
    bas = flopy.modflow.ModflowBas(mf)
    assert isinstance(bas, flopy.modflow.ModflowBas)
    lpf = flopy.modflow.ModflowLpf(mf)
    assert isinstance(lpf, flopy.modflow.ModflowLpf)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data={0: [[0, -100]]})
    assert isinstance(wel, flopy.modflow.ModflowWel)
    ghb = flopy.modflow.ModflowGhb(mf,
                                   stress_period_data={0: [[1, 5.9, 1000.]]})
    assert isinstance(ghb, flopy.modflow.ModflowGhb)
    oc = flopy.modflow.ModflowOc(mf)
    assert isinstance(oc, flopy.modflow.ModflowOc)
    sms = flopy.modflow.ModflowSms(mf)
    assert isinstance(sms, flopy.modflow.ModflowSms)
    # write well file
    wel.write_file()
    assert os.path.isfile(os.path.join(cpth, '{}.wel'.format(mf.name))) is True
    wel2 = flopy.modflow.ModflowWel.load(
        os.path.join(cpth, '{}.wel'.format(mf.name)), mf)
    assert wel2.stress_period_data[0] == wel.stress_period_data[0]
    # ghb file
    ghb.write_file(check=False)
    assert os.path.isfile(os.path.join(cpth, '{}.ghb'.format(mf.name))) is True
    ghb2 = flopy.modflow.ModflowGhb.load(
        os.path.join(cpth, '{}.ghb'.format(mf.name)), mf)
    assert ghb2.stress_period_data[0] == ghb.stress_period_data[0]
    return


if __name__ == '__main__':
    test_modflow_unstructured()
