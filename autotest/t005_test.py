# Test instantiation of flopy classes
import os

cpth = os.path.join("temp", "t005")
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)


def test_modflow_unstructured():
    import flopy
    import numpy as np

    mf = flopy.mfusg.MfUsg(structured=False, model_ws=cpth)
    assert isinstance(mf, flopy.mfusg.MfUsg)

    disu = flopy.mfusg.MfUsgDisU(
        mf, nodes=1, iac=[1], njag=1, ja=np.array([0]), cl12=[1.0], fahl=[1.0]
    )
    assert isinstance(disu, flopy.mfusg.MfUsgDisU)

    bas = flopy.modflow.ModflowBas(mf)
    assert isinstance(bas, flopy.modflow.ModflowBas)

    lpf = flopy.mfusg.MfUsgLpf(mf)
    assert isinstance(lpf, flopy.mfusg.MfUsgLpf)

    wel = flopy.mfusg.MfUsgWel(mf, stress_period_data={0: [[0, -100]]})
    assert isinstance(wel, flopy.mfusg.MfUsgWel)

    ghb = flopy.modflow.ModflowGhb(
        mf, stress_period_data={0: [[1, 5.9, 1000.0]]}
    )
    assert isinstance(ghb, flopy.modflow.ModflowGhb)

    oc = flopy.modflow.ModflowOc(mf)
    assert isinstance(oc, flopy.modflow.ModflowOc)

    sms = flopy.mfusg.MfUsgSms(mf)
    assert isinstance(sms, flopy.mfusg.MfUsgSms)

    # write well file
    wel.write_file()
    assert os.path.isfile(os.path.join(cpth, f"{mf.name}.wel")) is True
    wel2 = flopy.mfusg.MfUsgWel.load(os.path.join(cpth, f"{mf.name}.wel"), mf)
    assert wel2.stress_period_data[0] == wel.stress_period_data[0]

    # write ghb file
    ghb.write_file(check=False)
    assert os.path.isfile(os.path.join(cpth, f"{mf.name}.ghb")) is True
    ghb2 = flopy.modflow.ModflowGhb.load(
        os.path.join(cpth, f"{mf.name}.ghb"), mf
    )
    assert ghb2.stress_period_data[0] == ghb.stress_period_data[0]

    return


if __name__ == "__main__":
    test_modflow_unstructured()
