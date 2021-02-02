"""
Test MT3D model creation and file writing
"""

import os
import warnings
import flopy


def test_mt3d_create_withmfmodel():
    model_ws = os.path.join(".", "temp", "t013")

    # Create a MODFLOW model
    mf = flopy.modflow.Modflow(model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(mf)
    lpf = flopy.modflow.ModflowLpf(mf)

    # Create MT3D model
    mt = flopy.mt3d.Mt3dms(modflowmodel=mf, model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(mt)
    adv = flopy.mt3d.Mt3dAdv(mt)
    dsp = flopy.mt3d.Mt3dDsp(mt)
    ssm = flopy.mt3d.Mt3dSsm(mt)
    gcg = flopy.mt3d.Mt3dRct(mt)
    rct = flopy.mt3d.Mt3dGcg(mt)
    tob = flopy.mt3d.Mt3dTob(mt)

    # Write the output
    mt.write_input()

    # confirm that MT3D files exist
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, btn.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, adv.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, dsp.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, ssm.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, gcg.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, rct.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, tob.extension[0]))
        )
        is True
    )

    return


def test_mt3d_create_woutmfmodel():
    model_ws = os.path.join(".", "temp", "t013")

    # Create MT3D model
    mt = flopy.mt3d.Mt3dms(model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(
        mt,
        nlay=1,
        nrow=2,
        ncol=2,
        nper=1,
        delr=1.0,
        delc=1.0,
        htop=1.0,
        dz=1.0,
        laycon=0,
        perlen=1.0,
        nstp=1,
        tsmult=1.0,
    )
    adv = flopy.mt3d.Mt3dAdv(mt)
    dsp = flopy.mt3d.Mt3dDsp(mt)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        ssm = flopy.mt3d.Mt3dSsm(mt)

        assert len(w) == 1, "Number of warnings = {} instead of 1".format(
            len(w)
        )
        assert w[0].category == UserWarning, "Warning category: {}".format(
            w[0]
        )
        assert "mxss is None and modflowmodel is None." in str(w[0].message)

    gcg = flopy.mt3d.Mt3dRct(mt)
    rct = flopy.mt3d.Mt3dGcg(mt)
    tob = flopy.mt3d.Mt3dTob(mt)

    # Write the output
    mt.write_input()

    # confirm that MT3D files exist
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, btn.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, adv.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, dsp.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, ssm.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, gcg.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, rct.extension[0]))
        )
        is True
    )
    assert (
        os.path.isfile(
            os.path.join(model_ws, "{}.{}".format(mt.name, tob.extension[0]))
        )
        is True
    )

    return


def test_mt3d_pht3d():
    # Note: this test is incomplete!
    model_ws = os.path.join(".", "temp", "t013")

    # Create MT3D model
    mt = flopy.mt3d.Mt3dms(model_ws=model_ws)
    phc = flopy.mt3d.Mt3dPhc(mt, minkin=[[[1]], [[2]]])

    # Write the output
    mt.write_input()

    # confirm that MT3D files exist
    assert os.path.isfile(
        os.path.join(model_ws, "{}.{}".format(mt.name, phc.extension[0]))
    )

    return


if __name__ == "__main__":
    test_mt3d_create_withmfmodel()
    test_mt3d_create_woutmfmodel()
