import os
import flopy
import numpy as np


tpth = os.path.abspath(os.path.join("temp", "t016"))
if not os.path.isdir(tpth):
    os.makedirs(tpth)


exe_name = "mfusg"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_usg_disu_load():

    pthusgtest = os.path.join(
        "..", "examples", "data", "mfusg_test", "01A_nestedgrid_nognc"
    )
    fname = os.path.join(pthusgtest, "flow.disu")
    assert os.path.isfile(fname), "disu file not found {}".format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname="usgload", verbose=True)

    # Load the disu file
    disu = flopy.modflow.ModflowDisU.load(fname, m)
    assert isinstance(disu, flopy.modflow.ModflowDisU)

    # Change where model files are written
    model_ws = tpth
    m.model_ws = model_ws

    # Write the disu file
    disu.write_file()
    assert os.path.isfile(
        os.path.join(model_ws, "{}.{}".format(m.name, m.disu.extension[0]))
    )

    # Load disu file
    disu2 = flopy.modflow.ModflowDisU.load(fname, m)
    for (key1, value1), (key2, value2) in zip(
        disu2.__dict__.items(), disu.__dict__.items()
    ):
        if isinstance(value1, (flopy.utils.Util2d, flopy.utils.Util3d)):
            assert np.array_equal(value1.array, value2.array)
        elif isinstance(
            value1, list
        ):  # this is for the jagged _get_neighbours list
            assert np.all([np.all(v1 == v2) for v1, v2 in zip(value1, value2)])
        elif not isinstance(value1, flopy.utils.reference.TemporalReference):
            assert value1 == value2

    return


def test_usg_sms_load():

    pthusgtest = os.path.join(
        "..", "examples", "data", "mfusg_test", "01A_nestedgrid_nognc"
    )
    fname = os.path.join(pthusgtest, "flow.sms")
    assert os.path.isfile(fname), "sms file not found {}".format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname="usgload", verbose=True)

    # Load the sms file
    sms = flopy.modflow.ModflowSms.load(fname, m)
    assert isinstance(sms, flopy.modflow.ModflowSms)

    # Change where model files are written
    model_ws = tpth
    m.model_ws = model_ws

    # Write the sms file
    sms.write_file()
    assert os.path.isfile(
        os.path.join(model_ws, "{}.{}".format(m.name, m.sms.extension[0]))
    )

    # Load sms file
    sms2 = flopy.modflow.ModflowSms.load(fname, m)
    for (key1, value1), (key2, value2) in zip(
        sms2.__dict__.items(), sms.__dict__.items()
    ):
        assert (
            value1 == value2
        ), "key1 {}, value 1 {} != key2 {} value 2 {}".format(
            key1, value1, key2, value2
        )

    return


def test_usg_model():
    mf = flopy.modflow.Modflow(
        version="mfusg",
        structured=True,
        model_ws=tpth,
        modelname="simple",
        exe_name=v,
    )
    dis = flopy.modflow.ModflowDis(mf, nlay=1, nrow=11, ncol=11)
    bas = flopy.modflow.ModflowBas(mf)
    lpf = flopy.modflow.ModflowLpf(mf)
    wel = flopy.modflow.ModflowWel(
        mf, stress_period_data={0: [[0, 5, 5, -1.0]]}
    )
    ghb = flopy.modflow.ModflowGhb(
        mf,
        stress_period_data={
            0: [
                [0, 0, 0, 1.0, 1000.0],
                [0, 9, 9, 0.0, 1000.0],
            ]
        },
    )
    oc = flopy.modflow.ModflowOc(mf)
    sms = flopy.modflow.ModflowSms(mf, options="complex")

    # run with defaults
    mf.write_input()
    if run:
        success, buff = mf.run_model()
        assert success

    # try different complexity options; all should run successfully
    for complexity in ["simple", "moderate", "complex"]:
        print("testing MFUSG with sms complexity: " + complexity)
        sms = flopy.modflow.ModflowSms(mf, options=complexity)
        sms.write_file()
        if run:
            success, buff = mf.run_model()
            assert success


def test_usg_load_01B():
    print(
        "testing 1-layer unstructured mfusg model loading: 01A_nestedgrid_nognc.nam"
    )
    pthusgtest = os.path.join(
        "..", "examples", "data", "mfusg_test", "01A_nestedgrid_nognc"
    )
    fname = os.path.join(pthusgtest, "flow.nam")
    assert os.path.isfile(fname), "nam file not found {}".format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname="usgload_1b", verbose=True)

    # Load the model, with checking
    m = m.load(fname, check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, flopy.modflow.mfdisu.ModflowDisU), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, flopy.modflow.mflpf.ModflowLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, flopy.modflow.mfbas.ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, flopy.modflow.mfoc.ModflowOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, flopy.modflow.mfsms.ModflowSms), msg


def test_usg_load_45usg():
    print("testing 3-layer unstructured mfusg model loading: 45usg.nam")
    pthusgtest = os.path.join("..", "examples", "data", "mfusg_test", "45usg")
    fname = os.path.join(pthusgtest, "45usg.nam")
    assert os.path.isfile(fname), "nam file not found {}".format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname="45usg", verbose=True)

    # Load the model, with checking.
    m = m.load(fname, check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, flopy.modflow.mfdisu.ModflowDisU), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, flopy.modflow.mflpf.ModflowLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, flopy.modflow.mfbas.ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, flopy.modflow.mfoc.ModflowOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, flopy.modflow.mfsms.ModflowSms), msg
    msg = "flopy failed on loading mfusg drn package"
    assert isinstance(m.drn, flopy.modflow.mfdrn.ModflowDrn), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, flopy.modflow.mfwel.ModflowWel), msg


def test_usg_rch_evt_models01():
    # this test has RCH nrchop == 1, and EVT nevtop == 1
    print(
        "testing unstructured mfusg RCH nrchop == 1, and EVT nevtop == 1: \
usg_rch_evt.nam"
    )
    model_ws = os.path.join(
        "..", "examples", "data", "mfusg_test", "rch_evt_tests"
    )
    nam = "usg_rch_evt.nam"
    m = flopy.modflow.Modflow.load(
        nam, model_ws=model_ws, version="mfusg", exe_name=v
    )
    m.model_ws = tpth
    m.write_input()
    if run:
        success, buff = m.run_model()
        assert success


def test_usg_rch_evt_models02():
    # this test has RCH nrchop == 2, and EVT nevtop == 2
    print(
        "testing unstructured mfusg RCH nrchop == 2, and EVT nevtop == 2: \
usg_rch_evt_nrchop2.nam"
    )
    model_ws = os.path.join(
        "..", "examples", "data", "mfusg_test", "rch_evt_tests"
    )
    nam = "usg_rch_evt_nrchop2.nam"
    m = flopy.modflow.Modflow.load(
        nam, model_ws=model_ws, version="mfusg", exe_name=v
    )
    m.model_ws = tpth
    m.write_input()
    if run:
        success, buff = m.run_model()
        assert success


def test_usg_rch_evt_models02a():
    # this test has RCH nrchop == 2, and EVT nevtop == 2
    print(
        "testing unstructured mfusg RCH nrchop == 2, and EVT nevtop == 2,\
 but with fewer irch nodes: than in nodelay[0] usg_rch_evt_nrchop2.nam"
    )
    model_ws = os.path.join(
        "..", "examples", "data", "mfusg_test", "rch_evt_tests"
    )
    nam = "usg_rch_evt_nrchop2a.nam"
    m = flopy.modflow.Modflow.load(
        nam, model_ws=model_ws, version="mfusg", exe_name=v
    )
    m.model_ws = tpth
    m.write_input()
    if run:
        success, buff = m.run_model()
        assert success


if __name__ == "__main__":
    test_usg_disu_load()
    test_usg_sms_load()
    test_usg_model()
    test_usg_load_01B()
    test_usg_load_45usg()
    test_usg_rch_evt_models01()
    test_usg_rch_evt_models02()
    test_usg_rch_evt_models02a()
