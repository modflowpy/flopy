import os
import numpy as np
import flopy
from ci_framework import baseTestDir, flopyTest

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def test_mflist_external():
    model_ws = f"{baseDir}_test_mflist_external"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    ml = flopy.modflow.Modflow(
        "mflist_test",
        model_ws=model_ws,
        external_path=os.path.join(model_ws, "ref"),
    )
    dis = flopy.modflow.ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    wel_data = {
        0: [[0, 0, 0, -1], [1, 1, 1, -1]],
        1: [[0, 0, 0, -2], [1, 1, 1, -1]],
    }
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=wel_data)
    ml.write_input()

    ml1 = flopy.modflow.Modflow.load(
        "mflist_test.nam",
        model_ws=ml.model_ws,
        verbose=True,
        forgive=False,
        check=False,
    )

    assert np.array_equal(ml.wel[0], ml1.wel[0])
    assert np.array_equal(ml.wel[1], ml1.wel[1])

    ml1.write_input()


def test_single_mflist_entry_load():
    model_ws = f"{baseDir}_test_single_mflist_entry_load"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    pth = os.path.join("..", "examples", "data", "freyberg")
    m = flopy.modflow.Modflow.load(
        "freyberg.nam", model_ws=pth, load_only=["WEL"], check=False
    )
    w = m.wel
    spd = w.stress_period_data
    flopy.modflow.ModflowWel(m, stress_period_data={0: [0, 0, 0, 0.0]})
    m.external_path = "."
    m.change_model_ws(model_ws, reset_external=True)
    m.write_input()

    mm = flopy.modflow.Modflow.load(
        "freyberg.nam",
        model_ws=model_ws,
        forgive=False,
    )
    assert mm.wel.stress_period_data
    mm.write_input()


def test_mflist_add_record():
    ml = flopy.modflow.Modflow()
    _ = flopy.modflow.ModflowDis(ml, nper=2)
    wel = flopy.modflow.ModflowWel(ml)
    assert len(wel.stress_period_data.data) == 0

    wel.stress_period_data.add_record(0, [0, 1, 2], [1.0])
    assert len(wel.stress_period_data.data) == 1
    wel_dtype = [("k", int), ("i", int), ("j", int), ("flux", np.float32)]
    check0 = np.array([(0, 1, 2, 1.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)

    wel.stress_period_data.add_record(0, [0, 1, 1], [8.0])
    assert len(wel.stress_period_data.data) == 1
    check0 = np.array([(0, 1, 2, 1.0), (0, 1, 1, 8.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)

    wel.stress_period_data.add_record(1, [0, 1, 1], [5.0])
    assert len(wel.stress_period_data.data) == 2
    check1 = np.array([(0, 1, 1, 5.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)
    np.testing.assert_array_equal(wel.stress_period_data[1], check1)


if __name__ == "__main__":
    test_mflist_external()
    test_single_mflist_entry_load()
    test_mflist_add_record()
