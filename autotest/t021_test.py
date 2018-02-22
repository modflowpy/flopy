# Test modflow write adn run
import os
import numpy as np
import flopy

mpth = os.path.join('temp', 't021')
# make the directory if it does not exist
if not os.path.isdir(mpth):
    os.makedirs(mpth)


def test_mflist_external():
    ml = flopy.modflow.Modflow("mflist_test", model_ws=mpth,
                               external_path=os.path.join(mpth, "ref"))
    dis = flopy.modflow.ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    wel_data = {0: [[0, 0, 0, -1], [1, 1, 1, -1]],
                1: [[0, 0, 0, -2], [1, 1, 1, -1]]}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=wel_data)
    ml.write_input()

    ml1 = flopy.modflow.Modflow.load("mflist_test.nam",
                                     model_ws=ml.model_ws,
                                     verbose=True,
                                     forgive=False,check=False)

    assert np.array_equal(ml.wel[0], ml1.wel[0])
    assert np.array_equal(ml.wel[1], ml1.wel[1])

    ml1.write_input()


def test_single_mflist_entry_load():

    import os
    import numpy as np
    import flopy

    model_ws = os.path.join("..","examples","data","freyberg")
    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=model_ws,load_only=["WEL"],check=False)
    w = m.wel
    spd = w.stress_period_data
    flopy.modflow.ModflowWel(m,stress_period_data={0:[0,0,0,0.0]})
    m.external_path = "."
    m.change_model_ws("temp",reset_external=True)
    m.write_input()

    mm = flopy.modflow.Modflow.load("freyberg.nam", model_ws="temp",forgive=False)
    assert mm.wel.stress_period_data
    mm.write_input()

if __name__ == '__main__':
    test_mflist_external()
    test_single_mflist_entry_load()
