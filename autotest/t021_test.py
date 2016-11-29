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
                               external_path="ref")
    dis = flopy.modflow.ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    wel_data = {0: [[0, 0, 0, -1], [1, 1, 1, -1]],
                1: [[0, 0, 0, -2], [1, 1, 1, -1]]}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=wel_data)
    ml.write_input()

    ml1 = flopy.modflow.Modflow.load("mflist_test.nam",
                                     model_ws=ml.model_ws,
                                     verbose=True,
                                     forgive=False)
    assert np.array_equal(ml.wel[0], ml1.wel[0])
    assert np.array_equal(ml.wel[1], ml1.wel[1])


if __name__ == '__main__':
    test_mflist_external()
