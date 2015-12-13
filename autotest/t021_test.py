# Test modflow write adn run
import numpy as np

def test_mflist_external():
    import flopy.modflow as fmf
    ml = fmf.Modflow("mflist_test",model_ws="temp",external_path="ref")
    dis = fmf.ModflowDis(ml,1,10,10,nper=3,perlen=1.0)
    wel_data = {0:[[0,0,0,-1],[1,1,1,-1]],1:[[0,0,0,-2],[1,1,1,-1]]}
    wel = fmf.ModflowWel(ml,stress_period_data=wel_data)
    ml.write_input()

    ml1 = fmf.Modflow.load("mflist_test.nam",
                           model_ws=ml.model_ws,
                           verbose=True,
                           forgive=False)
    assert np.array_equal(ml.wel[0],ml1.wel[0])
    assert np.array_equal(ml.wel[1],ml1.wel[1])
if __name__ == '__main__':
    test_mflist_external()
