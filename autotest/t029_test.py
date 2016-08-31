import os
import flopy
import matplotlib.pyplot as plt


pthtest = os.path.join('..', 'examples', 'data', 'mfgrd_test')
newpth = os.path.join('.', 'temp')

def test_mfgrddisv():
    fn = os.path.join(pthtest, 'flow.disv.grb')
    disv = flopy.utils.MfGrdFile(fn, verbose=True)

    iverts, verts = disv.get_verts()
    errmsg = 'shape of flow.disv {} not equal to (156, 2).'.format(verts.shape)
    assert verts.shape == (156, 2), errmsg
    errmsg = 'ncells of flow.disv {} not equal to 218.'.format(len(iverts))
    assert len(iverts) == 218, errmsg

if __name__ == '__main__':
    test_mfgrddisv()
