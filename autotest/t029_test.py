import os
import flopy

pthtest = os.path.join('..', 'examples', 'data', 'mfgrd_test')


def test_mfgrddis():
    grbnam = 'nwtp3.dis.grb'
    fn = os.path.join(pthtest, grbnam)
    dis = flopy.utils.MfGrdFile(fn, verbose=True)

    iverts, verts = dis.get_verts()
    mg = dis.mg
    extents = mg.get_extent()
    vertc = dis.get_centroids()
    errmsg = 'extents {} of {} '.format(extents, grbnam) + \
             'does not equal (0.0, 8000.0, 0.0, 8000.0)'
    assert extents == (0.0, 8000.0, 0.0, 8000.0), errmsg
    errmsg = 'shape of {} {} '.format(grbnam, verts.shape) + \
             'not equal to (32000, 2).'
    assert verts.shape == (32000, 2), errmsg
    errmsg = 'ncells of {} {} '.format(grbnam, len(iverts)) + \
             'not equal to 6400.'
    assert len(iverts) == 6400, errmsg


def test_mfgrddisv():
    fn = os.path.join(pthtest, 'flow.disv.grb')
    disv = flopy.utils.MfGrdFile(fn, verbose=True)

    iverts, verts = disv.get_verts()
    errmsg = 'shape of flow.disv {} not equal to (156, 2).'.format(verts.shape)
    assert verts.shape == (156, 2), errmsg
    errmsg = 'ncells of flow.disv {} not equal to 218.'.format(len(iverts))
    assert len(iverts) == 218, errmsg


if __name__ == '__main__':
    test_mfgrddis()
    test_mfgrddisv()
