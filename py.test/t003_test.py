import flopy
import os.path

def test_loadfreyberg():
    pth = os.path.join('..', 'examples', 'freyberg')
    assert(os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'freyberg.nam'
    ml = flopy.modflow.Modflow.load(namefile)
    assert isinstance(ml, flopy.modflow.Modflow)
    return

if __name__ == '__main__':
    test_loadfreyberg()
