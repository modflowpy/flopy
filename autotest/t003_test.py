import flopy
import os.path

def test_loadfreyberg():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    assert(os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'freyberg.nam'
    ml = flopy.modflow.Modflow.load(namefile)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False
    return

def test_loadoahu():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'parameters')
    assert(os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'Oahu_01.nam'
    ml = flopy.modflow.Modflow.load(namefile)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False
    return

def test_loadtwrip():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'parameters')
    assert(os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'twrip.nam'
    ml = flopy.modflow.Modflow.load(namefile, verbose=True)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False
    return


if __name__ == '__main__':
    test_loadfreyberg()
    test_loadoahu()
    test_loadtwrip()
