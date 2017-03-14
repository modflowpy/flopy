# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import shutil
import flopy
try:
    import pymake
except:
    print('could not import pymake')

cpth = os.path.join('temp', 't049')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

def test_modpath():

    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    mfnam = 'freyberg.nam'

    mf2005_exe = 'mf2005'
    v = flopy.which(mf2005_exe)

    mpth_exe = 'mp6'
    v2 = flopy.which(mpth_exe)

    run = True
    if v is None or v2 is None:
        run = False
    try:
        import pymake
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
        pymake.setup(os.path.join(pth, mfnam), lpth)
    except:
        run = False
        lpth = pth

    m = flopy.modflow.Modflow.load(mfnam, model_ws=lpth, verbose=True,
                                   exe_name=mf2005_exe)
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'modflow model run did not terminate successfully'

    # create the forward modpath file
    mpnam = 'freybergmp'
    mp = flopy.modpath.Modpath(mpnam, exe_name=mpth_exe, modflowmodel=m,
                               model_ws=lpth)
    mpbas = flopy.modpath.ModpathBas(mp, hnoflo=m.bas6.hnoflo,
                                     hdry=m.lpf.hdry,
                                     ibound=m.bas6.ibound.array, prsity=0.2,
                                     prsityCB=0.2)
    sim = mp.create_mpsim(trackdir='forward', simtype='endpoint',
                          packages='RCH')

    # write forward particle track files
    mp.write_input()

    if run:
        try:
            success, buff = mp.run_model(silent=False)
        except:
            pass
        assert success, 'forward modpath model run ' + \
                        'did not terminate successfully'

    mp.run_model()

    mpnam = 'freybergmpp'
    mpp = flopy.modpath.Modpath(mpnam, exe_name=mpth_exe,
                                modflowmodel=m, model_ws=lpth)
    mpbas = flopy.modpath.ModpathBas(mpp, hnoflo=m.bas6.hnoflo,
                                     hdry=m.lpf.hdry,
                                     ibound=m.bas6.ibound.array, prsity=0.2,
                                     prsityCB=0.2)
    sim = mpp.create_mpsim(trackdir='backward', simtype='pathline',
                           packages='WEL')

    # write backward particle track files
    mpp.write_input()

    if run:
        try:
            success, buff = mpp.run_model(silent=False)
        except:
            pass
        assert success, 'backward modpath model run ' + \
                        'did not terminate successfully'

    # load modpath output files
    if run:
        endfile = os.path.join(lpth, mp.sim.endpoint_file)
        pthfile = os.path.join(lpth, mpp.sim.pathline_file)
    else:
        endfile = os.path.join('..', 'examples', 'data', 'mp6_examples',
                               'freybergmp.gitmpend')
        pthfile = os.path.join('..', 'examples', 'data', 'mp6_examples',
                               'freybergmpp.gitmppth')

    # load the endpoint data
    try:
        endobj = flopy.utils.EndpointFile(endfile)
    except:
        assert False, 'could not load endpoint file'
    ept = endobj.get_alldata()
    assert ept.shape == (695,), 'shape of endpoint file is not (695,)'

    # load the pathline data
    try:
        pthobj = flopy.utils.PathlineFile(pthfile)
    except:
        assert False, 'could not load pathline file'
    plines = pthobj.get_alldata()
    assert len(plines) == 576, 'there are not 576 particle pathlines in file'

    return


if __name__ == '__main__':
    test_modpath()
