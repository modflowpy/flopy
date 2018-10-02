# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import shutil
import flopy
import numpy as np
import matplotlib.pyplot as plt
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

mf2005_exe = 'mf2005'
v = flopy.which(mf2005_exe)

mpth_exe = 'mp6'
v2 = flopy.which(mpth_exe)

rung = True
if v is None or v2 is None:
    rung = False


def test_modpath():

    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    mfnam = 'freyberg.nam'

    run = rung
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

def test_pathline_plot():
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    mfnam = 'freyberg.nam'

    run = rung
    try:
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
    except:
        run = False
        lpth = pth

    nampath = os.path.join(lpth, mfnam)
    assert os.path.exists(nampath), "namefile {} doesn't exist.".format(nampath)
    # load the modflow files for model map
    m = flopy.modflow.Modflow.load(mfnam, model_ws=lpth, verbose=True, forgive=False,
                                   exe_name=mf2005_exe)

    # load modpath output files
    if run:
        pthfile = os.path.join(lpth, 'freybergmpp.mppth')
    else:
        pthfile = os.path.join('..', 'examples', 'data', 'mp6_examples',
                               'freybergmpp.gitmppth')

    # load the pathline data
    try:
        pthobj = flopy.utils.PathlineFile(pthfile)
    except:
        assert False, 'could not load pathline file'

    # determine version
    ver = pthobj.version
    assert ver == 6, '{} is not a MODPATH version 6 pathline file'.format(fpth)

    # get all pathline data
    plines = pthobj.get_alldata()

    mm = flopy.plot.ModelMap(model=m)
    try:
        mm.plot_pathline(plines, colors='blue', layer='all')
    except:
        assert False, 'could not plot pathline with layer="all"'

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, 'could not plot grid and ibound'

    try:
        fpth = os.path.join(lpth, 'pathline.png')
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, 'could not save plot as {}'.format(fpth)

    mm = flopy.plot.ModelMap(model=m)
    try:
        mm.plot_pathline(plines, colors='green', layer=0)
    except:
        assert False, 'could not plot pathline with layer=0'

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, 'could not plot grid and ibound'

    try:
        fpth = os.path.join(lpth, 'pathline2.png')
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, 'could not save plot as {}'.format(fpth)

    mm = flopy.plot.ModelMap(model=m)
    try:
        mm.plot_pathline(plines, colors='red')
    except:
        assert False, 'could not plot pathline'

    # plot the grid and ibound array
    try:
        mm.plot_grid()
        mm.plot_ibound()
    except:
        assert False, 'could not plot grid and ibound'

    try:
        fpth = os.path.join(lpth, 'pathline3.png')
        plt.savefig(fpth)
        plt.close()
    except:
        assert False, 'could not save plot as {}'.format(fpth)

    return

def test_mp5_load():

    # load the base freyberg model
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    # load the modflow files for model map
    m = flopy.modflow.Modflow.load('freyberg.nam', model_ws=pth, check=False,
                                   verbose=True, forgive=False)

    # load the pathline data
    fpth = os.path.join('..', 'examples', 'data', 'mp5', 'm.ptl')
    try:
        pthobj = flopy.utils.PathlineFile(fpth)
    except:
        assert False, 'could not load pathline file'

    # load endpoint data
    fpth = os.path.join('..', 'examples', 'data', 'mp5', 'm.ept')
    try:
        endobj = flopy.utils.EndpointFile(fpth, verbose=True)
    except:
        assert False, 'could not load endpoint file'

    # determine version
    ver = pthobj.version
    assert ver == 5, '{} is not a MODPATH version 5 pathline file'.format(fpth)

    # read all of the pathline and endpoint data
    plines = pthobj.get_alldata()
    epts = endobj.get_alldata()

    # determine the number of particles in the pathline file
    nptl = pthobj.nid.shape[0]
    assert nptl == 64, 'number of MODPATH 5 particles does not equal 64'

    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, nptl))

    # plot the pathlines one pathline at a time
    mm = flopy.plot.ModelMap(model=m)
    for n in range(nptl):
        p = pthobj.get_data(partid=n)
        e = endobj.get_data(partid=n)
        try:
            mm.plot_pathline(p, colors=colors[n], layer='all')
            mm.plot_endpoint(e)
        except:
            assert False, 'could not plot pathline {} '.format(n+1) + \
                          'with layer="all"'

    # plot the grid and ibound array
    try:
        mm.plot_grid(lw=0.5)
        mm.plot_ibound()
    except:
        assert False, 'could not plot grid and ibound'

    try:
        fpth = os.path.join(cpth, 'mp5.pathline.png')
        plt.savefig(fpth, dpi=300)
        plt.close()
    except:
        assert False, 'could not save plot as {}'.format(fpth)

    return

if __name__ == '__main__':
    #test_modpath()
    #test_pathline_plot()
    test_mp5_load()
