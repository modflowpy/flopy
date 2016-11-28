"""
Test loading and preserving existing unit numbers
"""
import os
import flopy

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp', 't036')

# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)

def test_unitnums_load_and_write():
    mfnam = 'testsfr2_tab.nam'

    # copy files
    exe_name = 'mf2005'
    v = flopy.which(exe_name)

    run = True
    if v is None:
        run = False
    try:
        import pymake
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
        apth = os.path.join(lpth, 'flopy')
        compth = lpth
        pymake.setup(os.path.join(pth, mfnam), lpth)
    except:
        run = False
        lpth = pth
        apth = cpth
        compth = cpth

    m = flopy.modflow.Modflow.load(mfnam, verbose=True, model_ws=lpth,
                                   exe_name=exe_name)
    assert m.load_fail is False, 'failed to load all packages'

    msg = 'modflow-2005 testsfr2_tab does not have 1 layer, 7 rows, and 100 colummns'
    v = (m.nlay, m.nrow, m.ncol, m.nper)
    assert v == (1, 7, 100, 50), msg

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn0 = os.path.join(lpth, mfnam)

    # rewrite files
    m.change_model_ws(apth, reset_external=True)
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn1 = os.path.join(apth, mfnam)

    if run:
        fsum = os.path.join(compth,
                            '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
        try:
            success = pymake.compare_budget(fn0, fn1,
                                            max_incpd=0.1, max_cumpd=0.1,
                                            outfile=fsum)
        except:
            print('could not perform ls'
                  'budget comparison')

        assert success, 'budget comparison failure'

    return

if __name__ == '__main__':
    test_unitnums_load_and_write()
