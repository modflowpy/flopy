"""
Some basic tests for LAKE load.
"""

import os
import flopy

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp', 't025')

mf_items = ['l1b2k_bath.nam', 'l2a_2k.nam', 'lakeex3.nam', 'l1b2k.nam',
            'l1a2k.nam']
pths = []
for mi in mf_items:
    pths.append(path)

exe_name = 'mf2005'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def load_lak(mfnam, pth, run):
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

    m = flopy.modflow.Modflow.load(mfnam, model_ws=lpth, verbose=True, forgive=False,
                                   exe_name=exe_name)
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            msg = 'could not run base model ' + \
                  '{}'.format(os.path.splitext(mfnam)[0])
            print(msg)
            pass
        msg = 'base model {} '.format(os.path.splitext(mfnam)[0]) + \
              'run did not terminate successfully'
        assert success, msg
        msg = 'base model {} '.format(os.path.splitext(mfnam)[0]) + \
              'run terminated successfully'
        print(msg)
        fn0 = os.path.join(lpth, mfnam)


    # write free format files - wont run without resetting to free format - evt externa file issue
    m.free_format_input = True

    # rewrite files
    m.change_model_ws(apth, reset_external=True)  # l1b2k_bath wont run without this
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            msg = 'could not run new model ' + \
                  '{}'.format(os.path.splitext(mfnam)[0])
            print(msg)
            pass
        msg = 'new model {} '.format(os.path.splitext(mfnam)[0]) + \
              'run did not terminate successfully'
        assert success, msg
        msg = 'new model {} '.format(os.path.splitext(mfnam)[0]) + \
              'run terminated successfully'
        print(msg)
        fn1 = os.path.join(apth, mfnam)

        fsum = os.path.join(compth,
                            '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
    if run:
        try:
            success = pymake.compare_budget(fn0, fn1,
                                            max_incpd=0.1, max_cumpd=0.1,
                                            outfile=fsum)
        except:
            print('could not perform budget comparison')

        assert success, 'budget comparison failure'


    return


def test_mf2005load():
    for namfile, pth in zip(mf_items, pths):
        yield load_lak, namfile, pth, run
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_lak(namfile, pth, run)
