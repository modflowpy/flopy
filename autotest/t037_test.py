"""
Some basic tests for SWR2 load.
"""

import os
import flopy

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp', 't037')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)

mf_items = ['swiex1.nam', 'swiex2_strat.nam', 'swiex3.nam']
pths = []
for val in mf_items:
    pths.append(path)


def load_swi(mfnam, pth):
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

    m = flopy.modflow.Modflow.load(mfnam, model_ws=lpth, verbose=True,
                                   exe_name=exe_name)
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn0 = os.path.join(lpth, mfnam)

    # write free format files -
    # won't run without resetting to free format - evt external file issue
    m.free_format_input = True

    # rewrite files
    m.change_model_ws(apth,
                      reset_external=True)  # l1b2k_bath wont run without this
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
            print('could not perform budget comparison')

        assert success, 'budget comparison failure'

    return


def test_mf2005swi2load():
    for namfile, pth in zip(mf_items, pths):
        yield load_swi, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_swi(namfile, pth)
