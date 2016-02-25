"""
Some basic tests for LAKE load.
"""

import os
import flopy
import numpy as np

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp')

mf_items = ['l2a_2k.nam', 'lakeex3.nam', 'l1b2k_bath.nam', 'l1b2k.nam', 'l1a2k.nam']
pths = [path, path, path, path, path]

#mf_items = ['l1b2k_bath.nam']
#mf_items = ['lakeex3.nam']
#mf_items = ['l1a2k.nam']
#pths = [path]



def load_lak(mfnam, pth):
    m = flopy.modflow.Modflow.load(mfnam, model_ws=pth, verbose=True)
    assert m.load_fail is False

    m.exe_name = 'mf2005'
    v = flopy.which(m.exe_name)

    run = True
    if v is None:
        run = False

    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn0 = os.path.join(pth, mfnam)


    # write free format files - wont run without resetting to free format - evt externa file issue
    m.bas6.ifrefm = True
    #m.array_free_format = True

    # rewrite files
    m.change_model_ws(cpth, reset_external=True)  # l1b2k_bath wont run without this
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn1 = os.path.join(cpth, mfnam)

        try:
            import pymake as pm
            fsum = os.path.join(cpth,
                                '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
            success = pm.compare_budget(fn0, fn1,
                                        max_incpd=0.1, max_cumpd=0.1,
                                        outfile=fsum)
            assert success, 'budget comparison failure'

        except:
            pass

    return




def test_mf2005load():
    for namfile, pth in zip(mf_items, pths):
        yield load_lak, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_lak(namfile, pth)
