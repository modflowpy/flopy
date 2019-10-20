"""
Test the observation process load and write
"""
import os
import shutil
import filecmp
import flopy
try:
    import pymake
except ImportError:
    print('could not import pymake')
    pymake = False

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp', 't048')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

mf_items = ['fhb.nam', 'l1a2k.nam', 'l1b2k.nam', 'l1b2k_bath.nam',
            'lakeex3.nam']
#mf_items = ['l1b2k_bath.nam']
pths = []
for val in mf_items:
    pths.append(path)


# test044 load and write of MODFLOW-2005 FHB example problem
def load_and_write_fhb(mfnam, pth):

    exe_name = 'mf2005'
    v = flopy.which(exe_name)

    if pymake:
        run = v is not None
        lpth = os.path.join(cpth, os.path.splitext(mfnam)[0])
        apth = os.path.join(lpth, 'flopy')
        compth = lpth
        pymake.setup(os.path.join(pth, mfnam), lpth)
    else:
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
            success = False
        assert success, 'base model run did not terminate successfully'
        fn0 = os.path.join(lpth, mfnam)

    # rewrite files
    m.change_model_ws(apth, reset_external=True)
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            success = False
        assert success, 'new model run did not terminate successfully'
        fn1 = os.path.join(apth, mfnam)

    if run:
        fsum = os.path.join(compth,
                            '{}.head.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_heads(fn0, fn1, outfile=fsum)
        except:
            success = False
            print('could not perform head comparison')

        assert success, 'head comparison failure'

        fsum = os.path.join(compth,
                            '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_budget(fn0, fn1,
                                            max_incpd=0.1, max_cumpd=0.1,
                                            outfile=fsum)
        except:
            success = False
            print('could not perform budget comparison')

        assert success, 'budget comparison failure'

    return

def test_mf2005fhbload():
    for namfile, pth in zip(mf_items, pths):
        yield load_and_write_fhb, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_and_write_fhb(namfile, pth)
