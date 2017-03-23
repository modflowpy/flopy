"""
Test the gmg load and write with an external summary file
"""
import os
import shutil
import flopy
try:
    import pymake
except:
    print('could not import pymake')

path = os.path.join('..', 'examples', 'data', 'freyberg')
cpth = os.path.join('temp', 't046')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

mf_items = ['freyberg.nam']
pths = []
for val in mf_items:
    pths.append(path)


def load_and_write(mfnam, pth):
    """
    test045 load and write of MODFLOW-2005 GMG example problem
    """
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

    # change model workspace
    m.change_model_ws(apth)

    # recreate oc file
    oc = m.oc
    unitnumber = [oc.unit_number[0], oc.iuhead, oc.iuddn, 0, 0]
    spd = {(0,0): ['save head', 'save drawdown']}
    chedfm = '(10(1X1PE13.5))'
    cddnfm = '(10(1X1PE13.5))'
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd,
                                 chedfm=chedfm, cddnfm=cddnfm,
                                 unitnumber=unitnumber)

    # rewrite files
    m.write_input()

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'new model run did not terminate successfully'
        fn1 = os.path.join(apth, mfnam)

    if run:
        # compare heads
        fsum = os.path.join(compth,
                            '{}.head.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_heads(fn0, fn1, outfile=fsum)
        except:
            print('could not perform head comparison')

        assert success, 'head comparison failure'

        # compare heads
        fsum = os.path.join(compth,
                            '{}.ddn.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_heads(fn0, fn1, outfile=fsum,
                                           text='drawdown')
        except:
            print('could not perform drawdown comparison')

        assert success, 'head comparison failure'

        # compare budgets
        fsum = os.path.join(compth,
                            '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
        success = False
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
        yield load_and_write, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_and_write(namfile, pth)
