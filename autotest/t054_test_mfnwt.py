"""
Try to load all of the MODFLOW-USG examples in ../examples/data/mfusg_test.
These are the examples that are distributed with MODFLOW-USG.
"""

import os
import flopy
import pymake

# make the working directory
tpth = os.path.join('temp', 't054')
if not os.path.isdir(tpth):
    os.makedirs(tpth)

# build list of name files to try and load
nwtpth = os.path.join('..', 'examples', 'data', 'mf2005_test')
nwt_files = []
m = flopy.modflow.Modflow('test', version='mfnwt')
for path, subdirs, files in os.walk(nwtpth):
    for name in files:
        if name.endswith('.nam'):
            pth = os.path.join(nwtpth, name)
            nf = flopy.utils.parsenamefile(pth, m.mfnam_packages)
            lpf = False
            wel = False
            for key, value in nf.items():
                if 'LPF' in value.filetype:
                    lpf = True
                if 'WEL' in value.filetype:
                    wel = True
            if lpf and wel:
                nwt_files.append(os.path.join(path, name))

mfnwt_exe = 'mfnwt'
v = flopy.which(mfnwt_exe)

run = True
if v is None:
    run = False


#
def test_mfnwt_model():
    for fnwt in nwt_files:
        d, f = os.path.split(fnwt)
        yield mfnwt_model, f, d

# function to load a MODFLOW-2005 model, convert to a MFNWT model,
# write it back out, run the MFNWT model, load the MFNWT model,
# and compare the results.
def mfnwt_model(namfile, model_ws):
    # load MODFLOW-2005 models as MODFLOW-NWT models
    m = flopy.modflow.Modflow.load(namfile, model_ws=model_ws,
                                   version='mfnwt', verbose=True, check=False,
                                   exe_name=mfnwt_exe)
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False
    # convert to MODFLOW-NWT model
    m.set_version('mfnwt')
    # extract data from existing flow package
    flowpaks = ['LPF']
    for pak in m.get_package_list():
        if pak == 'LPF':
            lpf = m.get_package(pak)
            layavg = lpf.layavg
            laytyp = lpf.laytyp
            layvka = lpf.layvka
            ss = lpf.ss
            sy = lpf.sy
            hk = lpf.hk
            vka = lpf.vka
            hani = lpf.hani
            chani = lpf.chani
            ipakcb = lpf.ipakcb
            unitnumber = lpf.unit_number[0]
            # remove existing package
            m.remove_package(pak)
            break
    # create UPW file from existing flow package
    upw = flopy.modflow.ModflowUpw(m, layavg=layavg, laytyp=laytyp,
                                   ipakcb=ipakcb, unitnumber=unitnumber,
                                   layvka=layvka, hani=hani, chani=chani,
                                   hk=hk, vka=vka, ss=ss, sy=sy)
    # remove the existing solver
    solvers = ['SIP', 'PCG', 'PCGN', 'GMG', 'DE4']
    for pak in m.get_package_list():
        solv = m.get_package(pak)
        if pak in solvers:
            unitnumber = solv.unit_number[0]
            m.remove_package(pak)
    nwt = flopy.modflow.ModflowNwt(m, unitnumber=unitnumber)

    # add specify option to the well package
    wel = m.get_package('WEL')
    wel.specify = True
    wel.phiramp = 1.0e-5
    wel.phiramp_unit = 2

    # change workspace and write MODFLOW-NWT model
    tdir = os.path.splitext(namfile)[0]
    pth = os.path.join(tpth, tdir)
    m.change_model_ws(pth)
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn0 = os.path.join(pth, namfile)

    # reload the model just written
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mfnwt', verbose=True, check=False,
                                   exe_name=mfnwt_exe)
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False

    # change workspace and write MODFLOW-NWT model
    pthf = os.path.join(pth, 'flopy')
    m.change_model_ws(pthf)
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, 'base model run did not terminate successfully'
        fn1 = os.path.join(pthf, namfile)

    if run:
        fsum = os.path.join(pth, '{}.head.out'.format(tdir))
        success = False
        try:
            success = pymake.compare_heads(fn0, fn1, outfile=fsum)
        except:
            print('could not perform head comparison')

        assert success, 'head comparison failure'

        fsum = os.path.join(pth, '{}.budget.out'.format(tdir))
        success = False
        try:
            success = pymake.compare_budget(fn0, fn1,
                                            max_incpd=0.1, max_cumpd=0.1,
                                            outfile=fsum)
        except:
            print('could not perform budget comparison')

        assert success, 'budget comparison failure'

    return

if __name__ == '__main__':
    for fnwt in nwt_files:
        d, f = os.path.split(fnwt)
        mfnwt_model(f, d)
