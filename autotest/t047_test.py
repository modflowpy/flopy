"""
Test the gmg load and write with an external summary file
"""
import os
import shutil
import numpy as np
import flopy
from flopy.utils.recarray_utils import recarray

cpth = os.path.join('temp', 't047')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

def get_namefile_entries(fpth):
    try:
        f = open(fpth, 'r')
    except:
        print('could not open...{}'.format(fpth))
        return None
    dtype = [('ftype', '|S12'), ('unit', np.int),
             ('filename', '|S128'), ('status', '|S10')]
    lines = f.readlines()
    data = []
    for line in lines:
        if line[0] == '#':
            continue
        t = line.rstrip().split()
        ftype = t[0]
        if isinstance(ftype, bytes):
            ftype = ftype.decode()
        iu = int(t[1])
        filename = t[2]
        if isinstance(filename, bytes):
            filename = filename.decode()
        if len(t) < 4:
            status = ''
        else:
            status = t[3]
            if isinstance(status, bytes):
                status = status.decode()

        data.append([ftype, iu, filename, status])
    data = recarray(data, dtype)
    return data

def test_gage():

    mnam = 'gage_test'

    m = flopy.modflow.Modflow(modelname=mnam, model_ws=cpth)
    dis = flopy.modflow.ModflowDis(m)
    spd = {(0, 0): ['print head'],
           (0, 1): [],
           (0, 249): ['print head'],
           (0, 250): [],
           (0, 499): ['print head', 'save ibound'],
           (0, 500): [],
           (0, 749): ['print head', 'ddreference'],
           (0, 750): [],
           (0, 999): ['print head', 'save budget', 'save drawdown']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, cboufm='(20i5)')

    gages = [[-1, -26, 1], [-2, -27, 1]]
    gage = flopy.modflow.ModflowGage(m, numgage=2,
                                     gage_data=gages)

    m.write_input()

    # check that the gage output units entries are in the name file
    fpth = os.path.join(cpth, '{}.nam'.format(mnam))
    entries = get_namefile_entries(fpth)
    for idx, g in enumerate(gages):
        if g[0] < 0:
            iu = abs(g[1])
        else:
            iu = abs(g[2])
        found = False
        iun = None
        for jdx, iut in enumerate(entries['unit']):
            if iut == iu:
                found = True
                iun = iut
                break
        assert found, '{} not in name file entries'.format(iu)

    return

def test_gage_files():

    mnam = 'gage_test_files'

    m = flopy.modflow.Modflow(modelname=mnam, model_ws=cpth)
    dis = flopy.modflow.ModflowDis(m)
    spd = {(0, 0): ['print head'],
           (0, 1): [],
           (0, 249): ['print head'],
           (0, 250): [],
           (0, 499): ['print head', 'save ibound'],
           (0, 500): [],
           (0, 749): ['print head', 'ddreference'],
           (0, 750): [],
           (0, 999): ['print head', 'save budget', 'save drawdown']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, cboufm='(20i5)')

    gages = [[-1, -26, 1], [-2, -27, 1]]
    files = ['gage1.go', 'gage2.go']
    gage = flopy.modflow.ModflowGage(m, numgage=2,
                                     gage_data=gages, files=files)

    m.write_input()

    # check that the gage output file entries are in the name file
    fpth = os.path.join(cpth, '{}.nam'.format(mnam))
    entries = get_namefile_entries(fpth)
    for idx, f in enumerate(files):
        found = False
        iun = None
        for jdx, fnn in enumerate(entries['filename']):
            if isinstance(fnn, bytes):
                fnn = fnn.decode()
            if fnn == f:
                found = True
                iun = entries[jdx]['unit']
                break
        assert found, '{} not in name file entries'.format(f)
        iu = abs(gages[idx][1])
        assert iu == iun, '{} unit not equal to {} '.format(f, iu) + \
                          '- name file unit = {}'.format(iun)

    return

def test_gage_filenames0():

    mnam = 'gage_test_filenames0'

    m = flopy.modflow.Modflow(modelname=mnam, model_ws=cpth)
    dis = flopy.modflow.ModflowDis(m)
    spd = {(0, 0): ['print head'],
           (0, 1): [],
           (0, 249): ['print head'],
           (0, 250): [],
           (0, 499): ['print head', 'save ibound'],
           (0, 500): [],
           (0, 749): ['print head', 'ddreference'],
           (0, 750): [],
           (0, 999): ['print head', 'save budget', 'save drawdown']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, cboufm='(20i5)')

    gages = [[-1, -126, 1], [-2, -127, 1]]
    filenames = 'mygages0.gage'
    gage = flopy.modflow.ModflowGage(m, numgage=2,
                                     gage_data=gages, filenames=filenames)

    m.write_input()

    # check that the gage output units entries are in the name file
    fpth = os.path.join(cpth, '{}.nam'.format(mnam))
    entries = get_namefile_entries(fpth)
    for idx, g in enumerate(gages):
        if g[0] < 0:
            iu = abs(g[1])
        else:
            iu = abs(g[2])
        found = False
        iun = None
        for jdx, iut in enumerate(entries['unit']):
            if iut == iu:
                found = True
                iun = iut
                break
        assert found, '{} not in name file entries'.format(iu)

    return

def test_gage_filenames():

    mnam = 'gage_test_filenames'

    m = flopy.modflow.Modflow(modelname=mnam, model_ws=cpth)
    dis = flopy.modflow.ModflowDis(m)
    spd = {(0, 0): ['print head'],
           (0, 1): [],
           (0, 249): ['print head'],
           (0, 250): [],
           (0, 499): ['print head', 'save ibound'],
           (0, 500): [],
           (0, 749): ['print head', 'ddreference'],
           (0, 750): [],
           (0, 999): ['print head', 'save budget', 'save drawdown']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd, cboufm='(20i5)')

    gages = [[-1, -126, 1], [-2, -127, 1]]
    filenames = ['mygages.gage', 'mygage1.go', 'mygage2.go']
    gage = flopy.modflow.ModflowGage(m, numgage=2,
                                     gage_data=gages, filenames=filenames)

    m.write_input()

    # check that the gage output file entries are in the name file
    fpth = os.path.join(cpth, '{}.nam'.format(mnam))
    entries = get_namefile_entries(fpth)
    for idx, f in enumerate(filenames[1:]):
        found = False
        iun = None
        for jdx, fnn in enumerate(entries['filename']):
            if isinstance(fnn, bytes):
                fnn = fnn.decode()
            if fnn == f:
                found = True
                iun = entries[jdx]['unit']
                break
        assert found, '{} not in name file entries'.format(f)
        iu = abs(gages[idx][1])
        assert iu == iun, '{} unit not equal to {} '.format(f, iu) + \
                          '- name file unit = {}'.format(iun)

    return


if __name__ == '__main__':
    test_gage()
    test_gage_files()
    test_gage_filenames()
