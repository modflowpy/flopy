import os
import sys
import time
import numpy as np

import flopy

base_pth = os.path.join('data', 'uspb', 'flopy')
cf_pth = os.path.join('data', 'uspb', 'cf')
res_pth = os.path.join('data', 'uspb', 'results')

def cf_model(model, k, i, j, base, Q=-100):
    wd = {1: [[k, i, j, Q]]}
    model.remove_package('WEL')
    wel = flopy.modflow.ModflowWel(model=model, stress_period_data=wd)
    wel.write_file()
    model.run_model(silent=True)
    # get the results
    hedObj = flopy.utils.HeadFile(os.path.join(cf_pth, 'DG.hds'), precision='double')
    cbcObj = flopy.utils.CellBudgetFile(os.path.join(cf_pth, 'DG.cbc'), precision='double')
    kk = hedObj.get_kstpkper()
    v = np.zeros((len(kk)), dtype=np.float)
    h = hedObj.get_ts((k, i, j))
    for idx, kon in enumerate(kk):
        if h[idx, 1] == model.lpf.hdry:
            v[idx] = np.nan
        else:
            v1 = cbcObj.get_data(kstpkper=kon, text='DRAINS', full3D=True)[0]
            v2 = cbcObj.get_data(kstpkper=kon, text='STREAM LEAKAGE', full3D=True)[0]
            v3 = cbcObj.get_data(kstpkper=kon, text='ET', full3D=True)[0]
            v[idx] = ((v1.sum() + v2.sum() + v3.sum()) - base) / (-Q)
    return v


ml = flopy.modflow.Modflow.load('DG.nam', version='mf2005', exe_name='mf2005dbl', 
                                verbose=True, model_ws=base_pth)

# set a few variables from the model
nrow, ncol = ml.dis.nrow, ml.dis.ncol
ibound = ml.bas6.ibound[3, :, :]

# create base model and run
ml.change_model_ws(cf_pth)
ml.write_input()
ml.run_model()

# get base model results
cbcObj = flopy.utils.CellBudgetFile(os.path.join(cf_pth, 'DG.cbc'), precision='double')
v1 = cbcObj.get_data(kstpkper=(0, 0), text='DRAINS', full3D=True)[0]
v2 = cbcObj.get_data(kstpkper=(0, 0), text='STREAM LEAKAGE', full3D=True)[0]
v3 = cbcObj.get_data(kstpkper=(0, 0), text='ET', full3D=True)[0]
baseQ = v1.sum() + v2.sum() + v3.sum()

# modify OC
ml.remove_package('OC')
stress_period_data = {(1,9): ['save head', 'save budget', 'print budget'],
                      (1,10): [],
                      (1,19): ['save head', 'save budget', 'print budget'], 
                      (1,20): [],
                      (1,29): ['save head', 'save budget', 'print budget'], 
                      (1,30): [],
                      (1,39): ['save head', 'save budget', 'print budget'], 
                      (1,40): [],
                      (1,49): ['save head', 'save budget', 'print budget'], 
                      (1,50): [],
                      (1,59): ['save head', 'save budget', 'print budget'], 
                      (1,60): [],
                      (1,69): ['save head', 'save budget', 'print budget'], 
                      (1,70): [],
                      (1,79): ['save head', 'save budget', 'print budget'], 
                      (1,80): [],
                      (1,89): ['save head', 'save budget', 'print budget'], 
                      (1,90): [],
                      (1,99): ['save head', 'save budget', 'print budget'], 
                      (1,100): []}
oc = flopy.modflow.ModflowOc(ml, stress_period_data=stress_period_data)
oc.write_file()

# calculate subset of model to run
nstep = 4
nrow2 = nrow // nstep
ncol2 = ncol // nstep

# open summary file
fs = open(os.path.join('data', 'uspb', 'uspb_capture_{}.out'.format(nstep)), 'w', 0)

# write some summary information
fs.write('Problem size: {} rows and {} columns.\n'.format(nrow, ncol))
fs.write('Capture fraction analysis performed every {} rows and columns.\n'.format(nstep))
fs.write('Maximum number of analyses: {} rows and {} columns.\n'.format(nrow2, ncol2))

# create array to store capture fraction data (subset of model)
cf_array = np.empty((10, nrow2, ncol2), dtype=np.float)
cf_array.fill(np.nan)

# timer for capture fraction analysis
start = time.time()

# capture fraction analysis
icnt = 0
jcnt = 0
idx = 0
for i in range(0, nrow, nstep):
    jcnt = 0
    for j in range(0, ncol, nstep):
        if ibound[i, j] < 1:
            sys.stdout.write('.')
        else:
            line = '\nrow {} of {} - col {} of {}\n'.format(icnt+1, nrow2, jcnt+1, ncol2)
            fs.write(line)
            sys.stdout.write(line)
            s0 = time.time()
            cf = cf_model(ml, 3, i, j, baseQ)
            s1 = time.time()
            line = '  model {} run time: {} seconds\n'.format(idx, s1-s0)
            fs.write(line)
            sys.stdout.write(line)
            idx += 1
            # add values to the array
            if icnt < nrow2 and jcnt < ncol2:
                cf_array[:, icnt, jcnt] = cf.copy()
        # increment jcnt
        jcnt += 1
    # increment icnt
    icnt += 1

# end timer for capture fraction analysis
end = time.time()
ets = end - start
line = '\n' + \
       'streamflow capture analysis took {} seconds.\n'.format(ets) + \
       'streamflow capture analysis took {} minutes.\n'.format(ets/60.) + \
       'streamflow capture analysis took {} hours.\n'.format(ets/3600.)
fs.write(line)
sys.stdout.write(line)

#close summary file
fs.close()

# clean up working directory
filelist = [f for f in os.listdir(cf_pth)]
for f in filelist:
    os.remove(os.path.join(cf_pth, f))

# create res_pth (if it doesn't exist) and save data
if not os.path.exists(res_pth):
    os.makedirs(res_pth)
for idx in range(10):
    fn = os.path.join(res_pth, 'USPB_capture_fraction_{:02d}_{:02d}.dat'.format(nstep, idx+1))
    print('saving capture fraction data to...{}'.format(os.path.basename(fn)))
    np.savetxt(fn, cf_array[idx, :, :], delimiter=' ')

