# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:29:35 2020

@author: Artesia
"""


import os
import numpy as np
import flopy
import matplotlib.pyplot as plt

def test_plotting_with_quasi3d_layers():
    modelname = 'model_mf'
    model_ws = os.path.join('.', 'temp', 't069a')
    exe_name = 'mf2005'
    mf = flopy.modflow.Modflow(modelname, model_ws=model_ws, exe_name=exe_name)
    
    # Model domain and grid definition
    Lx = 1000.
    Ly = 1000.
    ztop = 0.
    zbot = -30.
    nlay = 3
    nrow = 10
    ncol = 10
    delr = Lx / ncol
    delc = Ly / nrow
    laycbd = [0]*(nlay)
    laycbd[0] = 1
    botm = np.linspace(ztop, zbot, nlay + np.sum(laycbd) + 1)[1:]
    
    # Create the discretization object
    flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                             top=ztop, botm=botm, laycbd=laycbd)
    
    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.
    strt[:, :, -1] = 0.
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    # Add LPF package to the MODFLOW model
    flopy.modflow.ModflowLpf(mf, hk=10., vka=10., ipakcb=53, vkcb=10)
    
    # add a well
    row = int((nrow-1)/2)
    col = int((ncol-1)/2)
    spd = {0:[[1, row, col, -1000]]}
    flopy.modflow.ModflowWel(mf, stress_period_data=spd)
    
    # Add OC package to the MODFLOW model
    spd = {(0, 0): ['save head', 'save budget']}
    flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
    
    # Add PCG package to the MODFLOW model
    flopy.modflow.ModflowPcg(mf)
    
    # Write the MODFLOW model input files
    mf.write_input()
    
    # Run the MODFLOW model
    success, buff = mf.run_model()
    
    # read output
    hf = flopy.utils.HeadFile(os.path.join(mf.model_ws,'{}.hds'.format(mf.name)))
    head = hf.get_data(totim=1.0)
    cbb = flopy.utils.CellBudgetFile(os.path.join(mf.model_ws,'{}.cbc'.format(mf.name)))
    frf = cbb.get_data(text='FLOW RIGHT FACE', totim=1.0)[0]
    fff = cbb.get_data(text='FLOW FRONT FACE', totim=1.0)[0]
    flf = cbb.get_data(text='FLOW LOWER FACE', totim=1.0)[0]
    
    # plot a map
    plt.figure()
    mv = flopy.plot.PlotMapView(model=mf,layer=1)
    mv.plot_array(head)
    mv.plot_grid()
    mv.plot_array(head)
    mv.contour_array(head)
    mv.plot_ibound()
    mv.plot_bc('wel')
    mv.plot_discharge(frf,fff, head=head)
    
    # plot a cross-section
    plt.figure()
    cs = flopy.plot.PlotCrossSection(model=mf, line={'row':int((nrow-1)/2)})
    cs.plot_grid()
    cs.plot_array(head)
    cs.contour_array(head)
    cs.plot_ibound()
    cs.plot_bc('wel')
    cs.plot_discharge(frf, fff, flf, head=head)

