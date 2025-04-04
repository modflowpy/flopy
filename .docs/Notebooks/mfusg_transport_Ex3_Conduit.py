#!/usr/bin/env python
# coding: utf-8

# ## MF-USG Example Problems for Transport of Solute through a Conduit within a Multi-Aquifer System 

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other Enhancements to MODFLOW-USG, GSI Environmental, July 2024 http://www.gsi-net.com/en/software/free-software/USG-Transport.html
# 
# This test problem discusses transport of a dissolved chemical species through a conduit connecting two confined aquifers.One aquifer is contaminated, the other not. Problem sees how well acts as conduit (no pumping) to contaminate the other aquifer, while c=0 boundaries in the first aquifer cleans it up.
# 
# The simulation domain of 47,000 feet by 47,000 feet square is discretized by 2 layers, 100 rows, and 100 columns of dimensions Δx=Δy=470feet, and Δz=10feet. The two layers are separated by a confining aquitard with zero leakance and a thickness of 10 feet. The overlying aquifer has an ambient gradient from south to north with a constant head boundary value of 30 feet along the south boundary and a constant head boundary value of 10 feet along the north boundary. The lower aquifer has a constant head of 60 around the entire perimeter. A conduit of 1 foot diameter connects the two aquifers resulting in flow up through the conduit into layer 1 from the bottom aquifer as a result of the head differential, with a subsequent radial flow component due to the mound and a northward flow component due to the ambient gradient. The steady-state flow-field thus generated was used for the transport simulation.
# 
# The transport simulation with the BCT package was conducted for a period of 3,000 days with a fixed time-step size of 30 days. The concentration of water in the upper aquifer is zero at the start of the simulation, while the concentration of water in the lower aquifer is one at the start of the simulation. The upstream weighting scheme was used for the simulation with dispersivities in the longitudinal and transverse directions set to zero. Figure Ex 5 shows the results of the simulation at 3,000 days, in layer 1 indicating that solutes from layer 2 migrated into layer 1 through the conduit due to head gradients between the aquifers causing the resulting plume in layer 1.
# 
# A simulation was also conducted for this case with use of a nested grid. The region around the conduit was nested with each cell being further subdivided in two along the row and column directions. Note that the nesting is not ideal for this problem, as the plume crosses the nested region in the lateral and longitudinal directions. However, such a setup depicts the code accuracy in evaluating transport across nested regions. Figure Ex 6 shows the nested grid used for this simulation and the simulation results in layer 1 at 3,000 days. The results are very similar to those of Figure Ex 5.
# 

# In[1]:


import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.mfusg import MfUsg, MfUsgBcf, MfUsgBct, MfUsgCln, MfUsgOc, MfUsgSms, MfUsgWel
from flopy.modflow import ModflowBas, ModflowChd, ModflowDis
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import HeadFile

# In[2]:


model_ws = "Ex3_CLN_Conduit"
mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws= model_ws,
    modelname="Ex3_CLN_Conduit",
    exe_name="mfusg_gsi",
)


# In[3]:


nrow = 100
ncol = 100
delc = 470.0
delr = 470.0

nlay = 2
top = -100
botm = [-110.0,-120.0,-130.0]

dis = ModflowDis(mf,nlay,nrow,ncol, delr=delr, delc=delc, laycbd=[1,0], top=top, botm=botm,
                 itmuni=4, lenuni= 0, nper=1, perlen=3000.0, nstp=100)


# In[4]:


ibound = np.ones((nlay, nrow, ncol))
ibound[0, 0, :]  = -1
ibound[0, -1, :] = -1
ibound[1, 0, :]  = -1
ibound[1, -1, :] = -1
ibound[1, :, 0] = -1
ibound[1, :, -1] = -1

strt = np.full((nlay, nrow, ncol), 10.0)
strt[0, -1, :] = 30.0
strt[1, :, :] = 30.0
strt[1, 0, :]  = 60.0
strt[1, -1, :] = 60.0
strt[1, :, 0] = 60.0
strt[1, :, -1] = 60.0

bas = ModflowBas(mf,ibound=ibound,strt=strt)
bas.ibound.fmtin = "(25I3)"
bas.strt.fmtin = "(10e12.4)"


# In[5]:


ipakcb = 50        
bcf = MfUsgBcf(mf,ipakcb = ipakcb, wetfct=1.0, iwetit=5, hy=[100.0,400.0], vcont=0.0)


# In[6]:


##Generate a list of chd cells
lrcsc = [(ilay, irow, icol,  strt[ilay, irow, icol], strt[ilay, irow, icol], 10.0, ibound[ilay, irow, icol]) 
            for ilay in range(nlay) for irow in range(nrow) for icol in range(ncol)]
lrcsc = {0:[item[:5] for item in lrcsc if item[6] == -1]}
chd = ModflowChd(mf,ipakcb = ipakcb, stress_period_data=lrcsc)


# In[7]:


sms = MfUsgSms(mf, 
        hclose=1.0e-3,
        hiclose=1.0e-5,
        mxiter=250,
        iter1=600,
        iprsms=1,
        nonlinmeth=1,
        linmeth=1,
        theta=0.7,
        akappa=0.07,
        gamma=0.1,
        amomentum=0.0,
        numtrack=200,
        btol=1.1,
        breduc=0.2,
        reslim=1.0,
        iacl=1,
        norder=0,
        level=7,
        north=14,
        iredsys=0,
        rrctol=0.0,
        idroptol=1,
        epsrn=1.0e-3,
)


# In[8]:


bct = MfUsgBct(mf, ipakcb = 55, itvd = 0, prsity = 0.01, 
               dl =0.0, dt=0.0, conc=[[0.0,1.0]])


# In[9]:


unitnumber = [71, 35, 36, 0, 0, 37, 0]

node_prop = [
    [1, 1, 0, 10.0, -110.0, 1.57, 0, 0],
    [2, 1, 0, 10.0, -130.0, 1.57, 0, 0],
]
cln_gwc = [
    [1, 1, 50, 50, 0, 0, 10.0, 1.0, 0],
    [2, 2, 50, 50, 0, 0, 10.0, 1.0, 0],
]

nconduityp = 1
cln_circ = [[1, 0.5, 3.23e10]]

strt = [10.0, 30.0]
cln = flopy.mfusg.MfUsgCln(
    mf,
    ncln=1,
    iclnnds=-1,
    nndcln=2,
    nclngwc=2,
    node_prop=node_prop,
    cln_gwc=cln_gwc,
    cln_circ=cln_circ,
    dll = 0.0,
    dlm = 0.0,
    strt=strt,
    conc=[[0.0,1.0]],
    unitnumber=unitnumber,
)


# In[10]:


wel = MfUsgWel(mf, ipakcb=ipakcb, cln_stress_period_data={0:[1,0.0]})


# In[11]:


oc = MfUsgOc(mf, unitnumber= [14,30,31,0,0,33], save_every=1,save_conc=1,compact=False)


# In[12]:


mf.write_input()
success, buff = mf.run_model()


# In[13]:


concobj = HeadFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc = concobj.get_data()


# In[21]:


levels = [0.01,0.50,0.99]

fig = plt.figure(figsize=(4, 8), dpi=150)
for ilay in range(nlay) :
    ax = fig.add_subplot(nlay, 1, ilay + 1)
    ax.set_aspect('equal')
    pmv = PlotMapView(mf, layer=ilay, ax=ax)
    pmv.plot_array(simconc, cmap='jet')
    pmv.contour_array(simconc, levels=levels, colors="w", linewidths=1.0, linestyles="-")


# # With Dispersion

# In[22]:


mf.remove_package("SMS")
sms = MfUsgSms(mf, 
        hclose=1.0e-3,
        hiclose=1.0e-5,
        mxiter=220,
        iter1=600,
        iprsms=1,
        nonlinmeth=2,
        linmeth=1,
        theta=0.9,
        akappa=0.07,
        gamma=0.1,
        amomentum=0.0,
        numtrack=200,
        btol=1.1,
        breduc=0.2,
        reslim=1.0,
        iacl=2,
        norder=1,
        level=3,
        north=14,
        iredsys=0,
        rrctol=0.0,
        idroptol=0,
        epsrn=1.0e-3,
)


# In[23]:


mf.remove_package("BCT")
bct = MfUsgBct(mf, ipakcb = 55, itvd = 0, prsity = 0.01, 
               idsip =2, dlx=5000.0,dly=5000.0,dlz=5000.0,
               dtxy=100.0, dtyz=0.0, dtxz=0.0,
               conc=[[0.0,1.0]])


# In[24]:


mf.write_input()
success, buff = mf.run_model()


# In[25]:


concobj = HeadFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc = concobj.get_data()


# In[26]:


levels = [0.01,0.50,0.99]

fig = plt.figure(figsize=(4, 8), dpi=150)
for ilay in range(nlay) :
    ax = fig.add_subplot(nlay, 1, ilay + 1)
    ax.set_aspect('equal')
    pmv = PlotMapView(mf, layer=ilay, ax=ax)
    pmv.plot_array(simconc, cmap='jet')
    pmv.contour_array(simconc, levels=levels, colors="w", linewidths=1.0, linestyles="-")

