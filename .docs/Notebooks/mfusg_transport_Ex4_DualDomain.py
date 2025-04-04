#!/usr/bin/env python
# coding: utf-8

# ## Dual Domain Transport in a One-Dimensional, Uniform Flow Field

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other 
# Enhancements to MODFLOW-USG, GSI Environmental, July 2024 
# http://www.gsi-net.com/en/software/free-software/USG-Transport.html
# 
# This test problem discusses one dimensional dual domain transport in 
# a uniform steady-state flow-field. A 150 foot long horizontal soil 
# column is discretized into 1 layer, 1 row, and 300 columns using 
# dx=0.5feet, dy=1foot, and dz=1foot. The flow-field is setup using 
# a hydraulic conductivity of 1000 ft/day and constant head boundaries 
# of 10 feet and 9 feet at either end of the domain. The simulation 
# considers a dual porosity system with a mobile domain fraction of 0.4. 
# Transport related parameters for the mobile domain include a 
# longitudinal dispersivity of 0.5 feet, zero molecular diffusion, 
# a porosity value of 0.35, a soil bulk density value of 1.6 kg/L, 
# and an adsorption coefficient (kd) value of 0.1 L/kg. Transport 
# parameters for the immobile domain include a porosity value of 0.2, 
# a soil bulk density value of 1.6 kg/L, an adsorption coefficient (kd) 
# value of 0.1 L/kg, and a mass transfer rate of 0.1 day-1. The 
# concentration of water in both mobile and immobile domains is zero at 
# the start of the simulation.
# 
# A transport simulation was performed for this setup with a prescribed 
# species concentration of 1mg/L at the upstream end of the soil column 
# within the mobile domain, for a period of 20 days. Subsequently, the 
# concentration of inflow water was made to zero for a period of 30 days 
# to evaluate flushing of the system. Each stress period contains 100 time 
# steps of uniform size 0.04 day step size for the first stress period 
# when the component species front is advancing, and a 0.08 day step size 
# for the second stress period when the soil column is being flushed. 
# Simulation results were compared with results from a MT3D simulation of 
# the same setup, using the TVD solution scheme. Note that the mobile 
# porosity in MT3D is equal to the porosity of the mobile domain (0.35) 
# times the mobile domain fraction (0.4), and that the immobile porosity in
# MT3D is equal to the porosity of the immobile domain (0.2) times the 
# immobile domain fraction (which is one minus the mobile domain fraction 0.6). 
# Figure Ex 7 shows the concentration versus time plot in the mobile domain, 
# at the outlet of the domain. The MT3D and BCT Process simulation results 
# are almost the same.

# In[1]:


from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.mfusg import MfUsg, MfUsgBcf, MfUsgBct, MfUsgDpt, MfUsgOc, MfUsgPcb, MfUsgSms
from flopy.modflow import ModflowBas, ModflowChd, ModflowDis
from flopy.utils import HeadFile

# In[2]:


model_ws = "Ex4_DualDomain"

# temp_dir = TemporaryDirectory()
# model_ws = temp_dir.name


# In[3]:


mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws= model_ws,
    modelname="Ex4_DualDomain",
    exe_name="mfusg_gsi",
)


# In[4]:


nlay = 1 
nrow = 1
ncol = 300
delr = 0.5
delc = 1.0
top  = 1.0
botm = 0

nper=2
perlen = [20.0, 30.0]
nstp   = [100, 100]
tsmult = [1.5, 1.5]
lenuni =0

xcol = [i * delc for i in range(ncol)]

dis = ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm, 
                 nper=nper, perlen=perlen, nstp=nstp, tsmult=tsmult, lenuni=lenuni)


# In[5]:


ibound = np.ones((nlay, nrow, ncol))
ibound[:, :, 0]  = -1
ibound[:, :, -1] = -1

strt = np.full((nlay, nrow, ncol), 9.7)
strt[:, :, 0]  = 10.0
strt[:, :, -1] = 9.0

bas = ModflowBas(mf,ibound=ibound, strt=strt)


# In[6]:


tran   = 1000.0
bcf = MfUsgBcf(mf,ipakcb = 50, laycon=0, hdry=-999.9, wetfct=0.0, iwetit=0, tran=tran)


# In[7]:


sms = MfUsgSms(mf, options ="MODERATE",     
        hclose=1.0e-3,
        hiclose=1.0e-5,
        mxiter=250,
        iter1=600,
        iprsms=1,
        nonlinmeth=-1,
        linmeth=1,
        theta=0.9,
        akappa=0.0001,
        gamma=0.0,
        amomentum=0.0,
        numtrack=0,
        btol=0.0,
        breduc=0.0,
        reslim=0.0,
        iacl=2,
        norder=0,
        level=3,
        north=5,
        iredsys=1,
        rrctol=0.0,
        idroptol=1,
        epsrn=1.0e-3,
)


# In[8]:


lrcsc = {0:[[0,0,0,10.,10.],[0,0,299,9.,9.]]}

chd = ModflowChd(mf,ipakcb = 50,stress_period_data=lrcsc)


# In[9]:


prsity = 0.35
bulkd = 1.60
dl = 0.5
dt = 0.0
adsorb = 0.1
conc = 0
bct = MfUsgBct(mf, ipakcb = 55, itvd=20,cinact=-999.0, diffnc=0.0, prsity=prsity, dl = dl, dt = dt, 
               iadsorb=1,bulkd=bulkd, adsorb =adsorb, conc = conc)


# In[10]:


lrcsc = {0:[0,0,0,1,1.0],1:[0,0,0,1,0.0]}
pcb = MfUsgPcb(mf,stress_period_data=lrcsc)


# In[11]:


dpt = MfUsgDpt(mf,ipakcb=55, idptcon=0, icbndimflg=0,iadsorbim=1,icbundim=1,prsityim=0.2,bulkdim=1.6,ddtr=0.1, adsorbim=0.1)


# In[12]:


lrcsc = {(0,0): ["DELTAT 4.0E-2", "TMAXAT 2.0E00", "TMINAT 1.0E-4", "TADJAT 1.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"], 
     (1,0) : ["DELTAT 8.0E-2", "TADJAT 1.0", "SAVE HEAD", "SAVE BUDGET", "SAVE CONC"]}

oc = MfUsgOc(mf, atsa=1, npsteps=1, unitnumber= [14,30,31,0,0,35], stress_period_data = lrcsc)


# In[13]:


mf.write_input()
success, buff = mf.run_model()


# In[14]:


concobj = HeadFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc = concobj.get_ts((0,0,299))


# In[15]:


fig = plt.figure(figsize=(8, 5), dpi=150)
ax = fig.add_subplot(111)
ax.plot(simconc[:,0], simconc[:,1], label="BCT Result")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Normalized concentration")
ax.set_title("MODFLOW-USG Transport Simulation Results for Dual Domain Transport in a One-Dimensional, Uniform Flow Field")
ax.legend()


# In[ ]:




