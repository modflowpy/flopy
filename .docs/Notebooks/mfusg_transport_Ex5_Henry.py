#!/usr/bin/env python
# coding: utf-8

# ## Henry Problem for Density-Dependent Flow and Transport

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other Enhancements to MODFLOW-USG, GSI Environmental, July 2024 http://www.gsi-net.com/en/software/free-software/USG-Transport.html
# 
# The Henry Problem depicted by Guo and Langevin (2002) is replicated here to evaluate the ability of the BCT Process with density dependent flow capabilities coded into USG-Transport. A cross-sectional domain 2-m long, by 1-m high, and by 1-m wide is provided a constant flux of fresh ground water along the left boundary at a rate (Qin) of 5.702m3/d per meter with a concentration (Cin) equal to zero. A zero constant head boundary is applied to the right side of the cross-section to represent seawater hydrostatic conditions. The upper and lower model boundaries are no flow. The finite-difference model grid used to discretize the problem domain consists of 1 row with 21 columns and 10 layers. Each cell, with the exception of the cells in column 21, is 0.1 by 0.1 m in size. Cells in column 21 are 0.01-m horizontal by 0.1-m vertical.
# The narrow column of cells in column 21 was used to better locate the seawater hydrostatic boundary at a distance of 2 m. The WEL package was used to assign injection wells, with constant inflow rates of 0.5702 m3/d to each cell of column 1. Constant freshwater heads were assigned to the cells in column 21 using a head value of 1.0 m and a concentration of 35 kg/m3. The concentration for inflow from these constant head cells was specified at 35 kg/m3. An identical problem setup in SEAWAT was also simulated for comparison.

# In[1]:


import os
import shutil
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.mfusg import (
    MfUsg,
    MfUsgBct,
    MfUsgDdf,
    MfUsgDisU,
    MfUsgLpf,
    MfUsgOc,
    MfUsgPcb,
    MfUsgSms,
    MfUsgWel,
)
from flopy.modflow import ModflowBas, ModflowChd, ModflowDis
from flopy.plot import PlotCrossSection
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen

# In[2]:


model_ws = "Ex5_Henry"

# temp_dir = TemporaryDirectory()
# model_ws = temp_dir.name


# In[3]:


mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws= model_ws,
    modelname="Ex5_Henry",
    exe_name="mfusg_gsi",
)


# In[4]:


ms = flopy.modflow.Modflow()

nrow = 1
ncol = 21
delc = 1.0
delr = [0.1] * ncol
delr[ncol-1] = 0.01

nlay = 10
delv = 0.1
top = 1.0
botm = np.linspace(top - delv, 0.0, nlay)

dis = flopy.modflow.ModflowDis(ms,nlay,nrow,ncol, delr=delr, delc=delc, laycbd=0, top=top, botm=botm)


# In[5]:


gridgen_ws = os.path.join(model_ws, 'gridgen')
if not os.path.exists(gridgen_ws):
    os.mkdir(gridgen_ws)    
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
g.build()
gridprops = g.get_gridprops_disu5()


# In[6]:


disu = MfUsgDisU(mf, **gridprops, nper=1, perlen=2.0, nstp=40, tsmult = 1.1 ,steady=False)


# In[7]:


# gridprops_ug = g.get_gridprops_unstructuredgrid()
# ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug)
# mf.modelgrid = ugrid


# In[8]:


bas = ModflowBas(mf)


# In[9]:


ipakcb = 53
hk  = 864.0
vka = 864.0
ss  = 0.0
lpf = MfUsgLpf(mf,ipakcb = ipakcb, hk = hk,vka = vka,ss=ss)


# In[10]:


sms = MfUsgSms(mf, 
        hclose=1.0e-4,
        hiclose=1.0e-6,
        mxiter=250,
        iter1=600,
        iprsms=1,
        nonlinmeth=-1,
        linmeth=1,
        theta=0.7,
        akappa=0.07,
        gamma=0.1,
        amomentum=0.0,
        numtrack=200,
        btol=1.1,
        breduc=0.2,
        reslim=1.0,
        iacl=0,
        norder=0,
        level=1,
        north=14,
        iredsys=0,
        rrctol=0.0,
        idroptol=0,
        epsrn=1.0e-3,
)


# In[11]:


dtype = np.dtype([
    ("node", int),
    ("flux", np.float32),
    ("c01", np.float32)])

welflux = 0.5702
welconc = 0.0
lrcsc = []
for ilay in range(nlay) :
    innode = mf.modelgrid.get_layer_node_range(ilay)[0]
    lrcsc.append([innode,welflux,welconc])

wel = MfUsgWel(mf, ipakcb=ipakcb, options=[], dtype=dtype, stress_period_data={0:lrcsc})


# In[12]:


dtype = np.dtype([
    ("node", int),
    ("shead", np.float32),
    ("ehead", np.float32),
    ("c01", np.float32)])

chdhead = 1.0
chdconc = 35.0
lrcsc = []
for ilay in range(nlay) :
    outnode = mf.modelgrid.get_layer_node_range(ilay)[1] - 1
    lrcsc.append([outnode,chdhead,chdhead,chdconc])

chd = ModflowChd(mf,ipakcb = ipakcb, options=[], dtype=dtype, stress_period_data=lrcsc)


# In[13]:


prsity = 0.35
dl = 0.0
dt = 0.0
conc = 35.0
anglex = 0.0

bct = MfUsgBct(mf, ipakcb = 55, itvd=3,cinact=-999.9, anglex = anglex, prsity=prsity, dl = dl, dt = dt, 
               conc = conc)


# In[14]:


ddf = MfUsgDdf(mf)


# In[15]:


# outconc = 35.0
# lrcsc = []
# for ilay in range(nlay) :
#     outnode = mf.modelgrid.get_layer_node_range(ilay)[1]
#     lrcsc.append([outnode,1,outconc])
# pcb = MfUsgPcb(mf,stress_period_data=lrcsc)


# In[16]:


oc = MfUsgOc(mf, save_conc = 1, save_every=1, save_types = ["save head", "save budget"],unitnumber= [14,30,31,0,0,132])


# In[17]:


mf.write_input()
success, buff = mf.run_model()


# In[18]:


concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text='conc')
simconc = concobj.get_data()


# In[19]:


levels = [0.35,3.5, 8.75,17.5,31.5]

fig = plt.figure(figsize=(8, 5), dpi=150)
ax = fig.add_subplot(111)
pxs = PlotCrossSection(ms, ax=ax, line={"row": 0})
pxs.plot_array(simconc, cmap="jet", vmin=0, vmax=35)
pxs.contour_array(simconc, levels=levels, colors="w", linewidths=1.0, linestyles="-")


# In[ ]:




