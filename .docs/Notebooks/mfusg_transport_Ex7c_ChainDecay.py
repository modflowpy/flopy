#!/usr/bin/env python
# coding: utf-8

# ## MF-USG Example Problems for Matrix Diffusion Transport Package

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other Enhancements to MODFLOW-USG, GSI Environmental, July 2024 http://www.gsi-net.com/en/software/free-software/USG-Transport.html
#
# Several benchmark and verification simulations have been conducted with the MDT Package modules to test accuracy and performance. The code has been tested in 1-, 2-, and 3-dimensions, against analytical solutions as well as against other numerical codes; specifically,MT3D (Zheng and Wang, 1999). The following example problems are provided to demonstrate application of the MDT Process. It is recommended that users familiarize themselves with the different simulation options, code accuracy under various conditions, and input/output structures of the MDT Process via these test problems.

# In[1]:


import os
import shutil
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import flopy
import flopy.utils.binaryfile as bf
from flopy.mfusg import (
    MfUsg,
    MfUsgBct,
    MfUsgDisU,
    MfUsgLpf,
    MfUsgMdt,
    MfUsgOc,
    MfUsgPcb,
    MfUsgSms,
    MfUsgWel,
)
from flopy.modflow import ModflowBas, ModflowChd, ModflowDis
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen

# ## Example MD3: Demonstration of PCE Decay

# A hypothetical site with a tetrachloroethene (PCE) release is used to demonstrate MODFLOW-USG MDT package's ability to model the effects of sequential decay in addition to diffusion into and from low-k zones (Table 5). The transmissive zone is a sandy aquifer that is interbedded with clay lenses. Hydraulic conductivity of the transmissive zone is 12,500 m/yr and the hydraulic gradient across the site is 0.002 m/m. Porosities for the sand and clay are 0.33 and 0.4, respectively. A tortuosity of 0.7 is assumed for both sand and clay. PCE is continuously released into groundwater at a concentration of 100 mg/L. The source area is 10 m perpendicular to groundwater flow and 3 m deep.
#
# Approximately 40% of the transmissive zone comprises of clay lenses with a characteristic diffusion length of 0.5 m. Therefore, matrix diffusion was modeled as being from the clay lenses embedded in the transmissive zone. PCE is assumed to undergo reductive dechlorination to trichloroethene (TCE), then to cis-1,2-dicloroethene (cis-DCE), and finally to vinyl chloride (VC) with decay rates of 0.4 yr-1 (PCE), 0.15 yr-1 (TCE), 0.1 yr-1 (cis-DCE), and 0.2 yr-1 (VC) (Wiedemeier et al., 1999; Aziz et al., 2002). Retardation factors assigned to each of the constituents are shown in Table 5.
#
# The MODFLOW-USG MDT model contains 5 layers, 20 rows, and 100 columns. Input parameters are shown in Table 5. Comparison of the MODFLOW-USG MDT model concentrations with the semi-analytical model results over various years is shown on Figure Ex 14. As shown in the figure, the MODFLOW-USG MDT package was able to reproduce the constituent concentrations reasonably well for all constituents.

# In[2]:


model_ws = "Ex7_PCE"

# temp_dir = TemporaryDirectory()
# model_ws = temp_dir.name


# In[3]:


mf = MfUsg(
    version="mfusg",
    structured=True,
    model_ws=model_ws,
    modelname="Ex7_PCE",
    exe_name="mfusg_gsi",
)


# In[4]:


ms = flopy.modflow.Modflow()

nrow = 20
ncol = 100
delc = 5
delr = 20

nlay = 5
top = 15
delv = 3
botm = np.linspace(top - delv, 0.0, nlay)

dis = flopy.modflow.ModflowDis(
    ms, nlay, nrow, ncol, delr=delr, delc=delc, laycbd=0, top=top, botm=botm
)


# In[5]:


gridgen_ws = os.path.join(model_ws, "gridgen")
if not os.path.exists(gridgen_ws):
    os.mkdir(gridgen_ws)
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
g.build()


# In[6]:


gridprops = g.get_gridprops_disu5()
anglex = g.get_anglex()
nnodes = g.get_nodes()
gridx = g.get_cellxy(nnodes)[:, 0]


# In[7]:


disu = MfUsgDisU(mf, **gridprops, itmuni=5, lenuni=2, nper=1, perlen=50.0)


# In[8]:


# MODFLOW-USG does not have vertices, so we need to create
# and unstructured grid and then assign it to the model. This
# will allow plotting and other features to work properly.
gridprops_ug = g.get_gridprops_unstructuredgrid()
ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug)
mf.modelgrid = ugrid


# In[9]:


bas = ModflowBas(mf, strt=20.0)


# In[10]:


ipakcb = 50
hk = 12500.0
vka = 12500.0
lpf = MfUsgLpf(mf, ipakcb=ipakcb, constantcv=1, novfc=1, hk=hk, vka=vka)


# In[11]:


sms = MfUsgSms(
    mf,
    hclose=1.0e-6,
    hiclose=1.0e-8,
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


# In[12]:


dtype = np.dtype(
    [
        ("node", int),
        ("shead", np.float32),
        ("ehead", np.float32),
        ("C01", np.float32),
        ("C02", np.float32),
        ("C03", np.float32),
        ("C04", np.float32),
    ]
)

inhead = 18.96
outhead = 15.0
lrcsc = []
for ilay in range(nlay):
    istart, istop = mf.modelgrid.get_layer_node_range(ilay)
    for inode in range(istart, istop):
        if inode == 8001:
            lrcsc.append([inode, inhead, inhead, 0.1, 0.0, 0.0, 0.0])
        else:
            xcenters = mf.modelgrid.xcellcenters[inode]
            if xcenters < 20:
                lrcsc.append([inode, inhead, inhead, 0.0, 0.0, 0.0, 0.0])
            if xcenters > 1980:
                lrcsc.append([inode, outhead, outhead, 0.0, 0.0, 0.0, 0.0])

chd = ModflowChd(mf, ipakcb=ipakcb, options=[], dtype=dtype, stress_period_data=lrcsc)


# In[13]:


diffnc = 6.694036e-002
adsorb = [0.19375, 0.1166625, 0.03625, 0.0137]
fodrw = [0.4, 0.15, 0.1, 0.2]
bct = MfUsgBct(
    mf,
    itrnsp=3,
    mcomp=4,
    itvd=0,
    cinact=-999.9,
    diffnc=diffnc,
    prsity=0.33,
    bulkd=1.6,
    anglex=anglex,
    dl=0.0,
    dt=0.0,
    iadsorb=1,
    adsorb=adsorb,
    ifod=3,
    fodrw=fodrw,
    fodrs=0,
    timeweight=1.0,
    chaindecay=True,
    nparent=[0, 1, 1, 1],
    jparent=[0, 1, 2, 3],
    stotio=[0, 0.795, 0.737, 0.640],
)


# In[14]:


lrcsc = {0: [8000, 1, 0.1]}
pcb = MfUsgPcb(mf, stress_period_data=lrcsc)


# In[15]:


mdt = MfUsgMdt(
    mf,
    mdflag=2,
    frahk=True,
    volfracmd=0.4,
    pormd=0.4,
    rhobmd=1.6,
    difflenmd=0.5,
    tortmd=0.7,
    kdmd=0.19375,
    decaymd=0.4,
    yieldmd=0.0,
    diffmd=0.031558,
)


# In[16]:


lrcsc = {
    (0, 0): [
        "DELTAT 0.5",
        "TMAXAT 0.5",
        "TMINAT 0.5",
        "TADJAT 1.0",
        "TCUTAT 2.0",
        "SAVE HEAD",
        "SAVE BUDGET",
        "SAVE CONC",
    ]
}

oc = MfUsgOc(
    mf,
    atsa=1,
    npsteps=1,
    unitnumber=[14, 30, 31, 0, 0, 132],
    stress_period_data=lrcsc,
    compact=False,
)


# In[17]:


mf.write_input()
success, buff = mf.run_model()


# In[18]:


ilay = 0
comps = ["PCE", "TCE", "cis-DCE", "VC"]
fig = plt.figure(figsize=(8, 8), dpi=150)
for idx, comp in enumerate(comps):
    ax = fig.add_subplot(2, 2, idx + 1)
    concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text=f"conc0{idx+1}")
    simconc = concobj.get_data()
    pmv = PlotMapView(mf, layer=ilay, ax=ax)
    pmv.plot_array(simconc, cmap="jet")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{comp} concentration")


# In[19]:


irow = 0
comps = ["PCE", "TCE", "cis-DCE", "VC"]
fig = plt.figure(figsize=(8, 8), dpi=150)
for idx, comp in enumerate(comps):
    ax = fig.add_subplot(2, 2, idx + 1)
    concobj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text=f"conc0{idx+1}")
    simconc = concobj.get_data()
    pxs = PlotCrossSection(ms, line={"row": irow}, ax=ax)
    pxs.plot_array(simconc, cmap="jet")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"{comp} concentration")


# In[20]:


conc1obj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text="conc01")
conc2obj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text="conc02")
conc3obj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text="conc03")
conc4obj = HeadUFile(f"{mf.model_ws}/{mf.name}.con", text="conc04")

plottime = [1.0, 5.0, 25.0, 50.0]
xdist = mf.modelgrid.xcellcenters[0:99]

fig = plt.figure(figsize=(8, 8), dpi=150)
ilay = 3
for idx, t in enumerate(plottime):
    ax = fig.add_subplot(2, 2, idx + 1)
    simconc1 = conc1obj.get_data(totim=t)[ilay][0:99] * 1000
    simconc2 = conc2obj.get_data(totim=t)[ilay][0:99] * 1000
    simconc3 = conc3obj.get_data(totim=t)[ilay][0:99] * 1000
    simconc4 = conc4obj.get_data(totim=t)[ilay][0:99] * 1000
    ax.plot(xdist, simconc1, label="USG MDT PCE")
    ax.plot(xdist, simconc2, label="USG MDT TCE")
    ax.plot(xdist, simconc3, label="USG MDT cis-DCE")
    ax.plot(xdist, simconc4, label="USG MDT VC")
    plt.yscale("log")
    ax.set_ylabel("Concentration(mg/L)")
    ax.set_ylim(0.001, 100)
    ax.set_title(f"{t} Years After Source Release")
    ax.legend()


# In[ ]:
