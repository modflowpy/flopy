# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   metadata:
#     section: mfusg
#     authors:
#       - name: Hua Zhang
# ---

# ## MF-USG Example Problems for Solute Transport in LAKE Package

# Panday, S., 2024; USG-Transport Version 2.4.0: Transport and Other Enhancements to MODFLOW-USG, GSI Environmental, July 2024 http://www.gsi-net.com/en/software/free-software/USG-Transport.html
#
# A verification example is presented here. The example is based on the verification example provided in Merritt and Konikow (2000) and Bedekar et al (2016). The simulation considers a lake in a two-layer groundwater model simulated for a 5,000-day period. Initial conditions in the model are such that the lake stage is 50 ft below the fixed groundwater heads on the left side of the model boundary. The initial concentration of the lake is set to 100 mg/L. As the simulation begins, groundwater with zero concentration enters the lake causing dilution in the lake. As the lake fills over time lake stage and volume become stable and the lake stage rises sufficiently above the right boundary of the groundwater model to cause seepage from the lake into the groundwater system. Continued precipitation over the lake and lake seepage into ground causes dilution to continue at a low rate.
# Change in lake stage and volume over time is shown in Figure Ex 15. Lake stage and volume calculated with MODFLOW-USG are compared with a MODFLOW simulation to verify the flow solution of the LAK package of MODFLOW-USG. A good agreement is observed between MODFLOW-USG and MODFLOW. MT3D-USGS is used with the MODFLOW model to simulate solute transport in lakes. Dilution in lake concentration is shown in Figure 23. Results from MODFLOW-USG lake transport are compared to results obtained using MT3D-USGS. Manual calculations were also performed to verify the correctness of the solution. Lake concentration calculated by MODFLOW-USG show a good match with manual calculations and MT3D-USGS results.

# Bedekar, Vivek, Morway, E.D., Langevin, C.D., and Tonkin, Matt, 2016, MT3D-USGS version 1: A U.S. Geological Survey release of MT3DMS updated with new and expanded transport capabilities for use with MODFLOW: U.S. Geological Survey Techniques and Methods 6-A53, 69 p., http://dx.doi.org/10.3133/tm6A53.
#
# In this simple benchmark problem, initial groundwater concentrations are set equal to zero. Upon execution of the simulation, groundwater begins flowing into the lake as a result of a lake stage that is 50 ft below fixed groundwater heads on the left edge of the model boundary. The initial constituent concentration in the lake is 100 mg/L. Thus, the discharge of “clean” groundwater to the lake dilutes the lake concentrations for the remainder of the simulation. Streams entering or exiting the lake are not simulated. Precipitation and evaporation rates of 0.0115 ft/d and 0.0103 ft/d, respectively, remain constant throughout the simulation and have associated concentrations equal to zero. Thus, the analytical solution for this problem is easy to calculate and is shown as the black line in figure 33. The simulated LKT concentrations, depicted as green color-filled circles in figure 33, demonstrate the accuracy of MT3D-USGS.
#
# As the lake fills, seepage from the lake to the surficial aquifer begins to occur at approximately 1,230 days into the simulation, when the lake stage rises sufficiently above the fixed head boundary along the right-hand edge of the model domain. Furthermore, after 3,000 simulation days, the amount of precipitation falling on the lake, which has a significantly expanded surface area by this point in the simulation, plus groundwater inflow to the lake is roughly balanced by the combination of evaporation and seepage losses to the surficial aquifer below the lake. Hence, even after the lake stage levels off, the lake constituent concentration continues to drop as solute is continually lost to the groundwater system through seepage occurring in parts of the lakebed, and is diluted by precipitation and by groundwater inflow occurring in other parts of the lakebed with zero (or very low) concentrations. The simulation maintains a good mass balance and verifies lake transport related calculations in the absence of lake-stream interaction.
#

# +
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import flopy
import flopy.utils.binaryfile as bf
from flopy.mfusg import (
    MfUsg,
    MfUsgBas,
    MfUsgBcf,
    MfUsgBct,
    MfUsgDisU,
    MfUsgEvt,
    MfUsgLak,
    MfUsgOc,
    MfUsgPcb,
    MfUsgRch,
    MfUsgSms,
)
from flopy.modflow import ModflowChd, ModflowDis, ModflowFhb, ModflowGage
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import HeadUFile
from flopy.utils.gridgen import Gridgen

# -

# +
model_ws = "Ex8_Lake"

mf = MfUsg(
    version="mfusg",
    structured=False,
    model_ws=model_ws,
    modelname="Ex8_Lake",
    exe_name="mfusg",
)
# -

# +
ms = flopy.modflow.Modflow()

nrow = 17
ncol = 17
delc = [
    250.0,
    1000.0,
    1000.0,
    1000.0,
    1000.0,
    1000.0,
    500.0,
    500.0,
    500.0,
    500.0,
    500.0,
    1000.0,
    1000.0,
    1000.0,
    1000.0,
    1000.0,
    250.0,
]
delr = delc

nlay = 5
top = 500.0
botm = [107.0, 97.0, 87.0, 77.0, 67.0]

dis = flopy.modflow.ModflowDis(
    ms, nlay, nrow, ncol, delr=delr, delc=delc, laycbd=0, top=top, botm=botm
)
# -

# +
gridgen_ws = os.path.join(model_ws, "gridgen")
if not os.path.exists(gridgen_ws):
    os.mkdir(gridgen_ws)
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
g.build()
# -

# +
disu = g.get_disu(
    mf, itmuni=4, lenuni=0, nper=1, perlen=5000.0, nstp=100, tsmult=1.02, steady=False
)
disu.ivsd = -1
anglex = g.get_anglex()
# -

# +
# MODFLOW-USG does not have vertices, so we need to create
# and unstructured grid and then assign it to the model. This
# will allow plotting and other features to work properly.
gridprops_ug = g.get_gridprops_unstructuredgrid()
ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug)
mf.modelgrid = ugrid
# -

# +

# A 17-row by 17-column by 5-layer grid is used in the Lake Transport benchmark problem in A, plan and B, profile views. This problem first appeared in Merritt and Konikow (2000).

pmv = PlotMapView(mf)
pmv.plot_grid()
# -

# +
pxs = PlotCrossSection(ms, line={"row": 8})
pxs.plot_grid()
# -

# +
ibound = []
for ilay in range(nlay):
    laybnd = np.ones((nrow, ncol))
    laybnd[0, :] = -1  # CHD cells
    laybnd[-1, :] = -1  # CHD cells
    laybnd[:, 0] = -1  # CHD cells
    laybnd[:, -1] = -1  # CHD cells
    if ilay == 0:  # Lake cells
        laybnd[6:11, 6:11] = 0
    if ilay == 1:  # Lake cells
        laybnd[7:10, 7:10] = 0
    laybnd = laybnd.reshape(-1)
    ibound.append(laybnd)
bas = MfUsgBas(mf, ibound=ibound, strt=115.0)
bas.ibound.fmtin = "(17I4)"
# -

# +
ipakcb = 50
wetdry = []
for ilay in range(nlay):
    layv = np.ones((nrow, ncol))
    layv[0, :] = 0.0  # CHD cells
    layv[-1, :] = 0.0  # CHD cells
    layv[:, 0] = 0.0  # CHD cells
    layv[:, -1] = 0.0  # CHD cells
    if ilay == 0:  # Lake cells
        layv[6:11, 6:11] = 0.0
    if ilay == 1:  # Lake cells
        layv[7:10, 7:10] = 0.0
    layv = layv.reshape(-1)
    wetdry.append(layv)

bcf = MfUsgBcf(
    mf,
    ipakcb=ipakcb,
    laycon=[1, 3, 3, 3, 3],
    sf1=[0.2, 3e-4, 3e-4, 3e-4, 3e-4],
    sf2=0.2,
    hy=30.0,
    vcont=3.0,
    iwdflg=1,
    wetfct=1.0,
    wetdry=wetdry,
)
bcf.wetdry.fmtin = "(17F5.1)"
# -

# +
sms = MfUsgSms(
    mf,
    hclose=1.0e-6,
    hiclose=1.0e-5,
    mxiter=500,
    iter1=600,
    iprsms=3,
    nonlinmeth=0,
    linmeth=2,
    ipc=2,
    iscl=0,
    iord=0,
    rclosepcgu=1.0e-5,
)
# -

# +
diffnc = 6.694036e-002
adsorb = [0.19375, 0.1166625, 0.03625, 0.0137]
fodrw = [0.4, 0.15, 0.1, 0.2]
bct = MfUsgBct(
    mf, itvd=0, cinact=-999.9, diffnc=0.0, prsity=0.2, anglex=0, dl=0.0, dt=0.0
)
# -

# +
rch = MfUsgRch(mf, ipakcb=ipakcb, iconc=1, rech=0.0116, rchconc=0.0)
# -

# +

elev = np.round(
    [
        160.0,
        159.0,
        157.4,
        155.9,
        154.3,
        152.7,
        151.6,
        150.8,
        150.0,
        149.2,
        148.4,
        147.3,
        145.7,
        144.1,
        142.6,
        141.0,
        140.0,
    ],
    1,
)
etsurf = np.tile(elev, (nrow, 1))
etsurf[6:11, 6:11] = 105.0
etsurf[7:10, 7:10] = 95.0
etsurf = etsurf.reshape(-1)
evt = MfUsgEvt(mf, ipakcb=ipakcb, surf=etsurf, evtr=0.0141, exdp=15.0)
evt.surf.fmtin = "(17F6.1)"
# -

# +
fhbhead = np.tile(elev, (nrow, nlay)).reshape(-1)
lrcsc = []
inhead = 115.0
for ilay in range(nlay):
    istart, istop = mf.modelgrid.get_layer_node_range(ilay)
    for inode in range(istart, istop):
        xcenters = mf.modelgrid.xcellcenters[inode]
        ycenters = mf.modelgrid.ycellcenters[inode]
        if xcenters < 200 or ycenters < 200 or xcenters > 12800 or ycenters > 12800:
            lrcsc.append([inode, 0, fhbhead[inode]])

fhb = ModflowFhb(mf, ipakcb=ipakcb, nhed=len(lrcsc), ds7=lrcsc)
# -

# +
# lrcsc = []
# inhead = 115.0
# for ilay in range(nlay) :
#     istart, istop = mf.modelgrid.get_layer_node_range(ilay)
#     for inode in range(istart, istop) :
#         xcenters = mf.modelgrid.xcellcenters[inode]
#         ycenters = mf.modelgrid.ycellcenters[inode]
#         if xcenters <200 or ycenters <200 or xcenters>12800 or ycenters>12800:
#             lrcsc.append([inode,inhead,inhead])

# chd = ModflowChd(mf,ipakcb = ipakcb, stress_period_data=lrcsc)

lrcsc = []
inhead = 115.0
for ilay in range(nlay):
    istart, istop = mf.modelgrid.get_layer_node_range(ilay)
    for inode in range(istart, istop):
        xcenters = mf.modelgrid.xcellcenters[inode]
        ycenters = mf.modelgrid.ycellcenters[inode]
        if xcenters < 200 or ycenters < 200 or xcenters > 12800 or ycenters > 12800:
            lrcsc.append([inode, 1, 0.0])
pcb = MfUsgPcb(mf, stress_period_data=lrcsc)

# -

# +
lkarr1 = np.zeros((nrow, ncol))
lkarr1[6:11, 6:11] = 1
lkarr2 = np.zeros((nrow, ncol))
lkarr2[7:10, 7:10] = 1
lkarr = {0: [lkarr1.reshape(-1), lkarr2.reshape(-1), 0, 0, 0]}

flux_data = {0: [[0.0116, 0.0103, 0.0, 0.0]]}

conc_data = {0: {(0, 0): [0.0, 0.0]}}

lak = MfUsgLak(
    mf,
    ipakcb=ipakcb,
    theta=0,
    stages=110.0,
    clake=100.0,
    lakarr=lkarr,
    bdlknc=0.1,
    flux_data=flux_data,
    conc_data=conc_data,
)
lak.lakarr[0].fmtin = "(17I3)"
# -

# +
gages = [[-1, -37, 3]]
files = [f"{mf.name}.got"]
gage = ModflowGage(mf, numgage=1, gage_data=gages, files=files)

# -

# +
oc = MfUsgOc(
    mf, unitnumber=[14, 31, 32, 0, 0, 55], save_every=1, save_conc=1, compact=False
)

# -

# +
mf.write_input()
success, buff = mf.run_model()
# -

# +
got = np.genfromtxt(f"{mf.model_ws}/{mf.name}.got", skip_header=2)
# -

# +

time = [
    0.00e00,
    1.60e01,
    3.23e01,
    4.90e01,
    6.60e01,
    8.33e01,
    1.01e02,
    1.19e02,
    1.37e02,
    1.56e02,
    1.75e02,
    1.95e02,
    2.15e02,
    2.35e02,
    2.56e02,
    2.77e02,
    2.98e02,
    3.20e02,
    3.43e02,
    3.66e02,
    3.89e02,
    4.13e02,
    4.37e02,
    4.62e02,
    4.87e02,
    5.13e02,
    5.39e02,
    5.66e02,
    5.93e02,
    6.21e02,
    6.50e02,
    6.79e02,
    7.08e02,
    7.38e02,
    7.69e02,
    8.01e02,
    8.33e02,
    8.65e02,
    8.99e02,
    9.33e02,
    9.67e02,
    1.00e03,
    1.04e03,
    1.08e03,
    1.11e03,
    1.15e03,
    1.19e03,
    1.23e03,
    1.27e03,
    1.31e03,
    1.35e03,
    1.40e03,
    1.44e03,
    1.49e03,
    1.53e03,
    1.58e03,
    1.63e03,
    1.67e03,
    1.72e03,
    1.77e03,
    1.83e03,
    1.88e03,
    1.93e03,
    1.99e03,
    2.04e03,
    2.10e03,
    2.16e03,
    2.22e03,
    2.28e03,
    2.34e03,
    2.40e03,
    2.47e03,
    2.53e03,
    2.60e03,
    2.67e03,
    2.74e03,
    2.81e03,
    2.88e03,
    2.95e03,
    3.03e03,
    3.10e03,
    3.18e03,
    3.26e03,
    3.34e03,
    3.42e03,
    3.51e03,
    3.60e03,
    3.68e03,
    3.77e03,
    3.86e03,
    3.96e03,
    4.05e03,
    4.15e03,
    4.25e03,
    4.35e03,
    4.45e03,
    4.56e03,
    4.67e03,
    4.77e03,
    4.89e03,
    5.00e03,
]
manual_calculation = [
    100.000,
    96.6139,
    93.3794,
    90.1930,
    87.0376,
    83.9044,
    80.7895,
    77.6918,
    74.6126,
    71.5545,
    68.5221,
    65.5215,
    62.5610,
    59.6505,
    56.8017,
    54.0276,
    51.3421,
    48.7593,
    46.2923,
    43.9523,
    41.7488,
    39.6872,
    37.7690,
    35.9925,
    34.3533,
    32.8457,
    31.4636,
    30.2008,
    29.0501,
    28.0026,
    27.0492,
    26.1808,
    25.3893,
    24.6672,
    24.0078,
    23.4050,
    22.8535,
    22.3485,
    21.8857,
    21.4614,
    21.0721,
    20.7148,
    20.3868,
    20.0856,
    19.8090,
    19.5550,
    19.3218,
    19.1075,
    18.9085,
    18.7227,
    18.5489,
    18.3852,
    18.2304,
    18.0836,
    17.9438,
    17.8102,
    17.6823,
    17.5596,
    17.4411,
    17.3260,
    17.2140,
    17.1047,
    16.9978,
    16.8929,
    16.7899,
    16.6884,
    16.5882,
    16.4888,
    16.3900,
    16.2914,
    16.1930,
    16.0946,
    15.9961,
    15.8974,
    15.7982,
    15.6986,
    15.5984,
    15.4976,
    15.3960,
    15.2936,
    15.1904,
    15.0862,
    14.9810,
    14.8747,
    14.7674,
    14.6589,
    14.5493,
    14.4386,
    14.3266,
    14.2134,
    14.0990,
    13.9833,
    13.8663,
    13.7481,
    13.6287,
    13.5079,
    13.3859,
    13.2626,
    13.1381,
    13.0123,
    12.8852,
]

# -

# +
fig = plt.figure(figsize=(8, 5), dpi=150)
ax = fig.add_subplot(111)
ax.plot(got[:, 0], got[:, 3])
ax.plot(time, manual_calculation, linestyle="--")

ax2 = ax.twinx()
ax2.plot(got[:, 0], got[:, 2], color="b")
ax2.set_ylabel("Lake Volumn(ft3)", color="b")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Lake concentration")
