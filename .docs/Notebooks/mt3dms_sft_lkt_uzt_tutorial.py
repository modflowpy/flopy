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
#     section: mt3dms
#     authors:
#       - name: Eric Morway
# ---

# # MT3D-USGS: Transport with the SFR/LAK/UZF Packages (SFT/LKT/UZT), and chemical reactions (RCT)
# A more comprehensive demonstration of setting up an MT3D-USGS model that uses all of the new packages included in the first release of MT3D-USGS.  Also includes RCT.
#
# #### Problem Description:
# * 300 row x 300 col x 3 layer x 2 stress period model
# * Flow model uses SFR, LAK, and UZF with connections between all three
# * Transport model simulates streamflow transport (SFT), with connection to a single lake (LKT)
# * Transport model simulates overland runoff and spring discharge (UZT) to surface water network
#
# Start by importing some libraries:

# +
import os
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# Create a MODFLOW model and store it, in this case in the variable 'mf'.
# The modelname will be the name given to all MODFLOW files.
# The exe_name should be the name of the MODFLOW executable.
# In this case, we want to use version: 'mfnwt' for MODFLOW-NWT

# +
# temporary directory
temp_dir = TemporaryDirectory()
model_ws = temp_dir.name

modelpth = os.path.join(model_ws, "no3")
modelname = "no3"
mfexe = "mfnwt"
mtexe = "mt3dusgs"

# Make sure modelpth directory exists
if not os.path.isdir(modelpth):
    os.makedirs(modelpth, exist_ok=True)

# Instantiate MODFLOW object in flopy
mf = flopy.modflow.Modflow(
    modelname=modelname, exe_name=mfexe, model_ws=modelpth, version="mfnwt"
)
# -

# ### Set up model discretization

# +
Lx = 90000.0
Ly = 90000.0
nrow = 300
ncol = 300
nlay = 3

delr = Lx / ncol
delc = Ly / nrow

xmax = ncol * delr
ymax = nrow * delc

X, Y = np.meshgrid(
    np.linspace(delr / 2, xmax - delr / 2, ncol),
    np.linspace(ymax - delc / 2, 0 + delc / 2, nrow),
)
# -

# ### Instantiate output control (oc) package for MODFLOW-NWT

oc = flopy.modflow.ModflowOc(mf)

# ### Instantiate solver package for MODFLOW-NWT

# +
# Newton-Raphson Solver: Create a flopy nwt package object

headtol = 1.0e-4
fluxtol = 5
maxiterout = 5000
thickfact = 1e-06
linmeth = 2
iprnwt = 1
ibotav = 1

nwt = flopy.modflow.ModflowNwt(
    mf,
    headtol=headtol,
    fluxtol=fluxtol,
    maxiterout=maxiterout,
    thickfact=thickfact,
    linmeth=linmeth,
    iprnwt=iprnwt,
    ibotav=ibotav,
    options="SIMPLE",
)
# -

# ### Instantiate discretization (DIS) package for MODFLOW-NWT

# +
elv_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "dis_arrays",
    "grnd_elv.txt",
)

# Top of Layer 1 elevation determined using GW Vistas and stored locally
grndElv = np.loadtxt(elv_pth)

# Bottom of layer 1 elevation also determined from use of GUI and stored locally
bt1_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "dis_arrays",
    "bot1.txt",
)
bot1Elv = np.loadtxt(bt1_pth)

bot2Elv = np.ones(bot1Elv.shape) * 100
bot3Elv = np.zeros(bot2Elv.shape)

botm = [bot1Elv, bot2Elv, bot3Elv]
botm = np.array(botm)
Steady = [False, False]
nstp = [1, 1]
tsmult = [1.0, 1.0]

# Stress periods
perlen = [9131.25, 9131.25]

# Create the discretization object
# itmuni = 4 (days); lenuni = 1 (feet)
dis = flopy.modflow.ModflowDis(
    mf,
    nlay,
    nrow,
    ncol,
    nper=2,
    delr=delr,
    delc=delc,
    top=grndElv,
    botm=botm,
    laycbd=0,
    itmuni=4,
    lenuni=1,
    steady=Steady,
    nstp=nstp,
    tsmult=tsmult,
    perlen=perlen,
)
# -

# ### Instantiate upstream weighting (UPW) flow package for MODFLOW-NWT
#

# +
# UPW must be instantiated after DIS.  Otherwise, during the mf.write_input() procedures,
# flopy will crash.

# First line of UPW input is: IUPWCB HDRY NPUPW IPHDRY
hdry = -1.00e30
iphdry = 0

# Next variables are: LAYTYP, LAYAVG, CHANI, LAYVKA, LAYWET
laytyp = [1, 3, 3]  # >0: convertible
layavg = 0  #  0: harmonic mean
chani = 1.0  # >0: CHANI is the horizontal anisotropy for the entire layer
layvka = 0  # =0: indicates VKA is vertical hydraulic conductivity
laywet = 0  # Always set equal to zero in UPW package
hk = 20
# hani = 1          # Not needed because CHANI > 1
vka = 0.5  # Is equal to vert. K b/c LAYVKA = 0
ss = 0.00001
sy = 0.20

upw = flopy.modflow.ModflowUpw(
    mf,
    laytyp=laytyp,
    layavg=layavg,
    chani=chani,
    layvka=layvka,
    laywet=laywet,
    ipakcb=53,
    hdry=hdry,
    iphdry=iphdry,
    hk=hk,
    vka=vka,
    ss=ss,
    sy=sy,
)
# -

# ### Instantiate basic (BAS or BA6) package for MODFLOW-NWT

# +
ibnd1_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "bas_arrays",
    "ibnd_lay1.txt",
)
ibnd1 = np.loadtxt(ibnd1_pth)
ibnd2 = np.ones(ibnd1.shape)
ibnd3 = np.ones(ibnd2.shape)

ibnd = [ibnd1, ibnd2, ibnd3]
ibnd = np.array(ibnd)

StHd1_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "bas_arrays",
    "strthd1.txt",
)
StHd1 = np.loadtxt(StHd1_pth)

StHd2_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "bas_arrays",
    "strthd2.txt",
)
StHd2 = np.loadtxt(StHd2_pth)

StHd3_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "bas_arrays",
    "strthd3.txt",
)
StHd3 = np.loadtxt(StHd3_pth)

strtElev = [StHd1, StHd2, StHd3]
strtElev = np.array(strtElev)

hdry = 999.0

bas = flopy.modflow.ModflowBas(mf, ibound=ibnd, hnoflo=hdry, strt=strtElev)
# -

# ### Instantiate general head boundary (GHB) package for MODFLOW-NWT

# +
# GHB boundaries are located along the top (north) and bottom (south)
# edges of the domain, all 3 layers.

elev_stpt_row1 = 308.82281
elev_stpt_row300 = 239.13811
elev_slp = (308.82281 - 298.83649) / (ncol - 1)

sp = []
for k in [0, 1, 2]:  # These indices need to be adjusted for 0-based moronicism
    for i in [
        0,
        299,
    ]:  # These indices need to be adjusted for 0-based silliness
        for j in np.arange(
            0, 300, 1
        ):  # These indices need to be adjusted for 0-based foolishness
            # Skipping cells not satisfying the conditions below
            if (i == 1 and (j < 27 or j > 31)) or (
                i == 299 and (j < 26 or j > 31)
            ):
                if i % 2 == 0:
                    sp.append(
                        [
                            k,
                            i,
                            j,
                            elev_stpt_row1 - (elev_slp * (j - 1)),
                            11.3636,
                        ]
                    )
                else:
                    sp.append(
                        [
                            k,
                            i,
                            j,
                            elev_stpt_row300 - (elev_slp * (j - 1)),
                            11.3636,
                        ]
                    )


for k in [0, 1, 2]:
    for j in np.arange(26, 32, 1):
        sp.append([k, 299, j, 238.20, 3409.0801])

ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=sp)
# -

# ### Instantiate streamflow routing (SFR2) package for MODFLOW-NWT

# +
# Read pre-prepared reach data into numpy recarrays using numpy.genfromtxt()
# Remember that the cell indices stored in the pre-prepared NO3_ReachInput.csv file are based on 0-based indexing.
# Flopy will convert to 1-based when it writes the files

rpth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "sfr_data",
    "no3_reachinput.csv",
)
reach_data = np.genfromtxt(rpth, delimiter=",", names=True)
reach_data

# Read pre-prepared segment data into numpy recarrays using numpy.genfromtxt()

spth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "sfr_data",
    "no3_segmentdata.csv",
)
ss_segment_data = np.genfromtxt(spth, delimiter=",", names=True)
segment_data = {0: ss_segment_data, 1: ss_segment_data}
segment_data[0][0:1]["width1"]

nstrm = len(reach_data)
nss = len(segment_data[0])
nsfrpar = 0
const = 128390.4  # constant for manning's equation, units of cfs
dleak = 0.0001
ipakcb = 53  # flag for writing SFR output to cell-by-cell budget (on unit 53)
istcb2 = 37  # flag for writing SFR output to text file
isfropt = 1
dataset_5 = {
    0: [nss, 0, 0],
    1: [-1, 0, 0],
}  # dataset 5 (see online guide) (ITMP, IRDFLG, IPTFLG)

# Input arguments generally follow the variable names defined in the Online Guide to MODFLOW
sfr = flopy.modflow.ModflowSfr2(
    mf,
    nstrm=nstrm,
    nss=nss,
    const=const,
    dleak=dleak,
    ipakcb=ipakcb,
    istcb2=istcb2,
    isfropt=isfropt,
    reachinput=True,
    reach_data=reach_data,
    segment_data=segment_data,
    dataset_5=dataset_5,
    unit_number=15,
)
# -

# ### Instantiate Lake (LAK) package for MODFLOW-NWT

# +
# Read pre-prepared lake arrays
LakArr_pth = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "lak_arrays",
    "lakarr1.txt",
)
LakArr_lyr1 = np.loadtxt(LakArr_pth)
LakArr_lyr2 = np.zeros(LakArr_lyr1.shape)
LakArr_lyr3 = np.zeros(LakArr_lyr2.shape)

LakArr = [LakArr_lyr1, LakArr_lyr2, LakArr_lyr3]
LakArr = np.array(LakArr)

nlakes = int(np.max(LakArr))
ipakcb = ipakcb  # From above
theta = -1.0  # Implicit
nssitr = 10  # Maximum number of iterations for Newton’s method
sscncr = 1.000e-03  # Convergence criterion for equilibrium lake stage solution
surfdep = 2.000e00  # Height of small topological variations in lake-bottom
stages = 268.00  # Initial stage of each lake at the beginning of the run

# ITMP  > 0, read lake definition data
# ITMP1 ≥ 0, read new recharge, evaporation, runoff, and withdrawal data for each lake
# LWRT  > 0, suppresses printout from the lake package

bdlknc_lyr1 = LakArr_lyr1.copy()
bdlknc_lyr2 = LakArr_lyr1.copy()
bdlknc_lyr3 = np.zeros(LakArr_lyr1.shape)

# Need to expand bdlknc_lyr1 non-zero values by 1 in either direction
# (left/right and up/down)
for i in np.arange(0, LakArr_lyr1.shape[0]):
    for j in np.arange(0, LakArr_lyr1.shape[1]):
        im1 = i - 1
        ip1 = i + 1
        jm1 = j - 1
        jp1 = j + 1

        if im1 >= 0:
            if LakArr_lyr1[i, j] == 1 and LakArr_lyr1[im1, j] == 0:
                bdlknc_lyr1[im1, j] = 1

        if ip1 < LakArr_lyr1.shape[0]:
            if LakArr_lyr1[i, j] == 1 and LakArr_lyr1[ip1, j] == 0:
                bdlknc_lyr1[ip1, j] = 1

        if jm1 >= 0:
            if LakArr_lyr1[i, j] == 1 and LakArr_lyr1[i, jm1] == 0:
                bdlknc_lyr1[i, jm1] = 1

        if jp1 < LakArr_lyr1.shape[1]:
            if LakArr_lyr1[i, j] == 1 and LakArr_lyr1[i, jp1] == 0:
                bdlknc_lyr1[i, jp1] = 1


bdlknc = [bdlknc_lyr1, bdlknc_lyr2, bdlknc_lyr3]
bdlknc = np.array(bdlknc)

flux_data = {0: [[0.0073, 0.0073, 0.0, 0.0]], 1: [[0.0073, 0.0073, 0.0, 0.0]]}

lak = flopy.modflow.ModflowLak(
    mf,
    nlakes=nlakes,
    ipakcb=ipakcb,
    theta=theta,
    nssitr=nssitr,
    sscncr=sscncr,
    surfdep=surfdep,
    stages=stages,
    lakarr=LakArr,
    bdlknc=bdlknc,
    flux_data=flux_data,
    unit_number=16,
)
# -

# ### Instantiate gage package for use with MODFLOW-NWT package

# +
gages = [
    [1, 225, 90, 3],
    [2, 68, 91, 3],
    [3, 33, 92, 3],
    [4, 165, 93, 3],
    [5, 123, 94, 3],
    [6, 77, 95, 3],
    [7, 173, 96, 3],
    [8, 328, 97, 3],
    [9, 115, 98, 3],
    [-1, -101, 1],
]

# gages = [[1,38,61,1],[2,67,62,1], [3,176,63,1], [4,152,64,1], [5,186,65,1], [6,31,66,1]]
files = [
    "no3.gag",
    "seg1_gag.out",
    "seg2_gag.out",
    "seg3_gag.out",
    "seg4_gag.out",
    "seg5_gag.out",
    "seg6_gag.out",
    "seg7_gag.out",
    "seg8_gag.out",
    "seg9_gag.out",
    "lak1_gag.out",
]

numgage = len(gages)
gage = flopy.modflow.ModflowGage(
    mf, numgage=numgage, gage_data=gages, filenames=files
)
# -

# ### Instantiate Unsaturated-Zone Flow (UZF) package for MODFLOW-NWT

# +
nuztop = 2
iuzfopt = 2
irunflg = 1
ietflg = 0
iuzfcb = 52
iuzfcb2 = 0
ntrail2 = 20
nsets2 = 20
nuzgag = 2
surfdep = 2.0

eps = 3.0
thts = 0.30
thti = 0.13079

fname_uzbnd = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "uzf_arrays",
    "iuzbnd.txt",
)
fname_runbnd = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "uzf_arrays",
    "irunbnd.txt",
)

iuzfbnd = np.loadtxt(fname_uzbnd)
irunbnd = np.loadtxt(fname_runbnd)

uzgag = [[106, 160, 121, 3], [1, 1, -122, 1]]

finf = {0: 1.8250e-03, 1: 1.8250e-03}

uzf = flopy.modflow.ModflowUzf1(
    mf,
    nuztop=nuztop,
    iuzfopt=iuzfopt,
    irunflg=irunflg,
    ietflg=ietflg,
    ipakcb=iuzfcb,
    iuzfcb2=iuzfcb2,
    ntrail2=ntrail2,
    nsets=nsets2,
    surfdep=surfdep,
    uzgag=uzgag,
    iuzfbnd=1,
    irunbnd=0,
    vks=1.0e-6,
    eps=3.5,
    thts=0.35,
    thtr=0.15,
    thti=0.20,
)
# -

# ### Instantiate Drain (DRN) package for MODFLOW-NWT

# +
fname_drnElv = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "drn_arrays",
    "elv.txt",
)
fname_drnCond = os.path.join(
    "..",
    "..",
    "examples",
    "data",
    "mt3d_example_sft_lkt_uzt",
    "drn_arrays",
    "cond.txt",
)

drnElv = np.loadtxt(fname_drnElv)
drnCond = np.loadtxt(fname_drnCond)

drnElv_lst = pd.DataFrame(
    {
        "lay": 1,
        "row": np.nonzero(drnElv)[0] + 1,
        "col": np.nonzero(drnElv)[1] + 1,
        "elv": drnElv[np.nonzero(drnElv)],
        "cond": drnCond[np.nonzero(drnCond)],
    },
    columns=["lay", "row", "col", "elv", "cond"],
)

# Convert the DataFrame into a list of lists for the drn constructor
stress_period_data = drnElv_lst.values.tolist()

# Create a dictionary, 1 entry for each of the two stress periods.
stress_period_data = {0: stress_period_data, 1: stress_period_data}

drn = flopy.modflow.ModflowDrn(
    mf, ipakcb=ipakcb, stress_period_data=stress_period_data
)
# -

# ### Instantiate linkage with mass transport routing (LMT) package for MODFLOW-NWT (generates linker file)

lmt = flopy.modflow.ModflowLmt(
    mf,
    output_file_name="NO3.ftl",
    output_file_header="extended",
    output_file_format="formatted",
    package_flows=["all"],
)

# ## Now work on MT3D-USGS file creation

# +
# Start by setting up MT3D-USGS class and pass in MODFLOW-NWT object for setting up a number of the BTN arrays

mt = flopy.mt3d.Mt3dms(
    modflowmodel=mf,
    modelname=modelname,
    model_ws=modelpth,
    version="mt3d-usgs",
    namefile_ext="mtnam",
    exe_name=mtexe,
    ftlfilename="no3.ftl",
    ftlfree=True,
)
# -

# ### Instantiate basic transport (BTN) package for MT3D-USGS

# +
ncomp = 1
lunit = "FT"
sconc = 0.0
prsity = 0.3
cinact = -1.0
thkmin = 0.000001
nprs = -2
nprobs = 10
nprmas = 10
dt0 = 0.1
nstp = 1
mxstrn = 500
ttsmult = 1.2
ttsmax = 100

# These observations need to be entered with 0-based indexing
obs = [[0, 104, 158], [1, 104, 158], [2, 104, 158]]

btn = flopy.mt3d.Mt3dBtn(
    mt,
    MFStyleArr=True,
    DRYCell=True,
    lunit=lunit,
    sconc=sconc,
    ncomp=ncomp,
    prsity=prsity,
    cinact=cinact,
    obs=obs,
    thkmin=thkmin,
    nprs=nprs,
    nprobs=nprobs,
    chkmas=True,
    nprmas=nprmas,
    dt0=dt0,
    nstp=nstp,
    mxstrn=mxstrn,
    ttsmult=ttsmult,
    ttsmax=ttsmax,
)
# -

# ### Instantiate advection (ADV) package for MT3D-USGS

# +
mixelm = 0
percel = 1.0000
mxpart = 5000
nadvfd = 1  # (1 = Upstream weighting)

adv = flopy.mt3d.Mt3dAdv(
    mt, mixelm=mixelm, percel=percel, mxpart=mxpart, nadvfd=nadvfd
)
# -

# ### Instantiate generalized conjugate gradient solver (GCG) package for MT3D-USGS

# +
mxiter = 1
iter1 = 50
isolve = 3
ncrs = 0
accl = 1.000000
cclose = 1.00e-06
iprgcg = 5

gcg = flopy.mt3d.Mt3dGcg(
    mt,
    mxiter=mxiter,
    iter1=iter1,
    isolve=isolve,
    ncrs=ncrs,
    accl=accl,
    cclose=cclose,
    iprgcg=iprgcg,
)
# -

# ### Instantiate dispersion (DSP) package for MT3D-USGS

# +
al = 0.1  # longitudinal dispersivity
trpt = 0.1  # ratio of the horizontal transverse dispersivity to 'AL'
trpv = 0.1  # ratio of the vertical transverse dispersitvity to 'AL'
dmcoef = 1.0000e-10

dsp = flopy.mt3d.Mt3dDsp(
    mt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef, multiDiff=True
)
# -

# ### Instantiate source-sink mixing (SSM) package for MT3D-USGS

# +
# no user-specified concentrations associated with boundary conditions

mxss = 11199

ssm = flopy.mt3d.Mt3dSsm(mt, mxss=mxss)
# -

# ### Instantiate reaction (RCT) package for MT3D-USGS

# +
isothm = 0
ireact = 1
irctop = 2
igetsc = 0
ireaction = 0

rc1 = 6.3258e-04  # first-order reaction rate for the dissolved phase
rc2 = 0.0  # Decay on Soil Layer

rct = flopy.mt3d.Mt3dRct(
    mt, isothm=isothm, ireact=ireact, igetsc=igetsc, rc1=rc1, rc2=rc2
)
# -

# ### Instantiate streamflow transport (SFT) package for MT3D-USGS

# +
nsfinit = len(reach_data)
mxsfbc = len(reach_data)
icbcsf = 0
ioutobs = 92
isfsolv = 1
wimp = 0.5
wups = 1.0
cclosesf = 1.0e-6
mxitersf = 10
crntsf = 1.0
iprtxmd = 0
coldsf = 0
dispsf = 0
obs_sf = [225, 293, 326, 491, 614, 691, 864, 1192, 1307]
sf_stress_period_data = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}

gage_output = [None, None, "no3.sftobs"]

sft = flopy.mt3d.Mt3dSft(
    mt,
    nsfinit=nsfinit,
    mxsfbc=mxsfbc,
    icbcsf=icbcsf,
    ioutobs=ioutobs,
    isfsolv=isfsolv,
    wimp=wimp,
    wups=wups,
    cclosesf=cclosesf,
    mxitersf=mxitersf,
    crntsf=crntsf,
    iprtxmd=iprtxmd,
    coldsf=coldsf,
    dispsf=dispsf,
    nobssf=len(obs_sf),
    obs_sf=obs_sf,
    sf_stress_period_data=sf_stress_period_data,
    filenames=gage_output,
)
# -

# ### Instantiate unsaturated-zone transport (UZT) package for MT3D-USGS

# +
mxuzcon = np.count_nonzero(irunbnd)
icbcuz = 45
iet = 0
wc = np.ones((nlay, nrow, ncol)) * 0.29
sdh = np.ones((nlay, nrow, ncol))

uzt = flopy.mt3d.Mt3dUzt(
    mt,
    mxuzcon=mxuzcon,
    icbcuz=icbcuz,
    iet=iet,
    iuzfbnd=iuzfbnd,
    sdh=sdh,
    cuzinf=1.4158e-03,
    filenames="no3",
)
# -

# ### Instantiate lake transport (LKT) package for MT3D-USGS

# +
nlkinit = 1
mxlkbc = 720
icbclk = 81
ietlak = 1
coldlak = 1

lkt_flux_data = {0: [[0, 1, 0.01667]], 1: [[0, 1, 0.02667]]}

lkt = flopy.mt3d.Mt3dLkt(
    mt,
    nlkinit=nlkinit,
    mxlkbc=mxlkbc,
    icbclk=icbclk,
    ietlak=ietlak,
    coldlak=coldlak,
    lk_stress_period_data=lkt_flux_data,
)
# -

# #### Write the MT3D-USGS input files for inspecting and running

# + pycharm={"name": "#%%\n"}
mf.write_input()
mt.write_input()

# mf.run_model()
# mt.run_model()
