# OAHU island-wide GWRP model
#  using retarded units (ft) and a coarse test grid
#  simulating 1 short stress period (eventually steady-state)
#  simulating 1 layer of volcanic rock referenced to sea level
#  simulating 50% seawater salinity with SWI2
#    uses GH relation for initial salinity conditions
#    apply crude mask of Oahu coastline as ocean boundary
#    uniform recharge, add 2 wells, and 1 horizontal flow barrier
#  changed origin of grid in plot to upper left corner.
#
# uses FLOPY3, modified from FLOPY2 tutorial 2
# 
# Kolja Rotzoll (kolja@usgs.gov), 1/15/2015
#------------------------------------------------------
import sys, os 
import numpy as np
from pylab import *
from PIL import Image, ImageDraw

flopypath = os.path.join('..', '..')
if flopypath not in sys.path:
    print 'Adding to sys.path: ', flopypath
    sys.path.append(flopypath)

import flopy
import flopy.utils.binaryfile as bf

#--flopy objects
modelname = 'Oahu_01'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005', model_ws=os.path.join('data'))

#--model domain and grid definition
ztop = 30.                       # top of layer (ft rel to msl)
botm = -1000.                    # bottom of layer (ft rel to msl)
nlay = 1                         # number of layers (z)
nrow = 18                        # number of rows (y)
ncol = 20                        # number of columns (x)
delr = 16000  		             # row width of cell, in ft
delc = delr   		             # column width of cell, in ft
Lx = delr * ncol   		         # length of x model domain, in ft
Ly = delc * nrow   		         # length of y model domain, in ft

#--define the stress periods
nper   = 1
ts     = 1                       # length of time step, in days
nstp   = 1000                    # number of time steps
perlen = nstp*ts                 # length of simulation, in days
steady = True                    # steady state or transient
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm, nper=nper,
                               perlen=perlen, nstp=nstp, steady=steady)

#--hydraulic parameters (lpf or bcf)
hk = 1500. 						# horizontal K
sy = 0.05						# specific yield
ss = 1.e-5						# specific storage
layavg = 0						# 0 = harmonic mean, 1 = logarithmic mean, 
								# 2 = arithmetic mean of sat b and log-mean K
laytyp = 1                      # 0 = confined, 1 = convertible
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, sy=sy, ss=ss, laytyp=laytyp, layavg=layavg)
laycon = 2						# 0 = confined, 1 = unconfined T varies, 
								# 2 = convertible T const, 3 = convertible T varies

#--water/land interface (now replaced with coarse Oahu coastline)
polyg = [(6,13),(3,6),(6,6),(9,3),(12,8),(14,9),(16,13),(13,14),(11,13),(6,13)] # referenced to row/col
px, py = zip(*polyg)
colcell, rowcell = meshgrid(range(ncol), range(nrow))
mask = Image.new('L', (ncol, nrow), 0)
ImageDraw.Draw(mask).polygon(polyg, outline=1, fill=1)
index = np.array(mask)

#--BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)	# active cells
h_start = np.zeros((nrow, ncol), dtype=float)
peak = 15							# maximum expected water level 
h_start[:, :][index==1] = peak		# starting heads over land
h_start[:, :][index==0] = 0			# starting heads over ocean
#print h_start
bas = flopy.modflow.ModflowBas(mf,ibound=ibound,strt=h_start)

#--general head boundary
nghb = ncol * nrow - np.sum(index)
lrchc = np.zeros((nghb, 5))
lrchc[:, 0] = 0
lrchc[:, 1] = rowcell[index==0]
lrchc[:, 2] = colcell[index==0]
lrchc[:, 3] = 0.
lrchc[:, 4] = hk * 10
#print lrchc
ghb = flopy.modflow.ModflowGhb(mf,stress_period_data=lrchc)

#--recharge & withdrawal
Recharge =  600*133680.56    # Total recharge over the island, in ft^3/d
nrech = np.sum(index)
lrcq = np.zeros((nrech, 4))
lrcq[:, 0] = 0
lrcq[:, 1] = rowcell[index==1]
lrcq[:, 2] = colcell[index==1]
lrcq[:, 3] = Recharge/nrech
lrcq = np.vstack((lrcq,[0,8,7,-90*133680],[0,10,9,-80*133680]))  # add wells (row/col, zero-based)
#print lrcq
wel = flopy.modflow.ModflowWel(mf,stress_period_data=lrcq)

#--horizontal flow barrier
nhfb = 12
lrcrch = np.zeros((nhfb, 6))
lrcrch[:, 0] = 0                        # layer
lrcrch[:, 1] = arange(2,nhfb+2)           # row 1
lrcrch[:, 2] = ones(nhfb)*(ncol/2-1)    # col 1
lrcrch[:, 3] = arange(2,nhfb+2)           # row 2
lrcrch[:, 4] = ones(nhfb)*(ncol/2)      # col 2
lrcrch[:, 5] = 0.000001                  # hydrologic characteristics
#print lrcrch
hfb = flopy.modflow.ModflowHfb(mf,layer_row_column_data=lrcrch)

#--SWI input
z1 = np.zeros((nrow, ncol))
z1[index==1] = peak*(-40)				# 50% salinity from starting head
z = array([z1]) 						# zeta interfaces 
#print z
iso = np.zeros((nrow, ncol), dtype=np.int32)	# water type of sinks and sources
iso[:,:][index==1] = 1 					# land
iso[:,:][index==0] = -2 				# ocean (ghb)
#print iso
swi = flopy.modflow.ModflowSwi2(mf,nsrf=1,istrat=1,toeslope=0.04,tipslope=0.04,
	        				    nu=[0,0.025],zeta=z,ssz=0.05,isource=iso,nsolver=1)

#--output control & solver
oc = flopy.modflow.ModflowOc(mf, words=['head'], save_head_every=1,
							 item2=[[0, 1, 0, 0]], item3=[[0, 1, 0, 0]])
pcg = flopy.modflow.ModflowPcg(mf, hclose=1.0e-4, rclose=5.0e-0) # pre-conjugate gradient solver
#de4 = flopy.modflow.ModflowDe4(mf, itmx=1, hclose=1e-5)  		 # direct solver
#----------------------------------------------------------------------

#--write the model input files
mf.write_input()

print '\n\nfinished...\n'
