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
#     section: mf6
#     authors:
#       - name: Joseph Hughes
# ---

# # Embedded Lake Connections for a Structured Grid
#
# A lake embedded in the model domain replaces the cells it occupies, so the MODFLOW 6 Lake (LAK) Package needs a connection to every active cell that touches it. `get_lak_connections()` builds those connections from an array of lake numbers, and returns the idomain with the lake cells deactivated, the number of connections in each lake, and the connectiondata block for the package.
#
# We build a lake embedded in one layer of a structured grid, and put the simulated lake stage into the head array so that a single map shows the water surface. The companion notebook does the same thing on a vertex grid.

# +
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.discretization import StructuredGrid
from flopy.mf6.utils import get_lak_connections

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# ## Build the grid and the lake
#
# We use a grid of two layers with 7 rows and 7 columns. The lake occupies a three by three block in the upper layer, and is given lake number 0. Cells that are not part of a lake are masked.

# +
nlay, nrow, ncol = 2, 7, 7
delr = delc = np.full(7, 100.0)
top = np.full((nrow, ncol), 10.0)
botm = np.array([np.full((nrow, ncol), 0.0), np.full((nrow, ncol), -10.0)])

modelgrid = StructuredGrid(delr=delr, delc=delc, top=top, botm=botm, nlay=nlay)

lake_map = np.full((nlay, nrow, ncol), -1, dtype=np.int32)
lake_map[0, 2:5, 2:5] = 0
lake_map = np.ma.masked_where(lake_map < 0, lake_map)

print(f"Lake cells in layer 1: {(~lake_map.mask[0]).sum()}")
# -

# ## Build the lake connections
#
# The lake map is the same shape as the model grid, so the lake is embedded rather than sitting on top of the model. We give the same bed leakance to every connection.

# +
idomain, connection_dict, connectiondata = get_lak_connections(
    modelgrid,
    lake_map,
    idomain=np.ones((nlay, nrow, ncol), dtype=int),
    bedleak=0.1,
)

print(f"Connections in each lake: {connection_dict}")
print(f"connectiondata rows: {len(connectiondata)}")
# -

# A connection is horizontal where the lake meets a cell in the same layer, and vertical where it sits on the cell below. The three by three lake has twelve cells around its perimeter and nine beneath it.

# +
claktype = [row[3] for row in connectiondata]
for kind in ("horizontal", "vertical"):
    print(f"{kind:12s}{claktype.count(kind)}")

print()
print("lakeno iconn cellid          claktype     bedleak  connlen  connwidth")
for row in connectiondata[:4]:
    lakeno, iconn, cellid, kind, leak, _, _, connlen, connwidth = row
    print(
        f"{lakeno:6d}{iconn:6d} {cellid!s:15s} {kind:12s}"
        f"{leak:8.2f}{connlen:9.1f}{connwidth:10.1f}"
    )
# -

# The cells the lake occupies are deactivated in the returned idomain, so the lake replaces them rather than sharing the domain with them.

print(f"idomain in the lake cells: {np.unique(idomain[0, 2:5, 2:5])}")
print(f"active cells remaining   : {(idomain > 0).sum()} of {idomain.size}")

# ## Run the model
#
# The connectiondata and the idomain go straight into the MODFLOW 6 model. We
# set constant heads on the left and right edges to drive flow across the lake.

# +
temp_dir = TemporaryDirectory()
name = "dis_lake"

sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=temp_dir.name, exe_name="mf6")
flopy.mf6.ModflowTdis(sim)
flopy.mf6.ModflowIms(
    sim,
    linear_acceleration="bicgstab",
    outer_dvclose=1e-9,
    inner_dvclose=1e-10,
    outer_maximum=200,
)
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, newtonoptions="newton under_relaxation")
flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
    idomain=idomain,
)
flopy.mf6.ModflowGwfic(gwf, strt=8.0)
flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)
flopy.mf6.ModflowGwfchd(
    gwf,
    stress_period_data=[[(0, i, 0), 9.0] for i in range(nrow)]
    + [[(0, i, ncol - 1), 6.0] for i in range(nrow)],
)
lak = flopy.mf6.ModflowGwflak(
    gwf,
    stage_filerecord=f"{name}.lak.stage.bin",
    nlakes=1,
    packagedata=[[0, 7.5, connection_dict[0]]],
    connectiondata=connectiondata,
    perioddata={0: [[0, "rainfall", 0.001]]},
)
flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", saverecord=[("HEAD", "ALL")])

sim.write_simulation(silent=True)
success, buff = sim.run_simulation(silent=True)
assert success, "\n".join(buff[-15:])
print("model converged")
# -

# ## Put the lake stage into the head array
#
# The lake cells are inactive, so the head array has no value there. We fill
# those cells with the stage of the lake that occupies them, which gives a
# single array of the water surface that can be plotted in one pass.

# +
head = gwf.output.head().get_data()
stage = lak.output.stage().get_data().flatten()
for lake_number in np.unique(lake_map.compressed()):
    head[lake_map == lake_number] = stage[lake_number]

print(f"lake stage: {stage[0]:.3f}")
print(f"cells with no head value: {(np.abs(head) > 1e29).sum()}")
print(f"head range: {head.min():.3f} to {head.max():.3f}")
# -

# ## Head map
#
# The lake reads as part of the water surface rather than as a hole in it.

# +
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax, layer=0)
cb = pmv.plot_array(head[0])
pmv.plot_grid(colors="0.5", lw=0.5)
pmv.contour_array(head[0], colors="white", linewidths=1.0)
ax.set_title("Head with the lake stage embedded")
fig.colorbar(cb, ax=ax, shrink=0.7, label="head")
# -

try:
    temp_dir.cleanup()
except (PermissionError, NotADirectoryError):
    pass
