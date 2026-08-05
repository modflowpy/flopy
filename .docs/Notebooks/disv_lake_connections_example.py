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

# # Embedded Lake Connections for a Vertex Grid
#
# `get_lak_connections()` builds Lake (LAK) Package connection data for a lake embedded in a vertex grid as well as a structured one. The cells a lake touches are found from the cells that share an edge with it, and the width of a connection is the length of the shared edge.
#
# This notebook puts the same lake used in the [structured grid example](https://flopy.readthedocs.io/en/latest/Notebooks/dis_lake_connections_example.html) on a vertex grid, and checks that the two agree.

# +
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.discretization import StructuredGrid, VertexGrid
from flopy.mf6.utils import get_lak_connections

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# ## Build a vertex grid and the equivalent structured grid
#
# The vertex grid discretizes the same domain as the structured grid, one cell2d for every row and column, so the two must give the same lake connections.

# +
nlay, nrow, ncol = 2, 7, 7
delr = delc = np.full(7, 100.0)
top = np.full((nrow, ncol), 10.0)
botm = np.array([np.full((nrow, ncol), 0.0), np.full((nrow, ncol), -10.0)])
ncpl = nrow * ncol

structured = StructuredGrid(delr=delr, delc=delc, top=top, botm=botm, nlay=nlay)

xv = np.concatenate(([0.0], np.cumsum(delr)))
yv = delc.sum() - np.concatenate(([0.0], np.cumsum(delc)))
vertices, ivert = [], {}
for i in range(nrow + 1):
    for j in range(ncol + 1):
        ivert[(i, j)] = len(vertices)
        vertices.append((len(vertices), float(xv[j]), float(yv[i])))

cell2d = [
    (
        i * ncol + j,
        0.5 * (xv[j] + xv[j + 1]),
        0.5 * (yv[i] + yv[i + 1]),
        4,
        ivert[(i, j)],
        ivert[(i, j + 1)],
        ivert[(i + 1, j + 1)],
        ivert[(i + 1, j)],
    )
    for i in range(nrow)
    for j in range(ncol)
]

vertex = VertexGrid(
    vertices=vertices,
    cell2d=cell2d,
    top=top.flatten(),
    botm=botm.reshape(nlay, ncpl),
    nlay=nlay,
)
print(f"Vertex grid: {vertex.nlay} layers of {vertex.ncpl} cells")
# -

# ## Build the lake connections
#
# The lake occupies the same three by three block in the upper layer. On the vertex grid the lake map has one value for every cell2d rather than a row and a column.

# +
lake_map_dis = np.full((nlay, nrow, ncol), -1, dtype=np.int32)
lake_map_dis[0, 2:5, 2:5] = 0
lake_map_disv = lake_map_dis.reshape(nlay, ncpl)

dis_idomain, dis_conn, dis_data = get_lak_connections(
    structured,
    np.ma.masked_where(lake_map_dis < 0, lake_map_dis),
    idomain=np.ones((nlay, nrow, ncol), dtype=int),
    bedleak=0.1,
)
disv_idomain, disv_conn, disv_data = get_lak_connections(
    vertex,
    np.ma.masked_where(lake_map_disv < 0, lake_map_disv),
    idomain=np.ones((nlay, ncpl), dtype=int),
    bedleak=0.1,
)

print(f"structured connections: {dis_conn}")
print(f"vertex connections    : {disv_conn}")
print(f"first vertex record   : {disv_data[0]}")
# -

# The two grids give the same connections. A structured cellid is a layer, row, and column while a vertex cellid is a layer and a cell2d number, so the structured cellids are flattened before the two are compared.


# +
def normalize(row):
    lakeno, _, cellid, claktype, _, _, _, connlen, connwidth = row
    if len(cellid) == 3:
        k, i, j = cellid
        cellid = (k, i * ncol + j)
    return (lakeno, *cellid, claktype, connlen, connwidth)


assert sorted(map(normalize, dis_data)) == sorted(map(normalize, disv_data))
assert np.array_equal(dis_idomain.reshape(nlay, ncpl), disv_idomain)
print("The vertex grid reproduces the structured grid connections.")
# -

# ## Run the model
#
# The connectiondata and the idomain from the vertex grid go straight into a
# DISV model. Constant heads on the left and right edges drive flow across the
# lake.

# +
temp_dir = TemporaryDirectory()
name = "disv_lake"

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
flopy.mf6.ModflowGwfdisv(
    gwf,
    nlay=nlay,
    ncpl=ncpl,
    top=top.flatten(),
    botm=botm.reshape(nlay, ncpl),
    vertices=vertices,
    cell2d=cell2d,
    idomain=disv_idomain,
)
flopy.mf6.ModflowGwfic(gwf, strt=8.0)
flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)
flopy.mf6.ModflowGwfchd(
    gwf,
    stress_period_data=[[(0, i * ncol), 9.0] for i in range(nrow)]
    + [[(0, i * ncol + ncol - 1), 6.0] for i in range(nrow)],
)
lak = flopy.mf6.ModflowGwflak(
    gwf,
    stage_filerecord=f"{name}.lak.stage.bin",
    nlakes=1,
    packagedata=[[0, 7.5, disv_conn[0]]],
    connectiondata=disv_data,
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
# The lake cells are inactive, so the head array has no value there. Filling
# those cells with the stage of the lake that occupies them gives a single
# array of the water surface that can be plotted in one pass.

# +
lake_map_masked = np.ma.masked_where(lake_map_disv < 0, lake_map_disv)

head = gwf.output.head().get_data().reshape(nlay, ncpl)
stage = lak.output.stage().get_data().flatten()
for lake_number in np.unique(lake_map_masked.compressed()):
    head[lake_map_masked == lake_number] = stage[lake_number]

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
pmv = flopy.plot.PlotMapView(modelgrid=vertex, ax=ax, layer=0)
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
