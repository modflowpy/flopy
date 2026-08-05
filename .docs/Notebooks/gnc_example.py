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
#     section: dis
#     authors:
#       - name: Joseph Hughes
# ---

# # Ghost Node Correction (GNC) Data for MODFLOW 6
#
# The control volume finite difference formulation used by MODFLOW assumes that the line connecting two cell centers crosses the shared face at a right angle through the middle of the face. A quadtree grid violates that assumption wherever a coarse cell connects to a finer cell, because the shared face is offset from the center of the coarse cell. The Ghost Node Correction (GNC) Package corrects the resulting error by interpolating the head at a ghost node, which is the point in the coarse cell that does lie on the perpendicular through the middle of the face.
#
# FloPy builds GNC Package input two ways, and we demonstrate both here. The first uses the ghost node data GRIDGEN writes when it exports a grid. The second computes the ghost node data from a model grid, a grid conforming array of refinement levels, and the grid connectivity, and does not require GRIDGEN.
#
# We also compare the ghost node correction against XT3D, which is the other MODFLOW 6 option for improving accuracy on a quadtree grid, in terms of both the answer and what the correction costs.

# +
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import flopy
from flopy.utils import flopy_io, get_gnc, get_gridprops_gnc6
from flopy.utils.gridgen import Gridgen

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# The FloPy GRIDGEN module requires that the gridgen executable can be called using subprocess **(i.e., gridgen is in your path)**.

gridgen_exe = flopy.which("gridgen")
if gridgen_exe is None:
    msg = (
        "Warning, gridgen is not in your path. "
        "When you create the gridgen object you will need to "
        "provide a full path to the gridgen binary executable."
    )
    print(msg)
else:
    print(f"gridgen executable was found at: {flopy_io.relpath_safe(gridgen_exe)}")

# +
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)
gridgen_ws = workspace / "gridgen"
gridgen_ws.mkdir(parents=True, exist_ok=True)

print(f"Model workspace is : {flopy_io.scrub_login(str(workspace))}")
print(f"Gridgen workspace is : {flopy_io.scrub_login(str(gridgen_ws))}")
# -

# ## Build the quadtree grid
#
# GRIDGEN works from a base MODFLOW grid. We use a 3 layer grid of 20 rows and 20 columns and refine a square in the middle of the grid by three levels, which produces cells one eighth the width of the base grid cells.

# +
nlay, nrow, ncol = 3, 20, 20
delr = delc = 1.0
top = 1.0
botm = [top - (k + 1) * top / nlay for k in range(nlay)]

base_grid = flopy.discretization.StructuredGrid(
    delr=np.full(ncol, delr, dtype=float),
    delc=np.full(nrow, delc, dtype=float),
    top=np.full((nrow, ncol), top),
    botm=np.array([np.full((nrow, ncol), b) for b in botm]),
)
# -

# +
center, half_width = ncol / 2.0, 3.0
corners = [
    (center - half_width, center - half_width),
    (center + half_width, center - half_width),
    (center + half_width, center + half_width),
    (center - half_width, center + half_width),
]

g = Gridgen(base_grid, model_ws=str(gridgen_ws))
g.add_refinement_features([Polygon(corners)], "polygon", 3, range(nlay))
g.build(verbose=False)

disv_gridprops = g.get_gridprops_disv()
ncpl = disv_gridprops["ncpl"]
print(f"Number of cells per layer: {ncpl}")
# -

# ## Ghost node data from GRIDGEN
#
# GRIDGEN computes the ghost node data whenever it exports a grid and writes it to the `qtg.gnc.dat` file. The `get_gnc()` method reads that file and returns a record array with zero-based node numbers, where cell `n` contains the ghost node, cell `m` is the connecting cell, and cells `j0` and `j1` are the contributing cells whose heads are interpolated.

gnc = g.get_gnc()
print(f"Number of ghost nodes: {len(gnc)}")
print(gnc[:5])

# GRIDGEN always writes two contributing cells. When a ghost node has only one contributing cell, that cell is repeated and its contributing factor is halved, which MODFLOW accumulates into the same matrix position. The contributing factors always sum to less than one, because one minus the sum is the factor applied to the head in cell `n`.

alpha = gnc["alpha0"] + gnc["alpha1"]
print(f"Contributing factors range from {alpha.min():.4f} to {alpha.max():.4f}")

# A grid that is not refined has no ghost nodes and the record array is empty. The GNC Package should not be created in that case.

# The `get_gridprops_gnc6()` method converts the node numbers to cellids and returns a dictionary that can be unpacked directly into the `ModflowGwfgnc` constructor. Cellids are built for a DISV grid here; pass `dis_type="disu"` for a DISU grid.

gnc_gridprops = g.get_gridprops_gnc6(dis_type="disv")
print(f"numgnc: {gnc_gridprops['numgnc']}")
print(f"numalphaj: {gnc_gridprops['numalphaj']}")
print(f"first record: {gnc_gridprops['gncdata'][0]}")

# ## Ghost node data from the model grid
#
# The ghost node data can also be computed from the model grid, without running GRIDGEN. All that is needed is the grid, which provides the cell centers and the connectivity, and a grid conforming array of refinement levels. Cell areas are used when levels are not supplied.

vgrid = flopy.discretization.VertexGrid(**g.get_gridprops_vertexgrid())

# A vertex grid does not carry connectivity, so `get_gnc()` builds it from the cells that share an edge. Connectivity can also be passed with the `ia` or `iac` and `ja` arguments, which is what an unstructured grid already provides.
#
# The refinement level of each cell follows from the cell area, where level 0 is a base grid cell and each level halves the cell width. We compute the areas from the cell vertices so that nothing in this section depends on GRIDGEN. The levels are given for one layer, and `get_gnc()` applies them to every layer.


# +
def cell_area(icpl):
    x, y = np.array(vgrid.get_cell_vertices(icpl)).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


area = np.array([cell_area(icpl) for icpl in range(ncpl)])
level = np.round(np.log2(np.sqrt(area.max() / area))).astype(int)
print(f"Refinement levels present: {np.unique(level)}")
# -

# The `get_gnc()` function returns the same record array that GRIDGEN wrote. We ask for two contributing cells so the result can be compared directly.

gnc_grid = get_gnc(vgrid, level=level, numalphaj=2)
print(f"Number of ghost nodes: {len(gnc_grid)}")
print(gnc_grid[:5])


# The two record arrays hold the same ghost nodes. We sort the records because the two routines visit the cells in a different order, and we sort the contributing cells within each record because the two cells are sometimes listed in the opposite order. That ordering does not matter, since MODFLOW accumulates the contribution of each cell.


# +
def sort_gnc(recarray):
    nodes = np.sort(np.column_stack([recarray["j0"], recarray["j1"]]), axis=1)
    alpha = np.sort(np.column_stack([recarray["alpha0"], recarray["alpha1"]]), axis=1)
    key = np.column_stack([recarray["n"], recarray["m"], nodes, alpha])
    return key[np.lexsort(key.T[::-1])]


# GRIDGEN writes the contributing factors with six significant digits
assert np.allclose(sort_gnc(gnc_grid), sort_gnc(gnc), atol=2e-6)
print("The computed ghost node data matches the GRIDGEN ghost node data.")
# -

# The dictionary for the GNC Package is built with the `get_gridprops_gnc6()` function, which also verifies that each cell `n` is connected to cell `m` and that the contributing factors sum to less than one.

gnc_gridprops = get_gridprops_gnc6(gnc_grid, dis_type="disv", ncpl=ncpl)
print(f"numgnc: {gnc_gridprops['numgnc']}")

# ## Where the ghost nodes are
#
# Every ghost node lies on a connection between a coarse cell and a finer cell, so the ghost nodes trace the boundary of the refined area. We plot the connections in the upper layer.

# +
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=vgrid, ax=ax, layer=0)
pmv.plot_grid(colors="0.5", lw=0.5)

xc, yc = vgrid.xcellcenters, vgrid.ycellcenters
for rec in gnc_grid[gnc_grid["n"] < ncpl]:
    n, m = rec["n"], rec["m"]
    ax.plot([xc[n], xc[m]], [yc[n], yc[m]], color="C3", lw=1.0, zorder=2)
    ax.plot(xc[m], yc[m], "o", color="C3", ms=2.5, zorder=3)
    for j in (rec["j0"], rec["j1"]):
        ax.plot(xc[j], yc[j], "s", color="C0", ms=3.0, zorder=3)

ax.plot([], [], color="C3", lw=1.0, label="ghost node connection")
ax.plot([], [], "s", color="C0", ms=3.0, lw=0, label="contributing cell")
ax.legend(loc="upper right", framealpha=1.0)
ax.set_title("Ghost node connections in layer 1")
# -

# ## Effect of the correction
#
# We build the same model three ways and compare the results. The uncorrected model uses the standard formulation, the corrected model adds the GNC Package, and the third model uses XT3D. The correction is applied implicitly by default, so the BICGSTAB linear acceleration option is specified in the IMS Package.
#
# The model is confined and homogeneous, with constant heads on the left and right edges and no flow across the top and bottom edges. Head then varies linearly between the two constant head columns, which gives an exact solution to compare against.

# +
h_left, h_right = 1.0, 0.0
xcenters = vgrid.xcellcenters
left = [icpl for icpl in range(ncpl) if xcenters[icpl] < delr]
right = [icpl for icpl in range(ncpl) if xcenters[icpl] > ncol - delr]

chdspd = [[(k, icpl), h_left] for k in range(nlay) for icpl in left]
chdspd += [[(k, icpl), h_right] for k in range(nlay) for icpl in right]

x_left, x_right = xcenters[left].mean(), xcenters[right].mean()
exact = h_left + (h_right - h_left) * (xcenters - x_left) / (x_right - x_left)
exact = np.tile(exact, nlay)
print(f"Number of constant head cells: {len(chdspd)}")
# -


# MODFLOW 6 reports the memory it allocates at the end of the simulation listing file, which we read back for each model along with the simulated heads.


# +
def run_model(name, gnc=False, xt3d=False):
    ws = workspace / name
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(ws), exe_name="mf6")
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(
        sim,
        linear_acceleration="bicgstab",
        inner_maximum=1000,
        inner_dvclose=1e-10,
        outer_dvclose=1e-10,
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
    flopy.mf6.ModflowGwfic(gwf, strt=0.5 * (h_left + h_right))
    flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=xt3d, icelltype=0, k=1.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{name}.hds", saverecord=[("HEAD", "ALL")]
    )
    if gnc:
        flopy.mf6.ModflowGwfgnc(gwf, **gnc_gridprops)
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, f"{name} did not converge"

    listing = (ws / "mfsim.lst").open().read()
    memory = float(re.search(r"Total\s+([0-9.E+-]+)\s*\n\s*Virtual", listing).group(1))
    return gwf.output.head().get_data().flatten(), memory


# +
heads, error, memory = {}, {}, {}
for name, kwargs in [
    ("uncorrected", {}),
    ("gnc", {"gnc": True}),
    ("xt3d", {"xt3d": True}),
]:
    heads[name], memory[name] = run_model(name, **kwargs)
    error[name] = np.abs(heads[name] - exact)

print(f"{'variant':14s}{'max error':>12s}{'rms error':>12s}{'memory, MB':>13s}")
for name in ("uncorrected", "gnc", "xt3d"):
    rms = np.sqrt((error[name] ** 2).mean())
    print(f"{name:14s}{error[name].max():12.3e}{rms:12.3e}{memory[name]:13.1f}")
# -

# The ghost node correction removes about 18 times the head error introduced by the refinement. XT3D reproduces a linear head field exactly by construction, so it is exact on this problem; that is a property of this test rather than a general ranking of the two corrections.

for name in ("gnc", "xt3d"):
    removed = 1.0 - error[name].max() / error["uncorrected"].max()
    print(f"{name:5s} removes {100 * removed:.1f} percent of the error")

# ## Cost of the correction
#
# The two corrections reach a comparable answer by different means. XT3D replaces the flow calculation on every connection in the model, which extends the stencil of every cell. The ghost node correction only adds terms on the connections that have a ghost node, which are the connections between a coarse cell and a finer cell, and there are far fewer of those.

print(f"Cells in the model: {ncpl * nlay}")
print(f"Ghost nodes: {gnc_gridprops['numgnc']}")
print(
    f"Ghost nodes are on {100 * gnc_gridprops['numgnc'] / (ncpl * nlay):.1f} "
    "percent of the cells"
)

# That shows up in the memory MODFLOW 6 allocates. XT3D nearly doubles it, because the extended stencil applies to every cell in the model. The ghost node correction adds a couple of percent. Both corrections remove nearly all of the error introduced by the refinement, and the ghost node correction does so in about half the memory.

for name in ("gnc", "xt3d"):
    print(
        f"{name:5s} memory relative to the uncorrected model: "
        f"{memory[name] / memory['uncorrected']:.3f}"
    )
print(f"gnc memory relative to xt3d: {memory['gnc'] / memory['xt3d']:.3f}")

# Run times are not compared here. They depend on how many iterations the solver takes, and the ordering of the two corrections changes with the problem, so run time is not a reliable way to choose between them.
#
# The practical difference is in what each one asks of the user. XT3D is a single keyword in the NPF Package and needs no other input. The ghost node correction needs the ghost node data, which was the difficult part of using the GNC Package and is what the FloPy functionality shown in this notebook provides.

# ## Where the error is
#
# The error in the uncorrected model is concentrated on the boundary of the refined area, which is where the ghost nodes are. The ghost node correction removes most of it.

# +
vmax = error["uncorrected"].max()

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, name in zip(axes, ("uncorrected", "gnc")):
    ax.set_aspect("equal")
    pmv = flopy.plot.PlotMapView(modelgrid=vgrid, ax=ax, layer=0)
    cb = pmv.plot_array(error[name], cmap="magma_r", vmin=0.0, vmax=vmax)
    pmv.plot_grid(colors="0.5", lw=0.3, alpha=0.5)
    ax.set_title(f"{name}, layer 1")
fig.colorbar(cb, ax=axes, shrink=0.7, label="absolute head error")
# -

# Clean up the temporary workspace.

try:
    temp_dir.cleanup()
except (PermissionError, NotADirectoryError):
    pass
