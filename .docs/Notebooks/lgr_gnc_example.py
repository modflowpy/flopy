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

# # Ghost Node Correction (GNC) Data for a GWF-GWF Exchange
#
# The control volume finite difference formulation used by MODFLOW 6 assumes that the line connecting two cell centers crosses the shared face at a right angle through the middle of the face. A local grid refinement violates that assumption on the exchange between the parent and the child model, because the face a child cell shares with a parent cell is offset from the center of the parent cell. The Ghost Node Correction (GNC) Package of the exchange corrects the resulting error by interpolating the head at a ghost node, which is the point in the parent cell that does lie on the perpendicular through the middle of the face.
#
# The `Lgr` utility builds the ghost node data along with the exchange data, and we use it here to correct a parent and child model. The same data can be built for any two models joined by a GWF-GWF exchange with `flopy.utils.get_gnc_exchange`.
#
# The problem is confined and homogeneous, with constant heads on the left and right edges of the parent model and no flow across the top and bottom edges. Head then varies linearly between the two constant head columns, which gives an exact solution to compare against.

# +
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.utils import flopy_io
from flopy.utils.lgrutil import Lgr

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# +
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)
print(f"Model workspace is : {flopy_io.scrub_login(str(workspace))}")
# -

# ## Build the parent and child grids
#
# The parent grid is a single layer of 12 rows and 12 columns. The block of parent cells in the middle is made inactive and is replaced by a child model with three cells per parent cell in each direction.

# +
nlay, nrow, ncol = 1, 12, 12
delr = delc = 1.0
top = 1.0
botm = [0.0]
ncpp = 3

refine_mask = np.ones((nlay, nrow, ncol), dtype=int)
refine_mask[:, 4:8, 4:8] = 0

lgr = Lgr(
    nlay,
    nrow,
    ncol,
    delr,
    delc,
    np.full((nrow, ncol), top),
    np.zeros((nlay, nrow, ncol)),
    refine_mask,
    ncpp=ncpp,
    ncppl=[1],
)

parent, child = lgr.parent, lgr.child
print(f"Parent grid: {parent.nrow} rows, {parent.ncol} columns")
print(f"Child grid : {child.nrow} rows, {child.ncol} columns")
# -

# ## Exchange and ghost node data
#
# `get_exchange_data()` returns the connections between the two models, and `get_gnc_data()` returns the ghost node data for those connections.

# +
exchangedata = lgr.get_exchange_data(angldegx=True, cdist=True)
gnc_gridprops = lgr.get_gnc_data()

print(f"Number of exchanges: {len(exchangedata)}")
print(f"numgnc: {gnc_gridprops['numgnc']}")
print(f"numalphaj: {gnc_gridprops['numalphaj']}")
# -

# MODFLOW 6 requires one ghost node record for every exchange record, in the same order, so `numgnc` always equals the number of exchanges. A connection that needs no correction is written with a cellid of zero and a contributing factor of zero, which MODFLOW 6 skips.
#
# With three child cells across the face of a parent cell, the middle one is centered on that face and needs no correction, so one connection in three is written that way.

# +
corrected = [rec for rec in gnc_gridprops["gncdata"] if rec[-1] != 0.0]
skipped = [rec for rec in gnc_gridprops["gncdata"] if rec[-1] == 0.0]

print(f"Connections with a ghost node: {len(corrected)}")
print(f"Connections without one      : {len(skipped)}")
print(f"first corrected record: {corrected[0]}")
print(f"first skipped record  : {skipped[0]}")
# -

# The ghost node is in the parent cell, and so is the contributing cell whose head is interpolated with it. The contributing factor is the offset of the child cell from the center of the parent cell divided by the distance to the contributing cell.

alpha = np.array([rec[-1] for rec in corrected])
print(f"Contributing factors: {np.unique(np.round(alpha, 6))}")

# ## Where the ghost nodes are
#
# Every ghost node lies on a connection between a parent cell and a child cell, so the ghost nodes trace the boundary of the child model.

# +
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=parent, ax=ax)
pmv.plot_grid(colors="0.5", lw=0.6)
cmv = flopy.plot.PlotMapView(modelgrid=child, ax=ax)
cmv.plot_grid(colors="0.5", lw=0.3)

pxc, pyc = parent.xcellcenters, parent.ycellcenters
cxc, cyc = child.xcellcenters, child.ycellcenters
for cellidn, cellidm, cellidj, factor in corrected:
    xn, yn = pxc[cellidn[1], cellidn[2]], pyc[cellidn[1], cellidn[2]]
    xm, ym = cxc[cellidm[1], cellidm[2]], cyc[cellidm[1], cellidm[2]]
    xj, yj = pxc[cellidj[1], cellidj[2]], pyc[cellidj[1], cellidj[2]]
    ax.plot([xn, xm], [yn, ym], color="C3", lw=0.9, zorder=2)
    ax.plot(xj, yj, "s", color="C0", ms=3.5, zorder=3)

ax.plot([], [], color="C3", lw=0.9, label="ghost node connection")
ax.plot([], [], "s", color="C0", ms=3.5, lw=0, label="contributing cell")
ax.legend(loc="upper right", framealpha=1.0)
ax.set_title("Ghost node connections on the exchange")
# -

# ## Build and run the models
#
# The same simulation is built with and without the GNC Package of the exchange. The correction is applied implicitly, so the BICGSTAB linear acceleration option is specified in the IMS Package.

# +
h_left, h_right = 1.0, 0.0


def exact(grid):
    x0, x1 = parent.xcellcenters[0, 0], parent.xcellcenters[0, -1]
    return h_left + (h_right - h_left) * (grid.xcellcenters - x0) / (x1 - x0)


def build_simulation(name, gnc=False):
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(workspace / name), exe_name="mf6"
    )
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(
        sim,
        linear_acceleration="bicgstab",
        inner_maximum=1000,
        inner_dvclose=1e-11,
        outer_dvclose=1e-11,
    )

    gwfp = flopy.mf6.ModflowGwf(sim, modelname="parent", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwfp,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=refine_mask,
    )
    flopy.mf6.ModflowGwfic(gwfp, strt=0.5)
    flopy.mf6.ModflowGwfnpf(gwfp, icelltype=0, k=1.0)
    flopy.mf6.ModflowGwfchd(
        gwfp,
        stress_period_data=[[(0, i, 0), h_left] for i in range(nrow)]
        + [[(0, i, ncol - 1), h_right] for i in range(nrow)],
    )
    flopy.mf6.ModflowGwfoc(
        gwfp, head_filerecord="parent.hds", saverecord=[("HEAD", "ALL")]
    )

    gwfc = flopy.mf6.ModflowGwf(sim, modelname="child", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwfc,
        nlay=child.nlay,
        nrow=child.nrow,
        ncol=child.ncol,
        delr=child.delr,
        delc=child.delc,
        top=top,
        botm=botm,
        xorigin=child.xoffset,
        yorigin=child.yoffset,
    )
    flopy.mf6.ModflowGwfic(gwfc, strt=0.5)
    flopy.mf6.ModflowGwfnpf(gwfc, icelltype=0, k=1.0)
    flopy.mf6.ModflowGwfoc(
        gwfc, head_filerecord="child.hds", saverecord=[("HEAD", "ALL")]
    )

    exchange = flopy.mf6.ModflowGwfgwf(
        sim,
        exgtype="GWF6-GWF6",
        nexg=len(exchangedata),
        exgmnamea="parent",
        exgmnameb="child",
        exchangedata=exchangedata,
        auxiliary=["angldegx", "cdist"],
    )
    if gnc:
        exchange.gnc.initialize(filename=f"{name}.gnc", **gnc_gridprops)
    return sim


# +
heads = {}
for name, gnc in [("uncorrected", False), ("gnc", True)]:
    sim = build_simulation(name, gnc=gnc)
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, f"{name} did not converge"
    ws = workspace / name
    heads[name] = (
        flopy.utils.HeadFile(ws / "parent.hds").get_data()[0],
        flopy.utils.HeadFile(ws / "child.hds").get_data()[0],
    )
    print(f"{name} converged")
# -

# ## Effect of the correction
#
# The head varies linearly between the constant head columns, so the exact solution is known and the error of each model can be measured directly.

# +
active = refine_mask[0] > 0
print(f"{'variant':14s}{'parent max error':>18s}{'child max error':>17s}")
errors = {}
for name in ("uncorrected", "gnc"):
    head_parent, head_child = heads[name]
    parent_error = np.abs(head_parent - exact(parent))[active].max()
    child_error = np.abs(head_child - exact(child)).max()
    errors[name] = max(parent_error, child_error)
    print(f"{name:14s}{parent_error:18.3e}{child_error:17.3e}")
# -

# The correction removes the error the exchange introduces. What is left is at the level of the solver tolerance, so the linear head field is reproduced exactly once the ghost nodes are applied.

assert errors["gnc"] < 1.0e-6

# The error in the uncorrected model is largest along the boundary of the child model, which is where the ghost nodes are.

# +
error = np.abs(heads["uncorrected"][0] - exact(parent))
error[~active] = np.nan
vmax = np.nanmax(error)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, name in zip(axes, ("uncorrected", "gnc")):
    ax.set_aspect("equal")
    err = np.abs(heads[name][0] - exact(parent))
    err[~active] = np.nan
    pmv = flopy.plot.PlotMapView(modelgrid=parent, ax=ax)
    cb = pmv.plot_array(err, cmap="magma_r", vmin=0.0, vmax=vmax)
    pmv.plot_grid(colors="0.5", lw=0.3, alpha=0.5)
    ax.set_title(f"{name}, parent model")
fig.colorbar(cb, ax=axes, shrink=0.7, label="absolute head error")
# -

# Clean up the temporary workspace.

try:
    temp_dir.cleanup()
except (PermissionError, NotADirectoryError):
    pass
