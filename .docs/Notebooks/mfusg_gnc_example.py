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
#       - name: Joseph Hughes
# ---

# # MODFLOW-USG: Ghost Node Correction (GNC) Data for a Quadtree Grid
#
# The control volume finite difference formulation used by MODFLOW-USG assumes that the line connecting two cell centers crosses the shared face at a right angle through the middle of the face. A quadtree grid violates that assumption wherever a coarse cell connects to a finer cell, because the shared face is offset from the center of the coarse cell. The Ghost Node Correction (GNC) Package corrects the resulting error by interpolating the head at a ghost node, which is the point in the coarse cell that does lie on the perpendicular through the middle of the face.
#
# GRIDGEN computes the ghost node data for a quadtree grid, and FloPy converts it to GNC Package input. We build a quadtree grid, create the GNC Package, and compare the corrected and uncorrected solutions.
#
# The same ghost node data can be computed from a model grid without running GRIDGEN, which is shown in the [MODFLOW 6 ghost node correction example](https://flopy.readthedocs.io/en/latest/Notebooks/gnc_example.html).

# +
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import flopy
from flopy.utils import flopy_io
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
# GRIDGEN works from a base MODFLOW grid. We use a single layer grid of 10 rows and 10 columns and refine a square in the middle of the grid by three levels, which produces cells one eighth the width of the base grid cells.

# +
nlay, nrow, ncol = 1, 10, 10
delr = delc = 1.0

base_grid = flopy.discretization.StructuredGrid(
    delr=np.full(ncol, delr, dtype=float),
    delc=np.full(nrow, delc, dtype=float),
    top=np.full((nrow, ncol), 1.0),
    botm=np.zeros((nlay, nrow, ncol)),
)

g = Gridgen(base_grid, model_ws=str(gridgen_ws))
refinement = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
g.add_refinement_features(refinement, "polygon", 3, layers=[0])
g.build(verbose=False)

disu_gridprops = g.get_gridprops_disu5()
print(f"Number of cells: {g.get_nodes()}")
# -

# ## Ghost node data
#
# GRIDGEN computes the ghost node data whenever it exports a grid and writes it to the `qtg.gnc.dat` file. The `get_gnc()` method reads that file and returns a record array with zero-based node numbers, where cell `n` contains the ghost node, cell `m` is the connecting cell, and cells `j0` and `j1` are the contributing cells whose heads are interpolated.

gnc = g.get_gnc()
print(f"Number of ghost nodes: {len(gnc)}")
print(gnc[:5])

# Two contributing cells are always written. When a ghost node has only one contributing cell, that cell is repeated and its contributing factor is halved, which MODFLOW-USG accumulates into the same matrix position. MODFLOW-USG reads a fixed number of contributing cells per record and indexes `IBOUND` with each of them, so an unused slot cannot be filled with a dummy cell number of zero the way it can in MODFLOW 6. Repeating a cell keeps every slot valid.

single = gnc["j0"] == gnc["j1"]
print(f"{single.sum()} of {len(gnc)} ghost nodes have one contributing cell")

# The contributing factors always sum to less than one, because one minus the sum is the factor applied to the head in cell `n`.

alpha = gnc["alpha0"] + gnc["alpha1"]
print(f"Contributing factors range from {alpha.min():.4f} to {alpha.max():.4f}")

# The `get_gridprops_gnc5()` method returns a dictionary that can be unpacked directly into the `MfUsgGnc` constructor. GRIDGEN writes contributing factors rather than conductances, so `iflalphan` is always 0. The `i2kn` and `isymgncn` options can be set through the method.

gnc_gridprops = g.get_gridprops_gnc5()
for key in ("numgnc", "numalphaj", "i2kn", "isymgncn", "iflalphan"):
    print(f"{key}: {gnc_gridprops[key]}")

# ## Build and run the models
#
# We build the same model with and without the GNC Package. The default `isymgncn` of 0 updates the left-hand side matrix, which makes the matrix asymmetric, so the model is solved with the complex option of the SMS Package.

# +
chdspd = []
for x, y, head in [(0.0, 10.0, 1.0), (10.0, 0.0, 0.0)]:
    node = g.intersect([(x, y)], "point", 0)["nodenumber"][0]
    chdspd.append([node, head, head])
print(f"Constant head cells: {chdspd}")


def build_model(name, gnc=False):
    m = flopy.mfusg.MfUsg(
        modelname=name,
        model_ws=str(workspace / name),
        exe_name="mfusg",
        structured=False,
    )
    flopy.mfusg.MfUsgDisU(m, **disu_gridprops)
    flopy.mfusg.MfUsgBas(m)
    flopy.mfusg.MfUsgLpf(m)
    flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
    flopy.mfusg.MfUsgSms(m, options="COMPLEX")
    flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
    if gnc:
        flopy.mfusg.MfUsgGnc(m, **gnc_gridprops)
    return m


# +
heads = {}
for name, gnc_flag in [("uncorrected", False), ("gnc", True)]:
    m = build_model(name, gnc=gnc_flag)
    m.write_input()
    success, buff = m.run_model(silent=True)
    assert success, f"{name} did not converge"
    head_file = workspace / name / f"{name}.hds"
    heads[name] = np.concatenate(flopy.utils.HeadUFile(head_file).get_data())
    print(f"{name} converged")
# -

# The GNC Package file lists the cell containing the ghost node, the connecting cell, the two contributing cells, and the two contributing factors, using one-based node numbers.

gnc_file = workspace / "gnc" / "gnc.gnc"
print("".join(gnc_file.open().readlines()[:8]))

# ## Effect of the correction
#
# The correction changes the simulated heads around the refined area, where the ghost nodes are.

# +
diff = heads["gnc"] - heads["uncorrected"]
print(f"Maximum head difference: {np.abs(diff).max():.3e}")

ugrid = flopy.discretization.UnstructuredGrid(**g.get_gridprops_unstructuredgrid())
vmax = np.abs(diff).max()

fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

ax = axes[0]
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=ugrid, ax=ax, layer=0)
cb = pmv.plot_array(heads["gnc"], cmap="jet")
pmv.plot_grid(colors="0.5", lw=0.3, alpha=0.5)
pmv.contour_array(heads["gnc"], levels=[0.2, 0.4, 0.6, 0.8], colors="white")
ax.set_title("Corrected head")
fig.colorbar(cb, ax=ax, shrink=0.7, label="head")

ax = axes[1]
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=ugrid, ax=ax, layer=0)
cb = pmv.plot_array(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
pmv.plot_grid(colors="0.5", lw=0.3, alpha=0.5)
ax.set_title("Corrected minus uncorrected head")
fig.colorbar(cb, ax=ax, shrink=0.7, label="head difference")
# -

# Clean up the temporary workspace.

try:
    temp_dir.cleanup()
except (PermissionError, NotADirectoryError):
    pass
