"""Test functions in flopy/export/shapefile_utils.py
"""
import numpy as np
from modflow_devtools.markers import requires_pkg

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.export.shapefile_utils import shp2recarray
from flopy.utils.crs import get_shapefile_crs

from .test_grid import minimal_unstructured_grid_info, minimal_vertex_grid_info


@requires_pkg("pyproj", "pyshp", "shapely")
def test_write_grid_shapefile(
    minimal_unstructured_grid_info, minimal_vertex_grid_info, function_tmpdir
):
    import pyproj

    d = minimal_unstructured_grid_info
    delr = np.ones(10)
    delc = np.ones(10)
    crs = 26916
    shapefilename = function_tmpdir / "grid.shp"
    sg = StructuredGrid(delr=delr, delc=delc, nlay=1, crs=crs)
    sg.write_shapefile(shapefilename)
    data = shp2recarray(shapefilename)
    # check that row and column appear as integers in recarray
    assert np.issubdtype(data.dtype["row"], np.integer)
    assert np.issubdtype(data.dtype["column"], np.integer)
    assert len(data) == sg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    usg = UnstructuredGrid(**d, crs=crs)
    usg.write_shapefile(shapefilename)
    data = shp2recarray(shapefilename)
    assert len(data) == usg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    d = minimal_vertex_grid_info
    vg = VertexGrid(**d, crs=crs)
    vg.write_shapefile(shapefilename)
    data = shp2recarray(shapefilename)
    assert len(data) == vg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs
