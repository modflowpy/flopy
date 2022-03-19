"""
Test vtk export_model function without packages_names definition
"""

from ci_framework import FlopyTestSetup, base_test_dir
import os
import numpy as np
import flopy
from flopy.utils import import_optional_dependency

mf_exe_name = "mf6"
base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)
test_setup = FlopyTestSetup(verbose=True, test_dirs=base_dir)

def test_vtk_export_model_without_packages_names():
    dir_name = os.path.join(base_dir, "test_0")
    name = "mymodel"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=dir_name, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=3, ncol=3)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 2, 2), 0.0]]
    )

    # Export model without specifying packages_names parameter
    vtk = import_optional_dependency("vtk")
    if vtk is not None:

        from vtk.util.numpy_support import vtk_to_numpy

        # create the vtk output
        gwf = sim.get_model()
        vtkobj = flopy.export.vtk.Vtk(gwf, binary=False)
        vtkobj.add_model(gwf)
        f = os.path.join(dir_name, "gwf.vtk")
        vtkobj.write(f)

        # load the output using the vtk standard library
        f = os.path.join(dir_name, "gwf_000000.vtk")
        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(f)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"Found cell locations {cell_locations} in vtk file.")
        print(f"Expecting cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(9 * [42])
        print(f"Found cell types {cell_types} in vtk file.")
        print(f"Expecting cell types {cell_types_answer}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg

    # If the function executes without error then test was successful
    assert True


def test_vtk_export_disv1_model():
    dir_name = os.path.join(base_dir, "test_1")
    name = "mymodel"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=dir_name, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)

    nlay, nrow, ncol = 1, 3, 3
    from flopy.discretization import StructuredGrid
    mg = StructuredGrid(delc=np.array(nrow * [1]), delr=np.array(ncol * [1]),
                        top=np.zeros((nrow, ncol)), botm=np.zeros((nlay, nrow, ncol)) - 1,
                        idomain=np.ones((nlay, nrow, ncol)))

    from flopy.utils.cvfdutil import gridlist_to_disv_gridprops
    gridprops = gridlist_to_disv_gridprops([mg])
    gridprops["top"] = 0
    gridprops["botm"] = np.zeros((nlay, nrow*ncol), dtype=float) - 1
    gridprops["nlay"] = nlay

    disv = flopy.mf6.ModflowGwfdisv(gwf, **gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=10)
    npf = flopy.mf6.ModflowGwfnpf(gwf)

    # Export model without specifying packages_names parameter
    vtk = import_optional_dependency("vtk")
    if vtk is not None:

        from vtk.util.numpy_support import vtk_to_numpy

        # create the vtk output
        gwf = sim.get_model()
        vtkobj = flopy.export.vtk.Vtk(gwf, binary=False)
        vtkobj.add_model(gwf)
        f = os.path.join(dir_name, "gwf.vtk")
        vtkobj.write(f)

        # load the output using the vtk standard library
        f = os.path.join(dir_name, "gwf.vtk")
        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(f)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)
        #print(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"Found cell locations {cell_locations} in vtk file.")
        print(f"Expecting cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(9 * [42])
        print(f"Found cell types {cell_types} in vtk file.")
        print(f"Expecting cell types {cell_types_answer}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg

    # If the function executes without error then test was successful
    assert True


def grid2disvgrid(nrow, ncol):
    """Simple function to create disv verts and iverts for a regular grid of size nrow, ncol"""
    def lower_left_point(i, j, ncol):
        return i * (ncol + 1) + j

    mg = np.meshgrid(np.linspace(0, ncol, ncol + 1), np.linspace(0, nrow, nrow + 1))
    verts = np.vstack((mg[0].flatten(), mg[1].flatten())).transpose()

    # in the creation of iverts here, we intentionally do not close the cell polygon
    iverts = []
    for i in range(nrow):
        for j in range(ncol):
            iv_cell = []
            iv_cell.append(lower_left_point(i, j, ncol))
            iv_cell.append(lower_left_point(i, j + 1, ncol))
            iv_cell.append(lower_left_point(i + 1, j + 1, ncol))
            iv_cell.append(lower_left_point(i + 1, j, ncol))
            iverts.append(iv_cell)
    return verts, iverts


def test_vtk_export_disv2_model():
    # in this case, test for iverts that do not explicitly close the cell polygons
    dir_name = os.path.join(base_dir, "test_2")
    name = "mymodel"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=dir_name, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)

    nlay, nrow, ncol = 1, 3, 3
    verts, iverts = grid2disvgrid(3, 3)
    from flopy.utils.cvfdutil import get_disv_gridprops
    gridprops = get_disv_gridprops(verts, iverts)

    gridprops["top"] = 0
    gridprops["botm"] = np.zeros((nlay, nrow*ncol), dtype=float) - 1
    gridprops["nlay"] = nlay

    disv = flopy.mf6.ModflowGwfdisv(gwf, **gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=10)
    npf = flopy.mf6.ModflowGwfnpf(gwf)

    # Export model without specifying packages_names parameter
    vtk = import_optional_dependency("vtk")
    if vtk is not None:

        from vtk.util.numpy_support import vtk_to_numpy

        # create the vtk output
        gwf = sim.get_model()
        vtkobj = flopy.export.vtk.Vtk(gwf, binary=False)
        vtkobj.add_model(gwf)
        f = os.path.join(dir_name, "gwf.vtk")
        vtkobj.write(f)

        # load the output using the vtk standard library
        f = os.path.join(dir_name, "gwf.vtk")
        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(f)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)
        #print(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"Found cell locations {cell_locations} in vtk file.")
        print(f"Expecting cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(9 * [42])
        print(f"Found cell types {cell_types} in vtk file.")
        print(f"Expecting cell types {cell_types_answer}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg

    # If the function executes without error then test was successful
    assert True

def load_verts(fname):
    verts = np.genfromtxt(
        fname, dtype=[int, float, float], names=["iv", "x", "y"]
    )
    verts["iv"] -= 1  # zero based
    return verts


def load_iverts(fname, closed=False):
    f = open(fname, "r")
    iverts = []
    xc = []
    yc = []
    for line in f:
        ll = line.strip().split()
        if not closed:
            iverts.append([int(i) - 1 for i in ll[4:-1]])
        else:
            iverts.append([int(i) - 1 for i in ll[4:]])
        xc.append(float(ll[1]))
        yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


def test_vtk_export_disu1_grid():
    # test exporting open cell vertices
    dir_name = os.path.join(base_dir, "test_3")

    # load vertices
    u_data_ws = os.path.join("..", "examples", "data", "unstructured")
    fname = os.path.join(u_data_ws, "ugrid_verts.dat")
    verts = load_verts(fname)

    # load the index list into iverts, xc, and yc
    fname = os.path.join(u_data_ws, "ugrid_iverts.dat")
    iverts, xc, yc = load_iverts(fname)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones((nnodes),)
    botm = np.ones((nnodes),)

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    # create the modelgrid
    modelgrid = flopy.discretization.UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )

    # export grid
    vtk = import_optional_dependency("vtk")
    if vtk is not None:

        from vtk.util.numpy_support import vtk_to_numpy

        outfile = os.path.join(dir_name, "disu_grid.vtu")
        vtkobj = flopy.export.vtk.Vtk(
            modelgrid=modelgrid, vertical_exageration=2, binary=True, smooth=False
        )
        vtkobj.add_array(modelgrid.top, "top")
        vtkobj.add_array(modelgrid.botm, "botm")
        vtkobj.write(outfile)

        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(outfile)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"Found cell locations {cell_locations} in vtk file.")
        print(f"Expecting cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(654 * [42])
        print(f"Found cell types {cell_types[0:9]} in vtk file.")
        print(f"Expecting cell types {cell_types_answer[0:9]}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg


def test_vtk_export_disu2_grid():
    # test exporting closed cell vertices
    dir_name = os.path.join(base_dir, "test_4")

    # load vertices
    u_data_ws = os.path.join("..", "examples", "data", "unstructured")
    fname = os.path.join(u_data_ws, "ugrid_verts.dat")
    verts = load_verts(fname)

    # load the index list into iverts, xc, and yc
    fname = os.path.join(u_data_ws, "ugrid_iverts.dat")
    iverts, xc, yc = load_iverts(fname, closed=True)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones((nnodes),)
    botm = np.ones((nnodes),)

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    # create the modelgrid
    modelgrid = flopy.discretization.UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )

    # export grid
    vtk = import_optional_dependency("vtk")
    if vtk is not None:

        from vtk.util.numpy_support import vtk_to_numpy

        outfile = os.path.join(dir_name, "disu_grid.vtu")
        vtkobj = flopy.export.vtk.Vtk(
            modelgrid=modelgrid, vertical_exageration=2, binary=True, smooth=False
        )
        vtkobj.add_array(modelgrid.top, "top")
        vtkobj.add_array(modelgrid.botm, "botm")
        vtkobj.write(outfile)

        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(outfile)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"Found cell locations {cell_locations} in vtk file.")
        print(f"Expecting cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(654 * [42])
        print(f"Found cell types {cell_types[0:9]} in vtk file.")
        print(f"Expecting cell types {cell_types_answer[0:9]}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg


def test_vtk_export_disu_model():
    from flopy.utils.gridgen import Gridgen

    dir_name = os.path.join(base_dir, "test_5")
    name = "mymodel"

    Lx = 10000.0
    Ly = 10500.0
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, 0]

    ml5 = flopy.modflow.Modflow()
    dis5 = flopy.modflow.ModflowDis(
        ml5,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    g = Gridgen(dis5, model_ws=dir_name)

    rf0shp = os.path.join(dir_name, "rf0")
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [
        [[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
          (xmin, ymin)]]
    ]
    g.add_refinement_features(rfpoly, "polygon", 2, [0,])
    g.build(verbose=False)

    gridprops = g.get_gridprops_disu6()

    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=dir_name, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdisu(gwf, **gridprops)

    ic = flopy.mf6.ModflowGwfic(
        gwf, strt=np.random.random_sample(gwf.modelgrid.nnodes) * 350
    )
    npf = flopy.mf6.ModflowGwfnpf(
        gwf, k=np.random.random_sample(gwf.modelgrid.nnodes) * 10
    )

    # export grid
    vtk = import_optional_dependency("vtk")
    if vtk is not None:
        from vtk.util.numpy_support import vtk_to_numpy

        vtkobj = flopy.export.vtk.Vtk(gwf, binary=False)
        vtkobj.add_model(gwf)
        f = os.path.join(dir_name, "gwf.vtk")
        vtkobj.write(f)

        # load the output using the vtk standard library
        f = os.path.join(dir_name, "gwf.vtk")
        gridreader = vtk.vtkUnstructuredGridReader()
        gridreader.SetFileName(f)
        gridreader.Update()
        grid = gridreader.GetOutput()

        # get the points
        vtk_points = grid.GetPoints()
        vtk_points = vtk_points.GetData()
        vtk_points = vtk_to_numpy(vtk_points)
        # print(vtk_points)

        # get cell locations (ia format of point to cell relationship)
        cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
        cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        print(f"First nine cell locations {cell_locations} in vtk file.")
        print(f"Expecting first nine cell locations {cell_locations_answer}")
        errmsg = f"vtk cell locations do not match expected result."
        assert np.allclose(cell_locations, cell_locations_answer), errmsg

        cell_types = vtk_to_numpy(grid.GetCellTypesArray())
        cell_types_answer = np.array(1770 * [42])
        print(f"First nine cell types {cell_types[0:9]} in vtk file.")
        print(f"Expecting fist nine cell types {cell_types_answer[0:9]}")
        errmsg = f"vtk cell types do not match expected result."
        assert np.allclose(cell_types, cell_types_answer), errmsg

        # now check that the data is consistent with that in npf and ic

        k_vtk = vtk_to_numpy(grid.GetCellData().GetArray("k"))
        if not np.allclose(gwf.npf.k.array, k_vtk):
            raise AssertionError("'k' array not written in proper node order")

        strt_vtk = vtk_to_numpy(grid.GetCellData().GetArray("strt"))
        if not np.allclose(gwf.ic.strt.array, strt_vtk):
            raise AssertionError(
                "'strt' array not written in proper node order"
            )


if __name__ == "__main__":
    test_vtk_export_model_without_packages_names()
    test_vtk_export_disv1_model()
    test_vtk_export_disv2_model()
    test_vtk_export_disu1_grid()
    test_vtk_export_disu2_grid()
    test_vtk_export_disu_model()