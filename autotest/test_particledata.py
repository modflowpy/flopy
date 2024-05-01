from functools import reduce
from itertools import chain
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from autotest.test_grid_cases import GridCases
from flopy.discretization import StructuredGrid
from flopy.mf6.modflow.mfsimulation import MFSimulation
from flopy.modflow.mf import Modflow
from flopy.modflow.mfdis import ModflowDis
from flopy.modpath import (
    CellDataType,
    FaceDataType,
    LRCParticleData,
    Modpath7,
    Modpath7Bas,
    Modpath7Sim,
    NodeParticleData,
    ParticleData,
    ParticleGroupLRCTemplate,
    ParticleGroupNodeTemplate,
)
from flopy.modpath.mp7particlegroup import ParticleGroup
from flopy.utils.modpathfile import EndpointFile, PathlineFile

# utilities


def get_nn(grid: StructuredGrid, k, i, j):
    return k * grid.nrow * grid.ncol + i * grid.ncol + j


def flatten(a):
    return [
        [
            *chain.from_iterable(
                xx if isinstance(xx, tuple) else [xx] for xx in x
            )
        ]
        for x in a
    ]


# test initializers


structured_dtype = np.dtype(
    [
        ("k", "<i4"),
        ("i", "<i4"),
        ("j", "<i4"),
        ("localx", "<f4"),
        ("localy", "<f4"),
        ("localz", "<f4"),
        ("timeoffset", "<f4"),
        ("drape", "<i4"),
    ]
)
unstructured_dtype = np.dtype(
    [
        ("node", "<i4"),
        ("localx", "<f4"),
        ("localy", "<f4"),
        ("localz", "<f4"),
        ("timeoffset", "<f4"),
        ("drape", "<i4"),
    ]
)


def test_particledata_structured_ctor_with_partlocs_as_list_of_tuples():
    locs = [(0, 1, 1), (0, 1, 2)]
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 1, 1, 0.5, 0.5, 0.5, 0.0, 0),
                (0, 1, 2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=structured_dtype,
        ),
    )


def test_particledata_structured_ctor_with_partlocs_as_ndarray():
    locs = np.array([(0, 1, 1), (0, 1, 2)])
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 1, 1, 0.5, 0.5, 0.5, 0.0, 0),
                (0, 1, 2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=structured_dtype,
        ),
    )


def test_particledata_unstructured_ctor_with_partlocs_as_ndarray():
    locs = np.array([0, 1, 2])
    data = ParticleData(partlocs=locs, structured=False)

    assert data.particlecount == 3
    assert data.dtype == unstructured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 0.5, 0.5, 0.5, 0.0, 0),
                (1, 0.5, 0.5, 0.5, 0.0, 0),
                (2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=unstructured_dtype,
        ),
    )


def test_particledata_unstructured_ctor_with_partlocs_as_list():
    locs = [0, 1, 2]
    data = ParticleData(partlocs=locs, structured=False)

    assert data.particlecount == 3
    assert data.dtype == unstructured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 0.5, 0.5, 0.5, 0.0, 0),
                (1, 0.5, 0.5, 0.5, 0.0, 0),
                (2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=unstructured_dtype,
        ),
    )


def test_particledata_unstructured_ctor_with_partlocs_as_ndarray():
    locs = np.array([0, 1, 2])
    data = ParticleData(partlocs=locs, structured=False)

    assert data.particlecount == 3
    assert data.dtype == unstructured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 0.5, 0.5, 0.5, 0.0, 0),
                (1, 0.5, 0.5, 0.5, 0.0, 0),
                (2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=unstructured_dtype,
        ),
    )


def test_particledata_structured_ctor_with_partlocs_as_list_of_lists():
    locs = [list(p) for p in [(0, 1, 1), (0, 1, 2)]]
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert isinstance(data.particledata, pd.DataFrame)
    assert np.array_equal(
        data.particledata.to_records(index=False),
        np.core.records.fromrecords(
            [
                (0, 1, 1, 0.5, 0.5, 0.5, 0.0, 0),
                (0, 1, 2, 0.5, 0.5, 0.5, 0.0, 0),
            ],
            dtype=structured_dtype,
        ),
    )


# test to_prp()


def test_particledata_to_prp_dis_1():
    # model grid
    grid = GridCases().structured_small()

    # particle data
    cells = [(0, 1, 1), (0, 1, 2)]
    part_data = ParticleData(partlocs=cells, structured=True)

    # convert to global coordinates
    rpts_prt = flatten(list(part_data.to_prp(grid)))

    # check conversion
    assert len(rpts_prt) == len(cells)
    assert all(
        len(c) == 7 for c in rpts_prt
    )  # each coord should be a tuple (irpt, k, i, j, x, y, z)

    # expected
    exp = np.core.records.fromrecords(
        [
            (0, 1, 1, 0.5, 0.5, 0.5, 0.0, 0),
            (0, 1, 2, 0.5, 0.5, 0.5, 0.0, 0),
        ],
        dtype=structured_dtype,
    )

    for ci, cell in enumerate(cells):
        # check containing cell is correct
        rpt = rpts_prt[ci]
        assert cell == grid.intersect(*rpt[4:7])

        # check global coords are equivalent to local
        k, i, j = cell
        verts = grid.get_cell_vertices(i, j)
        xs, ys = list(zip(*verts))
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        minz, maxz = grid.botm[k, i, j], grid.top[i, j]
        assert np.isclose(rpt[4], minx + (exp[ci][3] * (maxx - minx)))
        assert np.isclose(rpt[5], miny + (exp[ci][4] * (maxy - miny)))
        assert np.isclose(rpt[6], minz + (exp[ci][5] * (maxz - minz)))


def test_particledata_to_prp_dis_9():
    # minimal structured grid
    grid = GridCases().structured_small()

    # release points in mp7 format (using cell-local coordinates)
    rpts_mp7 = [
        # k, i, j, localx, localy, localz
        # (0-based indexing converted to 1-based for mp7 by flopy)
        (0, 0, 0, float(f"0.{i + 1}"), float(f"0.{i + 1}"), 0.5)
        for i in range(9)
    ]

    # create particle data
    part_data = ParticleData(
        partlocs=[p[:3] for p in rpts_mp7],
        structured=True,
        localx=[p[3] for p in rpts_mp7],
        localy=[p[4] for p in rpts_mp7],
        localz=[p[5] for p in rpts_mp7],
        timeoffset=0,
        drape=0,
    )

    # expected release points in PRT format
    rpts_exp = [
        # release point index, k, i, j, x, y,
        [0, 0, 0, 0, 0.10000000149011612, 1.1000000014901161, 7.5],
        [1, 0, 0, 0, 0.20000000298023224, 1.2000000029802322, 7.5],
        [2, 0, 0, 0, 0.30000001192092896, 1.300000011920929, 7.5],
        [3, 0, 0, 0, 0.4000000059604645, 1.4000000059604645, 7.5],
        [4, 0, 0, 0, 0.5, 1.5, 7.5],
        [5, 0, 0, 0, 0.6000000238418579, 1.600000023841858, 7.5],
        [6, 0, 0, 0, 0.699999988079071, 1.699999988079071, 7.5],
        [7, 0, 0, 0, 0.800000011920929, 1.800000011920929, 7.5],
        [8, 0, 0, 0, 0.8999999761581421, 1.899999976158142, 7.5],
    ]

    # convert to prt format
    rpts_prt = flatten(list(part_data.to_prp(grid)))
    assert np.allclose(rpts_prt, rpts_exp, atol=1e-3)


@pytest.mark.parametrize("lx", [None, 0.5, 0.25])  # local x coord
@pytest.mark.parametrize("ly", [None, 0.5, 0.25])  # local y coord
@pytest.mark.parametrize(
    "localz", [False, True]
)  # whether to return local z coords
def test_particledata_to_prp_disv_1(lx, ly, localz):
    """
    1 particle in bottom left cell, testing with default
    location (middle), explicitly specifying middle, and
    offset in x and y directions. Also test the `localz`
    switch, which determines whether to return local z
    coordinates (on interval [0, 1]) rather than global.
    """

    # model grid
    grid = GridCases().vertex_small()

    # particle data
    locs = [4]
    lx = [lx] if lx else None
    ly = [ly] if ly else None
    part_data = ParticleData(
        partlocs=locs,
        structured=False,
        particleids=range(len(locs)),
        localx=lx,
        localy=ly,
    )

    # convert to global coordinates
    rpts_prt = flatten(list(part_data.to_prp(grid, localz=localz)))

    # check conversion succeeded
    assert len(rpts_prt) == len(locs)
    assert all(
        len(c) == 6 for c in rpts_prt
    )  # each coord should be a tuple (irpt, k, j, x, y, z)
    for rpt in rpts_prt:
        assert np.isclose(rpt[3], lx[0] if lx else 0.5)  # check x
        assert np.isclose(rpt[4], ly[0] if ly else 0.5)  # check y
        assert np.isclose(rpt[5], 0.5 if localz else 7.5)  # check z

    # debugging: plot grid, cell centers, and particle location
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # ax.set_aspect("equal")
    # grid.plot() # plot grid
    # xc, yc = ( # plot cell centers
    #     grid.get_xcellcenters_for_layer(0),
    #     grid.get_ycellcenters_for_layer(0)
    # )
    # xc = xc.flatten()
    # yc = yc.flatten()
    # for i in range(grid.ncpl):
    #     x, y = xc[i], yc[i]
    #     nn = grid.intersect(x, y, 0)[1]
    #     ax.plot(x, y, "ro")
    #     ax.annotate(str(nn + 1), (x, y), color="r") # 1-based node numbering
    # for c in coords:  # plot particle location(s)
    #     ax.plot(c[0], c[1], "bo")
    # plt.show()


@pytest.mark.parametrize(
    "localz", [False, True]
)  # whether to return local z coords
def test_particledata_to_prp_disv_9(localz):
    # minimal vertex grid
    grid = GridCases().vertex_small()

    # release points in mp7 format (using cell-local coordinates)
    rpts_mp7 = [
        # node number, localx, localy, localz
        # (0-based indexing converted to 1-based for mp7 by flopy)
        (0, float(f"0.{i + 1}"), float(f"0.{i + 1}"), 0.5)
        for i in range(9)
    ]

    # create particle data
    part_data = ParticleData(
        partlocs=[p[0] for p in rpts_mp7],
        structured=False,
        localx=[p[1] for p in rpts_mp7],
        localy=[p[2] for p in rpts_mp7],
        localz=[p[3] for p in rpts_mp7],
        timeoffset=0,
        drape=0,
    )

    # expected release points in PRT format; below we will use flopy
    # to convert from mp7 to prt format and make sure they are equal
    rpts_exp = [
        # particle index, k, j, x, y, z
        # (0-based indexing converted to 1-based for mf6 by flopy)
        (
            i,
            0,
            0,
            float(f"0.{i + 1}"),
            float(f"2.{i + 1}"),
            0.5 if localz else ((grid.xyzextent[5] - grid.xyzextent[4]) / 2),
        )
        for i in range(9)
    ]

    # convert to prt format
    rpts_prt = flatten(list(part_data.to_prp(grid, localz=localz)))
    assert np.allclose(rpts_prt, rpts_exp, atol=1e-3)


@pytest.mark.parametrize(
    "localz", [False, True]
)  # whether to return local z coords
def test_lrcparticledata_to_prp_divisions_defaults(localz, array_snapshot):
    sd_data = CellDataType()
    regions = [[0, 0, 1, 0, 1, 1]]
    part_data = LRCParticleData(
        subdivisiondata=[sd_data], lrcregions=[regions]
    )
    grid = GridCases().structured_small()
    rpts_prt = flatten(list(part_data.to_prp(grid, localz=localz)))
    num_cells = reduce(
        sum,
        [
            (lrc[3] - lrc[0] + 1)
            * (lrc[4] - lrc[1] + 1)
            * (lrc[5] - lrc[2] + 1)
            for lrc in regions
        ],
    )
    act_len = len(rpts_prt)
    exp_len = (
        num_cells
        * sd_data.rowcelldivisions
        * sd_data.columncelldivisions
        * sd_data.layercelldivisions
    )
    assert act_len == exp_len
    assert rpts_prt == array_snapshot


def test_lrcparticledata_to_prp_divisions_custom():
    # create particle data
    rel_minl = 0
    rel_maxl = 2
    rel_minr = 2
    rel_maxr = 3
    rel_minc = 2
    rel_maxc = 3
    cell_data = flopy.modpath.CellDataType(
        drape=0,
        rowcelldivisions=5,
        columncelldivisions=5,
        layercelldivisions=1,
    )
    regions = [[rel_minl, rel_minr, rel_minc, rel_maxl, rel_maxr, rel_maxc]]
    part_data = flopy.modpath.LRCParticleData(
        subdivisiondata=[cell_data],
        lrcregions=[regions],
    )

    # create grid
    nlay, nrow, ncol = 3, 21, 20
    delr = delc = 500.0
    top = 350.0
    botm = [220.0, 200.0, 0.0]
    m = Modflow()
    dis = ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    grid = m.modelgrid
    # todo once StructuredGrid initializer fixed,
    # remove dummy model/dis and uncomment below
    # grid = StructuredGrid(
    #     delc=delc,
    #     delr=delr,
    #     top=top,
    #     botm=botm,
    #     nlay=nlay,
    #     nrow=nrow,
    #     ncol=ncol
    # )

    # convert to PRP package data
    rpts_prt = list(part_data.to_prp(grid))
    rpts_prt = pd.DataFrame(rpts_prt)
    assert len(rpts_prt) == 300

    # plot release points (debugging)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(rpts_prt[4], rpts_prt[5], rpts_prt[6])
    # plt.show()


def test_lrcparticledata_to_prp_top_bottom():
    rd = 1
    cd = 1
    sddata = FaceDataType(
        horizontaldivisions1=0,
        verticaldivisions1=0,
        horizontaldivisions2=0,
        verticaldivisions2=0,
        horizontaldivisions3=0,
        verticaldivisions3=0,
        horizontaldivisions4=0,
        verticaldivisions4=0,
        rowdivisions5=rd,
        columndivisions5=cd,
        rowdivisions6=rd,
        columndivisions6=cd,
    )
    lrcregions = [[0, 1, 1, 0, 1, 1]]
    data = LRCParticleData(subdivisiondata=[sddata], lrcregions=[lrcregions])
    grid = GridCases().structured_small()
    rpts_prt = flatten(list(data.to_prp(grid)))

    num_cells = len(
        [
            (lrc[3] - lrc[0]) * (lrc[4] - lrc[1]) * (lrc[5] - lrc[2])
            for lrc in lrcregions
        ]
    )
    assert (
        len(rpts_prt) == num_cells * rd * cd * 2
    )  # 1 particle each on top and bottom faces

    # particle should be centered on each face
    verts = grid.get_cell_vertices(1, 1)
    xs, ys = list(zip(*verts))
    for coord in rpts_prt:
        assert np.isclose(coord[4], np.mean(xs))
        assert np.isclose(coord[5], np.mean(ys))

    # check elevation
    assert rpts_prt[0][6] == grid.top_botm[1, 1, 1]
    assert rpts_prt[1][6] == grid.top_botm[0, 1, 1]


def test_lrcparticledata_to_prp_1_per_face(array_snapshot):
    sddata = FaceDataType(
        horizontaldivisions1=1,
        verticaldivisions1=1,
        horizontaldivisions2=1,
        verticaldivisions2=1,
        horizontaldivisions3=1,
        verticaldivisions3=1,
        horizontaldivisions4=1,
        verticaldivisions4=1,
        rowdivisions5=1,
        columndivisions5=1,
        rowdivisions6=1,
        columndivisions6=1,
    )
    lrcregions = [[0, 1, 1, 0, 1, 1]]
    data = LRCParticleData(subdivisiondata=[sddata], lrcregions=[lrcregions])
    grid = GridCases().structured_small()
    rpts_prt = flatten(list(data.to_prp(grid)))
    num_cells = len(
        [
            (lrc[3] - lrc[0]) * (lrc[4] - lrc[1]) * (lrc[5] - lrc[2])
            for lrc in lrcregions
        ]
    )
    assert len(rpts_prt) == num_cells * 6  # 1 particle on each face
    assert rpts_prt == array_snapshot


@pytest.mark.parametrize(
    "localz", [False, True]
)  # whether to return local z coords
def test_nodeparticledata_to_prp_disv_defaults(
    function_tmpdir, example_data_path, localz
):
    """
    This test loads a GWF simulation, runs it, and feeds it to an MP7 simulation
    to get expected particle release locations. These could be hard-coded but it
    is not expensive to run the test003 model, and this way we show how expected
    values are obtained.
    """

    # create particle data
    pdat = NodeParticleData()

    # load gwf simulation, switch workspace, write input files, and run
    sim = MFSimulation.load(
        sim_ws=example_data_path / "mf6" / "test003_gwfs_disv"
    )
    gwf = sim.get_model("gwf_1")
    grid = gwf.modelgrid
    gwf_name = "gwf"
    gwf_ws = function_tmpdir / gwf_name
    gwf_ws.mkdir()
    sim.set_sim_path(gwf_ws)
    sim.write_simulation()
    sim.run_simulation()

    # create, write and run mp7 simulation to get expected start locations
    mp7_name = "mp7"
    mp7_ws = function_tmpdir / mp7_name
    mp7_ws.mkdir()
    pg = ParticleGroupNodeTemplate(
        particlegroupname="G1",
        particledata=pdat,
        filename=f"{mp7_name}.sloc",
    )
    mp = Modpath7(
        modelname=mp7_name,
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=mp7_ws,
    )
    mpbas = Modpath7Bas(
        mp,
        porosity=0.1,
    )
    mpsim = Modpath7Sim(
        mp,
        simulationtype="pathline",
        trackingdirection="forward",
        budgetoutputoption="summary",
        stoptimeoption="total",
        particlegroups=[pg],
    )
    mp.write_input()
    mp.run_model()

    # extract particle starting locations computed by mp7
    mp_pl_file = mp7_ws / f"{mp7_name}.mppth"
    assert mp_pl_file.exists()
    plf = PathlineFile(mp_pl_file)
    pldata = plf.get_alldata()
    mp7_pls = pd.concat([pd.DataFrame(ra) for ra in pldata])
    mp7_pls = mp7_pls.sort_values(by=["time", "particleid"]).head(27)
    mp7_rpts = [
        [0, r.k, r.x, r.y, r.zloc if localz else r.z]
        for r in mp7_pls.itertuples()
    ]  # omit rpt index
    mp7_rpts.sort()

    # convert particle data to prt format, flatten (remove cell ID tuples),
    # remove irpt as it is not guaranteed to match, and sort
    prt_rpts = flatten(list(pdat.to_prp(grid, localz=localz)))
    prt_rpts = [r[1:] for r in prt_rpts]
    prt_rpts.sort()
    assert np.allclose(prt_rpts, mp7_rpts)


def test_nodeparticledata_to_prp_dis_1_per_face():
    sddata = FaceDataType(
        horizontaldivisions1=1,
        verticaldivisions1=1,
        horizontaldivisions2=1,
        verticaldivisions2=1,
        horizontaldivisions3=1,
        verticaldivisions3=1,
        horizontaldivisions4=1,
        verticaldivisions4=1,
        rowdivisions5=1,
        columndivisions5=1,
        rowdivisions6=1,
        columndivisions6=1,
    )
    grid = GridCases().structured_small()
    nodes = [get_nn(grid, 0, 1, 1)]
    data = NodeParticleData(subdivisiondata=sddata, nodes=[nodes])

    rpts = flatten(list(data.to_prp(grid)))

    num_cells = len(nodes)
    assert len(rpts) == num_cells * 6


@requires_pkg("shapefile")
def test_nodeparticledata_prp_disv_big(function_tmpdir):
    Lx = 10000.0
    Ly = 10500.0
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, 0]

    ms = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    from flopy.utils.gridgen import Gridgen

    # create Gridgen workspace
    gridgen_ws = function_tmpdir
    gridgen_ws.mkdir(parents=True, exist_ok=True)

    # create Gridgen object
    g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)

    # add polygon for each refinement level
    outer_polygon = [
        [(3500, 4000), (3500, 6500), (6000, 6500), (6000, 4000), (3500, 4000)]
    ]
    g.add_refinement_features([outer_polygon], "polygon", 1, range(nlay))
    refshp0 = gridgen_ws / "rf0"

    middle_polygon = [
        [(4000, 4500), (4000, 6000), (5500, 6000), (5500, 4500), (4000, 4500)]
    ]
    g.add_refinement_features([middle_polygon], "polygon", 2, range(nlay))
    refshp1 = gridgen_ws / "rf1"

    inner_polygon = [
        [(4500, 5000), (4500, 5500), (5000, 5500), (5000, 5000), (4500, 5000)]
    ]
    g.add_refinement_features([inner_polygon], "polygon", 3, range(nlay))
    refshp2 = gridgen_ws / "rf2"

    # build the grid
    g.build(verbose=False)
    grid_props = g.get_gridprops_vertexgrid()
    disv_props = g.get_gridprops_disv()
    grid = flopy.discretization.VertexGrid(**grid_props)

    # define particle data
    coords = (4718.45, 5281.25)
    cells = grid.intersect(coords[0], coords[1], 0)
    nodew = grid.ncpl * 2 + cells[0]
    facedata = flopy.modpath.FaceDataType(
        drape=0,
        verticaldivisions1=10,
        horizontaldivisions1=10,
        verticaldivisions2=10,
        horizontaldivisions2=10,
        verticaldivisions3=10,
        horizontaldivisions3=10,
        verticaldivisions4=10,
        horizontaldivisions4=10,
        rowdivisions5=0,
        columndivisions5=0,
        rowdivisions6=4,
        columndivisions6=4,
    )
    pgdata = flopy.modpath.NodeParticleData(
        subdivisiondata=facedata, nodes=nodew
    )

    # convert to PRP package data
    rpts_prt = flatten(list(pgdata.to_prp(grid)))
    print(rpts_prt)


# test write()


def test_lrcparticledata_write(function_tmpdir):
    # create particle data
    rel_minl = 0
    rel_maxl = 2
    rel_minr = 2
    rel_maxr = 3
    rel_minc = 2
    rel_maxc = 3
    cell_data = flopy.modpath.CellDataType(
        drape=0,
        rowcelldivisions=5,
        columncelldivisions=5,
        layercelldivisions=1,
    )
    regions = [[rel_minl, rel_minr, rel_minc, rel_maxl, rel_maxr, rel_maxc]]
    part_data = flopy.modpath.LRCParticleData(
        subdivisiondata=[cell_data],
        lrcregions=[regions],
    )

    # write to a file
    p = function_tmpdir / "f.txt"
    with open(p, "w") as f:
        part_data.write(f)

    # check lines written
    lines = open(p).readlines()
    assert lines == ["1 1\n", "2 1 0\n", " 5 5 1\n", "1 3 3 3 4 4 \n"]


# To make it easier to compare MODFLOW 6 PRT and MODPATH 7 results,
# we want to_coords() and to_prp() to return release configurations
# in the same order as is generated by MODPATH 7. That is, an input
# file for MF6 PRT's PRP package should list particle release points
# in the same order that MODPATH 7 assigns particle IDs upon release
# from any input style. The tests below set up bare-bones MP7 models
# in endpoint mode and compare against their release points.


@pytest.fixture
def mf6_sim(function_tmpdir):
    name = "tiny-gwf"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=function_tmpdir, exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=1, ncol=1)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.0]])
    budget_file = name + ".bud"
    head_file = name + ".hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def get_mp7_sim(mf6_sim, groups):
    mp = Modpath7(
        modelname=f"{mf6_sim.name}_mp7",
        flowmodel=mf6_sim.get_model(),
        exe_name="mp7",
        model_ws=mf6_sim.sim_path,
    )
    mpbas = Modpath7Bas(mp)
    mpsim = Modpath7Sim(
        mp,
        simulationtype="endpoint",
        trackingdirection="forward",
        particlegroups=groups,
    )
    return mp


@requires_exe("mp7")
def test_lrcparticledata_celldatatype_to_coords_order(mf6_sim):
    mf6_sim.write_simulation()
    success, buff = mf6_sim.run_simulation()
    assert success, pformat(buff)

    pdata = flopy.modpath.LRCParticleData()
    pg = ParticleGroupLRCTemplate(particlegroupname="PG1", particledata=pdata)
    mp7_sim = get_mp7_sim(mf6_sim, [pg])
    mp7_sim.write_input()
    success, buff = mp7_sim.run_model()
    assert success, pformat(buff)

    gwf = mf6_sim.get_model()
    grid = gwf.modelgrid
    ep_file = EndpointFile(mf6_sim.sim_path / f"{mp7_sim.name}.mpend")
    expected = ep_file.get_destination_endpoint_data(range(grid.nnodes))[
        ["x0", "y0", "z0"]
    ].tolist()
    actual = list(pdata.to_coords(grid))
    assert len(expected) == len(actual) == 27
    assert np.allclose(expected, actual)


def test_lrcparticledata_facedatatype_to_coords_order(mf6_sim):
    mf6_sim.write_simulation()
    success, buff = mf6_sim.run_simulation()
    assert success, pformat(buff)

    pdata = flopy.modpath.LRCParticleData(
        subdivisiondata=[FaceDataType()],
    )
    pg = ParticleGroupLRCTemplate(particlegroupname="PG1", particledata=pdata)
    mp7_sim = get_mp7_sim(mf6_sim, [pg])
    mp7_sim.write_input()
    success, buff = mp7_sim.run_model()
    assert success, pformat(buff)

    gwf = mf6_sim.get_model()
    grid = gwf.modelgrid
    ep_file = EndpointFile(mf6_sim.sim_path / f"{mp7_sim.name}.mpend")
    expected = ep_file.get_destination_endpoint_data(range(grid.nnodes))[
        ["x0", "y0", "z0"]
    ].tolist()
    actual = list(pdata.to_coords(grid))
    assert len(expected) == len(actual) == 54
    assert np.allclose(expected, actual)
