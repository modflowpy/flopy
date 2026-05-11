import numpy as np
from modflow_devtools.markers import requires_exe

import flopy
from flopy.utils.hfb_util import make_hfb_array
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid


def structured_sim():
    lx = 100
    ly = 100
    nlay = 1
    nrow = 10
    ncol = 10
    delc = np.full((nrow,), ly / nrow)
    delr = np.full((ncol,), lx / ncol)
    top = np.full((nrow, ncol), 10)
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)

    sim = flopy.mf6.MFSimulation(sim_ws="tmp_struct")
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="hfb_model")
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
    )

    return sim


def vertex_sim(path):
    geom = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    tri = Triangle(angle=30, model_ws=path)
    tri.add_polygon(geom)
    tri.add_region((5, 5), 0, maximum_area=40)
    tri.build()

    vor = VoronoiGrid(tri)
    pkg_props = vor.get_disv_gridprops()
    ncpl = pkg_props["ncpl"]
    nlay = 1
    top = np.full((ncpl,), 10)
    botm = np.zeros((nlay, ncpl))
    idomain = np.ones(botm.shape, dtype=int)

    sim = flopy.mf6.MFSimulation(sim_ws="tmp_struct")
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="hfb_model")
    disv = flopy.mf6.ModflowGwfdisv(
        gwf, nlay=1, top=top, botm=botm, idomain=idomain, **pkg_props
    )

    return sim


def unstructured_sim(path):
    geom = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    tri = Triangle(angle=30, model_ws=path)
    tri.add_polygon(geom)
    tri.add_region((5, 5), 0, maximum_area=40)
    tri.build()

    vor = VoronoiGrid(tri)
    grid_props = vor.get_disu6_gridprops()

    top = np.full((grid_props["nodes"],), 10)
    botm = np.zeros((grid_props["nodes"]))
    idomain = np.ones(botm.shape, dtype=int)

    sim = flopy.mf6.MFSimulation(sim_ws="tmp_struct")
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="hfb_model")
    disu = flopy.mf6.ModflowGwfdisu(
        gwf, top=top, bot=botm, idomain=idomain, **grid_props
    )

    return sim


def test_simple_structured():
    validation = [
        (0, 0, 8, 0, 1, 8),
        (0, 0, 8, 0, 0, 9),
        (0, 1, 7, 0, 2, 7),
        (0, 1, 7, 0, 1, 8),
        (0, 2, 6, 0, 3, 6),
        (0, 2, 6, 0, 2, 7),
        (0, 3, 5, 0, 4, 5),
        (0, 3, 5, 0, 3, 6),
        (0, 4, 4, 0, 4, 5),
        (0, 4, 4, 0, 5, 4),
        (0, 5, 3, 0, 5, 4),
        (0, 5, 3, 0, 6, 3),
        (0, 6, 2, 0, 6, 3),
        (0, 6, 2, 0, 7, 2),
        (0, 7, 1, 0, 7, 2),
        (0, 7, 1, 0, 8, 1),
        (0, 8, 0, 0, 8, 1),
        (0, 8, 0, 0, 9, 0),
    ]

    fault = [(0, 4), (94.0, 100)]
    sim = structured_sim()
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


@requires_exe("triangle")
def test_simple_vertex(function_tmpdir):
    validation = [
        (0, 53, 0, 92),
        (0, 65, 0, 162),
        (0, 64, 0, 65),
        (0, 69, 0, 71),
        (0, 64, 0, 72),
        (0, 45, 0, 77),
        (0, 53, 0, 77),
        (0, 78, 0, 79),
        (0, 76, 0, 78),
        (0, 78, 0, 162),
        (0, 79, 0, 82),
        (0, 11, 0, 79),
        (0, 69, 0, 80),
        (0, 45, 0, 80),
        (0, 11, 0, 84),
        (0, 69, 0, 84),
        (0, 94, 0, 114),
        (0, 4, 0, 94),
        (0, 4, 0, 96),
        (0, 97, 0, 101),
        (0, 97, 0, 99),
        (0, 98, 0, 99),
        (0, 96, 0, 98),
        (0, 108, 0, 130),
        (0, 108, 0, 109),
        (0, 101, 0, 108),
        (0, 92, 0, 114),
        (0, 107, 0, 130),
        (0, 125, 0, 130),
        (0, 125, 0, 132),
        (0, 149, 0, 171),
        (0, 133, 0, 149),
        (0, 150, 0, 151),
        (0, 151, 0, 171),
        (0, 150, 0, 154),
        (0, 125, 0, 169),
        (0, 119, 0, 169),
        (0, 149, 0, 169),
    ]

    fault = [(0, 4), (94.0, 100)]
    sim = vertex_sim(function_tmpdir)
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


@requires_exe("triangle")
def test_simple_unstructured(function_tmpdir):
    validation = [
        (53, 92),
        (65, 162),
        (64, 65),
        (69, 71),
        (64, 72),
        (45, 77),
        (53, 77),
        (78, 79),
        (76, 78),
        (78, 162),
        (79, 82),
        (11, 79),
        (69, 80),
        (45, 80),
        (11, 84),
        (69, 84),
        (94, 114),
        (4, 94),
        (4, 96),
        (97, 101),
        (97, 99),
        (98, 99),
        (96, 98),
        (108, 130),
        (108, 109),
        (101, 108),
        (92, 114),
        (107, 130),
        (125, 130),
        (125, 132),
        (149, 171),
        (133, 149),
        (150, 151),
        (151, 171),
        (150, 154),
        (125, 169),
        (119, 169),
        (149, 169),
    ]
    fault = [(0, 4), (94.0, 100)]
    sim = unstructured_sim(function_tmpdir)
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


def test_multisegment_structured():
    validation = [
        (0, 0, 8, 0, 1, 8),
        (0, 0, 8, 0, 0, 9),
        (0, 1, 7, 0, 1, 8),
        (0, 1, 7, 0, 2, 7),
        (0, 2, 6, 0, 2, 7),
        (0, 3, 6, 0, 4, 6),
        (0, 3, 6, 0, 3, 7),
        (0, 4, 5, 0, 4, 6),
        (0, 4, 5, 0, 5, 5),
        (0, 5, 4, 0, 5, 5),
        (0, 6, 3, 0, 7, 3),
        (0, 6, 3, 0, 6, 4),
        (0, 5, 4, 0, 6, 4),
        (0, 7, 1, 0, 8, 1),
        (0, 6, 2, 0, 7, 2),
        (0, 7, 1, 0, 7, 2),
        (0, 8, 0, 0, 9, 0),
        (0, 8, 0, 0, 8, 1),
    ]

    fault = [[0, 11], [55, 45], [89, 100]]
    sim = structured_sim()
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


@requires_exe("triangle")
def test_multisegment_vertex(function_tmpdir):
    validation = [
        (0, 11, 0, 84),
        (0, 11, 0, 79),
        (0, 40, 0, 103),
        (0, 4, 0, 40),
        (0, 40, 0, 114),
        (0, 40, 0, 53),
        (0, 45, 0, 53),
        (0, 45, 0, 77),
        (0, 45, 0, 80),
        (0, 29, 0, 53),
        (0, 64, 0, 72),
        (0, 64, 0, 65),
        (0, 69, 0, 80),
        (0, 69, 0, 71),
        (0, 69, 0, 84),
        (0, 78, 0, 79),
        (0, 76, 0, 78),
        (0, 78, 0, 162),
        (0, 79, 0, 82),
        (0, 97, 0, 110),
        (0, 97, 0, 102),
        (0, 98, 0, 102),
        (0, 98, 0, 103),
        (0, 4, 0, 103),
        (0, 107, 0, 130),
        (0, 107, 0, 108),
        (0, 108, 0, 110),
        (0, 125, 0, 132),
        (0, 125, 0, 130),
        (0, 133, 0, 149),
        (0, 150, 0, 151),
        (0, 150, 0, 154),
        (0, 65, 0, 162),
        (0, 125, 0, 169),
        (0, 119, 0, 169),
        (0, 149, 0, 169),
        (0, 149, 0, 171),
        (0, 151, 0, 171),
    ]

    fault = [[0, 11], [55, 45], [89, 100]]
    sim = vertex_sim(function_tmpdir)
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


def test_colinear_hfb():
    validation = [
        (0, 0, 4, 0, 0, 5),
        (0, 1, 4, 0, 1, 5),
        (0, 2, 4, 0, 2, 5),
        (0, 3, 4, 0, 3, 5),
        (0, 4, 4, 0, 4, 5),
        (0, 5, 4, 0, 5, 5),
        (0, 6, 4, 0, 6, 5),
        (0, 7, 4, 0, 7, 5),
        (0, 8, 4, 0, 8, 5),
        (0, 9, 4, 0, 9, 5),
    ]

    fault = [(50, 0), (50, 100)]
    sim = structured_sim()
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )


def test_split_cell_hfb():
    validation = [
        (0, 4, 0, 0, 5, 0),
        (0, 4, 1, 0, 5, 1),
        (0, 4, 2, 0, 5, 2),
        (0, 4, 3, 0, 5, 3),
        (0, 4, 4, 0, 5, 4),
        (0, 4, 5, 0, 5, 5),
        (0, 4, 6, 0, 5, 6),
        (0, 4, 7, 0, 5, 7),
        (0, 4, 8, 0, 5, 8),
        (0, 4, 9, 0, 5, 9),
    ]

    fault = [(0, 55), (100, 55)]
    sim = structured_sim()
    gwf = sim.get_model()
    modelgrid = gwf.modelgrid
    hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
    for row in hfbs:
        cid1 = row.cellid1
        cid2 = row.cellid2
        x = sorted([cid1, cid2])
        test_set = x[0] + x[1]
        if test_set not in validation:
            raise AssertionError(
                f"HFB Line {x[0]} and {x[1]} are outside of validation data set"
            )

    if len(hfbs) != len(validation):
        raise AssertionError(
            f"HFB data length {len(hfbs)} not equal to validation {len(validation)}"
        )
