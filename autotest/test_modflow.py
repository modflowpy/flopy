import glob
import inspect
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from autotest.conftest import (
    excludes_platform,
    get_example_data_path,
    requires_exe,
    requires_pkg,
)
from matplotlib import pyplot as plt

from flopy.discretization import StructuredGrid
from flopy.mf6 import MFSimulation
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowDrn,
    ModflowGhb,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowRch,
    ModflowRiv,
    ModflowSfr2,
    ModflowWel,
)
from flopy.mt3d import Mt3dBtn, Mt3dms
from flopy.plot import PlotMapView
from flopy.seawat import Seawat
from flopy.utils import EndpointFile, PathlineFile, TimeseriesFile, Util2d


@pytest.fixture
def model_reference_path(example_data_path) -> Path:
    return example_data_path / "usgs.model.reference"


@pytest.fixture
def parameters_model_path(example_data_path):
    return example_data_path / "parameters"


@pytest.mark.parametrize(
    "namfile",
    [Path("freyberg") / "freyberg.nam"]
    + [
        Path("parameters") / f"{nf}.nam"
        for nf in ["Oahu_01", "twrip", "twrip_upw"]
    ],
)
def test_modflow_load(namfile, example_data_path):
    mpath = Path(example_data_path / namfile).parent
    model = Modflow.load(
        str(mpath / namfile.name),
        verbose=True,
        model_ws=str(mpath),
        check=False,
    )

    assert isinstance(model, Modflow)
    assert not model.load_fail


def test_modflow_load_when_nam_dne():
    with pytest.raises(OSError):
        Modflow.load("nonexistent.nam", check=False)


def test_mbase_modelgrid(tmpdir):
    ml = Modflow(
        modelname="test", xll=500.0, rotation=12.5, start_datetime="1/1/2016"
    )
    try:
        print(ml.modelgrid.xcentergrid)
    except:
        pass
    else:
        raise Exception("should have failed")

    dis = ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.proj4 is None
    ml.model_ws = tmpdir

    ml.write_input()
    ml1 = Modflow.load("test.nam", model_ws=ml.model_ws)
    assert str(ml1.modelgrid) == str(ml.modelgrid)
    assert ml1.start_datetime == ml.start_datetime
    assert ml1.modelgrid.proj4 is None


def test_mt_modelgrid(tmpdir):
    ml = Modflow(
        modelname="test",
        xll=500.0,
        proj4_str="epsg:2193",
        rotation=12.5,
        start_datetime="1/1/2016",
    )
    dis = ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.epsg == 2193
    assert ml.modelgrid.idomain is None
    ml.model_ws = tmpdir

    mt = Mt3dms(
        modelname="test_mt",
        modflowmodel=ml,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert mt.modelgrid.xoffset == ml.modelgrid.xoffset
    assert mt.modelgrid.yoffset == ml.modelgrid.yoffset
    assert mt.modelgrid.epsg == ml.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)

    # no modflowmodel
    swt = Seawat(
        modelname="test_swt",
        modflowmodel=None,
        mt3dmodel=None,
        model_ws=ml.model_ws,
        verbose=True,
    )
    assert swt.modelgrid is swt.dis is swt.bas6 is None

    # passing modflowmodel
    swt = Seawat(
        modelname="test_swt",
        modflowmodel=ml,
        mt3dmodel=mt,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert (
        swt.modelgrid.xoffset == mt.modelgrid.xoffset == ml.modelgrid.xoffset
    )
    assert (
        swt.modelgrid.yoffset == mt.modelgrid.yoffset == ml.modelgrid.yoffset
    )
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)

    # bas and btn present
    ibound = np.ones(ml.dis.botm.shape)
    ibound[0][0:5] = 0
    bas = ModflowBas(ml, ibound=ibound)
    assert ml.modelgrid.idomain is not None

    mt = Mt3dms(
        modelname="test_mt",
        modflowmodel=ml,
        model_ws=ml.model_ws,
        verbose=True,
    )
    btn = Mt3dBtn(mt, icbund=ml.bas6.ibound.array)

    # reload swt
    swt = Seawat(
        modelname="test_swt",
        modflowmodel=ml,
        mt3dmodel=mt,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert (
        ml.modelgrid.xoffset == mt.modelgrid.xoffset == swt.modelgrid.xoffset
    )
    assert (
        mt.modelgrid.yoffset == ml.modelgrid.yoffset == swt.modelgrid.yoffset
    )
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)


def test_free_format_flag(tmpdir):
    Lx = 100.0
    Ly = 100.0
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = Modflow(rotation=20.0)
    dis = ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    bas = ModflowBas(ms, ifrefm=True)
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = False
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = True
    bas.ifrefm = False
    assert ms.free_format_input == bas.ifrefm
    bas.ifrefm = True
    assert ms.free_format_input == bas.ifrefm

    ms.model_ws = tmpdir
    ms.write_input()
    ms1 = Modflow.load(ms.namefile, model_ws=ms.model_ws)
    assert ms1.free_format_input == ms.free_format_input
    assert ms1.free_format_input == ms1.bas6.ifrefm
    ms1.free_format_input = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = True
    assert ms1.free_format_input == ms1.bas6.ifrefm


def test_sr(tmpdir):
    ws = str(tmpdir)
    m = Modflow(
        "test",
        model_ws=ws,
        xll=12345,
        yll=12345,
        proj4_str="test test test",
    )
    ModflowDis(m, 10, 10, 10)
    m.write_input()
    mm = Modflow.load("test.nam", model_ws=ws)
    extents = mm.modelgrid.extent
    if extents[2] != 12345:
        raise AssertionError()
    if extents[3] != 12355:
        raise AssertionError()
    if mm.modelgrid.proj4 != "test test test":
        raise AssertionError()

    mm.dis.top = 5000

    if not np.allclose(mm.dis.top.array, mm.modelgrid.top):
        raise AssertionError("modelgrid failed dynamic update test")


def test_mf6_update_grid(example_data_path):
    ml_path = example_data_path / "mf6" / "test001a_Tharmonic"
    sim = MFSimulation.load(sim_ws=str(ml_path))
    gwf = sim.get_model("flow15")
    mg = gwf.modelgrid
    gwf.dis.top = 12

    assert np.allclose(
        gwf.dis.top.array, gwf.modelgrid.top
    ), "StructuredGrid failed dynamic update test"

    # disv
    ml_path = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=str(ml_path))
    gwf = sim.get_model("gwf_1")
    mg = gwf.modelgrid
    gwf.disv.top = 6.12

    assert np.allclose(
        gwf.disv.top.array, gwf.modelgrid.top
    ), "VertexGrid failed dynamic update test"

    # disu
    ml_path = example_data_path / "mf6" / "test006_gwf3"
    sim = MFSimulation.load(sim_ws=str(ml_path))
    gwf = sim.get_model("gwf_1")
    mg = gwf.modelgrid
    gwf.disu.top = 101

    assert np.allclose(
        gwf.disu.top.array, gwf.modelgrid.top
    ), "UnstructuredGrid failed dynamic update test"


def test_load_twri_grid(example_data_path):
    mpath = example_data_path / "mf2005_test"
    name = "twri.nam"
    ml = Modflow.load(name, model_ws=str(mpath), check=False)
    mg = ml.modelgrid
    assert isinstance(
        mg, StructuredGrid
    ), "modelgrid is not an StructuredGrid instance"
    shape = (3, 15, 15)
    assert (
        mg.shape == shape
    ), f"modelgrid shape {mg.shape} not equal to {shape}"
    thick = mg.thick
    shape = (5, 15, 15)
    assert (
        thick.shape == shape
    ), f"thickness shape {thick.shape} not equal to {shape}"


def test_mg(tmpdir):
    from flopy.utils import geometry

    Lx = 100.0
    Ly = 100.0
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = Modflow()
    dis = ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    bas = ModflowBas(ms, ifrefm=True)
    t = ms.modelgrid.thick

    # test instantiation of an empty basic Structured Grid
    mg = StructuredGrid(dis.delc.array, dis.delr.array)

    # test instantiation of Structured grid with offsets
    mg = StructuredGrid(dis.delc.array, dis.delr.array, xoff=1, yoff=1)

    assert abs(ms.modelgrid.extent[-1] - Ly) < 1e-3  # , txt
    ms.modelgrid.set_coord_info(xoff=111, yoff=0)
    assert ms.modelgrid.xoffset == 111
    ms.modelgrid.set_coord_info()

    xll, yll = 321.0, 123.0
    angrot = 20.0
    ms.modelgrid = StructuredGrid(
        delc=ms.dis.delc.array,
        delr=ms.dis.delr.array,
        xoff=xll,
        yoff=xll,
        angrot=angrot,
        lenuni=2,
    )

    # test that transform for arbitrary coordinates
    # is working in same as transform for model grid
    mg2 = StructuredGrid(
        delc=ms.dis.delc.array, delr=ms.dis.delr.array, lenuni=2
    )
    x = mg2.xcellcenters[0]
    y = mg2.ycellcenters[0]
    mg2.set_coord_info(xoff=xll, yoff=yll, angrot=angrot)
    xt, yt = geometry.transform(x, y, xll, yll, mg2.angrot_radians)

    assert np.sum(xt - ms.modelgrid.xcellcenters[0]) < 1e-3
    assert np.sum(yt - ms.modelgrid.ycellcenters[0]) < 1e-3

    # test inverse transform
    x0, y0 = 9.99, 2.49
    x1, y1 = geometry.transform(x0, y0, xll, yll, angrot)
    x2, y2 = geometry.transform(x1, y1, xll, yll, angrot, inverse=True)
    assert np.abs(x2 - x0) < 1e-6
    assert np.abs(y2 - y0) < 1e6

    ms.start_datetime = "1-1-2016"
    assert ms.start_datetime == "1-1-2016"
    assert ms.dis.start_datetime == "1-1-2016"

    ms.model_ws = tmpdir

    ms.write_input()
    ms1 = Modflow.load(ms.namefile, model_ws=ms.model_ws)

    assert str(ms1.modelgrid) == str(ms.modelgrid)
    assert ms1.start_datetime == ms.start_datetime
    assert ms1.modelgrid.lenuni == ms.modelgrid.lenuni


def test_dynamic_xll_yll():
    nlay, nrow, ncol = 1, 10, 5
    delr, delc = 250, 500
    xll, yll = 286.80, 29.03
    # test scaling of length units
    ms2 = Modflow()
    dis = ModflowDis(
        ms2, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )

    ms2.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=30.0)
    xll1, yll1 = ms2.modelgrid.xoffset, ms2.modelgrid.yoffset

    assert xll1 == xll, f"modelgrid.xoffset ({xll1}) is not equal to {xll}"
    assert yll1 == yll, f"modelgrid.yoffset ({yll1}) is not equal to {yll}"

    # check that xll, yll are being recomputed
    xll += 10.0
    yll += 21.0
    ms2.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=30.0)
    xll1, yll1 = ms2.modelgrid.xoffset, ms2.modelgrid.yoffset

    assert xll1 == xll, f"modelgrid.xoffset ({xll1}) is not equal to {xll}"
    assert yll1 == yll, f"modelgrid.yoffset ({yll1}) is not equal to {yll}"


def test_namfile_readwrite(tmpdir, example_data_path):
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    xll, yll = 272300, 5086000
    ws = str(tmpdir)
    m = Modflow(modelname="junk", model_ws=ws)
    dis = ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc)
    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        top=m.dis.top.array,
        botm=m.dis.botm.array,
        # lenuni=3,
        # length_multiplier=.3048,
        xoff=xll,
        yoff=yll,
        angrot=30,
    )

    # test reading and writing of SR information to namfile
    m.write_input()
    m2 = Modflow.load("junk.nam", model_ws=ws)

    t_value = abs(m2.modelgrid.xoffset - xll)
    msg = f"m2.modelgrid.xoffset ({m2.modelgrid.xoffset}) does not equal {xll}"
    assert t_value < 1e-2, msg

    t_value = abs(m2.modelgrid.yoffset - yll)
    msg = f"m2.modelgrid.yoffset ({m2.modelgrid.yoffset}) does not equal {yll}"
    assert t_value < 1e-2

    msg = f"m2.modelgrid.angrot ({m2.modelgrid.angrot}) does not equal 30"
    assert m2.modelgrid.angrot == 30, msg

    ml = Modflow.load(
        "freyberg.nam",
        model_ws=str(example_data_path / "freyberg_multilayer_transient"),
        verbose=False,
        check=False,
        exe_name="mfnwt",
    )

    assert ml.modelgrid.xoffset == ml.modelgrid._xul_to_xll(619653)
    assert ml.modelgrid.yoffset == ml.modelgrid._yul_to_yll(3353277)
    assert ml.modelgrid.angrot == 15.0


def test_read_usgs_model_reference(tmpdir, model_reference_path):
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    # xll, yll = 272300, 5086000

    mrf_path = tmpdir / model_reference_path.name
    shutil.copy(model_reference_path, mrf_path)

    xul, yul = 0, 0
    with open(mrf_path) as foo:
        for line in foo:
            if "xul" in line.lower():
                xul = float(line.strip().split()[1])
            elif "yul" in line.lower():
                yul = float(line.strip().split()[1])
            else:
                continue

    ws = str(tmpdir)
    m = Modflow(modelname="junk", model_ws=ws)
    # feet and days
    dis = ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        lenuni=1,
        itmuni=4,
    )
    m.write_input()

    # test reading of SR information from usgs.model.reference
    m2 = Modflow.load("junk.nam", model_ws=ws)
    from flopy.discretization import StructuredGrid

    mg = StructuredGrid(delr=dis.delr.array, delc=dis.delc.array)
    mg.read_usgs_model_reference_file(mrf_path)
    m2.modelgrid = mg

    if abs(mg.xvertices[0, 0] - xul) > 0.01:
        raise AssertionError()
    if abs(mg.yvertices[0, 0] - yul) > 0.01:
        raise AssertionError

    assert m2.modelgrid.xoffset == mg.xoffset
    assert m2.modelgrid.yoffset == mg.yoffset
    assert m2.modelgrid.angrot == mg.angrot
    assert m2.modelgrid.epsg == mg.epsg

    # test reading non-default units from usgs.model.reference
    shutil.copy(mrf_path, f"{mrf_path}_copy")
    with open(f"{mrf_path}_copy") as src:
        with open(mrf_path, "w") as dst:
            for line in src:
                if "epsg" in line:
                    line = line.replace("102733", "4326")
                dst.write(line)

    m2 = Modflow.load("junk.nam", model_ws=ws)
    m2.modelgrid.read_usgs_model_reference_file(mrf_path)

    assert m2.modelgrid.epsg == 4326
    # have to delete this, otherwise it will mess up other tests
    to_del = glob.glob(f"{mrf_path}*")
    for f in to_del:
        if os.path.exists(f):
            os.remove(os.path.join(f))


def mf2005_model_namfiles():
    path = get_example_data_path(Path(__file__)) / "mf2005_test"
    return [str(p) for p in path.glob("*.nam")]


def parameters_model_namfiles():
    path = get_example_data_path(Path(__file__)) / "parameters"
    skip = ["twrip.nam", "twrip_upw.nam"]  # TODO: why do these fail?
    return [str(p) for p in path.glob("*.nam") if p.name not in skip]


@requires_exe("mf2005")
@pytest.mark.parametrize(
    "namfile", mf2005_model_namfiles() + parameters_model_namfiles()
)
def test_mf2005_test_models_load(example_data_path, namfile):
    assert not Modflow.load(
        namfile,
        model_ws=str(example_data_path / "mf2005_test"),
        version="mf2005",
        verbose=True,
    ).load_fail


@requires_exe("mf2005")
@pytest.mark.parametrize("namfile", parameters_model_namfiles())
def test_parameters_models_load(parameters_model_path, namfile):
    assert not Modflow.load(
        namfile,
        model_ws=str(parameters_model_path),
        version="mf2005",
        verbose=True,
    ).load_fail


@pytest.mark.parametrize("namfile", mf2005_model_namfiles())
def test_mf2005_test_models_loadonly(example_data_path, namfile):
    assert not Modflow.load(
        str(namfile),
        model_ws=str(example_data_path / "mf2005_test"),
        version="mf2005",
        verbose=True,
        load_only=["bas6"],
        check=False,
    ).load_fail


@pytest.mark.slow
def test_write_irch(tmpdir, example_data_path):
    mpath = example_data_path / "freyberg_multilayer_transient"
    nam_file = "freyberg.nam"
    m = Modflow.load(
        nam_file,
        model_ws=str(mpath),
        check=False,
        forgive=False,
        verbose=True,
    )
    irch = {}
    for kper in range(m.nper):
        arr = np.random.randint(0, m.nlay, size=(m.nrow, m.ncol))
        irch[kper] = arr

    m.remove_package("RCH")
    ModflowRch(m, rech=0.1, irch=irch, nrchop=2)

    for kper in range(m.nper):
        arr = irch[kper]
        aarr = m.rch.irch[kper].array
        d = arr - aarr
        assert np.abs(d).sum() == 0

    m.change_model_ws(str(tmpdir))
    m.write_input()

    mm = Modflow.load(
        nam_file,
        model_ws=str(tmpdir),
        forgive=False,
        verbose=True,
        check=False,
    )
    for kper in range(m.nper):
        arr = irch[kper]
        aarr = mm.rch.irch[kper].array
        d = arr - aarr
        assert np.abs(d).sum() == 0


def test_mflist_external(tmpdir):
    ext = tmpdir / "ws"
    ml = Modflow(
        "mflist_test",
        model_ws=str(tmpdir),
        external_path=ext.name,
    )

    dis = ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    wel_data = {
        0: [[0, 0, 0, -1], [1, 1, 1, -1]],
        1: [[0, 0, 0, -2], [1, 1, 1, -1]],
    }
    wel = ModflowWel(ml, stress_period_data=wel_data)
    ml.change_model_ws(str(ext))
    ml.write_input()

    ml1 = Modflow.load(
        "mflist_test.nam",
        model_ws=str(ext),
        verbose=True,
        forgive=False,
        check=False,
    )

    assert np.array_equal(ml.wel[0], ml1.wel[0])
    assert np.array_equal(ml.wel[1], ml1.wel[1])

    ml1.write_input()

    # ml = Modflow(
    #     "mflist_test",
    #     model_ws=str(tmpdir),
    #     external_path=str(tmpdir / "ref"),
    # )
    # dis = ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    # wel_data = {
    #     0: [[0, 0, 0, -1], [1, 1, 1, -1]],
    #     1: [[0, 0, 0, -2], [1, 1, 1, -1]],
    # }
    # wel = ModflowWel(ml, stress_period_data=wel_data)
    # ml.write_input()

    # ml1 = Modflow.load(
    #     "mflist_test.nam",
    #     model_ws=ml.model_ws,
    #     verbose=True,
    #     forgive=False,
    #     check=False,
    # )

    # assert np.array_equal(ml.wel[0], ml1.wel[0])
    # assert np.array_equal(ml.wel[1], ml1.wel[1])

    # ml1.write_input()


@excludes_platform("windows", ci_only=True)
def test_single_mflist_entry_load(tmpdir, example_data_path):
    m = Modflow.load(
        "freyberg.nam",
        model_ws=str(example_data_path / "freyberg"),
        load_only=["WEL"],
        check=False,
    )
    w = m.wel
    spd = w.stress_period_data
    ModflowWel(m, stress_period_data={0: [0, 0, 0, 0.0]})
    m.external_path = "external"
    m.change_model_ws(str(tmpdir), reset_external=True)
    m.write_input()

    mm = Modflow.load(
        "freyberg.nam",
        model_ws=str(tmpdir),
        forgive=False,
    )
    assert mm.wel.stress_period_data
    mm.write_input()


def test_mflist_add_record():
    ml = Modflow()
    _ = ModflowDis(ml, nper=2)
    wel = ModflowWel(ml)
    assert len(wel.stress_period_data.data) == 0

    wel.stress_period_data.add_record(0, [0, 1, 2], [1.0])
    assert len(wel.stress_period_data.data) == 1
    wel_dtype = [("k", int), ("i", int), ("j", int), ("flux", np.float32)]
    check0 = np.array([(0, 1, 2, 1.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)

    wel.stress_period_data.add_record(0, [0, 1, 1], [8.0])
    assert len(wel.stress_period_data.data) == 1
    check0 = np.array([(0, 1, 2, 1.0), (0, 1, 1, 8.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)

    wel.stress_period_data.add_record(1, [0, 1, 1], [5.0])
    assert len(wel.stress_period_data.data) == 2
    check1 = np.array([(0, 1, 1, 5.0)], dtype=wel_dtype)
    np.testing.assert_array_equal(wel.stress_period_data[0], check0)
    np.testing.assert_array_equal(wel.stress_period_data[1], check1)


__mf2005_test_path = get_example_data_path(Path(__file__)) / "mf2005_test"


@pytest.mark.parametrize(
    "namfile",
    [
        os.path.join(__mf2005_test_path, f)
        for f in os.listdir(__mf2005_test_path)
        if f.endswith(".nam")
    ],
)
def test_checker_on_load(namfile):
    # load all of the models in the mf2005_test folder
    # model level checks are performed by default on load()
    f = os.path.basename(namfile)
    d = os.path.dirname(namfile)
    m = Modflow.load(f, model_ws=d)
    assert isinstance(m, Modflow), "Not a Modflow instance"


def test_bcs_check(tmpdir):
    mf = Modflow(version="mf2005", model_ws=str(tmpdir))

    # test check for isolated cells
    dis = ModflowDis(mf, nlay=2, nrow=3, ncol=3, top=100, botm=95)
    bas = ModflowBas(mf, ibound=np.ones((2, 3, 3), dtype=int))
    chk = bas.check()

    dis = ModflowDis(mf, nlay=3, nrow=5, ncol=5, top=100, botm=95)
    ibound = np.zeros((3, 5, 5), dtype=int)
    ibound[1, 1, 1] = 1  # fully isolated cell
    ibound[0:2, 4, 4] = 1  # cell connected vertically to one other cell
    bas = ModflowBas(mf, ibound=ibound)
    mf._mg_resync = True
    chk = bas.check()
    assert chk.summary_array["desc"][0] == "isolated cells in ibound array"
    assert (
        chk.summary_array.i[0] == 1
        and chk.summary_array.i[0] == 1
        and chk.summary_array.j[0] == 1
    )
    assert len(chk.summary_array) == 1

    ghb = ModflowGhb(mf, stress_period_data={0: [0, 0, 0, 100, 1]})
    riv = ModflowRiv(
        mf,
        stress_period_data={
            0: [[0, 0, 0, 101, 10, 100], [0, 0, 1, 80, 10, 90]]
        },
    )
    chk = ghb.check()
    assert chk.summary_array["desc"][0] == "BC in inactive cell"
    chk = riv.check()
    assert chk.summary_array["desc"][4] == "RIV stage below rbots"
    assert np.array_equal(chk.summary_array["j"], np.array([0, 1, 1, 1, 1]))


def test_properties_check(tmpdir):
    mf = Modflow(
        version="mf2005",
        model_ws=str(tmpdir),
    )
    dis = ModflowDis(
        mf,
        nrow=2,
        ncol=2,
        top=np.array([[100, np.nan], [100, 100]]),
        nper=3,
        steady=True,
    )
    chk = dis.check()
    assert len(chk.summary_array) == 1
    kij = (
        chk.summary_array["k"][0],
        chk.summary_array["i"][0],
        chk.summary_array["j"][0],
    )
    assert kij == (0, 0, 1)
    lpf = ModflowLpf(mf, sy=np.ones((2, 2)), ss=np.ones((2, 2)))
    chk = lpf.check()
    assert len(chk.summary_array) == 0

    # test k values check
    lpf = ModflowLpf(
        mf,
        hk=np.array([[1, 1e10], [1, -1]]),
        hani=np.array([[1, 1], [1, -1]]),
        vka=np.array([[1e10, 0], [1, 1e-20]]),
    )
    chk = lpf.check()
    ind1 = np.array(
        [
            True if list(inds) == [0, 1, 1] else False
            for inds in chk.view_summary_array_fields(["k", "i", "j"])
        ]
    )
    ind1_errors = chk.summary_array[ind1]["desc"]
    ind2 = np.array(
        [
            True if list(inds) == [0, 0, 1] else False
            for inds in chk.view_summary_array_fields(["k", "i", "j"])
        ]
    )
    ind2_errors = chk.summary_array[ind2]["desc"]
    ind3 = np.array(
        [
            True if list(inds) == [0, 0, 0] else False
            for inds in chk.view_summary_array_fields(["k", "i", "j"])
        ]
    )
    ind3_errors = chk.summary_array[ind3]["desc"]

    assert (
        "zero or negative horizontal hydraulic conductivity values"
        in ind1_errors
    )
    assert (
        "horizontal hydraulic conductivity values below checker threshold of 1e-11"
        in ind1_errors
    )
    assert "negative horizontal anisotropy values" in ind1_errors
    assert (
        "vertical hydraulic conductivity values below checker threshold of 1e-11"
        in ind1_errors
    )
    assert (
        "horizontal hydraulic conductivity values above checker threshold of 100000.0"
        in ind2_errors
    )
    assert (
        "zero or negative vertical hydraulic conductivity values"
        in ind2_errors
    )
    assert (
        "vertical hydraulic conductivity values above checker threshold of 100000.0"
        in ind3_errors
    )


def test_oc_check():
    m = Modflow()
    oc = ModflowOc(m)
    chk = oc.check()
    assert len(chk.summary_array) == 1, len(chk.summary_array)
    assert "DIS package not available" in chk.summary_array[0]["desc"]

    ModflowDis(m)
    oc.stress_period_data = {(0, 0): ["save head", "save budget"]}
    chk = oc.check()  # check passsed
    assert len(chk.summary_array) == 0, len(chk.summary_array)

    oc.stress_period_data = {(0, 0): ["save"]}
    chk = oc.check()
    assert len(chk.summary_array) == 1, len(chk.summary_array)
    assert "too few words" in chk.summary_array[0]["desc"]

    oc.stress_period_data = {(0, 0): ["save it"]}
    chk = oc.check()
    assert len(chk.summary_array) == 1, len(chk.summary_array)
    assert "action 'save it' ignored" in chk.summary_array[0]["desc"]

    oc.stress_period_data = {(1, 1): ["save head", "save budget"]}
    chk = oc.check()
    assert len(chk.summary_array) == 1, len(chk.summary_array)
    assert "OC stress_period_data ignored" in chk.summary_array[0]["desc"]


def test_rchload(tmpdir):
    nlay = 2
    nrow = 3
    ncol = 4
    nper = 2

    # create model 1
    ws = str(tmpdir)
    m1 = Modflow("rchload1", model_ws=ws)
    dis1 = ModflowDis(m1, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)
    a = np.random.random((nrow, ncol))
    rech1 = Util2d(
        m1, (nrow, ncol), np.float32, a, "rech", cnstnt=1.0, how="openclose"
    )
    rch1 = ModflowRch(m1, rech={0: rech1})
    m1.write_input()

    # load model 1
    m1l = Modflow.load("rchload1.nam", model_ws=ws)
    a1 = rech1.array
    a2 = m1l.rch.rech[0].array
    assert np.allclose(a1, a2)
    a2 = m1l.rch.rech[1].array
    assert np.allclose(a1, a2)

    m2 = Modflow("rchload2", model_ws=ws)
    dis2 = ModflowDis(m2, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)
    a = np.random.random((nrow, ncol))
    rech2 = Util2d(
        m2, (nrow, ncol), np.float32, a, "rech", cnstnt=2.0, how="openclose"
    )
    rch2 = ModflowRch(m2, rech={0: rech2})
    m2.write_input()

    # load model 2
    m2l = Modflow.load("rchload2.nam", model_ws=ws)
    a1 = rech2.array
    a2 = m2l.rch.rech[0].array
    assert np.allclose(a1, a2)
    a2 = m2l.rch.rech[1].array
    assert np.allclose(a1, a2)


@requires_pkg("pandas")
def test_mp5_load(tmpdir, example_data_path):
    # load the base freyberg model
    freyberg_ws = example_data_path / "freyberg"
    # load the modflow files for model map
    m = Modflow.load(
        "freyberg.nam",
        model_ws=str(freyberg_ws),
        check=False,
        verbose=True,
        forgive=False,
    )

    # load the pathline data
    pthobj = PathlineFile(str(example_data_path / "mp5" / "m.ptl"))

    # load endpoint data
    fpth = str(example_data_path / "mp5" / "m.ept")
    endobj = EndpointFile(fpth, verbose=True)

    # determine version
    ver = pthobj.version
    assert ver == 5, f"{fpth} is not a MODPATH version 5 pathline file"

    # read all of the pathline and endpoint data
    plines = pthobj.get_alldata()
    epts = endobj.get_alldata()

    # determine the number of particles in the pathline file
    nptl = pthobj.nid.shape[0]
    assert nptl == 64, "number of MODPATH 5 particles does not equal 64"

    hsv = plt.get_cmap("hsv")
    colors = hsv(np.linspace(0, 1.0, nptl))

    # plot the pathlines one pathline at a time
    mm = PlotMapView(model=m)
    for n in pthobj.nid:
        p = pthobj.get_data(partid=n)
        e = endobj.get_data(partid=n)
        try:
            mm.plot_pathline(p, colors=colors[n], layer="all")
        except:
            assert False, f'could not plot pathline {n + 1} with layer="all"'
        try:
            mm.plot_endpoint(e)
        except:
            assert False, f'could not plot endpoint {n + 1} with layer="all"'

    # plot the grid and ibound array
    mm.plot_grid(lw=0.5)
    mm.plot_ibound()

    fpth = os.path.join(str(tmpdir), "mp5.pathline.png")
    plt.savefig(fpth, dpi=300)
    plt.close()


@requires_pkg("pandas")
def test_mp5_timeseries_load(example_data_path):
    pth = str(example_data_path / "mp5")
    files = [
        os.path.join(pth, name)
        for name in sorted(os.listdir(pth))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)


@requires_pkg("pandas")
def test_mp6_timeseries_load(example_data_path):
    pth = str(example_data_path / "mp5")
    files = [
        os.path.join(pth, name)
        for name in sorted(os.listdir(pth))
        if ".timeseries" in name
    ]
    for file in files:
        print(file)
        eval_timeseries(file)


def eval_timeseries(file):
    ts = TimeseriesFile(file)
    assert isinstance(ts, TimeseriesFile), (
        f"{os.path.basename(file)} " "is not an instance of TimeseriesFile"
    )

    # get the all of the data
    tsd = ts.get_alldata()
    assert (
        len(tsd) > 0
    ), f"could not load data using get_alldata() from {os.path.basename(file)}."

    # get the data for the last particleid
    partid = ts.get_maxid()
    assert partid is not None, (
        "could not get maximum particleid using get_maxid() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_data(partid=partid)
    assert tsd.shape[0] > 0, (
        f"could not load data for particleid {partid} using get_data() from "
        f"{os.path.basename(file)}. Maximum partid = {ts.get_maxid()}."
    )

    timemax = ts.get_maxtime() / 2.0
    assert timemax is not None, (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_alldata(totim=timemax)
    assert len(tsd) > 0, (
        f"could not load data for totim>={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )

    timemax = ts.get_maxtime()
    assert timemax is not None, (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_alldata(totim=timemax, ge=False)
    assert len(tsd) > 0, (
        f"could not load data for totim<={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )


def test_default_oc_stress_period_data(tmpdir):
    m = Modflow(model_ws=str(tmpdir), verbose=True)
    dis = ModflowDis(m, nper=10, perlen=10.0, nstp=5)
    bas = ModflowBas(m)
    lpf = ModflowLpf(m, ipakcb=100)
    wel_data = {0: [[0, 0, 0, -1000.0]]}
    wel = ModflowWel(m, ipakcb=101, stress_period_data=wel_data)
    # spd = {(0, 0): ['save head', 'save budget']}
    oc = ModflowOc(m, stress_period_data=None)
    spd_oc = oc.stress_period_data
    tups = list(spd_oc.keys())
    kpers = [t[0] for t in tups]
    assert len(kpers) == m.nper
    kstps = [t[1] for t in tups]
    assert max(kstps) == 4
    assert min(kstps) == 4
    m.write_input()


def test_mfcbc(tmpdir):
    m = Modflow(verbose=True, model_ws=str(tmpdir))
    dis = ModflowDis(m)
    bas = ModflowBas(m)
    lpf = ModflowLpf(m, ipakcb=100)
    wel_data = {0: [[0, 0, 0, -1000.0]]}
    wel = ModflowWel(m, ipakcb=101, stress_period_data=wel_data)
    spd = {(0, 0): ["save head", "save budget"]}
    oc = ModflowOc(m, stress_period_data=spd)
    t = oc.get_budgetunit()
    assert t == [100, 101], f"budget units are {t} not [100, 101]"

    nlay = 3
    nrow = 3
    ncol = 3
    ml = Modflow(modelname="t1", model_ws=str(tmpdir), verbose=True)
    dis = ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=0, botm=[-1.0, -2.0, -3.0]
    )
    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    ibound[0, 1, 1] = 0
    ibound[0, 0, -1] = -1
    bas = ModflowBas(ml, ibound=ibound)
    lpf = ModflowLpf(ml, ipakcb=102)
    wel_data = {0: [[2, 2, 2, -1000.0]]}
    wel = ModflowWel(ml, ipakcb=100, stress_period_data=wel_data)
    oc = ModflowOc(ml)

    oc.reset_budgetunit(budgetunit=1053, fname="big.bin")

    msg = (
        f"wel ipakcb ({wel.ipakcb}) "
        "not set correctly to 1053 using oc.resetbudgetunit()"
    )
    assert wel.ipakcb == 1053, msg

    ml.write_input()


def test_load_with_list_reader(tmpdir):
    # Create an original model and then manually modify to use
    # advanced list reader capabilities
    nlay = 1
    nrow = 10
    ncol = 10
    nper = 3

    # create the ghbs
    ghbra = ModflowGhb.get_empty(20)
    l = 0
    for i in range(nrow):
        ghbra[l] = (0, i, 0, 1.0, 100.0 + i)
        l += 1
        ghbra[l] = (0, i, ncol - 1, 1.0, 200.0 + i)
        l += 1
    ghbspd = {0: ghbra}

    # create the drains
    drnra = ModflowDrn.get_empty(2)
    drnra[0] = (0, 1, int(ncol / 2), 0.5, 55.0)
    drnra[1] = (0, 2, int(ncol / 2), 0.5, 75.0)
    drnspd = {0: drnra}

    # create the wells
    welra = ModflowWel.get_empty(2)
    welra[0] = (0, 1, 1, -5.0)
    welra[1] = (0, nrow - 3, ncol - 3, -10.0)
    welspd = {0: welra}

    m = Modflow(
        modelname="original",
        model_ws=str(tmpdir),
        exe_name="mf2005",
    )
    dis = ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)
    bas = ModflowBas(m)
    lpf = ModflowLpf(m)
    ghb = ModflowGhb(m, stress_period_data=ghbspd)
    drn = ModflowDrn(m, stress_period_data=drnspd)
    wel = ModflowWel(m, stress_period_data=welspd)
    pcg = ModflowPcg(m)
    oc = ModflowOc(m)
    m.add_external("original.drn.dat", 71)
    m.add_external("original.wel.bin", 72, binflag=True, output=False)
    m.write_input()

    # rewrite ghb
    fname = os.path.join(str(tmpdir), "original.ghb")
    with open(fname, "w") as f:
        f.write(f"{ghbra.shape[0]} 0\n")
        for kper in range(nper):
            f.write(f"{ghbra.shape[0]} 0\n")
            f.write("open/close original.ghb.dat\n")

    # write ghb list
    sfacghb = 5
    fname = os.path.join(str(tmpdir), "original.ghb.dat")
    with open(fname, "w") as f:
        f.write(f"sfac {sfacghb}\n")
        for k, i, j, stage, cond in ghbra:
            f.write(f"{k + 1} {i + 1} {j + 1} {stage} {cond}\n")

    # rewrite drn
    fname = os.path.join(str(tmpdir), "original.drn")
    with open(fname, "w") as f:
        f.write(f"{drnra.shape[0]} 0\n")
        for kper in range(nper):
            f.write(f"{drnra.shape[0]} 0\n")
            f.write("external 71\n")

    # write drn list
    sfacdrn = 1.5
    fname = os.path.join(str(tmpdir), "original.drn.dat")
    with open(fname, "w") as f:
        for kper in range(nper):
            f.write(f"sfac {sfacdrn}\n")
            for k, i, j, stage, cond in drnra:
                f.write(f"{k + 1} {i + 1} {j + 1} {stage} {cond}\n")

    # rewrite wel
    fname = os.path.join(str(tmpdir), "original.wel")
    with open(fname, "w") as f:
        f.write(f"{drnra.shape[0]} 0\n")
        for kper in range(nper):
            f.write(f"{drnra.shape[0]} 0\n")
            f.write("external 72 (binary)\n")

    # create the wells, but use an all float dtype to write a binary file
    # use one-based values
    weldt = np.dtype(
        [
            ("k", "<f4"),
            ("i", "<f4"),
            ("j", "<f4"),
            ("q", "<f4"),
        ]
    )
    welra = np.recarray(2, dtype=weldt)
    welra[0] = (1, 2, 2, -5.0)
    welra[1] = (1, nrow - 2, ncol - 2, -10.0)
    fname = os.path.join(str(tmpdir), "original.wel.bin")
    with open(fname, "wb") as f:
        welra.tofile(f)
        welra.tofile(f)
        welra.tofile(f)

    # no need to run the model
    # success, buff = m.run_model(silent=True)
    # assert success, 'model did not terminate successfully'

    # the m2 model will load all of these external files, possibly using sfac
    # and just create regular list input files for wel, drn, and ghb
    fname = "original.nam"
    m2 = Modflow.load(fname, model_ws=str(tmpdir), verbose=False)
    m2.name = "new"
    m2.write_input()

    originalghbra = m.ghb.stress_period_data[0].copy()
    originalghbra["cond"] *= sfacghb
    assert np.array_equal(originalghbra, m2.ghb.stress_period_data[0])

    originaldrnra = m.drn.stress_period_data[0].copy()
    originaldrnra["cond"] *= sfacdrn
    assert np.array_equal(originaldrnra, m2.drn.stress_period_data[0])

    originalwelra = m.wel.stress_period_data[0].copy()
    assert np.array_equal(originalwelra, m2.wel.stress_period_data[0])


def get_basic_modflow_model(ws, name):
    m = Modflow(name, model_ws=ws)

    size = 100
    nlay = 10
    nper = 10
    nsfr = int((size**2) / 5)

    dis = ModflowDis(
        m,
        nper=nper,
        nlay=nlay,
        nrow=size,
        ncol=size,
        top=nlay,
        botm=list(range(nlay)),
    )

    rch = ModflowRch(
        m, rech={k: 0.001 - np.cos(k) * 0.001 for k in range(nper)}
    )

    ra = ModflowWel.get_empty(size**2)
    well_spd = {}
    for kper in range(nper):
        ra_per = ra.copy()
        ra_per["k"] = 1
        ra_per["i"] = (
            (np.ones((size, size)) * np.arange(size))
            .transpose()
            .ravel()
            .astype(int)
        )
        ra_per["j"] = list(range(size)) * size
        well_spd[kper] = ra
    wel = ModflowWel(m, stress_period_data=well_spd)

    # SFR package
    rd = ModflowSfr2.get_empty_reach_data(nsfr)
    rd["iseg"] = range(len(rd))
    rd["ireach"] = 1
    sd = ModflowSfr2.get_empty_segment_data(nsfr)
    sd["nseg"] = range(len(sd))
    sfr = ModflowSfr2(reach_data=rd, segment_data=sd, model=m)

    return m


@pytest.mark.slow
def test_model_init_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    benchmark(lambda: get_basic_modflow_model(ws=str(tmpdir), name=name))


@pytest.mark.slow
def test_model_write_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = get_basic_modflow_model(ws=str(tmpdir), name=name)
    benchmark(lambda: model.write_input())


@pytest.mark.slow
def test_model_load_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = get_basic_modflow_model(ws=str(tmpdir), name=name)
    model.write_input()
    benchmark(
        lambda: Modflow.load(f"{name}.nam", model_ws=str(tmpdir), check=False)
    )
