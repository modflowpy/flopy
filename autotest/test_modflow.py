import glob
import inspect
import os
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import excludes_platform, requires_exe
from modflow_devtools.misc import has_pkg

from autotest.conftest import get_example_data_path
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
from flopy.seawat import Seawat
from flopy.utils import Util2d

_example_data_path = get_example_data_path()


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

    # Paths
    model = Modflow.load(
        mpath / namfile.name,
        verbose=True,
        model_ws=mpath,
        check=False,
    )

    assert isinstance(model, Modflow)
    assert not model.load_fail
    assert model.model_ws == str(mpath)

    # string paths
    model = Modflow.load(
        str(mpath / namfile.name),
        verbose=True,
        model_ws=str(mpath),
        check=False,
    )

    assert isinstance(model, Modflow)
    assert not model.load_fail
    assert model.model_ws == str(mpath)


@pytest.mark.parametrize(
    "path,expected",
    [
        pytest.param(
            _example_data_path / "freyberg" / "freyberg.nam",
            {
                "crs": None,
                "epsg": None,
                "angrot": 0.0,
                "xoffset": 0.0,
                "yoffset": 0.0,
            },
            id="freyberg",
        ),
        pytest.param(
            _example_data_path
            / "freyberg_multilayer_transient"
            / "freyberg.nam",
            {
                "proj4": "+proj=utm +zone=14 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
                "angrot": 15.0,
                "xoffset": 622241.1904510253,
                "yoffset": 3343617.741737109,
            },
            id="freyberg_multilayer_transient",
        ),
        pytest.param(
            _example_data_path
            / "mt3d_test"
            / "mfnwt_mt3dusgs"
            / "sft_crnkNic"
            / "CrnkNic.nam",
            {
                "epsg": 26916,
                "angrot": 0.0,
                "xoffset": 0.0,
                "yoffset": 0.0,
            },
            id="CrnkNic",
        ),
    ],
)
def test_modflow_load_modelgrid(path, expected):
    """Check modelgrid metadata from NAM file."""
    model = Modflow.load(path.name, model_ws=path.parent, load_only=[])
    modelgrid = model.modelgrid
    for key, expected_value in expected.items():
        if key == "proj4" and has_pkg("pyproj"):
            # skip since pyproj will usually restructure proj4 attribute
            # otherwise continue test without pyproj, as it should be preserved
            continue
        modelgrid_value = getattr(modelgrid, key)
        if isinstance(modelgrid_value, float):
            assert modelgrid_value == pytest.approx(expected_value), key
        else:
            assert modelgrid_value == expected_value, key


def test_modflow_load_when_nam_dne():
    with pytest.raises(OSError):
        Modflow.load("nonexistent.nam", check=False)


def test_mbase_modelgrid(function_tmpdir):
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
    ml.model_ws = function_tmpdir

    ml.write_input()
    ml1 = Modflow.load("test.nam", model_ws=ml.model_ws)
    assert str(ml1.modelgrid) == str(ml.modelgrid)
    assert ml1.start_datetime == ml.start_datetime
    assert ml1.modelgrid.proj4 is None


def test_mt_modelgrid(function_tmpdir):
    ml = Modflow(
        modelname="test",
        xll=500.0,
        crs="epsg:2193",
        rotation=12.5,
        start_datetime="1/1/2016",
    )
    dis = ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.epsg == 2193
    assert ml.modelgrid.idomain is None
    ml.model_ws = function_tmpdir

    mt = Mt3dms(
        modelname="test_mt",
        modflowmodel=ml,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert mt.modelgrid.xoffset == ml.modelgrid.xoffset
    assert mt.modelgrid.yoffset == ml.modelgrid.yoffset
    assert mt.modelgrid.crs == ml.modelgrid.crs
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
    assert mt.modelgrid.crs == ml.modelgrid.crs == swt.modelgrid.crs
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
    assert mt.modelgrid.crs == ml.modelgrid.crs == swt.modelgrid.crs
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)


@requires_exe("mp7", "mf2005")
def test_exe_selection(example_data_path, function_tmpdir):
    model_path = example_data_path / "freyberg"
    namfile_path = model_path / "freyberg.nam"

    # no selection defaults to mf2005
    exe_name = "mf2005"
    assert Path(Modflow().exe_name).stem == exe_name
    assert Path(Modflow(exe_name=None).exe_name).stem == exe_name
    assert (
        Path(Modflow.load(namfile_path, model_ws=model_path).exe_name).stem
        == exe_name
    )
    assert (
        Path(
            Modflow.load(
                namfile_path, exe_name=None, model_ws=model_path
            ).exe_name
        ).stem
        == exe_name
    )

    # user-specified (just for testing - there is no legitimate reason
    # to use mp7 with Modflow but Modpath7 derives from BaseModel too)
    exe_name = "mp7"
    assert Path(Modflow(exe_name=exe_name).exe_name).stem == exe_name
    assert (
        Path(
            Modflow.load(
                namfile_path, exe_name=exe_name, model_ws=model_path
            ).exe_name
        ).stem
        == exe_name
    )

    # init/load should warn if exe DNE
    exe_name = "not_an_exe"
    with pytest.warns(UserWarning):
        ml = Modflow(exe_name=exe_name)
    with pytest.warns(UserWarning):
        ml = Modflow.load(namfile_path, exe_name=exe_name, model_ws=model_path)

    # run should error if exe DNE
    ml = Modflow.load(namfile_path, exe_name=exe_name, model_ws=model_path)
    ml.change_model_ws(function_tmpdir)
    ml.write_input()
    with pytest.raises(ValueError):
        ml.run_model()


def test_free_format_flag(function_tmpdir):
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

    ms.model_ws = function_tmpdir
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


def test_sr(function_tmpdir):
    ws = function_tmpdir
    m = Modflow(
        "test",
        model_ws=ws,
        xll=12345,
        yll=12345,
        crs=26916,
    )
    ModflowDis(m, 10, 10, 10)
    m.write_input()
    mm = Modflow.load("test.nam", model_ws=ws)
    extents = mm.modelgrid.extent
    if extents[2] != 12345:
        raise AssertionError()
    if extents[3] != 12355:
        raise AssertionError()
    assert mm.modelgrid.epsg == 26916

    mm.dis.top = 5000

    if not np.allclose(mm.dis.top.array, mm.modelgrid.top):
        raise AssertionError("modelgrid failed dynamic update test")


def test_mf6_update_grid(example_data_path):
    ml_path = example_data_path / "mf6" / "test001a_Tharmonic"
    sim = MFSimulation.load(sim_ws=ml_path)
    gwf = sim.get_model("flow15")
    mg = gwf.modelgrid
    gwf.dis.top = 12

    assert np.allclose(
        gwf.dis.top.array, gwf.modelgrid.top
    ), "StructuredGrid failed dynamic update test"

    # disv
    ml_path = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=ml_path)
    gwf = sim.get_model("gwf_1")
    mg = gwf.modelgrid
    gwf.disv.top = 6.12

    assert np.allclose(
        gwf.disv.top.array, gwf.modelgrid.top
    ), "VertexGrid failed dynamic update test"

    # disu
    ml_path = example_data_path / "mf6" / "test006_gwf3"
    sim = MFSimulation.load(sim_ws=ml_path)
    gwf = sim.get_model("gwf_1")
    mg = gwf.modelgrid
    gwf.disu.top = 101

    assert np.allclose(
        gwf.disu.top.array, gwf.modelgrid.top
    ), "UnstructuredGrid failed dynamic update test"


def test_load_twri_grid(example_data_path):
    mpath = example_data_path / "mf2005_test"
    name = "twri.nam"
    ml = Modflow.load(name, model_ws=mpath, check=False)
    mg = ml.modelgrid
    assert isinstance(
        mg, StructuredGrid
    ), "modelgrid is not an StructuredGrid instance"
    shape = (3, 15, 15)
    assert (
        mg.shape == shape
    ), f"modelgrid shape {mg.shape} not equal to {shape}"
    thickness = mg.cell_thickness
    shape = (5, 15, 15)
    assert (
        thickness.shape == shape
    ), f"cell_thickness shape {thickness.shape} not equal to {shape}"


def test_mg(function_tmpdir):
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
    t = ms.modelgrid.cell_thickness

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

    ms.model_ws = function_tmpdir

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


def test_namfile_readwrite(function_tmpdir, example_data_path):
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    xll, yll = 272300, 5086000
    ws = function_tmpdir
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

    # test reading and writing of modelgrid information to namfile
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
        model_ws=example_data_path / "freyberg_multilayer_transient",
        verbose=False,
        check=False,
        exe_name="mfnwt",
    )

    assert ml.modelgrid.xoffset == ml.modelgrid._xul_to_xll(619653)
    assert ml.modelgrid.yoffset == ml.modelgrid._yul_to_yll(3353277)
    assert ml.modelgrid.angrot == 15.0


def test_read_usgs_model_reference(function_tmpdir, model_reference_path):
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    # xll, yll = 272300, 5086000

    mrf_path = function_tmpdir / model_reference_path.name
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

    ws = function_tmpdir
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

    # test reading of proj4 string from usgs.model.reference
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
    assert m2.modelgrid.crs == mg.crs

    # test reading epsg code from usgs.model.reference
    shutil.copy(mrf_path, f"{mrf_path}_copy")
    with open(f"{mrf_path}_copy") as src:
        with open(mrf_path, "w") as dst:
            for line in src:
                if "epsg" in line:
                    line = "epsg 26916\n"
                if "proj4" in line:
                    line = "# proj4\n"
                dst.write(line)

    m2 = Modflow.load("junk.nam", model_ws=ws)
    m2.modelgrid.read_usgs_model_reference_file(mrf_path)

    assert m2.modelgrid.epsg == 26916
    # have to delete this, otherwise it will mess up other tests
    to_del = glob.glob(f"{mrf_path}*")
    for f in to_del:
        if os.path.exists(f):
            os.remove(os.path.join(f))


def mf2005_model_namfiles():
    path = _example_data_path / "mf2005_test"
    return [str(p) for p in path.glob("*.nam")]


def parameters_model_namfiles():
    path = _example_data_path / "parameters"
    skip = ["twrip.nam", "twrip_upw.nam"]  # TODO: why do these fail?
    return [str(p) for p in path.glob("*.nam") if p.name not in skip]


@requires_exe("mf2005")
@pytest.mark.parametrize(
    "namfile", mf2005_model_namfiles() + parameters_model_namfiles()
)
def test_mf2005_test_models_load(example_data_path, namfile):
    assert not Modflow.load(
        namfile,
        model_ws=example_data_path / "mf2005_test",
        version="mf2005",
        verbose=True,
    ).load_fail


@requires_exe("mf2005")
@pytest.mark.parametrize("namfile", parameters_model_namfiles())
def test_parameters_models_load(parameters_model_path, namfile):
    assert not Modflow.load(
        namfile,
        model_ws=parameters_model_path,
        version="mf2005",
        verbose=True,
    ).load_fail


@pytest.mark.parametrize("namfile", mf2005_model_namfiles())
def test_mf2005_test_models_loadonly(example_data_path, namfile):
    assert not Modflow.load(
        namfile,
        model_ws=example_data_path / "mf2005_test",
        version="mf2005",
        verbose=True,
        load_only=["bas6"],
        check=False,
    ).load_fail


@pytest.mark.slow
def test_write_irch(function_tmpdir, example_data_path):
    mpath = example_data_path / "freyberg_multilayer_transient"
    nam_file = "freyberg.nam"
    m = Modflow.load(
        nam_file,
        model_ws=mpath,
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

    m.change_model_ws(function_tmpdir)
    m.write_input()

    mm = Modflow.load(
        nam_file,
        model_ws=function_tmpdir,
        forgive=False,
        verbose=True,
        check=False,
    )
    for kper in range(m.nper):
        arr = irch[kper]
        aarr = mm.rch.irch[kper].array
        d = arr - aarr
        assert np.abs(d).sum() == 0


def test_mflist_external(function_tmpdir):
    ext = function_tmpdir / "ws"
    ml = Modflow(
        "mflist_test",
        model_ws=function_tmpdir,
        external_path=ext.name,
    )

    dis = ModflowDis(ml, 1, 10, 10, nper=3, perlen=1.0)
    wel_data = {
        0: [[0, 0, 0, -1], [1, 1, 1, -1]],
        1: [[0, 0, 0, -2], [1, 1, 1, -1]],
    }
    wel = ModflowWel(ml, stress_period_data=wel_data)
    ml.change_model_ws(ext)
    ml.write_input()

    ml1 = Modflow.load(
        "mflist_test.nam",
        model_ws=ext,
        verbose=True,
        forgive=False,
        check=False,
    )

    assert np.array_equal(ml.wel[0], ml1.wel[0])
    assert np.array_equal(ml.wel[1], ml1.wel[1])

    ml1.write_input()

    # ml = Modflow(
    #     "mflist_test",
    #     model_ws=str(function_tmpdir),
    #     external_path=str(function_tmpdir / "ref"),
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
def test_single_mflist_entry_load(function_tmpdir, example_data_path):
    m = Modflow.load(
        "freyberg.nam",
        model_ws=example_data_path / "freyberg",
        load_only=["WEL"],
        check=False,
    )
    w = m.wel
    spd = w.stress_period_data
    ModflowWel(m, stress_period_data={0: [0, 0, 0, 0.0]})
    m.external_path = "external"
    m.change_model_ws(function_tmpdir, reset_external=True)
    m.write_input()

    mm = Modflow.load(
        "freyberg.nam",
        model_ws=function_tmpdir,
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


_mf2005_test_path = _example_data_path / "mf2005_test"
_mf2005_namfiles = [
    Path(_mf2005_test_path) / f
    for f in _mf2005_test_path.rglob("*")
    if f.suffix == ".nam"
]


@pytest.mark.parametrize("namfile", _mf2005_namfiles)
def test_checker_on_load(namfile):
    # load all of the models in the mf2005_test folder
    # model level checks are performed by default on load()

    # with pathlib.Path
    model = Modflow.load(namfile, model_ws=namfile.parent)
    assert isinstance(model, Modflow), "Not a Modflow instance"

    # with str paths
    f = os.path.basename(namfile)
    d = os.path.dirname(namfile)
    model = Modflow.load(f, model_ws=d)
    assert isinstance(model, Modflow), "Not a Modflow instance"


@pytest.mark.parametrize("str_path", [True, False])
def test_manual_check(function_tmpdir, str_path):
    namfile_path = _mf2005_namfiles[0]
    summary_path = function_tmpdir / "summary"
    model = Modflow.load(namfile_path, model_ws=namfile_path.parent)
    model.change_model_ws(function_tmpdir)
    model.check(str(summary_path) if str_path else summary_path, verbose=True)
    assert summary_path.is_file()


def test_bcs_check(function_tmpdir):
    mf = Modflow(version="mf2005", model_ws=function_tmpdir)

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
    riv_spd = pd.DataFrame(
        [[0, 0, 0, 0, 101.0, 10.0, 100.0], [0, 0, 0, 1, 80.0, 10.0, 90.0]],
        columns=["per", "k", "i", "j", "stage", "cond", "rbot"],
    )

    pers = riv_spd.groupby("per")
    riv_spd = {i: pers.get_group(i).drop("per", axis=1) for i in [0]}
    riv = ModflowRiv(
        mf,
        stress_period_data=riv_spd,
        # stress_period_data={
        #     0: [[0, 0, 0, 101, 10, 100], [0, 0, 1, 80, 10, 90]]
        # },
    )
    chk = ghb.check()
    assert chk.summary_array["desc"][0] == "BC in inactive cell"
    chk = riv.check()
    assert chk.summary_array["desc"][4] == "RIV stage below rbots"
    assert np.array_equal(chk.summary_array["j"], np.array([0, 1, 1, 1, 1]))


def test_path_params_and_props(function_tmpdir, module_tmpdir):
    # properties should be set to string abspaths regardless of
    # pathlib.Path or str arguments

    mf = Modflow(
        version="mf2005", model_ws=function_tmpdir, external_path=module_tmpdir
    )
    assert mf.model_ws == str(function_tmpdir)
    assert mf.external_path == str(module_tmpdir)

    mf = Modflow(
        version="mf2005",
        model_ws=str(function_tmpdir),
        external_path=str(module_tmpdir),
    )
    assert mf.model_ws == str(function_tmpdir)
    assert mf.external_path == str(module_tmpdir)


def test_properties_check(function_tmpdir):
    mf = Modflow(
        version="mf2005",
        model_ws=function_tmpdir,
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
    chk = oc.check()  # check passed
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


def test_rchload(function_tmpdir):
    nlay = 2
    nrow = 3
    ncol = 4
    nper = 2

    # create model 1
    ws = function_tmpdir
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


def test_default_oc_stress_period_data(function_tmpdir):
    m = Modflow(model_ws=function_tmpdir, verbose=True)
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


def test_mfcbc(function_tmpdir):
    m = Modflow(verbose=True, model_ws=function_tmpdir)
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
    ml = Modflow(modelname="t1", model_ws=function_tmpdir, verbose=True)
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


def test_load_with_list_reader(function_tmpdir):
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
        model_ws=function_tmpdir,
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
    fname = os.path.join(function_tmpdir, "original.ghb")
    with open(fname, "w") as f:
        f.write(f"{ghbra.shape[0]} 0\n")
        for kper in range(nper):
            f.write(f"{ghbra.shape[0]} 0\n")
            f.write("open/close original.ghb.dat\n")

    # write ghb list
    sfacghb = 5
    fname = os.path.join(function_tmpdir, "original.ghb.dat")
    with open(fname, "w") as f:
        f.write(f"sfac {sfacghb}\n")
        for k, i, j, stage, cond in ghbra:
            f.write(f"{k + 1} {i + 1} {j + 1} {stage} {cond}\n")

    # rewrite drn
    fname = os.path.join(function_tmpdir, "original.drn")
    with open(fname, "w") as f:
        f.write(f"{drnra.shape[0]} 0\n")
        for kper in range(nper):
            f.write(f"{drnra.shape[0]} 0\n")
            f.write("external 71\n")

    # write drn list
    sfacdrn = 1.5
    fname = os.path.join(function_tmpdir, "original.drn.dat")
    with open(fname, "w") as f:
        for kper in range(nper):
            f.write(f"sfac {sfacdrn}\n")
            for k, i, j, stage, cond in drnra:
                f.write(f"{k + 1} {i + 1} {j + 1} {stage} {cond}\n")

    # rewrite wel
    fname = os.path.join(function_tmpdir, "original.wel")
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
    fname = os.path.join(function_tmpdir, "original.wel.bin")
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
    m2 = Modflow.load(fname, model_ws=function_tmpdir, verbose=False)
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


@requires_exe("mf2005")
@pytest.mark.parametrize(
    "container",
    ["recarray", "dataframe", "dict_of_recarray", "dict_of_dataframe"],
)
def test_pkg_data_containers(function_tmpdir, container):
    """Test various containers for package data (list, ndarray, recarray, dataframe, dict of such)"""

    nlay = 1
    nrow = 10
    ncol = 10
    nper = 3

    name = "pkg_data"
    ws = function_tmpdir

    # create the ghbs
    ghb_ra = ModflowGhb.get_empty(20)
    l = 0
    for i in range(nrow):
        ghb_ra[l] = (0, i, 0, 1.0, 100.0 + i)
        l += 1
        ghb_ra[l] = (0, i, ncol - 1, 1.0, 200.0 + i)
        l += 1
    ghb_spd = {0: ghb_ra}

    # well pkg, setup data per 'container' parameter
    # to test support for various container types
    wel_ra = ModflowWel.get_empty(2)
    wel_ra[0] = (0, 1, 1, -5.0)
    wel_ra[1] = (0, nrow - 3, ncol - 3, -10.0)
    wel_dtype = np.dtype(
        [
            ("k", int),
            ("i", int),
            ("j", int),
            ("q", np.float32),
        ]
    )
    df_per = pd.DataFrame(wel_ra)
    if "dict_of_recarray" in container:
        wel_spd = {0: wel_ra}
    elif "dict_of_dataframe" in container:
        wel_spd = {0: df_per}
    elif "recarray" in container:
        wel_spd = wel_ra
    elif "dataframe" in container:
        wel_spd = df_per

    m = Modflow(name, model_ws=ws)
    dis = ModflowDis(
        m,
        nper=nper,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        top=nlay,
        botm=list(range(nlay)),
    )
    ghb = ModflowGhb(m, stress_period_data=ghb_spd)
    wel = ModflowWel(m, stress_period_data=wel_spd)
    bas = ModflowBas(m)
    lpf = ModflowLpf(m)
    pcg = ModflowPcg(m)
    oc = ModflowOc(m)

    # write and run the model
    m.write_input()
    success, buff = m.run_model(silent=False, report=True)
    from pprint import pformat

    assert success, pformat(buff)


def get_perftest_model(ws, name):
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
def test_model_init_time(function_tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    benchmark(lambda: get_perftest_model(ws=function_tmpdir, name=name))


@pytest.mark.slow
def test_model_write_time(function_tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = get_perftest_model(ws=function_tmpdir, name=name)
    benchmark(lambda: model.write_input())


@pytest.mark.slow
def test_model_load_time(function_tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = get_perftest_model(ws=function_tmpdir, name=name)
    model.write_input()
    benchmark(
        lambda: Modflow.load(
            f"{name}.nam", model_ws=function_tmpdir, check=False
        )
    )
