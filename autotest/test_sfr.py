import glob
import io
import os
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from autotest.conftest import get_example_data_path, get_project_root_path
from flopy.discretization import StructuredGrid
from flopy.modflow import Modflow, ModflowDis, ModflowSfr2, ModflowStr
from flopy.modflow.mfsfr2 import check
from flopy.utils.recarray_utils import create_empty_recarray
from flopy.utils.sfroutputfile import SfrFile


@pytest.fixture(scope="session")
def mf2005_model_path(example_data_path):
    return example_data_path / "mf2005_test"


@pytest.fixture(scope="session")
def hydmod_model_path(example_data_path):
    return example_data_path / "hydmod_test"


@pytest.fixture(scope="session")
def sfr_examples_path(example_data_path):
    return example_data_path / "sfr_examples"


@pytest.fixture(scope="session")
def sfr_test_model_path(example_data_path):
    return example_data_path / "sfr_test"


def sfr_models():
    return {
        0: {"mfnam": "test1ss.nam", "sfrfile": "test1ss.sfr"},
        1: {"mfnam": "test1tr.nam", "sfrfile": "test1tr.sfr"},
        2: {"mfnam": "testsfr2_tab.nam", "sfrfile": "testsfr2_tab_ICALC1.sfr"},
        3: {"mfnam": "testsfr2_tab.nam", "sfrfile": "testsfr2_tab_ICALC2.sfr"},
        4: {"mfnam": "testsfr2.nam", "sfrfile": "testsfr2.sfr"},
        5: {"mfnam": "UZFtest2.nam", "sfrfile": "UZFtest2.sfr"},
    }


@pytest.fixture(scope="session")
def sfr_data():
    dtype = np.dtype(
        [("k", int), ("i", int), ("j", int), ("iseg", int), ("ireach", int)]
    )
    r = create_empty_recarray(27, dtype=dtype)
    r["i"] = [
        3,
        4,
        5,
        7,
        8,
        9,
        0,
        1,
        2,
        4,
        4,
        5,
        0,
        0,
        0,
        3,
        4,
        5,
        0,
        1,
        2,
        4,
        5,
        6,
        2,
        2,
        2,
    ]
    r["j"] = [
        0,
        1,
        2,
        6,
        6,
        6,
        6,
        6,
        6,
        3,
        4,
        5,
        9,
        8,
        7,
        6,
        6,
        6,
        0,
        0,
        0,
        6,
        6,
        6,
        9,
        8,
        7,
    ]
    r["iseg"] = sorted(list(range(1, 10)) * 3)
    r["ireach"] = [1, 2, 3] * 9

    d = create_empty_recarray(9, dtype=np.dtype([("nseg", int), ("outseg", int)]))
    d["nseg"] = range(1, 10)
    d["outseg"] = [4, 0, 6, 8, 3, 8, 1, 2, 8]
    return r, d


@pytest.mark.parametrize("case", list(sfr_models().values())[:-1])
def test_load_sfr(case, mf2005_model_path):
    m = Modflow()
    sfr = ModflowSfr2.load(mf2005_model_path / case["sfrfile"], m)


def test_sfr(function_tmpdir, mf2005_model_path, sfr_test_model_path):
    def sfr_process(mfnam, sfrfile, model_ws, outfolder):
        m = Modflow.load(mfnam, model_ws=model_ws, verbose=False)
        sfr = m.get_package("SFR")

        if not os.path.isdir(outfolder):
            os.makedirs(outfolder, exist_ok=True)
        outpath = os.path.join(outfolder, sfrfile)
        sfr.write_file(outpath)

        m.remove_package("SFR")
        sfr2 = ModflowSfr2.load(outpath, m)

        assert np.all(sfr2.reach_data == sfr.reach_data)
        assert np.all(sfr2.dataset_5 == sfr.dataset_5)
        for k, v in sfr2.segment_data.items():
            assert np.all(v == sfr.segment_data[k])
        for k, v in sfr2.channel_flow_data.items():
            assert np.all(v == sfr.channel_flow_data[k])
        for k, v in sfr2.channel_geometry_data.items():
            assert np.all(v == sfr.channel_geometry_data[k])

        return m, sfr

    m, sfr = sfr_process(
        "test1ss.nam", "test1ss.sfr", mf2005_model_path, function_tmpdir
    )

    m, sfr = sfr_process(
        "test1tr.nam", "test1tr.sfr", mf2005_model_path, function_tmpdir
    )

    m, sfr = sfr_process(
        "testsfr2_tab.nam",
        "testsfr2_tab_ICALC1.sfr",
        mf2005_model_path,
        function_tmpdir,
    )

    assert list(sfr.dataset_5.keys()) == list(range(0, 50))

    m, sfr = sfr_process(
        "testsfr2_tab.nam",
        "testsfr2_tab_ICALC2.sfr",
        mf2005_model_path,
        function_tmpdir,
    )

    assert sfr.channel_geometry_data[0][1] == [
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        [6.0, 4.5, 3.5, 0.0, 0.3, 3.5, 4.5, 6.0],
    ]

    m, sfr = sfr_process(
        "testsfr2.nam", "testsfr2.sfr", mf2005_model_path, function_tmpdir
    )

    assert round(sum(sfr.segment_data[49][0]), 7) == 3.9700007

    m, sfr = sfr_process(
        "UZFtest2.nam", "UZFtest2.sfr", mf2005_model_path, function_tmpdir
    )

    assert isinstance(sfr.plot()[0], matplotlib.axes.Axes)  # test the plot() method
    matplotlib.pyplot.close()

    def interpolate_to_reaches(sfr):
        reach_data = sfr.reach_data
        segment_data = sfr.segment_data[0]
        for reachvar, segvars in {
            "strtop": ("elevup", "elevdn"),
            "strthick": ("thickm1", "thickm2"),
            "strhc1": ("hcond1", "hcond2"),
        }.items():
            reach_data[reachvar] = sfr._interpolate_to_reaches(*segvars)
            for seg in segment_data.nseg:
                reaches = reach_data[reach_data.iseg == seg]
                dist = np.cumsum(reaches.rchlen) - 0.5 * reaches.rchlen
                fp = [
                    segment_data[segment_data["nseg"] == seg][segvars[0]][0],
                    segment_data[segment_data["nseg"] == seg][segvars[1]][0],
                ]
                xp = [dist[0], dist[-1]]
                assert (
                    np.sum(np.abs(reaches[reachvar] - np.interp(dist, xp, fp).tolist()))
                    < 0.01
                )
        return reach_data

    # trout lake example (only sfr file is included)
    # can add tests for sfr connection with lak package
    sfr = ModflowSfr2.load(sfr_test_model_path / "TL2009.sfr", Modflow())
    # convert sfr package to reach input
    sfr.reachinput = True
    sfr.isfropt = 1
    sfr.reach_data = interpolate_to_reaches(sfr)
    sfr.get_slopes(minimum_slope=-100, maximum_slope=100)
    reach_inds = 29
    outreach = sfr.reach_data.outreach[reach_inds]
    out_inds = np.asarray(sfr.reach_data.reachID == outreach).nonzero()
    assert (
        sfr.reach_data.slope[reach_inds]
        == (sfr.reach_data.strtop[reach_inds] - sfr.reach_data.strtop[out_inds])
        / sfr.reach_data.rchlen[reach_inds]
    )
    chk = sfr.check()
    assert sfr.reach_data.slope.min() < 0.0001 and "minimum slope" in chk.warnings
    # negative segments for lakes shouldn't be included in segment numbering order check
    assert "segment numbering order" not in chk.warnings
    sfr.reach_data.slope[0] = 1.1
    chk.slope(maximum_slope=1.0)
    assert "maximum slope" in chk.warnings


def test_sfr_renumbering():
    # test segment renumbering

    dtype = np.dtype([("iseg", int), ("ireach", int)])
    r = create_empty_recarray(27, dtype)
    r["iseg"] = sorted(list(range(1, 10)) * 3)
    r["ireach"] = [1, 2, 3] * 9

    dtype = np.dtype([("nseg", int), ("outseg", int)])
    d = create_empty_recarray(9, dtype)
    d["nseg"] = range(1, 10)
    d["outseg"] = [4, 0, 6, 8, 3, 8, 1, 2, 8]
    m = Modflow()
    sfr = ModflowSfr2(m, reach_data=r, segment_data={0: d})
    chk = sfr.check()
    assert "segment numbering order" in chk.warnings
    sfr.renumber_segments()
    chk = sfr.check()
    assert "continuity in segment and reach numbering" in chk.passed
    assert "segment numbering order" in chk.passed

    # test renumbering non-consecutive segment numbers
    r["iseg"] *= 2
    r["ireach"] = [1, 2, 3] * 9

    dtype = np.dtype([("nseg", int), ("outseg", int)])
    d = create_empty_recarray(9, dtype)
    d["nseg"] = np.arange(1, 10) * 2
    d["outseg"] = np.array([4, 0, 6, 8, 3, 8, 1, 2, 8]) * 2
    m = Modflow()
    sfr = ModflowSfr2(m, reach_data=r, segment_data={0: d})
    chk = sfr.check()
    assert "segment numbering order" in chk.warnings
    sfr.renumber_segments()
    chk = sfr.check()
    assert "continuity in segment and reach numbering" in chk.passed
    assert "segment numbering order" in chk.passed

    # test computing of outreaches
    assert np.array_equal(
        sfr.reach_data.outreach,
        np.array(
            [
                2,
                3,
                7,
                5,
                6,
                10,
                8,
                9,
                16,
                11,
                12,
                19,
                14,
                15,
                22,
                17,
                18,
                22,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                0,
            ]
        ),
    )
    # test slope
    sfr.reach_data["rchlen"] = [10] * 3 * 5 + [100] * 2 * 3 + [1] * 2 * 3
    strtop = np.zeros(len(sfr.reach_data))
    strtop[2] = 0.3
    strtop[21] = -0.2
    strtop[22] = -0.4
    sfr.reach_data["strtop"] = strtop
    default_slope = 0.0001
    sfr.get_slopes(default_slope=default_slope)
    sl1 = sfr.reach_data.slope[2]

    def isequal(v1, v2):
        return np.abs(v1 - v2) < 1e-6

    assert isequal(sfr.reach_data.slope[2], 0.03)
    assert isequal(sfr.reach_data.slope[14], 0.02)
    assert isequal(sfr.reach_data.slope[20], sfr.reach_data.slope[17])
    assert isequal(sfr.reach_data.slope[21], 0.2)
    assert isequal(sfr.reach_data.slope[-1], default_slope)


def test_const(sfr_data):
    m = Modflow()
    dis = ModflowDis(m, 1, 10, 10, lenuni=2, itmuni=4)
    m.modelgrid = StructuredGrid(
        delc=dis.delc.array,
        delr=dis.delr.array,
    )
    r, d = sfr_data
    sfr = ModflowSfr2(m, reach_data=r, segment_data={0: d})
    assert sfr.const == 86400.0
    m.dis.itmuni = 1.0
    m.sfr.const = None
    assert sfr.const == 1.0
    m.dis.lenuni = 1.0
    m.sfr.const = None
    assert sfr.const == 1.486
    m.dis.itmuni = 4.0
    m.sfr.const = None
    assert sfr.const == 1.486 * 86400.0
    assert True


@requires_pkg("pyshp", "shapely", name_map={"pyshp": "shapefile"})
def test_export(function_tmpdir, sfr_data):
    m = Modflow()
    dis = ModflowDis(m, 1, 10, 10, lenuni=2, itmuni=4)

    m.export(function_tmpdir / "grid.shp")
    r, d = sfr_data
    sfr = ModflowSfr2(m, reach_data=r, segment_data={0: d})
    sfr.segment_data[0]["flow"][-1] = 1e4
    sfr.stress_period_data.export(function_tmpdir / "sfr.shp", sparse=True)
    sfr.export_linkages(function_tmpdir / "linkages.shp")
    sfr.export_outlets(function_tmpdir / "outlets.shp")
    sfr.export_transient_variable(function_tmpdir / "inlets.shp", "flow")

    from flopy.export.shapefile_utils import shp2recarray

    ra = shp2recarray(function_tmpdir / "inlets.shp")
    assert ra.flow0[0] == 1e4
    ra = shp2recarray(function_tmpdir / "outlets.shp")
    assert ra.iseg[0] + ra.ireach[0] == 5
    ra = shp2recarray(function_tmpdir / "linkages.shp")
    crds = np.array(list(ra.geometry[2].coords))
    assert np.array_equal(crds, np.array([[2.5, 4.5], [3.5, 5.5]]))
    ra = shp2recarray(function_tmpdir / "sfr.shp")
    assert ra.iseg.sum() == sfr.reach_data.iseg.sum()
    assert ra.ireach.sum() == sfr.reach_data.ireach.sum()
    y = np.concatenate([np.array(g.exterior)[:, 1] for g in ra.geometry])
    x = np.concatenate([np.array(g.exterior)[:, 0] for g in ra.geometry])

    assert (x.min(), x.max(), y.min(), y.max()) == m.modelgrid.extent
    assert ra[(ra.iseg == 2) & (ra.ireach == 1)]["geometry"][0].bounds == (
        6.0,
        2.0,
        7.0,
        3.0,
    )


def test_example(mf2005_model_path):
    m = Modflow.load(
        "test1ss.nam",
        version="mf2005",
        exe_name="mf2005",
        model_ws=mf2005_model_path,
        load_only=["ghb", "evt", "rch", "dis", "bas6", "oc", "sip", "lpf"],
    )
    reach_data = np.genfromtxt(
        "../examples/data/sfr_examples/test1ss_reach_data.csv",
        delimiter=",",
        names=True,
    )
    segment_data = np.genfromtxt(
        "../examples/data/sfr_examples/test1ss_segment_data.csv",
        delimiter=",",
        names=True,
    )

    channel_flow_data = {
        0: {
            1: [
                [0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0],
                [0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.1, 1.25, 1.4, 1.7, 2.6],
                [3.0, 3.5, 4.2, 5.3, 7.0, 8.5, 12.0, 14.0, 17.0, 20.0, 22.0],
            ]
        }
    }
    channel_geometry_data = {
        0: {
            7: [
                [0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
                [20.0, 13.0, 10.0, 2.0, 0.0, 10.0, 13.0, 20.0],
            ],
            8: [
                [0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
                [25.0, 17.0, 13.0, 4.0, 0.0, 10.0, 16.0, 20.0],
            ],
        }
    }

    nstrm = len(reach_data)  # number of reaches
    nss = len(segment_data)  # number of segments
    nsfrpar = 0  # number of parameters (not supported)
    nparseg = 0
    const = 1.486  # constant for manning's equation, units of cfs
    dleak = 0.0001  # closure tolerance for stream stage computation
    ipakcb = 53  # flag for writing SFR output to cell-by-cell budget (on unit 53)
    istcb2 = 81  # flag for writing SFR output to text file
    dataset_5 = {0: [nss, 0, 0]}  # dataset 5 (see online guide)

    sfr = ModflowSfr2(
        m,
        nstrm=nstrm,
        nss=nss,
        const=const,
        dleak=dleak,
        ipakcb=ipakcb,
        istcb2=istcb2,
        reach_data=reach_data,
        segment_data=segment_data,
        channel_geometry_data=channel_geometry_data,
        channel_flow_data=channel_flow_data,
        dataset_5=dataset_5,
    )

    assert istcb2 in m.output_units
    assert True

    # test handling of a 0-D array (produced by genfromtxt sometimes)
    segment_data = np.array(segment_data[0])
    reach_data = reach_data[reach_data["iseg"] == 1]
    nss = 1
    sfr = ModflowSfr2(
        m,
        nstrm=nstrm,
        nss=nss,
        const=const,
        dleak=dleak,
        ipakcb=ipakcb,
        istcb2=istcb2,
        reach_data=reach_data,
        segment_data=segment_data,
        channel_geometry_data=channel_geometry_data,
        channel_flow_data=channel_flow_data,
        dataset_5=dataset_5,
    )

    # test default construction of dataset_5
    sfr2 = ModflowSfr2(
        m,
        nstrm=nstrm,
        nss=nss,
        const=const,
        dleak=dleak,
        ipakcb=ipakcb,
        istcb2=istcb2,
        reach_data=reach_data,
        segment_data=segment_data,
        channel_geometry_data=channel_geometry_data,
        channel_flow_data=channel_flow_data,
    )
    assert len(sfr2.dataset_5) == 1
    assert sfr2.dataset_5[0][0] == sfr2.nss
    nper = 9
    m.dis.nper = nper
    assert len(sfr2.dataset_5) == nper
    for i in range(1, nper):
        assert sfr2.dataset_5[i][0] == -1


def test_no_ds_6bc(function_tmpdir):
    """Test case where datasets 6b and 6c aren't read
    (e.g., see table at https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/sfr.htm)
    """
    sfrfiletxt = (
        "REACHINPUT\n"
        "2 2 0 0 128390 0.0001 119 0 3 10 1 30 0 4 0.75 91.54\n"
        "1 1 1 1 1 1.0 1.0 0.001 1 1 .3 0.02 3.5 0.7\n"
        "1 2 2 2 1 1.0 0.5 0.001 1 1 .3 0.02 3.5 0.7\n"
        "2 2 0\n"
        "1 2 2 0 0 0 0 0 0.041 0.111\n"
        "0 3.64 7.28 10.92 14.55 18.19 21.83 25.47\n"
        "2.55 1.02 0.76 0 0.25 0.76 1.02 2.55\n"
        "2 2 0 0 0 0 0 0 0.041 0.111\n"
        "0 3.96 7.92 11.88 15.83 19.79 23.75 27.71\n"
        "2.77 1.11 0.83 0 0.28 0.83 1.11 2.77\n"
    )
    sfrfile = io.StringIO(sfrfiletxt)
    m = Modflow("junk", model_ws=function_tmpdir)
    sfr = ModflowSfr2.load(sfrfile, model=m)
    assert len(sfr.segment_data[0]) == 2
    assert len(sfr.channel_geometry_data[0]) == 2
    assert len(sfr.channel_geometry_data[0][1]) == 2
    for i in range(2):
        assert len(sfr.channel_geometry_data[0][1][i]) == 8
        assert sum(sfr.channel_geometry_data[0][1][i]) > 0.0

    sfrfile2 = function_tmpdir / "junk.sfr"
    sfr.write_file()
    sfr = ModflowSfr2.load(sfrfile2, model=m)
    assert len(sfr.segment_data[0]) == 2
    assert len(sfr.channel_geometry_data[0]) == 2
    assert len(sfr.channel_geometry_data[0][1]) == 2
    for i in range(2):
        assert len(sfr.channel_geometry_data[0][1][i]) == 8
        assert sum(sfr.channel_geometry_data[0][1][i]) > 0.0


def test_ds_6d_6e_disordered(function_tmpdir, hydmod_model_path):
    m = Modflow.load("test1tr2.nam", model_ws=hydmod_model_path)
    m.change_model_ws(function_tmpdir)
    m.write_input()

    m2 = Modflow.load("test1tr2.nam", model_ws=function_tmpdir)

    sfr = m.get_package("SFR")
    sfr2 = m2.get_package("SFR")

    if len(sfr.graph) != len(sfr2.graph):
        raise AssertionError

    if len(sfr.segment_data[0]) != len(sfr2.segment_data[0]):
        raise AssertionError

    for kper, d in sfr.channel_flow_data.items():
        for seg, value in d.items():
            if not np.allclose(value, sfr2.channel_flow_data[kper][seg]):
                raise AssertionError

    for kper, d in sfr.channel_geometry_data.items():
        for seg, value in d.items():
            if not np.allclose(value, sfr2.channel_geometry_data[kper][seg]):
                raise AssertionError


def test_disordered_reachdata_fields(function_tmpdir, hydmod_model_path):
    m = Modflow.load("test1tr2.nam", model_ws=hydmod_model_path)
    sfr = m.get_package("SFR")
    orig_reach_data = sfr.reach_data
    # build shuffled rec array
    shuffled_fields = list(set(orig_reach_data.dtype.names))
    data = []
    names = []
    formats = []
    for field in shuffled_fields:
        data.append(orig_reach_data[field])
        names.append(field)
        formats.append(orig_reach_data.dtype[field].str)
    reach_data = np.rec.fromarrays(data, names=names, formats=formats)
    m.sfr.reach_data = reach_data
    m.change_model_ws(function_tmpdir)
    m.write_input()


def test_transient_example(function_tmpdir, mf2005_model_path):
    gpth = mf2005_model_path / "testsfr2.*"
    for f in glob.glob(str(gpth)):
        shutil.copy(f, function_tmpdir)
    m = Modflow.load("testsfr2.nam", model_ws=function_tmpdir)

    # test handling of unformatted output file
    m.sfr.istcb2 = -49
    m.set_output_attribute(unit=abs(m.sfr.istcb2), attr={"binflag": True})
    m.write_input()
    m2 = Modflow.load("testsfr2.nam", model_ws=function_tmpdir)
    assert m2.sfr.istcb2 == -49
    assert m2.get_output_attribute(unit=abs(m2.sfr.istcb2), attr="binflag")


@pytest.mark.skip("Pending https://github.com/modflowpy/flopy/issues/1471")
def test_assign_layers(function_tmpdir):
    m = Modflow(model_ws=function_tmpdir)
    m.dis = ModflowDis(
        nrow=1,
        ncol=6,
        nlay=7,
        botm=np.array(
            [
                [50.0, 49.0, 42.0, 27.0, 6.0, -33.0],
                [-196.0, -246.0, -297.0, -351.0, -405.0, -462.0],
                [-817.0, -881.0, -951.0, -1032.0, -1141.0, -1278.0],
                [-1305.0, -1387.0, -1466.0, -1546.0, -1629.0, -1720.0],
                [-2882.0, -2965.0, -3032.0, -3121.0, -3226.0, -3341.0],
                [-3273.0, -3368.0, -3451.0, -3528.0, -3598.0, -3670.0],
                [-3962.0, -4080.0, -4188.0, -4292.0, -4392.0, -4496.0],
            ]
        ),
        model=m,
    )
    reach_data = ModflowSfr2.get_empty_reach_data(5)
    seg_data = {0: ModflowSfr2.get_empty_segment_data(1)}
    seg_data[0]["outseg"] = 0
    reach_data["k"] = 0
    reach_data["i"] = 0
    reach_data["j"] = np.arange(5)
    reach_data["strtop"] = np.array([20, -250, 0.0, -3000.0, -4500.0])
    reach_data["strthick"] = 1.0
    sfr = ModflowSfr2(reach_data=reach_data, segment_data=seg_data, model=m)

    # TODO: this writes to the current working directory regardless of model workspace
    sfr.assign_layers()
    assert np.array_equal(sfr.reach_data.k, np.array([1, 2, 1, 4, 6]))

    l = m.dis.get_layer(0, 0, 0.0)
    assert l == 1
    l = m.dis.get_layer(0, [0, 1], 0.0)
    assert np.array_equal(l, np.array([1, 1]))


@requires_exe("mf2005")
def test_SfrFile(function_tmpdir, sfr_examples_path, mf2005_model_path):
    common_names = [
        "layer",
        "row",
        "column",
        "segment",
        "reach",
        "Qin",
        "Qaquifer",
        "Qout",
        "Qovr",
        "Qprecip",
        "Qet",
        "stage",
        "depth",
        "width",
        "Cond",
    ]
    sfrout = SfrFile(sfr_examples_path / "sfroutput2.txt")
    assert sfrout.ncol == 18, sfrout.ncol
    assert sfrout.names == common_names + [
        "Qwt",
        "delUzstor",
        "gw_head",
    ], sfrout.names
    assert sfrout.times == [(0, 0), (49, 1)], sfrout.times

    df = sfrout.get_dataframe()
    assert df.layer.values[0] == 1
    assert df.column.values[0] == 169
    assert df.Cond.values[0] == 74510.0
    assert df.gw_head.values[3] == 1.288e03

    sfrout = SfrFile(sfr_examples_path / "test1tr.flw")
    assert sfrout.ncol == 16, sfrout.ncol
    assert sfrout.names == common_names + ["gradient"], sfrout.names
    expected_times = [
        (0, 0),
        (4, 0),
        (9, 0),
        (12, 0),
        (14, 0),
        (19, 0),
        (24, 0),
        (29, 0),
        (32, 0),
        (34, 0),
        (39, 0),
        (44, 0),
        (49, 0),
        (0, 1),
        (4, 1),
        (9, 1),
        (12, 1),
        (14, 1),
        (19, 1),
        (24, 1),
        (29, 1),
        (32, 1),
        (34, 1),
        (39, 1),
        (44, 1),
        (45, 1),
        (46, 1),
        (47, 1),
        (48, 1),
        (49, 1),
    ]
    assert sfrout.times == expected_times, sfrout.times
    df = sfrout.get_dataframe()
    assert df.gradient.values[-1] == 5.502e-02
    assert df.shape == (1080, 20)

    ml = Modflow.load("test1tr.nam", model_ws=mf2005_model_path, exe_name="mf2005")
    ml.change_model_ws(function_tmpdir)
    ml.write_input()
    ml.run_model()

    sfrout = SfrFile(function_tmpdir / "test1tr.flw")
    assert sfrout.ncol == 16, sfrout.ncol
    assert sfrout.names == common_names + ["gradient"], sfrout.names
    expected_times = [
        (0, 0),
        (4, 0),
        (9, 0),
        (12, 0),
        (14, 0),
        (19, 0),
        (24, 0),
        (29, 0),
        (32, 0),
        (34, 0),
        (39, 0),
        (44, 0),
        (49, 0),
        (0, 1),
        (4, 1),
        (9, 1),
        (12, 1),
        (14, 1),
        (19, 1),
        (24, 1),
        (29, 1),
        (32, 1),
        (34, 1),
        (39, 1),
        (44, 1),
        (45, 1),
        (46, 1),
        (47, 1),
        (48, 1),
        (49, 1),
    ]
    assert sfrout.times == expected_times, sfrout.times


def test_sfr_plot(mf2005_model_path):
    m = Modflow.load(
        "test1ss.nam",
        model_ws=mf2005_model_path,
        verbose=False,
        check=False,
    )
    sfr = m.get_package("SFR")
    tv = sfr.plot(key="strtop")
    assert isinstance(tv[0], matplotlib.axes.SubplotBase)


def get_test_matrix():
    t = []
    for isfropt in range(6):
        for icalc in range(5):
            t.append((isfropt, icalc))
    return t


def test_sfrcheck(function_tmpdir, mf2005_model_path):
    m = Modflow.load("test1tr.nam", model_ws=mf2005_model_path, verbose=False)

    # run level=0 check
    m.model_ws = function_tmpdir
    fpth = "SFRchecker_results.txt"
    m.sfr.check(fpth, level=0)

    # test checks without modifications
    chk = check(m.sfr)
    chk.numbering()
    assert "continuity in segment and reach numbering" in chk.passed
    chk.routing()
    assert "circular routing" in chk.passed
    chk.overlapping_conductance()
    assert (
        "overlapping conductance" in chk.warnings
    )  # this example model has overlapping conductance
    chk.elevations()
    for test in [
        "segment elevations",
        "reach elevations",
        "reach elevations vs. grid elevations",
    ]:
        assert test in chk.passed
    chk.slope()
    assert "minimum slope" in chk.passed

    # create gaps in segment numbering
    m.sfr.segment_data[0]["nseg"][-1] += 1
    m.sfr.reach_data["ireach"][3] += 1

    # create circular routing instance
    m.sfr.segment_data[0]["outseg"][0] = 1
    m.sfr._graph = None  # weak, but the above shouldn't happen

    chk = check(m.sfr)
    chk.numbering()
    assert "continuity in segment and reach numbering" in chk.errors
    chk.routing()
    assert "circular routing" in chk.errors
    m.sfr.segment_data[0]["nseg"][-1] -= 1
    m.sfr.isfropt = 1.0
    chk = check(m.sfr)
    chk.elevations()
    # throw warning if isfropt=1 and strtop at default
    assert "maximum streambed top" in chk.warnings
    assert "minimum streambed top" in chk.warnings
    m.sfr.reach_data["strtop"] = m.sfr._interpolate_to_reaches("elevup", "elevdn")
    m.sfr.get_slopes()
    m.sfr.reach_data["strhc1"] = 1.0
    m.sfr.reach_data["strthick"] = 1.0
    chk = check(m.sfr)
    chk.elevations()
    assert "maximum streambed top" in chk.passed
    assert "minimum streambed top" in chk.passed
    m.sfr.reach_data["strtop"][2] = -99.0
    chk = check(m.sfr)
    chk.elevations()
    assert "minimum streambed top" in chk.warnings
    m.sfr.reach_data["strtop"][2] = 99999.0
    chk = check(m.sfr)
    chk.elevations()
    assert "maximum streambed top" in chk.warnings
    assert True


@pytest.mark.parametrize("i, case", list(sfr_models().items()))
def test_sfrloadcheck(function_tmpdir, mf2005_model_path, i, case):
    m = Modflow.load(case["mfnam"], model_ws=mf2005_model_path)
    m.model_ws = function_tmpdir
    checker_outfile = os.path.join(function_tmpdir, f"SFRcheck_{m.name}.txt")

    chk = m.sfr.check(checker_outfile, level=1)

    if i == 1:
        assert "overlapping conductance" in chk.warnings
    if i == 2:
        assert "segment elevations vs. model grid" in chk.warnings


@requires_exe("mfnwt")
@pytest.mark.parametrize("isfropt, icalc", get_test_matrix())
def test_isfropt_icalc(function_tmpdir, example_data_path, isfropt, icalc):
    pth = example_data_path / "sfr_test"
    nam = f"sfrtest{isfropt}{icalc}.nam"
    ml = Modflow.load(nam, check=False, model_ws=pth, exe_name="mfnwt")
    sfr = ml.get_package("SFR")
    if sfr is None:
        raise AssertionError()

    ws = function_tmpdir / f"sfrtest{isfropt}{icalc}"
    ml.change_model_ws(ws)
    ml.write_input()
    success = ml.run_model()[0]
    if not success:
        raise AssertionError(
            f"sfrtest{isfropt}{icalc}.nam is broken, please fix SFR 6a, 6bc logic!"
        )


__example_data_path = get_example_data_path()
__project_root_path = get_project_root_path()


@requires_exe("mf2005dbl")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    "namfile",
    [
        __example_data_path / "mf2005_test" / "str.nam",
        (
            __project_root_path
            / ".docs"
            / "groundwater_paper"
            / "uspb"
            / "flopy"
            / "DG.nam"
        ),
    ],
)
def test_mf2005(function_tmpdir, namfile):
    m = Modflow.load(
        namfile,
        exe_name="mf2005dbl",
        forgive=False,
        model_ws=Path(namfile).parent,
        verbose=True,
        check=False,
    )
    assert m.load_fail is False

    # rewrite files
    ws = function_tmpdir / "ws"
    m.model_ws = ws
    m.write_input()

    # attempt to run the model
    success, buff = m.run_model(silent=False)
    assert success

    # load files
    pth = ws / f"{m.name}.str"
    str2 = ModflowStr.load(pth, m)
    for name in str2.dtype.names:
        assert (
            np.array_equal(
                str2.stress_period_data[0][name],
                m.str.stress_period_data[0][name],
            )
            is True
        )
    for name in str2.dtype2.names:
        assert (
            np.array_equal(str2.segment_data[0][name], m.str.segment_data[0][name])
            is True
        )
