import numpy as np
import pytest
from modflow_devtools.markers import requires_exe


def test_get_icelltype_from_laytyp():
    from flopy.utils.classic_to_mf6 import get_icelltype_from_laytyp

    # 1D array
    laytyp = np.array([1, 0, 0, 1])
    icelltype = get_icelltype_from_laytyp(laytyp)
    assert icelltype.shape == laytyp.shape
    assert np.array_equal(icelltype, [1, 0, 0, 1])

    # scalar
    laytyp = 0
    icelltype = get_icelltype_from_laytyp(laytyp)
    assert icelltype == 0

    laytyp = 1
    icelltype = get_icelltype_from_laytyp(laytyp)
    assert icelltype == 1

    # negative laytyp is convertible (wetting disabled, not confined)
    laytyp = -1
    icelltype = get_icelltype_from_laytyp(laytyp)
    assert icelltype == 1

    # various LAYTYP values — any non-zero means convertible
    laytyp = np.array([0, 1, 2, 3, -1])
    icelltype = get_icelltype_from_laytyp(laytyp)
    # 0 stays 0, all others (positive or negative) map to 1
    assert np.array_equal(icelltype, [0, 1, 1, 1, 1])


def test_get_icelltype_from_laycon():
    from flopy.utils.classic_to_mf6 import get_icelltype_from_laycon

    # scalar confined (laycon=0)
    assert get_icelltype_from_laycon(0) == 0
    # scalar convertible (laycon=1)
    assert get_icelltype_from_laycon(1) == 1
    # laycon=2 is head-dependent-T confined — maps to 0
    assert get_icelltype_from_laycon(2) == 0
    # laycon=3 is fully convertible — maps to 1
    assert get_icelltype_from_laycon(3) == 1

    # full 4-value array covering all cases
    laycon = np.array([0, 1, 2, 3])
    result = get_icelltype_from_laycon(laycon)
    assert np.array_equal(result, [0, 1, 0, 1])

    # shape is preserved
    laycon_3d = np.array([[[0, 1], [2, 3]]])
    result_3d = get_icelltype_from_laycon(laycon_3d)
    assert result_3d.shape == laycon_3d.shape
    assert np.array_equal(result_3d, [[[0, 1], [0, 1]]])


def test_mfgrdfile_write_roundtrip(tmp_path):
    from flopy.mf6.utils import MfGrdFile, get_structured_connectivity

    nlay, nrow, ncol = 2, 3, 4
    delr = np.ones(ncol) * 100.0
    delc = np.ones(nrow) * 50.0
    top = np.ones((nrow, ncol)) * 10.0
    botm = np.zeros((nlay, nrow, ncol))
    botm[0] = 5.0

    ia, ja, nja = get_structured_connectivity(nlay, nrow, ncol)

    icelltype = np.zeros((nlay, nrow, ncol), dtype=np.int32)
    icelltype[0] = 1  # top layer convertible

    grb_file = tmp_path / "test.dis.grb"
    MfGrdFile.write_dis(
        str(grb_file),
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        top,
        botm,
        ia,
        ja,
        icelltype=icelltype,
    )

    grb = MfGrdFile(str(grb_file))

    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol
    assert grb.nodes == nlay * nrow * ncol
    assert grb.nja == nja

    np.testing.assert_array_almost_equal(grb.delr, delr)
    np.testing.assert_array_almost_equal(grb.delc, delc)

    # Build expected TOP for all cells (MF6 format)
    # Layer 1: model top, Layer 2+: bottom of layer above
    # In Fortran order (layer-interleaved): [L0[0,0], L1[0,0], L2[0,0], L0[0,1], ...]
    top_expected = np.zeros(nlay * nrow * ncol)
    top_flat = top.flatten(order="F")
    botm_flat = botm.flatten(order="F")

    for i in range(nrow * ncol):
        # Layer 0: use model top
        top_expected[i * nlay] = top_flat[i]
        # Layers 1+: use bottom of layer above
        for k in range(1, nlay):
            top_expected[i * nlay + k] = botm_flat[i * nlay + (k - 1)]

    np.testing.assert_array_almost_equal(grb.top, top_expected)
    np.testing.assert_array_almost_equal(grb.bot, botm.flatten(order="F"))

    np.testing.assert_array_equal(grb.ia, ia)
    np.testing.assert_array_equal(grb.ja, ja)

    icelltype_read = grb._datadict["ICELLTYPE"]
    np.testing.assert_array_equal(icelltype_read, icelltype.flatten(order="F"))


def test_mfgrdfile_write_with_idomain(tmp_path):
    from flopy.mf6.utils import MfGrdFile, get_structured_connectivity

    nlay, nrow, ncol = 1, 3, 3
    delr = np.ones(ncol)
    delc = np.ones(nrow)
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))

    # center cell inactive
    idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
    idomain[0, 1, 1] = 0

    ia, ja, nja = get_structured_connectivity(nlay, nrow, ncol, idomain)

    grb_file = tmp_path / "test_idomain.dis.grb"
    MfGrdFile.write_dis(
        str(grb_file), nlay, nrow, ncol, delr, delc, top, botm, ia, ja, idomain=idomain
    )

    grb = MfGrdFile(str(grb_file))
    idomain_read = grb._datadict["IDOMAIN"]
    np.testing.assert_array_equal(idomain_read, idomain.flatten(order="F"))


def test_mfgrdfile_write_validation():
    from flopy.mf6.utils import MfGrdFile, get_structured_connectivity

    nlay, nrow, ncol = 1, 2, 2
    delr = np.ones(ncol)
    delc = np.ones(nrow)
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))
    ia, ja, nja = get_structured_connectivity(nlay, nrow, ncol)

    with pytest.raises(ValueError, match="delr length"):
        MfGrdFile.write_dis(
            "test.grb",
            nlay,
            nrow,
            ncol,
            np.ones(3),  # Wrong length
            delc,
            top,
            botm,
            ia,
            ja,
        )

    with pytest.raises(ValueError, match="ia length"):
        MfGrdFile.write_dis(
            "test.grb",
            nlay,
            nrow,
            ncol,
            delr,
            delc,
            top,
            botm,
            np.ones(10),  # Wrong length (should be ncells + 1 = 5)
            ja,
        )


def test_nwt_to_mf6_converter_init(function_tmpdir):
    import numpy as np

    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    nlay, nrow, ncol = 2, 3, 4
    delr = np.ones(ncol) * 100.0
    delc = np.ones(nrow) * 100.0
    top = np.ones((nrow, ncol)) * 10.0
    botm = np.zeros((nlay, nrow, ncol))
    botm[0] = 5.0
    laytyp = np.array([1, 0])  # Top layer convertible

    hds_file = function_tmpdir / "test.hds"
    cbc_file = function_tmpdir / "test.cbc"

    head_data = np.ones((nlay, nrow, ncol)) * 8.0
    HeadFile.write(
        str(hds_file),
        {(1, 1): head_data},
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    from flopy.mf6.utils import get_structured_connectivity, get_structured_faceflows

    ia, ja, nja = get_structured_connectivity(nlay, nrow, ncol)
    flowja = np.ones(nja) * 0.5

    frf, fff, flf = get_structured_faceflows(
        flowja, grb_file=None, ia=ia, ja=ja, nlay=nlay, nrow=nrow, ncol=ncol
    )

    bud_data = [
        {
            "data": frf.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW RIGHT FACE",
            "imeth": 1,
        },
        {
            "data": fff.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW FRONT FACE",
            "imeth": 1,
        },
        {
            "data": flf.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW LOWER FACE",
            "imeth": 1,
        },
    ]

    CellBudgetFile.write(
        str(cbc_file),
        bud_data,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    converter = ClassicMfToMf6Converter(
        str(hds_file),
        str(cbc_file),
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        top,
        botm,
        laytyp,
        model_ws=function_tmpdir,
    )

    assert converter.nlay == nlay
    assert converter.nrow == nrow
    assert converter.ncol == ncol
    assert converter.ncells == nlay * nrow * ncol
    assert len(converter.times) == 1
    assert len(converter.kstpkper) == 1

    assert converter.icelltype.shape == (nlay,)
    assert np.array_equal(converter.icelltype, [1, 0])
    assert converter.icelltype_3d.shape == (nlay, nrow, ncol)
    assert np.all(converter.icelltype_3d[0] == 1)
    assert np.all(converter.icelltype_3d[1] == 0)


def test_nwt_to_mf6_converter_convert(function_tmpdir):
    import numpy as np

    from flopy.mf6.utils import (
        MfGrdFile,
        get_structured_connectivity,
        get_structured_faceflows,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    nlay, nrow, ncol = 2, 3, 4
    delr = np.ones(ncol) * 100.0
    delc = np.ones(nrow) * 100.0
    top = np.ones((nrow, ncol)) * 10.0
    botm = np.zeros((nlay, nrow, ncol))
    botm[0] = 5.0
    laytyp = np.array([1, 0])

    hds_file = function_tmpdir / "test.hds"
    cbc_file = function_tmpdir / "test.cbc"

    head_data = np.ones((nlay, nrow, ncol)) * 8.0
    HeadFile.write(str(hds_file), {(1, 1): head_data}, nlay=nlay, nrow=nrow, ncol=ncol)

    ia, ja, nja = get_structured_connectivity(nlay, nrow, ncol)
    flowja = np.ones(nja) * 0.5
    frf, fff, flf = get_structured_faceflows(
        flowja, grb_file=None, ia=ia, ja=ja, nlay=nlay, nrow=nrow, ncol=ncol
    )

    bud_data = [
        {
            "data": frf.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW RIGHT FACE",
            "imeth": 1,
        },
        {
            "data": fff.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW FRONT FACE",
            "imeth": 1,
        },
        {
            "data": flf.flatten(order="F"),
            "kstp": 1,
            "kper": 1,
            "text": "FLOW LOWER FACE",
            "imeth": 1,
        },
    ]
    CellBudgetFile.write(str(cbc_file), bud_data, nlay=nlay, nrow=nrow, ncol=ncol)

    converter = ClassicMfToMf6Converter(
        str(hds_file),
        str(cbc_file),
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        top,
        botm,
        laytyp,
        model_ws=function_tmpdir,
    )

    output_dir = function_tmpdir / "mf6_output"
    result = converter.convert(str(output_dir), verbose=False)

    assert result["grb"].exists()
    assert result["hds"].exists()
    assert result["bud"].exists()

    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    hds_mf6 = HeadFile(str(result["hds"]))
    head_read = hds_mf6.get_data(idx=0)  # Read first record
    np.testing.assert_array_almost_equal(head_read, head_data)

    bud_mf6 = CellBudgetFile(str(result["bud"]))
    texts = bud_mf6.textlist
    # Convert to strings and strip for comparison
    texts_str = [
        t.decode().strip() if isinstance(t, bytes) else str(t).strip() for t in texts
    ]
    assert "FLOW-JA-FACE" in texts_str
    assert "DATA-SAT" in texts_str
    # DATA-SPDIS skipped for now


@requires_exe("mfnwt")
@pytest.mark.slow
def test_nwt_to_mf6_watertable_model(function_tmpdir):
    from flopy.mf6.utils import MfGrdFile
    from flopy.modflow import (
        Modflow,
        ModflowBas,
        ModflowDis,
        ModflowGhb,
        ModflowNwt,
        ModflowOc,
        ModflowRch,
        ModflowUpw,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    modelname = "watertable"

    nlay, nrow, ncol = 1, 1, 100
    delr = 50.0
    delc = 1.0

    h1, h2 = 20.0, 11.0

    top = 25.0
    botm = 0.0
    hk = 50.0

    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    strt[0, 0, 0] = h1
    strt[0, 0, -1] = h2

    rchrate = 0.001

    h_adj1 = h1 - (h1 - h2) / ncol
    h_adj2 = h2 + (h1 - h2) / ncol

    b1 = 0.5 * (h1 + h_adj1)
    b2 = 0.5 * (h2 + h_adj2)
    c1 = hk * b1 * delc / (0.5 * delr)
    c2 = hk * b2 * delc / (0.5 * delr)

    ghb_dtype = ModflowGhb.get_default_dtype()
    stress_period_data = np.zeros((2), dtype=ghb_dtype)
    stress_period_data = stress_period_data.view(np.recarray)
    stress_period_data[0] = (0, 0, 0, h1, c1)
    stress_period_data[1] = (0, 0, ncol - 1, h2, c2)

    mf = Modflow(
        modelname=modelname,
        exe_name="mfnwt",
        model_ws=function_tmpdir,
        version="mfnwt",
    )
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    ModflowBas(mf, ibound=1, strt=strt)
    ModflowUpw(mf, hk=hk, laytyp=1, ipakcb=53)  # laytyp=1 for convertible
    ModflowGhb(mf, stress_period_data=stress_period_data)
    ModflowRch(mf, rech=rchrate, nrchop=1)
    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )
    oc.reset_budgetunit(budgetunit=53, fname=f"{modelname}.cbc")
    ModflowNwt(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    assert success, "NWT model run failed"

    hds_file = function_tmpdir / f"{modelname}.hds"
    cbc_file = function_tmpdir / f"{modelname}.cbc"
    assert hds_file.exists(), "Head file not created"
    assert cbc_file.exists(), "Budget file not created"

    converter = ClassicMfToMf6Converter.from_model(mf, hds_file, cbc_file)

    output_dir = function_tmpdir / "mf6_output"
    result = converter.convert(str(output_dir), verbose=True)

    assert result["grb"].exists(), "GRB file not created"
    assert result["hds"].exists(), "MF6 head file not created"
    assert result["bud"].exists(), "MF6 budget file not created"

    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    hds_mf6 = HeadFile(str(result["hds"]))
    head_mf6 = hds_mf6.get_data(idx=0)

    hds_nwt = HeadFile(str(hds_file))
    head_nwt = hds_nwt.get_data(idx=0)

    np.testing.assert_array_almost_equal(
        head_mf6, head_nwt, decimal=5, err_msg="MF6 heads don't match NWT heads"
    )

    bud_mf6 = CellBudgetFile(str(result["bud"]))
    texts = bud_mf6.textlist
    texts_str = [
        t.decode().strip() if isinstance(t, bytes) else str(t).strip() for t in texts
    ]

    assert "FLOW-JA-FACE" in texts_str, "FLOW-JA-FACE not in budget"
    assert "DATA-SAT" in texts_str, "DATA-SAT not in budget"

    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", idx=0)
    assert len(flowja) > 0, "Could not read FLOW-JA-FACE"

    kstpkper0 = bud_mf6.get_kstpkper()[0]
    sat_data = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper0)
    assert len(sat_data) > 0, "Could not read DATA-SAT"

    # Validate saturation values numerically.  For the single convertible layer
    # (laytyp=1, icelltype=1) with top=25 and botm=0:
    #   sat = clamp((head - botm) / (top - botm), 0, 1) = clamp(head / 25, 0, 1)
    # DATA-SAT is stored as imeth=6; the saturation values are in the "q" field.
    sat_vals = sat_data[0]["q"]
    expected_sat = np.clip(head_nwt.flatten() / (top - botm), 0.0, 1.0)
    np.testing.assert_array_almost_equal(
        sat_vals,
        expected_sat,
        decimal=5,
        err_msg="DATA-SAT values do not match expected (head - botm) / (top - botm)",
    )


@requires_exe("mfnwt")
@pytest.mark.slow
def test_nwt_to_mf6_multilayer_model(function_tmpdir):
    from flopy.mf6.utils import MfGrdFile
    from flopy.modflow import (
        Modflow,
        ModflowBas,
        ModflowDis,
        ModflowNwt,
        ModflowOc,
        ModflowRch,
        ModflowUpw,
        ModflowWel,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    modelname = "multilayer"

    # 3 layers: top convertible, middle/bottom confined
    nlay, nrow, ncol = 3, 10, 10
    delr = delc = 100.0

    top = np.ones((nrow, ncol)) * 100.0
    botm = np.zeros((nlay, nrow, ncol))
    botm[0] = 80.0
    botm[1] = 60.0
    botm[2] = 40.0

    strt = np.ones((nlay, nrow, ncol)) * 95.0

    laytyp = np.array([1, 0, 0])

    hk = np.ones((nlay, nrow, ncol)) * 10.0

    mf = Modflow(
        modelname=modelname,
        exe_name="mfnwt",
        model_ws=function_tmpdir,
        version="mfnwt",
    )
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    ModflowBas(mf, ibound=1, strt=strt)
    ModflowUpw(mf, hk=hk, laytyp=laytyp, ipakcb=53)

    # well in center
    wel_data = [(1, 5, 5, -1000.0)]  # Layer 2, center cell, pumping
    ModflowWel(mf, stress_period_data={0: wel_data})

    # recharge to top layer
    ModflowRch(mf, rech=0.001)

    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )
    oc.reset_budgetunit(budgetunit=53, fname=f"{modelname}.cbc")

    ModflowNwt(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    assert success

    hds_file = function_tmpdir / f"{modelname}.hds"
    cbc_file = function_tmpdir / f"{modelname}.cbc"

    assert hds_file.exists()
    assert cbc_file.exists()

    converter = ClassicMfToMf6Converter.from_model(mf, hds_file, cbc_file)

    output_dir = function_tmpdir / "mf6_output"
    result = converter.convert(str(output_dir), verbose=True)

    # Validate GRB file
    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    # Verify ICELLTYPE was set correctly
    icelltype = grb._datadict["ICELLTYPE"]

    # Layer 1 should be convertible (1), layers 2-3 confined (0)
    # ICELLTYPE is flattened in Fortran order, which interleaves layers
    # Pattern: [L0[0,0], L1[0,0], L2[0,0], L0[0,1], L1[0,1], L2[0,1], ...]
    ncells = nlay * nrow * ncol
    cells_per_layer = nrow * ncol

    # Extract layer data from Fortran-ordered flat array
    layer_1_icelltype = icelltype[0::nlay]  # Every nlay-th element starting at 0
    layer_2_icelltype = icelltype[1::nlay]  # Every nlay-th element starting at 1
    layer_3_icelltype = icelltype[2::nlay]  # Every nlay-th element starting at 2

    assert np.all(layer_1_icelltype == 1), (
        f"Layer 1 should be convertible, got {layer_1_icelltype[:10]}"
    )
    assert np.all(layer_2_icelltype == 0), (
        f"Layer 2 should be confined, got {layer_2_icelltype[:10]}"
    )
    assert np.all(layer_3_icelltype == 0), (
        f"Layer 3 should be confined, got {layer_3_icelltype[:10]}"
    )

    # Verify TOP array is expanded to all cells
    assert len(grb.top) == ncells, f"TOP should have {ncells} values"

    # Extract layer data from Fortran-ordered TOP array
    top_layer_1 = grb.top[0::nlay]  # Every nlay-th element starting at 0
    top_layer_2 = grb.top[1::nlay]  # Every nlay-th element starting at 1
    top_layer_3 = grb.top[2::nlay]  # Every nlay-th element starting at 2

    # Layer 1 top should match model top
    np.testing.assert_array_almost_equal(
        top_layer_1,
        top.flatten(order="F"),
        err_msg="Layer 1 TOP should match model top",
    )

    # Layer 2 top should match layer 1 bottom
    np.testing.assert_array_almost_equal(
        top_layer_2,
        botm[0].flatten(order="F"),
        err_msg="Layer 2 TOP should match layer 1 bottom",
    )

    # Layer 3 top should match layer 2 bottom
    np.testing.assert_array_almost_equal(
        top_layer_3,
        botm[1].flatten(order="F"),
        err_msg="Layer 3 TOP should match layer 2 bottom",
    )

    # Validate heads
    hds_nwt = HeadFile(str(hds_file))
    hds_mf6 = HeadFile(str(result["hds"]))

    head_nwt = hds_nwt.get_data(idx=0)
    head_mf6 = hds_mf6.get_data(idx=0)

    np.testing.assert_array_almost_equal(head_mf6, head_nwt, decimal=5)

    # Validate budget
    bud_mf6 = CellBudgetFile(str(result["bud"]))
    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", idx=0)
    assert flowja is not None

    # Verify saturation is correct for convertible layer
    kstpkper0 = bud_mf6.get_kstpkper()[0]
    sat_data = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper0)
    assert len(sat_data) > 0, "Could not read DATA-SAT"


# ---------------------------------------------------------------------------
# Fixtures for example-data models
# ---------------------------------------------------------------------------


@pytest.fixture
def freyberg_multilayer_path(example_data_path):
    return example_data_path / "freyberg_multilayer_transient"


@pytest.fixture
def mf2005_freyberg_path(example_data_path):
    return example_data_path / "freyberg"


# ---------------------------------------------------------------------------
# Integration tests against real pre-computed model output
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_classic_to_mf6_freyberg_multilayer(freyberg_multilayer_path, function_tmpdir):
    """
    Convert the pre-computed freyberg_multilayer_transient NWT outputs to MF6
    and verify that the face-flow roundtrip is exact.

    This test requires no executable — it uses the binary output files already
    committed to the repo.  It checks:
    - GRB file has the correct grid dimensions
    - Head values are preserved for a sample of time steps (first, middle, last)
    - get_structured_faceflows(flowja_converted) recovers the original NWT
      FLOW RIGHT/FRONT/LOWER FACE values (interior faces only)
    - DATA-SAT is present for sampled time steps
    """
    from flopy.mf6.utils import MfGrdFile, get_structured_faceflows
    from flopy.modflow import Modflow
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    model_ws = freyberg_multilayer_path

    # Load model to get grid parameters (no exe needed)
    mf = Modflow.load(
        "freyberg.nam",
        model_ws=str(model_ws),
        check=False,
        load_only=["dis", "upw"],
    )
    nlay = int(mf.dis.nlay)
    nrow = int(mf.dis.nrow)
    ncol = int(mf.dis.ncol)

    converter = ClassicMfToMf6Converter(
        hds_file=str(model_ws / "freyberg.hds"),
        cbc_file=str(model_ws / "freyberg.cbc"),
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=mf.dis.delr.array,
        delc=mf.dis.delc.array,
        top=mf.dis.top.array,
        botm=mf.dis.botm.array,
        laytyp=mf.upw.laytyp.array,
        hdry=float(mf.upw.hdry),
    )

    output_dir = function_tmpdir / "mf6"
    result = converter.convert(str(output_dir))

    # --- GRB dimensions ---
    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    hds_orig = HeadFile(str(model_ws / "freyberg.hds"))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(model_ws / "freyberg.cbc"))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    kstpkper_list = hds_orig.get_kstpkper()

    # --- Verify a sample of time steps (first, middle, last) ---
    check_indices = [0, len(kstpkper_list) // 2, len(kstpkper_list) - 1]
    for check_idx in check_indices:
        kstpkper = kstpkper_list[check_idx]

        # Head values are preserved exactly
        head_orig = hds_orig.get_data(kstpkper=kstpkper)
        head_mf6 = hds_mf6.get_data(kstpkper=kstpkper)
        np.testing.assert_array_almost_equal(
            head_mf6,
            head_orig,
            decimal=5,
            err_msg=f"Head mismatch at kstpkper={kstpkper}",
        )

        # Face-flow roundtrip: faceflows → flowja → faceflows should recover
        # the original values on interior faces.
        frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
        fff_orig = cbc_orig.get_data(text="FLOW FRONT FACE", kstpkper=kstpkper)[0]
        flf_orig = cbc_orig.get_data(text="FLOW LOWER FACE", kstpkper=kstpkper)[0]

        flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
        frf_rt, fff_rt, flf_rt = get_structured_faceflows(
            flowja, grb_file=str(result["grb"])
        )

        np.testing.assert_array_almost_equal(
            frf_rt[:, :, :-1],
            frf_orig[:, :, :-1],
            decimal=5,
            err_msg=f"FLOW RIGHT FACE mismatch at kstpkper={kstpkper}",
        )
        np.testing.assert_array_almost_equal(
            fff_rt[:, :-1, :],
            fff_orig[:, :-1, :],
            decimal=5,
            err_msg=f"FLOW FRONT FACE mismatch at kstpkper={kstpkper}",
        )
        np.testing.assert_array_almost_equal(
            flf_rt[:-1, :, :],
            flf_orig[:-1, :, :],
            decimal=5,
            err_msg=f"FLOW LOWER FACE mismatch at kstpkper={kstpkper}",
        )

        # DATA-SAT is present
        sat = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper)
        assert len(sat) > 0, f"DATA-SAT missing at kstpkper={kstpkper}"


@requires_exe("mf2005")
@pytest.mark.slow
def test_classic_to_mf6_freyberg_mf2005(mf2005_freyberg_path, function_tmpdir):
    """
    Run the single-layer steady-state Freyberg MODFLOW-2005 model, convert
    its outputs to MF6, and verify correctness.

    Tests that the converter works with the LPF package (vs UPW for NWT),
    confirming ClassicMfToMf6Converter handles both code paths.
    """
    import shutil

    from flopy.mf6.utils import MfGrdFile, get_structured_faceflows
    from flopy.modflow import Modflow
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    # Copy model to a writable temp directory and run it
    run_ws = function_tmpdir / "mf2005"
    shutil.copytree(mf2005_freyberg_path, run_ws)

    mf = Modflow.load(
        "freyberg.nam",
        model_ws=str(run_ws),
        exe_name="mf2005",
        check=False,
    )
    success, _ = mf.run_model(silent=True)
    assert success, "MODFLOW-2005 freyberg run failed"

    hds_path = run_ws / "freyberg.hds"
    cbc_path = run_ws / "freyberg.cbc"
    assert hds_path.exists(), "Head file not produced"
    assert cbc_path.exists(), "Budget file not produced"

    nlay = int(mf.dis.nlay)
    nrow = int(mf.dis.nrow)
    ncol = int(mf.dis.ncol)

    converter = ClassicMfToMf6Converter.from_model(mf, hds_path, cbc_path)

    output_dir = function_tmpdir / "mf6"
    result = converter.convert(str(output_dir))

    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    hds_orig = HeadFile(str(hds_path))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(cbc_path))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    kstpkper = hds_orig.get_kstpkper()[0]

    # Head roundtrip
    np.testing.assert_array_almost_equal(
        hds_mf6.get_data(kstpkper=kstpkper),
        hds_orig.get_data(kstpkper=kstpkper),
        decimal=5,
    )

    # Face-flow roundtrip (single layer, so no lower face)
    frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
    fff_orig = cbc_orig.get_data(text="FLOW FRONT FACE", kstpkper=kstpkper)[0]

    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
    frf_rt, fff_rt, _ = get_structured_faceflows(flowja, grb_file=str(result["grb"]))

    np.testing.assert_array_almost_equal(
        frf_rt[:, :, :-1], frf_orig[:, :, :-1], decimal=5
    )
    np.testing.assert_array_almost_equal(
        fff_rt[:, :-1, :], fff_orig[:, :-1, :], decimal=5
    )


@requires_exe("mf2000")
@pytest.mark.slow
def test_classic_to_mf6_mf2000_watertable(function_tmpdir):
    """
    Build and run a 1-layer unconfined watertable MODFLOW-2000 model, convert
    its outputs to MF6 format, and verify head, face-flow, and saturation.

    MODFLOW-2000 produces compact budget files with the same binary format as
    MODFLOW-NWT and MODFLOW-2005.  This test confirms ClassicMfToMf6Converter
    handles MF-2000 output identically to later variants.
    """
    from flopy.mf6.utils import MfGrdFile, get_structured_faceflows
    from flopy.modflow import (
        Modflow,
        ModflowBas,
        ModflowDis,
        ModflowGhb,
        ModflowLpf,
        ModflowOc,
        ModflowPcg,
        ModflowRch,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    modelname = "mf2000_watertable"
    nlay, nrow, ncol = 1, 1, 100
    delr, delc = 50.0, 1.0
    h1, h2 = 20.0, 11.0
    top, botm = 25.0, 0.0
    hk = 50.0

    # Linear initial heads to avoid convergence failures with PCG and unconfined cells
    strt = np.zeros((nlay, nrow, ncol))
    strt[0, 0, :] = np.linspace(h1, h2, ncol)

    # GHB boundary conditions at both ends
    h_adj1 = h1 - (h1 - h2) / ncol
    h_adj2 = h2 + (h1 - h2) / ncol
    b1 = 0.5 * (h1 + h_adj1)
    b2 = 0.5 * (h2 + h_adj2)
    c1 = hk * b1 * delc / (0.5 * delr)
    c2 = hk * b2 * delc / (0.5 * delr)
    ghb_dtype = ModflowGhb.get_default_dtype()
    spd = np.zeros(2, dtype=ghb_dtype).view(np.recarray)
    spd[0] = (0, 0, 0, h1, c1)
    spd[1] = (0, 0, ncol - 1, h2, c2)

    mf = Modflow(
        modelname=modelname,
        exe_name="mf2000",
        model_ws=str(function_tmpdir),
        version="mf2k",
    )
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    ModflowBas(mf, ibound=1, strt=strt)
    ModflowLpf(mf, hk=hk, laytyp=1, ipakcb=53)
    ModflowGhb(mf, stress_period_data=spd)
    ModflowRch(mf, rech=0.001, nrchop=1)
    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )
    oc.reset_budgetunit(budgetunit=53, fname=f"{modelname}.cbc")
    ModflowPcg(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    assert success, "MODFLOW-2000 watertable run failed"

    hds_path = function_tmpdir / f"{modelname}.hds"
    cbc_path = function_tmpdir / f"{modelname}.cbc"
    assert hds_path.exists()
    assert cbc_path.exists()

    converter = ClassicMfToMf6Converter.from_model(mf, hds_path, cbc_path)
    result = converter.convert(str(function_tmpdir / "mf6_output"))

    assert result["grb"].exists()
    assert result["hds"].exists()
    assert result["bud"].exists()

    hds_orig = HeadFile(str(hds_path))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(cbc_path))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    kstpkper = hds_orig.get_kstpkper()[0]

    # Head roundtrip
    head_orig = hds_orig.get_data(kstpkper=kstpkper)
    np.testing.assert_array_almost_equal(
        hds_mf6.get_data(kstpkper=kstpkper), head_orig, decimal=5
    )

    # Face-flow roundtrip (1 row, 1 layer → only right face flows exist)
    frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
    frf_rt, _, _ = get_structured_faceflows(flowja, grb_file=str(result["grb"]))
    np.testing.assert_array_almost_equal(
        frf_rt[:, :, :-1], frf_orig[:, :, :-1], decimal=5
    )

    # Saturation: sat = clamp(head / (top - botm), 0, 1)
    sat_data = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper)
    assert len(sat_data) > 0, "DATA-SAT missing"
    expected_sat = np.clip(head_orig.flatten() / (top - botm), 0.0, 1.0)
    np.testing.assert_array_almost_equal(sat_data[0]["q"], expected_sat, decimal=5)


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_key",
    [
        "mf6/test/test001a_Tharmonic",  # 1 layer, 1 stress period, steady
        "mf6/test/test001h_rch_array3",  # 1 layer, 4 stress periods
    ],
)
def test_classic_to_mf6_mf2005_testmodels(model_key, function_tmpdir):
    """
    Fetch a MODFLOW-2005/LPF companion model from the devtools registry,
    run it, convert outputs to MF6, and verify head and face-flow roundtrip.

    The mf6/test registry includes mf2005/ subdirectories alongside each MF6
    test case.  Running these models exercises the LPF code path across a
    variety of boundary conditions and confirms multi-stress-period conversion.
    """
    import shutil

    from modflow_devtools.models import copy_to

    from flopy.mf6.utils import get_structured_faceflows
    from flopy.modflow import Modflow
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    # Download the parent MF6 model; the mf2005/ companion is a subdirectory.
    model_ws = function_tmpdir / "model"
    copy_to(str(model_ws), model_key)
    mf2005_ws = model_ws / "mf2005"
    assert mf2005_ws.exists(), f"mf2005/ subdir not found for {model_key}"

    # Locate the namefile (there should be exactly one .nam in mf2005/)
    nam_files = list(mf2005_ws.glob("*.nam"))
    assert len(nam_files) == 1, f"Expected 1 .nam file in mf2005/, got {nam_files}"
    nam_name = nam_files[0].name

    # Run mf2005
    mf = Modflow.load(
        nam_name,
        model_ws=str(mf2005_ws),
        exe_name="mf2005",
        check=False,
    )

    # The companion models may not have "save budget" in their OC.  Force it so
    # that the CBC file is populated for the converter.
    nper = int(mf.dis.nper)
    nstp = mf.dis.nstp.array
    sp_data = {
        (kstp, kper): ["save head", "save budget"]
        for kper in range(nper)
        for kstp in range(nstp[kper])
    }
    from flopy.modflow import ModflowOc

    ModflowOc(mf, stress_period_data=sp_data, compact=True)
    mf.write_input()

    success, _ = mf.run_model(silent=True)
    assert success, f"mf2005 run failed for {model_key}"

    # Locate binary outputs written by the OC package
    hds_path = next(mf2005_ws.glob("*.hds"), None)
    cbc_path = next(mf2005_ws.glob("*.cbc"), None)
    assert hds_path is not None, "No .hds file produced"
    assert cbc_path is not None, "No .cbc file produced"

    nlay = int(mf.dis.nlay)
    nrow = int(mf.dis.nrow)
    ncol = int(mf.dis.ncol)

    converter = ClassicMfToMf6Converter(
        hds_file=str(hds_path),
        cbc_file=str(cbc_path),
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=mf.dis.delr.array,
        delc=mf.dis.delc.array,
        top=mf.dis.top.array,
        botm=mf.dis.botm.array,
        laytyp=mf.lpf.laytyp.array,
    )
    result = converter.convert(str(function_tmpdir / "mf6_output"))

    assert result["grb"].exists()
    assert result["hds"].exists()
    assert result["bud"].exists()

    hds_orig = HeadFile(str(hds_path))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(cbc_path))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    for kstpkper in hds_orig.get_kstpkper():
        # Head roundtrip
        np.testing.assert_array_almost_equal(
            hds_mf6.get_data(kstpkper=kstpkper),
            hds_orig.get_data(kstpkper=kstpkper),
            decimal=5,
            err_msg=f"Head mismatch at kstpkper={kstpkper}",
        )

        # Face-flow roundtrip (check only terms that the model actually wrote)
        cbc_texts = [
            t.decode().strip() if isinstance(t, bytes) else t.strip()
            for t in cbc_orig.textlist
        ]
        flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
        frf_rt, fff_rt, flf_rt = get_structured_faceflows(
            flowja, grb_file=str(result["grb"])
        )
        if "FLOW RIGHT FACE" in cbc_texts:
            frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
            np.testing.assert_array_almost_equal(
                frf_rt[:, :, :-1],
                frf_orig[:, :, :-1],
                decimal=5,
                err_msg=f"FLOW RIGHT FACE mismatch at kstpkper={kstpkper}",
            )
        if "FLOW FRONT FACE" in cbc_texts:
            fff_orig = cbc_orig.get_data(text="FLOW FRONT FACE", kstpkper=kstpkper)[0]
            np.testing.assert_array_almost_equal(
                fff_rt[:, :-1, :],
                fff_orig[:, :-1, :],
                decimal=5,
                err_msg=f"FLOW FRONT FACE mismatch at kstpkper={kstpkper}",
            )
        if "FLOW LOWER FACE" in cbc_texts:
            flf_orig = cbc_orig.get_data(text="FLOW LOWER FACE", kstpkper=kstpkper)[0]
            np.testing.assert_array_almost_equal(
                flf_rt[:-1, :, :],
                flf_orig[:-1, :, :],
                decimal=5,
                err_msg=f"FLOW LOWER FACE mismatch at kstpkper={kstpkper}",
            )

        # DATA-SAT present
        sat = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper)
        assert len(sat) > 0, f"DATA-SAT missing at kstpkper={kstpkper}"


@requires_exe("mf2005")
@pytest.mark.slow
def test_classic_to_mf6_mf2005_multilayer(function_tmpdir):
    """
    Build and run a 3-layer MODFLOW-2005/LPF model (1 convertible + 2 confined),
    convert its outputs to MF6, and verify head, face-flow, and saturation.

    Complements test_classic_to_mf6_freyberg_mf2005 (single layer) and the
    NWT multilayer tests by exercising the LPF code path with nlay > 1.
    """
    from flopy.mf6.utils import MfGrdFile, get_structured_faceflows
    from flopy.modflow import (
        Modflow,
        ModflowBas,
        ModflowDis,
        ModflowLpf,
        ModflowOc,
        ModflowPcg,
        ModflowRch,
        ModflowWel,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    modelname = "mf2005_multilayer"
    nlay, nrow, ncol = 3, 10, 10
    delr = delc = 100.0
    top = np.ones((nrow, ncol)) * 100.0
    botm = np.zeros((nlay, nrow, ncol))
    botm[0] = 80.0
    botm[1] = 60.0
    botm[2] = 40.0
    laytyp = np.array([1, 0, 0])  # top convertible, lower two confined
    strt = np.ones((nlay, nrow, ncol)) * 95.0
    hk = np.ones((nlay, nrow, ncol)) * 10.0

    mf = Modflow(
        modelname=modelname,
        exe_name="mf2005",
        model_ws=str(function_tmpdir),
        version="mf2005",
    )
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    ModflowBas(mf, ibound=1, strt=strt)
    ModflowLpf(mf, hk=hk, laytyp=laytyp, ipakcb=53)
    ModflowWel(mf, stress_period_data={0: [(1, 5, 5, -1000.0)]})
    ModflowRch(mf, rech=0.001)
    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )
    oc.reset_budgetunit(budgetunit=53, fname=f"{modelname}.cbc")
    ModflowPcg(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    assert success, "MODFLOW-2005 multilayer run failed"

    hds_path = function_tmpdir / f"{modelname}.hds"
    cbc_path = function_tmpdir / f"{modelname}.cbc"
    assert hds_path.exists()
    assert cbc_path.exists()

    converter = ClassicMfToMf6Converter.from_model(mf, hds_path, cbc_path)
    result = converter.convert(str(function_tmpdir / "mf6_output"))

    # GRB dimensions
    grb = MfGrdFile(str(result["grb"]))
    assert grb.nlay == nlay
    assert grb.nrow == nrow
    assert grb.ncol == ncol

    # ICELLTYPE pattern (Fortran interleaved: layer varies fastest)
    icelltype = grb._datadict["ICELLTYPE"]
    assert np.all(icelltype[0::nlay] == 1), "Layer 1 should be convertible"
    assert np.all(icelltype[1::nlay] == 0), "Layer 2 should be confined"
    assert np.all(icelltype[2::nlay] == 0), "Layer 3 should be confined"

    hds_orig = HeadFile(str(hds_path))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(cbc_path))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    kstpkper = hds_orig.get_kstpkper()[0]
    head_orig = hds_orig.get_data(kstpkper=kstpkper)

    # Head roundtrip
    np.testing.assert_array_almost_equal(
        hds_mf6.get_data(kstpkper=kstpkper), head_orig, decimal=5
    )

    # Face-flow roundtrip (3 layers → include lower face)
    frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
    fff_orig = cbc_orig.get_data(text="FLOW FRONT FACE", kstpkper=kstpkper)[0]
    flf_orig = cbc_orig.get_data(text="FLOW LOWER FACE", kstpkper=kstpkper)[0]
    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
    frf_rt, fff_rt, flf_rt = get_structured_faceflows(
        flowja, grb_file=str(result["grb"])
    )
    np.testing.assert_array_almost_equal(
        frf_rt[:, :, :-1], frf_orig[:, :, :-1], decimal=5
    )
    np.testing.assert_array_almost_equal(
        fff_rt[:, :-1, :], fff_orig[:, :-1, :], decimal=5
    )
    np.testing.assert_array_almost_equal(
        flf_rt[:-1, :, :], flf_orig[:-1, :, :], decimal=5
    )

    # Saturation: confined layers = 1.0; convertible layer = (head - botm) / thickness
    sat_data = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper)
    assert len(sat_data) > 0, "DATA-SAT missing"
    sat_flat = sat_data[0]["q"]  # Fortran order: layer varies fastest
    # Confined layers (1 and 2, 0-indexed) should all be 1.0
    np.testing.assert_array_equal(
        sat_flat[1::nlay], 1.0, err_msg="Layer 2 (confined) sat != 1"
    )
    np.testing.assert_array_equal(
        sat_flat[2::nlay], 1.0, err_msg="Layer 3 (confined) sat != 1"
    )
    # Convertible layer sat should match (head - botm[0]) / (top - botm[0])
    head_layer0 = head_orig[0].flatten(order="F")
    top_layer0 = top.flatten(order="F")
    bot_layer0 = botm[0].flatten(order="F")
    expected_sat_layer0 = np.clip(
        (head_layer0 - bot_layer0) / (top_layer0 - bot_layer0), 0.0, 1.0
    )
    np.testing.assert_array_almost_equal(
        sat_flat[0::nlay], expected_sat_layer0, decimal=5
    )


@requires_exe("mf2000")
@pytest.mark.slow
def test_classic_to_mf6_mf2000_bcf(function_tmpdir):
    """
    Build and run a MODFLOW-2000 model with the BCF package (laycon=3,
    fully convertible), convert via from_model(), and verify that
    get_icelltype_from_laycon() correctly maps laycon → ICELLTYPE and that
    head, face-flow, and saturation all round-trip correctly.

    This test exercises the BCF code path in from_model(), which differs
    from the LPF/UPW path: laycon values (0/1/2/3) are first converted to
    ICELLTYPE (0/1) via get_icelltype_from_laycon(), and hdry is read from
    the BCF package (default -1e30, not -999.0).
    """
    from flopy.mf6.utils import get_structured_faceflows
    from flopy.modflow import (
        Modflow,
        ModflowBas,
        ModflowBcf,
        ModflowDis,
        ModflowGhb,
        ModflowOc,
        ModflowPcg,
        ModflowRch,
    )
    from flopy.utils import ClassicMfToMf6Converter
    from flopy.utils.binaryfile import CellBudgetFile, HeadFile

    modelname = "mf2000_bcf"
    nlay, nrow, ncol = 1, 1, 100
    delr, delc = 50.0, 1.0
    h1, h2 = 20.0, 11.0
    top, botm = 25.0, 0.0
    hk = 50.0

    # Linear initial heads to avoid convergence failures with PCG and unconfined cells
    strt = np.zeros((nlay, nrow, ncol))
    strt[0, 0, :] = np.linspace(h1, h2, ncol)

    h_adj1 = h1 - (h1 - h2) / ncol
    h_adj2 = h2 + (h1 - h2) / ncol
    b1, b2 = 0.5 * (h1 + h_adj1), 0.5 * (h2 + h_adj2)
    c1 = hk * b1 * delc / (0.5 * delr)
    c2 = hk * b2 * delc / (0.5 * delr)
    ghb_dtype = ModflowGhb.get_default_dtype()
    spd = np.zeros(2, dtype=ghb_dtype).view(np.recarray)
    spd[0] = (0, 0, 0, h1, c1)
    spd[1] = (0, 0, ncol - 1, h2, c2)

    mf = Modflow(
        modelname=modelname,
        exe_name="mf2000",
        model_ws=str(function_tmpdir),
        version="mf2k",
    )
    ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        perlen=1,
        nstp=1,
        steady=True,
    )
    ModflowBas(mf, ibound=1, strt=strt)
    # laycon=3: fully convertible (T varies with saturated thickness)
    ModflowBcf(mf, laycon=3, hy=hk, ipakcb=53)
    ModflowGhb(mf, stress_period_data=spd)
    ModflowRch(mf, rech=0.001, nrchop=1)
    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )
    oc.reset_budgetunit(budgetunit=53, fname=f"{modelname}.cbc")
    ModflowPcg(mf)

    mf.write_input()
    success, _ = mf.run_model(silent=True)
    assert success, "MODFLOW-2000 BCF run failed"

    hds_path = function_tmpdir / f"{modelname}.hds"
    cbc_path = function_tmpdir / f"{modelname}.cbc"
    assert hds_path.exists()
    assert cbc_path.exists()

    # from_model() must detect BCF, call get_icelltype_from_laycon(laycon=3)→1,
    # and read hdry from bcf6 (default -1e30 for BCF).
    converter = ClassicMfToMf6Converter.from_model(mf, hds_path, cbc_path)
    assert converter.icelltype[0] == 1, "laycon=3 should give icelltype=1"
    assert converter.hdry == pytest.approx(-1e30), "hdry should come from BCF package"

    result = converter.convert(str(function_tmpdir / "mf6_output"))
    assert result["grb"].exists()
    assert result["hds"].exists()
    assert result["bud"].exists()

    hds_orig = HeadFile(str(hds_path))
    hds_mf6 = HeadFile(str(result["hds"]))
    cbc_orig = CellBudgetFile(str(cbc_path))
    bud_mf6 = CellBudgetFile(str(result["bud"]))

    kstpkper = hds_orig.get_kstpkper()[0]
    head_orig = hds_orig.get_data(kstpkper=kstpkper)

    # Head roundtrip
    np.testing.assert_array_almost_equal(
        hds_mf6.get_data(kstpkper=kstpkper), head_orig, decimal=5
    )

    # Face-flow roundtrip (1 row, 1 layer → only right face flows exist)
    frf_orig = cbc_orig.get_data(text="FLOW RIGHT FACE", kstpkper=kstpkper)[0]
    flowja = bud_mf6.get_data(text="FLOW-JA-FACE", kstpkper=kstpkper)[0]
    frf_rt, _, _ = get_structured_faceflows(flowja, grb_file=str(result["grb"]))
    np.testing.assert_array_almost_equal(
        frf_rt[:, :, :-1], frf_orig[:, :, :-1], decimal=5
    )

    # Saturation (laycon=3 → convertible → sat = clamp(head / 25, 0, 1))
    sat_data = bud_mf6.get_data(text="DATA-SAT", kstpkper=kstpkper)
    assert len(sat_data) > 0, "DATA-SAT missing"
    active = head_orig.flatten() > -1e29  # exclude hdry/hnoflo
    expected_sat = np.clip(head_orig.flatten()[active] / (top - botm), 0.0, 1.0)
    np.testing.assert_array_almost_equal(sat_data[0]["q"], expected_sat, decimal=5)
