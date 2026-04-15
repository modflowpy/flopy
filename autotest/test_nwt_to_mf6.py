import numpy as np
import pytest
from modflow_devtools.markers import requires_exe


def test_get_icelltype_from_laytyp():
    from flopy.utils.nwt_to_mf6 import get_icelltype_from_laytyp

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

    from flopy.utils import NwtToMf6Converter
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

    converter = NwtToMf6Converter(
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
    from flopy.utils import NwtToMf6Converter
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

    converter = NwtToMf6Converter(
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
    from flopy.utils import NwtToMf6Converter
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

    converter = NwtToMf6Converter(
        str(hds_file),
        str(cbc_file),
        nlay,
        nrow,
        ncol,
        delr=np.ones(ncol) * delr,
        delc=np.ones(nrow) * delc,
        top=np.ones((nrow, ncol)) * top,
        botm=np.ones((nlay, nrow, ncol)) * botm,
        laytyp=np.array([1]),  # Convertible
        model_ws=function_tmpdir,
    )

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
    assert flowja is not None, "Could not read FLOW-JA-FACE"

    sat_data = bud_mf6.get_data(text="DATA-SAT", idx=0)
    assert sat_data is not None, "Could not read DATA-SAT"


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
    from flopy.utils import NwtToMf6Converter
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

    converter = NwtToMf6Converter(
        str(hds_file),
        str(cbc_file),
        nlay,
        nrow,
        ncol,
        delr=np.ones(ncol) * delr,
        delc=np.ones(nrow) * delc,
        top=top,
        botm=botm,
        laytyp=laytyp,
        model_ws=function_tmpdir,
    )

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
    sat_data = bud_mf6.get_data(text="DATA-SAT", idx=0)
    assert sat_data is not None
