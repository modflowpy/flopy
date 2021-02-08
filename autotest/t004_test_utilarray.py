import os
import shutil
import numpy as np
import flopy
import warnings
from io import StringIO
from struct import pack
from tempfile import TemporaryFile
from textwrap import dedent
from flopy.utils.util_array import Util2d, Util3d, Transient2d, Transient3d

out_dir = os.path.join("temp", "t004")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)


def test_load_txt_free():
    a = np.ones((10,), dtype=np.float32) * 250.0
    fp = StringIO(u"10*250.0")
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(FREE)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.arange(10, dtype=np.int32).reshape((2, 5))
    fp = StringIO(
        dedent(
            u"""\
        0 1,2,3, 4
        5 6, 7,  8 9
    """
        )
    )
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(FREE)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.ones((2, 5), dtype=np.float32)
    a[1, 0] = 2.2
    fp = StringIO(
        dedent(
            u"""\
        5*1.0
        2.2 2*1.0, +1E-00 1.0
    """
        )
    )
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(FREE)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype


def test_load_txt_fixed():
    a = np.arange(10, dtype=np.int32).reshape((2, 5))
    fp = StringIO(
        dedent(
            u"""\
        01234X
        56789
    """
        )
    )
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(5I1)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    fp = StringIO(
        dedent(
            u"""\
        0123X
        4
        5678
        9
    """
        )
    )
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(4I1)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.array([[-1, 1, -2, 2, -3], [3, -4, 4, -5, 5]], np.int32)
    fp = StringIO(
        dedent(
            u"""\
        -1 1-2 2-3
        3 -44 -55
    """
        )
    )
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(5I2)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype


def test_load_block():
    a = np.ones((2, 5), dtype=np.int32) * 4
    fp = StringIO(
        dedent(
            u"""\
        1
        1 2 1 5 4
    """
        )
    )
    fa = Util2d.load_block(a.shape, fp, a.dtype)
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.ones((2, 5), dtype=np.float32) * 4
    a[0:2, 1:2] = 9.0
    a[0, 2:4] = 6.0
    fp = StringIO(
        dedent(
            u"""\
        3
        1 2 1 5 4.0
        1 2 2 2 9.0
        1 1 3 4 6.0
    """
        )
    )
    fa = Util2d.load_block(a.shape, fp, a.dtype)
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.zeros((2, 5), dtype=np.int32)
    a[0, 2:4] = 8
    fp = StringIO(
        dedent(
            u"""\
        1
        1 1 3 4 8
    """
        )
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fa = Util2d.load_block(a.shape, fp, a.dtype)
        assert len(w) == 1
        assert "blocks do not cover full array" in str(w[-1].message)
        np.testing.assert_equal(fa, a)
        assert fa.dtype == a.dtype


def test_load_bin():
    def temp_file(data):
        # writable file that is destroyed as soon as it is closed
        f = TemporaryFile()
        f.write(data)
        f.seek(0)
        return f

    # INTEGER
    a = np.arange(3 * 4, dtype=np.int32).reshape((3, 4)) - 1
    fp = temp_file(a.tobytes())
    fh, fa = Util2d.load_bin((3, 4), fp, np.int32)
    assert fh is None  # no header_dtype
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    # check warning if wrong integer type is used to read 4-byte integers
    # e.g. on platforms where int -> int64
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fp.seek(0)
        fh, fa = Util2d.load_bin((3, 4), fp, np.int64)
        fp.close()
        assert len(w) == 1
        assert a.dtype == np.int32
        assert fh is None  # no header_dtype
        np.testing.assert_equal(fa, a)

    # REAL
    real_header_fmt = "2i2f16s3i"
    header_data = (1, 2, 3.5, 4.5, b"Hello", 6, 7, 8)
    real_header = pack(real_header_fmt, *header_data)
    assert len(real_header) == 44

    a = np.arange(10).reshape((2, 5))
    fp = temp_file(real_header + pack("10f", *list(range(10))))
    fh, fa = Util2d.load_bin((2, 5), fp, np.float32, "Head")
    fp.close()
    for h1, h2 in zip(fh[0], header_data):
        assert h1 == h2
    np.testing.assert_equal(a.astype(np.float32), fa)
    assert fa.dtype == np.float32

    # DOUBLE PRECISION
    dbl_header_fmt = "2i2d16s3i"
    dbl_header = pack(dbl_header_fmt, *header_data)
    assert len(dbl_header) == 52

    fp = temp_file(real_header + pack("10d", *list(range(10))))
    fh, fa = Util2d.load_bin((2, 5), fp, np.float64, "Head")
    fp.close()
    for h1, h2 in zip(fh[0], header_data):
        assert h1 == h2
    np.testing.assert_equal(a.astype(np.float64), fa)
    assert fa.dtype == np.float64


def test_transient2d():
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=10, nrow=10, ncol=10, nper=3)
    t2d = Transient2d(ml, (10, 10), np.float32, 10.0, "fake")
    a1 = t2d.array
    assert a1.shape == (3, 1, 10, 10), a1.shape
    t2d.cnstnt = 2.0
    assert np.array_equal(t2d.array, np.zeros((3, 1, 10, 10)) + 20.0)

    t2d[0] = 1.0
    t2d[2] = 999
    assert np.array_equal(t2d[0].array, np.ones((ml.nrow, ml.ncol)))
    assert np.array_equal(t2d[2].array, np.ones((ml.nrow, ml.ncol)) * 999)

    m4d = t2d.array
    t2d2 = Transient2d.from_4d(ml, "rch", {"rech": m4d})
    m4d2 = t2d2.array
    assert np.array_equal(m4d, m4d2)


def test_transient3d():
    nlay = 3
    nrow = 4
    ncol = 5
    nper = 5
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)

    # Make a transient 3d array of a constant value
    t3d = Transient3d(ml, (nlay, nrow, ncol), np.float32, 10.0, "fake")
    a1 = t3d.array
    assert a1.shape == (nper, nlay, nrow, ncol), a1.shape

    # Make a transient 3d array with changing entries and then verify that
    # they can be reproduced through indexing
    a = np.arange((nlay * nrow * ncol), dtype=np.float32).reshape((nlay, nrow, ncol))
    t3d = {0: a, 2: 1025, 3: a, 4: 1000.0}
    t3d = Transient3d(ml, (nlay, nrow, ncol), np.float32, t3d, "fake")
    assert np.array_equal(t3d[0].array, a)
    assert np.array_equal(t3d[1].array, a)
    assert np.array_equal(t3d[2].array, np.zeros((nlay, nrow, ncol)) + 1025.0)
    assert np.array_equal(t3d[3].array, a)
    assert np.array_equal(t3d[4].array, np.zeros((nlay, nrow, ncol)) + 1000.0)

    # Test changing a value
    t3d[0] = 1.0
    assert np.array_equal(t3d[0].array, np.zeros((nlay, nrow, ncol)) + 1.0)

    # Check itmp and file_entry
    itmp, file_entry_dense = t3d.get_kper_entry(0)
    assert itmp == 1
    itmp, file_entry_dense = t3d.get_kper_entry(1)
    assert itmp == -1


def test_util2d():
    ml = flopy.modflow.Modflow()
    u2d = Util2d(ml, (10, 10), np.float32, 10.0, "test")
    a1 = u2d.array
    a2 = np.ones((10, 10), dtype=np.float32) * 10.0
    assert np.array_equal(a1, a2)

    # test external filenames - ascii and binary
    fname_ascii = os.path.join(out_dir, "test_a.dat")
    fname_bin = os.path.join(out_dir, "test_b.dat")
    np.savetxt(fname_ascii, a1, fmt="%15.6E")
    u2d.write_bin(a1.shape, fname_bin, a1, bintype="head")
    dis = flopy.modflow.ModflowDis(ml, 2, 10, 10)
    lpf = flopy.modflow.ModflowLpf(ml, hk=[fname_ascii, fname_bin])
    ml.lpf.hk[1].fmtin = "(BINARY)"
    assert np.array_equal(lpf.hk[0].array, a1)
    assert np.array_equal(lpf.hk[1].array, a1)

    # test external filenames - ascii and binary with model_ws and external_path
    ml = flopy.modflow.Modflow(
        model_ws=out_dir, external_path=os.path.join(out_dir, "ref")
    )
    u2d = Util2d(ml, (10, 10), np.float32, 10.0, "test")
    fname_ascii = os.path.join(out_dir, "test_a.dat")
    fname_bin = os.path.join(out_dir, "test_b.dat")
    np.savetxt(fname_ascii, a1, fmt="%15.6E")
    u2d.write_bin(a1.shape, fname_bin, a1, bintype="head")
    dis = flopy.modflow.ModflowDis(ml, 2, 10, 10)
    lpf = flopy.modflow.ModflowLpf(ml, hk=[fname_ascii, fname_bin])
    ml.lpf.hk[1].fmtin = "(BINARY)"
    assert np.array_equal(lpf.hk[0].array, a1)
    assert np.array_equal(lpf.hk[1].array, a1)

    # bin read write test
    fname = os.path.join(out_dir, "test.bin")
    u2d.write_bin((10, 10), fname, u2d.array)
    a3 = u2d.load_bin((10, 10), fname, u2d.dtype)[1]
    assert np.array_equal(a3, a1)
    # ascii read write test
    fname = os.path.join(out_dir, "text.dat")
    u2d.write_txt((10, 10), fname, u2d.array)
    a4 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(a1, a4)

    # fixed format read/write with touching numbers - yuck!
    data = np.arange(100).reshape(10, 10)
    u2d_arange = Util2d(ml, (10, 10), np.float32, data, "test")
    u2d_arange.write_txt(
        (10, 10), fname, u2d_arange.array, python_format=[7, "{0:10.4E}"]
    )
    a4a = u2d.load_txt((10, 10), fname, np.float32, "(7E10.6)")
    assert np.array_equal(u2d_arange.array, a4a)

    # test view vs copy with .array
    a5 = u2d.array
    a5 += 1
    assert not np.array_equal(a5, u2d.array)

    # Util2d.__mul__() overload
    new_2d = u2d * 2
    assert np.array_equal(new_2d.array, u2d.array * 2)

    # test the cnstnt application
    u2d.cnstnt = 2.0
    a6 = u2d.array
    assert not np.array_equal(a1, a6)
    u2d.write_txt((10, 10), fname, u2d.array)
    a7 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(u2d.array, a7)


def stress_util2d(ml, nlay, nrow, ncol):
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol)
    hk = np.ones((nlay, nrow, ncol))
    vk = np.ones((nlay, nrow, ncol)) + 1.0
    # save hk up one dir from model_ws
    fnames = []
    for i, h in enumerate(hk):
        fname = os.path.join(out_dir, "test_{0}.ref".format(i))
        fnames.append(fname)
        np.savetxt(fname, h, fmt="%15.6e", delimiter="")
        vk[i] = i + 1.0

    lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vk)
    # util2d binary check
    ml.lpf.vka[0].format.binary = True

    # util3d cnstnt propagation test
    ml.lpf.vka.cnstnt = 2.0
    ml.write_input()

    # check that binary is being respect - it can't get no respect!
    vka_1 = ml.lpf.vka[0]
    a = vka_1.array
    vka_1_2 = vka_1 * 2.0
    assert np.array_equal(a * 2.0, vka_1_2.array)

    if ml.external_path is not None:
        files = os.listdir(os.path.join(ml.model_ws, ml.external_path))
    else:
        files = os.listdir(ml.model_ws)

    print("\n\nexternal files: " + ",".join(files) + "\n\n")
    ml1 = flopy.modflow.Modflow.load(
        ml.namefile, model_ws=ml.model_ws, verbose=True, forgive=False
    )
    print("testing load")
    assert ml1.load_fail == False
    # check that both binary and cnstnt are being respected through
    # out the write and load process.
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.vka.array, ml.lpf.vka.array)
    assert np.array_equal(ml1.lpf.hk.array, hk)
    assert np.array_equal(ml1.lpf.hk.array, ml.lpf.hk.array)

    print("change model_ws")
    ml.model_ws = out_dir
    ml.write_input()
    if ml.external_path is not None:
        files = os.listdir(os.path.join(ml.model_ws, ml.external_path))
    else:
        files = os.listdir(ml.model_ws)
    print("\n\nexternal files: " + ",".join(files) + "\n\n")
    ml1 = flopy.modflow.Modflow.load(
        ml.namefile, model_ws=ml.model_ws, verbose=True, forgive=False
    )
    print("testing load")
    assert ml1.load_fail == False
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.hk.array, hk)

    # more binary testing
    ml.lpf.vka[0]._array[0, 0] *= 3.0
    ml.write_input()
    ml1 = flopy.modflow.Modflow.load(
        ml.namefile, model_ws=ml.model_ws, verbose=True, forgive=False
    )
    assert np.array_equal(ml.lpf.vka.array, ml1.lpf.vka.array)
    assert np.array_equal(ml.lpf.hk.array, ml1.lpf.hk.array)


def stress_util2d_for_joe_the_file_king(ml, nlay, nrow, ncol):
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol)
    hk = np.ones((nlay, nrow, ncol))
    vk = np.ones((nlay, nrow, ncol)) + 1.0
    # save hk up one dir from model_ws
    fnames = []
    for i, h in enumerate(hk):
        fname = os.path.join("test_{0}.ref".format(i))
        fnames.append(fname)
        np.savetxt(fname, h, fmt="%15.6e", delimiter="")
        vk[i] = i + 1.0

    lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vk)
    ml.lpf.vka[0].format.binary = True
    ml.lpf.vka.cnstnt = 2.0
    ml.write_input()

    assert np.array_equal(ml.lpf.hk.array, hk)
    assert np.array_equal(ml.lpf.vka.array, vk * 2.0)

    ml1 = flopy.modflow.Modflow.load(
        ml.namefile, model_ws=ml.model_ws, verbose=True, forgive=False
    )
    print("testing load")
    assert ml1.load_fail == False
    assert np.array_equal(ml1.lpf.vka.array, vk * 2.0)
    assert np.array_equal(ml1.lpf.hk.array, hk)
    assert np.array_equal(ml1.lpf.vka.array, ml.lpf.vka.array)
    assert np.array_equal(ml1.lpf.hk.array, ml.lpf.hk.array)

    # more binary testing
    ml.lpf.vka[0]._array[0, 0] *= 3.0
    ml.write_input()
    ml1 = flopy.modflow.Modflow.load(
        ml.namefile, model_ws=ml.model_ws, verbose=True, forgive=False
    )
    assert np.array_equal(ml.lpf.vka.array, ml1.lpf.vka.array)
    assert np.array_equal(ml.lpf.hk.array, ml1.lpf.hk.array)


def test_util2d_external_free():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ml = flopy.modflow.Modflow(model_ws=model_ws)
    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_free_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow()
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_free_path():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(model_ws=model_ws, external_path=ext_path)
    stress_util2d(ml, 1, 1, 1)

    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_free_path_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    base_dir = os.getcwd()
    os.chdir(out_dir)
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(external_path=ext_path)

    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_fixed():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ml = flopy.modflow.Modflow(model_ws=model_ws)
    ml.array_free_format = False

    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_fixed_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)

    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow()
    ml.array_free_format = False
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util2d_external_fixed_path():
    model_ws = os.path.join(out_dir, "extra_temp")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)
    ml = flopy.modflow.Modflow(model_ws=model_ws, external_path=ext_path)
    ml.array_free_format = False

    stress_util2d(ml, 1, 1, 1)
    stress_util2d(ml, 10, 1, 1)
    stress_util2d(ml, 1, 10, 1)
    stress_util2d(ml, 1, 1, 10)
    stress_util2d(ml, 10, 10, 1)
    stress_util2d(ml, 1, 10, 10)
    stress_util2d(ml, 10, 1, 10)
    stress_util2d(ml, 10, 10, 10)


def test_util2d_external_fixed_path_nomodelws():
    model_ws = os.path.join(out_dir)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.mkdir(model_ws)
    ext_path = "ref"
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path)

    base_dir = os.getcwd()
    os.chdir(out_dir)
    ml = flopy.modflow.Modflow(external_path=ext_path)
    ml.array_free_format = False
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 1)
    stress_util2d_for_joe_the_file_king(ml, 1, 10, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 1, 10)
    stress_util2d_for_joe_the_file_king(ml, 10, 10, 10)
    os.chdir(base_dir)


def test_util3d():
    ml = flopy.modflow.Modflow()
    u3d = Util3d(ml, (10, 10, 10), np.float32, 10.0, "test")
    a1 = u3d.array
    a2 = np.ones((10, 10, 10), dtype=np.float32) * 10.0
    assert np.array_equal(a1, a2)

    new_3d = u3d * 2.0
    assert np.array_equal(new_3d.array, u3d.array * 2)

    # test the mult list-based overload for Util3d
    mult = [2.0] * 10
    mult_array = (u3d * mult).array
    assert np.array_equal(mult_array, np.zeros((10, 10, 10)) + 20.0)
    u3d.cnstnt = 2.0
    assert not np.array_equal(a1, u3d.array)

    return


def test_arrayformat():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    u2d = Util2d(ml, (15, 2), np.float32, np.ones((15, 2)), "test")

    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.npl = 1
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.npl = 2
    u2d.format.width = 8
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.free = True
    u2d.format.width = 8
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.free = False
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.fmtin = "(10G15.6)"
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()

    u2d.format.binary = True
    fmt_fort = u2d.format.fortran
    cr = u2d.get_internal_cr()
    parsed = Util2d.parse_control_record(cr)
    print(fmt_fort, parsed["fmtin"])
    assert fmt_fort.upper() == parsed["fmtin"].upper()


def test_new_get_file_entry():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    u2d = Util2d(ml, (5, 2), np.float32, np.ones((5, 2)), "test", locat=99)
    print(u2d.get_file_entry(how="internal"))
    print(u2d.get_file_entry(how="constant"))
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = False
    print(u2d.get_file_entry(how="openclose"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="openclose"))

    ml.array_free_format = False
    u2d = Util2d(ml, (5, 2), np.float32, np.ones((5, 2)), "test", locat=99)
    print(u2d.get_file_entry(how="internal"))
    print(u2d.get_file_entry(how="constant"))
    print(u2d.get_file_entry(how="external"))
    u2d.format.binary = True
    print(u2d.get_file_entry(how="external"))


def test_append_mflist():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    dis = flopy.modflow.ModflowDis(ml, 10, 10, 10, 10)
    sp_data1 = {3: [1, 1, 1, 1.0], 5: [1, 2, 4, 4.0]}
    sp_data2 = {0: [1, 1, 3, 3.0], 8: [9, 2, 4, 4.0]}
    wel1 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data1)
    wel2 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data2)
    wel3 = flopy.modflow.ModflowWel(
        ml, stress_period_data=wel2.stress_period_data.append(wel1.stress_period_data)
    )
    ml.write_input()


def test_mflist():
    ml = flopy.modflow.Modflow(model_ws=out_dir)
    dis = flopy.modflow.ModflowDis(ml, 10, 10, 10, 10)
    sp_data = {0: [[1, 1, 1, 1.0], [1, 1, 2, 2.0], [1, 1, 3, 3.0]], 1: [1, 2, 4, 4.0]}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data)
    spd = wel.stress_period_data

    # verify dataframe can be cast when spd.data.keys() != to ml.nper
    # verify that dataframe is cast correctly by recreating spd.data items
    df = wel.stress_period_data.get_dataframe()
    df = df.set_index(["per", "k", "i", "j"])
    for per, data in spd.data.items():
        dfdata = (
            df.xs(per, level="per")
            .dropna(
                subset=[
                    "flux",
                ],
                axis=0,
            )
            .loc[
                :,
                [
                    "flux",
                ],
            ]
            .to_records(index=True)
            .astype(data.dtype)
        )
        errmsg = "data not equal:\n  {}\n  {}".format(dfdata, data)
        assert np.array_equal(dfdata, data), errmsg

    m4ds = ml.wel.stress_period_data.masked_4D_arrays
    sp_data = flopy.utils.MfList.masked4D_arrays_to_stress_period_data(
        flopy.modflow.ModflowWel.get_default_dtype(), m4ds
    )
    assert np.array_equal(sp_data[0], ml.wel.stress_period_data[0])
    assert np.array_equal(sp_data[1], ml.wel.stress_period_data[1])
    # the last entry in sp_data (kper==9) should equal the last entry
    # with actual data in the well file (kper===1)
    assert np.array_equal(sp_data[9], ml.wel.stress_period_data[1])

    pth = os.path.join("..", "examples", "data", "mf2005_test")
    ml = flopy.modflow.Modflow.load(os.path.join(pth, "swi2ex4sww.nam"), verbose=True)
    m4ds = ml.wel.stress_period_data.masked_4D_arrays

    sp_data = flopy.utils.MfList.masked4D_arrays_to_stress_period_data(
        flopy.modflow.ModflowWel.get_default_dtype(), m4ds
    )

    # make a new wel file
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data)
    flx1 = m4ds["flux"]
    flx2 = wel.stress_period_data.masked_4D_arrays["flux"]

    flx1 = np.nan_to_num(flx1)
    flx2 = np.nan_to_num(flx2)

    assert flx1.sum() == flx2.sum()

    # test get_dataframe() on mflist obj
    sp_data3 = {
        0: [1, 1, 1, 1.0],
        1: [[1, 1, 3, 3.0], [1, 1, 2, 6.0]],
        2: [
            [1, 2, 4, 8.0],
            [1, 2, 3, 4.0],
            [1, 2, 2, 4.0],
            [1, 1, 3, 3.0],
            [1, 1, 2, 6.0],
        ],
    }
    wel4 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data3)
    df = wel4.stress_period_data.get_dataframe()
    df = df.set_index(["per", "k", "i", "j"])
    assert df.loc[0, "flux"].sum() == 1.0
    assert df.loc[1, "flux"].sum() == 9.0
    assert df.loc[2, "flux"].sum() == 25.0
    sp_data4 = {
        0: [1, 1, 1, 1.0],
        1: [[1, 1, 3, 3.0], [1, 1, 3, 6.0]],
        2: [
            [1, 2, 4, 8.0],
            [1, 2, 4, 4.0],
            [1, 2, 4, 4.0],
            [1, 1, 3, 3.0],
            [1, 1, 3, 6.0],
        ],
    }
    wel5 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data4)
    df = wel5.stress_period_data.get_dataframe()
    df = df.groupby(["per", "k", "i", "j"]).sum()
    assert df.loc[0, "flux"].sum() == 1.0
    assert df.loc[1, "flux"].sum() == 9.0
    assert (
        df.loc[(2, 1, 1, 3), "flux"] == 9.0
        )
    assert (
        df.loc[(2, 1, 2, 4), "flux"] == 16.0
    )


def test_how():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(model_ws=out_dir)
    ml.array_free_format = False
    dis = flopy.modflow.ModflowDis(ml, nlay=2, nrow=10, ncol=10)

    arr = np.ones((ml.nrow, ml.ncol))
    u2d = flopy.utils.Util2d(ml, arr.shape, np.float32, arr, "test", locat=1)
    print(u2d.get_file_entry())
    u2d.how = "constant"
    print(u2d.get_file_entry())
    u2d.fmtin = "(binary)"
    print(u2d.get_file_entry())


def test_util3d_reset():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(model_ws=out_dir)
    ml.array_free_format = False
    dis = flopy.modflow.ModflowDis(ml, nlay=2, nrow=10, ncol=10)
    bas = flopy.modflow.ModflowBas(ml, strt=999)
    arr = np.ones((ml.nlay, ml.nrow, ml.ncol))
    ml.bas6.strt = arr


if __name__ == "__main__":
    # test_util3d_reset()
    test_mflist()
    # test_new_get_file_entry()
    # test_arrayformat()
    # test_util2d_external_free_nomodelws()
    # test_util2d_external_free_path_nomodelws()
    # test_util2d_external_free()
    # test_util2d_external_free_path()
    # test_util2d_external_fixed()
    # test_util2d_external_fixed_path()
    # test_util2d_external_fixed_nomodelws()
    # test_util2d_external_fixed_path_nomodelws()
    # test_transient2d()
    # test_transient3d()
    # test_util2d()
    # test_util3d()
    # test_how()
