import warnings
from io import StringIO
from struct import pack
from tempfile import TemporaryFile
from textwrap import dedent

import numpy as np

from flopy.utils.util_array import Util2d


def test_load_txt_free():
    a = np.ones((10,), dtype=np.float32) * 250.0
    fp = StringIO("10*250.0")
    fa = Util2d.load_txt(a.shape, fp, a.dtype, "(FREE)")
    np.testing.assert_equal(fa, a)
    assert fa.dtype == a.dtype

    a = np.arange(10, dtype=np.int32).reshape((2, 5))
    fp = StringIO(
        dedent(
            """\
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
            """\
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
            """\
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
            """\
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
            """\
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
            """\
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
            """\
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
            """\
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


def test_load_bin(function_tmpdir):
    def temp_file(data):
        # writable file that is destroyed as soon as it is closed
        f = TemporaryFile(dir=function_tmpdir)
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
