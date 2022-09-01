import numpy as np
import pytest

from flopy.mf6.utils import get_residuals, get_structured_faceflows

pytestmark = pytest.mark.mf6


def test_get_faceflows_empty():
    flowja = np.zeros(10, dtype=np.float64)
    with pytest.raises(ValueError):
        frf, fff, flf = get_structured_faceflows(flowja)


def test_get_faceflows_jaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        frf, fff, flf = get_structured_faceflows(flowja, ia=ia)


def test_get_faceflows_iaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ja = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = get_structured_faceflows(flowja, ja=ja)


def test_get_faceflows_flowja_size():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(5, dtype=np.int32)
    ja = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = get_structured_faceflows(flowja, ia=ia, ja=ja)


def test_get_residuals_jaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = get_residuals(flowja, ia=ia)


def test_get_residuals_iaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ja = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = get_residuals(flowja, ja=ja)
