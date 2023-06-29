import numpy as np

from flopy.modpath import ParticleData

structured_plocs = [(1, 1, 1), (1, 1, 2)]
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
structured_array = np.core.records.fromrecords(
    [
        (1, 1, 1, 0.5, 0.5, 0.5, 0.0, 0),
        (1, 1, 2, 0.5, 0.5, 0.5, 0.0, 0),
    ],
    dtype=structured_dtype,
)


def test_particledata_structured_partlocs_as_list_of_tuples():
    locs = structured_plocs
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert np.array_equal(data.particledata, structured_array)


def test_particledata_structured_partlocs_as_ndarray():
    locs = np.array(structured_plocs)
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert np.array_equal(data.particledata, structured_array)


def test_particledata_structured_partlocs_as_list_of_lists():
    locs = [list(p) for p in structured_plocs]
    data = ParticleData(partlocs=locs, structured=True)

    assert data.particlecount == 2
    assert data.dtype == structured_dtype
    assert np.array_equal(data.particledata, structured_array)
