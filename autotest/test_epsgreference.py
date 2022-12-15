from flaky import flaky

from flopy.export.shapefile_utils import CRS, EpsgReference


@flaky
def test_epsgreference():
    ep = EpsgReference()
    ep.reset()
    ep.show()

    prjtxt = CRS.getprj(32614)  # WGS 84 / UTM zone 14N
    if prjtxt is None:
        print("unable to retrieve CRS prj txt")
        return
    assert isinstance(prjtxt, str), type(prjtxt)
    prj = ep.to_dict()
    assert 32614 in prj
    ep.show()

    ep.add(9999, "junk")
    prj = ep.to_dict()
    assert 9999 in prj
    assert ep.get(9999) == "junk"
    ep.show()

    ep.remove(9999)
    prj = ep.to_dict()
    assert 9999 not in prj
    ep.show()

    assert ep.get(9999) is None

    ep.reset()
    prj = ep.to_dict()
    assert len(prj) == 0
    ep.show()
