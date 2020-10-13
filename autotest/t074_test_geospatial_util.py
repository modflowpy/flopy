polygon = {'type': 'Polygon',
           'coordinates': (((-121.389308, 38.560816),
                            (-121.363391, 38.568835),
                            (-121.358641, 38.565972),
                            (-121.359327, 38.562767),
                            (-121.369932, 38.560575),
                            (-121.370609, 38.557232),
                            (-121.385435, 38.555018),
                            (-121.389308, 38.560816)),)}

poly_w_hole = {'type': 'Polygon',
               'coordinates': (((-121.383097, 38.565764),
                                (-121.342866, 38.579086),
                                (-121.342739, 38.578995),
                                (-121.323309, 38.578953),
                                (-121.358295, 38.561163),
                                (-121.379047, 38.559053),
                                (-121.382318, 38.562934),
                                (-121.383097, 38.565764)),
                               ((-121.367281, 38.567214),
                                (-121.362633, 38.562622),
                                (-121.345857, 38.570301),
                                (-121.352168, 38.572258),
                                (-121.367281, 38.567214)))}

multipolygon = {'type': 'MultiPolygon',
                'coordinates': [[((-121.433775, 38.544254),
                                  (-121.424263, 38.547474),
                                  (-121.422917, 38.540376),
                                  (-121.433775, 38.544254))],
                                [((-121.456113, 38.552220),
                                  (-121.440092, 38.548303),
                                  (-121.440053, 38.537820),
                                  (-121.459991, 38.541350),
                                  (-121.456113, 38.552220))]]}

point = {'type': 'Point',
         'coordinates': (-121.358560, 38.567760)}

multipoint = {'type': 'MultiPoint',
              'coordinates': ((-121.366489, 38.565485),
                              (-121.365405, 38.563835),
                              (-121.363352, 38.566422),
                              (-121.362895, 38.564504),
                              (-121.360556, 38.565530))}

linestring = {'type': 'LineString',
              'coordinates': ((-121.360899, 38.563478),
                              (-121.358161, 38.566511),
                              (-121.355936, 38.564727),
                              (-121.354738, 38.567047),
                              (-121.356678, 38.568741),
                              (-121.361583, 38.568072),
                              (-121.363066, 38.565530),
                              (-121.364664, 38.567359))}

multilinestring = {'type': 'MultiLineString',
                   'coordinates': (((-121.370653, 38.566021),
                                    (-121.368600, 38.563255),
                                    (-121.364207, 38.563746),
                                    (-121.364550, 38.561605)),
                                   ((-121.370710, 38.560713),
                                    (-121.371338, 38.561873),
                                    (-121.370881, 38.563122),
                                    (-121.372478, 38.563345),
                                    (-121.373448, 38.560802),
                                    (-121.374361, 38.562363),
                                    (-121.373733, 38.564906),
                                    (-121.371052, 38.567091)))}


def test_import_geospatial_utils():

    from flopy.utils.geospatial_utils import GeoSpatialUtil, \
        GeoSpatialCollection

    return


def test_polygon():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, Polygon

    poly = Shape.from_geojson(polygon)
    gi1 = poly.__geo_interface__

    if not isinstance(poly, Polygon):
        raise AssertionError()

    gu = GeoSpatialUtil(poly)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'polygon').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil polygon conversion error")


def test_polygon_with_hole():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, Polygon

    poly = Shape.from_geojson(poly_w_hole)
    gi1 = poly.__geo_interface__

    if not isinstance(poly, Polygon):
        raise AssertionError()

    gu = GeoSpatialUtil(poly)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'polygon').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil polygon conversion error")


def test_multipolygon():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, MultiPolygon

    poly = Shape.from_geojson(multipolygon)
    gi1 = poly.__geo_interface__

    if not isinstance(poly, MultiPolygon):
        raise AssertionError()

    gu = GeoSpatialUtil(poly)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'multipolygon').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil multipolygon "
                                 "conversion error")


def test_point():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, Point

    pt = Shape.from_geojson(point)
    gi1 = pt.__geo_interface__

    if not isinstance(pt, Point):
        raise AssertionError()

    gu = GeoSpatialUtil(pt)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'point').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil point "
                                 "conversion error")


def test_multipoint():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, MultiPoint

    mpt = Shape.from_geojson(multipoint)
    gi1 = mpt.__geo_interface__

    if not isinstance(mpt, MultiPoint):
        raise AssertionError()

    gu = GeoSpatialUtil(mpt)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'multipoint').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil multipoint "
                                 "conversion error")


def test_linestring():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, LineString

    lstr = Shape.from_geojson(linestring)
    gi1 = lstr.__geo_interface__

    if not isinstance(lstr, LineString):
        raise AssertionError()

    gu = GeoSpatialUtil(lstr)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'linestring').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil linestring "
                                 "conversion error")


def test_multilinestring():
    from flopy.utils.geospatial_utils import GeoSpatialUtil
    from flopy.utils.geometry import Shape, MultiLineString

    mlstr = Shape.from_geojson(multilinestring)
    gi1 = mlstr.__geo_interface__

    if not isinstance(mlstr, MultiLineString):
        raise AssertionError()

    gu = GeoSpatialUtil(mlstr)

    shp = gu.shape
    shply = gu.shapely
    points = gu.points
    geojson = gu.geojson
    fp_geo = gu.flopy_geometry

    geo_types = [shp, shply, points, geojson, fp_geo]

    for geo in geo_types:
        if geo is None:  # if shapely or geojson is not installed
            continue
        t = GeoSpatialUtil(geo, 'multilinestring').flopy_geometry
        gi2 = t.__geo_interface__

        is_equal = gi1 == gi2

        if not is_equal:
            raise AssertionError("GeoSpatialUtil multilinestring "
                                 "conversion error")


def test_polygon_collection():
    from flopy.utils.geospatial_utils import GeoSpatialCollection
    from flopy.utils.geometry import Shape, Collection

    col = [Shape.from_geojson(polygon),
           Shape.from_geojson(poly_w_hole),
           Shape.from_geojson(multipolygon)]

    gi1 = [i.__geo_interface__ for i in col]
    col = Collection(col)

    gc1 = GeoSpatialCollection(col)
    shapetype = gc1.shapetype
    shp = gc1.shape
    shply = gc1.shapely
    points = gc1.points
    geojson = gc1.geojson
    fp_geo = gc1.flopy_geometry
    
    collections = [shp, shply, points, geojson, fp_geo]
    for col in collections:
        if col is None:  # if geojson or shapely is not installed
            continue

        gc2 = GeoSpatialCollection(col, shapetype)
        gi2 = [i.flopy_geometry.__geo_interface__ for i in gc2]

        for ix, gi in enumerate(gi2):
            is_equal = gi == gi1[ix]

            if not is_equal:
                raise AssertionError("GeoSpatialCollection Polygon "
                                     "conversion error")


def test_point_collection():
    from flopy.utils.geospatial_utils import GeoSpatialCollection
    from flopy.utils.geometry import Shape, Collection

    col = [Shape.from_geojson(point),
           Shape.from_geojson(multipoint)]

    gi1 = [i.__geo_interface__ for i in col]
    col = Collection(col)

    gc1 = GeoSpatialCollection(col)
    shapetype = gc1.shapetype
    shp = gc1.shape
    shply = gc1.shapely
    points = gc1.points
    geojson = gc1.geojson
    fp_geo = gc1.flopy_geometry

    collections = [shp, shply, points, geojson, fp_geo]
    for col in collections:
        if col is None:  # if geojson or shapely is not installed
            continue

        gc2 = GeoSpatialCollection(col, shapetype)
        gi2 = [i.flopy_geometry.__geo_interface__ for i in gc2]

        for ix, gi in enumerate(gi2):
            is_equal = gi == gi1[ix]

            if not is_equal:
                raise AssertionError("GeoSpatialCollection Point "
                                     "conversion error")


def test_linestring_collection():
    from flopy.utils.geospatial_utils import GeoSpatialCollection
    from flopy.utils.geometry import Shape, Collection

    col = [Shape.from_geojson(linestring),
           Shape.from_geojson(multilinestring)]

    gi1 = [i.__geo_interface__ for i in col]
    col = Collection(col)

    gc1 = GeoSpatialCollection(col)
    shapetype = gc1.shapetype
    shp = gc1.shape
    shply = gc1.shapely
    points = gc1.points
    geojson = gc1.geojson
    fp_geo = gc1.flopy_geometry

    collections = [shp, shply, points, geojson, fp_geo]
    for col in collections:
        if col is None:  # if geojson or shapely is not installed
            continue

        gc2 = GeoSpatialCollection(col, shapetype)
        gi2 = [i.flopy_geometry.__geo_interface__ for i in gc2]

        for ix, gi in enumerate(gi2):
            is_equal = gi == gi1[ix]

            if not is_equal:
                raise AssertionError("GeoSpatialCollection Linestring "
                                     "conversion error")


def test_mixed_collection():
    from flopy.utils.geospatial_utils import GeoSpatialCollection
    from flopy.utils.geometry import Shape, Collection

    col = [Shape.from_geojson(polygon),
           Shape.from_geojson(poly_w_hole),
           Shape.from_geojson(multipolygon),
           Shape.from_geojson(point),
           Shape.from_geojson(multipoint),
           Shape.from_geojson(linestring),
           Shape.from_geojson(multilinestring)]

    gi1 = [i.__geo_interface__ for i in col]
    col = Collection(col)

    gc1 = GeoSpatialCollection(col)
    shapetype = gc1.shapetype
    shp = gc1.shape
    shply = gc1.shapely
    points = gc1.points
    geojson = gc1.geojson
    fp_geo = gc1.flopy_geometry

    collections = [shp, shply, points, geojson, fp_geo]
    for col in collections:
        if col is None:  # if geojson or shapely is not installed
            continue

        gc2 = GeoSpatialCollection(col, shapetype)
        gi2 = [i.flopy_geometry.__geo_interface__ for i in gc2]

        for ix, gi in enumerate(gi2):
            is_equal = gi == gi1[ix]

            if not is_equal:
                raise AssertionError("GeoSpatialCollection conversion error")


if __name__ == "__main__":
    test_import_geospatial_utils()
    test_polygon()
    test_polygon_with_hole()
    test_multipolygon()
    test_point()
    test_multipoint()
    test_linestring()
    test_multilinestring()
    test_polygon_collection()
    test_point_collection()
    test_linestring_collection()
    test_mixed_collection()
