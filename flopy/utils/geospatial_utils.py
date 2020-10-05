try:
    import shapely
    from shapely.geometry import MultiPolygon, Polygon, Point, \
        MultiPoint, LineString, MultiLineString
except ImportError:
    shapely = None

try:
    import geojson
except ImportError:
    geojson = None

import shapefile
import numpy as np
from flopy.utils.geometry import Shape, Collection


geojson_classes = {}
if geojson is not None:
    geojson_classes = {'polygon': geojson.Polygon,
                       'multipolygon': geojson.MultiPolygon,
                       'point': geojson.Point,
                       'multipoint': geojson.MultiPoint,
                       'linestring': geojson.LineString,
                       'multilinestring': geojson.MultiLineString}

shape_types = {'multipolygon': "MultiPolygon",
               'polygon': "Polygon",
               "point": "Point",
               "multipoint": "MultiPoint",
               "linestring": "LineString",
               "multilinestring": "MultiLineString"}


class GeoSpatialUtil(object):
    """
    Geospatial utils are a unifying method to provide conversion between
    commonly used geospatial input types


    """
    def __init__(self, obj, shapetype=None):
        self.__obj = obj
        self.__geo_interface = {}
        self._geojson = None
        self._shapely = None
        self._shape = None
        self._flopy_geometry = None
        self._points = None
        self.__shapetype = None

        if shapetype is not None:
            shapetype = shapetype.lower()

        if isinstance(obj, shapefile.Shape):
            self.__geo_interface = self.__obj.__geo_interface__

        elif isinstance(obj, (Shape, Collection)):
            geo_interface = obj.__geo_interface__
            if geo_interface['type'] == "Shapes":
                raise TypeError("Mixed Shape collections not supported")

            self.__geo_interface = geo_interface

        elif isinstance(obj, (np.ndarray, list, tuple)):
            if shapetype is None or shapetype not in shape_types:
                err = "shapetype must be one of the following: " + \
                      " , ".join(geojson_classes.keys())
                raise AssertionError(err)

            self.__geo_interface = {'type': shape_types[shapetype],
                                    "coordinates": list(obj)}

        if geojson is not None:
            if isinstance(obj, geojson.Feature):
                self.__geo_interface = \
                    {"type": obj.geometry.type,
                     "coordinates": obj.geometry.coordinates}

            elif isinstance(obj, (geojson.Point, geojson.MultiPoint,
                                  geojson.Polygon, geojson.MultiPolygon,
                                  geojson.LineString,
                                  geojson.MultiLineString)):
                self.__geo_interface = {"type": obj.type,
                                        "coordinates": obj.coordinates}

        if shapely is not None:
            if isinstance(obj, (Point, MultiPoint, Polygon,
                                MultiPolygon, LineString, MultiLineString)):
                self.__geo_interface = obj.__geo_interface__

    @property
    def __geo_interface__(self):
        return self.__geo_interface

    @property
    def shapetype(self):
        """

        Returns
        -------

        """
        if self.__shapetype is None:
            self.__shapetype = self.__geo_interface['type']
        return self.__shapetype

    @property
    def points(self):
        """

        Returns
        -------

        """
        if self._points is None:
            self._points = self.__geo_interface['coordinates']
        return self._points

    @property
    def shapely(self):
        """

        Returns
        -------

        """
        if shapely is not None:
            if self._shapely is None:
                self._shapely = shapely.geometry.shape(self.__geo_interface)
            return self._shapely

    @property
    def geojson(self):
        if geojson is not None:
            if self._geojson is None:
                cls = geojson_classes[self.__geo_interface['type'].lower()]
                self._geojson = cls(self.__geo_interface['coordinates'])
            return self._geojson

    @property
    def shape(self):
        """

        Returns
        -------

        """
        if self._shape is None:
            self._shape = shapefile.Shape._from_geojson(self.__geo_interface)
        return self._shape

    @property
    def flopy_geometry(self):
        """

        Returns
        -------

        """
        if self._flopy_geometry is None:
            self._flopy_geometry = Shape.from_geojson(self.__geo_interface)
        return self._flopy_geometry


class GeoSpatialCollection(object):
    """

    """
    def __init__(self, obj, shapetype=None):
        self.__obj = obj
        self.__collection = []
        self._geojson = None
        self._shapely = None
        self._shape = None
        self._flopy_geometry = None
        self._points = None
        self.__shapetype = None

        if isinstance(obj, str):
            with shapefile.Reader(obj) as r:
                for shape in r.shapes():
                    self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, shapefile.Reader):
            for shape in obj.shapes():
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, shapefile.Shapes):
            for shape in obj:
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, Collection):
            for shape in obj:
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, (np.ndarray, list, tuple)):
            if shapetype is None:
                err = "a list of shapetypes must be provided"
                raise AssertionError(err)

            for ix, geom in enumerate(obj):
                self.__collection.append(GeoSpatialUtil(geom, shapetype[ix]))

        if geojson is not None:
            if isinstance(obj, (geojson.GeometryCollection,
                                geojson.FeatureCollection,
                                geojson.MultiLineString,
                                geojson.MultiPoint,
                                geojson.MultiPolygon)):
                for geom in obj.geometries:
                    self.__collection.append(GeoSpatialUtil(geom))

        if shapely is not None:
            if isinstance(obj, (shapely.geometry.collection.GeometryCollection,
                                MultiPoint,
                                MultiLineString,
                                MultiPolygon)):
                for geom in list(obj):
                    self.__collection.append(GeoSpatialUtil(geom))

    @property
    def shapetype(self):
        """

        Returns
        -------

        """
        if self.__shapetype is None:
            self.__shapetype = [i.shapetype for i in self.__collection]
        return self.__shapetype

    @property
    def points(self):
        """

        Returns
        -------

        """
        if self._points is None:
            self._points = [i.points for i in self.__collection]
        return self._points

    @property
    def shapely(self):
        """

        Returns
        -------

        """
        if shapely is not None:
            if self._shapely is None:
                self._shapely = shapely.geometry.collection.GeometryCollection(
                    [i.shapely for i in self.__collection]
                )

        return self._shapely

    @property
    def geojson(self):
        """

        Returns
        -------

        """
        if geojson is not None:
            if self._geojson is None:
                self._geojson = geojson.GeometryCollection(
                    [i.geojson for i in self.__collection]
                )
        return self._geojson

    @property
    def shape(self):
        """

        Returns
        -------

        """
        if self._shape is None:
            self._shape = shapefile.Shapes()
            for geom in self.__collection:
                self._shape.append(geom.shape)
        return self._shape

    @property
    def flopy_geometry(self):
        """

        Returns
        -------

        """
        if self._flopy_geometry is None:
            self._flopy_geometry = Collection(
                [i.flopy_geometry for i in self.__collection]
            )
        return self._flopy_geometry


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import os
    ws = r"C:\Users\jlarsen\Desktop\DataCollector"
    tests = ["polygon_test.shp", "hole_test.shp", "multipolygon_test.shp"]

    for t in tests:
        test = os.path.join(ws, 'data', t)

        with shapefile.Reader(test) as rr:
            gc = GeoSpatialCollection(rr)
            pt_col = gc.points
            sh_col = gc.shapely
            geo_col = gc.geojson
            shp_col = gc.shape
            fp_col = gc.flopy_geometry
            sh_type = gc.shapetype

            gc2 = GeoSpatialCollection(fp_col)
            pt_col = gc2.points
            sh_col = gc2.shapely
            geo_col = gc2.geojson
            shp_col = gc2.shape
            fp_col = gc2.flopy_geometry

        with shapefile.Reader(test) as r:
            for shape in r.shapes():
                gu = GeoSpatialUtil(shape)
                feat = gu.geojson
                sh_feat = gu.shapely
                shpe = gu.shape
                pnt = gu.points
                fpshp = gu.flopy_geometry

                gu2 = GeoSpatialUtil(sh_feat)
                feat = gu2.geojson
                sh_feat = gu2.shapely
                shpe = gu2.shape
                pnt = gu2.points
