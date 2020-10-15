try:
    import shapely
    from shapely.geometry import (
        MultiPolygon,
        Polygon,
        Point,
        MultiPoint,
        LineString,
        MultiLineString,
    )
except ImportError:
    shapely = None

try:
    import geojson
except ImportError:
    geojson = None

import numpy as np
from flopy.utils.geometry import Shape, Collection


geojson_classes = {}
if geojson is not None:
    geojson_classes = {
        "polygon": geojson.Polygon,
        "multipolygon": geojson.MultiPolygon,
        "point": geojson.Point,
        "multipoint": geojson.MultiPoint,
        "linestring": geojson.LineString,
        "multilinestring": geojson.MultiLineString,
    }

shape_types = {
    "multipolygon": "MultiPolygon",
    "polygon": "Polygon",
    "point": "Point",
    "multipoint": "MultiPoint",
    "linestring": "LineString",
    "multilinestring": "MultiLineString",
}


class GeoSpatialUtil(object):
    """
    Geospatial utils are a unifying method to provide conversion between
    commonly used geospatial input types

    Parameters
    ----------
    obj : geospatial object
        obj can accept any of the following objects:
            shapefile.Shape object
            flopy.utils.geometry objects
            list of vertices
            geojson geometry objects
            shapely.geometry objects

    shapetype : str
        shapetype is required when a list of vertices is supplied for obj

    """

    def __init__(self, obj, shapetype=None):
        from ..export.shapefile_utils import import_shapefile

        self.__shapefile = import_shapefile()
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

        if isinstance(obj, self.__shapefile.Shape):
            self.__geo_interface = self.__obj.__geo_interface__

        elif isinstance(obj, (Shape, Collection)):
            geo_interface = obj.__geo_interface__
            if geo_interface["type"] == "GeometryCollection":
                raise TypeError("GeometryCollections are not supported")

            self.__geo_interface = geo_interface

        elif isinstance(obj, (np.ndarray, list, tuple)):
            if shapetype is None or shapetype not in shape_types:
                err = "shapetype must be one of the following: " + " , ".join(
                    geojson_classes.keys()
                )
                raise AssertionError(err)

            self.__geo_interface = {
                "type": shape_types[shapetype],
                "coordinates": list(obj),
            }

        if geojson is not None:
            if isinstance(obj, geojson.Feature):
                self.__geo_interface = {
                    "type": obj.geometry.type,
                    "coordinates": obj.geometry.coordinates,
                }

            elif isinstance(
                obj,
                (
                    geojson.Point,
                    geojson.MultiPoint,
                    geojson.Polygon,
                    geojson.MultiPolygon,
                    geojson.LineString,
                    geojson.MultiLineString,
                ),
            ):
                self.__geo_interface = {
                    "type": obj.type,
                    "coordinates": obj.coordinates,
                }

        if shapely is not None:
            if isinstance(
                obj,
                (
                    Point,
                    MultiPoint,
                    Polygon,
                    MultiPolygon,
                    LineString,
                    MultiLineString,
                ),
            ):
                self.__geo_interface = obj.__geo_interface__

    @property
    def __geo_interface__(self):
        """
        Geojson standard representation of a geometry

        Returns
        -------
            dict
        """
        return self.__geo_interface

    @property
    def shapetype(self):
        """
        Shapetype string for a geometry

        Returns
        -------
            str
        """
        if self.__shapetype is None:
            self.__shapetype = self.__geo_interface["type"]
        return self.__shapetype

    @property
    def points(self):
        """
        Returns a list of vertices to the user

        Returns
        -------
            list
        """
        if self._points is None:
            self._points = self.__geo_interface["coordinates"]
        return self._points

    @property
    def shapely(self):
        """
        Returns a shapely.geometry object to the user

        Returns
        -------
            shapely.geometry.<shape>
        """
        if shapely is not None:
            if self._shapely is None:
                self._shapely = shapely.geometry.shape(self.__geo_interface)
            return self._shapely

    @property
    def geojson(self):
        """
        Returns a geojson object to the user

        Returns
        -------
            geojson.<shape>
        """
        if geojson is not None:
            if self._geojson is None:
                cls = geojson_classes[self.__geo_interface["type"].lower()]
                self._geojson = cls(self.__geo_interface["coordinates"])
            return self._geojson

    @property
    def shape(self):
        """
        Returns a shapefile.Shape object to the user

        Returns
        -------
            shapefile.shape
        """
        if self._shape is None:
            self._shape = self.__shapefile.Shape._from_geojson(
                self.__geo_interface
            )
        return self._shape

    @property
    def flopy_geometry(self):
        """
        Returns a flopy geometry object to the user

        Returns
        -------
            flopy.utils.geometry.<Shape>
        """
        if self._flopy_geometry is None:
            self._flopy_geometry = Shape.from_geojson(self.__geo_interface)
        return self._flopy_geometry


class GeoSpatialCollection(object):
    """
    The GeoSpatialCollection class allows a user to convert between
    Collection objects from common geospatial libraries.

    Parameters
    ----------
    obj : collection object
        obj can accept the following types

        str : shapefile name
        shapefile.Reader object
        list of [shapefile.Shape, shapefile.Shape,]
        shapefile.Shapes object
        flopy.utils.geometry.Collection object
        list of [flopy.utils.geometry, ...] objects
        geojson.GeometryCollection object
        geojson.FeatureCollection object
        shapely.GeometryCollection object
        list of [[vertices], ...]

    shapetype : list
        optional list of shapetypes that is required when vertices are
        supplied to the class as the obj parameter

    """

    def __init__(self, obj, shapetype=None):
        from ..export.shapefile_utils import import_shapefile

        self.__shapefile = import_shapefile()
        self.__obj = obj
        self.__collection = []
        self._geojson = None
        self._shapely = None
        self._shape = None
        self._flopy_geometry = None
        self._points = None
        self.__shapetype = None

        if isinstance(obj, str):
            with self.__shapefile.Reader(obj) as r:
                for shape in r.shapes():
                    self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, self.__shapefile.Reader):
            for shape in obj.shapes():
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, self.__shapefile.Shapes):
            for shape in obj:
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, Collection):
            for shape in obj:
                self.__collection.append(GeoSpatialUtil(shape))

        elif isinstance(obj, (np.ndarray, list, tuple)):
            if isinstance(obj[0], (Shape, Collection, self.__shapefile.Shape)):
                for shape in obj:
                    self.__collection.append(GeoSpatialUtil(shape))

            else:
                if shapetype is None:
                    err = "a list of shapetypes must be provided"
                    raise AssertionError(err)

                elif isinstance(shapetype, str):
                    shapetype = [shapetype] * len(obj)

                for ix, geom in enumerate(obj):
                    self.__collection.append(
                        GeoSpatialUtil(geom, shapetype[ix])
                    )

        if geojson is not None:
            if isinstance(
                obj,
                (
                    geojson.GeometryCollection,
                    geojson.FeatureCollection,
                    geojson.MultiLineString,
                    geojson.MultiPoint,
                    geojson.MultiPolygon,
                ),
            ):
                for geom in obj.geometries:
                    self.__collection.append(GeoSpatialUtil(geom))

        if shapely is not None:
            if isinstance(
                obj,
                (
                    shapely.geometry.collection.GeometryCollection,
                    MultiPoint,
                    MultiLineString,
                    MultiPolygon,
                ),
            ):
                for geom in list(obj):
                    self.__collection.append(GeoSpatialUtil(geom))

    def __iter__(self):
        """
        Iterator method that allows the user to get a list of GeoSpatialUtil
        objects from the GeoSpatialCollection object

        Returns
        -------
            GeoSpatialUtil
        """
        yield from self.__collection

    @property
    def shapetype(self):
        """
        Returns a list of shapetypes to the user

        Returns
        -------
            list of str
        """
        if self.__shapetype is None:
            self.__shapetype = [i.shapetype for i in self.__collection]
        return self.__shapetype

    @property
    def points(self):
        """
        Property returns a multidimensional list of vertices

        Returns
        -------
            list of vertices
        """
        if self._points is None:
            self._points = [i.points for i in self.__collection]
        return self._points

    @property
    def shapely(self):
        """
        Property that returns a shapely.geometry.collection.GeometryCollection
        object to the user

        Returns
        -------
            shapely.geometry.collection.GeometryCollection object
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
        Property that returns a geojson.GeometryCollection object to the user

        Returns
        -------
            geojson.GeometryCollection
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
        Property that returns a shapefile.Shapes object

        Returns
        -------
            shapefile.Shapes object
        """
        if self._shape is None:
            self._shape = self.__shapefile.Shapes()
            for geom in self.__collection:
                self._shape.append(geom.shape)
        return self._shape

    @property
    def flopy_geometry(self):
        """
        Property that returns a flopy.util.geometry.Collection object

        Returns
        -------
            flopy.util.geometry.Collectionnos object
        """
        if self._flopy_geometry is None:
            self._flopy_geometry = Collection(
                [i.flopy_geometry for i in self.__collection]
            )
        return self._flopy_geometry
