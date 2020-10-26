"""
Container objects for working with geometric information
"""
import numpy as np


class Shape(object):
    """
    Parent class for handling geo interfacing, do not instantiate directly

    Parameters:
    ----------
    type : str
        shapetype string
    coordinates : list or tuple
        list of tuple of point or linestring coordinates
    exterior : list or tuple
        2d list of polygon coordinates
    interiors : list or tuple
        2d or 3d list of polygon interiors

    """

    def __init__(
        self,
        shapetype,
        coordinates=None,
        exterior=None,
        interiors=None,
    ):
        self.__type = shapetype

        if shapetype == "Polygon":
            self.exterior = tuple(map(tuple, exterior))
            self.interiors = (
                tuple()
                if interiors is None
                else (tuple(map(tuple, i)) for i in interiors)
            )
            self.interiors = tuple(self.interiors)

        elif shapetype == "LineString":
            self.coords = list(map(tuple, coordinates))
            if len(self.coords[0]) == 3:
                self.has_z = True

        elif shapetype == "Point":
            while len(coordinates) == 1:
                coordinates = coordinates[0]

            self.coords = coordinates
            if len(coordinates) == 3:
                self.has_z = True
        else:
            err = (
                "Supported shape types are Polygon, LineString, "
                "and Point: Supplied shape type {}".format(shapetype)
            )
            raise TypeError(err)

    @property
    def __geo_interface__(self):
        """
        Creates the geojson standard representation of a shape

        Returns
        -------
            dict
        """
        geo_interface = {}

        if self.__type == "Polygon":
            geo_interface = {
                "coordinates": tuple(
                    [self.exterior] + [i for i in self.interiors]
                ),
                "type": self.__type,
            }

        elif self.__type == "LineString":
            geo_interface = {
                "coordinates": tuple(self.coords),
                "type": self.__type,
            }

        elif self.__type == "Point":
            geo_interface = {
                "coordinates": tuple(self.coords),
                "type": self.__type,
            }

        return geo_interface

    @property
    def geojson(self):
        return self.__geo_interface__

    @staticmethod
    def from_geojson(geo_interface):
        """
        Method to load from geojson

        Parameters
        ----------
        geo_interface : geojson, dict
            geojson compliant representation of a linestring

        Returns
        -------
            Polygon, LineString, or Point
        """
        if geo_interface["type"] in ("Polygon", "MultiPolygon"):
            coord_list = geo_interface["coordinates"]
            if geo_interface["type"] == "Polygon":
                coord_list = [coord_list]

            geoms = []
            for coords in coord_list:
                exteriors = coords[0]
                interiors = None
                if len(coords) > 1:
                    interiors = coords[1:]

                geoms.append(Polygon(exteriors, interiors))

            if len(geoms) == 1:
                shape = geoms[0]
            else:
                shape = MultiPolygon(geoms)

        elif geo_interface["type"] == "LineString":
            shape = LineString(geo_interface["coordinates"])

        elif geo_interface["type"] == "MultiLineString":
            geoms = [
                LineString(coords) for coords in geo_interface["coordinates"]
            ]
            shape = MultiLineString(geoms)

        elif geo_interface["type"] == "Point":
            shape = Point(geo_interface["coordinates"])

        elif geo_interface["type"] == "MultiPoint":
            geoms = [Point(coords) for coords in geo_interface["coordinates"]]
            shape = MultiPoint(geoms)

        else:
            err = (
                "Supported shape types are Polygon, LineString, and "
                "Point: Supplied shape type {}".format(geo_interface["type"])
            )
            raise TypeError(err)

        return shape


class Collection(list):
    """
    The collection object is container for a group of flopy geometries

    This class acts as a base class for MultiPoint, MultiLineString, and
    MultiPolygon classes. This class can also accept a mix of geometries
    and act as a stand alone container.

    Parameters
    ----------
    geometries : list
        list of flopy.util.geometry objects

    """

    def __init__(self, geometries=()):
        super(Collection, self).__init__(geometries)

    def __repr__(self):
        return "Shapes: {}".format(list(self))

    @property
    def __geo_interface__(self):
        return {
            "type": "GeometryCollection",
            "geometries": [g.__geo_interface__ for g in self],
        }

    @property
    def bounds(self):
        """
        Method to calculate the bounding box of the collection

        Returns
        -------
            tuple (xmin, ymin, xmax, ymax)
        """
        bbox = [geom.bounds for geom in self]
        xmin, ymin = np.min(bbox, axis=0)[0:2]
        xmax, ymax = np.max(bbox, axis=0)[2:]

        return xmin, ymin, xmax, ymax

    def plot(self, ax=None, **kwargs):
        """
        Plotting method for collection

        Parameters
        ----------
        ax : matplotlib.axes object
        kwargs : keyword arguments
            matplotlib keyword arguments

        Returns
        -------
            matplotlib.axes object
        """
        for g in self:
            ax = g.plot(ax=ax, **kwargs)

        xmin, ymin, xmax, ymax = self.bounds
        ax.set_ylim([ymin - 0.005, ymax + 0.005])
        ax.set_xlim([xmin - 0.005, xmax + 0.005])
        return ax


class MultiPolygon(Collection):
    """
    Container for housing and describing multipolygon geometries (e.g. to be
        read or written to shapefiles or other geographic data formats)

    Parameters:
    ----------
    polygons : list
        list of flopy.utils.geometry.Polygon objects
    """

    def __init__(self, polygons=()):
        for p in polygons:
            if not isinstance(p, Polygon):
                raise TypeError("Only Polygon instances are supported")
            super(MultiPolygon, self).__init__(polygons)

    def __repr__(self):
        return "MultiPolygon: {}".format(list(self))

    @property
    def __geo_interface__(self):
        return {
            "type": "MultiPolygon",
            "coordinates": [g.__geo_interface__["coordinates"] for g in self],
        }


class MultiLineString(Collection):
    """
    Container for housing and describing multilinestring geometries (e.g. to be
        read or written to shapefiles or other geographic data formats)

    Parameters:
    ----------
    polygons : list
        list of flopy.utils.geometry.LineString objects
    """

    def __init__(self, linestrings=()):
        for l in linestrings:
            if not isinstance(l, LineString):
                raise TypeError("Only LineString instances are supported")
            super(MultiLineString, self).__init__(linestrings)

    def __repr__(self):
        return "LineString: {}".format(list(self))

    @property
    def __geo_interface__(self):
        return {
            "type": "MultiLineString",
            "coordinates": [g.__geo_interface__["coordinates"] for g in self],
        }


class MultiPoint(Collection):
    """
    Container for housing and describing multipoint geometries (e.g. to be
        read or written to shapefiles or other geographic data formats)

    Parameters:
    ----------
    polygons : list
        list of flopy.utils.geometry.Point objects
    """

    def __init__(self, points=()):
        for p in points:
            if not isinstance(p, Point):
                raise TypeError("Only Point instances are supported")
            super(MultiPoint, self).__init__(points)

    def __repr__(self):
        return "MultiPoint: {}".format(list(self))

    @property
    def __geo_interface__(self):
        return {
            "type": "MultiPoint",
            "coordinates": [g.__geo_interface__["coordinates"] for g in self],
        }


class Polygon(Shape):
    type = "Polygon"
    shapeType = 5  # pyshp

    def __init__(self, exterior, interiors=None):
        """
        Container for housing and describing polygon geometries (e.g. to be
        read or written to shapefiles or other geographic data formats)

        Parameters
        ----------
        exterior : sequence
            Sequence of coordinates describing the outer ring of the polygon.
        interiors : sequence of sequences
            Describes one or more holes within the polygon
        Attributes
        ----------
        exterior : (x, y, z) coordinates of exterior
        interiors : tuple of (x, y, z) coordinates of each interior polygon
        patch : descartes.PolygonPatch representation
        bounds : (xmin, ymin, xmax, ymax)
            Tuple describing bounding box for polygon
        geojson : dict
            Returns a geojson representation of the feature
        pyshp_parts : list of lists
            Returns a list of all parts (each an individual polygon).
            Can be used as input for the shapefile.Writer.poly method
            (pyshp package)
        Methods
        -------
        get_patch
            Returns a descartes PolygonPatch object representation of the
            polygon. Accepts keyword arguments to descartes.PolygonPatch.
            Requires the descartes package (pip install descartes).
        plot
            Plots the feature using descartes (via get_patch) and
            matplotlib.pyplot. Accepts keyword arguments to
            descartes.PolygonPatch. Requires the descartes package
            (pip install descartes).
        Notes
        -----
        Multi-polygons not yet supported.
        z information is only stored if it was entered.
        """
        super(Polygon, self).__init__(
            self.type,
            coordinates=None,
            exterior=exterior,
            interiors=interiors,
        )

    def __eq__(self, other):
        if not isinstance(other, Polygon):
            return False
        if other.exterior != self.exterior:
            return False
        if other.interiors != self.interiors:
            return False
        return True

    @property
    def _exterior_x(self):
        return [x for x, y in self.exterior]

    @property
    def _exterior_y(self):
        return [y for x, y in self.exterior]

    @property
    def bounds(self):
        ymin = np.min(self._exterior_y)
        ymax = np.max(self._exterior_y)
        xmin = np.min(self._exterior_x)
        xmax = np.max(self._exterior_x)
        return xmin, ymin, xmax, ymax

    @property
    def pyshp_parts(self):
        from ..export.shapefile_utils import import_shapefile

        # exterior ring must be clockwise (negative area)
        # interiors rings must be counter-clockwise (positive area)

        shapefile = import_shapefile()

        exterior = list(self.exterior)
        if shapefile.signed_area(exterior) > 0:
            exterior.reverse()

        interiors = []
        for i in self.interiors:
            il = list(i)
            if shapefile.signed_area(il) < 0:
                il.reverse()
            interiors.append(il)

        result = [exterior]
        for i in interiors:
            result.append(i)
        return result

    @property
    def patch(self):
        return self.get_patch()

    def get_patch(self, **kwargs):
        try:
            from descartes import PolygonPatch
        except ImportError:
            print(
                'This feature requires descartes.\nTry "pip install descartes"'
            )
        return PolygonPatch(self.geojson, **kwargs)

    def plot(self, ax=None, **kwargs):
        """
        Plot the feature.
        Parameters
        ----------
        ax : matplotlib.pyplot axes instance
        Accepts keyword arguments to descartes.PolygonPatch. Requires the
        descartes package (pip install descartes).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("This feature requires matplotlib.")

        if ax is None:
            ax = plt.gca()

        try:
            ax.add_patch(self.get_patch(**kwargs))
            xmin, ymin, xmax, ymax = self.bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        except:
            print("could not plot polygon feature")

        return ax


class LineString(Shape):
    type = "LineString"
    shapeType = 3
    has_z = False

    def __init__(self, coordinates):
        """
        Container for housing and describing linestring geometries (e.g. to be
        read or written to shapefiles or other geographic data formats)

        Parameters
        ----------
        coordinates : sequence
            Sequence of coordinates describing a line
        Attributes
        ----------
        coords : list of (x, y, z) coordinates
        x : list of x coordinates
        y : list of y coordinates
        z : list of z coordinates
        bounds : (xmin, ymin, xmax, ymax)
            Tuple describing bounding box for linestring
        geojson : dict
            Returns a geojson representation of the feature
        pyshp_parts : list of lists
            Returns a list of all parts (each an individual linestring).
            Can be used as input for the shapefile.Writer.line method (pyshp package)
        Methods
        -------
        plot
            Plots the feature using matplotlib.pyplot.
            Accepts keyword arguments to pyplot.plot.
        Notes
        -----
        Multi-linestrings not yet supported.
        z information is only stored if it was entered.

        """
        super(LineString, self).__init__(self.type, coordinates)

    def __eq__(self, other):
        if not isinstance(other, LineString):
            return False
        if other.x != self.x:
            return False
        if other.y != self.y:
            return False
        if other.z != self.z:
            return False
        return True

    @property
    def x(self):
        return [c[0] for c in self.coords]

    @property
    def y(self):
        return [c[1] for c in self.coords]

    @property
    def z(self):
        return 0 if not self.has_z else [c[2] for c in self.coords]

    @property
    def bounds(self):
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        return xmin, ymin, xmax, ymax

    @property
    def pyshp_parts(self):
        return [self.coords]

    def plot(self, ax=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("This feature requires matplotlib.")

        if ax is None:
            ax = plt.gca()

        ax.plot(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return ax


class Point(Shape):
    type = "Point"
    shapeType = 1
    has_z = False

    def __init__(self, *coordinates):
        """
        Container for housing and describing point geometries (e.g. to be read
        or written to shapefiles or other geographic data formats)

        Parameters
        ----------
        coordinates : tuple
            x, y or x, y, z

        Attributes
        ----------
        coords : x, y, z coordinates

        x : x coordinate

        y : y coordinate

        z : z coordinate

        bounds : (xmin, ymin, xmax, ymax)
            Tuple describing bounding box

        geojson : dict
            Returns a geojson representation of the feature

        pyshp_parts : list of tuples
            Can be used as input for the shapefile.Writer.line method
            (pyshp package)

        Methods
        -------
        plot
            Plots the feature using matplotlib.pyplot.
            Accepts keyword arguments to pyplot.scatter.

        Notes
        -----
        z information is only stored if it was entered.
        """
        super(Point, self).__init__(self.type, coordinates)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        if other.x != self.x:
            return False
        if other.y != self.y:
            return False
        if other.z != self.z:
            return False
        return True

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return 0 if not self.has_z else self.coords[2]

    @property
    def bounds(self):
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        return xmin, ymin, xmax, ymax

    @property
    def pyshp_parts(self):
        return self.coords

    def plot(self, ax=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("This feature requires matplotlib.")

        if ax is None:
            ax = plt.gca()

        ax.scatter(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin - 1, xmax + 1)  # singular bounds otherwise
        ax.set_ylim(ymin - 1, ymax + 1)

        return ax


def rotate(x, y, xoff, yoff, angrot_radians):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.

    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    xrot = (
        xoff
        + np.cos(angrot_radians) * (x - xoff)
        - np.sin(angrot_radians) * (y - yoff)
    )
    yrot = (
        yoff
        + np.sin(angrot_radians) * (x - xoff)
        + np.cos(angrot_radians) * (y - yoff)
    )

    return xrot, yrot


def transform(
    x,
    y,
    xoff,
    yoff,
    angrot_radians,
    length_multiplier=1.0,
    inverse=False,
):
    """
    Given x and y array-like values calculate the translation about an
    arbitrary origin and then return the rotated coordinates.

    """
    if isinstance(x, list):
        x = np.array(x, dtype=float)
    if isinstance(y, list):
        y = np.array(y, dtype=float)

    if not np.isscalar(x):
        x, y = x.copy(), y.copy()

    if not inverse:
        x *= length_multiplier
        y *= length_multiplier
        x += xoff
        y += yoff
        xrot, yrot = rotate(x, y, xoff, yoff, angrot_radians)

    else:
        xrot, yrot = rotate(x, y, xoff, yoff, -angrot_radians)
        xrot -= xoff
        yrot -= yoff
        xrot /= length_multiplier
        yrot /= length_multiplier

    return xrot, yrot


def shape(pyshp_shpobj):
    """
    Convert a pyshp geometry object to a flopy geometry object.

    Parameters
    ----------
    pyshp_shpobj : shapefile._Shape instance

    Returns
    -------
    shape : flopy.utils.geometry Polygon, Linestring, or Point

    Notes
    -----
    Currently only regular Polygons, LineStrings and Points (pyshp types 5, 3, 1) supported.

    Examples
    --------
    >>> import shapefile as sf
    >>> from flopy.utils.geometry import shape
    >>> sfobj = sf.Reader('shapefile.shp')
    >>> flopy_geom = shape(list(sfobj.iterShapes())[0])

    """
    import warnings

    warnings.warn(
        "Method will be Deprecated, calling GeoSpatialUtil",
        DeprecationWarning,
    )

    from .geospatial_utils import GeoSpatialUtil

    return GeoSpatialUtil(pyshp_shpobj).flopy_geometry


def get_polygon_area(geom):
    """
    Calculate the area of a closed polygon

    Parameters
    ----------
    geom : geospatial representation of polygon
        accepted types:

        vertices np.array([(x, y),....])
        geojson.Polygon
        shapely.Polygon
        shapefile.Shape

    Returns
    -------
    area : float
        area of polygon centroid

    """
    from .geospatial_utils import GeoSpatialUtil

    if isinstance(geom, (list, tuple, np.ndarray)):
        geom = [geom]

    geom = GeoSpatialUtil(geom, shapetype="Polygon")
    verts = np.array(geom.points[0])

    nverts = verts.shape[0]
    a = 0.0
    for iv in range(nverts - 1):
        x = verts[iv, 0]
        y = verts[iv, 1]
        xp1 = verts[iv + 1, 0]
        yp1 = verts[iv + 1, 1]
        a += x * yp1 - xp1 * y
    a = abs(a * 0.5)
    return a


def get_polygon_centroid(geom):
    """
    Calculate the centroid of a closed polygon

    Parameters
    ----------
    geom : geospatial representation of polygon
        accepted types:

        vertices np.array([(x, y),....])
        geojson.Polygon
        shapely.Polygon
        shapefile.Shape

    Returns
    -------
    centroid : tuple
        (x, y) of polygon centroid

    """
    from .geospatial_utils import GeoSpatialUtil

    if isinstance(geom, (list, tuple, np.ndarray)):
        geom = [geom]

    geom = GeoSpatialUtil(geom, shapetype="Polygon")
    verts = np.array(geom.points[0])

    nverts = verts.shape[0]
    cx = 0.0
    cy = 0.0
    for i in range(nverts - 1):
        x = verts[i, 0]
        y = verts[i, 1]
        xp1 = verts[i + 1, 0]
        yp1 = verts[i + 1, 1]
        cx += (x + xp1) * (x * yp1 - xp1 * y)
        cy += (y + yp1) * (x * yp1 - xp1 * y)
    a = get_polygon_area(verts)
    cx = cx * 1.0 / 6.0 / a
    cy = cy * 1.0 / 6.0 / a
    return cx, cy


def is_clockwise(*geom):
    """
    Determine if a ring is defined clockwise

    Parameters
    ----------
    *geom : geospatial representation of polygon
        accepted types:

        vertices [(x, y),....]
        geojson.Polygon
        shapely.Polygon
        shapefile.Shape
        x and y vertices: [x1, x2, x3], [y1, y2, y3]

    Returns
    -------
    clockwise : bool
        True when the ring is defined clockwise, False otherwise

    """
    from .geospatial_utils import GeoSpatialUtil

    if len(geom) == 2:
        x, y = geom
    else:
        geom = GeoSpatialUtil(geom, shapetype="Polygon")
        x, y = np.array(geom.points[0]).T

    if not (x[0] == x[-1]) and (y[0] == y[-1]):
        # close the ring if needed
        x = np.append(x, x[-1])
        y = np.append(y, y[-1])
    return np.sum((np.diff(x)) * (y[1:] + y[:-1])) > 0
