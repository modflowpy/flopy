"""
Container objects for working with geometric information
"""
import numpy as np


class Polygon:
    type = 'Polygon'
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
        self.exterior = tuple(map(tuple, exterior))
        self.interiors = tuple() if interiors is None else (map(tuple, i) for i
                                                            in interiors)

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
    def geojson(self):
        return {'coordinates': tuple(
            [self.exterior] + [i for i in self.interiors]),
                'type': self.type}

    @property
    def pyshp_parts(self):
        from ..export.shapefile_utils import (import_shapefile,
                                              shapefile_version)

        # exterior ring must be clockwise (negative area)
        # interiors rings must be counter-clockwise (positive area)

        shapefile = import_shapefile()
        sfv = shapefile_version(shapefile)

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
                'This feature requires descartes.\nTry "pip install descartes"')
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
            print('This feature requires matplotlib.')
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        try:
            ax.add_patch(self.get_patch(**kwargs))
            xmin, ymin, xmax, ymax = self.bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            plt.show()
        except:
            print('could not plot polygon feature')


class LineString:
    type = 'LineString'
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
        self.coords = list(map(tuple, coordinates))
        if len(self.coords[0]) == 3:
            self.has_z = True

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
    def geojson(self):
        return {'coordinates': tuple(self.coords),
                'type': self.type}

    @property
    def pyshp_parts(self):
        return [self.coords]

    def plot(self, ax=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('This feature requires matplotlib.')
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        plt.plot(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # plt.show()


class Point:
    type = 'Point'
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
        while len(coordinates) == 1:
            coordinates = coordinates[0]

        self.coords = coordinates
        if len(coordinates) == 3:
            self.has_z = True

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
    def geojson(self):
        return {'coordinates': tuple(self.coords),
                'type': self.type}

    @property
    def pyshp_parts(self):
        return self.coords

    def plot(self, ax=None, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('This feature requires matplotlib.')
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        plt.scatter(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin - 1, xmax + 1)  # singular bounds otherwise
        ax.set_ylim(ymin - 1, ymax + 1)


def rotate(x, y, xoff, yoff, angrot_radians):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.

    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    xrot = xoff + np.cos(angrot_radians) * \
           (x - xoff) - np.sin(angrot_radians) * \
           (y - yoff)
    yrot = yoff + np.sin(angrot_radians) * \
           (x - xoff) + np.cos(angrot_radians) * \
           (y - yoff)

    return xrot, yrot


def transform(x, y, xoff, yoff, angrot_radians,
              length_multiplier=1., inverse=False):
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
    types = {5: Polygon,
             3: LineString,
             1: Point}
    flopy_geometype = types[pyshp_shpobj.shapeType]
    return flopy_geometype(pyshp_shpobj.points)


def get_polygon_area(verts):
    """
    Calculate the area of a closed polygon

    Parameters
    ----------
    verts : numpy.ndarray
        polygon vertices

    Returns
    -------
    area : float
        area of polygon centroid

    """
    nverts = verts.shape[0]
    a = 0.
    for iv in range(nverts - 1):
        x = verts[iv, 0]
        y = verts[iv, 1]
        xp1 = verts[iv + 1, 0]
        yp1 = verts[iv + 1, 1]
        a += (x * yp1 - xp1 * y)
    a = abs(a * 0.5)
    return a


def get_polygon_centroid(verts):
    """
    Calculate the centroid of a closed polygon

    Parameters
    ----------
    verts : numpy.ndarray
        polygon vertices

    Returns
    -------
    centroid : tuple
        (x, y) of polygon centroid

    """
    nverts = verts.shape[0]
    cx = 0.
    cy = 0.
    for i in range(nverts - 1):
        x = verts[i, 0]
        y = verts[i, 1]
        xp1 = verts[i + 1, 0]
        yp1 = verts[i + 1, 1]
        cx += (x + xp1) * (x * yp1 - xp1 * y)
        cy += (y + yp1) * (x * yp1 - xp1 * y)
    a = get_polygon_area(verts)
    cx = cx * 1. / 6. / a
    cy = cy * 1. / 6. / a
    return cx, cy


def is_clockwise(x, y):
    """
    Determine if a ring is defined clockwise

    Parameters
    ----------
    x : numpy ndarray
        The x-coordinates of the ring
    y : numpy ndarray
        The y-coordinate of the ring

    Returns
    -------
    clockwise : bool
        True when the ring is defined clockwise, False otherwise

    """
    if not (x[0] == x[-1]) and (y[0] == y[-1]):
        # close the ring if needed
        x = np.append(x, x[-1])
        y = np.append(y, y[-1])
    return np.sum((np.diff(x)) * (y[1:] + y[:-1])) > 0
