"""
Container objects for working with geometric information
"""
import numpy as np

class Polygon:

    type = 'Polygon'
    shapetype = 5 # pyshp

    def __init__(self, exterior, interiors=None):

        self.exterior = tuple(map(tuple, exterior))
        self.interiors = tuple() if interiors is None else (map(tuple, i) for i in interiors)
        self.patch = self.get_patch()

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
        return {'coordinates': tuple([self.exterior] + [i for i in self.interiors]),
                'type': self.type}

    @property
    def pyshp_parts(self):
        return [list(self.exterior) + [list(i) for i in self.interiors]]

    def get_patch(self, **kwargs):
        try:
            from descartes import PolygonPatch
        except ImportError:
            print('This feature requires descartes.\nTry "pip install descartes"')
        return PolygonPatch(self.geojson, **kwargs)

    def plot(self, **kwargs):
        """Convenience wrapper around the descartes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('This feature requires matplotlib.')
        fig, ax = plt.subplots()
        ax.add_patch(self.get_patch(**kwargs))
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()

class LineString:

    type = 'LineString'
    shapetype = 3

    def __init__(self, coordinates):

        self.coords = list(map(tuple, coordinates))

    @property
    def x(self):
        return [x for x, y in self.coords]

    @property
    def y(self):
        return [y for x, y in self.coords]

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

    def plot(self, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('This feature requires matplotlib.')
        fig, ax = plt.subplots()
        plt.plot(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()

class Point:

    type = 'Point'
    shapetype = 1

    def __init__(self, *coordinates):

        self.coords = coordinates
        if len(coordinates) == 2:
            self.has_z = True

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

    def plot(self, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('This feature requires matplotlib.')
        fig, ax = plt.subplots()
        plt.scatter(self.x, self.y, **kwargs)
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()