"""
Container objects for working with geometric information
"""
class Polygon:

    type = 'Polygon'
    shapetype = 5 # pyshp

    def __init__(self, exterior, interiors=None):

        self.exterior = tuple(map(tuple, exterior))
        self.interiors = tuple() if interiors is None else (map(tuple, i) for i in interiors)
        self.patch = self.get_patch()

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
        plt.show()