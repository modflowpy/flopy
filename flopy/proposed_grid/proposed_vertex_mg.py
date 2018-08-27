import numpy as np
from flopy.proposed_grid.proposed_sr import SpatialReference
from flopy.mf6.data.mfdataarray import MFArray
from flopy.mf6.data.mfdatalist import MFList



class VertexModelGrid(object):
    """

    """
    def __init__(self, verticies, cell2d, top=None, botm=None,
                 idomain=None, sr=None, xoffset=None, yoffset=None,
                 rotation=0., origin_location="ll",
                 length_multiplier=1.):

        self.__verticies = verticies
        self.__cell2d = cell2d
        self.__top = top
        self.__botm = botm
        self.__idomain = idomain
        self.__sr = sr

        if self.__sr is None:
            self.__sr = SpatialReference(xoffset=xoffset,
                                         yoffset=yoffset,
                                         rotation=rotation,
                                         origin_location=origin_location,
                                         length_multiplier=length_multiplier,
                                         model_grid=self)

        self.__xgrid = None
        self.__ygrid = None
        self.__zgrid = None
        self.__xcenters = None
        self.__ycenters = None
        self.__zcenters = None
        self.__xvertex_dict = None
        self.__yvertex_dict = None
        self.__xcenters_dict = None
        self.__ycenters_dict = None

    def __clear_cache(self):
        self.__xgrid = None
        self.__ygrid = None
        self.__zgrid = None
        self.__xcenters = None
        self.__ycenters = None
        self.__zcenters = None
        self.__xvertex_dict = None
        self.__yvertex_dict = None
        self.__xcenters_dict = None
        self.__ycenters_dict = None
        if self.__sr is not None:
            self.__sr.model_grid = self

    def __setattr__(self, key, value):
        clear_cache = True
        if key == "verticies":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__verticies", np.array(value))

            elif isinstance(value, MFList):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__verticies", value.array)


        elif key == "cell2d":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__cell2d", np.array(value))

            elif isinstance(value, MFList):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__cell2d", value.array)

        elif key == "top":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__top", np.array(value))

            elif isinstance(value, MFArray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__top", value.array)

            elif value is None:
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__top", value)

        elif key == "botm":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__botm", np.array(value))

            elif isinstance(value, MFArray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__botm", value.array)

            elif value is None:
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__botm", value)

        elif key == "idomain":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__idomain", np.array(value))

            elif isinstance(value, MFArray):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__idomain", value.array)

            elif value is None:
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__idomain", value)

        elif key == "sr":
            clear_cache = False
            if isinstance(value, SpatialReference):
                super(VertexModelGrid, self). \
                    __setattr__("_VertexModelGrid__sr", value)

        else:
            clear_cache = False
            if isinstance(value, MFArray) or \
                    isinstance(value, MFList):
                super(VertexModelGrid, self). \
                    __setattr__(key, value.array)

            else:
                super(VertexModelGrid, self).__setattr__(key, value)

        if clear_cache:
            self.__clear_cache()

    @property
    def grid_type(self):
        return "vertex"

    @property
    def extent(self):
        return (min(np.ravel(self.xgrid)),
                max(np.ravel(self.xgrid)),
                min(np.ravel(self.ygrid)),
                max(np.ravel(self.ygrid)))

    @property
    def nlay(self):
        if self.__botm is not None:
            return len(self.__botm + 1)

    @property
    def ncpl(self):
        return len(self.__cell2d)

    @property
    def top(self):
        return self.__top

    @property
    def botm(self):
        return self.__botm

    @property
    def idomain(self):
        return self.__idomain

    @property
    def sr(self):
        if self.__sr._SpatialReference__model_grid is None:
            self.__sr.model_grid = self
        return self.__sr

    @property
    def xcenters(self):
        if self.__xcenters is None:
            self.__xcenters = self.__get_grid_centers(dimension="x")
        return self.__xcenters

    @property
    def ycenters(self):
        if self.__ycenters is None:
            self.__ycenters = self.__get_grid_centers(dimension="y")
        return self.__ycenters

    @property
    def zcenters(self):
        if self.__zcenters is None:
            self.__zcenters = self.__get_grid_centers(dimension="z")
        return self.__zcenters

    @property
    def xgrid(self):
        if self.__xgrid is None:
            self.__xgrid = self.__get_grid(dimension="x")
        return self.__xgrid

    @property
    def ygrid(self):
        if self.__ygrid is None:
            self.__ygrid = self.__get_grid(dimension="y")
        return self.__ygrid

    @property
    def zgrid(self):
        if self.__zgrid is None:
            self.__zgrid = self.__get_grid(dimension="z")
        return self.__zgrid

    @property
    def xedges(self):
        err_msg = "Grid edge array is not calculated " \
                  "for vertex model grid, use vertices"
        raise ValueError(err_msg)

    @property
    def yedges(self):
        err_msg = "Grid edge array is not calculated " \
                  "for vertex model grid, use vertices"
        raise ValueError(err_msg)

    @property
    def grid_lines(self):
        """
        Creates a series of grid line vertices for drawing
        a model grid line collection

        Returns:
            list: grid line vertices
        """
        xgrid = self.xgrid
        ygrid = self.ygrid

        lines = []
        for ncell, verts in enumerate(xgrid):
            for ix, vert in enumerate(verts):
                lines.append([(xgrid[ncell][ix - 1], ygrid[ncell][ix - 1]),
                              (xgrid[ncell][ix], ygrid[ncell][ix])])

        return lines

    def plot_grid_lines(self, **kwargs):
        """
        Get a LineCollection of the model grid in
        model coordinates

        Parameters
            **kwargs: matplotlib.pyplot keyword arguments

        Returns
            matplotlib.collections.LineCollection
        """
        from flopy.plot.plotbase import PlotMapView

        map = PlotMapView(modelgrid=self)
        lc = map.plot_grid(**kwargs)
        return lc

    def plot_array(self, a, ax=None, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        from flopy.plot.plotutil import PlotUtilities

        ax = PlotUtilities._plot_array_helper(a, sr=self, axes=ax, **kwargs)
        return ax

    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        from flopy.plot import PlotMapView

        kwargs['ax'] = ax
        map = PlotMapView(sr=self)
        contour_set = map.contour_array(a=a, **kwargs)

        return contour_set

    def __set_vertex_dictionaries(self):
        """
        Set method to create vertex dictionary by node number
        """
        vertexdict = {v[0]: [v[1], v[2]]
                      for v in self.__verticies}
        xcenters = {}
        ycenters = {}
        xverticies = {}
        yverticies = {}

        for cell2d in self.__cell2d:
            cell2d = tuple(cell2d)
            xcenters[int(cell2d[0])] = cell2d[1]
            ycenters[int(cell2d[0])] = cell2d[2]

            vert_number = [int(i) for i in cell2d[4:]]
            xverticies[int(cell2d[0])] = (vertexdict[ix][0]
                                          for ix in vert_number)
            yverticies[int(cell2d[0])] = (vertexdict[ix][1]
                                          for ix in vert_number)

        self.__xcenters_dict = xcenters
        self.__ycenters_dict = ycenters
        self.__xvertex_dict = xverticies
        self.__yvertex_dict = yverticies

    def __get_grid(self, dimension="x"):
        """
        Internal method to get model grid verticies for x

        Returns:
            list of dimension ncpl by nvertices
        """
        grid = []
        if dimension.lower() == "x":
            if self.__xvertex_dict is None:
                self.__set_vertex_dictionaries()

            for key, verts in sorted(self.__xvertex_dict.items()):
                grid.append(tuple([i for i in verts]))

        elif dimension.lower() == "y":
            if self.__yvertex_dict is None:
                self.__set_vertex_dictionaries()

            for key, verts in sorted(self.__yvertex_dict.items()):
                grid.append(tuple([i for i in verts]))

        elif dimension.lower() == "z":
            if self.top is None and self.botm is None:
                err_msg = "Model top and botm array must be specified" \
                          "for zcenters to be calculated"
                raise AssertionError(err_msg)
            else:
                grid = np.concatenate((self.top,
                                       self.botm), axis=0)

        else:
            err_msg = "get_grid method not implemented" \
                      " for {} dimension".format(dimension)
            raise NotImplementedError(err_msg)

        return np.array(grid)

    def __get_grid_centers(self, dimension="x"):
        """
        Internal method to get cell centers and set to grid
        """
        grid = []
        if dimension.lower() == "x":
            if self.__xcenters_dict is None:
                self.__set_vertex_dictionaries()

            for key, vert in sorted(self.__xcenters_dict.items()):
                grid.append(vert)

        elif dimension.lower() == "y":
            if self.__ycenters_dict is None:
                self.__set_vertex_dictionaries()

            for key, vert in sorted(self.__ycenters_dict.items()):
                grid.append(vert)

        elif dimension.lower() == "z":
            if self.top is None and self.botm is None:
                err_msg = "Model top and botm array must be specified" \
                          "for zcenters to be calculated"
                raise AssertionError(err_msg)
            else:
                elev = np.concatenate((self.top,
                                       self.botm), axis=0)

                for ix in range(1, len(elev)):
                    grid.append((elev[ix - 1] + elev[ix]) / 2.)

        else:
            err_msg = "get_grid method not implemented" \
                      " for {} dimension".format(dimension)
            raise NotImplementedError(err_msg)

        return np.array(grid)


if __name__ == "__main__":
    import os
    import flopy as fp

    ws = "../../examples/data/mf6/test003_gwfs_disv"
    name = "mfsim.nam"

    sim = fp.mf6.modflow.MFSimulation.load(sim_name=name, sim_ws=ws)

    print(sim.model_names)
    ml = sim.get_model("gwf_1")

    dis = ml.dis

    t = VertexModelGrid(dis.vertices, dis.cell2d,
                        top=dis.top, botm=dis.botm,
                        idomain=dis.idomain, xoffset=10.,
                        yoffset=0, rotation=-25)

    print('break')
    # todo: build out model grid methods!
    x = t.xgrid
    y = t.ygrid
    z = t.zgrid
    xc = t.xcenters
    yc = t.ycenters
    zc = t.zcenters
    lc = t.grid_lines
    e = t.extent

    sr_x = t.sr.xgrid
    sr_y = t.sr.ygrid
    sr_xc = t.sr.xcenters
    sr_yc = t.sr.ycenters
    sr_lc = t.sr.grid_lines
    sr_e = t.sr.extent

    print('break')