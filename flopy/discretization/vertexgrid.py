import os
import copy
import numpy as np

try:
    from matplotlib.path import Path
except ImportError:
    Path = None

from .grid import Grid, CachedData
from ..utils.geometry import is_clockwise


class VertexGrid(Grid):
    """
    class for a vertex model grid

    Parameters
    ----------
    vertices
        list of vertices that make up the grid
    cell2d
        list of cells and their vertices

    Properties
    ----------
    vertices
        returns list of vertices that make up the grid
    cell2d
        returns list of cells and their vertices

    Methods
    ----------
    get_cell_vertices(cellid)
        returns vertices for a single cell at cellid.

    """

    def __init__(
        self,
        vertices=None,
        cell2d=None,
        top=None,
        botm=None,
        idomain=None,
        lenuni=None,
        epsg=None,
        proj4=None,
        prj=None,
        xoff=0.0,
        yoff=0.0,
        angrot=0.0,
        nlay=None,
        ncpl=None,
        cell1d=None,
    ):
        super().__init__(
            "vertex",
            top,
            botm,
            idomain,
            lenuni,
            epsg,
            proj4,
            prj,
            xoff,
            yoff,
            angrot,
        )
        self._vertices = vertices
        self._cell1d = cell1d
        self._cell2d = cell2d
        self._top = top
        self._botm = botm
        self._idomain = idomain
        if botm is None:
            self._nlay = nlay
            self._ncpl = ncpl
        else:
            self._nlay = None
            self._ncpl = None

    @property
    def is_valid(self):
        if self._vertices is not None and (
            self._cell2d is not None or self._cell1d is not None
        ):
            return True
        return False

    @property
    def is_complete(self):
        if (
            self._vertices is not None
            and (self._cell2d is not None or self._cell1d is not None)
            and super().is_complete
        ):
            return True
        return False

    @property
    def nlay(self):
        if self._cell1d is not None:
            return 1
        elif self._botm is not None:
            return len(self._botm)
        else:
            return self._nlay

    @property
    def ncpl(self):
        if self._cell1d is not None:
            return len(self._cell1d)
        if self._botm is not None:
            return len(self._botm[0])
        if self._cell2d is not None and self._nlay is None:
            return len(self._cell2d)
        else:
            return self._ncpl

    @property
    def nnodes(self):
        return self.nlay * self.ncpl

    @property
    def nvert(self):
        return len(self._vertices)

    @property
    def iverts(self):
        return [[t[0]] + t[4:] for t in self._cell2d]

    @property
    def verts(self):
        return np.array([t[1:] for t in self._vertices], dtype=float)

    @property
    def shape(self):
        return self.nlay, self.ncpl

    @property
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        # new_botm = np.expand_dims(self._botm, 0)
        return np.concatenate((new_top, self._botm), axis=0)

    @property
    def extent(self):
        self._copy_cache = False
        xvertices = np.hstack(self.xvertices)
        yvertices = np.hstack(self.yvertices)
        self._copy_cache = True
        return (
            np.min(xvertices),
            np.max(xvertices),
            np.min(yvertices),
            np.max(yvertices),
        )

    @property
    def grid_lines(self):
        """
        Creates a series of grid line vertices for drawing
        a model grid line collection

        Returns:
            list: grid line vertices
        """
        self._copy_cache = False
        xgrid = self.xvertices
        ygrid = self.yvertices

        lines = []
        for ncell, verts in enumerate(xgrid):
            for ix, vert in enumerate(verts):
                lines.append(
                    [
                        (xgrid[ncell][ix - 1], ygrid[ncell][ix - 1]),
                        (xgrid[ncell][ix], ygrid[ncell][ix]),
                    ]
                )
        self._copy_cache = True
        return lines

    @property
    def xyzcellcenters(self):
        """
        Method to get cell centers and set to grid
        """
        cache_index = "cellcenters"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            self._build_grid_geometry_info()
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyzvertices(self):
        """
        Method to get all grid vertices in a layer, arranged per cell

        Returns:
            list of size sum(nvertices per cell)
        """
        cache_index = "xyzgrid"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            self._build_grid_geometry_info()
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def map_polygons(self):
        """
        Get a list of matplotlib Polygon patches for plotting

        Returns
        -------
            list of Polygon objects
        """
        try:
            import matplotlib.path as mpath
        except ImportError:
            raise ImportError("matplotlib required to use this method")
        cache_index = "xyzgrid"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            self.xyzvertices
            self._polygons = None
        if self._polygons is None:
            self._polygons = [
                mpath.Path(self.get_cell_vertices(nn))
                for nn in range(self.ncpl)
            ]

        return copy.copy(self._polygons)

    def intersect(self, x, y, local=False, forgive=False):
        """
        Get the CELL2D number of a point with coordinates x and y

        When the point is on the edge of two cells, the cell with the lowest
        CELL2D number is returned.

        Parameters
        ----------
        x : float
            The x-coordinate of the requested point
        y : float
            The y-coordinate of the requested point
        local: bool (optional)
            If True, x and y are in local coordinates (defaults to False)
        forgive: bool (optional)
            Forgive x,y arguments that fall outside the model grid and
            return NaNs instead (defaults to False - will throw exception)

        Returns
        -------
        icell2d : int
            The CELL2D number

        """
        if Path is None:
            s = (
                "Could not import matplotlib.  Must install matplotlib "
                "in order to use VertexGrid.intersect() method"
            )
            raise ImportError(s)

        if local:
            # transform x and y to real-world coordinates
            x, y = super().get_coords(x, y)
        xv, yv, zv = self.xyzvertices
        for icell2d in range(self.ncpl):
            xa = np.array(xv[icell2d])
            ya = np.array(yv[icell2d])
            # x and y at least have to be within the bounding box of the cell
            if (
                np.any(x <= xa)
                and np.any(x >= xa)
                and np.any(y <= ya)
                and np.any(y >= ya)
            ):
                path = Path(np.stack((xa, ya)).transpose())
                # use a small radius, so that the edge of the cell is included
                if is_clockwise(xa, ya):
                    radius = -1e-9
                else:
                    radius = 1e-9
                if path.contains_point((x, y), radius=radius):
                    return icell2d
        if forgive:
            icell2d = np.nan
            return icell2d
        raise Exception("x, y point given is outside of the model area")

    def get_cell_vertices(self, cellid):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param cellid: (int) cellid number
        Returns
        ------- list of x,y cell vertices
        """
        while cellid >= self.ncpl:
            if cellid > self.nnodes:
                err = "cellid {} out of index for size {}".format(
                    cellid, self.nnodes
                )
                raise IndexError(err)

            cellid -= self.ncpl

        self._copy_cache = False
        cell_verts = list(zip(self.xvertices[cellid], self.yvertices[cellid]))
        self._copy_cache = True
        return cell_verts

    def plot(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
        kwargs : ax, colors.  The remaining kwargs are passed into the
            the LineCollection constructor.

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        from flopy.plot import PlotMapView

        mm = PlotMapView(modelgrid=self)
        return mm.plot_grid(**kwargs)

    def _build_grid_geometry_info(self):
        cache_index_cc = "cellcenters"
        cache_index_vert = "xyzgrid"

        xcenters = []
        ycenters = []
        xvertices = []
        yvertices = []

        if self._cell1d is not None:
            zcenters = []
            zvertices = []
            vertexdict = {v[0]: [v[1], v[2], v[3]] for v in self._vertices}
            for cell1d in self._cell1d:
                cell1d = tuple(cell1d)
                xcenters.append(cell1d[1])
                ycenters.append(cell1d[2])
                zcenters.append(cell1d[3])

                vert_number = []
                for i in cell1d[3:]:
                    if i is not None:
                        vert_number.append(int(i))

                xcellvert = []
                ycellvert = []
                zcellvert = []
                for ix in vert_number:
                    xcellvert.append(vertexdict[ix][0])
                    ycellvert.append(vertexdict[ix][1])
                    zcellvert.append(vertexdict[ix][2])
                xvertices.append(xcellvert)
                yvertices.append(ycellvert)
                zvertices.append(zcellvert)

        else:
            vertexdict = {v[0]: [v[1], v[2]] for v in self._vertices}
            # build xy vertex and cell center info
            for cell2d in self._cell2d:
                cell2d = tuple(cell2d)
                xcenters.append(cell2d[1])
                ycenters.append(cell2d[2])

                vert_number = []
                for i in cell2d[4:]:
                    if i is not None:
                        vert_number.append(int(i))

                xcellvert = []
                ycellvert = []
                for ix in vert_number:
                    xcellvert.append(vertexdict[ix][0])
                    ycellvert.append(vertexdict[ix][1])
                xvertices.append(xcellvert)
                yvertices.append(ycellvert)

            # build z cell centers
            zvertices, zcenters = self._zcoords()

        if self._has_ref_coordinates:
            # transform x and y
            xcenters, ycenters = self.get_coords(xcenters, ycenters)
            xvertxform = []
            yvertxform = []
            # vertices are a list within a list
            for xcellvertices, ycellvertices in zip(xvertices, yvertices):
                xcellvertices, ycellvertices = self.get_coords(
                    xcellvertices, ycellvertices
                )
                xvertxform.append(xcellvertices)
                yvertxform.append(ycellvertices)
            xvertices = xvertxform
            yvertices = yvertxform

        self._cache_dict[cache_index_cc] = CachedData(
            [np.array(xcenters), np.array(ycenters), np.array(zcenters)]
        )
        self._cache_dict[cache_index_vert] = CachedData(
            [xvertices, yvertices, zvertices]
        )

    def get_xvertices_for_layer(self, layer):
        xgrid = np.array(self.xvertices, dtype=object)
        return xgrid

    def get_yvertices_for_layer(self, layer):
        ygrid = np.array(self.yvertices, dtype=object)
        return ygrid

    def get_xcellcenters_for_layer(self, layer):
        xcenters = np.array(self.xcellcenters)
        return xcenters

    def get_ycellcenters_for_layer(self, layer):
        ycenters = np.array(self.ycellcenters)
        return ycenters

    def get_number_plottable_layers(self, a):
        """
        Calculate and return the number of 2d plottable arrays that can be
        obtained from the array passed (a)

        Parameters
        ----------
        a : ndarray
            array to check for plottable layers

        Returns
        -------
        nplottable : int
            number of plottable layers

        """
        nplottable = 0
        required_shape = self.get_plottable_layer_shape()
        if a.shape == required_shape:
            nplottable = 1
        else:
            nplottable = a.size / self.ncpl
            nplottable = int(nplottable)
        return nplottable

    def get_plottable_layer_array(self, a, layer):
        # ensure plotarray is 1d with length ncpl
        required_shape = self.get_plottable_layer_shape()
        if a.ndim == 3:
            if a.shape[0] == 1:
                a = np.squeeze(a, axis=0)
                plotarray = a[layer, :]
            elif a.shape[1] == 1:
                a = np.squeeze(a, axis=1)
                plotarray = a[layer, :]
            else:
                raise Exception(
                    "Array has 3 dimensions so one of them must be of size 1 "
                    "for a VertexGrid."
                )
        elif a.ndim == 2:
            plotarray = a[layer, :]
        elif a.ndim == 1:
            plotarray = a
            if plotarray.shape[0] == self.nnodes:
                plotarray = plotarray.reshape(self.nlay, self.ncpl)
                plotarray = plotarray[layer, :]
        else:
            raise Exception("Array to plot must be of dimension 1 or 2")
        msg = "{} /= {}".format(plotarray.shape[0], required_shape)
        assert plotarray.shape == required_shape, msg
        return plotarray

    # initialize grid from a grb file
    @classmethod
    def from_binary_grid_file(cls, file_path, verbose=False):
        """
        Instantiate a VertexGrid model grid from a MODFLOW 6 binary
        grid (*.grb) file.

        Parameters
        ----------
        file_path : str
            file path for the MODFLOW 6 binary grid file
        verbose : bool
            Write information to standard output.  Default is False.

        Returns
        -------
        return : VertexGrid

        """
        from ..mf6.utils.binarygrid_util import MfGrdFile

        grb_obj = MfGrdFile(file_path, verbose=verbose)
        if grb_obj.grid_type != "DISV":
            err_msg = (
                "Binary grid file ({}) ".format(os.path.basename(file_path))
                + "is not a vertex (DISV) grid."
            )
            raise ValueError(err_msg)

        idomain = grb_obj.idomain
        xorigin = grb_obj.xorigin
        yorigin = grb_obj.yorigin
        angrot = grb_obj.angrot

        nlay, ncpl = grb_obj.nlay, grb_obj.ncpl
        top = np.ravel(grb_obj.top)
        botm = grb_obj.bot
        botm.shape = (nlay, ncpl)
        vertices, cell2d = grb_obj.cell2d

        return cls(
            vertices,
            cell2d,
            top,
            botm,
            idomain,
            xoff=xorigin,
            yoff=yorigin,
            angrot=angrot,
        )
