import os
import copy
import numpy as np
from .grid import Grid, CachedData


class UnstructuredGrid(Grid):
    """
    Class for an unstructured model grid

    Parameters
    ----------
    vertices : list
        list of vertices that make up the grid.  Each vertex consists of three
        entries [iv, xv, yv] which are the vertex number, which should be
        zero-based, and the x and y vertex coordinates.
    iverts : list
        list of vertex numbers that comprise each cell.  This list must be of
        size nodes, if the grid_varies_by_nodes argument is true, or it must
        be of size ncpl[0] if the same 2d spatial grid is used for each layer.
    xcenters : list or ndarray
        list of x center coordinates for all cells in the grid if the grid
        varies by layer or for all cells in a layer if the same grid is used
        for all layers
    ycenters : list or ndarray
        list of y center coordinates for all cells in the grid if the grid
        varies by layer or for all cells in a layer if the same grid is used
        for all layers
    ncpl : ndarray
        one dimensional array of size nlay with the number of cells in each
        layer.  This can also be passed in as a tuple or list as long as it
        can be set using ncpl = np.array(ncpl, dtype=int).  The sum of ncpl
        must be equal to the number of cells in the grid.  ncpl is optional
        and if it is not passed in, then it is is set using
        ncpl = np.array([len(iverts)], dtype=int), which means that all
        cells in the grid are contained in a single plottable layer.
        If the model grid defined in verts and iverts applies for all model
        layers, then the length of iverts can be equal to ncpl[0] and there
        is no need to repeat all of the vertex information for cells in layers
        beneath the top layer.
    top : list or ndarray
        top elevations for all cells in the grid.
    botm : list or ndarray
        bottom elevations for all cells in the grid.

    Properties
    ----------
    vertices
        returns list of vertices that make up the grid
    cell2d
        returns list of cells and their vertices

    Methods
    -------
    get_cell_vertices(cellid)
        returns vertices for a single cell at cellid.

    Notes
    -----
    This class handles spatial representation of unstructured grids.  It is
    based on the concept of being able to support multiple model layers that
    may have a different number of cells in each layer.  The array ncpl is of
    size nlay and and its sum must equal nodes.  If the length of iverts is
    equal to ncpl[0] and the number of cells per layer is the same for each
    layer, then it is assumed that the grid does not vary by layer.  In this
    case, the xcenters and ycenters arrays must also be of size ncpl[0].
    This makes it possible to efficiently store spatial grid information
    for multiple layers.

    If the spatial grid is different for each model layer, then the
    grid_varies_by_layer flag will automatically be set to false, and iverts
    must be of size nodes. The arrays for xcenters and ycenters must also
    be of size nodes.

    """

    def __init__(
        self,
        vertices=None,
        iverts=None,
        xcenters=None,
        ycenters=None,
        top=None,
        botm=None,
        idomain=None,
        lenuni=None,
        ncpl=None,
        epsg=None,
        proj4=None,
        prj=None,
        xoff=0.0,
        yoff=0.0,
        angrot=0.0,
    ):
        super().__init__(
            "unstructured",
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

        # if any of these are None, then the grid is not valid
        self._vertices = vertices
        self._iverts = iverts
        self._xc = xcenters
        self._yc = ycenters

        # if either of these are None, then the grid is not complete
        self._top = top
        self._botm = botm

        self._ncpl = None
        if ncpl is not None:
            # ensure ncpl is a 1d integer array
            self.set_ncpl(ncpl)
        else:
            # ncpl is not specified, but if the grid is valid, then it is
            # assumed to be of size len(iverts)
            if self.is_valid:
                self.set_ncpl(len(iverts))

        if iverts is not None:
            if self.grid_varies_by_layer:
                msg = "Length of iverts must equal grid nodes ({} {})".format(
                    len(iverts), self.nnodes
                )
                assert len(iverts) == self.nnodes, msg
            else:
                msg = "Length of iverts must equal ncpl ({} {})".format(
                    len(iverts), self.ncpl
                )
                assert np.all([cpl == len(iverts) for cpl in self.ncpl]), msg

        return

    def set_ncpl(self, ncpl):
        if isinstance(ncpl, int):
            ncpl = np.array([ncpl], dtype=int)
        if isinstance(ncpl, (list, tuple, np.ndarray)):
            ncpl = np.array(ncpl, dtype=int)
        else:
            raise TypeError("ncpl must be a list, tuple or ndarray")
        assert ncpl.ndim == 1, "ncpl must be 1d"
        self._ncpl = ncpl
        self._require_cache_updates()
        return

    @property
    def is_valid(self):
        iv = True
        if self._iverts is None:
            iv = False
        if self._vertices is None:
            iv = False
        if self._xc is None:
            iv = False
        if self._yc is None:
            iv = False
        return iv

    @property
    def is_complete(self):
        if self.is_valid is not None and super().is_complete:
            return True
        return False

    @property
    def nlay(self):
        if self.ncpl is None:
            return None
        else:
            return self.ncpl.shape[0]

    @property
    def grid_varies_by_layer(self):
        gvbl = False
        if self.is_valid:
            if self.ncpl[0] == len(self._iverts):
                gvbl = False
            else:
                gvbl = True
        return gvbl

    @property
    def nnodes(self):
        if self.ncpl is None:
            return None
        else:
            return self.ncpl.sum()

    @property
    def nvert(self):
        return len(self._vertices)

    @property
    def iverts(self):
        return self._iverts

    @property
    def verts(self):
        if self._vertices is None:
            return self._vertices
        else:
            return np.array([t[1:] for t in self._vertices], dtype=float)

    @property
    def ia(self):
        if self._ia is None:
            self._set_unstructured_iaja()
        return self._ia

    @property
    def ja(self):
        if self._ja is None:
            self._set_unstructured_iaja()
        return self._ja

    @property
    def ncpl(self):
        return self._ncpl

    @property
    def shape(self):
        return (self.nnodes,)

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
        a model grid line collection.  If the grid varies by layer, then
        return a dictionary with keys equal to layers and values equal to
        grid lines.  Otherwise, just return the grid lines

        Returns:
            dict: grid lines or dictionary of lines by layer

        """
        self._copy_cache = False
        xgrid = self.xvertices
        ygrid = self.yvertices

        grdlines = None
        if self.grid_varies_by_layer:
            grdlines = {}
            icell = 0
            for ilay, numcells in enumerate(self.ncpl):
                lines = []
                for _ in range(numcells):
                    verts = xgrid[icell]
                    for ix in range(len(verts)):
                        lines.append(
                            [
                                (xgrid[icell][ix - 1], ygrid[icell][ix - 1]),
                                (xgrid[icell][ix], ygrid[icell][ix]),
                            ]
                        )
                    icell += 1
                grdlines[ilay] = lines
        else:
            grdlines = []
            for icell in range(self.ncpl[0]):
                verts = xgrid[icell]

                for ix in range(len(verts)):
                    grdlines.append(
                        [
                            (xgrid[icell][ix - 1], ygrid[icell][ix - 1]),
                            (xgrid[icell][ix], ygrid[icell][ix]),
                        ]
                    )

        self._copy_cache = True
        return grdlines

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
        Method to get model grid verticies

        Returns:
            list of dimension ncpl by nvertices
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
    def cross_section_vertices(self):
        """
        Method to get vertices for cross-sectional plotting

        Returns
        -------
            xvertices, yvertices
        """
        xv, yv = self.xyzvertices[0], self.xyzvertices[1]
        if len(xv) == self.ncpl[0]:
            xv *= self.nlay
            yv *= self.nlay
        return xv, yv

    def cross_section_lay_ncpl_ncb(self, ncb):
        """
        Get PlotCrossSection compatible layers, ncpl, and ncb
        variables

        Parameters
        ----------
        ncb : int
            number of confining beds

        Returns
        -------
            tuple : (int, int, int) layers, ncpl, ncb
        """
        return 1, self.nnodes, 0

    def cross_section_nodeskip(self, nlay, xypts):
        """
        Get a nodeskip list for PlotCrossSection. This is a correction
        for UnstructuredGridPlotting

        Parameters
        ----------
        nlay : int
            nlay is nlay + ncb
        xypts : dict
            dictionary of node number and xyvertices of a cross-section

        Returns
        -------
            list : n-dimensional list of nodes to not plot for each layer
        """
        strt = 0
        end = 0
        nodeskip = []
        for ncpl in self.ncpl:
            end += ncpl
            layskip = []
            for nn, verts in xypts.items():
                if strt <= nn < end:
                    continue
                else:
                    layskip.append(nn)

            strt += ncpl
            nodeskip.append(layskip)

        return nodeskip

    def cross_section_adjust_indicies(self, k, cbcnt):
        """
        Method to get adjusted indicies by layer and confining bed
        for PlotCrossSection plotting

        Parameters
        ----------
        k : int
            zero based model layer
        cbcnt : int
            confining bed counter

        Returns
        -------
            tuple: (int, int, int) (adjusted layer, nodeskip layer, node
            adjustment value based on number of confining beds and the layer)
        """
        return 1, k + 1, 0

    def cross_section_set_contour_arrays(
        self, plotarray, xcenters, head, elev, projpts
    ):
        """
        Method to set countour array centers for rare instances where
        matplotlib contouring is prefered over trimesh plotting

        Parameters
        ----------
        plotarray : np.ndarray
            array of data for contouring
        xcenters : np.ndarray
            xcenters array
        head : np.ndarray
            head array to adjust cell centers location
        elev : np.ndarray
            cell elevation array
        projpts : dict
            dictionary of projected cross sectional vertices

        Returns
        -------
            tuple: (np.ndarray, np.ndarray, np.ndarray, bool)
            plotarray, xcenter array, ycenter array, and a boolean flag
            for contouring
        """
        if self.ncpl[0] != self.nnodes:
            return plotarray, xcenters, None, False
        else:
            zcenters = []
            if isinstance(head, np.ndarray):
                head = head.reshape(1, self.nnodes)
                head = np.vstack((head, head))
            else:
                head = elev.reshape(2, self.nnodes)

            elev = elev.reshape(2, self.nnodes)
            for k, ev in enumerate(elev):
                if k == 0:
                    zc = [
                        ev[i] if head[k][i] > ev[i] else head[k][i]
                        for i in sorted(projpts)
                    ]
                else:
                    zc = [ev[i] for i in sorted(projpts)]
                zcenters.append(zc)

            plotarray = np.vstack((plotarray, plotarray))
            xcenters = np.vstack((xcenters, xcenters))
            zcenters = np.array(zcenters)

            return plotarray, xcenters, zcenters, True

    @property
    def map_polygons(self):
        """
        Property to get Matplotlib polygon objects for the modelgrid

        Returns
        -------
            list or dict of matplotlib.collections.Polygon
        """
        try:
            from matplotlib.path import Path
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
            if self.grid_varies_by_layer:
                self._polygons = {}
                ilay = 0
                lay_break = np.cumsum(self.ncpl)
                for nn in range(self.nnodes):
                    if nn in lay_break:
                        ilay += 1

                    if ilay not in self._polygons:
                        self._polygons[ilay] = []

                    p = Path(self.get_cell_vertices(nn))
                    self._polygons[ilay].append(p)
            else:
                self._polygons = [
                    Path(self.get_cell_vertices(nn))
                    for nn in range(self.ncpl[0])
                ]

        return copy.copy(self._polygons)

    def intersect(self, x, y, local=False, forgive=False):
        x, y = super().intersect(x, y, local, forgive)
        raise Exception("Not implemented yet")

    @property
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        new_botm = np.expand_dims(self._botm, 0)
        return np.concatenate((new_top, new_botm), axis=0)

    def get_cell_vertices(self, cellid):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param cellid: (int) cellid number
        Returns
        ------- list of x,y cell vertices
        """
        self._copy_cache = False
        cell_vert = list(zip(self.xvertices[cellid], self.yvertices[cellid]))
        self._copy_cache = True
        return cell_vert

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

        layer = 0
        if "layer" in kwargs:
            layer = kwargs.pop("layer")
        mm = PlotMapView(modelgrid=self, layer=layer)
        return mm.plot_grid(**kwargs)

    def _build_grid_geometry_info(self):
        cache_index_cc = "cellcenters"
        cache_index_vert = "xyzgrid"

        vertexdict = {int(v[0]): [v[1], v[2]] for v in self._vertices}
        xcenters = self._xc
        ycenters = self._yc
        xvertices = []
        yvertices = []

        # build xy vertex and cell center info
        for iverts in self._iverts:

            xcellvert = []
            ycellvert = []
            for ix in iverts:
                xcellvert.append(vertexdict[ix][0])
                ycellvert.append(vertexdict[ix][1])

            xvertices.append(xcellvert)
            yvertices.append(ycellvert)

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
            [xcenters, ycenters, zcenters]
        )
        self._cache_dict[cache_index_vert] = CachedData(
            [xvertices, yvertices, zvertices]
        )

    def get_layer_node_range(self, layer):
        node_layer_range = [0] + list(np.add.accumulate(self.ncpl))
        return node_layer_range[layer], node_layer_range[layer + 1]

    def get_xvertices_for_layer(self, layer):
        xgrid = np.array(self.xvertices, dtype=object)
        if self.grid_varies_by_layer:
            istart, istop = self.get_layer_node_range(layer)
            xgrid = xgrid[istart:istop]
        return xgrid

    def get_yvertices_for_layer(self, layer):
        ygrid = np.array(self.yvertices, dtype=object)
        if self.grid_varies_by_layer:
            istart, istop = self.get_layer_node_range(layer)
            ygrid = ygrid[istart:istop]
        return ygrid

    def get_xcellcenters_for_layer(self, layer):
        xcenters = self.xcellcenters
        if self.grid_varies_by_layer:
            istart, istop = self.get_layer_node_range(layer)
            xcenters = xcenters[istart:istop]
        return xcenters

    def get_ycellcenters_for_layer(self, layer):
        ycenters = self.ycellcenters
        if self.grid_varies_by_layer:
            istart, istop = self.get_layer_node_range(layer)
            ycenters = ycenters[istart:istop]
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
        if a.size == self.nnodes:
            nplottable = self.nlay
        return nplottable

    def get_plottable_layer_array(self, a, layer):
        if a.shape[0] == self.ncpl[layer]:
            # array is already the size to be plotted
            plotarray = a
        else:
            # reshape the array into size nodes and then reset range to
            # the part of the array for this layer
            plotarray = np.reshape(a, (self.nnodes,))
            istart, istop = self.get_layer_node_range(layer)
            plotarray = plotarray[istart:istop]
        assert plotarray.shape[0] == self.ncpl[layer]
        return plotarray

    def get_plottable_layer_shape(self, layer=None):
        """
        Determine the shape that is required in order to plot in 2d for
        this grid.

        Parameters
        ----------
        layer : int
            Has no effect unless grid changes by layer

        Returns
        -------
        shape : tuple
            required shape of array to plot for a layer
        """
        shp = (self.nnodes,)
        if layer is not None:
            shp = (self.ncpl[layer],)
        return shp

    @classmethod
    def from_argus_export(cls, fname, nlay=1):
        """
        Create a new UnstructuredGrid from an Argus One Trimesh file

        Parameters
        ----------
        fname : string
            File name

        nlay : int
            Number of layers to create

        Returns
        -------
        flopy.discretization.unstructuredgrid.UnstructuredGrid

        """
        from ..utils.geometry import get_polygon_centroid

        f = open(fname, "r")
        line = f.readline()
        ll = line.split()
        ncells, nverts = ll[0:2]
        ncells = int(ncells)
        nverts = int(nverts)
        verts = np.empty((nverts, 3), dtype=float)
        xc = np.empty((ncells), dtype=float)
        yc = np.empty((ncells), dtype=float)

        # read the vertices
        f.readline()
        for ivert in range(nverts):
            line = f.readline()
            ll = line.split()
            c, iv, x, y = ll[0:4]
            verts[ivert, 0] = int(iv) - 1
            verts[ivert, 1] = x
            verts[ivert, 2] = y

        # read the cell information and create iverts, xc, and yc
        iverts = []
        for icell in range(ncells):
            line = f.readline()
            ll = line.split()
            ivlist = []
            for ic in ll[2:5]:
                ivlist.append(int(ic) - 1)
            if ivlist[0] != ivlist[-1]:
                ivlist.append(ivlist[0])
            iverts.append(ivlist)
            xc[icell], yc[icell] = get_polygon_centroid(verts[ivlist, 1:])

        # close file and return spatial reference
        f.close()
        return cls(verts, iverts, xc, yc, ncpl=np.array(nlay * [len(iverts)]))

    @staticmethod
    def ncpl_from_ihc(ihc, iac):
        """
        Use the ihc and iac arrays to calculate the number of cells per layer
        array (ncpl) assuming that the plottable layer number is stored in
        the diagonal position of the ihc array.

        Parameters
        ----------
        ihc : ndarray
            horizontal indicator array.  If the plottable layer number is
            stored in the diagonal position, then this will be used to create
            the returned ncpl array.  plottable layer numbers must increase
            monotonically and be consecutive with node number
        iac : ndarray
            array of size nodes that has the number of connections for a cell,
            plus one for the cell itself

        Returns
        -------
        ncpl : ndarray
            number of cells per plottable layer

        """
        from flopy.utils.gridgen import get_ia_from_iac

        valid = False
        ia = get_ia_from_iac(iac)

        # look through the diagonal position of the ihc array, which is
        # assumed to represent the plottable zero-based layer number
        layers = ihc[ia[:-1]]

        # use np.unique to find the unique layer numbers and the occurrence
        # of each layer number
        unique_layers, ncpl = np.unique(layers, return_counts=True)

        # make sure unique layers numbers are monotonically increasing
        # and are consecutive integers
        if np.all(np.diff(unique_layers) == 1):
            valid = True
        if not valid:
            ncpl = None
        return ncpl

    # initialize grid from a grb file
    @classmethod
    def from_binary_grid_file(cls, file_path, verbose=False):
        """
        Instantiate a UnstructuredGrid model grid from a MODFLOW 6 binary
        grid (*.grb) file.

        Parameters
        ----------
        file_path : str
            file path for the MODFLOW 6 binary grid file
        verbose : bool
            Write information to standard output.  Default is False.

        Returns
        -------
        return : UnstructuredGrid

        """
        from ..mf6.utils.binarygrid_util import MfGrdFile

        grb_obj = MfGrdFile(file_path, verbose=verbose)
        if grb_obj.grid_type != "DISU":
            err_msg = (
                "Binary grid file ({}) ".format(os.path.basename(file_path))
                + "is not a vertex (DISU) grid."
            )
            raise ValueError(err_msg)

        iverts = grb_obj.iverts
        if iverts is not None:
            verts = grb_obj.verts
            vertc = grb_obj.cellcenters
            xc, yc = vertc[:, 0], vertc[:, 1]

            idomain = grb_obj.idomain
            xorigin = grb_obj.xorigin
            yorigin = grb_obj.yorigin
            angrot = grb_obj.angrot

            top = np.ravel(grb_obj.top)
            botm = grb_obj.bot

            return cls(
                vertices=verts,
                iverts=iverts,
                xcenters=xc,
                ycenters=yc,
                top=top,
                botm=botm,
                idomain=idomain,
                xoff=xorigin,
                yoff=yorigin,
                angrot=angrot,
            )
        else:
            err_msg = (
                "{} binary grid file".format(os.path.basename(file_path))
                + " does not include vertex data"
            )
            raise TypeError(err_msg)
