import os
import numpy as np
import subprocess
from ..mbase import which
from ..utils.cvfdutil import centroid_of_polygon
from ..utils.geospatial_utils import GeoSpatialUtil


class Triangle:
    """
    Class to work with the triangle program to unstructured triangular grids.
    Information on the triangle program can be found at
    https://www.cs.cmu.edu/~quake/triangle.html

    Parameters
    ----------
    model_ws : str
        workspace location for creating triangle files (default is '.')
    exe_name : str
        path and name of the triangle program. (default is triangle, which
        means that the triangle program must be in your path)
    maximum_area : float
        the maximum area for any triangle.  The default value is None, which
        means that the user must specify maximum areas for each region.
    angle : float
        Triangle will continue to add vertices until no angle is less than
        this specified value.  (default is 20 degrees)
    nodes : ndarray
        Two dimensional array of shape (npoints, 2) with x and y positions
        of fixed node locations to include in the resulting triangular mesh.
        (default is None)
    additional_args : list
        list of additional command line switches to pass to triangle

    Returns
    -------
    None

    """

    def __init__(
        self,
        model_ws=".",
        exe_name="triangle",
        maximum_area=None,
        angle=20.0,
        nodes=None,
        additional_args=None,
    ):
        self.model_ws = model_ws
        exe_name = which(exe_name)
        if exe_name is None:
            raise Exception("Cannot find triangle binary executable")
        self.exe_name = os.path.abspath(exe_name)
        self.angle = angle
        self.maximum_area = maximum_area
        self._nodes = nodes
        self.additional_args = additional_args
        self._initialize_vars()
        return

    def add_polygon(self, polygon):
        """
        Add a polygon

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            add polygon method accepts any of these geometries:

            a list of (x, y) points
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy.utils.geometry.Polygon object

        Returns
        -------
        None

        """
        if isinstance(polygon, (list, tuple, np.ndarray)):
            polygon = [polygon]

        geom = GeoSpatialUtil(polygon, shapetype="Polygon")
        polygon = geom.points
        self._polygons.append(polygon[0])
        if len(polygon) > 1:
            for hole in polygon[1:]:
                self.add_hole(hole)
        return

    def add_hole(self, hole):
        """
        Add a point that will turn enclosing polygon into a hole

        Parameters
        ----------
        hole : tuple
            (x, y)

        Returns
        -------
        None

        """
        self._holes.append(hole)
        return

    def add_region(self, point, attribute=0, maximum_area=None):
        """
        Add a point that will become a region with a maximum area, if
        specified.

        Parameters
        ----------
        point : tuple
            (x, y)

        attribute : integer or float
            integer value assigned to output elements

        maximum_area : float
            maximum area of elements in region

        Returns
        -------
        None

        """
        self._regions.append([point, attribute, maximum_area])
        return

    def build(self, verbose=False):
        """
        Build the triangular mesh

        Parameters
        ----------
        verbose : bool
            If true, print the results of the triangle command to the terminal
            (default is False)

        Returns
        -------
        None

        """

        # provide some protection by removing existing files
        self.clean()

        # write the active domain to a file
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.0.node")
        self._write_nodefile(fname)

        # poly file
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.0.poly")
        self._write_polyfile(fname)

        # Construct the triangle command
        cmds = [self.exe_name]
        if self.maximum_area is not None:
            cmds.append(f"-a{self.maximum_area}")
        else:
            cmds.append("-a")
        if self.angle is not None:
            cmds.append(f"-q{self.angle}")
        if self.additional_args is not None:
            cmds += self.additional_args
        cmds.append("-A")  # assign attributes
        cmds.append("-p")  # triangulate .poly file
        cmds.append("-V")  # verbose
        cmds.append("-D")  # delaunay triangles for finite volume
        cmds.append("-e")  # edge file
        cmds.append("-n")  # neighbor file
        cmds.append(f"{self.file_prefix}.0")  # output file name

        # run Triangle
        buff = subprocess.check_output(cmds, cwd=self.model_ws)
        buff = buff.decode()
        if verbose:
            print(buff)

        # load the results
        self._load_results()
        self.ncpl = self.ele.shape[0]
        self.nvert = self.node.shape[0]

        # create verts and iverts
        self.verts = self.node[["x", "y"]]
        self.verts = np.array(self.verts.tolist(), float)
        self.iverts = []
        for row in self.ele:
            self.iverts.append([row[1], row[2], row[3]])

        return

    def plot(
        self,
        ax=None,
        layer=0,
        edgecolor="k",
        facecolor="none",
        cmap="Dark2",
        a=None,
        masked_values=None,
        **kwargs,
    ):
        """
        Plot the grid.  This method will plot the grid using the shapefile
        that was created as part of the build method.

        Note that the layer option is not working yet.

        Parameters
        ----------
        ax : matplotlib.pyplot axis
            The plot axis.  If not provided it, plt.gca() will be used.
            If there is not a current axis then a new one will be created.
        layer : int
            Layer number to plot
        cmap : string
            Name of colormap to use for polygon shading (default is 'Dark2')
        edgecolor : string
            Color name.  (Default is 'scaled' to scale the edge colors.)
        facecolor : string
            Color name.  (Default is 'scaled' to scale the face colors.)
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        kwargs : dictionary
            Keyword arguments that are passed to
            PatchCollection.set(``**kwargs``).  Some common kwargs would be
            'linewidths', 'linestyles', 'alpha', etc.

        Returns
        -------
        None

        """
        from ..plot import PlotMapView
        from ..discretization import VertexGrid

        cell2d = self.get_cell2d()
        vertices = self.get_vertices()
        ncpl = len(cell2d)

        modelgrid = VertexGrid(
            vertices=vertices, cell2d=cell2d, ncpl=ncpl, nlay=1
        )

        pmv = PlotMapView(modelgrid=modelgrid, ax=ax, layer=layer)
        if a is None:
            pc = pmv.plot_grid(
                facecolor=facecolor, edgecolor=edgecolor, **kwargs
            )
        else:
            pc = pmv.plot_array(
                a,
                masked_values=masked_values,
                cmap=cmap,
                edgecolor=edgecolor,
                **kwargs,
            )

        return pc

    def get_boundary_marker_array(self):
        """
        Get an integer array that has boundary markers

        Returns
        -------
        iedge : ndarray
            integer array of size ncpl containing a boundary ids.  The array
            contains zeros for cells that do not touch a boundary.  The
            boundary ids are the segment numbers for each segment in each
            polygon that is added with the add_polygon method.

        """
        iedge = np.zeros((self.ncpl), dtype=int)
        boundary_markers = np.unique(self.edge["boundary_marker"])
        for ibm in boundary_markers:
            icells = self.get_edge_cells(ibm)
            iedge[icells] = ibm
        return iedge

    def plot_boundary(self, ibm, ax=None, **kwargs):
        """
        Plot a line and vertices for the specified boundary marker

        Parameters
        ----------
        ibm : integer
            plot the boundary for this boundary marker

        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use triangle.plot_boundary()"
            )
        if ax is None:
            ax = plt.gca()
        idx = np.where(self.edge["boundary_marker"] == ibm)[0]
        for i in idx:
            iv1 = self.edge["endpoint1"][i]
            iv2 = self.edge["endpoint2"][i]
            x1 = self.node["x"][iv1]
            x2 = self.node["x"][iv2]
            y1 = self.node["y"][iv1]
            y2 = self.node["y"][iv2]
            ax.plot([x1, x2], [y1, y2], **kwargs)
        return

    def plot_vertices(self, ax=None, **kwargs):
        """
        Plot the mesh vertices

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use triangle.plot_vertices()"
            )
        if ax is None:
            ax = plt.gca()
        ax.plot(self.node["x"], self.node["y"], lw=0, **kwargs)
        return

    def label_vertices(self, ax=None, onebased=True, **kwargs):
        """
        Label the mesh vertices with their vertex numbers

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        onebased : bool
            Make the labels one-based if True so that they correspond to
            what would be written to MODFLOW.

        kwargs : dictionary
            dictionary of arguments to pass to ax.text()

        Returns
        -------
        None

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use triangle.label_vertices()"
            )
        if ax is None:
            ax = plt.gca()
        for i in range(self.verts.shape[0]):
            x = self.verts[i, 0]
            y = self.verts[i, 1]
            s = i
            if onebased:
                s += 1
            ax.text(x, y, str(s), **kwargs)
        return

    def plot_centroids(self, ax=None, **kwargs):
        """
        Plot the cell centroids

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use triangle.plot_centroids()"
            )

        if ax is None:
            ax = plt.gca()
        xcyc = self.get_xcyc()
        ax.plot(xcyc[:, 0], xcyc[:, 1], lw=0, **kwargs)
        return

    def label_cells(self, ax=None, onebased=True, **kwargs):
        """
        Label the cells with their cell numbers

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        onebased : bool
            Make the labels one-based if True so that they correspond to
            what would be written to MODFLOW.

        kwargs : dictionary
            dictionary of arguments to pass to ax.text()

        Returns
        -------
        None

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use triangle.lavel_cells()"
            )
        if ax is None:
            ax = plt.gca()
        xcyc = self.get_xcyc()
        for i in range(xcyc.shape[0]):
            x = xcyc[i, 0]
            y = xcyc[i, 1]
            s = i
            if onebased:
                s += 1
            ax.text(x, y, str(s), **kwargs)
        return

    def get_xcyc(self):
        """
        Get a 2-dimensional array of x and y cell center coordinates.

        Returns
        -------
        xcyc : ndarray
            column 0 contains the x coordinates and column 1 contains the
            y coordinates

        """
        ncpl = len(self.iverts)
        xcyc = np.empty((ncpl, 2), dtype=float)
        for i, icell2d in enumerate(self.iverts):
            points = []
            for iv in icell2d:
                x = self.verts[iv, 0]
                y = self.verts[iv, 1]
                points.append((x, y))
            xc, yc = centroid_of_polygon(points)
            xcyc[i, 0] = xc
            xcyc[i, 1] = yc
        return xcyc

    def get_cell2d(self):
        """
        Get a list of the information needed for the MODFLOW DISV Package.

        Returns
        -------
        cell2d : list (of lists)
            innermost list contains cell number, x, y, number of vertices, and
            then the vertex numbers comprising the cell.

        """
        cell2d = []
        xcyc = self.get_xcyc()
        for i, icell2d in enumerate(self.iverts):
            ic2dr = icell2d[::-1]
            cell2d.append([i, xcyc[i, 0], xcyc[i, 1], len(icell2d)] + ic2dr)
        return cell2d

    def get_vertices(self):
        """
        Get a list of vertices in the form needed for the MODFLOW DISV Package.

        Returns
        -------
        vertices : list (of lists)
            innermost list contains vertex number, x, and y

        """
        vertices = []
        for i, row in enumerate(self.verts):
            vertices.append([i, row[0], row[1]])
        return vertices

    def get_edge_cells(self, ibm):
        """
        Get a list of cell numbers that correspond to the specified boundary
        marker.

        Parameters
        ----------
        ibm : integer
            boundary marker value

        Returns
        -------
        cell_list : list
            list of zero-based cell numbers

        """
        # Create the edge dictionary if it doesn't exist
        if self.edgedict is None:
            self._create_edge_dict()

        # Create a list of cells for boundary marker ibm
        cell_list = []
        edgedict = self.edgedict
        for n, ivlist in enumerate(self.iverts):
            itmp = ivlist + [ivlist[0]]
            for i in range(len(ivlist)):
                ie = (itmp[i], itmp[i + 1])
                if ie in edgedict:
                    if edgedict[ie] == ibm:
                        cell_list.append(n)

        return cell_list

    def get_cell_edge_length(self, n, ibm):
        """
        Get the length of the edge for cell n that corresponds to
        boundary marker ibm

        Parameters
        ----------
        n : int
            cell number.  0 <= n < self.ncpl

        ibm : integer
            boundary marker number

        Returns
        -------
        length : float
            Length of the edge along that boundary marker.  Will
            return None if cell n does not touch boundary marker.

        """

        assert 0 <= n < self.ncpl, "Not a valid cell number"

        # Create the edge dictionary if it doesn't exist
        if self.edgedict is None:
            self._create_edge_dict()

        ivlist = self.iverts[n]
        itmp = ivlist + [ivlist[0]]
        d = None
        for i in range(len(ivlist)):
            iv1 = itmp[i]
            iv2 = itmp[i + 1]
            ie = (itmp[i], itmp[i + 1])
            if ie in self.edgedict:
                if self.edgedict[ie] == ibm:
                    x1, y1 = self.verts[iv1]
                    x2, y2 = self.verts[iv2]
                    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    return d
        return d

    def get_attribute_array(self):
        """
        Return an array containing the attribute value for each cell.  These
        are the attribute values that are passed into the add_region() method.

        Returns
        -------
        attribute_array : ndarray

        """
        return self.ele["attribute"]

    def clean(self):
        """
        Remove the input and output files created by this class and by the
        Triangle program

        Returns
        -------
        None

        """
        # remove input files
        for ext in ["poly", "node"]:
            fname = os.path.join(self.model_ws, f"{self.file_prefix}0.{ext}")
            if os.path.isfile(fname):
                os.remove(fname)
                if os.path.isfile(fname):
                    print(f"Could not remove: {fname}")
        # remove output files
        for ext in ["poly", "ele", "node", "neigh", "edge"]:
            fname = os.path.join(self.model_ws, f"{self.file_prefix}1.{ext}")
            if os.path.isfile(fname):
                os.remove(fname)
                if os.path.isfile(fname):
                    print(f"Could not remove: {fname}")
        return

    def _initialize_vars(self):
        self.file_prefix = "_triangle"
        self.ncpl = 0
        self.nvert = 0
        self._active_domain = None
        self._polygons = []
        self._holes = []
        self._regions = []
        self.verts = None
        self.iverts = None
        self.edgedict = None
        return

    def _load_results(self):

        # node file
        ext = "node"
        dt = [("ivert", int), ("x", float), ("y", float)]
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.1.{ext}")
        setattr(self, ext, None)
        if os.path.isfile(fname):
            f = open(fname, "r")
            line = f.readline()
            f.close()
            ll = line.strip().split()
            nvert = int(ll[0])
            ndim = int(ll[1])
            assert ndim == 2, "Dimensions in node file is not 2"
            iattribute = int(ll[2])
            if iattribute == 1:
                dt.append(("attribute", int))
            ibm = int(ll[3])
            if ibm == 1:
                dt.append(("boundary_marker", int))
            a = np.loadtxt(fname, skiprows=1, comments="#", dtype=dt)
            assert a.shape[0] == nvert
            setattr(self, ext, a)

        # ele file
        ext = "ele"
        dt = [("icell", int), ("iv1", int), ("iv2", int), ("iv3", int)]
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.1.{ext}")
        setattr(self, ext, None)
        if os.path.isfile(fname):
            f = open(fname, "r")
            line = f.readline()
            f.close()
            ll = line.strip().split()
            ncells = int(ll[0])
            npt = int(ll[1])
            assert npt == 3, "Nodes per triangle in ele file is not 3"
            iattribute = int(ll[2])
            if iattribute == 1:
                dt.append(("attribute", int))
            a = np.loadtxt(fname, skiprows=1, comments="#", dtype=dt)
            assert a.shape[0] == ncells
            setattr(self, ext, a)

        # edge file
        ext = "edge"
        dt = [("iedge", int), ("endpoint1", int), ("endpoint2", int)]
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.1.{ext}")
        setattr(self, ext, None)
        if os.path.isfile(fname):
            f = open(fname, "r")
            line = f.readline()
            f.close()
            ll = line.strip().split()
            nedges = int(ll[0])
            ibm = int(ll[1])
            if ibm == 1:
                dt.append(("boundary_marker", int))
            a = np.loadtxt(fname, skiprows=1, comments="#", dtype=dt)
            assert a.shape[0] == nedges
            setattr(self, ext, a)

        # neighbor file
        ext = "neigh"
        dt = [
            ("icell", int),
            ("neighbor1", int),
            ("neighbor2", int),
            ("neighbor3", int),
        ]
        fname = os.path.join(self.model_ws, f"{self.file_prefix}.1.{ext}")
        setattr(self, ext, None)
        if os.path.isfile(fname):
            f = open(fname, "r")
            line = f.readline()
            f.close()
            ll = line.strip().split()
            ncells = int(ll[0])
            nnpt = int(ll[1])
            assert nnpt == 3, "Neighbors per triangle in neigh file is not 3"
            a = np.loadtxt(fname, skiprows=1, comments="#", dtype=dt)
            assert a.shape[0] == ncells
            setattr(self, ext, a)

        return

    def _write_nodefile(self, fname):
        f = open(fname, "w")
        nvert = 0
        for p in self._polygons:
            nvert += len(p)
        if self._nodes is not None:
            nvert += self._nodes.shape[0]
        s = f"{nvert} 2 0 0\n"
        f.write(s)
        ip = 0
        for p in self._polygons:
            for vertex in p:
                s = f"{ip} {vertex[0]} {vertex[1]}\n"
                f.write(s)
                ip += 1
        if self._nodes is not None:
            for i in range(self._nodes.shape[0]):
                s = f"{ip} {self._nodes[i, 0]} {self._nodes[i, 1]}\n"
                f.write(s)
                ip += 1
        f.close()

    def _write_polyfile(self, fname):
        f = open(fname, "w")

        # vertices, write zero to indicate read from node file
        s = "{} {} {} {}\n".format(0, 0, 0, 0)
        f.write(s)

        # segments
        nseg = 0
        for p in self._polygons:
            nseg += len(p)
        bm = 1
        s = f"{nseg} {bm}\n"
        f.write(s)

        iseg = 0
        ipstart = 0
        for p in self._polygons:
            nseg = len(p)
            for i in range(nseg):
                ep1 = i
                ep2 = i + 1
                if ep2 > nseg - 1:
                    ep2 = 0
                ep1 += ipstart
                ep2 += ipstart
                s = f"{iseg} {ep1} {ep2} {iseg + 1}\n"
                f.write(s)
                iseg += 1
            ipstart += len(p)

        # holes
        nholes = len(self._holes)
        s = f"{nholes}\n"
        f.write(s)
        for i, hole in enumerate(self._holes):
            s = f"{i} {hole[0]} {hole[1]}\n"
            f.write(s)

        # regions
        nregions = len(self._regions)
        s = f"{nregions}\n"
        f.write(s)
        for i, region in enumerate(self._regions):
            pt = region[0]
            attribute = region[1]
            maxarea = region[2]
            if maxarea is None:
                maxarea = -1.0
            s = f"{i} {pt[0]} {pt[1]} {attribute} {maxarea}\n"
            f.write(s)

        f.close()
        return

    def _create_edge_dict(self):
        """
        Create the edge dictionary

        """
        edgedict = {}
        for _, iv1, iv2, iseg in self.edge:
            if iseg != 0:
                edgedict[(iv1, iv2)] = iseg
                edgedict[(iv2, iv1)] = iseg
        self.edgedict = edgedict
        return
