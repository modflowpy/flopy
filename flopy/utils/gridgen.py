from __future__ import print_function
import os
import numpy as np
import subprocess

# flopy imports
from ..modflow.mfdisu import ModflowDisU
from ..mf6.modflow import ModflowGwfdis
from .util_array import Util2d  # read1d,
from ..export.shapefile_utils import shp2recarray
from ..mbase import which
from ..export.shapefile_utils import import_shapefile

shapefile = import_shapefile()


# todo
# creation of line and polygon shapefiles from features (holes!)
# program layer functionality for plot method
# support an asciigrid option for top and bottom interpolation
# add intersection capability


def read1d(f, a):
    """
    Quick file to array reader for reading gridgen output.  Much faster
    than the read1d function in util_array

    """
    dtype = a.dtype.type
    lines = f.readlines()
    l = []
    for line in lines:
        l += [dtype(i) for i in line.strip().split()]
    a[:] = np.array(l, dtype=dtype)
    return a


def features_to_shapefile(features, featuretype, filename):
    """
    Write a shapefile for the features of type featuretype.

    Parameters
    ----------
    features : list
        point, line, or polygon features. Method accepts
        feature can be:
             a list of geometries
             flopy.utils.geometry.Collection object
             shapely.geometry.Collection object
             geojson.GeometryCollection object
             list of shapefile.Shape objects
             shapefile.Shapes object
    featuretype : str
        Must be 'point', 'line', 'linestring', or 'polygon'
    filename : string
        name of the shapefile to write

    Returns
    -------
    None

    """
    from .geospatial_utils import GeoSpatialCollection

    if featuretype.lower() == "line":
        featuretype = "LineString"

    features = GeoSpatialCollection(features, featuretype).flopy_geometry

    if featuretype.lower() not in ["point", "line", "linestring", "polygon"]:
        raise Exception("Unrecognized feature type: {}".format(featuretype))

    if featuretype.lower() in ("line", 'linestring'):
        wr = shapefile.Writer(filename, shapeType=shapefile.POLYLINE)
        wr.field("SHAPEID", "N", 20, 0)
        for i, line in enumerate(features):
            wr.line(line.__geo_interface__['coordinates'])
            wr.record(i)

    elif featuretype.lower() == "point":
        wr = shapefile.Writer(filename, shapeType=shapefile.POINT)
        wr.field("SHAPEID", "N", 20, 0)
        for i, point in enumerate(features):
            wr.point(*point.__geo_interface__['coordinates'])
            wr.record(i)

    elif featuretype.lower() == "polygon":
        wr = shapefile.Writer(filename, shapeType=shapefile.POLYGON)
        wr.field("SHAPEID", "N", 20, 0)
        for i, polygon in enumerate(features):
            wr.poly(polygon.__geo_interface__['coordinates'])
            wr.record(i)

    wr.close()
    return


def ndarray_to_asciigrid(fname, a, extent, nodata=1.0e30):
    # extent info
    xmin, xmax, ymin, ymax = extent
    ncol, nrow = a.shape
    dx = (xmax - xmin) / ncol
    assert dx == (ymax - ymin) / nrow
    # header
    header = "ncols     {}\n".format(ncol)
    header += "nrows    {}\n".format(nrow)
    header += "xllcorner {}\n".format(xmin)
    header += "yllcorner {}\n".format(ymin)
    header += "cellsize {}\n".format(dx)
    header += "NODATA_value {}\n".format(np.float(nodata))
    # replace nan with nodata
    idx = np.isnan(a)
    a[idx] = np.float(nodata)
    # write
    with open(fname, "wb") as f:
        f.write(header.encode("ascii"))
        np.savetxt(f, a, fmt="%15.6e")
    return


class Gridgen(object):
    """
    Class to work with the gridgen program to create layered quadtree grids.

    Parameters
    ----------
    dis : flopy.modflow.ModflowDis
        Flopy discretization object
    model_ws : str
        workspace location for creating gridgen files (default is '.')
    exe_name : str
        path and name of the gridgen program. (default is gridgen)
    surface_interpolation : str
        Default gridgen method for interpolating elevations.  Valid options
        include 'replicate' (default) and 'interpolate'
    vertical_pass_through : bool
        If true, Gridgen's GRID_TO_USGDATA command will connect layers
        where intermediate layers are inactive.
        (default is False)

    Notes
    -----
    For the surface elevations, the top of a layer uses the same surface as
    the bottom of the overlying layer.

    """

    def __init__(
        self,
        dis,
        model_ws=".",
        exe_name="gridgen",
        surface_interpolation="replicate",
        vertical_pass_through=False,
    ):
        self.dis = dis
        if isinstance(dis, ModflowGwfdis):
            self.nlay = self.dis.nlay.get_data()
            self.nrow = self.dis.nrow.get_data()
            self.ncol = self.dis.ncol.get_data()
            self.modelgrid = self.dis.parent.modelgrid
        else:
            self.nlay = self.dis.nlay
            self.nrow = self.dis.nrow
            self.ncol = self.dis.ncol
            self.modelgrid = self.dis.parent.modelgrid

        self.nodes = 0
        self.nja = 0
        self.nodelay = np.zeros((self.nlay), dtype=np.int)
        self._vertdict = {}
        self.model_ws = model_ws
        exe_name = which(exe_name)
        if exe_name is None:
            raise Exception("Cannot find gridgen binary executable")
        self.exe_name = os.path.abspath(exe_name)

        # Set default surface interpolation for all surfaces (nlay + 1)
        surface_interpolation = surface_interpolation.upper()
        if surface_interpolation not in ["INTERPOLATE", "REPLICATE"]:
            raise Exception(
                "Error.  Unknown surface interpolation method: "
                "{}.  Must be INTERPOLATE or "
                "REPLICATE".format(surface_interpolation)
            )
        self.surface_interpolation = [
            surface_interpolation for k in range(self.nlay + 1)
        ]

        # Set export options
        self.vertical_pass_through = "False"
        if vertical_pass_through:
            self.vertical_pass_through = "True"

        # Set up a blank _active_domain list with None for each layer
        self._addict = {}
        self._active_domain = []
        for k in range(self.nlay):
            self._active_domain.append(None)

        # Set up a blank _refinement_features list with empty list for
        # each layer
        self._rfdict = {}
        self._refinement_features = []
        for k in range(self.nlay):
            self._refinement_features.append([])

        # Set up blank _elev and _elev_extent dictionaries
        self._asciigrid_dict = {}

        return

    def set_surface_interpolation(
        self, isurf, type, elev=None, elev_extent=None
    ):
        """
        Parameters
        ----------
        isurf : int
            surface number where 0 is top and nlay + 1 is bottom
        type : str
            Must be 'INTERPOLATE', 'REPLICATE' or 'ASCIIGRID'.
        elev : numpy.ndarray of shape (nr, nc) or str
            Array that is used as an asciigrid.  If elev is a string, then
            it is assumed to be the name of the asciigrid.
        elev_extent : list-like
            List of xmin, xmax, ymin, ymax extents of the elev grid.
            Must be specified for ASCIIGRID; optional otherwise.

        Returns
        -------
        None

        """

        assert 0 <= isurf <= self.nlay + 1
        type = type.upper()
        if type not in ["INTERPOLATE", "REPLICATE", "ASCIIGRID"]:
            raise Exception(
                "Error.  Unknown surface interpolation type: "
                "{}.  Must be INTERPOLATE or "
                "REPLICATE".format(type)
            )
        else:
            self.surface_interpolation[isurf] = type

        if type == "ASCIIGRID":
            if isinstance(elev, np.ndarray):
                if elev_extent is None:
                    raise Exception(
                        "Error.  ASCIIGRID was specified but "
                        "elev_extent was not."
                    )
                try:
                    xmin, xmax, ymin, ymax = elev_extent
                except:
                    raise Exception(
                        "Cannot cast elev_extent into xmin, xmax, "
                        "ymin, ymax: {}".format(elev_extent)
                    )

                nm = "_gridgen.lay{}.asc".format(isurf)
                fname = os.path.join(self.model_ws, nm)
                ndarray_to_asciigrid(fname, elev, elev_extent)
                self._asciigrid_dict[isurf] = nm

            elif isinstance(elev, str):
                if not os.path.isfile(os.path.join(self.model_ws, elev)):
                    raise Exception(
                        "Error.  elev is not a valid file: "
                        "{}".format(os.path.join(self.model_ws, elev))
                    )
                self._asciigrid_dict[isurf] = elev
            else:
                raise Exception(
                    "Error.  ASCIIGRID was specified but "
                    "elev was not specified as a numpy ndarray or"
                    "valid asciigrid file."
                )
        return

    def add_active_domain(self, feature, layers):
        """
        Parameters
        ----------
        feature : str or list
            feature can be:
                 a string containing the name of a polygon
                 a list of polygons
                 flopy.utils.geometry.Collection object of Polygons
                 shapely.geometry.Collection object of Polygons
                 geojson.GeometryCollection object of Polygons
                 list of shapefile.Shape objects
                 shapefile.Shapes object

        layers : list
            A list of layers (zero based) for which this active domain
            applies.

        Returns
        -------
        None

        """
        # set nodes and nja to 0 to indicate that grid must be rebuilt
        self.nodes = 0
        self.nja = 0

        # Create shapefile or set shapefile to feature
        adname = "ad{}".format(len(self._addict))
        if isinstance(feature, list):
            # Create a shapefile
            adname_w_path = os.path.join(self.model_ws, adname)
            features_to_shapefile(feature, "polygon", adname_w_path)
            shapefile = adname
        else:
            shapefile = feature

        self._addict[adname] = shapefile
        sn = os.path.join(self.model_ws, shapefile + ".shp")
        assert os.path.isfile(sn), "Shapefile does not exist: {}".format(sn)

        for k in layers:
            self._active_domain[k] = adname

        return

    def add_refinement_features(self, features, featuretype, level, layers):
        """
        Parameters
        ----------
        features : str, list, or collection object
            features can be either a string containing the name of a shapefile
            or it can be:
                a list of points, lines, or polygons
                flopy.utils.geometry.Collection object
                a list of flopy.utils.geometry objects
                shapely.geometry.Collection object
                geojson.GeometryCollection object
                a list of shapefile.Shape objects
                shapefile.Shapes object

        featuretype : str
            Must be either 'point', 'line', or 'polygon'
        level : int
            The level of refinement for this features
        layers : list
            A list of layers (zero based) for which this refinement features
            applies.

        Returns
        -------
        None

        """
        # set nodes and nja to 0 to indicate that grid must be rebuilt
        self.nodes = 0
        self.nja = 0

        # todo: update the features to shapefile method for GeoSpatialCollection...
        # Create shapefile or set shapefile to feature
        rfname = "rf{}".format(len(self._rfdict))
        if isinstance(features, list):
            rfname_w_path = os.path.join(self.model_ws, rfname)
            features_to_shapefile(features, featuretype, rfname_w_path)
            shapefile = rfname
        else:
            shapefile = features

        self._rfdict[rfname] = [shapefile, featuretype, level]
        sn = os.path.join(self.model_ws, shapefile + ".shp")
        assert os.path.isfile(sn), "Shapefile does not exist: {}".format(sn)

        for k in layers:
            self._refinement_features[k].append(rfname)

        return

    def build(self, verbose=False):
        """
        Build the quadtree grid

        Parameters
        ----------
        verbose : bool
            If true, print the results of the gridgen command to the terminal
            (default is False)

        Returns
        -------
        None

        """
        fname = os.path.join(self.model_ws, "_gridgen_build.dfn")
        f = open(fname, "w")

        # Write the basegrid information
        f.write(self._mfgrid_block())
        f.write(2 * "\n")

        # Write the quadtree builder block
        f.write(self._builder_block())
        f.write(2 * "\n")

        # Write the active domain blocks
        f.write(self._ad_blocks())
        f.write(2 * "\n")

        # Write the refinement features
        f.write(self._rf_blocks())
        f.write(2 * "\n")
        f.close()

        # Command: gridgen quadtreebuilder _gridgen_build.dfn
        qtgfname = os.path.join(self.model_ws, "quadtreegrid.dfn")
        if os.path.isfile(qtgfname):
            os.remove(qtgfname)
        cmds = [self.exe_name, "quadtreebuilder", "_gridgen_build.dfn"]
        buff = subprocess.check_output(cmds, cwd=self.model_ws)
        if verbose:
            print(buff)
        assert os.path.isfile(qtgfname)

        # Export the grid to shapefiles, usgdata, and vtk files
        self.export(verbose)

        # Create a dictionary that relates nodenumber to vertices
        self._mkvertdict()

        # read and save nodelay array to self
        fname = os.path.join(self.model_ws, "qtg.nodesperlay.dat")
        f = open(fname, "r")
        self.nodelay = read1d(f, self.nodelay)
        f.close()

        # Create a recarray of the grid polygon shapefile
        shapename = os.path.join(self.model_ws, "qtgrid")
        self.qtra = shp2recarray(shapename)

        return

    def get_vertices(self, nodenumber):
        """
        Return a list of 5 vertices for the cell.  The first vertex should
        be the same as the last vertex.

        Parameters
        ----------
        nodenumber

        Returns
        -------
        list of vertices : list

        """
        return self._vertdict[nodenumber]

    def get_center(self, nodenumber):
        """
        Return the cell center x and y coordinates

        Parameters
        ----------
        nodenumber

        Returns
        -------
         (x, y) : tuple

        """
        vts = self.get_vertices(nodenumber)
        xmin = min(vts[0][0], vts[1][0], vts[2][0], vts[3][0])
        xmax = max(vts[0][0], vts[1][0], vts[2][0], vts[3][0])
        ymin = min(vts[0][1], vts[1][1], vts[2][1], vts[3][1])
        ymax = max(vts[0][1], vts[1][1], vts[2][1], vts[3][1])
        return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)

    def export(self, verbose=False):
        """
        Export the quadtree grid to shapefiles, usgdata, and vtk

        Parameters
        ----------
        verbose : bool
            If true, print the results of the gridgen command to the terminal
            (default is False)

        Returns
        -------
        None

        """

        # Create the export definition file
        fname = os.path.join(self.model_ws, "_gridgen_export.dfn")
        f = open(fname, "w")
        f.write("LOAD quadtreegrid.dfn\n")
        f.write("\n")
        f.write(self._grid_export_blocks())
        f.close()
        assert os.path.isfile(
            fname
        ), "Could not create export dfn file: {}".format(fname)

        # Export shapefiles
        cmds = [self.exe_name, "grid_to_shapefile_poly", "_gridgen_export.dfn"]
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            if verbose:
                print(buff)
            fn = os.path.join(self.model_ws, "qtgrid.shp")
            assert os.path.isfile(fn)
        except:
            print("Error.  Failed to export polygon shapefile of grid", buff)

        cmds = [
            self.exe_name,
            "grid_to_shapefile_point",
            "_gridgen_export.dfn",
        ]
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            if verbose:
                print(buff)
            fn = os.path.join(self.model_ws, "qtgrid_pt.shp")
            assert os.path.isfile(fn)
        except:
            print("Error.  Failed to export polygon shapefile of grid", buff)

        # Export the usg data
        cmds = [self.exe_name, "grid_to_usgdata", "_gridgen_export.dfn"]
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            if verbose:
                print(buff)
            fn = os.path.join(self.model_ws, "qtg.nod")
            assert os.path.isfile(fn)
        except:
            print("Error.  Failed to export usgdata", buff)

        # Export vtk
        cmds = [self.exe_name, "grid_to_vtk", "_gridgen_export.dfn"]
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            if verbose:
                print(buff)
            fn = os.path.join(self.model_ws, "qtg.vtu")
            assert os.path.isfile(fn)
        except:
            print("Error.  Failed to export vtk file", buff)

        cmds = [self.exe_name, "grid_to_vtk_sv", "_gridgen_export.dfn"]
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            if verbose:
                print(buff)
            fn = os.path.join(self.model_ws, "qtg_sv.vtu")
            assert os.path.isfile(fn)
        except:
            print("Error.  Failed to export shared vertex vtk file", buff)

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
        **kwargs
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
        pc : matplotlib.collections.PatchCollection

        """
        try:
            import matplotlib.pyplot as plt
        except:
            err_msg = "matplotlib must be installed to " + "use gridgen.plot()"
            raise ImportError(err_msg)

        from ..plot import plot_shapefile, shapefile_extents

        if ax is None:
            ax = plt.gca()
        shapename = os.path.join(self.model_ws, "qtgrid")
        xmin, xmax, ymin, ymax = shapefile_extents(shapename)

        idx = np.where(self.qtra.layer == layer)[0]

        pc = plot_shapefile(
            shapename,
            ax=ax,
            edgecolor=edgecolor,
            facecolor=facecolor,
            cmap=cmap,
            a=a,
            masked_values=masked_values,
            idx=idx,
            **kwargs
        )
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        return pc

    def get_nod_recarray(self):
        """
        Load the qtg.nod file and return as a numpy recarray

        Returns
        -------
        node_ra : ndarray
            Recarray representation of the node file with zero-based indexing

        """

        # nodes, nlay, ivsd, itmuni, lenuni, idsymrd, laycbd
        fname = os.path.join(self.model_ws, "qtg.nod")
        f = open(fname, "r")
        dt = np.dtype(
            [
                ("node", np.int),
                ("layer", np.int),
                ("x", np.float),
                ("y", np.float),
                ("z", np.float),
                ("dx", np.float),
                ("dy", np.float),
                ("dz", np.float),
            ]
        )
        node_ra = np.genfromtxt(fname, dtype=dt, skip_header=1)
        node_ra["layer"] -= 1
        node_ra["node"] -= 1
        return node_ra

    def get_disu(
        self,
        model,
        nper=1,
        perlen=1,
        nstp=1,
        tsmult=1,
        steady=True,
        itmuni=4,
        lenuni=2,
    ):
        """
        Create a MODFLOW-USG DISU flopy object.

        Parameters
        ----------
        model : Flopy model object
            The Flopy model object (of type :class:`flopy.modflow.mf.Modflow`)
            to which this package will be added.
        nper : int
            Number of model stress periods (default is 1).
        perlen : float or array of floats (nper)
            A single value or array of the stress period lengths
            (default is 1).
        nstp : int or array of ints (nper)
            Number of time steps in each stress period (default is 1).
        tsmult : float or array of floats (nper)
            Time step multiplier (default is 1.0).
        steady : boolean or array of boolean (nper)
            True or False indicating whether or not stress period is
            steady state (default is True).
        itmuni : int
            Time units, default is days (4)
        lenuni : int
            Length units, default is meters (2)

        Returns
        -------
        disu : Flopy ModflowDisU object.
        """

        # nodes, nlay, ivsd, itmuni, lenuni, idsymrd, laycbd
        fname = os.path.join(self.model_ws, "qtg.nod")
        f = open(fname, "r")
        line = f.readline()
        ll = line.strip().split()
        nodes = int(ll.pop(0))
        f.close()
        nlay = self.nlay
        ivsd = 0
        idsymrd = 0
        laycbd = 0

        # Save nodes
        self.nodes = nodes

        # nodelay
        nodelay = np.empty((nlay), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.nodesperlay.dat")
        f = open(fname, "r")
        nodelay = read1d(f, nodelay)
        f.close()

        # top
        top = [0] * nlay
        for k in range(nlay):
            fname = os.path.join(
                self.model_ws, "quadtreegrid.top{}.dat".format(k + 1)
            )
            f = open(fname, "r")
            tpk = np.empty((nodelay[k]), dtype=np.float32)
            tpk = read1d(f, tpk)
            f.close()
            if tpk.min() == tpk.max():
                tpk = tpk.min()
            else:
                tpk = Util2d(
                    model,
                    (nodelay[k],),
                    np.float32,
                    np.reshape(tpk, (nodelay[k],)),
                    name="top {}".format(k + 1),
                )
            top[k] = tpk

        # bot
        bot = [0] * nlay
        for k in range(nlay):
            fname = os.path.join(
                self.model_ws, "quadtreegrid.bot{}.dat".format(k + 1)
            )
            f = open(fname, "r")
            btk = np.empty((nodelay[k]), dtype=np.float32)
            btk = read1d(f, btk)
            f.close()
            if btk.min() == btk.max():
                btk = btk.min()
            else:
                btk = Util2d(
                    model,
                    (nodelay[k],),
                    np.float32,
                    np.reshape(btk, (nodelay[k],)),
                    name="bot {}".format(k + 1),
                )
            bot[k] = btk

        # area
        area = [0] * nlay
        fname = os.path.join(self.model_ws, "qtg.area.dat")
        f = open(fname, "r")
        anodes = np.empty((nodes), dtype=np.float32)
        anodes = read1d(f, anodes)
        f.close()
        istart = 0
        for k in range(nlay):
            istop = istart + nodelay[k]
            ark = anodes[istart:istop]
            if ark.min() == ark.max():
                ark = ark.min()
            else:
                ark = Util2d(
                    model,
                    (nodelay[k],),
                    np.float32,
                    np.reshape(ark, (nodelay[k],)),
                    name="area layer {}".format(k + 1),
                )
            area[k] = ark
            istart = istop

        # iac
        iac = np.empty((nodes), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.iac.dat")
        f = open(fname, "r")
        iac = read1d(f, iac)
        f.close()

        # Calculate njag and save as nja to self
        njag = iac.sum()
        self.nja = njag

        # ja
        ja = np.empty((njag), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.ja.dat")
        f = open(fname, "r")
        ja = read1d(f, ja)
        f.close()

        # ivc
        fldr = np.empty((njag), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.fldr.dat")
        f = open(fname, "r")
        fldr = read1d(f, fldr)
        ivc = np.where(abs(fldr) == 3, 1, 0)
        f.close()

        cl1 = None
        cl2 = None
        # cl12
        cl12 = np.empty((njag), dtype=np.float32)
        fname = os.path.join(self.model_ws, "qtg.c1.dat")
        f = open(fname, "r")
        cl12 = read1d(f, cl12)
        f.close()

        # fahl
        fahl = np.empty((njag), dtype=np.float32)
        fname = os.path.join(self.model_ws, "qtg.fahl.dat")
        f = open(fname, "r")
        fahl = read1d(f, fahl)
        f.close()

        # create dis object instance
        disu = ModflowDisU(
            model,
            nodes=nodes,
            nlay=nlay,
            njag=njag,
            ivsd=ivsd,
            nper=nper,
            itmuni=itmuni,
            lenuni=lenuni,
            idsymrd=idsymrd,
            laycbd=laycbd,
            nodelay=nodelay,
            top=top,
            bot=bot,
            area=area,
            iac=iac,
            ja=ja,
            ivc=ivc,
            cl1=cl1,
            cl2=cl2,
            cl12=cl12,
            fahl=fahl,
            perlen=perlen,
            nstp=nstp,
            tsmult=tsmult,
            steady=steady,
        )

        # return dis object instance
        return disu

    def get_nodes(self):
        """
        Get the number of nodes

        Returns
        -------
        nodes : int

        """
        fname = os.path.join(self.model_ws, "qtg.nod")
        f = open(fname, "r")
        line = f.readline()
        ll = line.strip().split()
        nodes = int(ll.pop(0))
        f.close()
        return nodes

    def get_nlay(self):
        """
        Get the number of layers

        Returns
        -------
        nlay : int

        """
        return self.nlay

    def get_nodelay(self):
        """
        Return the nodelay array, which is an array of size nlay containing
        the number of nodes in each layer.

        Returns
        -------
        nodelay : ndarray
            Number of nodes in each layer

        """
        nlay = self.get_nlay()
        nodelay = np.empty((nlay), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.nodesperlay.dat")
        f = open(fname, "r")
        nodelay = read1d(f, nodelay)
        f.close()
        return nodelay

    def get_top(self):
        """
        Get the top array

        Returns
        -------
        top : ndarray
            A 1D vector of cell top elevations of size nodes

        """
        nodes = self.get_nodes()
        nlay = self.get_nlay()
        nodelay = self.get_nodelay()
        top = np.empty((nodes), dtype=np.float32)
        istart = 0
        for k in range(nlay):
            istop = istart + nodelay[k]
            fname = os.path.join(
                self.model_ws, "quadtreegrid.top{}.dat".format(k + 1)
            )
            f = open(fname, "r")
            tpk = np.empty((nodelay[k]), dtype=np.float32)
            tpk = read1d(f, tpk)
            f.close()
            top[istart:istop] = tpk
            istart = istop
        return top

    def get_bot(self):
        """
        Get the bot array

        Returns
        -------
        bot : ndarray
            A 1D vector of cell bottom elevations of size nodes

        """
        nodes = self.get_nodes()
        nlay = self.get_nlay()
        nodelay = self.get_nodelay()
        bot = np.empty((nodes), dtype=np.float32)
        istart = 0
        for k in range(nlay):
            istop = istart + nodelay[k]
            fname = os.path.join(
                self.model_ws, "quadtreegrid.bot{}.dat".format(k + 1)
            )
            f = open(fname, "r")
            btk = np.empty((nodelay[k]), dtype=np.float32)
            btk = read1d(f, btk)
            f.close()
            bot[istart:istop] = btk
            istart = istop
        return bot

    def get_area(self):
        """
        Get the area array

        Returns
        -------
        area : ndarray
            A 1D vector of cell areas of size nodes

        """
        nodes = self.get_nodes()
        fname = os.path.join(self.model_ws, "qtg.area.dat")
        f = open(fname, "r")
        area = np.empty((nodes), dtype=np.float32)
        area = read1d(f, area)
        f.close()
        return area

    def get_iac(self):
        """
        Get the iac array

        Returns
        -------
        iac : ndarray
            A 1D vector of the number of connections (plus 1) for each cell

        """
        nodes = self.get_nodes()
        iac = np.empty((nodes), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.iac.dat")
        f = open(fname, "r")
        iac = read1d(f, iac)
        f.close()
        return iac

    def get_ja(self, nja=None):
        """
        Get the ja array

        Parameters
        ----------
        nja : int
            Number of connections.  If None, then it is read from gridgen
            output.

        Returns
        -------
        ja : ndarray
            A 1D vector of the cell connectivity (one-based)

        """
        if nja is None:
            iac = self.get_iac()
            nja = iac.sum()
        ja = np.empty((nja), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.ja.dat")
        f = open(fname, "r")
        ja = read1d(f, ja)
        f.close()
        return ja

    def get_fldr(self):
        """
        Get the fldr array

        Returns
        -------
        fldr : ndarray
            A 1D vector indicating the direction of the connection 1, 2, and 3
            are plus x, y, and z directions.  -1, -2, and -3 are negative
            x, y, and z directions.

        """
        iac = self.get_iac()
        njag = iac.sum()
        fldr = np.empty((njag), dtype=np.int)
        fname = os.path.join(self.model_ws, "qtg.fldr.dat")
        f = open(fname, "r")
        fldr = read1d(f, fldr)
        f.close()
        return fldr

    def get_ivc(self, fldr=None):
        """
        Get the MODFLOW-USG ivc array

        Parameters
        ----------
        fldr : ndarray
            Flow direction indicator array.  If None, then it is read from
            gridgen output.

        Returns
        -------
        ivc : ndarray
            A 1D vector indicating the direction of the connection where 1 is
            vertical and 0 is horizontal.

        """
        if fldr is None:
            fldr = self.get_fldr()
        ivc = np.zeros(fldr.shape, dtype=np.int)
        idx = abs(fldr) == 3
        ivc[idx] = 1
        return ivc

    def get_ihc(self, fldr=None):
        """
        Get the ihc array

        Parameters
        ----------
        fldr : ndarray
            Flow direction indicator array.  If None, then it is read from
            gridgen output.

        Returns
        -------
        ihc : ndarray
            A 1D vector indicating the direction of the connection where
            0 is vertical, 1 is a regular horizontal connection and 2 is a
            vertically staggered horizontal connection.

        """
        if fldr is None:
            fldr = self.get_fldr()
        ihc = np.empty(fldr.shape, dtype=np.int)
        ihc = np.where(abs(fldr) == 0, 0, ihc)
        ihc = np.where(abs(fldr) == 1, 1, ihc)
        ihc = np.where(abs(fldr) == 2, 1, ihc)
        ihc = np.where(abs(fldr) == 3, 0, ihc)
        return ihc

    def get_cl12(self):
        """
        Get the cl12 array

        Returns
        -------
        cl12 : ndarray
            A 1D vector of the cell connection distances, which are from the
            center of cell n to its shared face will cell m

        """
        iac = self.get_iac()
        njag = iac.sum()
        cl12 = np.empty((njag), dtype=np.float32)
        fname = os.path.join(self.model_ws, "qtg.c1.dat")
        f = open(fname, "r")
        cl12 = read1d(f, cl12)
        f.close()
        return cl12

    def get_fahl(self):
        """
        Get the fahl array

        Returns
        -------
        fahl : ndarray
            A 1D vector of the cell connection information, which is flow area
            for a vertical connection and horizontal length for a horizontal
            connection

        """
        iac = self.get_iac()
        njag = iac.sum()
        fahl = np.empty((njag), dtype=np.float32)
        fname = os.path.join(self.model_ws, "qtg.fahl.dat")
        f = open(fname, "r")
        fahl = read1d(f, fahl)
        f.close()
        return fahl

    def get_hwva(self, ja=None, ihc=None, fahl=None, top=None, bot=None):
        """
        Get the hwva array

        Parameters
        ----------
        ja : ndarray
            Cell connectivity.  If None, it will be read from gridgen output.
        ihc : ndarray
            Connection horizontal indicator array.  If None it will be read
            and calculated from gridgen output.
        fahl : ndarray
            Flow area, horizontal width array required by MODFLOW-USG.  If none
            then it will be read from the gridgen output.  Default is None.
        top : ndarray
            Cell top elevation.  If None, it will be read from gridgen output.
        bot : ndarray
            Cell bottom elevation.  If None, it will be read from gridgen
            output.

        Returns
        -------
        fahl : ndarray
            A 1D vector of the cell connection information, which is flow area
            for a vertical connection and horizontal length for a horizontal
            connection

        """
        iac = self.get_iac()
        nodes = iac.shape[0]

        if ja is None:
            ja = self.get_ja()
        if ihc is None:
            ihc = self.get_ihc()
        if fahl is None:
            fahl = self.get_fahl()
        if top is None:
            top = self.get_top()
        if bot is None:
            bot = self.get_bot()

        hwva = fahl.copy()
        ipos = 0
        for n in range(nodes):
            for j in range(iac[n]):
                if j == 0:
                    pass
                elif ihc[ipos] == 0:
                    pass
                else:
                    m = ja[ipos] - 1
                    dzn = top[n] - bot[n]
                    dzm = top[m] - bot[m]
                    dzavg = 0.5 * (dzn + dzm)
                    hwva[ipos] = hwva[ipos] / dzavg
                ipos += 1
        return hwva

    def get_angldegx(self, fldr=None):
        """
        Get the angldegx array

        Parameters
        ----------
        fldr : ndarray
            Flow direction indicator array.  If None, then it is read from
            gridgen output.

        Returns
        -------
        angldegx : ndarray
            A 1D vector indicating the angle (in degrees) between the x
            axis and an outward normal to the face.

        """
        if fldr is None:
            fldr = self.get_fldr()
        angldegx = np.zeros(fldr.shape, dtype=np.float)
        angldegx = np.where(fldr == 0, 1.0e30, angldegx)
        angldegx = np.where(abs(fldr) == 3, 1.0e30, angldegx)
        angldegx = np.where(fldr == 2, 90, angldegx)
        angldegx = np.where(fldr == -1, 180, angldegx)
        angldegx = np.where(fldr == -2, 270, angldegx)
        return angldegx

    def get_verts_iverts(self, ncells, verbose=False):
        """
        Return a 2d array of x and y vertices and a list of size ncells that
        has the list of vertices for each cell.

        Parameters
        ----------
        ncells : int
            The number of entries in iverts.  This should be ncpl for a layered
            model and nodes for a disu model.
        verbose : bool
            Print information as its working

        Returns
        -------
        verts, iverts : tuple
            verts is a 2d array of x and y vertex pairs (nvert, 2) and iverts
            is a list of vertices that comprise each cell

        """
        from .cvfdutil import to_cvfd

        verts, iverts = to_cvfd(
            self._vertdict, nodestop=ncells, verbose=verbose
        )
        return verts, iverts

    def get_cellxy(self, ncells):
        """

        Parameters
        ----------
        ncells : int
            Number of cells for which to create the list of cell centers

        Returns
        -------
        cellxy : ndarray
            x and y cell centers.  Shape is (ncells, 2)

        """
        cellxy = np.empty((ncells, 2), dtype=np.float)
        for n in range(ncells):
            x, y = self.get_center(n)
            cellxy[n, 0] = x
            cellxy[n, 1] = y
        return cellxy

    def get_gridprops(self):
        """
        Get a dictionary of information needed to create a MODFLOW-USG DISU
        Package

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nlay = self.get_nlay()
        nodes = self.get_nodes()
        nodelay = self.get_nodelay()

        gridprops["nodes"] = nodes
        gridprops["nlay"] = nlay
        gridprops["nodelay"] = nodelay

        # top
        top = self.get_top()
        gridprops["top"] = top

        # bot
        bot = self.get_bot()
        gridprops["bot"] = bot

        # area
        area = self.get_area()
        gridprops["area"] = area

        # iac
        iac = self.get_iac()
        gridprops["iac"] = iac

        # Calculate njag and save as nja to self
        njag = iac.sum()
        gridprops["nja"] = njag

        # ja
        ja = self.get_ja(njag)
        gridprops["ja"] = ja

        # fldr
        fldr = self.get_fldr()
        gridprops["fldr"] = fldr

        # ivc
        ivc = self.get_ivc(fldr=fldr)
        gridprops["ivc"] = ivc

        cl1 = None
        cl2 = None
        # cl12
        cl12 = self.get_cl12()
        gridprops["cl12"] = cl12

        # fahl
        fahl = self.get_fahl()
        gridprops["fahl"] = fahl

        return gridprops

    def get_gridprops_disu6(self):
        """
        Return a dictionary containing all of the information required to
        create a MODFLOW 6 DISU Package

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nodes = self.get_nodes()
        gridprops["nodes"] = nodes

        # top
        top = self.get_top()
        gridprops["top"] = top

        # bot
        bot = self.get_bot()
        gridprops["bot"] = bot

        # area
        area = self.get_area()
        gridprops["area"] = area

        # iac
        iac = self.get_iac()
        gridprops["iac"] = iac

        # Calculate njag and save as nja to self
        njag = iac.sum()
        gridprops["nja"] = njag

        # ja
        ja = self.get_ja(njag)
        gridprops["ja"] = ja

        # cl12
        cl12 = self.get_cl12()
        gridprops["cl12"] = cl12

        # fldr
        fldr = self.get_fldr()

        # ihc
        ihc = self.get_ihc(fldr)
        gridprops["ihc"] = ihc

        # hwva
        hwva = self.get_hwva(ja=ja, ihc=ihc, fahl=None, top=top, bot=bot)
        gridprops["hwva"] = hwva

        # angldegx
        angldegx = self.get_angldegx(fldr)
        gridprops["angldegx"] = angldegx

        # vertices -- not optimized for redundant vertices yet
        nvert = nodes * 4
        vertices = []
        ivert = 0
        for n in range(nodes):
            vs = self.get_vertices(n)
            for x, y in vs[:-1]:  # do not include last vertex
                vertices.append([ivert, x, y])
                ivert += 1
        gridprops["nvert"] = nvert
        gridprops["vertices"] = vertices

        # cell2d information
        cell2d = []
        iv = 1
        for n in range(nodes):
            xc, yc = self.get_center(n)
            cell2d.append([n, xc, yc, 4, iv, iv + 1, iv + 2, iv + 3])
            iv += 4
        gridprops["cell2d"] = cell2d

        return gridprops

    def to_disu6(self, fname, writevertices=True):
        """
        Create a MODFLOW 6 DISU file

        Parameters
        ----------
        fname : str
            name of file to write
        writevertices : bool
            include vertices in the DISU file. (default is True)

        Returns
        -------

        """

        gridprops = self.get_gridprops_disu6()
        f = open(fname, "w")

        # opts
        f.write("BEGIN OPTIONS\n")
        f.write("END OPTIONS\n\n")

        # dims
        f.write("BEGIN DIMENSIONS\n")
        f.write("  NODES {}\n".format(gridprops["nodes"]))
        f.write("  NJA {}\n".format(gridprops["nja"]))
        if writevertices:
            f.write("  NVERT {}\n".format(gridprops["nvert"]))
        f.write("END DIMENSIONS\n\n")

        # griddata
        f.write("BEGIN GRIDDATA\n")
        for prop in ["top", "bot", "area"]:
            f.write("  {}\n".format(prop.upper()))
            f.write("    INTERNAL\n")
            a = gridprops[prop]
            for aval in a:
                f.write("{} ".format(aval))
            f.write("\n")
        f.write("END GRIDDATA\n\n")

        # condata
        f.write("BEGIN CONNECTIONDATA\n")
        for prop in ["iac", "ja", "ihc", "cl12", "hwva", "angldegx"]:
            f.write("  {}\n".format(prop.upper()))
            f.write("    INTERNAL\n")
            a = gridprops[prop]
            for aval in a:
                f.write("{} ".format(aval))
            f.write("\n")
        f.write("END CONNECTIONDATA\n\n")

        if writevertices:
            # vertices -- not optimized for redundant vertices yet
            f.write("BEGIN VERTICES\n")
            vertices = gridprops["vertices"]
            for i, row in enumerate(vertices):
                x = row[0]
                y = row[1]
                s = "  {} {} {}\n".format(i + 1, x, y)
                f.write(s)
            f.write("END VERTICES\n\n")

            # celldata -- not optimized for redundant vertices yet
            f.write("BEGIN CELL2D\n")
            iv = 1
            for n in range(gridprops["nodes"]):
                xc, yc = self.get_center(n)
                s = "  {} {} {} {} {} {} {} {}\n".format(
                    n + 1, xc, yc, 4, iv, iv + 1, iv + 2, iv + 3
                )
                f.write(s)
                iv += 4
            f.write("END CELL2D\n\n")

        f.close()
        return

    def get_gridprops_disv(self, verbose=False):
        gridprops = {}

        nlay = self.get_nlay()
        nodelay = self.get_nodelay()
        ncpl = nodelay.min()
        assert ncpl == nodelay.max(), "Cannot create DISV properties "
        "because the number of cells is not the same for all layers"

        gridprops["nlay"] = nlay
        gridprops["ncpl"] = ncpl

        # top
        top = np.empty(ncpl, dtype=np.float32)
        k = 0
        fname = os.path.join(
            self.model_ws, "quadtreegrid.top{}.dat".format(k + 1)
        )
        f = open(fname, "r")
        top = read1d(f, top)
        f.close()
        gridprops["top"] = top

        # botm
        botm = []
        istart = 0
        for k in range(nlay):
            istop = istart + nodelay[k]
            fname = os.path.join(
                self.model_ws, "quadtreegrid.bot{}.dat".format(k + 1)
            )
            f = open(fname, "r")
            btk = np.empty((nodelay[k]), dtype=np.float32)
            btk = read1d(f, btk)
            f.close()
            botm.append(btk)
            istart = istop
        gridprops["botm"] = botm

        # cell xy locations
        cellxy = self.get_cellxy(ncpl)

        # verts and iverts
        verts, iverts = self.get_verts_iverts(ncpl)

        nvert = verts.shape[0]
        vertices = [[i, verts[i, 0], verts[i, 1]] for i in range(nvert)]
        gridprops["nvert"] = nvert
        gridprops["vertices"] = vertices

        # cell2d information
        cell2d = [
            [n, cellxy[n, 0], cellxy[n, 1], len(ivs)] + ivs
            for n, ivs in enumerate(iverts)
        ]
        gridprops["cell2d"] = cell2d

        return gridprops

    def to_disv6(self, fname, verbose=False):
        """
        Create a MODFLOW 6 DISV file

        Parameters
        ----------
        fname : str
            name of file to write

        Returns
        -------

        """

        if verbose:
            print("Loading properties from gridgen output.")
        gridprops = self.get_gridprops()
        f = open(fname, "w")

        # determine sizes
        nlay = gridprops["nlay"]
        nodelay = gridprops["nodelay"]
        ncpl = nodelay.min()
        assert ncpl == nodelay.max(), "Cannot create DISV package "
        "because the number of cells is not the same for all layers"

        # use the cvfdutil helper to eliminate redundant vertices and add
        # hanging nodes
        from .cvfdutil import to_cvfd

        verts, iverts = to_cvfd(self._vertdict, nodestop=ncpl, verbose=verbose)
        nvert = verts.shape[0]

        # opts
        if verbose:
            print("writing options.")
        f.write("BEGIN OPTIONS\n")
        f.write("END OPTIONS\n\n")

        # dims
        if verbose:
            print("writing dimensions.")
        f.write("BEGIN DIMENSIONS\n")
        f.write("  NCPL {}\n".format(ncpl))
        f.write("  NLAY {}\n".format(nlay))
        f.write("  NVERT {}\n".format(nvert))
        f.write("END DIMENSIONS\n\n")

        # griddata
        if verbose:
            print("writing griddata.")
        f.write("BEGIN GRIDDATA\n")
        for prop in ["top", "bot"]:
            a = gridprops[prop]
            if prop == "bot":
                prop = "botm"
            f.write("  {}\n".format(prop.upper()))
            f.write("    INTERNAL\n")
            if prop == "top":
                a = a[0:ncpl]
            for aval in a:
                f.write("{} ".format(aval))
            f.write("\n")
        f.write("END GRIDDATA\n\n")

        # vertices
        if verbose:
            print("writing vertices.")
        f.write("BEGIN VERTICES\n")
        for i, row in enumerate(verts):
            x = row[0]
            y = row[1]
            s = "  {} {} {}\n".format(i + 1, x, y)
            f.write(s)
        f.write("END VERTICES\n\n")

        # celldata
        if verbose:
            print("writing cell2d.")
        f.write("BEGIN CELL2D\n")
        for icell, icellverts in enumerate(iverts):
            xc, yc = self.get_center(icell)
            s = "  {} {} {} {}".format(icell + 1, xc, yc, len(icellverts))
            for iv in icellverts:
                s += " {}".format(iv + 1)
            f.write(s + "\n")
        f.write("END CELL2D\n\n")

        if verbose:
            print("done writing disv.")
        f.close()
        return

    def intersect(self, features, featuretype, layer):
        """
        Parameters
        ----------
        features : str or list
            features can be either a string containing the name of a shapefile
            or it can be a list of points, lines, or polygons
        featuretype : str
            Must be either 'point', 'line', or 'polygon'
        layer : int
            Layer (zero based) to intersect with.  Zero based.

        Returns
        -------
        result : np.recarray
            Recarray of the intersection properties.

        """
        ifname = "intersect_feature"
        if isinstance(features, list):
            ifname_w_path = os.path.join(self.model_ws, ifname)
            if os.path.exists(ifname_w_path + ".shp"):
                os.remove(ifname_w_path + ".shp")
            features_to_shapefile(features, featuretype, ifname_w_path)
            shapefile = ifname
        else:
            shapefile = features

        sn = os.path.join(self.model_ws, shapefile + ".shp")
        assert os.path.isfile(sn), "Shapefile does not exist: {}".format(sn)

        fname = os.path.join(self.model_ws, "_intersect.dfn")
        if os.path.isfile(fname):
            os.remove(fname)
        f = open(fname, "w")
        f.write("LOAD quadtreegrid.dfn\n")
        f.write(1 * "\n")
        f.write(self._intersection_block(shapefile, featuretype, layer))
        f.close()

        # Intersect
        cmds = [self.exe_name, "intersect", "_intersect.dfn"]
        buff = []
        fn = os.path.join(self.model_ws, "intersection.ifo")
        if os.path.isfile(fn):
            os.remove(fn)
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
        except:
            print("Error.  Failed to perform intersection", buff)

        # Make sure new intersection file was created.
        if not os.path.isfile(fn):
            s = ("Error.  Failed to perform intersection", buff)
            raise Exception(s)

        # Calculate the number of columns to import
        # The extra comma causes one too many columns, so calculate the length
        f = open(fn, "r")
        line = f.readline()
        f.close()
        ncol = len(line.strip().split(",")) - 1

        # Load the intersection results as a recarray, convert nodenumber
        # to zero-based and return
        result = np.genfromtxt(
            fn,
            dtype=None,
            names=True,
            delimiter=",",
            usecols=tuple(range(ncol)),
        )
        result = np.atleast_1d(result)
        result = result.view(np.recarray)
        result["nodenumber"] -= 1
        return result

    def _intersection_block(self, shapefile, featuretype, layer):
        s = ""
        s += "BEGIN GRID_INTERSECTION intersect" + "\n"
        s += "  GRID = quadtreegrid\n"
        s += "  LAYER = {}\n".format(layer + 1)
        s += "  SHAPEFILE = {}\n".format(shapefile)
        s += "  FEATURE_TYPE = {}\n".format(featuretype)
        s += "  OUTPUT_FILE = {}\n".format("intersection.ifo")
        s += "END GRID_INTERSECTION intersect" + "\n"
        return s

    def _mfgrid_block(self):
        # flopy modelgrid object uses same xoff, yoff, and rotation convention
        # as gridgen

        xoff = self.modelgrid.xoffset
        yoff = self.modelgrid.yoffset
        angrot = self.modelgrid.angrot

        s = ""
        s += "BEGIN MODFLOW_GRID basegrid" + "\n"
        s += "  ROTATION_ANGLE = {}\n".format(angrot)
        s += "  X_OFFSET = {}\n".format(xoff)
        s += "  Y_OFFSET = {}\n".format(yoff)
        s += "  NLAY = {}\n".format(self.nlay)
        s += "  NROW = {}\n".format(self.nrow)
        s += "  NCOL = {}\n".format(self.ncol)

        # delr
        delr = self.dis.delr.array
        if delr.min() == delr.max():
            s += "  DELR = CONSTANT {}\n".format(delr.min())
        else:
            s += "  DELR = OPEN/CLOSE delr.dat\n"
            fname = os.path.join(self.model_ws, "delr.dat")
            np.savetxt(fname, np.atleast_2d(delr))

        # delc
        delc = self.dis.delc.array
        if delc.min() == delc.max():
            s += "  DELC = CONSTANT {}\n".format(delc.min())
        else:
            s += "  DELC = OPEN/CLOSE delc.dat\n"
            fname = os.path.join(self.model_ws, "delc.dat")
            np.savetxt(fname, np.atleast_2d(delc))

        # top
        top = self.dis.top.array
        if top.min() == top.max():
            s += "  TOP = CONSTANT {}\n".format(top.min())
        else:
            s += "  TOP = OPEN/CLOSE top.dat\n"
            fname = os.path.join(self.model_ws, "top.dat")
            np.savetxt(fname, top)

        # bot
        botm = self.dis.botm.array
        for k in range(self.nlay):
            if isinstance(self.dis, ModflowGwfdis):
                bot = botm[k]
            else:
                bot = botm[k]
            if bot.min() == bot.max():
                s += "  BOTTOM LAYER {} = CONSTANT {}\n".format(
                    k + 1, bot.min()
                )
            else:
                s += "  BOTTOM LAYER {0} = OPEN/CLOSE bot{0}.dat\n".format(
                    k + 1
                )
                fname = os.path.join(self.model_ws, "bot{}.dat".format(k + 1))
                np.savetxt(fname, bot)

        s += "END MODFLOW_GRID" + "\n"
        return s

    def _rf_blocks(self):
        s = ""
        for rfname, rf in self._rfdict.items():
            shapefile, featuretype, level = rf
            s += "BEGIN REFINEMENT_FEATURES {}\n".format(rfname)
            s += "  SHAPEFILE = {}\n".format(shapefile)
            s += "  FEATURE_TYPE = {}\n".format(featuretype)
            s += "  REFINEMENT_LEVEL = {}\n".format(level)
            s += "END REFINEMENT_FEATURES\n"
            s += 2 * "\n"
        return s

    def _ad_blocks(self):
        s = ""
        for adname, shapefile in self._addict.items():
            s += "BEGIN ACTIVE_DOMAIN {}\n".format(adname)
            s += "  SHAPEFILE = {}\n".format(shapefile)
            s += "  FEATURE_TYPE = {}\n".format("polygon")
            s += "  INCLUDE_BOUNDARY = {}\n".format("True")
            s += "END ACTIVE_DOMAIN\n"
            s += 2 * "\n"
        return s

    def _builder_block(self):
        s = "BEGIN QUADTREE_BUILDER quadtreebuilder\n"
        s += "  MODFLOW_GRID = basegrid\n"

        # Write active domain information
        for k, adk in enumerate(self._active_domain):
            if adk is None:
                continue
            s += "  ACTIVE_DOMAIN LAYER {} = {}\n".format(k + 1, adk)

        # Write refinement feature information
        for k, rfkl in enumerate(self._refinement_features):
            if len(rfkl) == 0:
                continue
            s += "  REFINEMENT_FEATURES LAYER {} = ".format(k + 1)
            for rf in rfkl:
                s += rf + " "
            s += "\n"

        s += "  SMOOTHING = full\n"

        for k in range(self.nlay):
            if self.surface_interpolation[k] == "ASCIIGRID":
                grd = self._asciigrid_dict[k]
            else:
                grd = "basename"
            s += "  TOP LAYER {} = {} {}\n".format(
                k + 1, self.surface_interpolation[k], grd
            )

        for k in range(self.nlay):
            if self.surface_interpolation[k + 1] == "ASCIIGRID":
                grd = self._asciigrid_dict[k + 1]
            else:
                grd = "basename"
            s += "  BOTTOM LAYER {} = {} {}\n".format(
                k + 1, self.surface_interpolation[k + 1], grd
            )

        s += "  GRID_DEFINITION_FILE = quadtreegrid.dfn\n"
        s += "END QUADTREE_BUILDER\n"
        return s

    def _grid_export_blocks(self):
        s = "BEGIN GRID_TO_SHAPEFILE grid_to_shapefile_poly\n"
        s += "  GRID = quadtreegrid\n"
        s += "  SHAPEFILE = qtgrid\n"
        s += "  FEATURE_TYPE = polygon\n"
        s += "END GRID_TO_SHAPEFILE\n"
        s += "\n"
        s += "BEGIN GRID_TO_SHAPEFILE grid_to_shapefile_point\n"
        s += "  GRID = quadtreegrid\n"
        s += "  SHAPEFILE = qtgrid_pt\n"
        s += "  FEATURE_TYPE = point\n"
        s += "END GRID_TO_SHAPEFILE\n"
        s += "\n"
        s += "BEGIN GRID_TO_USGDATA grid_to_usgdata\n"
        s += "  GRID = quadtreegrid\n"
        s += "  USG_DATA_PREFIX = qtg\n"
        s += "  VERTICAL_PASS_THROUGH = {0}\n".format(
            self.vertical_pass_through
        )
        s += "END GRID_TO_USGDATA\n"
        s += "\n"
        s += "BEGIN GRID_TO_VTKFILE grid_to_vtk\n"
        s += "  GRID = quadtreegrid\n"
        s += "  VTKFILE = qtg\n"
        s += "  SHARE_VERTEX = False\n"
        s += "END GRID_TO_VTKFILE\n"
        s += "\n"
        s += "BEGIN GRID_TO_VTKFILE grid_to_vtk_sv\n"
        s += "  GRID = quadtreegrid\n"
        s += "  VTKFILE = qtg_sv\n"
        s += "  SHARE_VERTEX = True\n"
        s += "END GRID_TO_VTKFILE\n"
        return s

    def _mkvertdict(self):
        """
        Create the self._vertdict dictionary that maps the nodenumber to
        the vertices

        Returns
        -------
        None

        """
        # ensure there are active leaf cells from gridgen
        fname = os.path.join(self.model_ws, "qtg.nod")
        if not os.path.isfile(fname):
            raise Exception(
                "File {} should have been created by gridgen.".format(fname)
            )
        f = open(fname, "r")
        line = f.readline()
        ll = line.strip().split()
        nodes = int(ll[0])
        if nodes == 0:
            raise Exception("Gridgen resulted in no active cells.")

        # ensure shape file was created by gridgen
        fname = os.path.join(self.model_ws, "qtgrid.shp")
        assert os.path.isfile(fname), "gridgen shape file does not exist"

        # read vertices from shapefile
        sf = shapefile.Reader(fname)
        shapes = sf.shapes()
        fields = sf.fields
        attributes = [l[0] for l in fields[1:]]
        records = sf.records()
        idx = attributes.index("nodenumber")
        for i in range(len(shapes)):
            nodenumber = int(records[i][idx]) - 1
            self._vertdict[nodenumber] = shapes[i].points
        return
