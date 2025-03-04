import os
import subprocess
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from ..discretization import StructuredGrid
from ..export.shapefile_utils import shp2recarray
from ..mbase import resolve_exe
from ..mf6.modflow import ModflowGwfdis
from ..mfusg.mfusgdisu import MfUsgDisU
from ..modflow import ModflowDis
from ..utils import import_optional_dependency
from ..utils.flopy_io import relpath_safe
from .util_array import Util2d

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


def features_to_shapefile(features, featuretype, filename: Union[str, os.PathLike]):
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
    filename : str or PathLike
        The shapefile to write (extension is optional)

    Returns
    -------
    None

    """
    from .geospatial_utils import GeoSpatialCollection

    shapefile = import_optional_dependency("shapefile")

    if featuretype.lower() == "line":
        featuretype = "LineString"

    features = GeoSpatialCollection(features, featuretype).flopy_geometry

    if featuretype.lower() not in ["point", "line", "linestring", "polygon"]:
        raise ValueError(f"Unrecognized feature type: {featuretype}")

    if featuretype.lower() in ("line", "linestring"):
        wr = shapefile.Writer(str(filename), shapeType=shapefile.POLYLINE)
        wr.field("SHAPEID", "N", 20, 0)
        for i, line in enumerate(features):
            wr.line([line.__geo_interface__["coordinates"]])
            wr.record(i)

    elif featuretype.lower() == "point":
        wr = shapefile.Writer(str(filename), shapeType=shapefile.POINT)
        wr.field("SHAPEID", "N", 20, 0)
        for i, point in enumerate(features):
            wr.point(*point.__geo_interface__["coordinates"])
            wr.record(i)

    elif featuretype.lower() == "polygon":
        wr = shapefile.Writer(str(filename), shapeType=shapefile.POLYGON)
        wr.field("SHAPEID", "N", 20, 0)
        for i, polygon in enumerate(features):
            wr.poly(polygon.__geo_interface__["coordinates"])
            wr.record(i)

    wr.close()


def ndarray_to_asciigrid(fname: Union[str, os.PathLike], a, extent, nodata=1.0e30):
    # extent info
    xmin, xmax, ymin, ymax = extent
    ncol, nrow = a.shape
    dx = (xmax - xmin) / ncol
    assert dx == (ymax - ymin) / nrow
    # header
    header = f"ncols     {ncol}\n"
    header += f"nrows    {nrow}\n"
    header += f"xllcorner {xmin}\n"
    header += f"yllcorner {ymin}\n"
    header += f"cellsize {dx}\n"
    header += f"NODATA_value {float(nodata)}\n"
    # replace nan with nodata
    idx = np.isnan(a)
    a[idx] = float(nodata)
    # write
    with open(fname, "wb") as f:
        f.write(header.encode("ascii"))
        np.savetxt(f, a, fmt="%15.6e")


def get_ia_from_iac(iac):
    ia = [0]
    for ncon in iac:
        ia.append(ia[-1] + ncon)
    ia = np.array(ia)
    return ia


def get_isym(ia, ja):
    isym = -1 * np.zeros(ja.shape, ja.dtype)
    for n in range(ia.shape[0] - 1):
        for ii in range(ia[n], ia[n + 1]):
            m = ja[ii]
            if m != n:
                isym[ii] = 0
                for jj in range(ia[m], ia[m + 1]):
                    if ja[jj] == n:
                        isym[ii] = jj
                        break
            else:
                isym[ii] = ii
    return isym


def is_symmetrical(isym, a, atol=0):
    assert isym.shape == a.shape
    for ipos, val in enumerate(a):
        isympos = isym[ipos]
        diff = val - a[isympos]
        if not np.allclose(diff, 0, atol=atol):
            return False
    return True


def repair_array_asymmetry(isym, a, atol=0):
    assert isym.shape == a.shape
    for ipos, val in enumerate(a):
        isympos = isym[ipos]
        diff = val - a[isympos]
        if not np.allclose(diff, 0, atol=atol):
            a[isympos] = val
    return a


class Gridgen:
    """
    Class to work with the gridgen program to create layered quadtree grids.

    Parameters
    ----------
    modelgrid : flopy.discretization.StructuredGrid
        Flopy StructuredGrid object. Note this also accepts ModflowDis and
        ModflowGwfdis objects, however it is deprecated and support will be
        removed in version 3.3.7
    model_ws : str or PathLike
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
    **kwargs
        smoothing_level_vertical : int
            maximum level difference between two vertically adjacent cells.
            Adjust with caution, as adjustments can cause unexpected results
            to simulated flows
        smoothing_level_horizontal : int
            maximum level difference between two horizontally adjacent cells.
            Adjust with caution, as adjustments can cause unexpected results
            to simulated flows

    Notes
    -----
    For the surface elevations, the top of a layer uses the same surface as
    the bottom of the overlying layer.

    """

    def __init__(
        self,
        modelgrid,
        model_ws: Union[str, os.PathLike] = os.curdir,
        exe_name: Union[str, os.PathLike] = "gridgen",
        surface_interpolation="replicate",
        vertical_pass_through=False,
        **kwargs,
    ):
        if isinstance(modelgrid, StructuredGrid):
            if modelgrid.top is None or modelgrid.botm is None:
                raise AssertionError(
                    "A complete modelgrid must be supplied to use Gridgen"
                )

            self.modelgrid = modelgrid

        elif isinstance(modelgrid, (ModflowGwfdis, ModflowDis)):
            warnings.warn(
                "Supplying a dis object is deprecated, and support will be "
                "removed in version 3.3.7. Please supply StructuredGrid."
            )
            # this is actually a DIS file
            self.modelgrid = modelgrid.parent.modelgrid

        else:
            raise TypeError("A StructuredGrid object must be supplied to Gridgen")

        self.nlay = self.modelgrid.nlay
        self.nrow = self.modelgrid.nrow
        self.ncol = self.modelgrid.ncol

        self.nodes = 0
        self.nja = 0
        self.nodelay = np.zeros((self.nlay,), dtype=int)
        self._vertdict = {}
        self.model_ws = Path(model_ws).expanduser().absolute()
        self.exe_name = resolve_exe(exe_name)

        # Set default surface interpolation for all surfaces (nlay + 1)
        surface_interpolation = surface_interpolation.upper()
        if surface_interpolation not in ["INTERPOLATE", "REPLICATE"]:
            raise ValueError(
                f"Unknown surface interpolation method {surface_interpolation}, "
                "expected 'INTERPOLATE' or 'REPLICATE'"
            )
        self.surface_interpolation = [
            surface_interpolation for k in range(self.nlay + 1)
        ]

        # Set export options
        self.vertical_pass_through = "False"
        if vertical_pass_through:
            self.vertical_pass_through = "True"

        self.smoothing_level_vertical = kwargs.pop("smoothing_level_vertical", 1)
        self.smoothing_level_horizontal = kwargs.pop("smoothing_level_horizontal", 1)
        # Set up a blank _active_domain list with None for each layer
        self._addict = {}
        self._active_domain = []
        for k in range(self.nlay):
            self._active_domain.append(None)

        # Set up a blank _refinement_features list with empty list for
        # each layer
        self._rfdict = {}
        self._refinement_features = [[] for _ in range(self.nlay)]

        # Set up blank _elev and _elev_extent dictionaries
        self._asciigrid_dict = {}

    def set_surface_interpolation(self, isurf, type, elev=None, elev_extent=None):
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
            raise ValueError(
                "Unknown surface interpolation type "
                f"{type}, expected 'INTERPOLATE',"
                "'REPLICATE', or 'ASCIIGRID'"
            )
        else:
            self.surface_interpolation[isurf] = type

        if type == "ASCIIGRID":
            if isinstance(elev, np.ndarray):
                if elev_extent is None:
                    raise ValueError("ASCIIGRID was specified but elev_extent was not.")
                try:
                    xmin, xmax, ymin, ymax = elev_extent
                except:
                    raise ValueError(
                        "Cannot unpack elev_extent as tuple (xmin, xmax, ymin, ymax): "
                        f"{elev_extent}"
                    )

                nm = f"_gridgen.lay{isurf}.asc"
                fname = self.model_ws / nm
                ndarray_to_asciigrid(fname, elev, elev_extent)
                self._asciigrid_dict[isurf] = nm

            elif isinstance(elev, str):
                if not os.path.isfile(os.path.join(self.model_ws, elev)):
                    raise ValueError(
                        f"Elevation file not found: {os.path.join(self.model_ws, elev)}"
                    )
                self._asciigrid_dict[isurf] = elev
            else:
                raise ValueError(
                    "ASCIIGRID was specified but elevation was not provided "
                    "as a numpy ndarray or asciigrid file."
                )

    def resolve_shapefile_path(self, p):
        def _resolve(p):
            # try expanding absolute path
            path = Path(p).expanduser().absolute()
            # try looking in workspace
            return path if path.is_file() else self.model_ws / p

        path = _resolve(p)
        path = path if path.is_file() else _resolve(Path(p).with_suffix(".shp"))
        return path if path.is_file() else None

    def add_active_domain(self, feature, layers):
        """
        Parameters
        ----------
        feature : str, path-like or array-like
            feature can be:
                a shapefile name (str) or Pathlike
                a list of polygons
                a flopy.utils.geometry.Collection object of Polygons
                a shapely.geometry.Collection object of Polygons
                a geojson.GeometryCollection object of Polygons
                a list of shapefile.Shape objects
                a shapefile.Shapes object
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

        # expand shapefile path or create one from polygon feature
        if isinstance(feature, (str, os.PathLike)):
            shapefile_path = self.resolve_shapefile_path(feature)
        elif isinstance(feature, (list, tuple, np.ndarray)):
            shapefile_path = self.model_ws / f"ad{len(self._addict)}.shp"
            features_to_shapefile(feature, "polygon", shapefile_path)
        else:
            raise ValueError(
                "Feature must be a pathlike (shapefile) or array-like of geometries"
            )

        # make sure shapefile exists
        assert shapefile_path and shapefile_path.is_file(), (
            f"Shapefile does not exist: {shapefile_path}"
        )

        # store shapefile info
        self._addict[shapefile_path.stem] = relpath_safe(shapefile_path, self.model_ws)
        for k in layers:
            self._active_domain[k] = shapefile_path.stem

    def add_refinement_features(self, features, featuretype, level, layers):
        """
        Parameters
        ----------
        features : str, path-like or array-like
            features can be
                a shapefile name (str) or Pathlike
                a list of points, lines, or polygons
                a flopy.utils.geometry.Collection object
                a list of flopy.utils.geometry objects
                a shapely.geometry.Collection object
                a geojson.GeometryCollection object
                a list of shapefile.Shape objects
                a shapefile.Shapes object
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

        # Create shapefile or set shapefile to feature
        if isinstance(features, (str, os.PathLike)):
            shapefile_path = self.resolve_shapefile_path(features)
        elif isinstance(features, (list, tuple, np.ndarray)):
            shapefile_path = self.model_ws / f"rf{len(self._rfdict)}.shp"
            features_to_shapefile(features, featuretype, shapefile_path)
        else:
            raise ValueError(
                "Features must be a pathlike (shapefile) or array-like of geometries"
            )

        # make sure shapefile exists
        assert shapefile_path and shapefile_path.is_file(), (
            f"Shapefile does not exist: {shapefile_path}"
        )

        # store shapefile info
        self._rfdict[shapefile_path.stem] = [
            relpath_safe(shapefile_path, self.model_ws),
            featuretype,
            level,
        ]
        for k in layers:
            self._refinement_features[k].append(shapefile_path.stem)

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
        self.nodelay = self.read_qtg_nodesperlay_dat(
            model_ws=self.model_ws, nlay=self.nlay
        )

        # Create a recarray of the grid polygon shapefile
        shapename = os.path.join(self.model_ws, "qtgrid")
        self.qtra = shp2recarray(shapename)

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
        assert os.path.isfile(fname), f"Could not create export dfn file: {fname}"

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

        cmds = [self.exe_name, "grid_to_shapefile_point", "_gridgen_export.dfn"]
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
        pc : matplotlib.collections.PatchCollection

        """
        import matplotlib.pyplot as plt

        from ..plot import plot_shapefile, shapefile_extents

        if ax is None:
            ax = plt.gca()
        shapename = os.path.join(self.model_ws, "qtgrid")
        xmin, xmax, ymin, ymax = shapefile_extents(shapename)

        idx = np.asarray(self.qtra.layer == layer).nonzero()[0]

        pc = plot_shapefile(
            shapename,
            ax=ax,
            edgecolor=edgecolor,
            facecolor=facecolor,
            cmap=cmap,
            a=a,
            masked_values=masked_values,
            idx=idx,
            **kwargs,
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
        node_ra = self.read_qtg_nod(model_ws=self.model_ws, nodes_only=False)
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
        nodes = self.read_qtg_nod(model_ws=self.model_ws, nodes_only=True)
        nlay = self.nlay
        ivsd = 0
        idsymrd = 0
        laycbd = 0

        # Save nodes
        self.nodes = nodes

        # nodelay
        nodelay = self.read_qtg_nodesperlay_dat(model_ws=self.model_ws, nlay=nlay)

        # top
        top = [0] * nlay
        for k in range(nlay):
            tpk = self.read_quadtreegrid_top_dat(
                model_ws=self.model_ws, nodelay=nodelay, lay=k
            )
            if tpk.min() == tpk.max():
                tpk = tpk.min()
            else:
                tpk = Util2d(
                    model,
                    (nodelay[k],),
                    np.float32,
                    np.reshape(tpk, (nodelay[k],)),
                    name=f"top {k + 1}",
                )
            top[k] = tpk

        # bot
        bot = [0] * nlay
        for k in range(nlay):
            btk = self.read_quadtreegrid_bot_dat(
                model_ws=self.model_ws, nodelay=nodelay, lay=k
            )
            if btk.min() == btk.max():
                btk = btk.min()
            else:
                btk = Util2d(
                    model,
                    (nodelay[k],),
                    np.float32,
                    np.reshape(btk, (nodelay[k],)),
                    name=f"bot {k + 1}",
                )
            bot[k] = btk

        # area
        area = [0] * nlay
        anodes = self.read_qtg_area_dat(model_ws=self.model_ws, nodes=nodes)
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
                    name=f"area layer {k + 1}",
                )
            area[k] = ark
            istart = istop

        # iac
        iac = self.read_qtg_iac_dat(model_ws=self.model_ws, nodes=nodes)
        # Calculate njag and save as nja to self
        nja = iac.sum()
        self.nja = nja

        # ja -- this is being read is as one-based, which is also what is
        # expected by the ModflowDisu constructor
        ja = self.read_qtg_ja_dat(model_ws=self.model_ws, nja=nja)

        # ivc
        fldr = self.read_qtg_fldr_dat(model_ws=self.model_ws, nja=nja)
        ivc = np.where(abs(fldr) == 3, 1, 0)

        cl1 = None
        cl2 = None
        # cl12
        cl12 = self.read_qtg_cl_dat(model_ws=self.model_ws, nja=nja)

        # fahl
        fahl = self.read_qtg_fahl_dat(model_ws=self.model_ws, nja=nja)

        # create dis object instance
        disu = MfUsgDisU(
            model,
            nodes=nodes,
            nlay=nlay,
            njag=nja,
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

        # return disu object instance
        return disu

    def get_nodes(self):
        """
        Get the number of nodes

        Returns
        -------
        nodes : int

        """
        nodes = self.read_qtg_nod(model_ws=self.model_ws, nodes_only=True)
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
        nodelay = self.read_qtg_nodesperlay_dat(model_ws=self.model_ws, nlay=nlay)
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
            tpk = self.read_quadtreegrid_top_dat(
                model_ws=self.model_ws, nodelay=nodelay, lay=k
            )
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
            btk = self.read_quadtreegrid_bot_dat(
                model_ws=self.model_ws,
                nodelay=nodelay,
                lay=k,
            )
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
        area = self.read_qtg_area_dat(model_ws=self.model_ws, nodes=nodes)
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
        iac = self.read_qtg_iac_dat(model_ws=self.model_ws, nodes=nodes)
        return iac

    def get_ja(self, nja=None):
        """
        Get the zero-based ja array

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
        ja = self.read_qtg_ja_dat(model_ws=self.model_ws, nja=nja)
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
        nja = iac.sum()
        fldr = self.read_qtg_fldr_dat(model_ws=self.model_ws, nja=nja)
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
        ivc = np.where(abs(fldr) == 3, 1, 0)
        return ivc

    def get_ihc(self, nodelay=None, ia=None, fldr=None):
        """
        Get the ihc array

        Parameters
        ----------
        nodelay : ndarray
            Number of nodes in each layer. If None, then it is read from
            gridgen output.
        ia : ndarray
            Starting location of a row in the matrix. If None,
            then it is read from gridgen output.
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
        ihc = np.empty(fldr.shape, dtype=int)
        ihc = np.where(abs(fldr) == 0, 0, ihc)
        ihc = np.where(abs(fldr) == 1, 1, ihc)
        ihc = np.where(abs(fldr) == 2, 1, ihc)
        ihc = np.where(abs(fldr) == 3, 0, ihc)

        # fill the diagonal position of the ihc array with the layer number
        if nodelay is None:
            nodelay = self.get_nodelay()
        if ia is None:
            iac = self.get_iac()
            ia = get_ia_from_iac(iac)
        nodes = ia.shape[0] - 1
        nlayers = nodelay.shape[0]
        layers = -1 * np.ones(nodes, dtype=int)
        node_layer_range = [0] + list(np.add.accumulate(nodelay))
        for ilay in range(nlayers):
            istart = node_layer_range[ilay]
            istop = node_layer_range[ilay + 1]
            layers[istart:istop] = ilay
        assert np.all(layers >= 0)
        for node in range(nodes):
            ipos = ia[node]
            ihc[ipos] = layers[node]
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
        nja = iac.sum()
        cl12 = self.read_qtg_cl_dat(model_ws=self.model_ws, nja=nja)
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
        nja = iac.sum()
        fahl = self.read_qtg_fahl_dat(model_ws=self.model_ws, nja=nja)

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
                    m = ja[ipos]
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
        angldegx = np.zeros(fldr.shape, dtype=float)
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

        verts, iverts = to_cvfd(self._vertdict, nodestop=ncells, verbose=verbose)
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
        cellxy = np.empty((ncells, 2), dtype=float)
        for n in range(ncells):
            x, y = self.get_center(n)
            cellxy[n, 0] = x
            cellxy[n, 1] = y
        return cellxy

    @staticmethod
    def gridarray_to_flopyusg_gridarray(nodelay, a):
        nlay = nodelay.shape[0]
        istop = 0
        layerlist = []
        for k in range(nlay):
            istart = istop
            istop = istart + nodelay[k]
            ak = a[istart:istop]
            if ak.min() == ak.max():
                ak = ak.min()
            layerlist.append(ak)
        return layerlist

    def get_gridprops_disu5(self):
        """
        Get a dictionary of information needed to create a MODFLOW-USG DISU
        Package.  The returned dictionary can be unpacked directly into the
        ModflowDisU constructor.  The ja dictionary entry will be returned
        as zero-based.

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nodes = self.get_nodes()
        nlay = self.get_nlay()
        iac = self.get_iac()
        nja = iac.sum()
        ja = self.get_ja(nja)
        nodelay = self.get_nodelay()
        top = self.get_top()
        top = self.gridarray_to_flopyusg_gridarray(nodelay, top)
        bot = self.get_bot()
        bot = self.gridarray_to_flopyusg_gridarray(nodelay, bot)
        area = self.get_area()
        area = self.gridarray_to_flopyusg_gridarray(nodelay, area)
        fldr = self.get_fldr()
        ivc = np.where(abs(fldr) == 3, 1, 0)
        cl12 = self.get_cl12()
        fahl = self.get_fahl()

        gridprops["nodes"] = nodes
        gridprops["nlay"] = nlay
        gridprops["njag"] = nja
        gridprops["ivsd"] = 0
        gridprops["idsymrd"] = 0
        gridprops["iac"] = iac
        gridprops["ja"] = ja
        gridprops["nodelay"] = nodelay
        gridprops["top"] = top
        gridprops["bot"] = bot
        gridprops["area"] = area
        gridprops["ivc"] = ivc
        gridprops["cl12"] = cl12
        gridprops["fahl"] = fahl

        return gridprops

    def get_gridprops_disu6(self, repair_asymmetry=True):
        """
        Get a dictionary of information needed to create a MODFLOW 6 DISU
        Package.  The returned dictionary can be unpacked directly into the
        ModflowGwfdisu constructor.

        Parameters
        ----------
        repair_asymmetry : bool
            MODFLOW 6 checks for symmetry in the hwva array, and round off
            errors in the floating point calculations can result in small
            errors.  If this flag is true, then symmetry will be forced by
            setting the symmetric counterparts to the same value (the first
            one encountered).

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

        # Calculate nja and save as nja to self
        nja = iac.sum()
        gridprops["nja"] = nja

        # ja
        ja = self.get_ja(nja)
        gridprops["ja"] = ja

        # cl12
        cl12 = self.get_cl12()
        gridprops["cl12"] = cl12

        # fldr
        fldr = self.get_fldr()

        # ihc
        nodelay = self.get_nodelay()
        ia = get_ia_from_iac(iac)
        ihc = self.get_ihc(nodelay, ia, fldr)
        gridprops["ihc"] = ihc

        # hwva
        hwva = self.get_hwva(ja=ja, ihc=ihc, fahl=None, top=top, bot=bot)
        if repair_asymmetry:
            isym = get_isym(ia, ja)
            hwva = repair_array_asymmetry(isym, hwva)
        gridprops["hwva"] = hwva

        # angldegx
        angldegx = self.get_angldegx(fldr)
        gridprops["angldegx"] = angldegx

        # vertices -- not optimized for redundant vertices yet
        vertices = []
        ivert = 0
        for n in range(nodes):
            vs = self.get_vertices(n)
            for x, y in vs[:-1]:  # do not include last vertex
                vertices.append([ivert, x, y])
                ivert += 1
        nvert = len(vertices)
        gridprops["nvert"] = nvert
        gridprops["vertices"] = vertices

        # cell2d information
        cell2d = []
        iv = 0
        for n in range(nodes):
            xc, yc = self.get_center(n)
            cell2d.append([n, xc, yc, 5, iv, iv + 1, iv + 2, iv + 3, iv])
            iv += 4
        gridprops["cell2d"] = cell2d

        return gridprops

    def get_gridprops_disv(self):
        """
        Get a dictionary of information needed to create a MODFLOW 6 DISV
        Package.  The returned dictionary can be unpacked directly into the
        ModflowGwfdisv constructor.

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nlay = self.get_nlay()
        nodelay = self.get_nodelay()
        ncpl = nodelay.min()
        assert ncpl == nodelay.max(), "Cannot create DISV properties "
        "because the number of cells is not the same for all layers"

        gridprops["nlay"] = nlay
        gridprops["ncpl"] = ncpl

        # top (only need ncpl values)
        top = self.get_top()
        gridprops["top"] = top[:ncpl]

        # botm
        botm = self.get_bot()
        botm = botm.reshape((nlay, ncpl))
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

    def get_gridprops_vertexgrid(self):
        """
        Get a dictionary of information needed to create a flopy VertexGrid.
        The returned dictionary can be unpacked directly into the
        flopy.discretization.VertexGrid() constructor.

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nlay = self.get_nlay()
        nodelay = self.get_nodelay()
        ncpl = nodelay.min()
        assert ncpl == nodelay.max(), "Cannot create properties "
        "because the number of cells is not the same for all layers"

        # top (only need ncpl values)
        top = self.get_top()
        top = top[:ncpl]

        # botm
        botm = self.get_bot()
        botm = botm.reshape((nlay, ncpl))

        # cell xy locations
        cellxy = self.get_cellxy(ncpl)

        # verts and iverts
        verts, iverts = self.get_verts_iverts(ncpl)

        nvert = verts.shape[0]
        vertices = [[i, verts[i, 0], verts[i, 1]] for i in range(nvert)]

        # cell2d information
        cell2d = [
            [n, cellxy[n, 0], cellxy[n, 1], len(ivs)] + ivs
            for n, ivs in enumerate(iverts)
        ]

        gridprops["nlay"] = nlay
        gridprops["ncpl"] = ncpl
        gridprops["top"] = top
        gridprops["botm"] = botm
        gridprops["vertices"] = vertices
        gridprops["cell2d"] = cell2d

        return gridprops

    def get_gridprops_unstructuredgrid(self):
        """
        Get a dictionary of information needed to create a flopy
        UnstructuredGrid.  The returned dictionary can be unpacked directly
        into the flopy.discretization.UnstructuredGrid() constructor.

        Returns
        -------
        gridprops : dict

        """
        gridprops = {}

        nodes = self.get_nodes()
        ncpl = self.get_nodelay()
        xcyc = self.get_cellxy(nodes)
        xcenters = xcyc[:, 0]
        ycenters = xcyc[:, 1]
        top = self.get_top()
        bot = self.get_bot()
        verts, iverts = self.get_verts_iverts(nodes)
        nvert = verts.shape[0]
        vertices = [[i, verts[i, 0], verts[i, 1]] for i in range(nvert)]

        gridprops["vertices"] = vertices
        gridprops["iverts"] = iverts
        gridprops["ncpl"] = ncpl
        gridprops["xcenters"] = xcenters
        gridprops["ycenters"] = ycenters
        gridprops["top"] = top
        gridprops["botm"] = bot
        gridprops["iac"] = self.get_iac()
        gridprops["ja"] = self.get_ja()

        return gridprops

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
            if os.path.exists(f"{ifname_w_path}.shp"):
                os.remove(f"{ifname_w_path}.shp")
            features_to_shapefile(features, featuretype, ifname_w_path)
            shapefile = ifname
        else:
            shapefile = features

        sn = os.path.join(self.model_ws, f"{shapefile}.shp")
        assert os.path.isfile(sn), f"Shapefile does not exist: {sn}"

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
            fn, dtype=None, names=True, delimiter=",", usecols=tuple(range(ncol))
        )
        result = np.atleast_1d(result)
        result = result.view(np.recarray)
        result["nodenumber"] -= 1
        return result

    def _intersection_block(self, shapefile, featuretype, layer):
        s = ""
        s += "BEGIN GRID_INTERSECTION intersect\n"
        s += "  GRID = quadtreegrid\n"
        s += f"  LAYER = {layer + 1}\n"
        s += f"  SHAPEFILE = {shapefile}\n"
        s += f"  FEATURE_TYPE = {featuretype}\n"
        s += "  OUTPUT_FILE = intersection.ifo\n"
        s += "END GRID_INTERSECTION intersect\n"
        return s

    def _mfgrid_block(self):
        # flopy modelgrid object uses same xoff, yoff, and rotation convention
        # as gridgen

        xoff = self.modelgrid.xoffset
        yoff = self.modelgrid.yoffset
        angrot = self.modelgrid.angrot

        s = ""
        s += "BEGIN MODFLOW_GRID basegrid\n"
        s += f"  ROTATION_ANGLE = {angrot}\n"
        s += f"  X_OFFSET = {xoff}\n"
        s += f"  Y_OFFSET = {yoff}\n"
        s += f"  NLAY = {self.nlay}\n"
        s += f"  NROW = {self.nrow}\n"
        s += f"  NCOL = {self.ncol}\n"

        # delr
        delr = self.modelgrid.delr
        if delr.min() == delr.max():
            s += f"  DELR = CONSTANT {delr.min()}\n"
        else:
            s += "  DELR = OPEN/CLOSE delr.dat\n"
            fname = os.path.join(self.model_ws, "delr.dat")
            np.savetxt(fname, np.atleast_2d(delr))

        # delc
        delc = self.modelgrid.delc
        if delc.min() == delc.max():
            s += f"  DELC = CONSTANT {delc.min()}\n"
        else:
            s += "  DELC = OPEN/CLOSE delc.dat\n"
            fname = os.path.join(self.model_ws, "delc.dat")
            np.savetxt(fname, np.atleast_2d(delc))

        # top
        top = self.modelgrid.top
        if top.min() == top.max():
            s += f"  TOP = CONSTANT {top.min()}\n"
        else:
            s += "  TOP = OPEN/CLOSE top.dat\n"
            fname = os.path.join(self.model_ws, "top.dat")
            np.savetxt(fname, top)

        # bot
        botm = self.modelgrid.botm
        for k in range(self.nlay):
            bot = botm[k]
            if bot.min() == bot.max():
                s += f"  BOTTOM LAYER {k + 1} = CONSTANT {bot.min()}\n"
            else:
                s += "  BOTTOM LAYER {0} = OPEN/CLOSE bot{0}.dat\n".format(k + 1)
                fname = os.path.join(self.model_ws, f"bot{k + 1}.dat")
                np.savetxt(fname, bot)

        s += "END MODFLOW_GRID\n"
        return s

    def _rf_blocks(self):
        s = ""
        for rfname, rf in self._rfdict.items():
            shapefile, featuretype, level = rf
            s += f"BEGIN REFINEMENT_FEATURES {rfname}\n"
            s += f"  SHAPEFILE = {shapefile}\n"
            s += f"  FEATURE_TYPE = {featuretype}\n"
            s += f"  REFINEMENT_LEVEL = {level}\n"
            s += "END REFINEMENT_FEATURES\n"
            s += 2 * "\n"
        return s

    def _ad_blocks(self):
        s = ""
        for adname, shapefile in self._addict.items():
            s += f"BEGIN ACTIVE_DOMAIN {adname}\n"
            s += f"  SHAPEFILE = {shapefile}\n"
            s += "  FEATURE_TYPE = polygon\n"
            s += "  INCLUDE_BOUNDARY = True\n"
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
            s += f"  ACTIVE_DOMAIN LAYER {k + 1} = {adk}\n"

        # Write refinement feature information
        for k, rfkl in enumerate(self._refinement_features):
            if len(rfkl) == 0:
                continue
            s += f"  REFINEMENT_FEATURES LAYER {k + 1} = "
            for rf in rfkl:
                s += f"{rf} "
            s += "\n"

        s += "  SMOOTHING = full\n"
        s += f"  SMOOTHING_LEVEL_VERTICAL = {self.smoothing_level_vertical}\n"
        s += f"  SMOOTHING_LEVEL_HORIZONTAL = {self.smoothing_level_horizontal}\n"

        for k in range(self.nlay):
            if self.surface_interpolation[k] == "ASCIIGRID":
                grd = self._asciigrid_dict[k]
            else:
                grd = "basename"
            s += f"  TOP LAYER {k + 1} = {self.surface_interpolation[k]} {grd}\n"

        for k in range(self.nlay):
            if self.surface_interpolation[k + 1] == "ASCIIGRID":
                grd = self._asciigrid_dict[k + 1]
            else:
                grd = "basename"
            s += f"  BOTTOM LAYER {k + 1} = {self.surface_interpolation[k + 1]} {grd}\n"

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
        s += f"  VERTICAL_PASS_THROUGH = {self.vertical_pass_through}\n"
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
        nodes = self.read_qtg_nod(model_ws=self.model_ws, nodes_only=True)
        if nodes == 0:
            raise Exception("Gridgen resulted in no active cells.")

        sf = self.read_qtgrid_shp(model_ws=self.model_ws)
        shapes = sf.shapes()
        fields = sf.fields
        attributes = [l[0] for l in fields[1:]]
        records = sf.records()
        idx = attributes.index("nodenumber")
        for i in range(len(shapes)):
            nodenumber = int(records[i][idx]) - 1
            self._vertdict[nodenumber] = shapes[i].points

    @staticmethod
    def read_qtg_nod(model_ws: Union[str, os.PathLike], nodes_only: bool = False):
        """Read qtg.nod file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nodes_only : bool, optional
            Read only the number of nodes from file, by default False which
            reads the entire file

        Returns
        -------
        int or numpy recarray

        """

        fname = os.path.join(model_ws, "qtg.nod")

        with open(fname, "r") as f:
            if nodes_only:
                line = f.readline()
                ll = line.strip().split()
                nodes = int(ll[0])
            else:
                dt = np.dtype(
                    [
                        ("node", int),
                        ("layer", int),
                        ("x", float),
                        ("y", float),
                        ("z", float),
                        ("dx", float),
                        ("dy", float),
                        ("dz", float),
                    ]
                )
                nodes = np.genfromtxt(fname, dtype=dt, skip_header=1)
                nodes["layer"] -= 1
                nodes["node"] -= 1
            return nodes

    @staticmethod
    def read_qtgrid_shp(model_ws: Union[str, os.PathLike]):
        """Read qtgrid.shp file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored

        Returns
        -------
        shapefile
        """
        shapefile = import_optional_dependency("shapefile")

        # ensure shape file was created by gridgen
        fname = os.path.join(model_ws, "qtgrid.shp")
        # read vertices from shapefile
        return shapefile.Reader(fname)

    @staticmethod
    def read_qtg_nodesperlay_dat(model_ws: Union[str, os.PathLike], nlay: int):
        """Read qtgrid.shp file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nlay : int
            Number of layers

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.nodesperlay.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nlay), dtype=int))

    @staticmethod
    def read_quadtreegrid_top_dat(
        model_ws: Union[str, os.PathLike], nodelay: list[int], lay: int
    ):
        """Read quadtreegrid.top_.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nodelay : list[int]
            Number of nodes in each layer
        lay : int
            Layer

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, f"quadtreegrid.top{lay + 1}.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nodelay[lay]), dtype=np.float32))

    @staticmethod
    def read_quadtreegrid_bot_dat(
        model_ws: Union[str, os.PathLike], nodelay: list[int], lay: int
    ):
        """Read quadtreegrid.bot_.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nodelay : list[int]
            Number of nodes in each layer
        lay : int
            Layer

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, f"quadtreegrid.bot{lay + 1}.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nodelay[lay]), dtype=np.float32))

    @staticmethod
    def read_qtg_area_dat(model_ws: Union[str, os.PathLike], nodes: int):
        """Read qtg.area.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nodes : int
            Number of nodes

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.area.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nodes), dtype=np.float32))

    @staticmethod
    def read_qtg_iac_dat(model_ws: Union[str, os.PathLike], nodes: int):
        """Read qtg.iac.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nodes : int
            Number of nodes

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.iac.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nodes), dtype=int))

    @staticmethod
    def read_qtg_ja_dat(model_ws: Union[str, os.PathLike], nja: int):
        """Read qtg.ja.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nja : int
            Number of connections

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.ja.dat")
        with open(fname, "r") as f:
            ja = read1d(f=f, a=np.empty((nja), dtype=int)) - 1
            return ja

    @staticmethod
    def read_qtg_fldr_dat(model_ws: Union[str, os.PathLike], nja: int):
        """Read qtg.fldr.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nja : int
            Number of connections

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.fldr.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nja), dtype=int))

    @staticmethod
    def read_qtg_cl_dat(model_ws: Union[str, os.PathLike], nja: int):
        """Read qtg.c1.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nja : int
            Number of connections

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.c1.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nja), dtype=np.float32))

    @staticmethod
    def read_qtg_fahl_dat(model_ws: Union[str, os.PathLike], nja: int):
        """Read qtg.fahl.dat file

        Parameters
        ----------
        model_ws : Union[str, os.PathLike]
            Directory where file is stored
        nja : int
            Number of connections

        Returns
        -------
        np.ndarray
        """
        fname = os.path.join(model_ws, "qtg.fahl.dat")
        with open(fname, "r") as f:
            return read1d(f=f, a=np.empty((nja), dtype=np.float32))
