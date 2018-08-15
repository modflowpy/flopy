import copy
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
except ImportError:
    plt = None

from flopy.plot import plotutil
# from flopy.plot.plotutil import bc_color_dict
from flopy.utils import SpatialReference as DepreciatedSpatialReference
from flopy.grid.structuredmodelgrid import StructuredModelGrid
from flopy.grid.reference import SpatialReference
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class VertexMapView(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    sr : flopy.utils.reference.SpatialReference
        The spatial reference class (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    Notes
    -----
    ModelMap must know the position and rotation of the grid in order to make
    the plot.  This information is contained in the SpatialReference class
    (sr), which can be passed.  If sr is None, then it looks for sr in dis.
    If dis is None, then it looks for sr in model.dis.  If all of these
    arguments are none, then it uses xul, yul, and rotation.  If none of these
    arguments are provided, then it puts the lower-left-hand corner of the
    grid at (0, 0).

    """
    def __init__(self, sr=None, ax=None, model=None, dis=None, modelgrid=None,
                 layer=0, extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., length_multiplier=1.):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise Exception(s)

        self.model = model
        self.layer = layer
        self.dis = dis
        self.mg = None
        self.sr = None

        if model is not None:
            self.mg = model.modelgrid
            self.sr = model.modelgrid.sr

        elif modelgrid is not None:
            self.mg = modelgrid
            # todo: remove statement once model grid/spatial reference is finalized
            try:
                self.sr = modelgrid.sr
            except:
                self.sr = None
        elif dis is not None:
            self.mg = dis.parent.modelgrid
            self.sr = dis.parent.modelgrid.sr

        elif sr is not None:
            if isinstance(sr, DepreciatedSpatialReference):
                self.mg = copy.deepcopy(sr)
                self.sr = copy.deepcopy(sr)

            else:
                self.sr = sr

                # todo: change this to VertexModelGrid
                self.mg = StructuredModelGrid(delc=np.array([]), delr=np.array([]),
                                              top=np.array([]), botm=np.array([]),
                                              idomain=np.array([]), sr=self.sr)

        else:
            self.sr = SpatialReference(delc=np.array([]), xll=xll, xul=xul,
                                       yul=yul, rotation=rotation,
                                       length_multiplier=length_multiplier)

            # todo: change this to VertexModelGrid
            self.mg = StructuredModelGrid(delc=np.array([]), delr=np.array([]),
                                          top=np.array([]), botm=np.array([]),
                                          idomain=np.array([]), sr=self.sr)

        # model map override spatial reference settings
        if any(elem is not None for elem in (xul, yul, xll, yll)) or \
                rotation != 0 or length_multiplier != 1.:
            self.sr.length_multiplier = length_multiplier
            if isinstance(sr, DepreciatedSpatialReference):
                self.sr.set_spatialreference(xul=xul, yul=yul,
                                             xll=xll, yll=yll,
                                             rotation=rotation)
            else:
                self.sr.set_spatialreference(delc=self.mg.delc,
                                             xul=xul, yul=yul,
                                             xll=xll, yll=yll,
                                             rotation=rotation)
                self.mg.sr = self.sr

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect('equal')
            except:
                self.ax = plt.subplot(1, 1, 1, aspect='equal', axisbg="white")
        else:
            self.ax = ax

        if extent is not None:
            self._extent = extent
        else:
            self._extent = None

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.mg.sr.extent
        return self._extent


    def plot_array(self, a, masked_values=None, **kwargs):
        """
        Plot an array.  If the array is three-dimensional, then the method
        will plot the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.patchcollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        xgrid = self.sr.xgrid
        ygrid = self.sr.ygrid

        patches = [Polygon(list(zip(xgrid[i], ygrid[i])), closed=True)
                   for i in range(xgrid.shape[0])]

        p = PatchCollection(patches)
        p.set_array(plotarray)

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        p.set_clim(vmin=vmin, vmax=vmax)
        # send rest of kwargs to quadmesh
        p.set(**kwargs)

        ax.add_collection(p)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return ax

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        import matplotlib.tri as tri

        xcentergrid = self.mg.sr.xcenters
        ycentergrid = self.mg.sr.ycenters

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' in kwargs.keys():
            if 'cmap' in kwargs.keys():
                kwargs.pop('cmap')

        plot_triplot = False
        if 'plot_triplot' in kwargs:
            plot_triplot = kwargs.pop('plot_triplot')

        if 'extent' in kwargs:
            extent = kwargs.pop('extent')

            idx = (xcentergrid >= extent[0]) & (
                    xcentergrid <= extent[1]) & (
                          ycentergrid >= extent[2]) & (
                          ycentergrid <= extent[3])
            a = a[idx].flatten()
            xcentergrid = xcentergrid[idx].flatten()
            ycentergrid = ycentergrid[idx].flatten()

        triang = tri.Triangulation(xcentergrid, ycentergrid)

        try:
            amask = a.mask
            mask = [False for i in range(triang.triangles.shape[0])]
            for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                if amask[n0] or amask[n1] or amask[n2]:
                    mask[ipos] = True
            triang.set_mask(mask)
        except:
            pass

        contour_set = ax.tricontour(triang, plotarray, **kwargs)

        if plot_triplot:
            ax.triplot(triang, color="black", marker="o", lw=0.75)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, idomain=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.pyplot.axes object

        """
        if idomain is None:
            idomain = self.model.dis.idomain.array

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain <= 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        xgrid = self.sr.xgrid
        ygrid = self.sr.ygrid

        patches = [Polygon(list(zip(xgrid[i], ygrid[i])), closed=True)
                   for i in range(xgrid.shape[0])]

        p = PatchCollection(patches, cmap=cmap, norm=norm)
        p.set_array(plotarray)
        p.set(**kwargs)

        ax.add_collection(p)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return ax

    def plot_ibound(self, idomain=None, color_noflow='black', color_ch="blue",
                    color_vpt="red", **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_vpt : string
            Color for vertical pass through cells (Default is 'red'.)

        Returns
        -------
        ax : matplotlib.pyplot.axes object

        """
        if idomain is None:
            idomain = self.model.dis.idomain.array

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain == 0)
        idx2 = (idomain < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_vpt])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        xgrid = self.sr.xgrid
        ygrid = self.sr.ygrid

        patches = [Polygon(list(zip(xgrid[i], ygrid[i])), closed=True)
                   for i in range(xgrid.shape[0])]

        p = PatchCollection(patches, cmap=cmap, norm=norm)
        p.set_array(plotarray)
        p.set(**kwargs)

        ax.add_collection(p)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return ax

    def plot_grid(self, **kwargs):
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
        from matplotlib.collections import LineCollection

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        lc = LineCollection(self.mg.sr.grid_lines, **kwargs)

        ax.add_collection(lc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return ax

    def plot_bc(self, ftype=None, package=None, kper=0, color=None,
                plotAll=False, **kwargs):
        """
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        ftype : string
            Package name string ('WEL', 'GHB', etc.). (Default is None)
        package : flopy.modflow.Modflow package class instance
            flopy package class instance. (Default is None)
        kper : int
            Stress period to plot
        color : string
            matplotlib color string. (Default is None)
        plotAll : bool
            Boolean used to specify that boundary condition locations for all
            layers will be plotted on the current ModelMap layer.
            (Default is False)
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        #find package to plot
        if package is not None:
            p = package
            ftype = p.name[0]

        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            ftype = ftype.upper()
            p = self.model.get_package(ftype)

        else:
            raise Exception('Cannot find package to plot')

        arr_dict = p.stress_period_data.to_array(kper)
        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        nlay = self.mg.nlay

        # Plot the list locations
        plotarray = np.zeros((nlay, self.mg.ncpl), dtype=np.int)

        if plotAll:
            t = np.sum(fluxes, axis=0)
            pa = np.zeros((self.mg.ncpl,), dtype=np.int)
            pa[t != 0] = 1
            for k in range(nlay):
                plotarray[k, :] = pa.copy()
        else:
            plotarray[fluxes != 0] = 1

        # mask the plot array
        plotarray = np.ma.masked_equal(plotarray, 0)

        # set the colormap
        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color

        cmap = matplotlib.colors.ListedColormap(['0', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        ax = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return ax

    def plot_shapefile(self, shp, **kwargs):
        return NotImplementedError()

    def plot_cvfd(self, verts, iverts, **kwargs):
        return NotImplementedError()

    def contour_array_cvfd(self, vertc, a, masked_values=None, **kwargs):
        return NotImplementedError()

    def plot_discharge(self, fja, dis=None, head=None, istep=1,
                       normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        fja : numpy.ndarray
            MODFLOW's 'flow ja face'
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        # todo: change this to reference the modelgrid instance if dis does not exist
        if dis is None:
            if self.model is not None:
                dis = self.model.dis
            else:
                err_msg = "ModelMap.plot_quiver() error: DIS package not found"
                raise AssertionError(err_msg)

        top = dis.top.array
        botm = dis.botm.array

        fja = np.array(fja)
        nlay = self.mg.nlay
        ncpl = self.mg.ncpl

        delr = np.tile([np.max(i) - np.min(i) for i in self.mg.ygrid], (nlay, 1))
        delc = np.tile([np.max(i) - np.min(i) for i in self.mg.xgrid], (nlay, 1))

        # todo: get hnoflow and hdry from the proper place
        hnoflo = 999.
        hdry = 999.

        if head is None:
            head = np.zeros(botm.shape)

        if len(head.shape) == 3:
            head.shape = (nlay, -1)

        if len(fja.shape) == 4:
            fja = fja[0][0][0]

        laytyp = np.zeros((nlay,))

        if self.model is not None:
            if self.model.sto is not None:
                laytyp = np.zeros((nlay,))# self.model.sto.iconvert.array

        # todo: update saturated thickness for the new iconvert array!
        sat_thk = plotutil.PlotUtilities.\
            saturated_thickness(head, top,
                                botm, laytyp,
                                mask_values=[hnoflo, hdry])

        frf, fff, flf = plotutil.UnstructuredPlotUtilities.\
            vectorize_flow(fja, model_grid=self.mg,
                           idomain=dis.idomain.array)

        qx, qy, qz = plotutil.UnstructuredPlotUtilities.\
            specific_discharge(frf, fff, flf,
                               delr, delc, sat_thk)

        # Select the correct layer slice
        u = qx[self.layer, :]
        v = qy[self.layer, :]

        # apply step
        x = self.mg.sr.xcenters[::istep]
        y = self.mg.sr.ycenters[::istep]
        u = u[::istep]
        v = v[::istep]
        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # mask discharge in inactive cells
        idomain = dis.idomain.array
        idx = (idomain[self.layer, ::istep] == 0)
        idx[idomain[self.layer, ::istep] == -1] = 1

        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot
        urot, vrot = self.mg.sr.rotate(u, v, self.mg.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, scale=1, units='xy', pivot=pivot, **kwargs)
        return quiver

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        return NotImplementedError()

    def plot_endpoint(self, ep, direction="ending", selection=None,
                      selection_direction=None, **kwargs):
        return NotImplementedError()


if __name__ == "__main__":
    import os
    import flopy as fp
    from flopy.plot.plotbase import PlotMapView
    import flopy.utils.binaryfile as bf
    from flopy.proposed_grid.proposed_vertex_mg import VertexModelGrid

    ws = "../../examples/data/mf6/test003_gwfs_disv"
    name = "mfsim.nam"

    sim = fp.mf6.modflow.MFSimulation.load(sim_name=name, sim_ws=ws)

    print(sim.model_names)
    ml = sim.get_model("gwf_1")

    dis = ml.dis
    sto = ml.sto

    print('break')

    t = VertexModelGrid(dis.vertices, dis.cell2d,
                        top=dis.top, botm=dis.botm,
                        idomain=dis.idomain, xoffset=10.,
                        yoffset=0, rotation=-25)

    # todo: build out model grid methods!
    x = t.xgrid
    y = t.ygrid
    t0 = t.top
    t1 = t.botm
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

    map = PlotMapView(modelgrid=t, layer=0)
    #ax = map.plot_array(a=dis.botm.array)
    #plt.show()

    #arr = np.random.rand(100) * 100
    #ax = map.contour_array(a=arr)
    #plt.show()

    #idomain = np.ones(100, dtype=np.int)
    #r = np.random.randint(0, 100, size=25)
    #r1 = np.random.randint(0, 100, size=10)
    #idomain[r] = 0
    #idomain[r1] = -1
    #ax = map.plot_ibound(idomain)
    #plt.show()

    #ax = map.plot_grid()
    #plt.show()

    #chd = ml.get_package("CHD")
    #ax = map.plot_bc(package=chd)
    #plt.show()

    cbc = os.path.join(ws, "model.cbc")
    hds = os.path.join(ws, "model.hds")

    cbc = bf.CellBudgetFile(cbc, precision="double")
    hds = bf.HeadFile(hds)

    print(cbc.get_unique_record_names())

    fja = cbc.get_data(text="FLOW-JA-FACE")
    head = hds.get_alldata()[0]
    head.shape = (4, -1)
    print(head.ndim)

    ax = map.plot_discharge(fja=fja, head=head, dis=dis)
    plt.show()
    print('break')
