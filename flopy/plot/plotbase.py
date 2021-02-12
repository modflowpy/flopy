import numpy as np
from ..plot.crosssection import _StructuredCrossSection
from ..plot.vcrosssection import _VertexCrossSection
from ..plot import plotutil

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None


class PlotCrossSection(object):
    """
    Class to create a cross section of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.discretization.Grid object
        can be a StructuredGrid, VertexGrid, or UnstructuredGrid object
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section.
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.
    geographic_coords : bool
        boolean flag to allow the user to plot cross section lines in
        geographic coordinates. If False (default), cross section is plotted
        as the distance along the cross section line.

    """

    def __init__(
        self,
        model=None,
        modelgrid=None,
        ax=None,
        line=None,
        extent=None,
        geographic_coords=False,
    ):
        if plt is None:
            s = (
                "Could not import matplotlib.  Must install matplotlib "
                + " in order to use ModelMap method"
            )
            raise ImportError(s)

        if modelgrid is None and model is not None:
            modelgrid = model.modelgrid

        # update this after unstructured grid is finished!
        tmp = modelgrid.grid_type

        if tmp == "structured":
            self.__cls = _StructuredCrossSection(
                ax=ax,
                model=model,
                modelgrid=modelgrid,
                line=line,
                extent=extent,
                geographic_coords=geographic_coords,
            )

        elif tmp == "unstructured":
            raise NotImplementedError("Unstructured xc not yet implemented")

        elif tmp == "vertex":
            self.__cls = _VertexCrossSection(
                ax=ax,
                model=model,
                modelgrid=modelgrid,
                line=line,
                extent=extent,
                geographic_coords=geographic_coords,
            )

        else:
            raise ValueError("Unknown modelgrid type {}".format(tmp))

        self.model = self.__cls.model
        self.mg = self.__cls.mg
        self.ax = self.__cls.ax
        self.direction = self.__cls.direction
        self.pts = self.__cls.pts
        self.xpts = self.__cls.xpts
        self.d = self.__cls.d
        self.ncb = self.__cls.ncb
        self.laycbd = self.__cls.laycbd
        self.active = self.__cls.active
        self.elev = self.__cls.elev
        self.layer0 = self.__cls.layer0
        self.layer1 = self.__cls.layer1
        self.zpts = self.__cls.zpts
        self.xcentergrid = self.__cls.xcentergrid
        self.zcentergrid = self.__cls.zcentergrid
        self.geographic_coords = self.__cls.geographic_coords
        self.extent = self.__cls.extent

    def plot_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Plot a three-dimensional array as a patch collection.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        return self.__cls.plot_array(
            a=a, masked_values=masked_values, head=head, **kwargs
        )

    def plot_surface(self, a, masked_values=None, **kwargs):
        """
        Plot a two- or three-dimensional array as line(s).

        Parameters
        ----------
        a : numpy.ndarray
            Two- or three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        plot : list containing matplotlib.plot objects

        """
        return self.__cls.plot_surface(
            a=a, masked_values=masked_values, **kwargs
        )

    def plot_fill_between(
        self,
        a,
        colors=("blue", "red"),
        masked_values=None,
        head=None,
        **kwargs
    ):
        """
        Plot a three-dimensional array as lines.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        colors: list
            matplotlib fill colors, two required
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        plot : list containing matplotlib.fillbetween objects

        """
        return self.__cls.plot_fill_between(
            a=a,
            colors=colors,
            masked_values=masked_values,
            head=head,
            **kwargs
        )

    def contour_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Contour a three-dimensional array.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.contour

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        return self.__cls.contour_array(
            a=a, masked_values=masked_values, head=head, **kwargs
        )

    def plot_inactive(self, ibound=None, color_noflow="black", **kwargs):
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
        quadmesh : matplotlib.collections.QuadMesh

        """
        if ibound is None:
            if self.mg.idomain is None:
                raise AssertionError("An idomain array must be provided")
            else:
                ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=int)
        idx1 = ibound == 0
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(["0", color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return patches

    def plot_ibound(
        self,
        ibound=None,
        color_noflow="black",
        color_ch="blue",
        color_vpt="red",
        head=None,
        **kwargs
    ):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.model

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if ibound is None:
            if self.model is not None:
                if self.model.version == "mf6":
                    color_ch = color_vpt

            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

            ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=int)
        idx1 = ibound == 0
        idx2 = ibound < 0
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(
            ["none", color_noflow, color_ch]
        )
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # mask active cells
        patches = self.plot_array(
            plotarray,
            masked_values=[0],
            head=head,
            cmap=cmap,
            norm=norm,
            **kwargs
        )
        return patches

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
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        col = self.get_grid_line_collection(**kwargs)
        if col is not None:
            ax.add_collection(col)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return col

    def plot_bc(
        self, name=None, package=None, kper=0, color=None, head=None, **kwargs
    ):
        """
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        name : string
            Package name string ('WEL', 'GHB', etc.). (Default is None)
        package : flopy.modflow.Modflow package class instance
            flopy package class instance. (Default is None)
        kper : int
            Stress period to plot
        color : string
            matplotlib color string. (Default is None)
        head : numpy.ndarray
            Three-dimensional array (structured grid) or
            Two-dimensional array (vertex grid)
            to set top of patches to the minimum of the top of a\
            layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if "ftype" in kwargs and name is None:
            name = kwargs.pop("ftype")

        # Find package to plot
        if package is not None:
            p = package
        elif self.model is not None:
            if name is None:
                raise Exception("ftype not specified")
            name = name.upper()
            p = self.model.get_package(name)
        else:
            raise Exception("Cannot find package to plot")

        # trap for mf6 'cellid' vs mf2005 'k', 'i', 'j' convention
        if isinstance(p, list) or p.parent.version == "mf6":
            if not isinstance(p, list):
                p = [p]

            idx = np.array([])
            for pp in p:
                if pp.package_type in ("lak", "sfr", "maw", "uzf"):
                    t = plotutil.advanced_package_bc_helper(pp, self.mg, kper)
                else:
                    try:
                        mflist = pp.stress_period_data.array[kper]
                    except Exception as e:
                        raise Exception(
                            "Not a list-style boundary package: " + str(e)
                        )
                    if mflist is None:
                        return

                    t = np.array(
                        [list(i) for i in mflist["cellid"]], dtype=int
                    ).T

                if len(idx) == 0:
                    idx = np.copy(t)
                else:
                    idx = np.append(idx, t, axis=1)

        else:
            # modflow-2005 structured and unstructured grid
            if p.package_type in ("uzf", "lak"):
                idx = plotutil.advanced_package_bc_helper(p, self.mg, kper)
            else:
                try:
                    mflist = p.stress_period_data[kper]
                except Exception as e:
                    raise Exception(
                        "Not a list-style boundary package: " + str(e)
                    )
                if mflist is None:
                    return
                if len(self.mg.shape) == 3:
                    idx = [mflist["k"], mflist["i"], mflist["j"]]
                else:
                    idx = mflist["node"]

        # Plot the list locations, change this to self.mg.shape
        if len(self.mg.shape) != 3:
            plotarray = np.zeros((self.mg.nlay, self.mg.ncpl), dtype=int)
            plotarray[tuple(idx)] = 1
        else:
            plotarray = np.zeros(
                (self.mg.nlay, self.mg.nrow, self.mg.ncol), dtype=int
            )
            plotarray[idx[0], idx[1], idx[2]] = 1

        plotarray = np.ma.masked_equal(plotarray, 0)
        if color is None:
            key = name[:3].upper()
            if key in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[key]
            else:
                c = plotutil.bc_color_dict["default"]
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(["none", c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(
            plotarray,
            masked_values=[0],
            head=head,
            cmap=cmap,
            norm=norm,
            **kwargs
        )

        return patches

    def plot_vector(
        self,
        vx,
        vy,
        vz,
        head=None,
        kstep=1,
        hstep=1,
        normalize=False,
        masked_values=None,
        **kwargs
    ):
        """
        Plot a vector.

        Parameters
        ----------
        vx : np.ndarray
            x component of the vector to be plotted (non-rotated)
            array shape must be (nlay, nrow, ncol) for a structured grid
            array shape must be (nlay, ncpl) for a unstructured grid
        vy : np.ndarray
            y component of the vector to be plotted (non-rotated)
            array shape must be (nlay, nrow, ncol) for a structured grid
            array shape must be (nlay, ncpl) for a unstructured grid
        vz : np.ndarray
            y component of the vector to be plotted (non-rotated)
            array shape must be (nlay, nrow, ncol) for a structured grid
            array shape must be (nlay, ncpl) for a unstructured grid
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then the quivers will be
            plotted in the cell center.
        kstep : int
            layer frequency to plot (default is 1)
        hstep : int
            horizontal frequency to plot (default is 1)
        normalize : bool
            boolean flag used to determine if vectors should be normalized
            using the vector magnitude in each cell (default is False)
        masked_values : iterable of floats
            values to mask
        kwargs : matplotlib.pyplot keyword arguments for the
            plt.quiver method

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            result of the quiver function

        """
        if "pivot" in kwargs:
            pivot = kwargs.pop("pivot")
        else:
            pivot = "middle"

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        # this function does not support arbitrary cross-sections, so check it
        arbitrary = False
        if self.mg.grid_type == "structured":
            if not (self.direction == "x" or self.direction == "y"):
                arbitrary = True
        else:
            # check within a tolerance
            pts = self.pts
            xuniform = [
                True if abs(pts.T[0, 0] - i) < 1 else False for i in pts.T[0]
            ]
            yuniform = [
                True if abs(pts.T[1, 0] - i) < 1 else False for i in pts.T[1]
            ]
            if not np.all(xuniform) and not np.all(yuniform):
                arbitrary = True
        if arbitrary:
            err_msg = (
                "plot_specific_discharge() does not "
                "support arbitrary cross-sections"
            )
            raise AssertionError(err_msg)

        # get the actual values to plot
        if self.direction == "x":
            u_tmp = vx
        elif self.direction == "y":
            u_tmp = -1.0 * vy
        v_tmp = vz
        if self.mg.grid_type == "structured":
            if isinstance(head, np.ndarray):
                zcentergrid = self.__cls.set_zcentergrid(head)
            else:
                zcentergrid = self.zcentergrid

            if self.geographic_coords:
                xcentergrid = self.__cls.geographic_xcentergrid
            else:
                xcentergrid = self.xcentergrid

            if self.mg.nlay == 1:
                x = []
                z = []
                for k in range(self.mg.nlay):
                    for i in range(xcentergrid.shape[1]):
                        x.append(xcentergrid[k, i])
                        z.append(
                            0.5 * (zcentergrid[k, i] + zcentergrid[k + 1, i])
                        )
                x = np.array(x).reshape((1, xcentergrid.shape[1]))
                z = np.array(z).reshape((1, xcentergrid.shape[1]))
            else:
                x = xcentergrid
                z = zcentergrid

            u = []
            v = []
            xedge, yedge = self.mg.xyedges
            for k in range(self.mg.nlay):
                u.append(
                    plotutil.cell_value_points(
                        self.xpts, xedge, yedge, u_tmp[k, :, :]
                    )
                )
                v.append(
                    plotutil.cell_value_points(
                        self.xpts, xedge, yedge, v_tmp[k, :, :]
                    )
                )
            u = np.array(u)
            v = np.array(v)
            x = x[::kstep, ::hstep]
            z = z[::kstep, ::hstep]
            u = u[::kstep, ::hstep]
            v = v[::kstep, ::hstep]

            # upts and vpts has a value for the left and right
            # sides of a cell. Sample every other value for quiver
            u = u[:, ::2]
            v = v[:, ::2]

        else:
            # kstep implementation for vertex grid
            projpts = {
                key: value
                for key, value in self.__cls.projpts.items()
                if (key // self.mg.ncpl) % kstep == 0
            }

            # set x and z centers
            if isinstance(head, np.ndarray):
                # pipe kstep to set_zcentergrid to assure consistent array size
                zcenters = self.__cls.set_zcentergrid(
                    np.ravel(head), kstep=kstep
                )
            else:
                zcenters = [
                    np.mean(np.array(v).T[1])
                    for i, v in sorted(projpts.items())
                ]

            u = np.array([u_tmp.ravel()[cell] for cell in sorted(projpts)])

            x = np.array(
                [np.mean(np.array(v).T[0]) for i, v in sorted(projpts.items())]
            )

            z = np.ravel(zcenters)
            v = np.array([v_tmp.ravel()[cell] for cell in sorted(projpts)])

            x = x[::hstep]
            z = z[::hstep]
            u = u[::hstep]
            v = v[::hstep]

        # mask values
        if masked_values is not None:
            for mval in masked_values:
                to_mask = np.logical_or(u == mval, v == mval)
                u[to_mask] = np.nan
                v[to_mask] = np.nan

        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2.0 + v ** 2.0)
            idx = vmag > 0.0
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        # plot with quiver
        quiver = ax.quiver(x, z, u, v, pivot=pivot, **kwargs)

        return quiver

    def plot_specific_discharge(
        self, spdis, head=None, kstep=1, hstep=1, normalize=False, **kwargs
    ):
        """
        DEPRECATED. Use plot_vector() instead, which should follow after
        postprocessing.get_specific_discharge().

        Use quiver to plot vectors.

        Parameters
        ----------
        spdis : np.recarray
            numpy recarray of specific discharge information. This
            can be grabbed directly from the CBC file if SAVE_SPECIFIC_DISCHARGE
            is used in the MF6 NPF file.
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then the quivers will be plotted
            in the cell center.
        kstep : int
            layer frequency to plot. (Default is 1.)
        hstep : int
            horizontal frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors

        """
        import warnings

        warnings.warn(
            "plot_specific_discharge() has been deprecated. Use "
            "plot_vector() instead, which should follow after "
            "postprocessing.get_specific_discharge()",
            DeprecationWarning,
        )

        if "pivot" in kwargs:
            pivot = kwargs.pop("pivot")
        else:
            pivot = "middle"

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if isinstance(spdis, list):
            print(
                "Warning: Selecting the final stress period from Specific"
                " Discharge list"
            )
            spdis = spdis[-1]

        if self.mg.grid_type == "structured":
            ncpl = self.mg.nrow * self.mg.ncol

        else:
            ncpl = self.mg.ncpl

        nlay = self.mg.nlay

        qx = np.zeros((nlay * ncpl))
        qz = np.zeros((nlay * ncpl))
        ib = np.zeros((nlay * ncpl), dtype=bool)

        idx = np.array(spdis["node"]) - 1

        # check that vertex grid cross sections are not arbitrary
        # within a tolerance!
        if self.mg.grid_type != "structured":
            pts = self.pts
            xuniform = [
                True if abs(pts.T[0, 0] - i) < 1 else False for i in pts.T[0]
            ]
            yuniform = [
                True if abs(pts.T[1, 0] - i) < 1 else False for i in pts.T[1]
            ]
            if not np.all(xuniform):
                if not np.all(yuniform):
                    err_msg = (
                        "plot_specific_discharge does not "
                        "support aribtrary cross sections"
                    )
                    raise AssertionError(err_msg)

        if self.direction == "x":
            qx[idx] = spdis["qx"]
        elif self.direction == "y":
            qx[idx] = spdis["qy"] * -1
        else:
            err_msg = (
                "plot_specific_discharge does not "
                "support arbitrary cross-sections"
            )
            raise AssertionError(err_msg)

        qz[idx] = spdis["qz"]
        ib[idx] = True

        if self.mg.grid_type == "structured":
            qx.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            qz.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            ib.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)

            if isinstance(head, np.ndarray):
                zcentergrid = self.__cls.set_zcentergrid(head)
            else:
                zcentergrid = self.zcentergrid

            if self.geographic_coords:
                xcentergrid = self.__cls.geographic_xcentergrid
            else:
                xcentergrid = self.xcentergrid

            if nlay == 1:
                x = []
                z = []
                for k in range(nlay):
                    for i in range(xcentergrid.shape[1]):
                        x.append(xcentergrid[k, i])
                        z.append(
                            0.5 * (zcentergrid[k, i] + zcentergrid[k + 1, i])
                        )
                x = np.array(x).reshape((1, xcentergrid.shape[1]))
                z = np.array(z).reshape((1, xcentergrid.shape[1]))
            else:
                x = xcentergrid
                z = zcentergrid

            u = []
            v = []
            ibx = []
            xedge, yedge = self.mg.xyedges
            for k in range(self.mg.nlay):
                u.append(
                    plotutil.cell_value_points(
                        self.xpts, xedge, yedge, qx[k, :, :]
                    )
                )
                v.append(
                    plotutil.cell_value_points(
                        self.xpts, xedge, yedge, qz[k, :, :]
                    )
                )
                ibx.append(
                    plotutil.cell_value_points(
                        self.xpts, xedge, yedge, ib[k, :, :]
                    )
                )
            u = np.array(u)
            v = np.array(v)
            ibx = np.array(ibx)
            x = x[::kstep, ::hstep]
            z = z[::kstep, ::hstep]
            u = u[::kstep, ::hstep]
            v = v[::kstep, ::hstep]
            ib = ibx[::kstep, ::hstep]

            # upts and vpts has a value for the left and right
            # sides of a cell. Sample every other value for quiver
            u = u[:, ::2]
            v = v[:, ::2]
            ib = ib[:, ::2]

        else:
            # kstep implementation for vertex grid
            projpts = {
                key: value
                for key, value in self.__cls.projpts.items()
                if (key // ncpl) % kstep == 0
            }

            # set x and z centers
            if isinstance(head, np.ndarray):
                # pipe kstep to set_zcentergrid to assure consistent array size
                zcenters = self.__cls.set_zcentergrid(
                    np.ravel(head), kstep=kstep
                )
            else:
                zcenters = [
                    np.mean(np.array(v).T[1])
                    for i, v in sorted(projpts.items())
                ]

            u = np.array([qx[cell] for cell in sorted(projpts)])

            x = np.array(
                [np.mean(np.array(v).T[0]) for i, v in sorted(projpts.items())]
            )

            z = np.ravel(zcenters)
            v = np.array([qz[cell] for cell in sorted(projpts)])
            ib = np.array([ib[cell] for cell in sorted(projpts)])

            x = x[::hstep]
            z = z[::hstep]
            u = u[::hstep]
            v = v[::hstep]
            ib = ib[::hstep]

        if normalize:
            vmag = np.sqrt(u ** 2.0 + v ** 2.0)
            idx = vmag > 0.0
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        # mask with an ibound array
        u[~ib] = np.nan
        v[~ib] = np.nan

        quiver = ax.quiver(x, z, u, v, pivot=pivot, **kwargs)

        return quiver

    def plot_discharge(
        self,
        frf,
        fff,
        flf=None,
        head=None,
        kstep=1,
        hstep=1,
        normalize=False,
        **kwargs
    ):
        """
        DEPRECATED. Use plot_vector() instead, which should follow after
        postprocessing.get_specific_discharge().

        Use quiver to plot vectors.

        Parameters
        ----------
        frf : numpy.ndarray
            MODFLOW's 'flow right face'
        fff : numpy.ndarray
            MODFLOW's 'flow front face'
        flf : numpy.ndarray
            MODFLOW's 'flow lower face' (Default is None.)
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        kstep : int
            layer frequency to plot. (Default is 1.)
        hstep : int
            horizontal frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors

        """
        import warnings

        warnings.warn(
            "plot_discharge() has been deprecated. Use "
            "plot_vector() instead, which should follow after "
            "postprocessing.get_specific_discharge()",
            DeprecationWarning,
        )

        if self.mg.grid_type != "structured":
            err_msg = "Use plot_specific_discharge for " "{} grids".format(
                self.mg.grid_type
            )
            raise NotImplementedError(err_msg)

        else:
            ib = np.ones((self.mg.nlay, self.mg.nrow, self.mg.ncol))
            if self.mg.idomain is not None:
                ib = self.mg.idomain

            delr = self.mg.delr
            delc = self.mg.delc
            top = self.mg.top
            botm = self.mg.botm
            if not np.all(self.active == 1):
                botm = botm[self.active == 1]
            nlay = botm.shape[0]
            laytyp = None
            hnoflo = 999.0
            hdry = 999.0

            if self.model is not None:
                if self.model.laytyp is not None:
                    laytyp = self.model.laytyp

                if self.model.hnoflo is not None:
                    hnoflo = self.model.hnoflo

                if self.model.hdry is not None:
                    hdry = self.model.hdry

            # If no access to head or laytyp, then calculate confined saturated
            # thickness by setting laytyp to zeros
            if head is None or laytyp is None:
                head = np.zeros(botm.shape, np.float32)
                laytyp = np.zeros((nlay), dtype=int)
                head[0, :, :] = top
                if nlay > 1:
                    head[1:, :, :] = botm[:-1, :, :]

            sat_thk = plotutil.PlotUtilities.saturated_thickness(
                head, top, botm, laytyp, [hnoflo, hdry]
            )

            # Calculate specific discharge
            qx, qy, qz = plotutil.PlotUtilities.centered_specific_discharge(
                frf, fff, flf, delr, delc, sat_thk
            )

            if qz is None:
                qz = np.zeros((qx.shape), dtype=float)

            ib = ib.ravel()
            qx = qx.ravel()
            qy = qy.ravel() * -1
            qz = qz.ravel()

            temp = []
            for ix, val in enumerate(ib):
                if val != 0:
                    temp.append((ix + 1, qx[ix], -qy[ix], qz[ix]))

            spdis = np.recarray(
                (len(temp),),
                dtype=[
                    ("node", int),
                    ("qx", float),
                    ("qy", float),
                    ("qz", float),
                ],
            )
            for ix, tup in enumerate(temp):
                spdis[ix] = tup

            self.plot_specific_discharge(
                spdis,
                head=head,
                kstep=kstep,
                hstep=hstep,
                normalize=normalize,
                **kwargs
            )

    def get_grid_patch_collection(self, zpts, plotarray, **kwargs):
        """
        Get a PatchCollection of plotarray in unmasked cells

        Parameters
        ----------
        zpts : numpy.ndarray
            array of z elevations that correspond to the x, y, and horizontal
            distance along the cross-section (self.xpts). Constructed using
            plotutil.cell_value_points().
        plotarray : numpy.ndarray
            Three-dimensional array to attach to the Patch Collection.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if self.mg.grid_type == "structured":
            return self.__cls.get_grid_patch_collection(
                zpts=zpts, plotarray=plotarray, **kwargs
            )
        elif self.mg.grid_type == "unstructured":
            raise NotImplementedError()

        else:
            return self.__cls.get_grid_patch_collection(
                projpts=zpts, plotarray=plotarray, **kwargs
            )

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid

        Parameters
        ----------
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.LineCollection

        Returns
        -------
        linecollection : matplotlib.collections.LineCollection
        """
        return self.__cls.get_grid_line_collection(**kwargs)


class DeprecatedCrossSection(PlotCrossSection):
    """
    Deprecation handler for the PlotCrossSection class

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
    model : flopy.modflow.Modflow object
    modelgrid : flopy.discretization.Grid object
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section.
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """

    def __init__(
        self, ax=None, model=None, modelgrid=None, line=None, extent=None
    ):
        super(DeprecatedCrossSection, self).__init__(
            ax=ax, model=model, modelgrid=modelgrid, line=line, extent=extent
        )
