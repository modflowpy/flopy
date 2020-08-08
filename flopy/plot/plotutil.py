"""
Module containing helper functions for plotting model data
using ModelMap and ModelCrossSection. Functions for plotting
shapefiles are also included.

"""
from __future__ import print_function
import os
import sys
import math
import numpy as np
from ..utils import Util3d
from ..datbase import DataType, DataInterface

try:
    import shapefile
except ImportError:
    shapefile = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

bc_color_dict = {
    "default": "black",
    "WEL": "red",
    "DRN": "yellow",
    "RIV": "teal",
    "GHB": "cyan",
    "CHD": "navy",
    "STR": "purple",
    "SFR": "teal",
    "UZF": "peru",
    "LAK": "royalblue",
}


class PlotException(Exception):
    def __init__(self, message):
        super(PlotException, self).__init__(message)


class PlotUtilities(object):
    """
    Class which groups a collection of plotting utilities
    which Flopy and Flopy6 can use to generate map based plots
    """

    @staticmethod
    def _plot_simulation_helper(simulation, model_list, SelPackList, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        model input data from a model instance

        Parameters
        ----------
        simulation : flopy.mf6.Simulation object
        model_list : list
            list of model names to plot
        SelPackList : list
            list of package names to plot, if none
            all packages will be plotted

        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return.
                (default is zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.
        """
        defaults = {
            "kper": 0,
            "mflay": None,
            "filename_base": None,
            "file_extension": "png",
            "key": None,
        }

        for key in defaults:
            if key in kwargs:
                if key == "file_extension":
                    defaults[key] = kwargs[key].replace(".", "")
                else:
                    defaults[key] = kwargs[key]

                kwargs.pop(key)

        filename_base = defaults["filename_base"]

        if model_list is None:
            model_list = simulation.model_names

        axes = []
        ifig = 0
        for model_name in model_list:
            model = simulation.get_model(model_name)

            model_filename_base = None
            if filename_base is not None:
                model_filename_base = filename_base + "_" + model_name

            if model.verbose:
                print("   Plotting Model:   ", model_name)

            caxs = PlotUtilities._plot_model_helper(
                model,
                SelPackList=SelPackList,
                kper=defaults["kper"],
                mflay=defaults["mflay"],
                filename_base=model_filename_base,
                file_extension=defaults["file_extension"],
                key=defaults["key"],
                initial_fig=ifig,
                model_name=model_name,
                **kwargs
            )

            if isinstance(caxs, list):
                for c in caxs:
                    axes.append(c)
            else:
                axes.append(caxs)

            ifig = len(axes) + 1

        return axes

    @staticmethod
    def _plot_model_helper(model, SelPackList, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        model input data from a model instance

        Parameters
        ----------
        model : Flopy model instance
        SelPackList : list
            list of package names to plot, if none
            all packages will be plotted

        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return.
                (default is zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.
        """
        # valid keyword arguments
        defaults = {
            "kper": 0,
            "mflay": None,
            "filename_base": None,
            "file_extension": "png",
            "key": None,
            "model_name": "",
            "initial_fig": 0,
        }

        for key in defaults:
            if key in kwargs:
                if key == "file_extension":
                    defaults[key] = kwargs[key].replace(".", "")
                else:
                    defaults[key] = kwargs[key]

                kwargs.pop(key)

        axes = []
        ifig = defaults["initial_fig"]
        if SelPackList is None:
            for p in model.packagelist:
                caxs = PlotUtilities._plot_package_helper(
                    p,
                    initial_fig=ifig,
                    filename_base=defaults["filename_base"],
                    file_extension=defaults["file_extension"],
                    kper=defaults["kper"],
                    mflay=defaults["mflay"],
                    key=defaults["key"],
                    model_name=defaults["model_name"],
                )
                # unroll nested lists of axes into a single list of axes
                if isinstance(caxs, list):
                    for c in caxs:
                        axes.append(c)
                else:
                    axes.append(caxs)
                # update next active figure number
                ifig = len(axes) + 1

        else:
            for pon in SelPackList:
                for p in model.packagelist:
                    if pon in p.name:
                        if model.verbose:
                            print("   Plotting Package: ", p.name[0])
                        caxs = PlotUtilities._plot_package_helper(
                            p,
                            initial_fig=ifig,
                            filename_base=defaults["filename_base"],
                            file_extension=defaults["file_extension"],
                            kper=defaults["kper"],
                            mflay=defaults["mflay"],
                            key=defaults["key"],
                            model_name=defaults["model_name"],
                        )

                        # unroll nested lists of axes into a single list of axes
                        if isinstance(caxs, list):
                            for c in caxs:
                                axes.append(c)
                        else:
                            axes.append(caxs)
                        # update next active figure number
                        ifig = len(axes) + 1
                        break
        if model.verbose:
            print(" ")
        return axes

    @staticmethod
    def _plot_package_helper(package, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        package input data

        Parameters
        ----------
        package: flopy.pakbase.Package
            package instance supplied for plotting

        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return. (default is
                zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.

        """
        defaults = {
            "kper": 0,
            "filename_base": None,
            "file_extension": "png",
            "mflay": None,
            "key": None,
            "initial_fig": 0,
            "model_name": "",
        }

        for key in defaults:
            if key in kwargs:
                if key == "file_extension":
                    defaults[key] = kwargs[key].replace(".", "")
                elif key == "initial_fig":
                    defaults[key] = int(kwargs[key])
                else:
                    defaults[key] = kwargs[key]

                kwargs.pop(key)

        model_name = defaults.pop("model_name")

        inc = package.parent.modelgrid.nlay
        if defaults["mflay"] is not None:
            inc = 1

        axes = []
        for item, value in package.__dict__.items():
            caxs = []
            # trap non-flopy specific data_types.

            if isinstance(value, list):
                for v in value:
                    if isinstance(v, Util3d):
                        if package.parent.verbose:
                            print(
                                "plotting {} package Util3d instance: {}".format(
                                    package.name[0], item
                                )
                            )
                        fignum = list(
                            range(
                                defaults["initial_fig"],
                                defaults["initial_fig"] + inc,
                            )
                        )
                        defaults["initial_fig"] = fignum[-1] + 1
                        caxs.append(
                            PlotUtilities._plot_util3d_helper(
                                v,
                                filename_base=defaults["filename_base"],
                                file_extension=defaults["file_extension"],
                                mflay=defaults["mflay"],
                                fignum=fignum,
                                model_name=model_name,
                                colorbar=True,
                            )
                        )

            elif isinstance(value, DataInterface):
                if (
                    value.data_type == DataType.transientlist
                ):  # isinstance(value, (MfList, MFTransientList)):
                    if package.parent.verbose:
                        print(
                            "plotting {} package MfList instance: {}".format(
                                package.name[0], item
                            )
                        )
                    if defaults["key"] is None:
                        names = [
                            "{} {} location stress period {} layer {}".format(
                                model_name,
                                package.name[0],
                                defaults["kper"] + 1,
                                k + 1,
                            )
                            for k in range(package.parent.modelgrid.nlay)
                        ]
                        colorbar = False
                    else:
                        names = [
                            "{} {} {} data stress period {} layer {}".format(
                                model_name,
                                package.name[0],
                                defaults["key"],
                                defaults["kper"] + 1,
                                k + 1,
                            )
                            for k in range(package.parent.modelgrid.nlay)
                        ]
                        colorbar = True

                    fignum = list(
                        range(
                            defaults["initial_fig"],
                            defaults["initial_fig"] + inc,
                        )
                    )
                    defaults["initial_fig"] = fignum[-1] + 1
                    # need to keep this as value.plot() because of mf6 datatype issues
                    ax = value.plot(
                        defaults["key"],
                        names,
                        defaults["kper"],
                        filename_base=defaults["filename_base"],
                        file_extension=defaults["file_extension"],
                        mflay=defaults["mflay"],
                        fignum=fignum,
                        colorbar=colorbar,
                        **kwargs
                    )

                    if ax is not None:
                        caxs.append(ax)

                elif (
                    value.data_type == DataType.array3d
                ):  # isinstance(value, Util3d):
                    if value.array is not None:
                        if package.parent.verbose:
                            print(
                                "plotting {} package Util3d instance: {}".format(
                                    package.name[0], item
                                )
                            )
                        # fignum = list(range(ifig, ifig + inc))
                        fignum = list(
                            range(
                                defaults["initial_fig"],
                                defaults["initial_fig"] + value.array.shape[0],
                            )
                        )
                        defaults["initial_fig"] = fignum[-1] + 1

                        caxs.append(
                            PlotUtilities._plot_util3d_helper(
                                value,
                                filename_base=defaults["filename_base"],
                                file_extension=defaults["file_extension"],
                                mflay=defaults["mflay"],
                                fignum=fignum,
                                model_name=model_name,
                                colorbar=True,
                            )
                        )

                elif (
                    value.data_type == DataType.array2d
                ):  # isinstance(value, Util2d):
                    if value.array is not None:
                        if len(value.array.shape) == 2:  # is this necessary?
                            if package.parent.verbose:
                                print(
                                    "plotting {} package Util2d instance: {}".format(
                                        package.name[0], item
                                    )
                                )
                            fignum = list(
                                range(
                                    defaults["initial_fig"],
                                    defaults["initial_fig"] + 1,
                                )
                            )
                            defaults["initial_fig"] = fignum[-1] + 1

                            caxs.append(
                                PlotUtilities._plot_util2d_helper(
                                    value,
                                    filename_base=defaults["filename_base"],
                                    file_extension=defaults["file_extension"],
                                    fignum=fignum,
                                    model_name=model_name,
                                    colorbar=True,
                                )
                            )

                elif (
                    value.data_type == DataType.transient2d
                ):  # isinstance(value, Transient2d):
                    if value.array is not None:
                        if package.parent.verbose:
                            print(
                                "plotting {} package Transient2d instance: {}".format(
                                    package.name[0], item
                                )
                            )
                        fignum = list(
                            range(
                                defaults["initial_fig"],
                                defaults["initial_fig"] + inc,
                            )
                        )
                        defaults["initial_fig"] = fignum[-1] + 1

                        caxs.append(
                            PlotUtilities._plot_transient2d_helper(
                                value,
                                filename_base=defaults["filename_base"],
                                file_extension=defaults["file_extension"],
                                kper=defaults["kper"],
                                fignum=fignum,
                                colorbar=True,
                            )
                        )

                else:
                    pass

            else:
                pass

            # unroll nested lists os axes into a single list of axes
            if isinstance(caxs, list):
                for c in caxs:
                    if isinstance(c, list):
                        for cc in c:
                            axes.append(cc)
                    else:
                        axes.append(c)
            else:
                axes.append(caxs)

        return axes

    @staticmethod
    def _plot_mflist_helper(
        mflist,
        key=None,
        names=None,
        kper=0,
        filename_base=None,
        file_extension=None,
        mflay=None,
        **kwargs
    ):
        """
        Plot stress period boundary condition (MfList) data for a specified
        stress period

        Parameters
        ----------
        mflist: flopy.utils.util_list.MfList object

        key : str
            MfList dictionary key. (default is None)
        names : list
            List of names for figure titles. (default is None)
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        mflay : int
            MODFLOW zero-based layer number to return.  If None, then all
            all layers will be included. (default is None)
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        """
        if file_extension is not None:
            fext = file_extension
        else:
            fext = "png"

        model_name = ""
        if "model_name" in kwargs:
            model_name = kwargs.pop("model_name") + " "

        filenames = None
        if filename_base is not None:
            if mflay is not None:
                i0 = int(mflay)
                if i0 + 1 >= mflist.model.modelgrid.nlay:
                    i0 = mflist.model.modelgrid.nlay - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = mflist.model.modelgrid.nlay
            # build filenames
            package_name = mflist.package.name[0].upper()
            filenames = [
                "{}_{}_StressPeriod{}_Layer{}.{}".format(
                    filename_base, package_name, kper + 1, k + 1, fext
                )
                for k in range(i0, i1)
            ]

        if names is None:
            if key is None:
                names = [
                    "{}{} location stress period: {} layer: {}".format(
                        model_name, mflist.package.name[0], kper + 1, k + 1
                    )
                    for k in range(mflist.model.modelgrid.nlay)
                ]
            else:
                names = [
                    "{}{} {} stress period: {} layer: {}".format(
                        model_name,
                        mflist.package.name[0],
                        key,
                        kper + 1,
                        k + 1,
                    )
                    for k in range(mflist.model.modelgrid.nlay)
                ]

        if key is None:
            axes = PlotUtilities._plot_bc_helper(
                mflist.package,
                kper,
                names=names,
                filenames=filenames,
                mflay=mflay,
                **kwargs
            )
        else:
            arr_dict = mflist.to_array(kper, mask=True)

            try:
                arr = arr_dict[key]
            except:
                err_msg = "Cannot find key to plot\n"
                err_msg += "  Provided key={}\n  Available keys=".format(key)
                for name, arr in arr_dict.items():
                    err_msg += "{}, ".format(name)
                err_msg += "\n"
                raise PlotException(err_msg)

            axes = PlotUtilities._plot_array_helper(
                arr,
                model=mflist.model,
                names=names,
                filenames=filenames,
                mflay=mflay,
                **kwargs
            )
        return axes

    @staticmethod
    def _plot_util2d_helper(
        util2d,
        title=None,
        filename_base=None,
        file_extension=None,
        fignum=None,
        **kwargs
    ):
        """
        Plot 2-D model input data

        Parameters
        ----------
        util2d : flopy.util.util_array.Util2d object
        title : str
            Plot title. If a plot title is not provide one will be
            created based on data name (self.name). (default is None)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        fignum : list
            list of figure numbers
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        """
        model_name = ""
        if "model_name" in kwargs:
            model_name = kwargs.pop("model_name") + " "

        if title is None:
            title = "{}{}".format(model_name, util2d.name)

        if file_extension is not None:
            fext = file_extension
        else:
            fext = "png"

        filename = None
        if filename_base is not None:
            filename = "{}_{}.{}".format(filename_base, util2d.name, fext)

        axes = PlotUtilities._plot_array_helper(
            util2d.array,
            util2d.model,
            names=title,
            filenames=filename,
            fignum=fignum,
            **kwargs
        )
        return axes

    @staticmethod
    def _plot_util3d_helper(
        util3d,
        filename_base=None,
        file_extension=None,
        mflay=None,
        fignum=None,
        **kwargs
    ):
        """
        Plot 3-D model input data

        Parameters
        ----------
        util3d : flopy.util.util_array.Util3d object
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        mflay : int
            MODFLOW zero-based layer number to return.  If None, then all
            all layers will be included. (default is None)
        fignum : list
            list of figure numbers
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        """
        model_name = ""
        if "model_name" in kwargs:
            model_name = kwargs.pop("model_name")

        if file_extension is not None:
            fext = file_extension
        else:
            fext = "png"

        # flopy6 adaption
        array = util3d.array
        name = util3d.name
        if isinstance(name, str):
            name = [name] * array.shape[0]

        names = [
            "{}{} layer {}".format(model_name, name[k], k + 1)
            for k in range(array.shape[0])
        ]

        filenames = None
        if filename_base is not None:
            if mflay is not None:
                i0 = int(mflay)
                if i0 + 1 >= array.shape[0]:
                    i0 = array.shape[0] - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = array.shape[0]
            # build filenames, use local "name" variable (flopy6 adaptation)
            filenames = [
                "{}_{}_Layer{}.{}".format(filename_base, name[k], k + 1, fext)
                for k in range(i0, i1)
            ]

        axes = PlotUtilities._plot_array_helper(
            array,
            util3d.model,
            names=names,
            filenames=filenames,
            mflay=mflay,
            fignum=fignum,
            **kwargs
        )
        return axes

    @staticmethod
    def _plot_transient2d_helper(
        transient2d,
        filename_base=None,
        file_extension=None,
        kper=0,
        fignum=None,
        **kwargs
    ):
        """
        Plot transient 2-D model input data

        Parameters
        ----------
        transient2d : flopy.utils.util_array.Transient2D object
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        kper : int
            zero based stress period number
        fignum : list
            list of figure numbers
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.
            kper : str
                MODFLOW zero-based stress period number to return. If
                kper='all' then data for all stress period will be
                extracted. (default is zero).

        Returns
        -------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        """
        if file_extension is not None:
            fext = file_extension
        else:
            fext = "png"

        if isinstance(kper, int):
            k0 = kper
            k1 = kper + 1

        elif isinstance(kper, str):
            if kper.lower() == "all":
                k0 = 0
                k1 = transient2d.model.nper

            else:
                k0 = int(kper)
                k1 = k0 + 1

        else:
            k0 = int(kper)
            k1 = k0 + 1

        if fignum is not None:
            if not isinstance(fignum, list):
                fignum = list(fignum)
        else:
            fignum = list(range(k0, k1))

        if "mflay" in kwargs:
            kwargs.pop("mflay")

        axes = []
        for idx, kper in enumerate(range(k0, k1)):
            title = "{} stress period {:d}".format(
                transient2d.name.replace("_", "").upper(), kper + 1
            )

            if filename_base is not None:
                filename = filename_base + "_{:05d}.{}".format(kper + 1, fext)
            else:
                filename = None

            axes.append(
                PlotUtilities._plot_array_helper(
                    transient2d.array[kper],
                    transient2d.model,
                    names=title,
                    filenames=filename,
                    fignum=fignum[idx],
                    **kwargs
                )
            )
        return axes

    @staticmethod
    def _plot_scalar_helper(
        scalar, filename_base=None, file_extension=None, **kwargs
    ):
        """
        Helper method to plot scalar objects

        Parameters
        ----------
        scalar : flopy.mf6.data.mfscalar object
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')

        Returns
        -------
         axes: list matplotlib.axes object

        """
        if file_extension is not None:
            fext = file_extension
        else:
            fext = "png"

        if "mflay" in kwargs:
            kwargs.pop("mflay")

        title = "{}".format(scalar.name.replace("_", "").upper())

        if filename_base is not None:
            filename = filename_base + ".{}".format(fext)
        else:
            filename = None

        axes = PlotUtilities._plot_array_helper(
            scalar.array,
            scalar.model,
            names=title,
            filenames=filename,
            **kwargs
        )
        return axes

    @staticmethod
    def _plot_array_helper(
        plotarray,
        model=None,
        modelgrid=None,
        axes=None,
        names=None,
        filenames=None,
        fignum=None,
        mflay=None,
        **kwargs
    ):
        """
        Helper method to plot array objects

        Parameters
        ----------
        plotarray : np.array object
        model: fp.modflow.Modflow object
            optional if spatial reference is provided
        modelgrid: fp.discretization.ModelGrid object
            object that defines the spatial orientation of a modflow
            grid within flopy. Optional if model object is provided
        axes: matplotlib.axes object
            existing matplotlib axis object to layer additional
            plotting on to. Optional.
        names: list
            list of figure titles (optional)
        filenames: list
            list of filenames to save figures to (optional)
        fignum:
            list of figure numbers (optional)
        mflay: int
            modflow model layer
        **kwargs:
            keyword arguments

        Returns:
         axes: list matplotlib.axes object

        """
        from .map import PlotMapView

        defaults = {
            "figsize": None,
            "masked_values": None,
            "pcolor": True,
            "inactive": True,
            "contour": False,
            "clabel": False,
            "colorbar": False,
            "grid": False,
            "levels": None,
            "colors": "black",
            "dpi": None,
            "fmt": "%1.3f",
            "modelgrid": None,
        }

        # check that matplotlib is installed
        if plt is None:
            err_msg = (
                "Could not import matplotlib. "
                "Must install matplotlib "
                + " in order to plot LayerFile data."
            )
            raise PlotException(err_msg)

        for key in defaults:
            if key in kwargs:
                defaults[key] = kwargs.pop(key)

        plotarray = plotarray.astype(float)

        # test if this is vertex or structured grid
        if model is not None:
            grid_type = model.modelgrid.grid_type
            hnoflo = model.hnoflo
            hdry = model.hdry
            if defaults["masked_values"] is None:
                t = []
                if hnoflo is not None:
                    t.append(hnoflo)
                if hdry is not None:
                    t.append(hdry)
                if t:
                    defaults["masked_values"] = t
            else:
                if hnoflo is not None:
                    defaults["masked_values"].append(hnoflo)
                if hdry is not None:
                    defaults["masked_values"].append(hdry)

        elif modelgrid is not None:
            grid_type = modelgrid.grid_type

        else:
            grid_type = "structured"

        ib = None
        if modelgrid is not None:
            if modelgrid.idomain is not None:
                ib = modelgrid.idomain

        else:
            if ib is None:
                try:
                    ib = model.modelgrid.idomain
                except:
                    pass

        # reshape 2d arrays to 3d for convenience
        if len(plotarray.shape) == 2 and grid_type == "structured":
            plotarray = plotarray.reshape(
                (1, plotarray.shape[0], plotarray.shape[1])
            )

        # setup plotting routines
        # consider refactoring maxlay to nlay
        maxlay = plotarray.shape[0]
        i0, i1 = PlotUtilities._set_layer_range(mflay, maxlay)
        names = PlotUtilities._set_names(names, maxlay)
        filenames = PlotUtilities._set_names(filenames, maxlay)
        fignum = PlotUtilities._set_fignum(fignum, maxlay, i0, i1)
        axes = PlotUtilities._set_axes(
            axes, mflay, maxlay, i0, i1, defaults, names, fignum
        )

        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(num=fignum[idx])
            pmv = PlotMapView(
                ax=axes[idx], model=model, modelgrid=modelgrid, layer=k
            )
            if defaults["pcolor"]:
                cm = pmv.plot_array(
                    plotarray[k],
                    masked_values=defaults["masked_values"],
                    ax=axes[idx],
                    **kwargs
                )

                if defaults["colorbar"]:
                    label = ""
                    if not isinstance(defaults["colorbar"], bool):
                        label = str(defaults["colorbar"])
                    plt.colorbar(cm, ax=axes[idx], shrink=0.5, label=label)

            if defaults["contour"]:
                cl = pmv.contour_array(
                    plotarray[k],
                    masked_values=defaults["masked_values"],
                    ax=axes[idx],
                    colors=defaults["colors"],
                    levels=defaults["levels"],
                    **kwargs
                )
                if defaults["clabel"]:
                    axes[idx].clabel(cl, fmt=defaults["fmt"], **kwargs)

            if defaults["grid"]:
                pmv.plot_grid(ax=axes[idx])

            if defaults["inactive"]:
                if ib is not None:
                    pmv.plot_inactive(ibound=ib, ax=axes[idx])

        if len(axes) == 1:
            axes = axes[0]

        if filenames is not None:
            for idx, k in enumerate(range(i0, i1)):
                fig = plt.figure(num=fignum[idx])
                fig.savefig(filenames[idx], dpi=defaults["dpi"])
                print(
                    "    created...{}".format(os.path.basename(filenames[idx]))
                )
            # there will be nothing to return when done
            axes = None
            plt.close("all")

        return axes

    @staticmethod
    def _plot_bc_helper(
        package,
        kper,
        axes=None,
        names=None,
        filenames=None,
        fignum=None,
        mflay=None,
        **kwargs
    ):
        """
        Helper method to plot bc objects from flopy packages

        Parameters
        ----------
        package : flopy.pakbase.Package objects
        kper : int
            zero based stress period number
        axes: matplotlib.axes object
            existing matplotlib axis object to layer additional
            plotting on to. Optional.
        names: list
            list of figure titles (optional)
        filenames: list
            list of filenames to save figures to (optional)
        fignum:
            list of figure numbers (optional)
        mflay: int
            modflow model layer
        **kwargs:
            keyword arguments

        Returns
        -------
        axes: list matplotlib.axes object
        """

        from .map import PlotMapView

        if plt is None:
            s = (
                "Could not import matplotlib.  Must install matplotlib "
                + " in order to plot boundary condition data."
            )
            raise PlotException(s)

        defaults = {
            "figsize": None,
            "inactive": True,
            "grid": False,
            "dpi": None,
            "masked_values": None,
        }

        # parse kwargs
        for key in defaults:
            if key in kwargs:
                defaults[key] = kwargs.pop(key)

        ftype = package.name[0]

        color = "black"
        if "CHD" in ftype.upper():
            color = bc_color_dict[ftype.upper()[:3]]

        # flopy-modflow vs. flopy-modflow6 trap
        try:
            model = package.parent
        except AttributeError:
            model = package._model_or_sim

        nlay = model.modelgrid.nlay

        # set up plotting routines
        i0, i1 = PlotUtilities._set_layer_range(mflay, nlay)
        names = PlotUtilities._set_names(names, nlay)
        filenames = PlotUtilities._set_names(filenames, i1 - i0)
        fignum = PlotUtilities._set_fignum(fignum, i1 - i0, i0, i1)
        axes = PlotUtilities._set_axes(
            axes, mflay, nlay, i0, i1, defaults, names, fignum
        )

        for idx, k in enumerate(range(i0, i1)):
            pmv = PlotMapView(ax=axes[idx], model=model, layer=k)
            fig = plt.figure(num=fignum[idx])
            pmv.plot_bc(
                ftype=ftype,
                package=package,
                kper=kper,
                ax=axes[idx],
                color=color,
            )

            if defaults["grid"]:
                pmv.plot_grid(ax=axes[idx])

            if defaults["inactive"]:
                if model.modelgrid is not None:
                    ib = model.modelgrid.idomain
                    if ib is not None:
                        pmv.plot_inactive(ibound=ib, ax=axes[idx])

        if len(axes) == 1:
            axes = axes[0]

        if filenames is not None:
            for idx, k in enumerate(range(i0, i1)):
                fig = plt.figure(num=fignum[idx])
                fig.savefig(filenames[idx], dpi=defaults["dpi"])
                plt.close(fignum[idx])
                print(
                    "    created...{}".format(os.path.basename(filenames[idx]))
                )
            # there will be nothing to return when done
            axes = None
            plt.close("all")

        return axes

    @staticmethod
    def _set_layer_range(mflay, maxlay):
        """
        Re-usable method to check for mflay and set
        the range of plottable layers

        Parameters
        ----------
        mflay : int
            zero based layer number
        maxlay : int
            maximum number of layers in the plotting array

        Returns
        -------
        i0, i1 :  int, int
            minimum and maximum bounds on the layer range

        """
        if mflay is not None:
            i0 = int(mflay)
            if i0 + 1 >= maxlay:
                i0 = maxlay - 1
            i1 = i0 + 1
        else:
            i0 = 0
            i1 = maxlay

        return i0, i1

    @staticmethod
    def _set_names(names, maxlay):
        """
        Checks the supplied name variable for shape

        Parameters
        ----------
        names : list of str
            if names is not none, asserts that there is
            a name supplied for each plot that will be
            generated

        maxlay : int
            maximum number of layers in the plotting array

        Returns
        -------
        names :  list or None
            list of names or None

        """
        if names is not None:
            if not isinstance(names, list):
                if maxlay > 1:
                    names = [
                        "{} layer {}".format(names, i + 1)
                        for i in range(maxlay)
                    ]
                else:
                    names = [names]
            assert len(names) == maxlay
        return names

    @staticmethod
    def _set_fignum(fignum, maxlay, i0, i1):
        """
        Method to generate a list of matplotlib figure
        numbers to join to figure objects. Checks
        for existing figures.

        Parameters
        ----------
        fignum : list
            list of figure numbers
        maxlay : int
            maximum number of layers in the plotting array
        i0 : int
            minimum layer range
        i1 : int
            maximum layer range

        Returns
        -------
        fignum : list

        """
        if fignum is not None:
            if not isinstance(fignum, list):
                fignum = [fignum]
            assert len(fignum) == maxlay
            # check for existing figures
            f0 = fignum[0]
            for i in plt.get_fignums():
                if i >= f0:
                    f0 = i + 1
            finc = f0 - fignum[0]
            for idx, _ in enumerate(fignum):
                fignum[idx] += finc
        else:
            # check for existing figures
            f0 = 0
            for i in plt.get_fignums():
                if i >= f0:
                    f0 += 1
            f1 = f0 + (i1 - i0)
            fignum = np.arange(f0, f1)

        return fignum

    @staticmethod
    def _set_axes(axes, mflay, maxlay, i0, i1, defaults, names, fignum):
        """
        Method to prepare axes objects for plotting

        Parameters
        ----------
        axes : list
            matplotlib.axes objects
        mflay : int
            layer to plot or None
        i0 : int
            minimum range of layers to plot
        i1 : int
            maximum range of layers to plot
        defaults : dict
            the default dictionary from the parent plotting method
        fignum : list
            list of figure numbers

        Returns
        -------
        axes : list
            matplotlib.axes objects

        """
        if axes is not None:
            if not isinstance(axes, list):
                axes = [axes]
            assert len(axes) == maxlay

        else:
            # prepare some axis objects for use
            axes = []
            for idx, k in enumerate(range(i0, i1)):
                plt.figure(figsize=defaults["figsize"], num=fignum[idx])
                ax = plt.subplot(1, 1, 1, aspect="equal")
                if names is not None:
                    title = names[k]
                else:
                    klay = k
                    if mflay is not None:
                        klay = int(mflay)
                    title = "{} Layer {}".format("data", klay + 1)
                ax.set_title(title)
                axes.append(ax)

        return axes

    @staticmethod
    def saturated_thickness(head, top, botm, laytyp, mask_values=None):
        """
        Calculate the saturated thickness.

        Parameters
        ----------
        head : numpy.ndarray
            head array
        top : numpy.ndarray
            top array of shape (nrow, ncol)
        botm : numpy.ndarray
            botm array of shape (nlay, nrow, ncol)
        laytyp : numpy.ndarray
            confined (0) or convertible (1) of shape (nlay)
        mask_values : list of floats
            If head is one of these values, then set sat to top - bot

        Returns
        -------
        sat_thk : numpy.ndarray
            Saturated thickness of shape (nlay, nrow, ncol).

        """
        if head.ndim == 3:
            head = np.copy(head)
            nlay, nrow, ncol = head.shape
            ncpl = nrow * ncol
            head.shape = (nlay, ncpl)
            top.shape = (ncpl,)
            botm.shape = (nlay, ncpl)
            if laytyp.ndim == 3:
                laytyp.shape = (nlay, ncpl)

        else:
            nrow, ncol = None, None
            nlay, ncpl = head.shape

        # cast a laytyp flag for each cell if modflow-2005 based,
        # which makes it consistent with the mf6 iconvert array
        if laytyp.ndim == 1:
            t = np.zeros(head.shape)
            for ix, _ in enumerate(laytyp):
                t[ix, :] = laytyp[ix]
            laytyp = t
            del t

        sat_thk_conf = np.empty(head.shape, dtype=head.dtype)
        sat_thk_unconf = np.empty(head.shape, dtype=head.dtype)

        for k in range(nlay):
            if k == 0:
                t = top
            else:
                t = botm[k - 1, :]
            sat_thk_conf[k, :] = t - botm[k, :]

        for k in range(nlay):
            dh = np.zeros((ncpl,), dtype=head.dtype)
            s = sat_thk_conf[k, :]

            for mv in mask_values:
                idx = head[k, :] == mv
                dh[idx] = s[idx]

            if k == 0:
                t = top
            else:
                t = botm[k - 1, :]

            t = np.where(head[k, :] > t, t, head[k, :])
            dh = np.where(dh == 0, t - botm[k, :], dh)
            sat_thk_unconf[k, :] = dh[:]

        sat_thk = np.where(laytyp != 0, sat_thk_unconf, sat_thk_conf)

        if nrow is not None and ncol is not None:
            sat_thk.shape = (nlay, nrow, ncol)

        return sat_thk

    @staticmethod
    def centered_specific_discharge(Qx, Qy, Qz, delr, delc, sat_thk):
        """
        DEPRECATED. Use postprocessing.get_specific_discharge() instead.

        Using the MODFLOW discharge, calculate the cell centered specific discharge
        by dividing by the flow width and then averaging to the cell center.

        Parameters
        ----------
        Qx : numpy.ndarray
            MODFLOW 'flow right face'
        Qy : numpy.ndarray
            MODFLOW 'flow front face'.  The sign on this array will be flipped
            by this function so that the y axis is positive to north.
        Qz : numpy.ndarray
            MODFLOW 'flow lower face'.  The sign on this array will be flipped by
            this function so that the z axis is positive in the upward direction.
        delr : numpy.ndarray
            MODFLOW delr array
        delc : numpy.ndarray
            MODFLOW delc array
        sat_thk : numpy.ndarray
            Saturated thickness for each cell

        Returns
        -------
        (qx, qy, qz) : tuple of numpy.ndarrays
            Specific discharge arrays that have been interpolated to cell centers.

        """
        import warnings

        warnings.warn(
            "centered_specific_discharge() has been deprecated. Use "
            "postprocessing.get_specific_discharge() instead.",
            DeprecationWarning,
        )

        qx = None
        qy = None
        qz = None

        if Qx is not None:

            nlay, nrow, ncol = Qx.shape
            qx = np.zeros(Qx.shape, dtype=Qx.dtype)

            for k in range(nlay):
                for j in range(ncol - 1):
                    area = (
                        delc[:]
                        * 0.5
                        * (sat_thk[k, :, j] + sat_thk[k, :, j + 1])
                    )
                    idx = area > 0.0
                    qx[k, idx, j] = Qx[k, idx, j] / area[idx]

            qx[:, :, 1:] = 0.5 * (qx[:, :, 0 : ncol - 1] + qx[:, :, 1:ncol])
            qx[:, :, 0] = 0.5 * qx[:, :, 0]

        if Qy is not None:

            nlay, nrow, ncol = Qy.shape
            qy = np.zeros(Qy.shape, dtype=Qy.dtype)

            for k in range(nlay):
                for i in range(nrow - 1):
                    area = (
                        delr[:]
                        * 0.5
                        * (sat_thk[k, i, :] + sat_thk[k, i + 1, :])
                    )
                    idx = area > 0.0
                    qy[k, i, idx] = Qy[k, i, idx] / area[idx]

            qy[:, 1:, :] = 0.5 * (qy[:, 0 : nrow - 1, :] + qy[:, 1:nrow, :])
            qy[:, 0, :] = 0.5 * qy[:, 0, :]
            qy = -qy

        if Qz is not None:
            qz = np.zeros(Qz.shape, dtype=Qz.dtype)
            dr = delr.reshape((1, delr.shape[0]))
            dc = delc.reshape((delc.shape[0], 1))
            area = dr * dc
            for k in range(nlay):
                qz[k, :, :] = Qz[k, :, :] / area[:, :]
            qz[1:, :, :] = 0.5 * (qz[0 : nlay - 1, :, :] + qz[1:nlay, :, :])
            qz[0, :, :] = 0.5 * qz[0, :, :]
            qz = -qz

        return (qx, qy, qz)


class UnstructuredPlotUtilities(object):
    """
    Collection of unstructured grid and vertex grid compatible
    plotting helper functions
    """

    @staticmethod
    def line_intersect_grid(ptsin, xgrid, ygrid):
        """
        Uses cross product method to find which cells intersect with the
        line and then uses the parameterized line equation to caluculate
        intersection x, y vertex points. Should be quite fast for large model
        grids!

        Parameters
        ----------
        pts : list
            list of tuple line vertex pairs (ex. [(1, 0), (10, 0)]
        xgrid : np.array
            model grid x vertices
        ygrid : np.array
            model grid y vertices

        Returns
        -------
        vdict : dict of cell vertices

        """
        # make sure xedge and yedge are numpy arrays
        if not isinstance(xgrid, np.ndarray):
            xgrid = np.array(xgrid)
        if not isinstance(ygrid, np.ndarray):
            ygrid = np.array(ygrid)

        npts = len(ptsin)

        # use a vector cross product to find which
        # cells intersect the line
        vdict = {}
        for ix in range(1, npts):
            xmin = np.min([ptsin[ix - 1][0], ptsin[ix][0]])
            xmax = np.max([ptsin[ix - 1][0], ptsin[ix][0]])
            ymin = np.min([ptsin[ix - 1][1], ptsin[ix][1]])
            ymax = np.max([ptsin[ix - 1][1], ptsin[ix][1]])
            x1 = np.ones(xgrid.shape) * ptsin[ix - 1][0]
            y1 = np.ones(ygrid.shape) * ptsin[ix - 1][1]
            x2 = np.ones(xgrid.shape) * ptsin[ix][0]
            y2 = np.ones(ygrid.shape) * ptsin[ix][1]
            x3 = xgrid
            y3 = ygrid
            x4 = np.zeros(xgrid.shape)
            y4 = np.zeros(ygrid.shape)
            x4[:, :-1] = xgrid[:, 1:]
            x4[:, -1] = xgrid[:, 0]
            y4[:, :-1] = ygrid[:, 1:]
            y4[:, -1] = ygrid[:, 0]

            # find where intersection is
            v1 = [x2 - x1, y2 - y1]
            v2 = [x2 - x3, y2 - y3]
            xp = v1[0] * v2[1] - v1[1] * v2[0]

            # loop finds which edges the line intersects
            cells = []
            cell_vertex_ix = []
            for cell, cpv in enumerate(xp):
                if np.all([t < 0 for t in cpv]):
                    continue
                elif np.all([t > 0 for t in cpv]):
                    continue

                else:
                    # only cycle through the cells that intersect
                    # the infinite line
                    cvert_ix = []
                    for ix in range(len(cpv)):
                        if cpv[ix - 1] < 0 and cpv[ix] > 0:
                            cvert_ix.append(ix - 1)
                        elif cpv[ix - 1] > 0 and cpv[ix] < 0:
                            cvert_ix.append(ix - 1)
                        elif cpv[ix - 1] == 0 and cpv[ix] == 0:
                            cvert_ix += [ix - 1, ix]
                        else:
                            pass

                    if cvert_ix:
                        cells.append(cell)
                        cell_vertex_ix.append(cvert_ix)

            # find interesection vertices
            numa = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            numb = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            ua = numa / denom
            # ub = numb / denom
            del numa
            del numb
            del denom

            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)

            for ix, cell in enumerate(cells):
                xc = x[cell]
                yc = y[cell]
                verts = [
                    (xt, yt)
                    for xt, yt in zip(
                        xc[cell_vertex_ix[ix]], yc[cell_vertex_ix[ix]]
                    )
                ]

                if cell in vdict:
                    for i in verts:
                        # finally check that verts are
                        # within the line segment range
                        if i[0] < xmin or i[0] > xmax:
                            continue
                        elif i[1] < ymin or i[1] > ymax:
                            continue
                        elif i in vdict[cell]:
                            continue
                        elif (
                            np.isnan(i[0])
                            or np.isinf(i[0])
                            or np.isinf(i[1])
                            or np.isnan(i[1])
                        ):
                            continue
                        else:
                            vdict[cell].append(i)
                else:
                    # finally check that verts are
                    # within the line segment range
                    t = []
                    for i in verts:
                        if i[0] < xmin or i[0] > xmax:
                            continue
                        elif i[1] < ymin or i[1] > ymax:
                            continue
                        elif i in t:
                            continue
                        elif (
                            np.isnan(i[0])
                            or np.isinf(i[0])
                            or np.isinf(i[1])
                            or np.isnan(i[1])
                        ):
                            continue
                        else:
                            t.append(i)

                    if t:
                        vdict[cell] = t

        return vdict

    @staticmethod
    def irregular_shape_patch(xverts, yverts):
        """
        Patch for vertex cross section plotting when
        we have an irregular shape type throughout the
        model grid or multiple shape types.

        Parameters
        ----------
        xverts : list
            xvertices
        yverts : list
            yvertices

        Returns
        -------
            xverts, yverts as np.ndarray

        """
        max_verts = 0

        for xv in xverts:
            if len(xv) > max_verts:
                max_verts = len(xv)

        for yv in yverts:
            if len(yv) > max_verts:
                max_verts = len(yv)

        adj_xverts = []
        for xv in xverts:
            if len(xv) < max_verts:
                n = max_verts - len(xv)
                adj_xverts.append(xv + [xv[-1]] * n)
            else:
                adj_xverts.append(xv)

        adj_yverts = []
        for yv in yverts:
            if len(yv) < max_verts:
                n = max_verts - len(yv)
                adj_yverts.append(yv + [yv[-1]] * n)
            else:
                adj_yverts.append(yv)

        xverts = np.array(adj_xverts)
        yverts = np.array(adj_yverts)

        return xverts, yverts

    @staticmethod
    def arctan2(verts):
        """
        Reads 2 dimensional set of verts and orders them using the arctan 2 method

        Parameters
        ----------
        verts : np.array of floats
            Nx2 array of verts

        Returns
        -------
        verts : np.array of float
            Nx2 array of verts

        """
        center = verts.mean(axis=0)
        x = verts.T[0] - center[0]
        z = verts.T[1] - center[1]

        angles = np.arctan2(z, x) * 180 / np.pi
        angleidx = angles.argsort()

        verts = verts[angleidx]
        return verts


class SwiConcentration:
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files

    """

    def __init__(self, model=None, botm=None, istrat=1, nu=None):
        if model is None:
            if isinstance(botm, list):
                botm = np.array(botm)
            self.__botm = botm
            if isinstance(nu, list):
                nu = np.array(nu)
            self.__nu = nu
            self.__istrat = istrat
            if istrat == 1:
                self.__nsrf = self.nu.shape - 1
            else:
                self.__nsrf = self.nu.shape - 2
        else:
            try:
                dis = model.get_package("DIS")
            except:
                sys.stdout.write("Error: DIS package not available.\n")
            self.__botm = np.zeros(
                (dis.nlay + 1, dis.nrow, dis.ncol), np.float
            )
            self.__botm[0, :, :] = dis.top.array
            self.__botm[1:, :, :] = dis.botm.array
            try:
                swi = model.get_package("SWI2")
                self.__nu = swi.nu.array
                self.__istrat = swi.istrat
                self.__nsrf = swi.nsrf
            except (AttributeError, ValueError):
                sys.stdout.write("Error: SWI2 package not available...\n")
        self.__nlay = self.__botm.shape[0] - 1
        self.__nrow = self.__botm[0, :, :].shape[0]
        self.__ncol = self.__botm[0, :, :].shape[1]
        self.__b = self.__botm[0:-1, :, :] - self.__botm[1:, :, :]

    def calc_conc(self, zeta, layer=None):
        """
        Calculate concentrations for a given time step using passed zeta.

        Parameters
        ----------
        zeta : dictionary of numpy arrays
            Dictionary of zeta results. zeta keys are zero-based zeta surfaces.
        layer : int
            Concentration will be calculated for the specified layer.  If layer
            is None, then the concentration will be calculated for all layers.
            (default is None).

        Returns
        -------
        conc : numpy array
            Calculated concentration.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('test')
        >>> c = flopy.plot.SwiConcentration(model=m)
        >>> conc = c.calc_conc(z, layer=0)

        """
        conc = np.zeros((self.__nlay, self.__nrow, self.__ncol), np.float)

        pct = {}
        for isrf in range(self.__nsrf):
            z = zeta[isrf]
            pct[isrf] = (self.__botm[:-1, :, :] - z[:, :, :]) / self.__b[
                :, :, :
            ]
        for isrf in range(self.__nsrf):
            p = pct[isrf]
            if self.__istrat == 1:
                conc[:, :, :] += self.__nu[isrf] * p[:, :, :]
                if isrf + 1 == self.__nsrf:
                    conc[:, :, :] += self.__nu[isrf + 1] * (1.0 - p[:, :, :])
            # TODO linear option
        if layer is None:
            return conc
        else:
            return conc[layer, :, :]


def shapefile_extents(shp):
    """
    Determine the extents of a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.

    Returns
    -------
    extents : tuple
        tuple with xmin, xmax, ymin, ymax from shapefile.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> extent = flopy.plot.plotutil.shapefile_extents(fshp)

    """
    if shapefile is None:
        s = "Could not import shapefile.  Must install pyshp in order to plot shapefiles."
        raise PlotException(s)

    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    xmin, xmax, ymin, ymax = 1.0e20, -1.0e20, 1.0e20, -1.0e20

    for n in range(nshp):
        for p in shapes[n].points:
            xmin, xmax = min(xmin, p[0]), max(xmax, p[0])
            ymin, ymax = min(ymin, p[1]), max(ymax, p[1])
    return xmin, xmax, ymin, ymax


def shapefile_get_vertices(shp):
    """
    Get vertices for the features in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to extract shapefile feature vertices.

    Returns
    -------
    vertices : list
        Vertices is a list with vertices for each feature in the shapefile.
        Individual feature vertices are x, y tuples and contained in a list.
        A list with a single x, y tuple is returned for point shapefiles. A
        list with multiple x, y tuples is returned for polyline and polygon
        shapefiles.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> lines = flopy.plot.plotutil.shapefile_get_vertices(fshp)

    """
    if shapefile is None:
        s = "Could not import shapefile.  Must install pyshp in order to plot shapefiles."
        raise PlotException(s)

    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    vertices = []
    for n in range(nshp):
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            # points
            for p in shapes[n].points:
                vertices.append([(p[0], p[1])])
        elif st in [3, 13, 23]:
            # line
            line = []
            for p in shapes[n].points:
                line.append((p[0], p[1]))
            line = np.array(line)
            vertices.append(line)
        elif st in [5, 25, 31]:
            # polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                vertices.append(pts[par[pij] : par[pij + 1]])
    return vertices


def shapefile_to_patch_collection(shp, radius=500.0, idx=None):
    """
    Create a patch collection from the shapes in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.
    radius : float
        Radius of circle for points in the shapefile.  (Default is 500.)
    idx : iterable int
        A list or array that contains shape numbers to include in the
        patch collection.  Return all shapes if not specified.

    Returns
    -------
        pc : matplotlib.collections.PatchCollection
            Patch collection of shapes in the shapefile

    """
    if shapefile is None:
        s = "Could not import shapefile.  Must install pyshp in order to plot shapefiles."
        raise PlotException(s)
    if plt is None:
        err_msg = (
            "matplotlib must be installed to "
            + "use shapefile_to_patch_collection()"
        )
        raise ImportError(err_msg)
    else:
        from matplotlib.patches import Polygon, Circle, Path, PathPatch
        from matplotlib.collections import PatchCollection
    if isinstance(shp, str):
        sf = shapefile.Reader(shp)
    else:
        sf = shp
    shapes = sf.shapes()
    nshp = len(shapes)
    ptchs = []
    if idx is None:
        idx = range(nshp)
    for n in idx:
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            # points
            for p in shapes[n].points:
                ptchs.append(Circle((p[0], p[1]), radius=radius))
        elif st in [3, 13, 23]:
            # line
            vertices = []
            for p in shapes[n].points:
                vertices.append([p[0], p[1]])
            vertices = np.array(vertices)
            path = Path(vertices)
            ptchs.append(PathPatch(path, fill=False))
        elif st in [5, 25, 31]:
            # polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                ptchs.append(Polygon(pts[par[pij] : par[pij + 1]]))
    pc = PatchCollection(ptchs)
    return pc


def plot_shapefile(
    shp,
    ax=None,
    radius=500.0,
    cmap="Dark2",
    edgecolor="scaled",
    facecolor="scaled",
    a=None,
    masked_values=None,
    idx=None,
    **kwargs
):
    """
    Generic function for plotting a shapefile.

    Parameters
    ----------
    shp : string
        Name of the shapefile to plot.
    ax : matplolib.pyplot.axes object

    radius : float
        Radius of circle for points.  (Default is 500.)
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
    idx : iterable int
        A list or array that contains shape numbers to include in the
        patch collection.  Return all shapes if not specified.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------

    """

    if shapefile is None:
        s = "Could not import shapefile.  Must install pyshp in order to plot shapefiles."
        raise PlotException(s)

    if "vmin" in kwargs:
        vmin = kwargs.pop("vmin")
    else:
        vmin = None

    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    else:
        vmax = None

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc = shapefile_to_patch_collection(shp, radius=radius, idx=idx)
    pc.set(**kwargs)
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1.0 * np.arange(nshp) / nshp)
        if facecolor == "scaled":
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == "scaled":
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == "scaled":
            pc.set_edgecolor("none")
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a)
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    return pc


def cvfd_to_patch_collection(verts, iverts):
    """
    Create a patch collection from control volume vertices and incidence list

    Parameters
    ----------
    verts : ndarray
        2d array of x and y points.
    iverts : list of lists
        should be of len(ncells) with a list of vertex numbers for each cell

    """
    if plt is None:
        err_msg = (
            "matplotlib must be installed to "
            + "use cvfd_to_patch_collection()"
        )
        raise ImportError(err_msg)
    else:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

    ptchs = []
    for ivertlist in iverts:
        points = []
        for iv in ivertlist:
            points.append((verts[iv, 0], verts[iv, 1]))
        # close the polygon, if necessary
        if ivertlist[0] != ivertlist[-1]:
            iv = ivertlist[0]
            points.append((verts[iv, 0], verts[iv, 1]))
        ptchs.append(Polygon(points))
    pc = PatchCollection(ptchs)
    return pc


def plot_cvfd(
    verts,
    iverts,
    ax=None,
    layer=0,
    cmap="Dark2",
    edgecolor="scaled",
    facecolor="scaled",
    a=None,
    masked_values=None,
    **kwargs
):
    """
    Generic function for plotting a control volume finite difference grid of
    information.

    Parameters
    ----------
    verts : ndarray
        2d array of x and y points.
    iverts : list of lists
        should be of len(ncells) with a list of vertex number for each cell
    ax : matplotlib.pylot axis
        matplotlib.pyplot axis instance. Default is None
    layer : int
        layer to extract. Used in combination to the optional ncpl
        parameter. Default is 0
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
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------

    """
    if plt is None:
        err_msg = "matplotlib must be installed to use plot_cvfd()"
        raise ImportError(err_msg)

    if "vmin" in kwargs:
        vmin = kwargs.pop("vmin")
    else:
        vmin = None

    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    else:
        vmax = None

    if "ncpl" in kwargs:
        nlay = layer + 1
        ncpl = kwargs.pop("ncpl")
        if isinstance(ncpl, int):
            i = int(ncpl)
            ncpl = np.ones((nlay), dtype=np.int) * i
        elif isinstance(ncpl, list) or isinstance(ncpl, tuple):
            ncpl = np.array(ncpl)
        i0 = 0
        i1 = 0
        for k in range(nlay):
            i0 = i1
            i1 = i0 + ncpl[k]
        # retain iverts in selected layer
        iverts = iverts[i0:i1]
        # retain vertices in selected layer
        tverts = []
        for iv in iverts:
            for iloc in iv:
                tverts.append((verts[iloc, 0], verts[iloc, 1]))
        verts = np.array(tverts)
        # calculate offset for starting vertex in layer based on
        # global vertex numbers
        iadj = iverts[0][0]
        # reset iverts to relative vertices in selected layer
        tiverts = []
        for iv in iverts:
            i = []
            for t in iv:
                i.append(t - iadj)
            tiverts.append(i)
        iverts = tiverts
    else:
        i0 = 0
        i1 = len(iverts)

    # get current axis
    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)

    pc = cvfd_to_patch_collection(verts, iverts)
    pc.set(**kwargs)

    # set colors
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1.0 * np.arange(nshp) / nshp)
        if facecolor == "scaled":
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == "scaled":
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

        # add NaN values to mask
        a = np.ma.masked_where(np.isnan(a), a)

        if edgecolor == "scaled":
            pc.set_edgecolor("none")
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a[i0:i1])
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    return pc


def findrowcolumn(pt, xedge, yedge):
    """
    Find the MODFLOW cell containing the x- and y- point provided.

    Parameters
    ----------
    pt : list or tuple
        A list or tuple containing a x- and y- coordinate
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.

    Returns
    -------
    irow, jcol : int
        Row and column location containing x- and y- point passed to function.

    Examples
    --------
    >>> import flopy
    >>> irow, jcol = flopy.plotutil.findrowcolumn(pt, xedge, yedge)

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # find column
    jcol = -100
    for jdx, xmf in enumerate(xedge):
        if xmf > pt[0]:
            jcol = jdx - 1
            break

    # find row
    irow = -100
    for jdx, ymf in enumerate(yedge):
        if ymf < pt[1]:
            irow = jdx - 1
            break
    return irow, jcol


def line_intersect_grid(ptsin, xedge, yedge, returnvertices=False):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    ptsin : list
        A list of x, y points defining the vertices of a polyline that will be
        intersected with the rectilinear MODFLOW grid
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    returnvertices: bool
        Return the original polyline vertices in the list of numpy.ndarray
        containing vertices resulting from intersection of the provided
        polygon and the MODFLOW model grid if returnvertices=True.
        (default is False).

    Returns
    -------
    (x, y, dlen) : numpy.ndarray of tuples
        numpy.ndarray of tuples containing the x, y, and segment length of the
        intersection of the provided polyline with the rectilinear MODFLOW
        grid.

    Examples
    --------
    >>> import flopy
    >>> ptsout = flopy.plotutil.line_intersect_grid(ptsin, xedge, yedge)

    """

    small_value = 1.0e-4

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # build list of points along current line
    pts = []
    npts = len(ptsin)
    dlen = 0.0
    for idx in range(1, npts):
        x0 = ptsin[idx - 1][0]
        x1 = ptsin[idx][0]
        y0 = ptsin[idx - 1][1]
        y1 = ptsin[idx][1]
        a = x1 - x0
        b = y1 - y0
        c = math.sqrt(math.pow(a, 2.0) + math.pow(b, 2.0))
        # find cells with (x0, y0) and (x1, y1)
        irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
        irow1, jcol1 = findrowcolumn((x1, y1), xedge, yedge)
        # determine direction to go in the x- and y-directions
        jx = 0
        incx = abs(small_value * a / c)
        iy = 0
        incy = -abs(small_value * b / c)
        if a == 0.0:
            incx = 0.0
        # go to the right
        elif a > 0.0:
            jx = 1
            incx *= -1.0
        if b == 0.0:
            incy = 0.0
        # go down
        elif b < 0.0:
            iy = 1
            incy *= -1.0
        # process data
        if irow0 >= 0 and jcol0 >= 0:
            iadd = True
            if idx > 1 and returnvertices:
                iadd = False
            if iadd:
                pts.append((x0, y0, dlen))
        icnt = 0
        while True:
            icnt += 1
            dx = xedge[jcol0 + jx] - x0
            dlx = 0.0
            if a != 0.0:
                dlx = c * dx / a
            dy = yedge[irow0 + iy] - y0
            dly = 0.0
            if b != 0.0:
                dly = c * dy / b
            if dlx != 0.0 and dly != 0.0:
                if abs(dlx) < abs(dly):
                    dy = dx * b / a
                else:
                    dx = dy * a / b
            xt = x0 + dx + incx
            yt = y0 + dy + incy
            dl = math.sqrt(math.pow((xt - x0), 2.0) + math.pow((yt - y0), 2.0))
            dlen += dl
            if not returnvertices:
                pts.append((xt, yt, dlen))
            x0, y0 = xt, yt
            xt = x0 - 2.0 * incx
            yt = y0 - 2.0 * incy
            dl = math.sqrt(math.pow((xt - x0), 2.0) + math.pow((yt - y0), 2.0))
            dlen += dl
            x0, y0 = xt, yt
            irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
            if irow0 >= 0 and jcol0 >= 0:
                if not returnvertices:
                    pts.append((xt, yt, dlen))
            elif irow1 < 0 or jcol1 < 0:
                dl = math.sqrt(
                    math.pow((x1 - x0), 2.0) + math.pow((y1 - y0), 2.0)
                )
                dlen += dl
                break
            if irow0 == irow1 and jcol0 == jcol1:
                dl = math.sqrt(
                    math.pow((x1 - x0), 2.0) + math.pow((y1 - y0), 2.0)
                )
                dlen += dl
                pts.append((x1, y1, dlen))
                break
    return np.array(pts)


def cell_value_points(pts, xedge, yedge, vdata):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    pts : list
        A list of x, y points and polyline length to extract defining the
        vertices of a polyline that
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. The shape of xedge is
        (NCOL + 1). If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. The shape of yedge is
        (NROW + 1). If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    vdata : numpy.ndarray
        Data (i.e., head, hk, etc.) for a rectilinear MODFLOW model grid. The
        shape of vdata is (NROW, NCOL). If vdata is not a numpy.ndarray it is
        converted to a numpy.ndarray.

    Returns
    -------
    vcell : numpy.ndarray
        numpy.ndarray of of data values from the vdata numpy.ndarray at x- and
        y-coordinate locations in pts.

    Examples
    --------
    >>> import flopy
    >>> vcell = flopy.plotutil.cell_value_points(xpts, xedge, yedge, head[0, :, :])

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)
    if not isinstance(vdata, np.ndarray):
        vdata = np.array(vdata)

    vcell = []
    for (xt, yt, _) in pts:
        # find the modflow cell containing point
        irow, jcol = findrowcolumn((xt, yt), xedge, yedge)
        if irow >= 0 and jcol >= 0:
            if np.isnan(vdata[irow, jcol]):
                vcell.append(np.nan)
            else:
                v = np.asarray(vdata[irow, jcol])
                vcell.append(v)

    return np.array(vcell)


def _set_coord_info(mg, xul, yul, xll, yll, rotation):
    """

    Parameters
    ----------
    mg : fp.discretization.Grid object

    xul : float
        upper left x-coordinate location
    yul : float
        upper left y-coordinate location
    xll : float
        lower left x-coordinate location
    yll : float
        lower left y-coordinate location
    rotation : float
        model grid rotation

    Returns
    -------
    mg : fp.discretization.Grid object
    """
    import warnings

    if xul is not None and yul is not None:
        warnings.warn(
            "xul/yul have been deprecated. Use xll/yll instead.",
            DeprecationWarning,
        )
        if rotation is not None:
            mg._angrot = rotation

        mg.set_coord_info(
            xoff=mg._xul_to_xll(xul), yoff=mg._yul_to_yll(yul), angrot=rotation
        )
    elif xll is not None and xll is not None:
        mg.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)

    elif rotation is not None:
        mg.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)

    return mg


def _depreciated_dis_handler(modelgrid, dis):
    """
    PlotMapView handler for the deprecated dis parameter
    which adds top and botm information to the modelgrid

    Parmaeter
    ---------
    modelgrid : fp.discretization.Grid object

    dis : fp.modflow.ModflowDis object

    Returns
    -------
    modelgrid : fp.discretization.Grid

    """
    # creates a new modelgrid instance with the dis information
    from ..discretization import StructuredGrid, VertexGrid, UnstructuredGrid
    import warnings

    warnings.warn(
        "the dis parameter has been depreciated.", PendingDeprecationWarning
    )
    if modelgrid.grid_type == "vertex":
        modelgrid = VertexGrid(
            modelgrid.vertices,
            modelgrid.cell2d,
            dis.top.array,
            dis.botm.array,
            idomain=modelgrid.idomain,
            xoff=modelgrid.xoffset,
            yoff=modelgrid.yoffset,
            angrot=modelgrid.angrot,
        )
    if modelgrid.grid_type == "unstructured":
        modelgrid = UnstructuredGrid(
            modelgrid._vertices,
            modelgrid._iverts,
            modelgrid._xc,
            modelgrid._yc,
            dis.top.array,
            dis.botm.array,
            idomain=modelgrid.idomain,
            xoff=modelgrid.xoffset,
            yoff=modelgrid.yoffset,
            angrot=modelgrid.angrot,
        )
    else:
        modelgrid = StructuredGrid(
            delc=dis.delc.array,
            delr=dis.delr.array,
            top=dis.top.array,
            botm=dis.botm.array,
            idomain=modelgrid.idomain,
            xoff=modelgrid.xoffset,
            yoff=modelgrid.yoffset,
            angrot=modelgrid.angrot,
        )
    return modelgrid


def advanced_package_bc_helper(pkg, modelgrid, kper):
    """
    Helper function for plotting boundary conditions from "advanced" packages

    Parameters
    ----------
    pkg : flopy Package objects
    modelgrid : flopy.discretization.Grid object

    Returns
    -------
    """
    if pkg.package_type in ("sfr", "uzf"):
        if pkg.parent.version == "mf6":
            mflist = pkg.packagedata.array
            idx = np.array([list(i) for i in mflist["cellid"]], dtype=int).T
        else:
            iuzfbnd = pkg.iuzfbnd.array
            idx = np.where(iuzfbnd != 0)
            idx = np.append([[0] * idx[-1].size], idx, axis=0)
    elif pkg.package_type in ("lak", "maw"):
        if pkg.parent.version == "mf6":
            mflist = pkg.connectiondata.array
            idx = np.array([list(i) for i in mflist["cellid"]], dtype=int).T
        else:
            lakarr = pkg.lakarr.array[kper]
            idx = np.where(lakarr != 0)
            idx = np.array(idx)
    else:
        raise NotImplementedError(
            "Pkg {} not implemented for bc plotting".format(pkg.package_type)
        )
    return idx
