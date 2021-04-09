try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except (ImportError, ModuleNotFoundError):
    plt = None

import os
import platform


class styles:
    """Styles class for custom matplotlib styling

    The class contains both custom styles and plotting methods
    for custom formatting using a specific matplotlib style

    Additional styles can be easily added to the mplstyle folder and
    accessed using the plt.style.context() method.

    """

    _ws = os.path.abspath(os.path.dirname(__file__))
    _map_style = os.path.join(_ws, "mplstyle", "usgsmap.mplstyle")
    _plot_style = os.path.join(_ws, "mplstyle", "usgsplot.mplstyle")
    if platform.system() == "linux":
        _map_style = os.path.join(_ws, "mplstyle", "usgsmap_linux.mplstyle")
        _plot_style = os.path.join(_ws, "mplstyle", "usgsplot_linux.mplstyle")

    @classmethod
    def USGSMap(cls):
        return plt.style.context(styles._map_style)

    @classmethod
    def USGSPlot(cls):
        return plt.style.context(styles._map_style)

    @classmethod
    def set_font_type(cls, family, fontname):
        """
        Method to set the matplotlib font type for the current style

        Note: this method only works when adding text using the styles
        methods.

        Parameters
        ----------
        family : str
            matplotlib.rcparams font.family
        font : str
            matplotlib.rcparams font.fontname

        Returns
        -------
            None
        """
        mpl.rcParams["font.family"] = family
        mpl.rcParams["font." + family] = fontname
        return mpl.rcParams

    @classmethod
    def heading(
        self,
        ax=None,
        letter=None,
        heading=None,
        x=0.00,
        y=1.01,
        idx=None,
        fontsize=9,
    ):
        """Add a USGS-style heading to a matplotlib axis object

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        letter : str
            string that defines the subplot (A, B, C, etc.)
        heading : str
            text string
        x : float
            location of the heading in the x-direction in normalized plot
            dimensions ranging from 0 to 1 (default is 0.00)
        y : float
            location of the heading in the y-direction in normalized plot
            dimensions ranging from 0 to 1 (default is 1.01)
        idx : int
            index for programatically generating the heading letter when letter
            is None and idx is not None. idx = 0 will generate A
            (default is None)

        Returns
        -------
        text : object
            matplotlib text object

        """
        if ax is None:
            ax = plt.gca()

        if letter is None and idx is not None:
            letter = chr(ord("A") + idx)

        font = styles.__set_fontspec(
            bold=True, italic=False, fontsize=fontsize
        )

        if letter is not None:
            if heading is None:
                text = letter.replace(".", "")
            else:
                letter = letter.rstrip()
                if not letter.endswith("."):
                    letter += "."
                text = letter + " " + heading
        else:
            text = heading

        if text is None:
            return

        text = ax.text(
            x,
            y,
            text,
            va="bottom",
            ha="left",
            fontdict=font,
            transform=ax.transAxes,
        )
        return text

    @classmethod
    def xlabel(cls, ax=None, label="", bold=False, italic=False, **kwargs):
        """Method to set the xlabel using the styled fontdict

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        label : str
            axis label for the chart
        bold : bool
            flag to switch to boldface test
        italic : bool
            flag to use italic text
        kwargs : dict
            keyword arguments for the matplotlib set_xlabel method

        Returns
        -------
            None
        """
        if ax is None:
            ax = plt.gca()
        fontsize = kwargs.pop("fontsize", 9)
        fontspec = styles.__set_fontspec(
            bold=bold, italic=italic, fontsize=fontsize
        )
        ax.set_xlabel(label, fontdict=fontspec, **kwargs)

    @classmethod
    def ylabel(cls, ax=None, label="", bold=False, italic=False, **kwargs):
        """Method to set the ylabel using the styled fontdict

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        label : str
            axis label for the chart
        bold : bool
            flag to switch to boldface test
        italic : bool
            flag to use italic text
        kwargs : dict
            keyword arguments for the matplotlib set_xlabel method

        Returns
        -------
            None
        """
        if ax is None:
            ax = plt.gca()

        fontsize = kwargs.pop("fontsize", 9)
        fontspec = styles.__set_fontspec(
            bold=bold, italic=italic, fontsize=fontsize
        )
        ax.set_ylabel(label, fontdict=fontspec, **kwargs)

    @classmethod
    def graph_legend(cls, ax=None, handles=None, labels=None, **kwargs):
        """Add a USGS-style legend to a matplotlib axis object

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        handles : list
            list of legend handles
        labels : list
            list of labels for legend handles
        kwargs : kwargs
            matplotlib legend kwargs

        Returns
        -------
        leg : object
            matplotlib legend object

        """
        if ax is None:
            ax = plt.gca()

        fontspec = styles.__set_fontspec(bold=True, italic=False, family=True)

        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, prop=fontspec, **kwargs)

        # add title to legend
        if "title" in kwargs:
            title = kwargs.pop("title")
        else:
            title = None
        leg = styles.graph_legend_title(leg, title=title)
        return leg

    @classmethod
    def graph_legend_title(cls, leg, title=None):
        """Set the legend title for a matplotlib legend object

        Parameters
        ----------
        leg : legend object
            matplotlib legend object
        title : str
            title for legend

        Returns
        -------
        leg : object
            matplotlib legend object

        """
        if title is None:
            title = "EXPLANATION"
        elif title.lower() == "none":
            title = None

        fontspec = styles.__set_fontspec(bold=True, italic=False, family=True)

        leg.set_title(title, prop=fontspec)
        return leg

    @classmethod
    def add_text(
        cls,
        ax=None,
        text="",
        x=0.0,
        y=0.0,
        transform=True,
        bold=True,
        italic=True,
        fontsize=9,
        ha="left",
        va="bottom",
        **kwargs
    ):
        """Add USGS-style text to a axis object

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        text : str
            text string
        x : float
            x-location of text string (default is 0.)
        y : float
            y-location of text string (default is 0.)
        transform : bool
            boolean that determines if a transformed (True) or data (False)
            coordinate system is used to define the (x, y) location of the
            text string (default is True)
        bold : bool
            boolean indicating if bold font (default is True)
        italic : bool
            boolean indicating if italic font (default is True)
        fontsize : int
            font size (default is 9 points)
        ha : str
            matplotlib horizontal alignment keyword (default is left)
        va : str
            matplotlib vertical alignment keyword (default is bottom)
        kwargs : dict
            dictionary with valid matplotlib text object keywords

        Returns
        -------
        text_obj : object
            matplotlib text object

        """
        if ax is None:
            ax = plt.gca()

        if transform:
            transform = ax.transAxes
        else:
            transform = ax.transData

        font = styles.__set_fontspec(
            bold=bold, italic=italic, fontsize=fontsize
        )

        text_obj = ax.text(
            x,
            y,
            text,
            va=va,
            ha=ha,
            fontdict=font,
            transform=transform,
            **kwargs
        )
        return text_obj

    @classmethod
    def add_annotation(
        cls,
        ax=None,
        text="",
        xy=None,
        xytext=None,
        bold=True,
        italic=True,
        fontsize=9,
        ha="left",
        va="bottom",
        **kwargs
    ):
        """Add an annotation to a axis object

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)
        text : str
            text string
        xy : tuple
            tuple with the location of the annotation (default is None)
        xytext : tuple
            tuple with the location of the text
        bold : bool
            boolean indicating if bold font (default is True)
        italic : bool
            boolean indicating if italic font (default is True)
        fontsize : int
            font size (default is 9 points)
        ha : str
            matplotlib horizontal alignment keyword (default is left)
        va : str
            matplotlib vertical alignment keyword (default is bottom)
        kwargs : dict
            dictionary with valid matplotlib annotation object keywords

        Returns
        -------
        ann_obj : object
            matplotlib annotation object

        """
        if ax is None:
            ax = plt.gca()

        if xy is None:
            xy = (0.0, 0.0)

        if xytext is None:
            xytext = (0.0, 0.0)

        fontspec = styles.__set_fontspec(
            bold=bold, italic=italic, fontsize=fontsize
        )
        # add font information to kwargs
        if kwargs is None:
            kwargs = fontspec
        else:
            for key, value in fontspec.items():
                kwargs[key] = value

        # create annotation
        ann_obj = ax.annotate(text, xy, xytext, va=va, ha=ha, **kwargs)

        return ann_obj

    @classmethod
    def remove_edge_ticks(cls, ax=None):
        """Remove unnecessary ticks on the edges of the plot

        Parameters
        ----------
        ax : axis object
            matplotlib axis object (default is None)

        Returns
        -------
        ax : axis object
            matplotlib axis object

        """
        if ax is None:
            ax = plt.gca()

        ax.tick_params(axis="both", which="both", length=0)

    @classmethod
    def __set_fontspec(cls, bold=True, italic=True, fontsize=9, family=False):
        """Create fontspec dictionary for matplotlib pyplot objects

        Parameters
        ----------
        bold : bool
            boolean indicating if font is bold (default is True)
        italic : bool
            boolean indicating if font is italic (default is True)
        fontsize : int
            font size (default is 9 point)


        Returns
        -------
            dict
        """
        family = mpl.rcParams["font.family"][0]
        font = mpl.rcParams["font." + family][0]

        if bold:
            weight = "bold"
        else:
            weight = "normal"

        if italic:
            style = "italic"
        else:
            style = "normal"

        # define fontspec dictionary
        fontspec = {
            "fontname": font,
            "size": fontsize,
            "weight": weight,
            "style": style,
        }

        if family:
            fontspec.pop("fontname")
            fontspec["family"] = family

        return fontspec


if plt is None:
    styles = None
