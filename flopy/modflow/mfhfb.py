"""
mfhfb module.  Contains the ModflowHfb class. Note that the user can access
the ModflowHfb class as `flopy.modflow.ModflowHfb`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/hfb6.html>`_.

"""
import numpy as np
from numpy.lib.recfunctions import stack_arrays

from ..pakbase import Package
from ..utils.flopy_io import line_parse
from ..utils.recarray_utils import create_empty_recarray
from .mfparbc import ModflowParBc as mfparbc


class ModflowHfb(Package):
    """
    MODFLOW HFB6 - Horizontal Flow Barrier Package

    Parameters
    ----------
    model : model object
        The model object (of type: class:`flopy.modflow.mf.Modflow` or
        `flopy.mfusg.MfUsg`) to which this package will be added.
    nphfb : int
        Number of horizontal-flow barrier parameters. Note that for an HFB
        parameter to have an effect in the simulation, it must be defined
        and made active using NACTHFB to have an effect in the simulation
        (default is 0).
    mxfb : int
        Maximum number of horizontal-flow barrier barriers that will be
        defined using parameters (default is 0).
    nhfbnp: int
        Number of horizontal-flow barriers not defined by parameters. This
        is calculated automatically by FloPy based on the information in
        layer_row_column_data (default is 0).
    hfb_data : list of records

        In its most general form, this is a list of horizontal-flow
        barrier records. A barrier is conceptualized as being located on
        the boundary between two adjacent finite difference cells in the
        same layer. The innermost list is the layer, row1, column1, row2,
        column2, and hydrologic characteristics for a single hfb between
        the cells. The hydraulic characteristic is the barrier hydraulic
        conductivity divided by the width of the horizontal-flow barrier.
        (default is None).
        For a structured model, this gives the form of::
            hfb_data = [
                        [lay, row1, col1, row2, col2, hydchr],
                        [lay, row1, col1, row2, col2, hydchr],
                        [lay, row1, col1, row2, col2, hydchr],
                       ].
        Or for unstructured (mfusg) models::
            hfb_data = [
                        [node1, node2, hydchr],
                        [node1, node2, hydchr],
                        [node1, node2, hydchr],
                       ].
    nacthfb : int
        The number of active horizontal-flow barrier parameters
        (default is 0).
    no_print : boolean
        When True or 1, a list of horizontal flow barriers will not be
        written to the Listing File (default is False)
    options : list of strings
        Package options (default is None).
    extension : string
        Filename extension (default is 'hfb').
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.


    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> hfb_data = [[0, 10, 4, 10, 5, 0.01],[1, 10, 4, 10, 5, 0.01]]
    >>> hfb = flopy.modflow.ModflowHfb(m, hfb_data=hfb_data)

    """

    def __init__(
        self,
        model,
        nphfb=0,
        mxfb=0,
        nhfbnp=0,
        hfb_data=None,
        nacthfb=0,
        no_print=False,
        options=None,
        extension="hfb",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowHfb._defaultunit()

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )

        self._generate_heading()
        self.url = "hfb6.html"

        self.nphfb = nphfb
        self.mxfb = mxfb

        self.nacthfb = nacthfb

        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append("NOPRINT")
        self.options = options

        aux_names = []
        it = 0
        while it < len(options):
            if "aux" in options[it].lower():
                aux_names.append(options[it + 1].lower())
                it += 1
            it += 1

        if hfb_data is None:
            raise Exception("Failed to specify hfb_data.")

        self.nhfbnp = len(hfb_data)
        self.hfb_data = ModflowHfb.get_empty(
            self.nhfbnp, structured=self.parent.structured
        )
        for ibnd, t in enumerate(hfb_data):
            self.hfb_data[ibnd] = tuple(t)

        self.parent.add_package(self)

    def _ncells(self):
        """Maximum number of cell pairs that have horizontal flow barriers
         (developed for MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of hfb cells

        """
        return self.nhfbnp

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        structured = self.parent.structured
        f_hfb = open(self.fn_path, "w")
        f_hfb.write(f"{self.heading}\n")
        f_hfb.write(f"{self.nphfb:10d}{self.mxfb:10d}{self.nhfbnp:10d}")
        for option in self.options:
            f_hfb.write(f"  {option}")
        f_hfb.write("\n")
        for a in self.hfb_data:
            if structured:
                f_hfb.write(
                    "{:10d}{:10d}{:10d}{:10d}{:10d}{:13.6g}\n".format(
                        a[0] + 1, a[1] + 1, a[2] + 1, a[3] + 1, a[4] + 1, a[5]
                    )
                )
            else:
                f_hfb.write(
                    "{:10d}{:10d}{:13.6g}\n".format(a[0] + 1, a[1] + 1, a[2])
                )
        f_hfb.write(f"{self.nacthfb:10d}")
        f_hfb.close()

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        """
        Get an empty recarray that corresponds to hfb dtype and has
        been extended to include aux variables and associated
        aux names.

        """
        dtype = ModflowHfb.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    @staticmethod
    def get_default_dtype(structured=True):
        """
        Get the default dtype for hfb data

        """
        if structured:
            dtype = np.dtype(
                [
                    ("k", int),
                    ("irow1", int),
                    ("icol1", int),
                    ("irow2", int),
                    ("icol2", int),
                    ("hydchr", np.float32),
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("node1", int),
                    ("node2", int),
                    ("hydchr", np.float32),
                ]
            )
        return dtype

    @staticmethod
    def _get_sfac_columns():
        return ["hydchr"]

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type: class:`flopy.modflow.mf.Modflow`)
            to which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        hfb : ModflowHfb object
            ModflowHfb object (of type :class:`flopy.modflow.mfbas.ModflowHfb`)

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> hfb = flopy.modflow.ModflowHfb.load('test.hfb', m)

        """

        if model.verbose:
            print("loading hfb6 package file...")

        structured = model.structured

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        # dataset 1
        t = line_parse(line)
        nphfb = int(t[0])
        mxfb = int(t[1])
        nhfbnp = int(t[2])
        # check for no-print suppressor
        options = []
        aux_names = []
        if len(t) > 2:
            it = 2
            while it < len(t):
                toption = t[it]
                # print it, t[it]
                if toption.lower() == "noprint":
                    options.append(toption)
                elif "aux" in toption.lower():
                    options.append(" ".join(t[it : it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1
        # data set 2 and 3
        if nphfb > 0:
            dt = ModflowHfb.get_empty(1, structured=structured).dtype
            pak_parms = mfparbc.load(
                f,
                nphfb,
                dt,
                model,
                ext_unit_dict=ext_unit_dict,
                verbose=model.verbose,
            )
        # data set 4
        bnd_output = None
        if nhfbnp > 0:
            specified = ModflowHfb.get_empty(nhfbnp, structured=structured)
            for ibnd in range(nhfbnp):
                line = f.readline()
                if "open/close" in line.lower():
                    raise NotImplementedError(
                        "load() method does not support 'open/close'"
                    )
                t = line.strip().split()
                specified[ibnd] = tuple(t[: len(specified.dtype.names)])

            # convert indices to zero-based
            if structured:
                specified["k"] -= 1
                specified["irow1"] -= 1
                specified["icol1"] -= 1
                specified["irow2"] -= 1
                specified["icol2"] -= 1
            else:
                specified["node1"] -= 1
                specified["node2"] -= 1

            bnd_output = np.recarray.copy(specified)

        if nphfb > 0:
            partype = ["hydchr"]
            line = f.readline()
            t = line.strip().split()
            nacthfb = int(t[0])
            for iparm in range(nacthfb):
                line = f.readline()
                t = line.strip().split()
                pname = t[0].lower()
                iname = "static"
                par_dict, current_dict = pak_parms.get(pname)
                data_dict = current_dict[iname]
                par_current = ModflowHfb.get_empty(
                    par_dict["nlst"], structured=structured
                )

                #
                if model.mfpar.pval is None:
                    parval = float(par_dict["parval"])
                else:
                    try:
                        parval = float(model.mfpar.pval.pval_dict[pname])
                    except:
                        parval = float(par_dict["parval"])

                # fill current parameter data (par_current)
                for ibnd, t in enumerate(data_dict):
                    t = tuple(t)
                    par_current[ibnd] = tuple(
                        t[: len(par_current.dtype.names)]
                    )

                # convert indices to zero-based
                if structured:
                    par_current["k"] -= 1
                    par_current["irow1"] -= 1
                    par_current["icol1"] -= 1
                    par_current["irow2"] -= 1
                    par_current["icol2"] -= 1
                else:
                    par_current["node1"] -= 1
                    par_current["node2"] -= 1

                for ptype in partype:
                    par_current[ptype] *= parval

                if bnd_output is None:
                    bnd_output = np.recarray.copy(par_current)
                else:
                    bnd_output = stack_arrays(
                        (bnd_output, par_current),
                        asrecarray=True,
                        usemask=False,
                    )

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowHfb._ftype()
            )

        return cls(
            model,
            nphfb=0,
            mxfb=0,
            nhfbnp=len(bnd_output),
            hfb_data=bnd_output,
            nacthfb=0,
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "HFB6"

    @staticmethod
    def _defaultunit():
        return 29
