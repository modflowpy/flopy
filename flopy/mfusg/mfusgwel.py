"""
Mfusgwel module.

Contains the MfUsgWel class. Note that the user can access
the MfUsgWel class as `flopy.mfusg.MfUsgWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/wel.html>`_.
"""
from copy import deepcopy

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from ..modflow.mfparbc import ModflowParBc as mfparbc
from ..modflow.mfwel import ModflowWel
from ..utils import MfList
from ..utils.flopy_io import ulstrd
from ..utils.utils_def import get_open_file_object
from .mfusg import MfUsg


class MfUsgWel(ModflowWel):
    """MODFLOW-USG Well Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    stress_period_data : list of boundaries, or recarray of boundaries, or
        dictionary of boundaries
        For structured grid, each well is defined through definition of
        layer (int), row (int), column (int), flux (float).
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of:

            stress_period_data =
            {0: [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ],
            1:  [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ], ...
            kper:
                [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ]
            }

        For unstructured grid,
            stress_period_data =
            {0: [
                [node, flux],
                [node, flux],
                [node, flux]
                ],
            1:  [
                [node, flux],
                [node, flux],
                [node, flux]
                ], ...
            kper:
                [
                [node, flux],
                [node, flux],
                [node, flux]
                ]
            }

        Note that if the number of lists is smaller than the number of stress
        periods, then the last list of wells will apply until the end of the
        simulation. Full details of all options to specify stress_period_data
        can be found in the flopy3 boundaries Notebook in the basic
        subdirectory of the examples directory
    cln_stress_period_data : list of boundaries, or recarray of boundaries, or
        dictionary of boundaries
        Stress period data of wells simulated as Connected Linear Network (CLN)
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of:

            cln_stress_period_data =
            {0: [
                [iclnnode, flux],
                [iclnnode, flux],
                [iclnnode, flux]
                ],
            1:  [
                [iclnnode, flux],
                [iclnnode, flux],
                [iclnnode, flux]
                ], ...
            kper:
                [
                [iclnnode, flux],
                [iclnnode, flux],
                [iclnnode, flux]
                ]
            }
    dtype : custom datatype of stress_period_data.
        If None the default well datatype will be applied (default is None).
    cln_dtype : custom datatype of cln_stress_period_data.
        If None the default well datatype will be applied (default is None).
    extension : string
        Filename extension (default is 'wel')
    options : list of strings
        Package options (default is None).
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.
    add_package : bool
        Flag to add the initialised package object to the parent model object.
        Default is True.

    Attributes
    ----------
    mxactw : int
        Maximum number of wells for a stress period.  This is calculated
        automatically by FloPy based on the information in
        stress_period_data.

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mfusg.MfUsg()
    >>> lrcq = {0:[[2, 3, 4, -100.]], 1:[[2, 3, 4, -100.]]}
    >>> wel = flopy.mfusg.MfUsgWel(m, stress_period_data=lrcq)

    """

    def __init__(
        self,
        model,
        ipakcb=None,
        stress_period_data=None,
        cln_stress_period_data=None,
        dtype=None,
        cln_dtype=None,
        extension="wel",
        options=None,
        binary=False,
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        """Package constructor."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # set filenames
        filenames = self._prepare_filenames(filenames)

        super().__init__(
            model,
            ipakcb=ipakcb,
            stress_period_data=stress_period_data,
            dtype=dtype,
            extension=extension,
            options=options,
            binary=binary,
            unitnumber=unitnumber,
            filenames=filenames,
            add_package=False,
        )

        self.autoflowreduce = False
        self.iunitafr = 0

        for opt in self.options:
            if "autoflowreduce" in opt.lower():
                self.autoflowreduce = True
            if "iunitafr" in opt.lower():
                line_text = opt.strip().split()
                self.iunitafr = int(line_text[1])

        if self.iunitafr > 0:
            model.add_output_file(
                self.iunitafr,
                fname=filenames[1],
                extension="afr",
                binflag=False,
                package=self._ftype(),
            )

        # initialize CLN MfList
        # CLN WELs are always read as CLNNODE Q, so dtype must be of unstructured form
        if cln_dtype is None:
            cln_dtype = MfUsgWel.get_default_dtype(structured=False)
        self.dtype = cln_dtype

        # determine if any aux variables in cln_dtype
        options = self._check_for_aux(options, cln=True)
        self.cln_stress_period_data = MfList(
            self, cln_stress_period_data, binary=binary
        )

        # reset self.dtype for cases where the model is structured but CLN WELs are used
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured
            )

        # determine if any aux variables in dtype
        options = self._check_for_aux(options)

        self.options = options

        # initialize MfList
        self.stress_period_data = MfList(
            self, stress_period_data, binary=binary
        )

        if add_package:
            self.parent.add_package(self)

    def _check_for_aux(self, options, cln=False):
        """Check dtype for auxiliary variables, and add to options.

        Parameters:
        ----------
            options: (list) package options

        Returns
        -------
            options: list
                Package options strings

        """
        if cln:
            dt = self.get_default_dtype(structured=False)
        else:
            dt = self.get_default_dtype(structured=self.parent.structured)
        if len(self.dtype.names) > len(dt.names):
            for name in self.dtype.names[len(dt.names) :]:
                ladd = True
                for option in options:
                    if name.lower() in option.lower():
                        ladd = False
                        break
                if ladd:
                    options.append(f"aux {name} ")

        return options

    def write_file(self, f=None):
        """Write the package file.

        Parameters:
        ----------
            f: (str) optional file name

        Returns
        -------
        None
        """
        if f is None:
            f_wel = open(self.fn_path, "w")
        elif isinstance(f, str):
            f_wel = open(f, "w")
        else:
            f_wel = f

        f_wel.write(f"{self.heading}\n")

        mxact = (
            self.stress_period_data.mxact + self.cln_stress_period_data.mxact
        )

        line = f" {mxact:9d} {self.ipakcb:9d} "
        if self.options is None:
            self.options = []
        for opt in self.options:
            line += " " + str(opt)
        line += "\n"
        f_wel.write(line)

        # gwf wels (and possibly cln wells)
        self.stress_period_data.write_transient(
            f_wel, cln_data=self.cln_stress_period_data
        )

        f_wel.close()
