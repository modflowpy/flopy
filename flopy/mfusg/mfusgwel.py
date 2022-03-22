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
    """MODFLOW Well Package Class.

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
        dtype_cln = MfUsgWel.get_default_dtype(structured=False)
        self.dtype = dtype_cln
        self.cln_stress_period_data = MfList(
            self, cln_stress_period_data, binary=binary
        )
        # reset self.dtype for cases where the model is structured but CLN WELs are used
        self.dtype = MfUsgWel.get_default_dtype(structured=model.structured)

        if add_package:
            self.parent.add_package(self)

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
        for opt in self.options:
            line += " " + str(opt)
        line += "\n"
        f_wel.write(line)

        _, _, _, nper = self.parent.get_nrow_ncol_nlay_nper()

        kpers = list(self.stress_period_data.data.keys())
        if len(kpers) > 0:
            kpers.sort()
            first = kpers[0]
            last = max(kpers) + 1
        else:
            first = -1
            last = -1

        cln_kpers = list(self.cln_stress_period_data.data.keys())
        if len(cln_kpers) > 0:
            cln_kpers.sort()
            cln_first = cln_kpers[0]
            cln_last = max(cln_kpers) + 1
        else:
            cln_first = -1
            cln_last = -1

        maxper = max(nper, last, cln_last)

        if first < 0:
            first = maxper
        if cln_first < 0:
            cln_first = maxper

        fmt_string = self.stress_period_data.fmt_string
        cln_fmt_string = self.cln_stress_period_data.fmt_string

        for kper in range(maxper):

            # gw cell wells
            itmp, kper_data = self._get_kper_data(
                kper, first, self.stress_period_data
            )

            # cln wells
            itmpcln, cln_kper_data = self._get_kper_data(
                kper, cln_first, self.cln_stress_period_data
            )

            f_wel.write(
                f" {itmp:9d} {0:9d} {itmpcln:9d} # stress period {kper + 1}\n"
            )

            if itmp > 0:
                np.savetxt(f_wel, kper_data, fmt=fmt_string, delimiter="")
            if itmpcln > 0:
                np.savetxt(
                    f_wel, cln_kper_data, fmt=cln_fmt_string, delimiter=""
                )

        f_wel.close()

    @staticmethod
    def _get_kper_data(kper, first, stress_period_data):
        """
        Gets boundary condition stress period data for a given stress period.

        Parameters:
        ----------
            kper: int
                stress period (base 0)
            first : int
                First stress period for which stress period data is defined
            stress_period_data : Numpy recarray or int or None
                Flopy boundary condition stress_period_data object
                (with a "data" attribute that is keyed on kper)
        Returns
        -------
        itmp : int
            Number of boundary conditions for stress period kper
        kper_data : Numpy recarray
            Boundary condition data for stress period kper
        """
        kpers = list(stress_period_data.data.keys())
        # Fill missing early kpers with 0
        kper_data = None
        if kper < first:
            itmp = 0
        elif kper in kpers:
            kper_data = deepcopy(stress_period_data.data[kper])
            kper_vtype = stress_period_data.vtype[kper]
            if kper_vtype == np.recarray:
                itmp = kper_data.shape[0]
                lnames = [name.lower() for name in kper_data.dtype.names]
                for idx in ["k", "i", "j", "node"]:
                    if idx in lnames:
                        kper_data[idx] += 1
            elif (kper_vtype == int) or (kper_vtype is None):
                itmp = kper_data
        # Fill late missing kpers with -1
        else:
            itmp = -1

        return itmp, kper_data

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None, check=True):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        wel : MfUsgWel object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.mfusg.MfUsgWel.load('test.wel', m)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading wel package file...")
        f_obj = get_open_file_object(f, "r")

        # dataset 0 -- header
        while True:
            line = f_obj.readline()
            if line[0] != "#":
                break

        # dataset 1a -- check for parameters
        npwel = 0
        if "parameter" in line.lower():
            line_text = line.strip().split()
            npwel = int(line_text[1])
            if npwel > 0:
                if model.verbose:
                    print(
                        f"   Parameters detected. Number of parameters = {npwel}"
                    )
            line = f_obj.readline()

        # dataset 2 -- MXACTW IWELCB [Option]
        ipakcb, options, aux_names, iunitafr = cls._load_dataset2(line)

        # dataset 3 and 4 -- read parameter data
        pak_parms = None
        if npwel > 0:
            par_dt = MfUsgWel.get_empty(
                1, aux_names=aux_names, structured=model.structured
            ).dtype
            # dataset 4 --
            pak_parms = mfparbc.load(
                f_obj, npwel, par_dt, model, ext_unit_dict, model.verbose
            )

        if nper is None:
            _, _, _, nper = model.get_nrow_ncol_nlay_nper()

        # dataset 5 -- read data for every stress period
        stress_period_data, cln_stress_period_data = cls._load_item5(
            f_obj, model, nper, aux_names, ext_unit_dict, pak_parms
        )

        f_obj.close()

        # set package unit number
        filenames = [None, None, None]
        unitnumber = MfUsgWel._defaultunit()
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )
            if ipakcb > 0:
                _, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)
            if iunitafr > 0:
                _, filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iunitafr
                )
                model.add_pop_key_list(iunitafr)

        wel = cls(
            model,
            ipakcb=ipakcb,
            stress_period_data=stress_period_data,
            cln_stress_period_data=cln_stress_period_data,
            dtype=cls.get_default_dtype(structured=model.structured),
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        if check:
            wel.check(
                f=f"{wel.name[0]}.chk",
                verbose=model.verbose,
                level=0,
            )

        return wel

    @staticmethod
    def _load_dataset2(line):
        """Load mfusgwel dataset 2 from line."""
        # dataset 2 -- MXACTW IWELCB [Option]
        line_text = line.strip().split()
        n_items = 2
        ipakcb = 0
        if len(line_text) > 1:
            ipakcb = int(line_text[1])

        options = []
        aux_names = []
        iunitafr = 0
        if len(line_text) > n_items:
            item_n = n_items
            while item_n < len(line_text):
                toption = line_text[item_n]
                if toption.lower() == "noprint":
                    options.append(toption.lower())
                elif toption.lower() == "autoflowreduce":
                    options.append(toption.lower())
                elif toption.lower() == "iunitafr":
                    options.append(" ".join(line_text[item_n : item_n + 2]))
                    iunitafr = int(line_text[item_n + 1])
                    item_n += 1
                elif "aux" in toption.lower():
                    options.append(" ".join(line_text[item_n : item_n + 2]))
                    aux_names.append(line_text[item_n + 1].lower())
                    item_n += 1
                item_n += 1

        return ipakcb, options, aux_names, iunitafr

    @staticmethod
    def _load_itmp_itmpp_itmpcln(line):
        """Reads itmp, itmpp and itmpcln from line. Returns None for blank line."""
        # strip comments from line
        line = line[: line.find("#")]

        if line == "":
            return None, None, None
        line_text = line.strip().split()
        # strip non-numerics from line items
        line_text = [item for item in line_text if item.isdigit()]

        itmp = int(line_text[0])
        itmpp = 0
        if len(line_text) > 1:
            itmpp = int(line_text[1])
        itmpcln = 0
        if len(line_text) > 2:
            itmpcln = int(line_text[2])

        return itmp, itmpp, itmpcln

    @classmethod
    def _load_item5(
        cls, f_obj, model, nper, aux_names, ext_unit_dict=None, pak_parms=None
    ):
        """Loads Wel item 5."""
        bnd_output = None
        stress_period_data = {}

        cln_bnd_output = None
        cln_stress_period_data = {}

        for iper in range(nper):
            if model.verbose:
                msg = f"   loading well data for kper {iper + 1:5d}"
                print(msg)
            line = f_obj.readline()
            itmp, itmpp, itmpcln = cls._load_itmp_itmpp_itmpcln(line)
            if itmp is None:
                break

            # dataset 6a -- read well data
            bnd_output = cls._load_dataset6(
                f_obj, itmp, model, aux_names, ext_unit_dict, model.structured
            )

            # dataset 6c -- read CLN well data
            cln_bnd_output = cls._load_dataset6(
                f_obj, itmpcln, model, aux_names, ext_unit_dict, False
            )

            # dataset 7 -- parameter data
            if itmpp > 0:
                bnd_output = cls._load_dataset7(
                    f_obj, itmpp, model, aux_names, pak_parms, bnd_output
                )

            if bnd_output is None:
                stress_period_data[iper] = itmp
            else:
                stress_period_data[iper] = bnd_output

            if cln_bnd_output is None:
                cln_stress_period_data[iper] = itmpcln
            else:
                cln_stress_period_data[iper] = cln_bnd_output

        return stress_period_data, cln_stress_period_data

    @staticmethod
    def _load_dataset6(
        f_obj, itmp, model, aux_names, ext_unit_dict, structured=False
    ):
        """
        Reads dataset 6(a, b or c) from open file

        Parameters
        ----------
        f_obj : open file handle
        itmp : int
            Number of items to read from dataset6a
        model : Flopy model object
        aux_names : list of auxillary variable names
        ext_unit_dict : dictionary, optional
            External unit dictionary.
        structured : bool, default if False.
            model.structured, or if loading CLNs from dataset6a
            even for a structured model, False.

        Returns
        -------
        bnd_output : Numpy recarray
            Wel Dataset 6a for itmp values
        """

        # get the list columns that should be scaled with sfac
        sfac_columns = MfUsgWel._get_sfac_columns()

        if itmp == 0:
            bnd_output = None
            current = MfUsgWel.get_empty(
                itmp, aux_names=aux_names, structured=structured
            )
        elif itmp > 0:
            current = MfUsgWel.get_empty(
                itmp, aux_names=aux_names, structured=structured
            )
            current = ulstrd(
                f_obj, itmp, current, model, sfac_columns, ext_unit_dict
            )
            if structured:
                current["k"] -= 1
                current["i"] -= 1
                current["j"] -= 1
            else:
                current["node"] -= 1
            bnd_output = np.recarray.copy(current)
        else:
            if current is None:
                bnd_output = None
            else:
                bnd_output = np.recarray.copy(current)

        return bnd_output

    @staticmethod
    def _load_dataset7(
        f_obj, itmpp, model, aux_names, pak_parms, bnd_output=None
    ):
        """
        Reads named parameters from wel file

        Parameters
        ----------
        f_obj : open file handle
        itmpp : int, must be > 0
            Number of named parameters to read from dataset7
        model : Flopy model object
        aux_names : list of auxillary variable names
        pak_parms : named package parameters
            read using mfparbc.load()
        bnd_output : Numpy recarray of non-Named Parameter stress period data
            Named parameters will be stacked on top of this.

        Returns
        -------
        bnd_output : Numpy recarray
            Wel Dataset 7 for itmpp values
        """

        if itmpp <= 0:
            return bnd_output

        iparm = 0
        while iparm < itmpp:
            line = f_obj.readline()
            line_text = line.strip().split()
            pname = line_text[0].lower()
            iname = "static"
            try:
                kper_iname = line_text[1].lower()
                instance_dict = pak_parms.bc_parms[pname][1]
                if kper_iname in instance_dict:
                    iname = kper_iname
                else:
                    iname = "static"
            except ValueError:
                if model.verbose:
                    print(f"  implicit static instance for parameter {pname}")

            par_dict, current_dict = pak_parms.get(pname)
            data_dict = current_dict[iname]

            par_current = MfUsgWel.get_empty(
                par_dict["nlst"],
                aux_names=aux_names,
                structured=model.structured,
            )

            #  get appropriate parval
            if model.mfpar.pval is None:
                parval = float(par_dict["parval"])
            else:
                try:
                    parval = float(model.mfpar.pval.pval_dict[pname])
                except ValueError:
                    parval = float(par_dict["parval"])

            # fill current parameter data (par_current)
            for ibnd, vals in enumerate(data_dict):
                vals = tuple(vals)
                par_current[ibnd] = tuple(vals[: len(par_current.dtype.names)])

            if model.structured:
                par_current["k"] -= 1
                par_current["i"] -= 1
                par_current["j"] -= 1
            else:
                par_current["node"] -= 1

            par_current["flux"] *= parval

            if bnd_output is None:
                bnd_output = np.recarray.copy(par_current)
            else:
                bnd_output = stack_arrays(
                    (bnd_output, par_current),
                    asrecarray=True,
                    usemask=False,
                )

            iparm += 1

        return bnd_output
