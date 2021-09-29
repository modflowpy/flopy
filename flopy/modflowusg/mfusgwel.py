"""
mfusgwel module.  Contains the ModflowUsgWel class. Note that the user can access
the ModflowUsgWel class as `flopy.modflowusg.ModflowUsgWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?wel.htm>`_.

"""
import numpy as np
from copy import deepcopy
from ..utils import MfList
from ..pakbase import Package
from ..modflow.mfwel import ModflowWel
from ..utils.recarray_utils import create_empty_recarray
from ..utils.flopy_io import ulstrd
import warnings

from ..modflow.mfparbc import ModflowParBc as mfparbc


class ModflowUsgWel(ModflowWel):
    """
    MODFLOW Well Package Class.

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
    >>> m = flopy.modflowusg.ModflowUsg()
    >>> lrcq = {0:[[2, 3, 4, -100.]], 1:[[2, 3, 4, -100.]]}
    >>> wel = flopy.modflowusg.ModflowUsgWel(m, stress_period_data=lrcq)

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
    ):
        """
        Package constructor.

        """

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
        )

        self.autoflowreduce = False
        self.iunitafr = 0

        for idx, opt in enumerate(self.options):
            if "autoflowreduce" in opt.lower():
                self.autoflowreduce = True
            if "iunitafr" in opt.lower():
                t = opt.strip().split()
                self.iunitafr = int(t[1])

        if self.iunitafr > 0:
            fname = self.filenames[1]
            model.add_output_file(
                self.iunitafr,
                fname=fname,
                extension="afr",
                binflag=False,
                package=ModflowUsgWel._ftype(),
            )

        # initialize CLN MfList
        # CLN WELs are always read as CLNNODE Q, so dtype must be of unstructured form
        dtype_model = self.dtype
        dtype_cln = ModflowUsgWel.get_default_dtype(structured=False)
        self.dtype = dtype_cln
        self.cln_stress_period_data = MfList(
            self, cln_stress_period_data, binary=binary
        )
        # reset self.dtype for cases where the model is structured but CLN WELs are used
        self.dtype = ModflowUsgWel.get_default_dtype(
            structured=model.structured
        )

        self.parent.add_package(self)

    def write_file(self, f=None):
        """
        Write the package file.

        Parameters:
            f: (str) optional file name

        Returns
        -------
        None

        """
        if f is not None:
            if isinstance(f, str):
                f_wel = open(f, "w")
            else:
                f_wel = f
        else:
            f_wel = open(self.fn_path, "w")

        f_wel.write(f"{self.heading}\n")

        mxact = (
            self.stress_period_data.mxact + self.cln_stress_period_data.mxact
        )

        line = f" {mxact:9d} {self.ipakcb:9d} "
        for opt in self.options:
            line += " " + str(opt)
        line += "\n"
        f_wel.write(line)

        nrow, ncol, nlay, nper = self.parent.get_nrow_ncol_nlay_nper()

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
        loop_over_kpers = list(range(0, maxper))

        if first < 0:
            first = maxper
        if cln_first < 0:
            cln_first = maxper

        fmt_string = self.stress_period_data.fmt_string
        cln_fmt_string = self.cln_stress_period_data.fmt_string

        for kper in loop_over_kpers:
            # Fill missing early kpers with 0
            if kper < first:
                itmp = 0
            elif kper in kpers:
                kper_data = deepcopy(self.stress_period_data.data[kper])
                kper_vtype = self.stress_period_data.vtype[kper]
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

            # Fill missing early kpers with 0
            if kper < cln_first:
                itmpcln = 0
            elif kper in cln_kpers:
                cln_kper_data = deepcopy(
                    self.cln_stress_period_data.data[kper]
                )
                cln_kper_vtype = self.cln_stress_period_data.vtype[kper]
                if cln_kper_vtype == np.recarray:
                    itmpcln = cln_kper_data.shape[0]
                    lnames = [
                        name.lower() for name in cln_kper_data.dtype.names
                    ]
                    for idx in ["k", "i", "j", "node"]:
                        if idx in lnames:
                            cln_kper_data[idx] += 1
                elif (cln_kper_vtype == int) or (cln_kper_vtype is None):
                    itmpcln = cln_kper_data
            # Fill late missing kpers with -1
            else:
                itmpcln = -1

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

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None):
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
        wel : ModflowUsgWel object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflowusg.ModflowUsgWel.load('test.wel', m)

        """

        if model.verbose:
            print("loading wel package file...")

        # open the file if not already open
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        elif hasattr(f, "name"):
            filename = f.name
        else:
            filename = "?"

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # dataset 1a -- check for parameters
        npwel = 0
        if "parameter" in line.lower():
            t = line.strip().split()
            npwel = int(t[1])
            mxl = 0
            if npwel > 0:
                mxl = int(t[2])
                if model.verbose:
                    print(
                        f"   Parameters detected. Number of parameters = {npwel}"
                    )
            line = f.readline()

        # dataset 2 -- MXACTW IWELCB [Option]
        t = line.strip().split()
        imax = 2
        ipakcb = 0
        try:
            ipakcb = int(t[1])
        except:
            if model.verbose:
                print(f"   implicit ipakcb in {filename}")

        options = []
        aux_names = []
        iunitafr = 0
        if len(t) > imax:
            it = imax
            while it < len(t):
                toption = t[it]
                if toption.lower() == "noprint":
                    options.append(toption.lower())
                elif toption.lower() == "autoflowreduce":
                    options.append(toption.lower())
                elif toption.lower() == "iunitafr":
                    options.append(" ".join(t[it : it + 2]))
                    iunitafr = int(t[it + 1])
                    it += 1
                elif "aux" in toption.lower():
                    options.append(" ".join(t[it : it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1

        # get the list columns that should be scaled with sfac
        sfac_columns = ModflowUsgWel._get_sfac_columns()

        # dataset 3 -- read parameter data
        if npwel > 0:
            dt = ModflowUsgWel.get_empty(
                1, aux_names=aux_names, structured=model.structured
            ).dtype
            # dataset 4 --
            pak_parms = mfparbc.load(
                f, npwel, dt, model, ext_unit_dict, model.verbose
            )

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # dataset 5 -- read data for every stress period
        bnd_output = None
        stress_period_data = {}
        current = None

        cln_bnd_output = None
        cln_stress_period_data = {}
        cln_current = None

        for iper in range(nper):
            if model.verbose:
                msg = f"   loading well data for kper {iper + 1:5d}"
                print(msg)
            line = f.readline()
            if line == "":
                break
            t = line.strip().split()
            itmp = int(t[0])
            itmpp = 0
            try:
                itmpp = int(t[1])
            except:
                if model.verbose:
                    print(f"   implicit itmpp in {filename}")

            try:
                itmpcln = int(t[2])
            except:
                itmpcln = 0

            # dataset 6a -- read well data
            if itmp == 0:
                bnd_output = None
                current = ModflowUsgWel.get_empty(
                    itmp, aux_names=aux_names, structured=model.structured
                )
            elif itmp > 0:
                current = ModflowUsgWel.get_empty(
                    itmp, aux_names=aux_names, structured=model.structured
                )
                current = ulstrd(
                    f, itmp, current, model, sfac_columns, ext_unit_dict
                )
                if model.structured:
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

            # dataset 6c -- read CLN well data
            if itmpcln == 0:
                cln_bnd_output = None
                cln_current = ModflowUsgWel.get_empty(
                    itmp, aux_names=aux_names, structured=False
                )
            elif itmpcln > 0:
                cln_current = ModflowUsgWel.get_empty(
                    itmpcln, aux_names=aux_names, structured=False
                )
                cln_current = ulstrd(
                    f, itmpcln, cln_current, model, sfac_columns, ext_unit_dict
                )
                cln_current["node"] -= 1
                cln_bnd_output = np.recarray.copy(cln_current)
            else:
                if cln_current is None:
                    cln_bnd_output = None
                else:
                    cln_bnd_output = np.recarray.copy(cln_current)

            # dataset 7 -- parameter data
            for iparm in range(itmpp):
                line = f.readline()
                t = line.strip().split()
                pname = t[0].lower()
                iname = "static"
                try:
                    tn = t[1]
                    c = tn.lower()
                    instance_dict = pak_parms.bc_parms[pname][1]
                    if c in instance_dict:
                        iname = c
                    else:
                        iname = "static"
                except:
                    if model.verbose:
                        print(
                            f"  implicit static instance for parameter {pname}"
                        )

                par_dict, current_dict = pak_parms.get(pname)
                data_dict = current_dict[iname]

                par_current = ModflowUsgWel.get_empty(
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
                    except:
                        parval = float(par_dict["parval"])

                # fill current parameter data (par_current)
                for ibnd, t in enumerate(data_dict):
                    t = tuple(t)
                    par_current[ibnd] = tuple(
                        t[: len(par_current.dtype.names)]
                    )

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

            if bnd_output is None:
                stress_period_data[iper] = itmp
            else:
                stress_period_data[iper] = bnd_output

            if cln_bnd_output is None:
                cln_stress_period_data[iper] = itmpcln
            else:
                cln_stress_period_data[iper] = cln_bnd_output

        if openfile:
            f.close()

        # set package unit number
        filenames = [None, None, None]
        unitnumber = ModflowUsgWel._defaultunit()
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowUsgWel._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)
            if iunitafr > 0:
                iu, filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iunitafr
                )
                model.add_pop_key_list(iunitafr)

        wel = cls(
            model,
            ipakcb=ipakcb,
            stress_period_data=stress_period_data,
            cln_stress_period_data=cln_stress_period_data,
            dtype=ModflowUsgWel.get_default_dtype(structured=model.structured),
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        return wel
