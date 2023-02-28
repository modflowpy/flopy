"""
mfhyd module.  Contains the ModflowHydclass. Note that the user can access
the ModflowHyd class as `flopy.modflow.ModflowHyd`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/hyd.html>`_.

"""
import numpy as np

from ..pakbase import Package
from ..utils.recarray_utils import create_empty_recarray


class ModflowHyd(Package):
    """
    MODFLOW HYDMOD (HYD) Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    nhyd : int
        the maximum number of observation points. (default is 1).
    ihydun : int
        A flag that is used to determine if hydmod data should be saved.
        If ihydun is non-zero hydmod data will be saved. (default is 1).
    hydnoh : float
        is a user-specified value that is output if a value cannot be computed
        at a hydrograph location. For example, the cell in which the hydrograph
        is located may be a no-flow cell. (default is -999.)
    obsdata : list of lists, numpy array, or numpy recarray (nhyd, 7)
        Each row of obsdata includes data defining pckg (3 character string),
        arr (2 character string), intyp (1 character string) klay (int),
        xl (float), yl (float), hydlbl (14 character string) for each
        observation.

        pckg : str
            is a 3-character flag to indicate which package is to be addressed
            by hydmod for the hydrograph of each observation point.
        arr : str
            is a text code indicating which model data value is to be accessed
            for the hydrograph of each observation point.
        intyp : str
            is a 1-character value to indicate how the data from the specified
            feature are to be accessed; The two options are 'I' for
            interpolated value or 'C' for cell value (intyp must be 'C' for
            STR and SFR Package hydrographs.
        klay : int
            is the layer sequence number (zero-based) of the array to be
            addressed by HYDMOD.
        xl : float
            is the coordinate of the hydrograph point in model units of length
            measured parallel to model rows, with the origin at the lower left
            corner of the model grid.
        yl : float
            is the coordinate of the hydrograph point in model units of length
            measured parallel to model columns, with the origin at the lower
            left corner of the model grid.
        hydlbl : str
            is used to form a label for the hydrograph.


        The simplest form is a list of lists. For example, if nhyd=3 this
        gives the form of::

            obsdata =
            [
                [pckg, arr, intyp, klay, xl, yl, hydlbl],
                [pckg, arr, intyp, klay, xl, yl, hydlbl],
                [pckg, arr, intyp, klay, xl, yl, hydlbl]
            ]

    extension : list string
        Filename extension (default is ['hyd', 'hyd.bin'])
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the hydmod output name will be created using
        the model name and .hyd.bin extension (for example,
        modflowtest.hyd.bin). If a single string is passed the package will be
        set to the string and hydmod output name will be created using the
        model name and .hyd.bin extension. To define the names for all package
        files (input and output) the length of the list of strings should be 2.
        Default is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> hyd = flopy.modflow.ModflowHyd(m)

    """

    def __init__(
        self,
        model,
        nhyd=1,
        ihydun=None,
        hydnoh=-999.0,
        obsdata=[["BAS", "HD", "I", 0, 0.0, 0.0, "HOBS1"]],
        extension=["hyd", "hyd.bin"],
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowHyd._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # set ihydun to a default unit number if it isn't specified
        if ihydun is None:
            ihydun = 536

        # update external file information with hydmod output
        model.add_output_file(
            ihydun,
            fname=filenames[1],
            extension="hyd.bin",
            package=self._ftype(),
        )

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        self._generate_heading()
        self.url = "hyd.html"

        self.nhyd = nhyd
        self.ihydun = ihydun
        self.hydnoh = hydnoh

        dtype = ModflowHyd.get_default_dtype()
        obs = ModflowHyd.get_empty(nhyd)
        if isinstance(obsdata, list):
            if len(obsdata) != nhyd:
                raise RuntimeError(
                    f"ModflowHyd: nhyd ({nhyd}) does not equal "
                    f"length of obsdata ({len(obsdata)})."
                )
            for idx in range(nhyd):
                obs["pckg"][idx] = obsdata[idx][0]
                obs["arr"][idx] = obsdata[idx][1]
                obs["intyp"][idx] = obsdata[idx][2]
                obs["klay"][idx] = int(obsdata[idx][3])
                obs["xl"][idx] = float(obsdata[idx][4])
                obs["yl"][idx] = float(obsdata[idx][5])
                obs["hydlbl"][idx] = obsdata[idx][6]
            obsdata = obs
        elif isinstance(obsdata, np.ndarray):
            if obsdata.dtype == object:
                if obsdata.shape[1] != len(dtype):
                    raise IndexError("Incorrect number of fields for obsdata")
                obsdata = obsdata.transpose()
                obs["pckg"] = obsdata[0]
                obs["arr"] = obsdata[1]
                obs["intyp"] = obsdata[2]
                obs["klay"] = obsdata[3]
                obs["xl"] = obsdata[4]
                obs["yl"] = obsdata[5]
                obs["hydlbl"] = obsdata[6]
            else:
                inds = ["pckg", "arr", "intyp", "klay", "xl", "yl", "hydlbl"]
                for idx in inds:
                    obs["pckg"] = obsdata["pckg"]
                    obs["arr"] = obsdata["arr"]
                    obs["intyp"] = obsdata["intyp"]
                    obs["klay"] = obsdata["klay"]
                    obs["xl"] = obsdata["xl"]
                    obs["yl"] = obsdata["yl"]
                    obs["hydlbl"] = obsdata["hydlbl"]
            obsdata = obs
            obsdata = obsdata.view(dtype=dtype)
        self.obsdata = obsdata

        # add package
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing

        f = open(self.fn_path, "w")

        # write dataset 1
        f.write(f"{self.nhyd} {self.ihydun} {self.hydnoh} {self.heading}\n")

        # write dataset 2
        for idx in range(self.nhyd):
            f.write(f"{self.obsdata['pckg'][idx].decode()} ")
            f.write(f"{self.obsdata['arr'][idx].decode()} ")
            f.write(f"{self.obsdata['intyp'][idx].decode()} ")
            f.write(f"{self.obsdata['klay'][idx] + 1} ")
            f.write(f"{self.obsdata['xl'][idx]} ")
            f.write(f"{self.obsdata['yl'][idx]} ")
            f.write(f"{self.obsdata['hydlbl'][idx].decode()} ")
            f.write("\n")

        # close hydmod file
        f.close()

    @staticmethod
    def get_empty(ncells=0):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowHyd.get_default_dtype()
        return create_empty_recarray(ncells, dtype)

    @staticmethod
    def get_default_dtype():
        # PCKG ARR INTYP KLAY XL YL HYDLBL
        dtype = np.dtype(
            [
                ("pckg", "|S3"),
                ("arr", "|S2"),
                ("intyp", "|S1"),
                ("klay", int),
                ("xl", np.float32),
                ("yl", np.float32),
                ("hydlbl", "|S14"),
            ]
        )
        return dtype

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        hyd : ModflowHyd object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> hyd = flopy.modflow.ModflowHyd.load('test.hyd', m)

        """

        if model.verbose:
            print("loading hydmod package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # --read dataset 1
        # NHYD IHYDUN HYDNOH
        if model.verbose:
            print("  loading hydmod dataset 1")
        line = f.readline()
        t = line.strip().split()
        nhyd = int(t[0])
        ihydun = int(t[1])
        model.add_pop_key_list(ihydun)
        hydnoh = float(t[2])

        obs = ModflowHyd.get_empty(nhyd)

        for idx in range(nhyd):
            line = f.readline()
            t = line.strip().split()
            obs["pckg"][idx] = t[0].strip()
            obs["arr"][idx] = t[1].strip()
            obs["intyp"][idx] = t[2].strip()
            obs["klay"][idx] = int(t[3]) - 1
            obs["xl"][idx] = float(t[4])
            obs["yl"][idx] = float(t[5])
            obs["hydlbl"][idx] = t[6].strip()

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowHyd._ftype()
            )
            if ihydun > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ihydun
                )
                model.add_pop_key_list(ihydun)

        # return hyd instance
        return cls(
            model,
            nhyd=nhyd,
            ihydun=ihydun,
            hydnoh=hydnoh,
            obsdata=obs,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "HYD"

    @staticmethod
    def _defaultunit():
        return 36
