"""
mfusgpcb module.  Contains the MfUsgPcb class. Note that the user can access
the MfUsgPcb class as `flopy.mfusg.MfUsgPcb`.

"""

import numpy as np

from ..pakbase import Package
from ..utils import MfList
from ..utils.recarray_utils import create_empty_recarray
from .mfusg import MfUsg


class MfUsgPcb(Package):
    """
    MODFLOW USG Transport - Prescribed Concentration Boundary (PCB) Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int, optional
        Toggles whether cell-by-cell budget data should be saved. If None or zero,
        budget data will not be saved (default is None).
    stress_period_data : list, recarray, dataframe or dictionary of boundaries.
        Each pcb cell is defined through definition of
        layer(int), row(int), column(int), stage(float), conductance(float)
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of::

            stress_period_data =
            {0: [
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                ],
            1:  [
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                ], ...
            kper:
                [
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                [lay, row, col, iSpec, conc],
                ]
            }

        Note that if no values are specified for a certain stress period, then
        the list of boundaries for the previous stress period for which values
        were defined is used. Full details of all options to specify
        stress_period_data can be found in the flopy3boundaries Notebook in
        the basic subdirectory of the examples directory
    dtype : dtype definition
        if data type is different from default
    options : list of strings
        Package options. (default is None).
    extension : string
        Filename extension (default is 'pcb')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcb is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
        created using the model name and .cbc extension, if ipakcb is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.

    Attributes
    ----------

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
    >>> ml = flopy.mfusg.MfUsg()
    >>> lrcsc = {0:[2, 3, 4, 1, 100.]}  #this pcb will be applied to all
    >>>                                   #stress periods
    >>> pcb = flopy.mfusg.MfUsgPcb(ml, stress_period_data=lrcsc)

    """

    def __init__(
        self,
        model,
        ipakcb=None,
        stress_period_data=None,
        dtype=None,
        no_print=False,
        options=None,
        extension="pcb",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = MfUsgPcb._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # cbc output file
        self.set_cbc_output_file(ipakcb, model, filenames[1])

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        self._generate_heading()

        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append("NOPRINT")
        self.options = options
        self.parent.add_package(self)
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(structured=self.parent.structured)
        self.stress_period_data = MfList(self, stress_period_data)

    def _ncells(self):
        """Maximum number of cells that have general head boundaries
        (developed for MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of pcb cells

        """
        return self.stress_period_data.mxact

    def write_file(self, check=False):
        """
        Write the package file.

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        if check:  # allows turning off package checks when writing files at model level
            self.check(
                f=f"{self.name[0]}.chk",
                verbose=self.parent.verbose,
                level=1,
            )
        f_pcb = open(self.fn_path, "w")
        f_pcb.write(f"{self.heading}\n")
        f_pcb.write(f"{self.stress_period_data.mxact:10d}{self.ipakcb:10d}")
        for option in self.options:
            f_pcb.write(f"  {option}")
        f_pcb.write("\n")
        self.stress_period_data.write_transient(f_pcb)
        f_pcb.close()

    def add_record(self, kper, index, values):
        try:
            self.stress_period_data.add_record(kper, index, values)
        except Exception as e:
            raise Exception(f"MfUsgPcb error adding record to list: {e!s}")

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype = MfUsgPcb.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("iSpec", int),
                    ("conc", np.float32),
                ]
            )
        else:
            dtype = np.dtype([("node", int), ("iSpec", int), ("conc", np.float32)])
        return dtype

    @staticmethod
    def _get_sfac_columns():
        return ["conc"]

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None, check=False):
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        pcb : MfUsgPcb object

        Examples
        --------

        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg(e)
        >>> dis = flopy.modflow.ModflowDis.load('Test1.dis', ml)
        >>> pcb = flopy.mfusg.MfUsgPcb.load('Test1.pcb', ml)

        """

        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading pcb package file...")

        if model.version != "mfusg":
            print(
                "Warning: model version was reset from '{}' to 'mfusg' "
                "in order to load a DDF file".format(model.version)
            )
            model.version = "mfusg"

        return Package.load(
            f,
            model,
            cls,
            nper=nper,
            check=check,
            ext_unit_dict=ext_unit_dict,
        )

    @staticmethod
    def _ftype():
        return "PCB"

    @staticmethod
    def _defaultunit():
        return 154
