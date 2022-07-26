"""
mffhb module.  Contains the ModflowFhb class. Note that the user can access
the ModflowFhb class as `flopy.modflow.ModflowFhb`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/fhb.html>`_.

"""
import numpy as np

from ..pakbase import Package
from ..utils import read1d
from ..utils.recarray_utils import create_empty_recarray


class ModflowFhb(Package):
    """
    MODFLOW Flow and Head Boundary Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.ModflowFhb`) to
        which this package will be added.
    nbdtim : int
        The number of times at which flow and head will be specified for all
        selected cells. (default is 1)
    nflw : int
        Number of cells at which flows will be specified. (default is 0)
    nhed: int
        Number of cells at which heads will be specified. (default is 0)
    ifhbss : int
        FHB steady-state option flag. If the simulation includes any
        transient-state stress periods, the flag is read but not used; in
        this case, specified-flow, specified-head, and auxiliary-variable
        values will be interpolated for steady-state stress periods in the
        same way that values are interpolated for transient stress periods.
        If the simulation includes only steady-state stress periods, the flag
        controls how flow, head, and auxiliary-variable values will be
        computed for each steady-state solution. (default is 0)
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is None).
    nfhbx1 : int
        Number of auxiliary variables whose values will be computed for each
        time step for each specified-flow cell. Auxiliary variables are
        currently not supported. (default is 0)
    nfhbx2 : int
        Number of auxiliary variables whose values will be computed for each
        time step for each specified-head cell. Auxiliary variables are
        currently not supported. (default is 0)
    ifhbpt : int
        Flag for printing values of data list. Applies to datasets 4b, 5b, 6b,
        7b, and 8b. If ifhbpt > 0, datasets read at the beginning of the
        simulation will be printed. Otherwise, the datasets will not be
        printed. (default is 0).
    bdtimecnstm : float
        A constant multiplier for data list bdtime. (default is 1.0)
    bdtime : float or list of floats
        Simulation time at which values of specified flow and (or) values of
        specified head will be read. nbdtim values are required.
        (default is 0.0)
    cnstm5 : float
        A constant multiplier for data list flwrat. (default is 1.0)
    ds5 : list or numpy array or recarray
        Each FHB flwrat cell (dataset 5) is defined through definition of
        layer(int), row(int), column(int), iaux(int), flwrat[nbdtime](float).
        There should be nflw entries. (default is None)
        The simplest form is a list of lists with the FHB flow boundaries.
        This gives the form of::

            ds5 =
            [
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)],
                [lay, row, col, iaux, flwrat1, flwra2, ..., flwrat(nbdtime)]
            ]

    cnstm7 : float
        A constant multiplier for data list sbhedt. (default is 1.0)
    ds7 : list or numpy array or recarray
        Each FHB sbhed cell (dataset 7) is defined through definition of
        layer(int), row(int), column(int), iaux(int), sbhed[nbdtime](float).
        There should be nhed entries. (default is None)
        The simplest form is a list of lists with the FHB flow boundaries.
        This gives the form of::

            ds7 =
            [
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)],
                [lay, row, col, iaux, sbhed1, sbhed2, ..., sbhed(nbdtime)]
            ]

    extension : string
        Filename extension (default is 'fhb')
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
    >>> m = flopy.modflow.Modflow()
    >>> fhb = flopy.modflow.ModflowFhb(m)

    """

    def __init__(
        self,
        model,
        nbdtim=1,
        nflw=0,
        nhed=0,
        ifhbss=0,
        ipakcb=None,
        nfhbx1=0,
        nfhbx2=0,
        ifhbpt=0,
        bdtimecnstm=1.0,
        bdtime=[0.0],
        cnstm5=1.0,
        ds5=None,
        cnstm7=1.0,
        ds7=None,
        extension="fhb",
        unitnumber=None,
        filenames=None,
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowFhb._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            model.add_output_file(
                ipakcb, fname=filenames[1], package=self._ftype()
            )
        else:
            ipakcb = 0

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        self._generate_heading()
        self.url = "fhb.html"

        self.nbdtim = nbdtim
        self.nflw = nflw
        self.nhed = nhed
        self.ifhbss = ifhbss
        self.ipakcb = ipakcb
        if nfhbx1 != 0:
            nfhbx1 = 0
        self.nfhbx1 = nfhbx1
        if nfhbx2 != 0:
            nfhbx2 = 0
        self.nfhbx2 = nfhbx2
        self.ifhbpt = ifhbpt
        self.bdtimecnstm = bdtimecnstm
        if isinstance(bdtime, float):
            bdtime = [bdtime]
        self.bdtime = bdtime
        self.cnstm5 = cnstm5
        self.cnstm7 = cnstm7

        # check the type of dataset 5
        if ds5 is not None:
            dtype = ModflowFhb.get_default_dtype(
                nbdtim=nbdtim, head=False, structured=model.structured
            )
            if isinstance(ds5, (float, int, str)):
                msg = "dataset 5 must be a list of lists or a numpy array"
                raise TypeError(msg)
            elif isinstance(ds5, list):
                ds5 = np.array(ds5)
            # convert numpy array to a recarray
            if ds5.dtype != dtype:
                ds5 = np.core.records.fromarrays(ds5.transpose(), dtype=dtype)

        # assign dataset 5
        self.ds5 = ds5

        # check the type of dataset 7
        if ds7 is not None:
            dtype = ModflowFhb.get_default_dtype(
                nbdtim=nbdtim, head=True, structured=model.structured
            )
            if isinstance(ds7, (float, int, str)):
                msg = "dataset 7 must be a list of lists or a numpy array"
                raise TypeError(msg)
            elif isinstance(ds7, list):
                ds7 = np.array(ds7)
            # convert numpy array to a recarray
            if ds7.dtype != dtype:
                ds7 = np.core.records.fromarrays(ds7.transpose(), dtype=dtype)

        # assign dataset 7
        self.ds7 = ds7

        # perform some simple verification
        if len(self.bdtime) != self.nbdtim:
            raise ValueError(
                "bdtime has {} entries but requires "
                "{} entries.".format(len(self.bdtime), self.nbdtim)
            )

        if self.nflw > 0:
            if self.ds5 is None:
                raise TypeError(
                    f"dataset 5 is not specified but nflw > 0 ({self.nflw})"
                )

            if self.ds5.shape[0] != self.nflw:
                raise ValueError(
                    "dataset 5 has {} rows but requires "
                    "{} rows.".format(self.ds5.shape[0], self.nflw)
                )
            nc = self.nbdtim
            if model.structured:
                nc += 4
            else:
                nc += 2
            if len(self.ds5.dtype.names) != nc:
                raise ValueError(
                    "dataset 5 has {} columns but requires "
                    "{} columns.".format(len(self.ds5.dtype.names), nc)
                )

        if self.nhed > 0:
            if self.ds7 is None:
                raise TypeError(
                    f"dataset 7 is not specified but nhed > 0 ({self.nhed})"
                )
            if self.ds7.shape[0] != self.nhed:
                raise ValueError(
                    "dataset 7 has {} rows but requires "
                    "{} rows.".format(self.ds7.shape[0], self.nhed)
                )
            nc = self.nbdtim
            if model.structured:
                nc += 4
            else:
                nc += 2
            if len(self.ds7.dtype.names) != nc:
                raise ValueError(
                    "dataset 7 has {} columns but requires "
                    "{} columns.".format(len(self.ds7.dtype.names), nc)
                )

        self.parent.add_package(self)

    @staticmethod
    def get_empty(ncells=0, nbdtim=1, structured=True, head=False):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowFhb.get_default_dtype(
            nbdtim=nbdtim, structured=structured, head=head
        )
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    @staticmethod
    def get_default_dtype(nbdtim=1, structured=True, head=False):
        if structured:
            dtype = [("k", int), ("i", int), ("j", int)]
        else:
            dtype = [("node", int)]
        dtype.append(("iaux", int))
        for n in range(nbdtim):
            if head:
                name = f"sbhed{n + 1}"
            else:
                name = f"flwrat{n + 1}"
            dtype.append((name, np.float32))
        return np.dtype(dtype)

    def _ncells(self):
        """Maximum number of cells that have fhb (developed for MT3DMS
        SSM package).

        Returns
        -------
        ncells: int
            maximum number of fhb cells

        """
        return self.nflw + self.nhed

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        f = open(self.fn_path, "w")
        # f.write('{0:s}\n'.format(self.heading))

        # Data set 1
        f.write(f"{self.nbdtim} ")
        f.write(f"{self.nflw} ")
        f.write(f"{self.nhed} ")
        f.write(f"{self.ifhbss} ")
        f.write(f"{self.ipakcb} ")
        f.write(f"{self.nfhbx1} ")
        f.write(f"{self.nfhbx2}\n")

        # Dataset 2 - flow auxiliary names

        # Dataset 3 - head auxiliary names

        # Dataset 4a IFHBUN CNSTM IFHBPT
        f.write(f"{self.unit_number[0]} ")
        f.write(f"{self.bdtimecnstm} ")
        f.write(f"{self.ifhbpt}\n")

        # Dataset 4b
        for n in range(self.nbdtim):
            f.write(f"{self.bdtime[n]} ")
        f.write("\n")

        # Dataset 5 and 6
        if self.nflw > 0:
            # Dataset 5a IFHBUN CNSTM IFHBPT
            f.write(f"{self.unit_number[0]} ")
            f.write(f"{self.cnstm5} ")
            f.write(f"{self.ifhbpt}\n")

            # Dataset 5b
            for n in range(self.nflw):
                for name in self.ds5.dtype.names:
                    v = self.ds5[n][name]
                    if name in ["k", "i", "j", "node"]:
                        v += 1
                    f.write(f"{v} ")
                f.write("\n")

            # Dataset 6a and 6b - flow auxiliary data
            if self.nfhbx1 > 0:
                i = 0

        # Dataset 7
        if self.nhed > 0:
            # Dataset 7a IFHBUN CNSTM IFHBPT
            f.write(f"{self.unit_number[0]} ")
            f.write(f"{self.cnstm7} ")
            f.write(f"{self.ifhbpt}\n")

            # Dataset 7b IFHBUN CNSTM IFHBPT
            for n in range(self.nhed):
                for name in self.ds7.dtype.names:
                    v = self.ds7[n][name]
                    if name in ["k", "i", "j", "node"]:
                        v += 1
                    f.write(f"{v} ")
                f.write("\n")

            # Dataset 8a and 8b - head auxiliary data
            if self.nfhbx2 > 0:
                i = 1

        f.close()

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
        fhb : ModflowFhb object
            ModflowFhb object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> fhb = flopy.modflow.ModflowFhb.load('test.fhb', m)

        """
        if model.verbose:
            print("loading fhb package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # determine package unit number
        iufhb = None
        if ext_unit_dict is not None:
            iufhb, fname = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowFhb._ftype()
            )

        # Dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # dataset 1
        if model.verbose:
            print("loading fhb dataset 1")
        raw = line.strip().split()
        nbdtim = int(raw[0])
        nflw = int(raw[1])
        nhed = int(raw[2])
        ifhbss = int(raw[3])
        ipakcb = int(raw[4])
        nfhbx1 = int(raw[5])
        nfhbx2 = int(raw[6])

        ifhbpt = 0

        # Dataset 2
        flow_aux = []
        if nfhbx1 > 0:
            if model.verbose:
                print("loading fhb dataset 2")
            print("dataset 2 will not be preserved in the created fhb object.")
            for idx in range(nfhbx1):
                line = f.readline()
                raw = line.strip().split()
                varnam = raw[0]
                if len(varnam) > 16:
                    varnam = varnam[0:16]
                weight = float(raw[1])
                flow_aux.append([varnam, weight])

        # Dataset 3
        head_aux = []
        if nfhbx2 > 0:
            if model.verbose:
                print("loading fhb dataset 3")
            print("dataset 3 will not be preserved in the created fhb object.")
            for idx in range(nfhbx2):
                line = f.readline()
                raw = line.strip().split()
                varnam = raw[0]
                if len(varnam) > 16:
                    varnam = varnam[0:16]
                weight = float(raw[1])
                head_aux.append([varnam, weight])

        # Dataset 4a IFHBUN CNSTM IFHBPT
        if model.verbose:
            print("loading fhb dataset 4a")
        line = f.readline()
        raw = line.strip().split()
        ifhbun = int(raw[0])
        if ifhbun != iufhb:
            raise ValueError(
                "fhb dataset 4a must be in the fhb file (unit={}) "
                "fhb data is specified in unit={}".format(iufhb, ifhbun)
            )
        bdtimecnstm = float(raw[1])
        ifhbpt = max(ifhbpt, int(raw[2]))

        # Dataset 4b
        if model.verbose:
            print("loading fhb dataset 4b")

        bdtime = read1d(f, np.zeros((nbdtim,)))

        # Dataset 5 and 6
        cnstm5 = None
        ds5 = None
        ds6 = None
        if nflw > 0:
            if model.verbose:
                print("loading fhb dataset 5a")
            # Dataset 5a IFHBUN CNSTM IFHBPT
            line = f.readline()
            raw = line.strip().split()
            ifhbun = int(raw[0])
            if ifhbun != iufhb:
                raise ValueError(
                    "fhb dataset 5a must be in the fhb file (unit={}) "
                    "fhb data is specified in unit={}".format(iufhb, ifhbun)
                )
            cnstm5 = float(raw[1])
            ifhbpt = max(ifhbpt, int(raw[2]))

            if model.verbose:
                print("loading fhb dataset 5b")

            ds5 = ModflowFhb.get_empty(
                ncells=nflw,
                nbdtim=nbdtim,
                head=False,
                structured=model.structured,
            )
            for n in range(nflw):
                tds5 = read1d(f, np.zeros((nbdtim + 4)))
                ds5[n] = tuple(tds5)

            if model.structured:
                ds5["k"] -= 1
                ds5["i"] -= 1
                ds5["j"] -= 1
            else:
                ds5["node"] -= 1

            # Dataset 6
            if nfhbx1 > 0:
                cnstm6 = []
                ds6 = []
                dtype = []
                for name, weight in flow_aux:
                    dtype.append((name, object))
                for naux in range(nfhbx1):
                    if model.verbose:
                        print(f"loading fhb dataset 6a - aux {naux + 1}")
                    print(
                        "dataset 6a will not be preserved in "
                        "the created fhb object."
                    )
                    # Dataset 6a IFHBUN CNSTM IFHBPT
                    line = f.readline()
                    raw = line.strip().split()
                    ifhbun = int(raw[0])
                    if ifhbun != iufhb:
                        raise ValueError(
                            "fhb dataset 6a must be in the fhb file (unit={}) "
                            "fhb data is specified in "
                            "unit={}".format(iufhb, ifhbun)
                        )
                    cnstm6.append(float(raw[1]))
                    ifhbpt = max(ifhbpt, int(raw[2]))

                    if model.verbose:
                        print(f"loading fhb dataset 6b - aux {naux + 1}")
                    print(
                        "dataset 6b will not be preserved in "
                        "the created fhb object."
                    )
                    current = np.recarray(nflw, dtype=dtype)
                    for n in range(nflw):
                        ds6b = read1d(f, np.zeros((nbdtim,)))
                        current[n] = (tuple(ds6b),)
                    ds6.append(current.copy())

        # Dataset 7
        cnstm7 = None
        ds7 = None
        if nhed > 0:
            if model.verbose:
                print("loading fhb dataset 7a")
            # Dataset 7a IFHBUN CNSTM IFHBPT
            line = f.readline()
            raw = line.strip().split()
            ifhbun = int(raw[0])
            if ifhbun != iufhb:
                raise ValueError(
                    "fhb dataset 7a must be in the fhb file (unit={}) "
                    "fhb data is specified in unit={}".format(iufhb, ifhbun)
                )
            cnstm7 = float(raw[1])
            ifhbpt = max(ifhbpt, int(raw[2]))

            if model.verbose:
                print("loading fhb dataset 7b")

            ds7 = ModflowFhb.get_empty(
                ncells=nhed,
                nbdtim=nbdtim,
                head=True,
                structured=model.structured,
            )
            for n in range(nhed):
                tds7 = read1d(f, np.empty((nbdtim + 4)))
                ds7[n] = tuple(tds7)

            if model.structured:
                ds7["k"] -= 1
                ds7["i"] -= 1
                ds7["j"] -= 1
            else:
                ds7["node"] -= 1

            # Dataset 8
            if nfhbx2 > 0:
                cnstm8 = []
                ds8 = []
                dtype = []
                for name, weight in head_aux:
                    dtype.append((name, object))
                for naux in range(nfhbx1):
                    if model.verbose:
                        print(f"loading fhb dataset 8a - aux {naux + 1}")
                    print(
                        "dataset 8a will not be preserved in "
                        "the created fhb object."
                    )
                    # Dataset 6a IFHBUN CNSTM IFHBPT
                    line = f.readline()
                    raw = line.strip().split()
                    ifhbun = int(raw[0])
                    if ifhbun != iufhb:
                        raise ValueError(
                            "fhb dataset 8a must be in the fhb file (unit={}) "
                            "fhb data is specified in "
                            "unit={}".format(iufhb, ifhbun)
                        )
                    cnstm8.append(float(raw[1]))
                    ifhbpt6 = int(raw[2])
                    ifhbpt = max(ifhbpt, ifhbpt6)

                    if model.verbose:
                        print(f"loading fhb dataset 8b - aux {naux + 1}")
                    print(
                        "dataset 8b will not be preserved in "
                        "the created fhb object."
                    )
                    current = np.recarray(nflw, dtype=dtype)
                    for n in range(nhed):
                        ds8b = read1d(f, np.zeros((nbdtim,)))
                        current[n] = (tuple(ds8b),)
                    ds8.append(current.copy())

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowFhb._ftype()
            )
        if ipakcb > 0:
            iu, filenames[1] = model.get_ext_dict_attr(
                ext_unit_dict, unit=ipakcb
            )
            model.add_pop_key_list(ipakcb)

        # auxiliary data are not passed to load instantiation
        nfhbx1 = 0
        nfhbx2 = 0

        fhb = cls(
            model,
            nbdtim=nbdtim,
            nflw=nflw,
            nhed=nhed,
            ifhbss=ifhbss,
            ipakcb=ipakcb,
            nfhbx1=nfhbx1,
            nfhbx2=nfhbx2,
            ifhbpt=ifhbpt,
            bdtimecnstm=bdtimecnstm,
            bdtime=bdtime,
            cnstm5=cnstm5,
            ds5=ds5,
            cnstm7=cnstm7,
            ds7=ds7,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        # return fhb object
        return fhb

    @staticmethod
    def _ftype():
        return "FHB"

    @staticmethod
    def _defaultunit():
        return 40
