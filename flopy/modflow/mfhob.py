import sys
import collections
import numpy as np
from ..pakbase import Package
from ..utils.recarray_utils import create_empty_recarray


class ModflowHob(Package):
    """
    Head Observation package class

    Parameters
    ----------
    iuhobsv : int
        unit number where output is saved. If iuhobsv is None, a unit number
        will be assigned (default is None).
    hobdry : float
        Value of the simulated equivalent written to the observation output
        file when the observation is omitted because a cell is dry
        (default is 0).
    tomulth : float
        Time step multiplier for head observations. The product of tomulth and
        toffset must produce a time value in units consistent with other model
        input. tomulth can be dimensionless or can be used to convert the units
        of toffset to the time unit used in the simulation (default is 1).
    obs_data : HeadObservation or list of HeadObservation instances
        A single HeadObservation instance or a list of HeadObservation
        instances containing all of the data for each observation. If obs_data
        is None a default HeadObservation with an observation in layer, row,
        column (0, 0, 0) and a head value of 0 at totim 0 will be created
        (default is None).
    hobname : str
        Name of head observation output file. If iuhobsv is greater than 0,
        and hobname is None, the model basename with a '.hob.out' extension
        will be used (default is None).
    extension : string
        Filename extension (default is hob)
    no_print : boolean
        When True or 1, a list of head observations will not be
        written to the Listing File (default is False)
    options : list of strings
        Package options (default is None).
    unitnumber : int
        File unit number (default is None)
    filenames : str or list of str
        Filenames to use for the package and the output files. If filenames
        is None the package name will be created using the model name and
        package extension and the hob output name will be created using the
        model name and .hob.out extension (for example, modflowtest.hob.out),
        if iuhobsv is a number greater than zero. If a single string is passed
        the package will be set to the string and hob output name will be
        created using the model name and .hob.out extension, if iuhobsv is a
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

    Examples
    --------

    >>> import flopy
    >>> model = flopy.modflow.Modflow()
    >>> dis = flopy.modflow.ModflowDis(model, nlay=1, nrow=11, ncol=11, nper=2,
    ...                                perlen=[1,1])
    >>> tsd = [[1.,54.4], [2., 55.2]]
    >>> obsdata = flopy.modflow.HeadObservation(model, layer=0, row=5,
    ...                                         column=5, time_series_data=tsd)
    >>> hob = flopy.modflow.ModflowHob(model, iuhobsv=51, hobdry=-9999.,
    ...                                obs_data=obsdata)


    """

    def __init__(
        self,
        model,
        iuhobsv=None,
        hobdry=0,
        tomulth=1.0,
        obs_data=None,
        hobname=None,
        extension="hob",
        no_print=False,
        options=None,
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor
        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowHob._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # set filenames[1] to hobname if filenames[1] is not None
        if filenames[1] is None:
            if hobname is not None:
                filenames[1] = hobname

        if iuhobsv is not None:
            fname = filenames[1]
            model.add_output_file(
                iuhobsv,
                fname=fname,
                extension="hob.out",
                binflag=False,
                package=ModflowHob._ftype(),
            )
        else:
            iuhobsv = 0

        # Fill namefile items
        name = [ModflowHob._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent,
        # extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.url = "hob.htm"
        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )

        self.iuhobsv = iuhobsv
        self.hobdry = hobdry
        self.tomulth = tomulth

        # create default
        if obs_data is None:
            obs_data = HeadObservation(model)

        # make sure obs_data is a list
        if isinstance(obs_data, HeadObservation):
            obs_data = [obs_data]

        # set self.obs_data
        self.obs_data = obs_data

        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append("NOPRINT")
        self.options = options

        # add checks for input compliance (obsnam length, etc.)
        self.parent.add_package(self)

    def _set_dimensions(self):
        """
        Set the length of the obs_data list

        Returns
        -------
        None

        """
        # make sure each entry of obs_data list is a HeadObservation instance
        # and calculate nh, mobs, and maxm
        msg = ""
        self.nh = 0
        self.mobs = 0
        self.maxm = 0
        for idx, obs in enumerate(self.obs_data):
            if not isinstance(obs, HeadObservation):
                msg += (
                    "ModflowHob: obs_data entry {} ".format(idx)
                    + "is not a HeadObservation instance.\n"
                )
                continue
            self.nh += obs.nobs
            if obs.multilayer:
                self.mobs += obs.nobs
            self.maxm = max(self.maxm, obs.maxm)
        if msg != "":
            raise ValueError(msg)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # determine the dimensions of HOB data
        self._set_dimensions()

        # open file for writing
        f = open(self.fn_path, "w")

        # write dataset 0
        f.write("{}\n".format(self.heading))

        # write dataset 1
        f.write("{:10d}".format(self.nh))
        f.write("{:10d}".format(self.mobs))
        f.write("{:10d}".format(self.maxm))
        f.write("{:10d}".format(self.iuhobsv))
        f.write("{:10.4g}".format(self.hobdry))
        if self.no_print or "NOPRINT" in self.options:
            f.write("{: >10}".format("NOPRINT"))
        f.write("\n")

        # write dataset 2
        f.write("{:10.4g}\n".format(self.tomulth))

        # write datasets 3-6
        for idx, obs in enumerate(self.obs_data):
            # dataset 3
            obsname = obs.obsname
            if isinstance(obsname, bytes):
                obsname = obsname.decode("utf-8")
            line = "{:12s}   ".format(obsname)
            layer = obs.layer
            if layer >= 0:
                layer += 1
            line += "{:10d} ".format(layer)
            line += "{:10d} ".format(obs.row + 1)
            line += "{:10d} ".format(obs.column + 1)
            irefsp = obs.irefsp
            if irefsp >= 0:
                irefsp += 1
            line += "{:10d} ".format(irefsp)
            if obs.nobs == 1:
                toffset = obs.time_series_data[0]["toffset"]
                hobs = obs.time_series_data[0]["hobs"]
            else:
                toffset = 0.0
                hobs = 0.0
            line += "{:20} ".format(toffset)
            line += "{:10.4f} ".format(obs.roff)
            line += "{:10.4f} ".format(obs.coff)
            line += "{:10.4f} ".format(hobs)
            line += "  # DATASET 3 - Observation {}".format(idx + 1)
            f.write("{}\n".format(line))

            # dataset 4
            if len(obs.mlay.keys()) > 1:
                line = ""
                for key, value in iter(obs.mlay.items()):
                    line += "{:5d}{:10.4f}".format(key + 1, value)
                line += "  # DATASET 4 - Observation {}".format(idx + 1)
                f.write("{}\n".format(line))

            # dataset 5
            if irefsp < 0:
                line = "{:10d}".format(obs.itt)
                line += 103 * " "
                line += "  # DATASET 5 - Observation {}".format(idx + 1)
                f.write("{}\n".format(line))

            # dataset 6:
            if obs.nobs > 1:
                for jdx, t in enumerate(obs.time_series_data):
                    obsname = t["obsname"]
                    if isinstance(obsname, bytes):
                        obsname = obsname.decode("utf-8")
                    line = "{:12s}   ".format(obsname)
                    line += "{:10d} ".format(t["irefsp"] + 1)
                    line += "{:20} ".format(t["toffset"])
                    line += "{:10.4f} ".format(t["hobs"])
                    line += 55 * " "
                    line += "  # DATASET 6 - " + "Observation {}.{}".format(
                        idx + 1, jdx + 1
                    )
                    f.write("{}\n".format(line))

        # close the hob package file
        f.close()

        return

    @classmethod
    def load(cls, f, model, ext_unit_dict=None, check=True):
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        hob : ModflowHob package object
            ModflowHob package object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> hobs = flopy.modflow.ModflowHob.load('test.hob', m)

        """

        if model.verbose:
            sys.stdout.write("loading hob package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # read dataset 1
        t = line.strip().split()
        nh = int(t[0])
        iuhobsv = None
        hobdry = 0
        if len(t) > 3:
            iuhobsv = int(t[3])
            hobdry = float(t[4])

        # read dataset 2
        line = f.readline()
        t = line.strip().split()
        tomulth = float(t[0])

        # read observation data
        obs_data = []

        # read datasets 3-6
        nobs = 0
        while True:
            # read dataset 3
            line = f.readline()
            t = line.strip().split()
            obsnam = t[0]
            layer = int(t[1])
            row = int(t[2]) - 1
            col = int(t[3]) - 1
            irefsp0 = int(t[4])
            toffset = float(t[5])
            roff = float(t[6])
            coff = float(t[7])
            hob = float(t[8])

            # read dataset 4 if multilayer obs
            if layer > 0:
                layer -= 1
                mlay = {layer: 1.0}
            else:
                line = f.readline()
                t = line.strip().split()
                mlay = collections.OrderedDict()
                if len(t) >= abs(layer) * 2:
                    for j in range(0, abs(layer) * 2, 2):
                        k = int(t[j]) - 1
                        # catch case where the same layer is specified
                        # more than once. In this case add previous
                        # value to the current value
                        keys = list(mlay.keys())
                        v = 0.0
                        if k in keys:
                            v = mlay[k]
                        mlay[k] = float(t[j + 1]) + v
                else:
                    for j in range(abs(layer)):
                        k = int(t[0]) - 1
                        keys = list(mlay.keys())
                        v = 0.0
                        if k in keys:
                            v = mlay[k]
                        mlay[k] = float(t[1]) + v

                        if j != abs(layer) - 1:
                            line = f.readline()
                            t = line.strip().split()
                # reset layer
                layer = -len(list(mlay.keys()))

            # read datasets 5 & 6. Index loop variable
            if irefsp0 > 0:
                itt = 1
                irefsp0 -= 1
                totim = model.dis.get_totim_from_kper_toffset(
                    irefsp0, toffset * tomulth
                )
                names = [obsnam]
                tsd = [totim, hob]
                nobs += 1
            else:
                names = []
                tsd = []
                # read data set 5
                line = f.readline()
                t = line.strip().split()
                itt = int(t[0])
                # dataset 6
                for j in range(abs(irefsp0)):
                    line = f.readline()
                    t = line.strip().split()
                    names.append(t[0])
                    irefsp = int(t[1]) - 1
                    toffset = float(t[2])
                    totim = model.dis.get_totim_from_kper_toffset(
                        irefsp, toffset * tomulth
                    )
                    hob = float(t[3])
                    tsd.append([totim, hob])
                    nobs += 1

            obs_data.append(
                HeadObservation(
                    model,
                    tomulth=tomulth,
                    layer=layer,
                    row=row,
                    column=col,
                    roff=roff,
                    coff=coff,
                    obsname=obsnam,
                    mlay=mlay,
                    itt=itt,
                    time_series_data=tsd,
                    names=names,
                )
            )
            if nobs == nh:
                break

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowHob._ftype()
            )
            if iuhobsv is not None:
                if iuhobsv > 0:
                    iu, filenames[1] = model.get_ext_dict_attr(
                        ext_unit_dict, unit=iuhobsv
                    )
                    model.add_pop_key_list(iuhobsv)

        return cls(
            model,
            iuhobsv=iuhobsv,
            hobdry=hobdry,
            tomulth=tomulth,
            obs_data=obs_data,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "HOB"

    @staticmethod
    def _defaultunit():
        return 39


class HeadObservation(object):
    """
    Create single HeadObservation instance from a time series array. A list of
    HeadObservation instances are passed to the ModflowHob package.

    Parameters
    ----------
    tomulth : float
        Time-offset multiplier for head observations. Default is 1.
    obsname : string
        Observation name. Default is 'HOBS'
    layer : int
        The zero-based layer index of the cell in which the head observation
        is located. If layer is less than zero, hydraulic heads from multiple
        layers are combined to calculate a simulated value. The number of
        layers equals the absolute value of layer, or abs(layer). Default is 0.
    row : int
        The zero-based row index for the observation. Default is 0.
    column : int
        The zero-based column index of the observation. Default is 0.
    irefsp : int
        The zero-based stress period to which the observation time is
        referenced.
    roff : float
        Fractional offset from center of cell in Y direction (between rows).
        Default is 0.
    coff : float
        Fractional offset from center of cell in X direction (between columns).
        Default is 0.
    itt : int
        Flag that identifies whether head or head changes are used as
        observations. itt = 1 specified for heads and itt = 2 specified
        if initial value is head and subsequent changes in head. Only
        specified if irefsp is < 0. Default is 1.
    mlay : dictionary of length (abs(irefsp))
        Key represents zero-based layer numbers for multilayer observations and
        value represents the fractional value for each layer of multilayer
        observations. If mlay is None, a default mlay of {0: 1.} will be
        used (default is None).
    time_series_data : list or numpy array
        Two-dimensional list or numpy array containing the simulation time of
        the observation and the observed head [[totim, hob]]. If
        time_series_dataDefault is None, a default observation of 0. at
        totim 0. will be created (default is None).
    names : list
        List of specified observation names. If names is None, observation
        names will be automatically generated from obsname and the order
        of the timeseries data (default is None).

    Returns
    -------
    obs : HeadObservation
        HeadObservation object.

    Examples
    --------

    >>> import flopy
    >>> model = flopy.modflow.Modflow()
    >>> dis = flopy.modflow.ModflowDis(model, nlay=1, nrow=11, ncol=11, nper=2,
    ...                                perlen=[1,1])
    >>> tsd = [[1.,54.4], [2., 55.2]]
    >>> obsdata = flopy.modflow.HeadObservation(model, layer=0, row=5,
    ...                                         column=5, time_series_data=tsd)

    """

    def __init__(
        self,
        model,
        tomulth=1.0,
        obsname="HOBS",
        layer=0,
        row=0,
        column=0,
        irefsp=None,
        roff=0.0,
        coff=0.0,
        itt=1,
        mlay=None,
        time_series_data=None,
        names=None,
    ):
        """
        Object constructor
        """

        if mlay is None:
            mlay = {0: 1.0}
        if time_series_data is None:
            time_series_data = [[0.0, 0.0]]
        if irefsp is None:
            if len(time_series_data) == 1:
                irefsp = 1
            else:
                irefsp = -1 * len(time_series_data)

        # set class attributes
        self.obsname = obsname
        self.layer = layer
        self.row = row
        self.column = column
        self.irefsp = irefsp
        self.roff = roff
        self.coff = coff
        self.itt = itt
        self.mlay = mlay
        self.maxm = 0

        # check if multilayer observation
        self.multilayer = False
        if len(self.mlay.keys()) > 1:
            self.maxm = len(self.mlay.keys())
            self.multilayer = True
            tot = 0.0
            for key, value in self.mlay.items():
                tot += value
            if not (np.isclose(tot, 1.0, rtol=0)):
                msg = (
                    "sum of dataset 4 proportions must equal 1.0 - "
                    + "sum of dataset 4 proportions = {tot} for "
                    + "observation name {obsname}."
                ).format(tot=tot, obsname=self.obsname)
                raise ValueError(msg)

        # convert passed time_series_data to a numpy array
        if isinstance(time_series_data, list):
            time_series_data = np.array(time_series_data, dtype=float)

        # if a single observation is passed as a list reshape to a
        # two-dimensional numpy array
        if len(time_series_data.shape) == 1:
            time_series_data = np.reshape(time_series_data, (1, 2))

        # find indices of time series data that are valid
        tmax = model.dis.get_final_totim()
        keep_idx = time_series_data[:, 0] <= tmax
        time_series_data = time_series_data[keep_idx, :]

        # set the number of observations in this time series
        shape = time_series_data.shape
        self.nobs = shape[0]

        # construct names if not passed
        if names is None:
            if self.nobs == 1:
                names = [obsname]
            else:
                names = []
                for idx in range(self.nobs):
                    names.append("{}.{}".format(obsname, idx + 1))
        # make sure the length of names is greater than or equal to nobs
        else:
            if isinstance(names, str):
                names = [names]
            elif not isinstance(names, list):
                msg = (
                    "HeadObservation names must be a "
                    + "string or a list of strings"
                )
                raise ValueError(msg)
            if len(names) < self.nobs:
                msg = (
                    "a name must be specified for every valid "
                    + "observation - {} ".format(len(names))
                    + "names were passed but at least "
                    + "{} names are required.".format(self.nobs)
                )
                raise ValueError(msg)

        # create time_series_data
        self.time_series_data = self._get_empty(ncells=shape[0])
        for idx in range(self.nobs):
            t = time_series_data[idx, 0]
            kstp, kper, toffset = model.dis.get_kstp_kper_toffset(t)
            self.time_series_data[idx]["totim"] = t
            self.time_series_data[idx]["irefsp"] = kper
            self.time_series_data[idx]["toffset"] = toffset / tomulth
            self.time_series_data[idx]["hobs"] = time_series_data[idx, 1]
            self.time_series_data[idx]["obsname"] = names[idx]

        if self.nobs > 1:
            self.irefsp = -self.nobs
        else:
            self.irefsp = self.time_series_data[0]["irefsp"]

    def _get_empty(self, ncells=0):
        """
        Get an empty time_series_data recarray for a HeadObservation

        Parameters
        ----------
        ncells : int
            number of time entries in a HeadObservation

        Returns
        -------
        d : np.recarray

        """
        # get an empty recarray that corresponds to dtype
        dtype = self._get_dtype()
        d = create_empty_recarray(ncells, dtype, default_value=-1.0e10)
        d["obsname"] = ""
        return d

    def _get_dtype(self):
        """
        Get the dtype for HeadObservation time_series_data


        Returns
        -------
        dtype : np.dtype

        """
        # get the default HOB dtype
        dtype = np.dtype(
            [
                ("totim", np.float32),
                ("irefsp", int),
                ("toffset", np.float32),
                ("hobs", np.float32),
                ("obsname", "|S12"),
            ]
        )
        return dtype
