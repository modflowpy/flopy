"""
mfgage module.  Contains the ModflowGage class. Note that the user can access
the ModflowGage class as `flopy.modflow.ModflowGage`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/gage.htm>`_.

"""
import os
import sys

import numpy as np

from ..pakbase import Package
from ..utils import read_fixed_var, write_fixed_var
from ..utils.recarray_utils import create_empty_recarray


class ModflowGage(Package):
    """
    MODFLOW Gage Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    numgage : int
        The total number of gages included in the gage file (default is 0).
    gage_data : list or numpy array
        data for dataset 2a and 2b in the gage package. If a list is provided
        then the list includes 2 to 3 entries (LAKE UNIT [OUTTYPE]) for each
        LAK Package entry and 4 entries (GAGESEG GAGERCH UNIT OUTTYPE) for
        each SFR Package entry. If a numpy array it passed each gage location
        must have 4 entries, where LAK Package gages can have any value for the
        second column. The numpy array can be created using the get_empty()
        method available in ModflowGage. Default is None
    files : list of strings
        Names of gage output files. A file name must be provided for each gage.
        If files are not provided and filenames=None then a gage name will be
        created using the model name and the gage number (for example,
        modflowtest.gage1.go). Default is None.
    extension : string
        Filename extension (default is 'gage')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and gage output names will be created using the
        model name and the gage number (for example, modflowtest.gage1.go).
        If a single string is passed the package will be set to the string
        and gage output names will be created using the model name and the
        gage number. To define the names for all gage files (input and output)
        the length of the list of strings should be numgage + 1.
        Default is None.

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
    >>> gages = [[-1,  -26, 1], [-2,  -27, 1]]
    >>> files = ['gage1.go', 'gage2.go']
    >>> gage = flopy.modflow.ModflowGage(m, numgage=2,
    >>>                                  gage_data=gages, files=files)

    """

    def __init__(
        self,
        model,
        numgage=0,
        gage_data=None,
        files=None,
        extension="gage",
        unitnumber=None,
        filenames=None,
        **kwargs
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowGage._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None for x in range(numgage + 1)]
        elif isinstance(filenames, str):
            filenames = [filenames] + [None for x in range(numgage)]
        elif isinstance(filenames, list):
            if len(filenames) < numgage + 1:
                for idx in range(len(filenames), numgage + 2):
                    filenames.append(None)

        # process gage output files
        dtype = ModflowGage.get_default_dtype()
        if numgage > 0:
            # check the provided file entries
            if filenames[1] is None:
                if files is None:
                    files = []
                    for idx in range(numgage):
                        files.append(
                            "{}.gage{}.go".format(model.name, idx + 1)
                        )
                if isinstance(files, np.ndarray):
                    files = files.flatten().tolist()
                elif isinstance(files, str):
                    files = [files]
                elif isinstance(files, int) or isinstance(files, float):
                    files = ["{}.go".format(files)]
                if len(files) < numgage:
                    raise Exception(
                        "a filename needs to be provided for {} gages - {} "
                        "filenames were provided".format(numgage, len(files))
                    )
            else:
                if len(filenames) < numgage + 1:
                    raise Exception(
                        "filenames must have a length of {} the length "
                        "provided is {}".format(numgage + 1, len(filenames))
                    )
                else:
                    files = []
                    for n in range(numgage):
                        files.append(filenames[n + 1])

            # convert gage_data to a recarray, if necessary
            if isinstance(gage_data, np.ndarray):
                if not gage_data.dtype == dtype:
                    gage_data = np.core.records.fromarrays(
                        gage_data.transpose(), dtype=dtype
                    )
            elif isinstance(gage_data, list):
                d = ModflowGage.get_empty(ncells=numgage)
                for n in range(len(gage_data)):
                    t = gage_data[n]
                    gageloc = int(t[0])
                    if gageloc < 0:
                        gagerch = 0
                        iu = int(t[1])
                        outtype = 0
                        if iu < 0:
                            outtype = int(t[2])
                    else:
                        gagerch = int(t[1])
                        iu = int(t[2])
                        outtype = int(t[3])

                    d["gageloc"][n] = gageloc
                    d["gagerch"][n] = gagerch
                    d["unit"][n] = iu
                    d["outtype"][n] = outtype
                gage_data = d
            else:
                raise Exception(
                    "gage_data must be a numpy record array, numpy array "
                    "or a list"
                )

            # add gage output files to model
            for n in range(numgage):
                iu = abs(gage_data["unit"][n])
                fname = files[n]
                model.add_output_file(
                    iu,
                    fname=fname,
                    binflag=False,
                    package=ModflowGage._ftype(),
                )

        # Fill namefile items
        name = [ModflowGage._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.url = "gage.htm"

        self.numgage = numgage
        self.files = files

        self.dtype = self.get_default_dtype()

        self.gage_data = gage_data

        self.parent.add_package(self)

        return

    @staticmethod
    def get_default_dtype():
        dtype = np.dtype(
            [
                ("gageloc", int),
                ("gagerch", int),
                ("unit", int),
                ("outtype", int),
            ]
        )
        return dtype

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowGage.get_default_dtype()
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    def _ncells(self):
        """Maximum number of cells that have gages (developed for MT3DMS
        SSM package). Return zero because gage is not added to SSM package.

        Returns
        -------
        ncells: int
            0

        """
        return 0

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, "w")

        # # dataset 0
        # vn = self.parent.version_types[self.parent.version]
        # self.heading = '# {} package for '.format(self.name[0]) + \
        #                '{}, generated by Flopy.'.format(vn)
        # f.write('{0}\n'.format(self.heading))

        # dataset 1
        f.write(write_fixed_var([self.numgage], free=True))

        # dataset 2
        for n in range(self.numgage):
            gageloc = self.gage_data["gageloc"][n]
            gagerch = self.gage_data["gagerch"][n]
            iu = self.gage_data["unit"][n]
            outtype = self.gage_data["outtype"][n]
            t = [gageloc]
            if gageloc < 0:
                t.append(iu)
                if iu < 0:
                    t.append(outtype)
            else:
                t.append(gagerch)
                t.append(iu)
                t.append(outtype)
            f.write(write_fixed_var(t, free=True))

        # close the gage file
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
        str : ModflowStr object
            ModflowStr object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> gage = flopy.modflow.ModflowGage.load('test.gage', m)

        """

        if model.verbose:
            sys.stdout.write("loading gage package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r", errors="replace")

        # dataset 0 -- header
        while True:
            line = f.readline().rstrip()
            if line[0] != "#":
                break

        # read dataset 1
        if model.verbose:
            print("   reading gage dataset 1")
        t = read_fixed_var(line, free=True)
        numgage = int(t[0])

        if numgage == 0:
            gage_data = None
            files = None
        else:
            # read dataset 2
            if model.verbose:
                print("   reading gage dataset 2")

            gage_data = ModflowGage.get_empty(ncells=numgage)
            files = []

            for n in range(numgage):
                line = f.readline().rstrip()
                t = read_fixed_var(line, free=True)
                gageloc = int(t[0])
                if gageloc < 0:
                    gagerch = 0
                    iu = int(t[1])
                    outtype = 0
                    if iu < 0:
                        outtype = int(t[2])
                else:
                    gagerch = int(t[1])
                    iu = int(t[2])
                    outtype = int(t[3])
                gage_data["gageloc"][n] = gageloc
                gage_data["gagerch"][n] = gagerch
                gage_data["unit"][n] = iu
                gage_data["outtype"][n] = outtype

                for key, value in ext_unit_dict.items():
                    if key == abs(iu):
                        model.add_pop_key_list(abs(iu))
                        relpth = os.path.relpath(
                            value.filename, model.model_ws
                        )
                        files.append(relpth)
                        break

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = []
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowGage._ftype():
                    unitnumber = key
                    filenames.append(os.path.basename(value.filename))
        for file in files:
            filenames.append(os.path.basename(file))

        return cls(
            model,
            numgage=numgage,
            gage_data=gage_data,
            filenames=filenames,
            unitnumber=unitnumber,
        )

    @staticmethod
    def _ftype():
        return "GAGE"

    @staticmethod
    def _defaultunit():
        return 120
