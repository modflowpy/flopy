"""
mfmlt module.  Contains the ModflowMlt class. Note that the user can access
the ModflowMlt class as `flopy.modflow.ModflowMlt`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/mult.htm>`_.

"""
import numpy as np

from ..pakbase import Package
from ..utils import Util2d


class ModflowMlt(Package):
    """
    MODFLOW Mult Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mult_dict : dict
        Dictionary with mult data for the model. mult_dict is typically
        instantiated using load method.
    extension : string
        Filename extension (default is 'drn')
    unitnumber : int
        File unit number (default is 21).


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
    >>> mltdict = flopy.modflow.ModflowZon(m, mult_dict=mult_dict)

    """

    def __init__(
        self,
        model,
        mult_dict=None,
        extension="mlt",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowMlt._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowMlt._ftype()]
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

        self._generate_heading()
        self.url = "mult.htm"

        self.nml = 0
        if mult_dict is not None:
            self.nml = len(mult_dict)
            self.mult_dict = mult_dict
            # print mult_dict
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        Notes
        -----
        Not implemented because parameters are only supported on load

        """
        pass

    @classmethod
    def load(cls, f, model, nrow=None, ncol=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        nrow : int
            number of rows. If not specified it will be retrieved from
            the model object. (default is None).
        ncol : int
            number of columns. If not specified it will be retrieved from
            the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        zone : ModflowMult dict

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> mlt = flopy.modflow.ModflowMlt.load('test.mlt', m)

        """

        if model.verbose:
            print("loading mult package file...")

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
        t = line.strip().split()
        nml = int(t[0])

        # get nlay,nrow,ncol if not passed
        if nrow is None and ncol is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # read zone data
        mult_dict = {}
        for n in range(nml):
            line = f.readline()
            t = line.strip().split()
            if len(t[0]) > 10:
                mltnam = t[0][0:10].lower()
            else:
                mltnam = t[0].lower()
            if model.verbose:
                print(f'   reading data for "{mltnam:<10s}" mult')
            readArray = True
            kwrd = None
            if len(t) > 1:
                if "function" in t[1].lower() or "expression" in t[1].lower():
                    readArray = False
                    kwrd = t[1].lower()
            # load data
            if readArray:
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, mltnam, ext_unit_dict
                )
                # add unit number to list of external files in
                # ext_unit_dict to remove.
                if t.locat is not None:
                    model.add_pop_key_list(t.locat)
            else:
                line = f.readline()
                t = [kwrd, line]
                t = ModflowMlt.mult_function(mult_dict, line)
            mult_dict[mltnam] = t

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowMlt._ftype()
            )

        return cls(
            model,
            mult_dict=mult_dict,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def mult_function(mult_dict, line):
        """
        Construct a multiplier for the 'FUNCTION' option

        """
        t = line.strip().split()
        basename = t.pop(0).lower()[0:10]
        multarray = mult_dict[basename]
        try:
            multarray = multarray.array.copy()
        except:
            multarray = multarray.copy()
        # Construct the multiplier array
        while True:
            if len(t) < 2:
                break
            op = t.pop(0)
            multname = t.pop(0)[0:10]
            try:
                atemp = mult_dict[multname.lower()].array
            except:
                atemp = mult_dict[multname.lower()]
            if op == "+":
                multarray = multarray + atemp
            elif op == "*":
                multarray = multarray * atemp
            elif op == "-":
                multarray = multarray - atemp
            elif op == "/":
                multarray = multarray / atemp
            elif op == "^":
                multarray = multarray ** atemp
            else:
                raise Exception(f"Invalid MULT operation {op}")
        return multarray

    @staticmethod
    def _ftype():
        return "MULT"

    @staticmethod
    def _defaultunit():
        return 1002
