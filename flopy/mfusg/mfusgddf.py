"""
mfusgddf module.

Contains the MfUsgDdf class. Note that the user can access
the MfUsgDdf class as `flopy.mfusg.MfUsgDdf`.
"""

from ..pakbase import Package
from ..utils.flopy_io import line_parse
from ..utils.utils_def import (
    get_unitnumber_from_ext_unit_dict,
    type_from_iterable,
)
from .mfusg import MfUsg


class MfUsgDdf(Package):
    """MODFLOW-USG Transport Density Driven Flow (DDF) package class.

    parameters
    ----------
    model : model object
        the model object (of type :class:`flopy.mfusg.MfUsg`) to which
        this package will be added.
    rhofresh : density of freshwater
    rhostd   : density of standard solution
    cstd     : concentration of standard solution
    ithickav : a flag indicating if thickness weighted averaging for density term
              0 = arithmetic averaging; 1 = thickness weighting averaging
    imphdd   : a flag of hydraulic head term in the density formulation
              0 = explicit (on the right-hand side vector) symmetry of the matrix
              1 = implicit (on the left-hand side matrix) asymmetric matrix
    extension : str, optional
        file extension (default is 'ddf').
    unitnumber : int, optional
        fortran unit number for this package (default is none).
    filenames : str or list of str
        filenames to use for the package. if filenames=none the package name
        will be created using the model name and package extension. if a
        single string is passed the package will be set to the string.
        default is none.

    attributes
    ----------

    methods
    -------

    see also
    --------

    notes
    -----

    examples
    --------

    >>> import flopy
    >>> m = flopy.mfusg.mfusg()
    >>> ddf = flopy.mfusg.mfusgddf(m)
    """

    def __init__(
        self,
        model,
        rhofresh=1000.0,
        rhostd=1025.0,
        cstd=35.0,
        ithickav=1,
        imphdd=0,
        extension="ddf",
        unitnumber=None,
        filenames=None,
    ):
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = self._defaultunit()

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )

        self._generate_heading()
        self.rhofresh = rhofresh
        self.rhostd = rhostd
        self.cstd = cstd
        self.ithickav = ithickav
        self.imphdd = imphdd

        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, "w")
        f.write(f"{self.heading}\n")
        f.write(
            f" {self.rhofresh:9.2f} {self.rhostd:9.2f} {self.cstd:9.2f}"
            f" {self.ithickav:9d} {self.imphdd:9d}\n"
        )
        f.close()

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
        ddf : MfUsgDdf object

        Examples
        --------

        >>> import flopy
        >>> ml = flopy.mfusg.MfUsg()
        >>> ddf = flopy.mfusg.MfUsgDdf.load('Henry.ddf', ml)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading ddf package file...")

        if model.version != "mfusg":
            print(
                "Warning: model version was reset from '{}' to 'mfusg' "
                "in order to load a DDF file".format(model.version)
            )
            model.version = "mfusg"

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        line = f.readline().upper()
        while line.startswith("#"):
            line = f.readline().upper()

        if model.verbose:
            print("   loading RHOFRESH RHOSTD CSTD ITHICKAV IMPHDD...")

        ll = line_parse(line)
        rhofresh = float(ll.pop(0))
        rhostd = float(ll.pop(0))
        cstd = float(ll.pop(0))
        ithickav = type_from_iterable(ll, index=3, _type=int, default_val=1)
        imphdd = type_from_iterable(ll, index=4, _type=int, default_val=0)
        if model.verbose:
            print(
                f"   RHOFRESH {rhofresh} \n   RHOSTD {rhostd} \n   CSTD {cstd} \n"
                f"   ITHICKAV {ithickav} \n   IMPHDD {imphdd}"
            )

        if openfile:
            f.close()

        # set package unit number
        unitnumber, filenames = get_unitnumber_from_ext_unit_dict(
            model,
            cls,
            ext_unit_dict,
        )

        return cls(
            model,
            rhofresh=rhofresh,
            rhostd=rhostd,
            cstd=cstd,
            ithickav=ithickav,
            imphdd=imphdd,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "DDF"

    @staticmethod
    def _defaultunit():
        return 152
