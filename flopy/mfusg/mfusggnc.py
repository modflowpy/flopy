"""
Mfusggnc module.

This is for the Ghost Node Correction (GNC) Package for MODFLOW-USG.
Contains the MfUsgGnc class. Note that the user can access
the MfUsgGnc class as `flopy.mfusg.MfUsgGnc`.
"""
import numpy as np

from ..modflow.mfparbc import ModflowParBc as mfparbc
from ..pakbase import Package
from ..utils.flopy_io import ulstrd
from ..utils.recarray_utils import create_empty_recarray
from .mfusg import MfUsg, fmt_string


class MfUsgGnc(Package):
    """MODFLOW USG Ghost Node Correction (GNC) Package Class.

    Parameters
    ----------
    numgnc : integer
        numgnc (integer) is the number of GNC entries.
    numalphaj : integer
        numalphaj (integer) is the number of contributing factors.
    i2kn : integer
        0 : second-order correction not applied to unconfined transmissivity.
        1 : second-order correction applied to unconfined transmissivity.
    isymgncn : integer
        0 : implicit update on left-hand side matrix for asymmetric systems.
        1 : explicit update on right-hand side vector for symmetric systems.
    iflalphan : integer
        0 : AlphaJ is contributing factors from all adjacent contributing nodes.
        1 : AlphaJ represent the saturated conductances between the ghost node
            location and node j, and the contributing factors are computed
            internally using the equations for the unconfined conductances.
    gncdata : [cellidn, cellidm, cellidsj, alphasj]
        * cellidn ((integer, ...)) is the cellid of the cell in which the ghost
          node is located.
        * cellidm ((integer, ...)) is the cellid of the connecting cell
        * cellidsj ((integer, ...)) is the array of CELLIDS for the
          contributing j cells. This Item is repeated for each of the numalphaj
          adjacent contributing cells of the ghost node.
        * alphasj (double) is the contributing factors for each contributing
          node in CELLIDSJ. This Item is repeated for each of the numalphaj
          adjacent contributing cells of the ghost node.
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    extension : str, optional
        File extension (default is 'gnc'.
    unitnumber : int, optional
        FORTRAN unit number for this package (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
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
    >>> gnc = flopy.mfusg.MfUsgGnc(m)
    """

    def __init__(
        self,
        model,
        numgnc=0,
        numalphaj=1,
        i2kn=0,
        isymgncn=0,
        iflalphan=0,
        gncdata=None,
        extension="gnc",
        options=None,
        unitnumber=None,
        filenames=None,
    ):
        """Package constructor."""
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = self._defaultunit()

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )

        self._generate_heading()

        if options is None:
            options = []

        self.options = options

        if numgnc > 0:
            self.numgnc = numgnc
        else:
            raise Exception("mfgnc: number of GNC cell must be larger than 0")

        if 0 < numalphaj < 6:
            self.numalphaj = numalphaj
        else:
            raise Exception(
                "mfgnc: incorrect number of adjacent contributing nodes"
            )

        self.i2kn = i2kn
        self.isymgncn = isymgncn
        self.iflalphan = iflalphan

        if gncdata is None:
            raise Exception("mfgnc: GNC data must be provided")

        if len(gncdata) != self.numgnc:
            raise Exception(
                "mfgnc: Length of GNC data must equal number of GNC nodes"
            )

        self.dtype = MfUsgGnc.get_default_dtype(self.numalphaj, self.iflalphan)

        self.gncdata = np.array(gncdata, self.dtype)

        self.parent.add_package(self)

    def write_file(self, f=None, check=False):
        """Write the package file.

        Parameters
        ----------
        f : filename or file handle
            File to write to.

        Returns
        -------
        None

        """
        if f is not None:
            if isinstance(f, str):
                f_gnc = open(f, "w")
            else:
                f_gnc = f
        else:
            f_gnc = open(self.fn_path, "w")

        if check:
            raise NotImplementedError(
                "Warning: mfgnc package check not yet implemented."
            )

        f_gnc.write(f"{self.heading}\n")

        f_gnc.write(
            f" {0:9d} {0:9d} {self.numgnc:9d} {self.numalphaj:9d}"
            f" {self.i2kn:9d} {self.isymgncn:9d} {self.iflalphan:9d}"
            f" {self.options}\n"
        )

        gdata = self.gncdata.copy()

        gdata["NodeN"] += 1
        gdata["NodeM"] += 1
        for idx in range(self.numalphaj):
            gdata[f"Node{idx:d}"] += 1

        np.savetxt(f_gnc, gdata, fmt=fmt_string(gdata), delimiter="")

        f_gnc.write("\n")
        f_gnc.close()

    @staticmethod
    def get_default_dtype(numalphaj, iflalphan):
        """Returns default GNC dtypes."""
        dtype = np.dtype(
            [
                ("NodeN", int),
                ("NodeM", int),
            ]
        ).descr

        for idx in range(numalphaj):
            dtype.append((f"Node{idx:d}", "<i4"))
        for idx in range(numalphaj):
            dtype.append((f"Alpha{idx:d}", "<f4"))

        if iflalphan == 1:
            dtype.append(("AlphaN", "<f4"))

        return np.dtype(dtype)

    @staticmethod
    def get_empty(numgnc=0, numalphaj=1, iflalphan=0):
        """Returns empty GNC recarray of defualt dtype."""
        # get an empty recarray that corresponds to dtype
        dtype = MfUsgGnc.get_default_dtype(numalphaj, iflalphan)
        return create_empty_recarray(numgnc, dtype, default_value=-1.0e10)

    @classmethod
    def load(cls, f, model, pak_type="gnc", ext_unit_dict=None, **kwargs):
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
        gnc : MfUsgGnc object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> gnc = flopy.modflow.ModflowGnc.load('test.gnc', m)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading gnc package file...")

        if model.version != "mfusg":
            print(
                "Warning: model version was reset from"
                f"'{model.version}' to 'mfusg' to load a GNC file"
            )
            model.version = "mfusg"

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Item 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # Item 1 --NPGNCn MXGNn NGNCNPn MXADJn I2Kn ISYMGNCn IFLALPHAn [NOPRINT]
        line_text = line.strip().split()
        imax = 7
        npgncn, mxgnn, numgnc, numalphaj, i2kn, isymgncn, iflalphan = (
            int(line_text[0]),
            int(line_text[1]),
            int(line_text[2]),
            int(line_text[3]),
            int(line_text[4]),
            int(line_text[5]),
            int(line_text[6]),
        )

        options = []
        if len(line_text) > imax:
            if line_text[7].lower() == "noprint":
                options.append("noprint")

        # Item 2 -- read parameter data
        if npgncn > 0:
            dtype = MfUsgGnc.get_empty(npgncn, mxgnn, iflalphan).dtype
            # Item 3 --
            mfparbc.load(f, npgncn, dtype, model, ext_unit_dict, model.verbose)

        # Item 4 -- read GNC data
        gncdata = MfUsgGnc.get_empty(numgnc, numalphaj, iflalphan)

        gncdata = ulstrd(f, numgnc, gncdata, model, [], ext_unit_dict)

        gncdata["NodeN"] -= 1
        gncdata["NodeM"] -= 1
        for idx in range(numalphaj):
            gncdata[f"Node{idx:d}"] -= 1

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )

        return cls(
            model,
            numgnc=numgnc,
            numalphaj=numalphaj,
            i2kn=i2kn,
            isymgncn=isymgncn,
            iflalphan=iflalphan,
            gncdata=gncdata,
            options=options,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "GNC"

    @staticmethod
    def _defaultunit():
        return 72
