"""
mfzon module.  Contains the ModflowZone class. Note that the user can access
the ModflowZone class as `flopy.modflow.ModflowZone`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/zone.htm>`_.

"""
import numpy as np
from ..pakbase import Package
from ..utils import Util2d


class ModflowZon(Package):
    """
    MODFLOW Zone Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    zone_dict : dict
        Dictionary with zone data for the model. zone_dict is typically
        instantiated using load method.
    extension : string
        Filename extension (default is 'zon')
    unitnumber : int
        File unit number (default is None).
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
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> zonedict = flopy.modflow.ModflowZon(m, zone_dict=zone_dict)

    """

    def __init__(
        self,
        model,
        zone_dict=None,
        extension="zon",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowZon._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowZon._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
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
        self.url = "zone.htm"

        self.nzn = 0
        if zone_dict is not None:
            self.nzn = len(zone_dict)
            self.zone_dict = zone_dict
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
        return

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
        zone : ModflowZone dict

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> zon = flopy.modflow.ModflowZon.load('test.zon', m)

        """

        if model.verbose:
            print("loading zone package file...")

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
        nzn = int(t[0])

        # get nlay,nrow,ncol if not passed
        if nrow is None and ncol is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # read zone data
        zone_dict = {}
        for n in range(nzn):
            line = f.readline()
            t = line.strip().split()
            if len(t[0]) > 10:
                zonnam = t[0][0:10].lower()
            else:
                zonnam = t[0].lower()
            if model.verbose:
                print(f'   reading data for "{zonnam:<10s}" zone')
            # load data
            t = Util2d.load(
                f, model, (nrow, ncol), np.int32, zonnam, ext_unit_dict
            )
            # add unit number to list of external files in ext_unit_dict
            # to remove.
            if t.locat is not None:
                model.add_pop_key_list(t.locat)
            zone_dict[zonnam] = t

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowZon._ftype()
            )

        return cls(
            model,
            zone_dict=zone_dict,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "ZONE"

    @staticmethod
    def _defaultunit():
        return 1001
