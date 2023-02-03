"""
mp7bas module.  Contains the Modpath7Bas class.

"""
import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d


class Modpath7Bas(Package):
    """
    MODPATH 7 Basic Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modpath.Modpath7`) to which
        this package will be added.
    porosity : float or array of floats (nlay, nrow, ncol)
        The porosity array (the default is 0.30).
    defaultiface : dict
        Dictionary with keys that are the text string used by MODFLOW in
        the budget output file to label flow rates for a stress package
        and the values are the cell face (iface) on which to assign flows
        (the default is None).
    extension : str, optional
        File extension (default is 'mpbas').

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow.load('mf2005.nam')
    >>> mp = flopy.modpath.Modpath7('mf2005_mp', flowmodel=m)
    >>> mpbas = flopy.modpath.Modpath7Bas(mp)

    """

    def __init__(
        self, model, porosity=0.30, defaultiface=None, extension="mpbas"
    ):

        unitnumber = model.next_unit()

        super().__init__(model, extension, "MPBAS", unitnumber)

        shape = model.shape
        if len(shape) == 3:
            shape3d = shape
        elif len(shape) == 2:
            shape3d = (shape[0], 1, shape[1])
        else:
            shape3d = (1, 1, shape[0])

        self._generate_heading()

        self.laytyp = Util2d(
            self.parent,
            (shape[0],),
            np.int32,
            model.laytyp,
            name="bas - laytype",
            locat=self.unit_number[0],
        )
        if model.flowmodel.version != "mf6":
            self.ibound = Util3d(
                model,
                shape3d,
                np.int32,
                model.ibound,
                name="IBOUND",
                locat=self.unit_number[0],
            )

        self.porosity = Util3d(
            model,
            shape3d,
            np.float32,
            porosity,
            name="POROSITY",
            locat=self.unit_number[0],
        )

        # validate and set defaultiface
        if defaultiface is None:
            defaultifacecount = 0
        else:
            if not isinstance(defaultiface, dict):
                raise ValueError(
                    "defaultiface must be a dictionary with package "
                    "name keys and values between 0 and 6"
                )
            defaultifacecount = len(defaultiface.keys())
            for key, value in defaultiface.items():
                # check iface value
                if value < 0 or value > 6:
                    raise ValueError(
                        "defaultiface for package {} must be between 0 and 1 "
                        "({} specified)".format(key, value)
                    )

        self.defaultifacecount = defaultifacecount
        self.defaultiface = defaultiface

        self.parent.add_package(self)

    def write_file(self, check=False):
        """
        Write the package file

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default False)

        Returns
        -------
        None

        """
        # Open file for writing
        f = open(self.fn_path, "w")
        f.write(f"# {self.heading}\n")
        if self.parent.flowmodel.version != "mf6":
            f.write(f"{self.parent.hnoflo:g} {self.parent.hdry:g}\n")

        # default IFACE
        f.write(f"{self.defaultifacecount:<20d}# DEFAULTIFACECOUNT\n")
        if self.defaultifacecount > 0:
            for key, value in self.defaultiface.items():
                f.write(f"{key:20s}# PACKAGE LABEL\n")
                f.write(f"{value:<20d}# DEFAULT IFACE VALUE\n")

        # laytyp
        if self.parent.flow_version != "mf6":
            f.write(self.laytyp.string)

        # ibound
        if self.parent.flow_version != "mf6":
            f.write(self.ibound.get_file_entry())

        # porosity
        f.write(self.porosity.get_file_entry())

        f.close()
