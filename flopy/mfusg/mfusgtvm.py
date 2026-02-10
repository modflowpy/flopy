"""
mfusgtvm module.

Contains the MfUsgTvm class. Note that the user can access
the MfUsgTvm class as `flopy.mfusg.MfUsgTvm`.

Time-Varying Materials (TVM) Package for MODFLOW-USG.
"""

import numpy as np

from ..pakbase import Package
from .mfusg import MfUsg


class MfUsgTvm(Package):
    """
    MODFLOW-USG Time-Varying Materials Package Class.

    The TVM package allows hydraulic properties (HK, VKA, SS, SY, DDFTR, POR)
    to vary during a simulation. Properties are interpolated between stress
    period boundaries.

    Parameters
    ----------
    model : MfUsg
        Model object (must be of type flopy.mfusg.MfUsg)
    itvmprint : int
        Print flag. 0=no output, 1=summary, 2=detailed (default 0)
    tvmlogbasehk : float
        Interpolation type for HK.
        0 = linear interpolation
        > 0 = logarithmic interpolation with this base
        < 0 = step function (default 0)
    tvmlogbasevka : float
        Interpolation type for VKA. Same options as tvmlogbasehk (default 0)
    tvmlogbasess : float
        Interpolation type for SS. 0=linear, >0=log base.
        Note: step function not supported for SS (default 0)
    tvmlogbasesy : float
        Interpolation type for SY. 0=linear, >0=log base.
        Note: step function not supported for SY (default 0)
    tvmddftr : float
        Interpolation type for DDFTR (dual domain flow transfer rate).
        Same options as tvmlogbasehk (default 0)
    tvmlogbasepor : float
        Interpolation type for porosity. 0=linear, >0=log base.
        Note: step function not supported for porosity (default 0)
    stress_period_data : dict
        Dictionary keyed by stress period boundary index.
        Boundary 0 = start of stress period 1
        Boundary 1 = end of stress period 1 (= start of SP2)
        Boundary N = end of stress period N

        Each value is a dict with optional keys:
        'hk', 'vka', 'ss', 'sy', 'ddftr', 'por'

        Each property is a list of tuples: [(node, value), ...]

        Example::

            stress_period_data = {
                0: {'hk': [(100, 1e-5), (101, 1e-5)]},  # Start SP1
                1: {'hk': [(100, 1e-6), (101, 1e-6)]},  # End SP1
                5: {'hk': [(100, 1e-7)]},               # End SP5
            }

    extension : str
        Filename extension (default 'tvm')
    unitnumber : int
        File unit number (default None)
    filenames : str or list of str
        Filenames to use for the package (default None)
    add_package : bool
        Flag to add the package to the model (default True)

    Attributes
    ----------
    heading : str
        Package heading

    Methods
    -------
    write_file()
        Write the package file

    See Also
    --------
    flopy.mfusg.MfUsgLpf : Layer Property Flow package
    flopy.mfusg.MfUsgBcf : Block Centered Flow package

    Notes
    -----
    The TVM package was developed by Damian Merrick (HydroAlgorithmics)
    and Vivek Bedekar for MODFLOW-USG.

    Properties change at stress period boundaries and are interpolated
    during time steps within each stress period.

    Interpolation options:
    - Linear (0): Value changes linearly between boundaries
    - Logarithmic (>0): Value changes logarithmically with specified base
    - Step (<0): Value maintains start value until end of stress period

    Examples
    --------
    >>> import flopy
    >>> m = flopy.mfusg.MfUsg(modelname='test', model_ws='.')
    >>> # Change HK for nodes 100-102 gradually from SP1 to SP5
    >>> spd = {
    ...     0: {'hk': [(100, 1e-4), (101, 1e-4), (102, 1e-4)]},
    ...     5: {'hk': [(100, 1e-6), (101, 1e-6), (102, 1e-6)]},
    ... }
    >>> tvm = flopy.mfusg.MfUsgTvm(m, stress_period_data=spd)

    """

    def __init__(
        self,
        model,
        itvmprint=0,
        tvmlogbasehk=0.0,
        tvmlogbasevka=0.0,
        tvmlogbasess=0.0,
        tvmlogbasesy=0.0,
        tvmddftr=0.0,
        tvmlogbasepor=0.0,
        stress_period_data=None,
        extension="tvm",
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        """Package constructor."""
        # Validate model type
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        # Set default unit number
        if unitnumber is None:
            unitnumber = MfUsgTvm._defaultunit()

        # Prepare filenames
        filenames = MfUsgTvm._prepare_filenames(filenames)

        # Call parent constructor
        super().__init__(
            model,
            extension=extension,
            name=MfUsgTvm._ftype(),
            unit_number=unitnumber,
            filenames=filenames,
        )

        self._generate_heading()

        # Store parameters
        self.itvmprint = int(itvmprint)
        self.tvmlogbasehk = float(tvmlogbasehk)
        self.tvmlogbasevka = float(tvmlogbasevka)
        self.tvmlogbasess = float(tvmlogbasess)
        self.tvmlogbasesy = float(tvmlogbasesy)
        self.tvmddftr = float(tvmddftr)
        self.tvmlogbasepor = float(tvmlogbasepor)
        self.stress_period_data = stress_period_data if stress_period_data else {}

        # Add package to model
        if add_package:
            self.parent.add_package(self)

    @staticmethod
    def _prepare_filenames(filenames):
        """Prepare filenames for the package."""
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]
        return filenames

    def write_file(self):
        """
        Write the TVM package file.

        Returns
        -------
        None
        """
        # Open file
        f_tvm = open(self.fn_path, "w")

        # Write header comment
        f_tvm.write(f"# {self.heading}\n")

        # Line 1: Interpolation options
        # ITVMPRINT TVMLOGBASEHK TVMLOGBASEVKA TVMLOGBASESS ...
        f_tvm.write(
            f"{self.itvmprint:10d}"
            f"{self.tvmlogbasehk:10.3f}"
            f"{self.tvmlogbasevka:10.3f}"
            f"{self.tvmlogbasess:10.3f}"
            f"{self.tvmlogbasesy:10.3f}"
            f"{self.tvmddftr:10.3f}"
            f"{self.tvmlogbasepor:10.3f}\n"
        )

        # Get number of stress periods
        nper = self.parent.nper

        # Write data for each stress period boundary
        # Boundary 0 = start of SP1, Boundary 1 = end of SP1, etc.
        for iper in range(nper + 1):
            spd = self.stress_period_data.get(iper, {})

            # Get data for each property
            hk_data = spd.get("hk", [])
            vka_data = spd.get("vka", [])
            ss_data = spd.get("ss", [])
            sy_data = spd.get("sy", [])
            ddftr_data = spd.get("ddftr", [])
            por_data = spd.get("por", [])

            # Line 2: Counts for each property
            # NTVMHK NTVMVKA NTVMSS NTVMSY NTVMDDFTR NTVMPOR
            f_tvm.write(
                f"{len(hk_data):10d}"
                f"{len(vka_data):10d}"
                f"{len(ss_data):10d}"
                f"{len(sy_data):10d}"
                f"{len(ddftr_data):10d}"
                f"{len(por_data):10d}\n"
            )

            # Write HK data (ITVMHK HKNEW)
            for node, value in hk_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

            # Write VKA data (ITVMVKA VKANEW)
            for node, value in vka_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

            # Write SS data (ITVMSS SSNEW)
            for node, value in ss_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

            # Write SY data (ITVMSY SYNEW)
            for node, value in sy_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

            # Write DDFTR data (ITVMDDFTR DDFTRNEW)
            for node, value in ddftr_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

            # Write POR data (ITVMPOR PORNEW)
            for node, value in por_data:
                f_tvm.write(f"{node:10d} {value:15.6E}\n")

        f_tvm.close()

    @staticmethod
    def _ftype():
        return "TVM"

    @staticmethod
    def _defaultunit():
        return 71

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing TVM package from a file.

        Parameters
        ----------
        f : str or file-like object
            Filename or file handle
        model : MfUsg
            Model object
        ext_unit_dict : dict, optional
            External unit dictionary

        Returns
        -------
        tvm : MfUsgTvm
            MfUsgTvm object

        Examples
        --------
        >>> tvm = MfUsgTvm.load('model.tvm', model)

        """
        raise NotImplementedError(
            "TVM load method not yet implemented. "
            "Contributions welcome at https://github.com/modflowpy/flopy"
        )
