"""
mfusgtib module.

Contains the MfUsgTib class. Note that the user can access
the MfUsgTib class as `flopy.mfusg.MfUsgTib`.

Transient IBOUND (TIB) Package for MODFLOW-USG.
"""

import numpy as np

from ..pakbase import Package
from .mfusg import MfUsg


class MfUsgTib(Package):
    """
    MODFLOW-USG Transient IBOUND Package Class.

    The TIB package allows the IBOUND array to change during a simulation.
    Cells can be activated, deactivated, or converted to constant head.
    This is useful for simulating:
    - Excavation or reclamation
    - Well drilling or plugging
    - Lake/reservoir filling or draining
    - Mining operations

    Parameters
    ----------
    model : MfUsg
        Model object (must be of type flopy.mfusg.MfUsg)
    stress_period_data : dict
        Dictionary keyed by stress period number (0-based).
        Each value is a dict with optional keys:

        - 'ib0': list of int
            Node numbers to deactivate (set IBOUND=0)
        - 'ib1': list of dict
            Nodes to activate (set IBOUND=1). Each dict has:
            - 'node': int (required) - node number
            - 'head': float (optional) - initial head value
            - 'avhead': bool (optional) - use average head of neighbors
        - 'ibm1': list of dict
            Nodes to set as constant head (IBOUND=-1). Each dict has:
            - 'node': int (required) - node number
            - 'head': float (optional) - prescribed head value
            - 'avhead': bool (optional) - use average head of neighbors

        For transport simulations, additional keys are available:
        - 'icb0': list of int - deactivate ICBUND
        - 'icb1': list of dict - activate ICBUND
        - 'icbm1': list of dict - set prescribed concentration

        Example::

            stress_period_data = {
                0: {
                    'ib0': [100, 101, 102],  # Deactivate nodes
                },
                5: {
                    'ibm1': [
                        {'node': 200, 'head': 2900.0},
                        {'node': 201, 'head': 2900.0},
                    ],
                },
                10: {
                    'ib1': [
                        {'node': 200, 'avhead': True},  # Use avg head
                    ],
                },
            }

    extension : str
        Filename extension (default 'tib')
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
    flopy.mfusg.MfUsgBas : Basic package with initial IBOUND

    Notes
    -----
    The TIB package provides flexibility to change the IBOUND array
    during the simulation. This is particularly useful for:

    - Simulating excavation/reclamation by activating/deactivating cells
    - Simulating well drilling/plugging
    - Simulating filling/draining of lakes or reservoirs
    - Mining operations where voids are created or filled

    When activating a cell (IBOUND=1 or -1), you must provide either:
    - A specific head value using the 'head' option
    - The 'avhead' option to use average head of connected active cells

    If the cell was inactive in the previous stress period and no head
    option is provided, the simulation will abort with an error.

    Examples
    --------
    >>> import flopy
    >>> m = flopy.mfusg.MfUsg(modelname='test', model_ws='.', nper=10)
    >>> # Simulate lake filling: convert cells to constant head in SP 3
    >>> spd = {
    ...     3: {
    ...         'ibm1': [
    ...             {'node': 100, 'head': 2900.0},
    ...             {'node': 101, 'head': 2900.0},
    ...             {'node': 102, 'head': 2900.0},
    ...         ]
    ...     },
    ...     # Simulate lake draining: reactivate cells in SP 8
    ...     8: {
    ...         'ib1': [
    ...             {'node': 100, 'avhead': True},
    ...             {'node': 101, 'avhead': True},
    ...             {'node': 102, 'avhead': True},
    ...         ]
    ...     }
    ... }
    >>> tib = flopy.mfusg.MfUsgTib(m, stress_period_data=spd)

    """

    def __init__(
        self,
        model,
        stress_period_data=None,
        extension="tib",
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
            unitnumber = MfUsgTib._defaultunit()

        # Prepare filenames
        filenames = MfUsgTib._prepare_filenames(filenames)

        # Call parent constructor
        super().__init__(
            model,
            extension=extension,
            name=MfUsgTib._ftype(),
            unit_number=unitnumber,
            filenames=filenames,
        )

        self._generate_heading()

        # Store parameters
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
        Write the TIB package file.

        Returns
        -------
        None
        """
        # Open file
        f_tib = open(self.fn_path, "w")

        # Write header comment
        f_tib.write(f"# {self.heading}\n")

        # Get number of stress periods
        nper = self.parent.nper

        # Write data for each stress period
        for iper in range(nper):
            spd = self.stress_period_data.get(iper, {})

            # Get data for each IBOUND type
            ib0_data = spd.get("ib0", [])  # Deactivate (IBOUND=0)
            ib1_data = spd.get("ib1", [])  # Activate (IBOUND=1)
            ibm1_data = spd.get("ibm1", [])  # Constant head (IBOUND=-1)

            # For transport (ICBUND)
            icb0_data = spd.get("icb0", [])
            icb1_data = spd.get("icb1", [])
            icbm1_data = spd.get("icbm1", [])

            # Line 1: NIB0 NIB1 NIBM1 NICB0 NICB1 NICBM1
            f_tib.write(
                f"{len(ib0_data):10d}"
                f"{len(ib1_data):10d}"
                f"{len(ibm1_data):10d}"
                f"{len(icb0_data):10d}"
                f"{len(icb1_data):10d}"
                f"{len(icbm1_data):10d}\n"
            )

            # Write IB0 data (nodes to deactivate) - as array
            if len(ib0_data) > 0:
                # Write nodes, 10 per line
                for i, node in enumerate(ib0_data):
                    f_tib.write(f"{node:10d}")
                    if (i + 1) % 10 == 0:
                        f_tib.write("\n")
                if len(ib0_data) % 10 != 0:
                    f_tib.write("\n")

            # Write IB1 data (nodes to activate)
            for item in ib1_data:
                node = item["node"]
                line = f"{node:10d}"
                if "head" in item:
                    line += f" HEAD {item['head']:15.6E}"
                elif item.get("avhead", False):
                    line += " AVHEAD"
                f_tib.write(line + "\n")

            # Write IBM1 data (nodes to set as constant head)
            for item in ibm1_data:
                node = item["node"]
                line = f"{node:10d}"
                if "head" in item:
                    line += f" HEAD {item['head']:15.6E}"
                elif item.get("avhead", False):
                    line += " AVHEAD"
                else:
                    raise ValueError(
                        f"Node {node} in IBM1 requires either 'head' value "
                        "or 'avhead': True option"
                    )
                f_tib.write(line + "\n")

            # Write ICB0 data (transport - deactivate ICBUND)
            if len(icb0_data) > 0:
                for i, node in enumerate(icb0_data):
                    f_tib.write(f"{node:10d}")
                    if (i + 1) % 10 == 0:
                        f_tib.write("\n")
                if len(icb0_data) % 10 != 0:
                    f_tib.write("\n")

            # Write ICB1 data (transport - activate ICBUND)
            for item in icb1_data:
                node = item["node"]
                line = f"{node:10d}"
                if "conc" in item:
                    # conc can be a single value or list for multiple species
                    conc = item["conc"]
                    if isinstance(conc, (list, tuple)):
                        conc_str = " ".join(f"{c:15.6E}" for c in conc)
                    else:
                        conc_str = f"{conc:15.6E}"
                    line += f" CONC {conc_str}"
                elif item.get("avconc", False):
                    line += " AVCONC"
                f_tib.write(line + "\n")

            # Write ICBM1 data (transport - prescribed concentration)
            for item in icbm1_data:
                node = item["node"]
                line = f"{node:10d}"
                if "conc" in item:
                    conc = item["conc"]
                    if isinstance(conc, (list, tuple)):
                        conc_str = " ".join(f"{c:15.6E}" for c in conc)
                    else:
                        conc_str = f"{conc:15.6E}"
                    line += f" CONC {conc_str}"
                elif item.get("avconc", False):
                    line += " AVCONC"
                else:
                    raise ValueError(
                        f"Node {node} in ICBM1 requires either 'conc' value "
                        "or 'avconc': True option"
                    )
                f_tib.write(line + "\n")

        f_tib.close()

    @staticmethod
    def _ftype():
        return "TIB"

    @staticmethod
    def _defaultunit():
        return 72

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing TIB package from a file.

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
        tib : MfUsgTib
            MfUsgTib object

        Examples
        --------
        >>> tib = MfUsgTib.load('model.tib', model)

        """
        raise NotImplementedError(
            "TIB load method not yet implemented. "
            "Contributions welcome at https://github.com/modflowpy/flopy"
        )
