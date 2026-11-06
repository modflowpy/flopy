"""
MfUsgTib module.

Contains the MfUsgTib class for the Transient IBOUND (TIB) Package.
The TIB package allows changing IBOUND and ICBUND values at any stress
period in MODFLOW-USG. Use cases include excavation/reclamation,
well drilling/plugging, and transient prescribed head/concentration cells.

Note that the user can access the MfUsgTib class as `flopy.mfusg.MfUsgTib`.
"""

from os import PathLike

import numpy as np

from ..pakbase import Package
from ..utils import Util2d
from ..utils.utils_def import get_open_file_object
from .mfusg import MfUsg

# Keys for the six data categories, in file order
_IB_KEYS = ("ib0", "ib1", "ibm1", "icb0", "icb1", "icbm1")

# Which categories use U2DINT (node arrays) vs line-by-line records
_ARRAY_KEYS = {"ib0", "icb0"}


class MfUsgTib(Package):
    """Transient IBOUND (TIB) Package Class for MODFLOW-USG.

    The TIB package allows IBOUND and ICBUND arrays to be modified at any
    stress period during a simulation.

    Parameters
    ----------
    model : MfUsg
        The model object to which this package will be added.
    stress_period_data : dict of dict, optional
        Dictionary keyed by zero-based stress period number. Each value is
        a dict with up to six keys:

        - ``"ib0"``: array-like of int — global node numbers set to IBOUND=0
        - ``"ib1"``: list of tuples — nodes set to IBOUND=1,
          each ``(node,)`` or ``(node, "HEAD", value)`` or ``(node, "AVHEAD")``
        - ``"ibm1"``: list of tuples — nodes set to IBOUND=-1,
          same tuple forms as ib1
        - ``"icb0"``: array-like of int — global node numbers set to ICBUND=0
        - ``"icb1"``: list of tuples — nodes set to ICBUND=1,
          each ``(node,)`` or ``(node, "CONC", [c1,...])`` or ``(node, "AVCONC")``
        - ``"icbm1"``: list of tuples — nodes set to ICBUND=-1,
          same tuple forms as icb1

        All node numbers are global (0-based in Python, converted to 1-based
        on write), as required by the TIB specification. For structured grids,
        use ``model.dis.get_node()`` to convert (layer, row, col) tuples to
        global node numbers. Missing keys default to 0 entries. Missing stress
        periods mean no TIB changes for that period.
    extension : str, default "tib"
        File extension.
    unitnumber : int, optional
        File unit number.
    filenames : str or list of str, optional
        Filenames for the package.
    add_package : bool, default True
        Whether to add the package to the parent model.

    Examples
    --------
    >>> import flopy
    >>> m = flopy.mfusg.MfUsg()
    >>> dis = flopy.mfusg.MfUsgDis(m, nlay=1, nrow=10, ncol=10, nper=3)
    >>> # For structured grids, convert (lay, row, col) to global nodes:
    >>> nodes = dis.get_node([(0, 2, 3), (0, 5, 1)])
    >>> spd = {
    ...     0: {"ib0": nodes, "ib1": [(20, "HEAD", 10.0)]},
    ...     2: {"ibm1": [(30, "AVHEAD")]},
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
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if unitnumber is None:
            unitnumber = MfUsgTib._defaultunit()

        filenames = self._prepare_filenames(filenames)

        super().__init__(
            model,
            extension,
            self._ftype(),
            unitnumber,
            filenames,
        )
        self._generate_heading()

        if stress_period_data is None:
            stress_period_data = {}

        self._validate_stress_period_data(stress_period_data)
        self.stress_period_data = stress_period_data

        if add_package:
            self.parent.add_package(self)

    @staticmethod
    def _validate_stress_period_data(spd):
        """Check that stress_period_data has valid structure."""
        if not isinstance(spd, dict):
            raise ValueError(
                "stress_period_data must be a dict keyed by stress period."
            )
        for kper, data in spd.items():
            if not isinstance(kper, (int, np.integer)):
                raise ValueError(f"Stress period key must be int, got {type(kper)}.")
            if not isinstance(data, dict):
                raise ValueError(
                    f"Data for stress period {kper} must be a dict, got {type(data)}."
                )
            for key in data:
                if key not in _IB_KEYS:
                    raise ValueError(
                        f"Unknown key '{key}' in stress period {kper}. "
                        f"Valid keys: {_IB_KEYS}"
                    )

    @staticmethod
    def _count_for_key(data, key):
        """Return the number of entries for a given key in one stress period's data."""
        val = data.get(key)
        if val is None:
            return 0
        if isinstance(val, np.ndarray):
            return val.size
        return len(val)

    def write_file(self, f=None):
        """Write the TIB package file.

        Parameters
        ----------
        f : str or file handle, optional
            File to write to. If None, uses the default package path.
        """
        opened_here = f is None or isinstance(f, (str, PathLike))
        if f is None:
            f_obj = open(self.fn_path, "w")
        elif isinstance(f, (str, PathLike)):
            f_obj = open(f, "w")
        else:
            f_obj = f

        try:
            f_obj.write(f"{self.heading}\n")

            nper = self.parent.nper

            for kper in range(nper):
                data = self.stress_period_data.get(kper, {})

                counts = [self._count_for_key(data, key) for key in _IB_KEYS]
                count_line = " ".join(f"{c:9d}" for c in counts)
                f_obj.write(f"{count_line}   Stress period {kper + 1}\n")

                # Write each category
                for key, count in zip(_IB_KEYS, counts):
                    if count == 0:
                        continue
                    vals = data[key]
                    if key in _ARRAY_KEYS:
                        self._write_node_array(f_obj, vals, key)
                    else:
                        self._write_list_records(f_obj, vals, key)
        finally:
            if opened_here:
                f_obj.close()

    def _write_node_array(self, f_obj, vals, name):
        """Write a node array (ib0 or icb0) as U2DINT via Util2d."""
        arr = np.asarray(vals, dtype=np.int32)
        # Convert from 0-based to 1-based node numbers for the file
        arr_1based = arr + 1
        u2d = Util2d(
            self.parent,
            (arr_1based.size,),
            np.int32,
            arr_1based,
            name=name,
        )
        f_obj.write(u2d.get_file_entry())

    @staticmethod
    def _write_list_records(f_obj, records, key):
        """Write line-by-line records for ib1/ibm1/icb1/icbm1."""
        is_icb = key.startswith("icb")
        for rec in records:
            if not isinstance(rec, (list, tuple)):
                rec = (rec,)
            # Node number: convert 0-based → 1-based
            node_1based = int(rec[0]) + 1
            if len(rec) == 1:
                f_obj.write(f" {node_1based:9d}\n")
            elif len(rec) >= 2:
                keyword = str(rec[1]).upper()
                if is_icb and keyword == "CONC":
                    if len(rec) < 3:
                        raise ValueError(
                            f"'CONC' record for key '{key}' requires "
                            f"concentration values: {rec}"
                        )
                    conc_vals = rec[2]
                    if isinstance(conc_vals, (int, float)):
                        conc_vals = [conc_vals]
                    conc_str = " ".join(f"{c:14.6g}" for c in conc_vals)
                    f_obj.write(f" {node_1based:9d} CONC {conc_str}\n")
                elif not is_icb and keyword == "HEAD":
                    if len(rec) < 3:
                        raise ValueError(
                            f"'HEAD' record for key '{key}' requires "
                            f"a head value: {rec}"
                        )
                    hvalue = float(rec[2])
                    f_obj.write(f" {node_1based:9d} HEAD {hvalue:14.6g}\n")
                elif keyword in ("AVHEAD", "AVCONC"):
                    f_obj.write(f" {node_1based:9d} {keyword}\n")
                else:
                    raise ValueError(f"Unknown keyword '{keyword}' for key '{key}'.")

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None):
        """Load a TIB package from a file.

        Parameters
        ----------
        f : str or file handle
            File to load.
        model : MfUsg
            The model object.
        nper : int, optional
            Number of stress periods. If None, read from model.
        ext_unit_dict : dict, optional
            External unit dictionary.

        Returns
        -------
        MfUsgTib
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading tib package file...")

        opened_here = not (hasattr(f, "read") or hasattr(f, "write"))
        f_obj = get_open_file_object(f, "r")

        if nper is None:
            nper = model.nper

        try:
            # Skip comment/heading lines
            line = f_obj.readline()
            while line.startswith("#"):
                line = f_obj.readline()

            # The first non-comment line is the counts for stress period 1
            # (we already read it, so process it)
            stress_period_data = {}

            for kper in range(nper):
                if kper > 0:
                    line = f_obj.readline()

                t = line.split()
                # First 6 integers are counts for the 6 categories
                counts = {key: int(t[i]) for i, key in enumerate(_IB_KEYS)}

                data = {}
                for key in _IB_KEYS:
                    n = counts[key]
                    if n == 0:
                        continue
                    if key in _ARRAY_KEYS:
                        data[key] = cls._load_node_array(
                            f_obj, model, n, key, ext_unit_dict
                        )
                    else:
                        is_icb = key.startswith("icb")
                        data[key] = cls._load_list_records(f_obj, n, is_icb)

                if data:
                    stress_period_data[kper] = data
        finally:
            if opened_here:
                f_obj.close()

        return cls(
            model,
            stress_period_data=stress_period_data,
            add_package=True,
        )

    @staticmethod
    def _load_node_array(f_obj, model, n, name, ext_unit_dict):
        """Load a U2DINT node array and convert from 1-based to 0-based."""
        u2d = Util2d.load(f_obj, model, (n,), np.int32, name, ext_unit_dict)
        # Convert from 1-based (file) to 0-based (Python)
        return u2d.array - 1

    @staticmethod
    def _load_list_records(f_obj, n, is_icb):
        """Load line-by-line records for ib1/ibm1/icb1/icbm1."""
        records = []
        for _ in range(n):
            line = f_obj.readline().strip()
            parts = line.split()
            # Node number: convert 1-based → 0-based
            node = int(parts[0]) - 1
            if len(parts) == 1:
                records.append((node,))
            else:
                keyword = parts[1].upper()
                if is_icb and keyword == "CONC":
                    conc_vals = [float(x) for x in parts[2:]]
                    records.append((node, "CONC", conc_vals))
                elif not is_icb and keyword == "HEAD":
                    hvalue = float(parts[2])
                    records.append((node, "HEAD", hvalue))
                elif keyword in ("AVHEAD", "AVCONC"):
                    records.append((node, keyword))
                else:
                    raise ValueError(
                        f"Unknown keyword '{keyword}' in TIB record: {line}"
                    )
        return records

    @staticmethod
    def get_empty(nib0=0, nib1=0, nibm1=0, nicb0=0, nicb1=0, nicbm1=0):
        """Return a template stress-period dict with empty arrays/lists.

        Parameters
        ----------
        nib0 : int
            Number of nodes for IBOUND=0 category.
        nib1 : int
            Number of entries for IBOUND=1 category.
        nibm1 : int
            Number of entries for IBOUND=-1 category.
        nicb0 : int
            Number of nodes for ICBUND=0 category.
        nicb1 : int
            Number of entries for ICBUND=1 category.
        nicbm1 : int
            Number of entries for ICBUND=-1 category.

        Returns
        -------
        dict
            Template dict with keys matching the TIB data categories.
        """
        data = {}
        if nib0 > 0:
            data["ib0"] = np.zeros(nib0, dtype=np.int32)
        if nib1 > 0:
            data["ib1"] = [(0,)] * nib1
        if nibm1 > 0:
            data["ibm1"] = [(0,)] * nibm1
        if nicb0 > 0:
            data["icb0"] = np.zeros(nicb0, dtype=np.int32)
        if nicb1 > 0:
            data["icb1"] = [(0,)] * nicb1
        if nicbm1 > 0:
            data["icbm1"] = [(0,)] * nicbm1
        return data

    @staticmethod
    def _ftype():
        return "TIB"

    @staticmethod
    def _defaultunit():
        return 160
