import os
import copy
import numpy as np
from itertools import groupby
from .utils_def import totim_to_datetime


class ZoneBudget:
    """
    ZoneBudget class

    Parameters
    ----------
    cbc_file : str or CellBudgetFile object
        The file name or CellBudgetFile object for which budgets will be
        computed.
    z : ndarray
        The array containing to zones to be used.
    kstpkper : tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    totim : float
        The simulation time.
    aliases : dict
        A dictionary with key, value pairs of zones and aliases. Replaces
        the corresponding record and field names with the aliases provided.
        When using this option in conjunction with a list of zones, the
        zone(s) passed may either be all strings (aliases), all integers,
        or mixed.

    Returns
    -------
    None

    Examples
    --------

    >>> from flopy.utils.zonbud import ZoneBudget
    >>> zon = ZoneBudget.read_zone_file('zone_input_file')
    >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
    >>> zb.to_csv('zonebudtest.csv')
    >>> zb_mgd = zb * 7.48052 / 1000000
    """

    def __init__(
        self,
        cbc_file,
        z,
        kstpkper=None,
        totim=None,
        aliases=None,
        verbose=False,
        **kwargs,
    ):
        from .binaryfile import CellBudgetFile

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)
        else:
            raise Exception(f"Cannot load cell budget file: {cbc_file}.")

        if isinstance(z, np.ndarray):
            assert np.issubdtype(
                z.dtype, np.integer
            ), "Zones dtype must be integer"
        else:
            e = (
                "Please pass zones as a numpy ndarray of (positive)"
                " integers. {}".format(z.dtype)
            )
            raise Exception(e)

        # Check for negative zone values
        if np.any(z < 0):
            raise Exception(
                "Negative zone value(s) found:", np.unique(z[z < 0])
            )

        self.dis = None
        if "model" in kwargs.keys():
            self.model = kwargs.pop("model")
            self.dis = self.model.dis
        if "dis" in kwargs.keys():
            self.dis = kwargs.pop("dis")
        if len(kwargs.keys()) > 0:
            args = ",".join(kwargs.keys())
            raise Exception(f"LayerFile error: unrecognized kwargs: {args}")

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.cbc.get_data(idx=0, full3D=True)[0].shape
        self.nlay, self.nrow, self.ncol = self.cbc_shape
        self.cbc_times = self.cbc.get_times()
        self.cbc_kstpkper = self.cbc.get_kstpkper()
        self.kstpkper = None
        self.totim = None

        if kstpkper is not None:
            if isinstance(kstpkper, tuple):
                kstpkper = [kstpkper]
            for kk in kstpkper:
                s = f"The specified time step/stress period does not exist {kk}"
                assert kk in self.cbc.get_kstpkper(), s
            self.kstpkper = kstpkper
        elif totim is not None:
            if isinstance(totim, float):
                totim = [totim]
            elif isinstance(totim, int):
                totim = [float(totim)]
            for t in totim:
                s = f"The specified simulation time does not exist {t}"
                assert t in self.cbc.get_times(), s
            self.totim = totim
        else:
            # No time step/stress period or simulation time pass
            self.kstpkper = self.cbc.get_kstpkper()

        # Set float and integer types
        self.float_type = np.float32
        self.int_type = np.int32

        # Check dimensions of input zone array
        s = (
            "Row/col dimensions of zone array {}"
            " do not match model row/col dimensions {}".format(
                z.shape, self.cbc_shape
            )
        )
        assert z.shape[-2] == self.nrow and z.shape[-1] == self.ncol, s

        if z.shape == self.cbc_shape:
            izone = z.copy()
        elif len(z.shape) == 2:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[:, :]
        elif len(z.shape) == 3 and z.shape[0] == 1:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[0, :, :]
        else:
            e = f"Shape of the zone array is not recognized: {z.shape}"
            raise Exception(e)

        self.izone = izone
        self.allzones = np.unique(izone)
        self._zonenamedict = {z: f"ZONE_{z}" for z in self.allzones}

        if aliases is not None:
            s = (
                "Input aliases not recognized. Please pass a dictionary "
                "with key,value pairs of zone/alias."
            )
            assert isinstance(aliases, dict), s
            # Replace the relevant field names (ignore zone 0)
            seen = []
            for z, a in iter(aliases.items()):
                if z != 0 and z in self._zonenamedict.keys():
                    if z in seen:
                        raise Exception(
                            "Zones may not have more than 1 alias."
                        )
                    self._zonenamedict[z] = "_".join(a.split())
                    seen.append(z)

        # self._iflow_recnames = self._get_internal_flow_record_names()

        # All record names in the cell-by-cell budget binary file
        self.record_names = [
            n.strip() for n in self.cbc.get_unique_record_names(decode=True)
        ]

        # Get imeth for each record in the CellBudgetFile record list
        self.imeth = {}
        for record in self.cbc.recordarray:
            self.imeth[record["text"].strip().decode("utf-8")] = record[
                "imeth"
            ]

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS
        # ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO--- terms are used by the SWI2 groundwater flow process.
        internal_flow_terms = [
            "CONSTANT HEAD",
            "FLOW RIGHT FACE",
            "FLOW FRONT FACE",
            "FLOW LOWER FACE",
            "SWIADDTOCH",
            "SWIADDTOFRF",
            "SWIADDTOFFF",
            "SWIADDTOFLF",
        ]

        # Source/sink/storage term record names
        # These are all of the terms that are not related to constant
        # head cells or face flow terms
        self.ssst_record_names = [
            n for n in self.record_names if n not in internal_flow_terms
        ]

        # Initialize budget recordarray
        array_list = []
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                recordarray = self._initialize_budget_recordarray(
                    kstpkper=kk, totim=None
                )
                array_list.append(recordarray)
        elif self.totim is not None:
            for t in self.totim:
                recordarray = self._initialize_budget_recordarray(
                    kstpkper=None, totim=t
                )
                array_list.append(recordarray)
        self._budget = np.concatenate(array_list, axis=0)

        # Update budget record array
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                if verbose:
                    s = (
                        "Computing the budget for"
                        " time step {} in stress period {}".format(
                            kk[0] + 1, kk[1] + 1
                        )
                    )
                    print(s)
                self._compute_budget(kstpkper=kk)
        elif self.totim is not None:
            for t in self.totim:
                if verbose:
                    s = f"Computing the budget for time {t}"
                    print(s)
                self._compute_budget(totim=t)

    def _compute_budget(self, kstpkper=None, totim=None):
        """
        Creates a budget for the specified zone array. This function only
        supports the use of a single time step/stress period or time.

        Parameters
        ----------
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------
        None

        """
        # Initialize an array to track where the constant head cells
        # are located.
        ich = np.zeros(self.cbc_shape, self.int_type)
        swiich = np.zeros(self.cbc_shape, self.int_type)

        if "CONSTANT HEAD" in self.record_names:
            """
            C-----CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
            C-----CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
            C-----INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
            C-----HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
            C-----FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
            """
            chd = self.cbc.get_data(
                text="CONSTANT HEAD",
                full3D=True,
                kstpkper=kstpkper,
                totim=totim,
            )[0]
            ich[np.ma.where(chd != 0.0)] = 1
        if "FLOW RIGHT FACE" in self.record_names:
            self._accumulate_flow_frf("FLOW RIGHT FACE", ich, kstpkper, totim)
        if "FLOW FRONT FACE" in self.record_names:
            self._accumulate_flow_fff("FLOW FRONT FACE", ich, kstpkper, totim)
        if "FLOW LOWER FACE" in self.record_names:
            self._accumulate_flow_flf("FLOW LOWER FACE", ich, kstpkper, totim)
        if "SWIADDTOCH" in self.record_names:
            swichd = self.cbc.get_data(
                text="SWIADDTOCH", full3D=True, kstpkper=kstpkper, totim=totim
            )[0]
            swiich[swichd != 0] = 1
        if "SWIADDTOFRF" in self.record_names:
            self._accumulate_flow_frf("SWIADDTOFRF", swiich, kstpkper, totim)
        if "SWIADDTOFFF" in self.record_names:
            self._accumulate_flow_fff("SWIADDTOFFF", swiich, kstpkper, totim)
        if "SWIADDTOFLF" in self.record_names:
            self._accumulate_flow_flf("SWIADDTOFLF", swiich, kstpkper, totim)

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        # iterate over remaining items in the list
        for recname in self.ssst_record_names:
            self._accumulate_flow_ssst(recname, kstpkper, totim)

        # Compute mass balance terms
        self._compute_mass_balance(kstpkper, totim)

        return

    def _add_empty_record(
        self, recordarray, recname, kstpkper=None, totim=None
    ):
        """
        Build an empty records based on the specified flow direction and
        record name for the given list of zones.

        Parameters
        ----------
        recordarray :
        recname :
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------
        recordarray : np.recarray

        """
        if kstpkper is not None:
            if len(self.cbc_times) > 0:
                totim = self.cbc_times[self.cbc_kstpkper.index(kstpkper)]
            else:
                totim = 0.0
        elif totim is not None:
            if len(self.cbc_times) > 0:
                kstpkper = self.cbc_kstpkper[self.cbc_times.index(totim)]
            else:
                kstpkper = (0, 0)

        row = [totim, kstpkper[0], kstpkper[1], recname]
        row += [0.0 for _ in self._zonenamedict.values()]
        recs = np.array(tuple(row), dtype=recordarray.dtype)
        recordarray = np.append(recordarray, recs)
        return recordarray

    def _initialize_budget_recordarray(self, kstpkper=None, totim=None):
        """
        Initialize the budget record array which will store all of the
        fluxes in the cell-budget file.

        Parameters
        ----------
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------

        """

        # Create empty array for the budget terms.
        dtype_list = [
            ("totim", "<f4"),
            ("time_step", "<i4"),
            ("stress_period", "<i4"),
            ("name", (str, 50)),
        ]
        dtype_list += [
            (n, self.float_type) for n in self._zonenamedict.values()
        ]
        dtype = np.dtype(dtype_list)
        recordarray = np.array([], dtype=dtype)

        # Add "from" records
        if "STORAGE" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "FROM_STORAGE", kstpkper, totim
            )
        if "CONSTANT HEAD" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "FROM_CONSTANT_HEAD", kstpkper, totim
            )
        for recname in self.ssst_record_names:
            if recname != "STORAGE":
                recordarray = self._add_empty_record(
                    recordarray,
                    "FROM_" + "_".join(recname.split()),
                    kstpkper,
                    totim,
                )

        for z, n in self._zonenamedict.items():
            if z == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(
                    recordarray, "FROM_" + "_".join(n.split()), kstpkper, totim
                )
        recordarray = self._add_empty_record(
            recordarray, "TOTAL_IN", kstpkper, totim
        )

        # Add "out" records
        if "STORAGE" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "TO_STORAGE", kstpkper, totim
            )
        if "CONSTANT HEAD" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "TO_CONSTANT_HEAD", kstpkper, totim
            )
        for recname in self.ssst_record_names:
            if recname != "STORAGE":
                recordarray = self._add_empty_record(
                    recordarray,
                    "TO_" + "_".join(recname.split()),
                    kstpkper,
                    totim,
                )

        for z, n in self._zonenamedict.items():
            if z == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(
                    recordarray, "TO_" + "_".join(n.split()), kstpkper, totim
                )
        recordarray = self._add_empty_record(
            recordarray, "TOTAL_OUT", kstpkper, totim
        )

        recordarray = self._add_empty_record(
            recordarray, "IN-OUT", kstpkper, totim
        )
        recordarray = self._add_empty_record(
            recordarray, "PERCENT_DISCREPANCY", kstpkper, totim
        )
        return recordarray

    @staticmethod
    def _filter_circular_flow(fz, tz, f):
        """

        Parameters
        ----------
        fz
        tz
        f

        Returns
        -------

        """
        e = np.equal(fz, tz)
        fz = fz[np.logical_not(e)]
        tz = tz[np.logical_not(e)]
        f = f[np.logical_not(e)]
        return fz, tz, f

    def _update_budget_fromfaceflow(
        self, fz, tz, f, kstpkper=None, totim=None
    ):
        """

        Parameters
        ----------
        fz
        tz
        f
        kstpkper
        totim

        Returns
        -------

        """

        # No circular flow within zones
        fz, tz, f = self._filter_circular_flow(fz, tz, f)

        if len(f) == 0:
            return

        # Inflows
        idx = tz != 0
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = ["FROM_" + self._zonenamedict[z] for z in fzi]
        colnames = [self._zonenamedict[z] for z in tzi]
        fluxes = f[idx]
        self._update_budget_recordarray(
            rownames, colnames, fluxes, kstpkper, totim
        )

        # Outflows
        idx = fz != 0
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = ["TO_" + self._zonenamedict[z] for z in tzi]
        colnames = [self._zonenamedict[z] for z in fzi]
        fluxes = f[idx]
        self._update_budget_recordarray(
            rownames, colnames, fluxes, kstpkper, totim
        )
        return

    def _update_budget_fromssst(self, fz, tz, f, kstpkper=None, totim=None):
        """

        Parameters
        ----------
        fz
        tz
        f
        kstpkper
        totim

        Returns
        -------

        """
        if len(f) == 0:
            return
        self._update_budget_recordarray(fz, tz, f, kstpkper, totim)
        return

    def _update_budget_recordarray(
        self, rownames, colnames, fluxes, kstpkper=None, totim=None
    ):
        """
        Update the budget record array with the flux for the specified
        flow direction (in/out), record name, and column.

        Parameters
        ----------
        rownames
        colnames
        fluxes
        kstpkper
        totim

        Returns
        -------
        None

        """
        try:

            if kstpkper is not None:
                for rn, cn, flux in zip(rownames, colnames, fluxes):
                    rowidx = np.where(
                        (self._budget["time_step"] == kstpkper[0])
                        & (self._budget["stress_period"] == kstpkper[1])
                        & (self._budget["name"] == rn)
                    )
                    self._budget[cn][rowidx] += flux
            elif totim is not None:
                for rn, cn, flux in zip(rownames, colnames, fluxes):
                    rowidx = np.where(
                        (self._budget["totim"] == totim)
                        & (self._budget["name"] == rn)
                    )
                    self._budget[cn][rowidx] += flux

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_frf(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.ncol >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
                # COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
                # ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
                # 1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K

                k, i, j = np.where(
                    self.izone[:, :, 1:] > self.izone[:, :, :-1]
                )

                # Adjust column values to account for the starting position of "nz"
                j += 1

                # Define the zone to which flow is going
                nz = self.izone[k, i, j]

                # Define the zone from which flow is coming
                jl = j - 1
                nzl = self.izone[k, i, jl]

                # Get the face flow
                q = data[k, i, jl]

                # Get indices where flow face values are positive (flow out of higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzl[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # FLOW BETWEEN NODE J,I,K AND J+1,I,K
                k, i, j = np.where(
                    self.izone[:, :, :-1] > self.izone[:, :, 1:]
                )

                # Define the zone from which flow is coming
                nz = self.izone[k, i, j]

                # Define the zone to which flow is going
                jr = j + 1
                nzr = self.izone[k, i, jr]

                # Get the face flow
                q = data[k, i, j]

                # Get indices where flow face values are positive (flow out of higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzr[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[j > 0], i[j > 0], j[j > 0]
                jl = j - 1
                nzl = self.izone[k, i, jl]
                nz = self.izone[k, i, j]
                q = data[k, i, jl]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi[tzi != 0]]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[j < self.ncol - 1],
                    i[j < self.ncol - 1],
                    j[j < self.ncol - 1],
                )
                nz = self.izone[k, i, j]
                jr = j + 1
                nzr = self.izone[k, i, jr]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_fff(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.nrow >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW FRONT FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
                k, i, j = np.where(
                    self.izone[:, 1:, :] < self.izone[:, :-1, :]
                )
                i += 1
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nza[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
                k, i, j = np.where(
                    self.izone[:, :-1, :] < self.izone[:, 1:, :]
                )
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzb[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[i > 0], i[i > 0], j[i > 0]
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[i < self.nrow - 1],
                    i[i < self.nrow - 1],
                    j[i < self.nrow - 1],
                )
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_flf(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.nlay >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW LOWER FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
                k, i, j = np.where(
                    self.izone[1:, :, :] < self.izone[:-1, :, :]
                )
                k += 1
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nza[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
                k, i, j = np.where(
                    self.izone[:-1, :, :] < self.izone[1:, :, :]
                )
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzb[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[k > 0], i[k > 0], j[k > 0]
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[k < self.nlay - 1],
                    i[k < self.nlay - 1],
                    j[k < self.nlay - 1],
                )
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_ssst(self, recname, kstpkper, totim):

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE

        imeth = self.imeth[recname]

        data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)
        if len(data) == 0:
            # Empty data, can occur during the first time step of a transient
            # model when storage terms are zero and not in the cell-budget
            # file.
            return
        else:
            data = data[0]

        if imeth == 2 or imeth == 5:
            # LIST
            qin = np.ma.zeros(
                (self.nlay * self.nrow * self.ncol), self.float_type
            )
            qout = np.ma.zeros(
                (self.nlay * self.nrow * self.ncol), self.float_type
            )
            for [node, q] in zip(data["node"], data["q"]):
                idx = node - 1
                if q > 0:
                    qin.data[idx] += q
                elif q < 0:
                    qout.data[idx] += q
            qin = np.ma.reshape(qin, (self.nlay, self.nrow, self.ncol))
            qout = np.ma.reshape(qout, (self.nlay, self.nrow, self.ncol))
        elif imeth == 0 or imeth == 1:
            # FULL 3-D ARRAY
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            qin[data > 0] = data[data > 0]
            qout[data < 0] = data[data < 0]
        elif imeth == 3:
            # 1-LAYER ARRAY WITH LAYER INDICATOR ARRAY
            rlay, rdata = data[0], data[1]
            data = np.ma.zeros(self.cbc_shape, self.float_type)
            for (r, c), l in np.ndenumerate(rlay):
                data[l - 1, r, c] = rdata[r, c]
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            qin[data > 0] = data[data > 0]
            qout[data < 0] = data[data < 0]
        elif imeth == 4:
            # 1-LAYER ARRAY THAT DEFINES LAYER 1
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            r, c = np.where(data > 0)
            qin[0, r, c] = data[r, c]
            r, c = np.where(data < 0)
            qout[0, r, c] = data[r, c]
        else:
            # Should not happen
            raise Exception(
                f'Unrecognized "imeth" for {recname} record: {imeth}'
            )

        # Inflows
        fz = []
        tz = []
        f = []
        for z in self.allzones:
            if z != 0:
                flux = qin[(self.izone == z)].sum()
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.0
                fz.append("FROM_" + "_".join(recname.split()))
                tz.append(self._zonenamedict[z])
                f.append(flux)
        fz = np.array(fz)
        tz = np.array(tz)
        f = np.array(f)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        # Outflows
        fz = []
        tz = []
        f = []
        for z in self.allzones:
            if z != 0:
                flux = qout[(self.izone == z)].sum()
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.0
                fz.append("TO_" + "_".join(recname.split()))
                tz.append(self._zonenamedict[z])
                f.append(flux)
        fz = np.array(fz)
        tz = np.array(tz)
        f = np.array(f)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

    def _compute_mass_balance(self, kstpkper, totim):
        # Returns a record array with total inflow, total outflow,
        # and percent error summed by column.
        skipcols = ["time_step", "stress_period", "totim", "name"]

        # Compute inflows
        recnames = self.get_record_names()
        innames = [n for n in recnames if n.startswith("FROM_")]
        outnames = [n for n in recnames if n.startswith("TO_")]
        if kstpkper is not None:
            rowidx = np.where(
                (self._budget["time_step"] == kstpkper[0])
                & (self._budget["stress_period"] == kstpkper[1])
                & np.in1d(self._budget["name"], innames)
            )
        elif totim is not None:
            rowidx = np.where(
                (self._budget["totim"] == totim)
                & np.in1d(self._budget["name"], innames)
            )
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx]
        )
        intot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["TOTAL_IN"] * len(tz))
        self._update_budget_fromssst(fz, tz, intot, kstpkper, totim)

        # Compute outflows
        if kstpkper is not None:
            rowidx = np.where(
                (self._budget["time_step"] == kstpkper[0])
                & (self._budget["stress_period"] == kstpkper[1])
                & np.in1d(self._budget["name"], outnames)
            )
        elif totim is not None:
            rowidx = np.where(
                (self._budget["totim"] == totim)
                & np.in1d(self._budget["name"], outnames)
            )
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx]
        )
        outot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["TOTAL_OUT"] * len(tz))
        self._update_budget_fromssst(fz, tz, outot, kstpkper, totim)

        # Compute IN-OUT
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        f = intot - outot
        fz = np.array(["IN-OUT"] * len(tz))
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        # Compute percent discrepancy
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["PERCENT_DISCREPANCY"] * len(tz))
        in_minus_out = intot - outot
        in_plus_out = intot + outot
        f = 100 * in_minus_out / (in_plus_out / 2.0)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

    def get_model_shape(self):
        """Get model shape

        Returns
        -------
        nlay : int
            Number of layers
        nrow : int
            Number of rows
        ncol : int
            Number of columns

        """
        return self.nlay, self.nrow, self.ncol

    def get_record_names(self, stripped=False):
        """
        Get a list of water budget record names in the file.

        Returns
        -------
        out : list of strings
            List of unique text names in the binary file.

        Examples
        --------

        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> recnames = zb.get_record_names()

        """
        return _get_record_names(self._budget, stripped=stripped)

    def get_budget(self, names=None, zones=None, net=False, pivot=False):
        """
        Get a list of zonebudget record arrays.

        Parameters
        ----------

        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.
        pivot : boolean
            If True, returns data in a more user friendly format

        Returns
        -------
        budget_list : list of record arrays
            A list of the zonebudget record arrays.

        Examples
        --------

        >>> names = ['FROM_CONSTANT_HEAD', 'RIVER_LEAKAGE_OUT']
        >>> zones = ['ZONE_1', 'ZONE_2']
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> bud = zb.get_budget(names=names, zones=zones)

        """
        recarray = _get_budget(
            self._budget, self._zonenamedict, names=names, zones=zones, net=net
        )

        if pivot:
            recarray = _pivot_recarray(recarray)

        return recarray

    def get_volumetric_budget(
        self, modeltime, recarray=None, extrapolate_kper=False
    ):
        """
        Method to generate a volumetric budget table based on flux information

        Parameters
        ----------
        modeltime : flopy.discretization.ModelTime object
            ModelTime object for calculating volumes
        recarray : np.recarray
            optional, user can pass in a numpy recarray to calculate volumetric
            budget. recarray must be pivoted before passing to
            get_volumetric_budget
        extrapolate_kper : bool
            flag to determine if we fill in data gaps with other
            timestep information from the same stress period.
            if True, we assume that flux is constant throughout a stress period
            and the pandas dataframe returned contains a
            volumetric budget per stress period

            if False, calculates volumes from available flux data

        Returns
        -------
            pd.DataFrame

        """
        if recarray is None:
            recarray = self.get_budget(pivot=True)
        return _volumetric_flux(recarray, modeltime, extrapolate_kper)

    def to_csv(self, fname):
        """
        Saves the budget record arrays to a formatted
        comma-separated values file.

        Parameters
        ----------
        fname : str
            The name of the output comma-separated values file.

        Returns
        -------
        None

        """
        # Needs updating to handle the new budget list structure. Write out
        # budgets for all kstpkper if kstpkper is None or pass list of
        # kstpkper/totim to save particular budgets.
        with open(fname, "w") as f:
            # Write header
            f.write(",".join(self._budget.dtype.names) + "\n")
            # Write rows
            for rowidx in range(self._budget.shape[0]):
                s = (
                    ",".join([str(i) for i in list(self._budget[:][rowidx])])
                    + "\n"
                )
                f.write(s)
        return

    def get_dataframes(
        self,
        start_datetime=None,
        timeunit="D",
        index_key="totim",
        names=None,
        zones=None,
        net=False,
        pivot=False,
    ):
        """
        Get pandas dataframes.

        Parameters
        ----------

        start_datetime : str
            Datetime string indicating the time at which the simulation starts.
        timeunit : str
            String that indicates the time units used in the model.
        index_key : str
            Indicates the fields to be used (in addition to "record") in the
            resulting DataFrame multi-index.
        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.
        pivot : bool
            If True, returns dataframe in a more user friendly format

        Returns
        -------
        df : Pandas DataFrame
            Pandas DataFrame with the budget information.

        Examples
        --------
        >>> from flopy.utils.zonbud import ZoneBudget
        >>> zon = ZoneBudget.read_zone_file('zone_input_file')
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> df = zb.get_dataframes()

        """
        recarray = self.get_budget(names, zones, net, pivot=pivot)
        return _recarray_to_dataframe(
            recarray,
            self._zonenamedict,
            start_datetime=start_datetime,
            timeunit=timeunit,
            index_key=index_key,
            zones=zones,
            pivot=pivot,
        )

    @classmethod
    def _get_otype(cls, fname):
        """
        Method to automatically distinguish output type based on the
        zonebudget header

        Parameters
        ----------
        fname : str
            zonebudget output file name

        Returns
        -------
        otype : int

        """
        with open(fname) as foo:
            line = foo.readline()
            if "zonebudget version" in line.lower():
                otype = 0
            elif "time step" in line.lower():
                otype = 1
            elif "totim" in line.lower():
                otype = 2
            else:
                raise AssertionError("Cant distinguish output type")
        return otype

    @classmethod
    def read_output(cls, fname, net=False, dataframe=False, **kwargs):
        """
        Method to read a zonebudget output file into a recarray or pandas
        dataframe

        Parameters
        ----------
        fname : str
            zonebudget output file name
        net : bool
            boolean flag for net budget
        dataframe : bool
            boolean flag to return a pandas dataframe

        **kwargs
            pivot : bool

            start_datetime : str
                Datetime string indicating the time at which the simulation
                starts. Can be used when pandas dataframe is requested
            timeunit : str
                String that indicates the time units used in the model.


        Returns
        -------
        np.recarray
        """
        otype = ZoneBudget._get_otype(fname)
        if otype == 0:
            recarray = _read_zb_zblst(fname)
        elif otype == 1:
            recarray = _read_zb_csv(fname)
        else:
            add_prefix = kwargs.pop("add_prefix", True)
            recarray = _read_zb_csv2(fname, add_prefix=add_prefix)

        zonenamdict = {
            int(i.split("_")[-1]): i
            for i in recarray.dtype.names
            if i.startswith("ZONE")
        }
        pivot = kwargs.pop("pivot", False)
        recarray = _get_budget(recarray, zonenamdict, net=net)
        if pivot:
            recarray = _pivot_recarray(recarray)

        if not dataframe:
            return recarray
        else:
            start_datetime = kwargs.pop("start_datetime", None)
            timeunit = kwargs.pop("timeunit", "D")
            return _recarray_to_dataframe(
                recarray,
                zonenamdict,
                start_datetime=start_datetime,
                timeunit=timeunit,
                pivot=pivot,
            )

    @classmethod
    def read_zone_file(cls, fname):
        """Method to read a zonebudget zone file into memory

        Parameters
        ----------
        fname : str
            zone file name

        Returns
        -------
        zones : np.array

        """
        with open(fname, "r") as f:
            lines = f.readlines()

        # Initialize layer
        lay = 0

        # Initialize data counter
        totlen = 0
        i = 0

        # First line contains array dimensions
        dimstring = lines.pop(0).strip().split()
        nlay, nrow, ncol = [int(v) for v in dimstring]
        zones = np.zeros((nlay, nrow, ncol), dtype=np.int32)

        # The number of values to read before placing
        # them into the zone array
        datalen = nrow * ncol

        # List of valid values for LOCAT
        locats = ["CONSTANT", "INTERNAL", "EXTERNAL"]

        # ITERATE OVER THE ROWS
        for line in lines:
            rowitems = line.strip().split()

            # Skip blank lines
            if len(rowitems) == 0:
                continue

            # HEADER
            if rowitems[0].upper() in locats:
                vals = []
                locat = rowitems[0].upper()

                if locat == "CONSTANT":
                    iconst = int(rowitems[1])
                else:
                    fmt = rowitems[1].strip("()")
                    fmtin, iprn = [int(v) for v in fmt.split("I")]

            # ZONE DATA
            else:
                if locat == "CONSTANT":
                    vals = np.ones((nrow, ncol), dtype=int) * iconst
                    lay += 1
                elif locat == "INTERNAL":
                    # READ ZONES
                    rowvals = [int(v) for v in rowitems]
                    s = "Too many values encountered on this line."
                    assert len(rowvals) <= fmtin, s
                    vals.extend(rowvals)

                elif locat == "EXTERNAL":
                    # READ EXTERNAL FILE
                    fname = rowitems[0]
                    if not os.path.isfile(fname):
                        errmsg = f'Could not find external file "{fname}"'
                        raise Exception(errmsg)
                    with open(fname, "r") as ext_f:
                        ext_flines = ext_f.readlines()
                    for ext_frow in ext_flines:
                        ext_frowitems = ext_frow.strip().split()
                        rowvals = [int(v) for v in ext_frowitems]
                        vals.extend(rowvals)
                    if len(vals) != datalen:
                        errmsg = (
                            "The number of values read from external "
                            'file "{}" does not match the expected '
                            "number.".format(len(vals))
                        )
                        raise Exception(errmsg)
                else:
                    # Should not get here
                    raise Exception(f"Locat not recognized: {locat}")

                    # IGNORE COMPOSITE ZONES

                if len(vals) == datalen:
                    # place values for the previous layer into the zone array
                    vals = np.array(vals, dtype=int).reshape((nrow, ncol))
                    zones[lay, :, :] = vals[:, :]
                    lay += 1
                totlen += len(rowitems)
            i += 1
        s = (
            "The number of values read ({:,.0f})"
            " does not match the number expected"
            " ({:,.0f})".format(totlen, nlay * nrow * ncol)
        )
        assert totlen == nlay * nrow * ncol, s
        return zones

    @classmethod
    def write_zone_file(cls, fname, array, fmtin=None, iprn=None):
        """
        Saves a numpy array in a format readable by the zonebudget program
        executable.

        File format:
        line 1: nlay, nrow, ncol
        line 2: INTERNAL (format)
        line 3: begin data
        .
        .
        .

        example from NACP:
        19 250 500
        INTERNAL      (10I7)
        199     199     199     199     199     199     199     199     199
        199     199     199     199     199     199     199     199     199
        ...
        INTERNAL      (10I7)
        199     199     199     199     199     199     199     199     199
        199     199     199     199     199     199     199     199     199
        ...

        Parameters
        ----------
        array : array
            The array of zones to be written.
        fname :  str
            The path and name of the file to be written.
        fmtin : int
            The number of values to write to each line.
        iprn : int
            Padding space to add between each value.

        Returns
        -------

        """
        if len(array.shape) == 2:
            b = np.zeros((1, array.shape[0], array.shape[1]), dtype=np.int32)
            b[0, :, :] = array[:, :]
            array = b.copy()
        elif len(array.shape) < 2 or len(array.shape) > 3:
            raise Exception(
                f"Shape of the input array is not recognized: {array.shape}"
            )
        if np.ma.is_masked(array):
            array = np.ma.filled(array, 0)

        nlay, nrow, ncol = array.shape

        if fmtin is not None:
            assert fmtin <= ncol, (
                "The specified width is greater than the "
                "number of columns in the array."
            )
        else:
            fmtin = ncol

        iprnmin = len(str(array.max()))
        if iprn is None or iprn <= iprnmin:
            iprn = iprnmin + 1

        formatter_str = f"{{:>{iprn}}}"
        formatter = formatter_str.format

        with open(fname, "w") as f:
            header = f"{nlay} {nrow} {ncol}\n"
            f.write(header)
            for lay in range(nlay):
                record_2 = f"INTERNAL\t({fmtin}I{iprn})\n"
                f.write(record_2)
                if fmtin < ncol:
                    for row in range(nrow):
                        rowvals = array[lay, row, :].ravel()
                        start = 0
                        end = start + fmtin
                        vals = rowvals[start:end]
                        while len(vals) > 0:
                            s = (
                                "".join([formatter(int(val)) for val in vals])
                                + "\n"
                            )
                            f.write(s)
                            start = end
                            end = start + fmtin
                            vals = rowvals[start:end]

                elif fmtin == ncol:
                    for row in range(nrow):
                        vals = array[lay, row, :].ravel()
                        f.write(
                            "".join([formatter(int(val)) for val in vals])
                            + "\n"
                        )

    def copy(self):
        """
        Return a deepcopy of the object.
        """
        return copy.deepcopy(self)

    def export(self, f, ml, **kwargs):
        """
        Method to export a netcdf file, or add zonebudget output to
        an open netcdf file instance

        Parameters
        ----------
        f : str or flopy.export.netcdf.NetCdf object
        ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf object
        **kwargs :
            logger : flopy.export.netcdf.Logger instance
            masked_vals : list
                list of values to mask

        Returns
        -------
            flopy.export.netcdf.NetCdf object

        """
        from flopy.export.utils import output_helper

        if isinstance(f, str):
            if not f.endswith(".nc"):
                raise AssertionError(
                    "File extension must end with .nc to "
                    "export a netcdf file"
                )

        zbncfobj = dataframe_to_netcdf_fmt(
            self.get_dataframes(pivot=True), self.izone, flux=True
        )
        oudic = {"zbud": zbncfobj}
        return output_helper(f, ml, oudic, **kwargs)

    def __deepcopy__(self, memo):
        """
        Over-rides the default deepcopy behavior. Copy all attributes except
        the CellBudgetFile object which does not copy nicely.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        ignore_attrs = ["cbc"]
        for k, v in self.__dict__.items():
            if k not in ignore_attrs:
                setattr(result, k, copy.deepcopy(v, memo))

        # Set CellBudgetFile object attribute manually. This is object
        # read-only so should not be problems with pointers from
        # multiple objects.
        result.cbc = self.cbc
        return result

    def __mul__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) * other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __truediv__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __div__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __add__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) + other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __sub__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) - other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj


class ZoneBudget6:
    """
    Model class for building, editing and running MODFLOW 6 zonebuget

    Parameters
    ----------
    name : str
        model name for zonebudget
    model_ws : str
        path to model
    exe_name : str
        excutable name
    extension : str
        name file extension
    """

    def __init__(
        self,
        name="zonebud",
        model_ws=".",
        exe_name="zbud6",
        extension=".zbnam",
    ):
        from ..mf6.utils import MfGrdFile
        from .binaryfile import CellBudgetFile

        self._name = name
        self._zon = None
        self._grb = None
        self._bud = None
        self._model_ws = model_ws
        self._exe_name = exe_name

        if not extension.startswith("."):
            extension = "." + extension

        self._extension = extension
        self.zbnam_packages = {
            "zon": ZoneFile6,
            "bud": CellBudgetFile,
            "grb": MfGrdFile,
        }
        self.package_dict = {}
        if self._zon is not None:
            self.package_dict["zon"] = self._zon
        if self._grb is not None:
            self.package_dict["grb"] = self._grb
        if self._bud is not None:
            self.package_dict["bud"] = self._bud

        self._recarray = None

    def run_model(self, exe_name=None, nam_file=None, silent=False):
        """
        Method to run a zonebudget model

        Parameters
        ----------
        exe_name : str
            optional zonebudget executable name
        nam_file : str
            optional zonebudget name file name
        silent : bool
            optional flag to silence output

        Returns
        -------
            tuple
        """
        from ..mbase import run_model

        if exe_name is None:
            exe_name = self._exe_name
        if nam_file is None:
            nam_file = os.path.join(self._name + self._extension)
        return run_model(
            exe_name, nam_file, model_ws=self._model_ws, silent=silent
        )

    def __setattr__(self, key, value):
        if key in ("zon", "bud", "grb", "cbc"):
            self.add_package(key, value)
            return
        elif key == "model_ws":
            raise AttributeError("please use change_model_ws() method")
        elif key == "name":
            self.change_model_name(value)
        super().__setattr__(key, value)

    def __getattr__(self, item):
        if item in ("zon", "bud", "grb", "name", "model_ws"):
            item = f"_{item}"
        return super().__getattribute__(item)

    def add_package(self, pkg_name, pkg):
        """
        Method to add a package to the ZoneBudget6 object

        Parameters
        ----------
        pkg_name : str
            three letter package abbreviation
        pkg : str or object
            either a package file name or package object

        """
        pkg_name = pkg_name.lower()
        if pkg_name not in self.zbnam_packages:
            if pkg_name == "cbc":
                pkg_name = "bud"
            else:
                raise KeyError(
                    f"{pkg_name} package is not valid for zonebudget"
                )

        if isinstance(pkg, str):
            if os.path.exists(os.path.join(self._model_ws, pkg)):
                pkg = os.path.join(self._model_ws, pkg)

            func = self.zbnam_packages[pkg_name]
            if pkg_name in ("bud", "grb"):
                pkg = func(pkg, precision="double")
            else:
                pkg = func.load(pkg, self)

        else:
            pass

        pkg_name = f"_{pkg_name}"
        self.__setattr__(pkg_name, pkg)
        if pkg is not None:
            self.package_dict[pkg_name[1:]] = pkg

    def change_model_ws(self, model_ws):
        """
        Method to change the model ws for writing a zonebudget
        model.

        Parameters
        ----------
        model_ws : str
            new model directory

        """
        self._model_ws = model_ws

    def change_model_name(self, name):
        """
        Method to change the model name for writing a zonebudget
        model.

        Parameters
        ----------
        name : str
            new model name

        """
        self._name = name
        if self._zon is not None:
            self._zon.filename = f"{name}.{self._zon.filename.split('.')[-1]}"

    def get_dataframes(
        self,
        start_datetime=None,
        timeunit="D",
        index_key="totim",
        names=None,
        zones=None,
        net=False,
        pivot=False,
    ):
        """
        Get pandas dataframes.

        Parameters
        ----------

        start_datetime : str
            Datetime string indicating the time at which the simulation starts.
        timeunit : str
            String that indicates the time units used in the model.
        index_key : str
            Indicates the fields to be used (in addition to "record") in the
            resulting DataFrame multi-index.
        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.
        pivot : bool
            If True, returns data in a more user friendly fashion

        Returns
        -------
        df : Pandas DataFrame
            Pandas DataFrame with the budget information.

        Examples
        --------
        >>> from flopy.utils.zonbud import ZoneBudget6
        >>> zb6 = ZoneBudget6.load("my_nam_file", model_ws="my_model_ws")
        >>> zb6.run_model()
        >>> df = zb6.get_dataframes()

        """
        recarray = self.get_budget(
            names=names, zones=zones, net=net, pivot=pivot
        )

        return _recarray_to_dataframe(
            recarray,
            self._zon._zonenamedict,
            start_datetime=start_datetime,
            timeunit=timeunit,
            index_key=index_key,
            zones=zones,
            pivot=pivot,
        )

    def get_budget(
        self, f=None, names=None, zones=None, net=False, pivot=False
    ):
        """
        Method to read and get zonebudget output

        Parameters
        ----------
        f : str
            zonebudget output file name
        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.
        pivot : bool
            Method to pivot recordarray into a more user friendly method
            for working with data

        Returns
        -------
            np.recarray
        """
        aliases = None
        if self._zon is not None:
            aliases = self._zon.aliases

        if f is None and self._recarray is None:
            f = os.path.join(self._model_ws, f"{self._name}.csv")
            self._recarray = _read_zb_csv2(
                f, add_prefix=False, aliases=aliases
            )
        elif f is None:
            pass
        else:
            self._recarray = _read_zb_csv2(
                f, add_prefix=False, aliases=aliases
            )

        recarray = _get_budget(
            self._recarray,
            self._zon._zonenamedict,
            names=names,
            zones=zones,
            net=net,
        )

        if pivot:
            recarray = _pivot_recarray(recarray)

        return recarray

    def get_volumetric_budget(
        self, modeltime, recarray=None, extrapolate_kper=False
    ):
        """
        Method to generate a volumetric budget table based on flux information

        Parameters
        ----------
        modeltime : flopy.discretization.ModelTime object
            ModelTime object for calculating volumes
        recarray : np.recarray
            optional, user can pass in a numpy recarray to calculate volumetric
            budget. recarray must be pivoted before passing to
            get_volumetric_budget
        extrapolate_kper : bool
            flag to determine if we fill in data gaps with other
            timestep information from the same stress period.
            if True, we assume that flux is constant throughout a stress period
            and the pandas dataframe returned contains a
            volumetric budget per stress period

            if False, calculates volumes from available flux data

        Returns
        -------
            pd.DataFrame

        """
        if recarray is None:
            recarray = self.get_budget(pivot=True)
        return _volumetric_flux(recarray, modeltime, extrapolate_kper)

    def write_input(self, line_length=20):
        """
        Method to write a ZoneBudget 6 model to file

        Parameters
        ----------
        line_length : int
            length of line for izone array

        """
        nam = []
        for pkg_nam, pkg in self.package_dict.items():
            if pkg_nam in ("grb", "bud"):
                path = os.path.relpath(pkg.filename, self._model_ws)
            else:
                path = pkg.filename
                pkg.write_input(line_length=line_length)
            nam.append(f"  {pkg_nam.upper()}   {path}\n")

        path = os.path.join(self._model_ws, self._name + self._extension)
        with open(path, "w") as foo:
            foo.write("BEGIN ZONEBUDGET\n")
            foo.writelines(nam)
            foo.write("END ZONEBUDGET\n")

    @staticmethod
    def load(nam_file, model_ws="."):
        """
        Method to load a zonebudget model from namefile

        Parameters
        ----------
        nam_file : str
            zonebudget name file
        model_ws : str
            model workspace path

        Returns
        -------
            ZoneBudget6 object
        """
        from ..utils.flopy_io import multi_line_strip

        name = nam_file.split(".")[0]
        zb6 = ZoneBudget6(name=name, model_ws=model_ws)
        with open(os.path.join(model_ws, nam_file)) as foo:
            line = multi_line_strip(foo)
            if "begin" in line:
                while True:
                    t = multi_line_strip(foo).split()
                    if t[0] == "end":
                        break
                    else:
                        zb6.add_package(t[0], t[1])

        return zb6

    def export(self, f, ml, **kwargs):
        """
        Method to export a netcdf file, or add zonebudget output to
        an open netcdf file instance

        Parameters
        ----------
        f : str or flopy.export.netcdf.NetCdf object
        ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf object
        **kwargs :
            logger : flopy.export.netcdf.Logger instance
            masked_vals : list
                list of values to mask

        Returns
        -------
            flopy.export.netcdf.NetCdf object

        """
        from flopy.export.utils import output_helper

        if isinstance(f, str):
            if not f.endswith(".nc"):
                raise AssertionError(
                    "File extension must end with .nc to "
                    "export a netcdf file"
                )

        zbncfobj = dataframe_to_netcdf_fmt(
            self.get_dataframes(pivot=True), self._zon.izone, flux=True
        )
        oudic = {"zbud": zbncfobj}
        return output_helper(f, ml, oudic, **kwargs)


class ZoneFile6:
    """
    Class to build, read, write and edit MODFLOW 6 zonebudget zone files

    Parameters
    ----------
    model : ZoneBudget6 object
        model object
    izone : np.array
        numpy array of zone numbers
    extension : str
        zone file extension name, defaults to ".zon"
    aliases : dict
        optional dictionary of zone aliases. ex. {1 : "nw_model"}
    """

    def __init__(self, model, izone, extension=".zon", aliases=None):
        self.izone = izone

        if not extension.startswith("."):
            extension = "." + extension

        self._extension = extension
        self._parent = model
        self._parent.add_package("zon", self)
        self.filename = self._parent.name + extension
        self.aliases = aliases
        self.allzones = [int(z) for z in np.unique(izone) if z != 0]
        self._zonenamedict = {z: f"ZONE_{z}" for z in self.allzones}

        if aliases is not None:
            if not isinstance(aliases, dict):
                raise TypeError("aliases parameter must be a dictionary")

            pop_list = []
            for zn, alias in aliases.items():
                if zn in self._zonenamedict:
                    self._zonenamedict[zn] = "_".join(alias.split())
                    self.aliases[zn] = "_".join(alias.split())
                else:
                    pop_list.append(zn)
                    print(f"warning: zone number {zn} not found")

            for p in pop_list:
                aliases.pop(p)

    @property
    def ncells(self):
        """
        Method to get number of model cells

        """
        return self.izone.size

    def write_input(self, f=None, line_length=20):
        """
        Method to write the zonebudget 6 file

        Parameters
        ----------
        f : str
            zone file name
        line_length : int
            maximum length of line to write in izone array
        """
        if f is None:
            f = os.path.join(self._parent.model_ws, self.filename)

        with open(f, "w") as foo:
            bfmt = ["  {:d}"]
            foo.write(
                f"BEGIN DIMENSIONS\n    NCELLS  {self.ncells}\n"
                "END DIMENSIONS\n\n"
            )

            foo.write("BEGIN GRIDDATA\n  IZONE\n")
            foo.write("  INTERNAL FACTOR 1 IPRN 0\n")
            izone = np.ravel(self.izone)
            i0 = 0
            i1 = line_length
            while i1 < self.izone.size:
                fmt = "".join(bfmt * line_length)
                foo.write(fmt.format(*izone[i0:i1]))
                foo.write("\n")
                i0 = i1
                i1 += line_length
            i1 = self.izone.size - i0
            fmt = "".join(bfmt * i1)
            foo.write(fmt.format(*izone[i0:]))
            foo.write("\nEND GRIDDATA\n")

    @staticmethod
    def load(f, model):
        """
        Method to load a Zone file for zonebudget 6.

        Parameter
        ---------
        f : str
            zone file name
        model : ZoneBudget6 object
            zonebudget 6 model object

        Returns
        -------
        ZoneFile6 object

        """
        from ..utils.flopy_io import multi_line_strip

        pkg_ws = os.path.split(f)[0]
        with open(f) as foo:
            t = [0]
            while t[0] != "ncells":
                t = multi_line_strip(foo).split()

            ncells = int(t[1])

            t = [0]
            while t[0] != "izone":
                t = multi_line_strip(foo).split()

            method = multi_line_strip(foo).split()[0]

            if method in ("internal", "open/close"):
                izone = np.zeros((ncells,), dtype=int)
                i = 0
                fobj = foo
                if method == "open/close":
                    fobj = open(os.path.join(pkg_ws, t[1]))
                while i < ncells:
                    t = multi_line_strip(fobj)
                    if t[0] == "open/close":
                        if fobj != foo:
                            fobj.close()
                        fobj = open(os.path.join(pkg_ws, t[1]))
                    for zn in t:
                        izone[i] = zn
                        i += 1
            else:
                izone = np.array([t[1]] * ncells, dtype=int)

        zon = ZoneFile6(model, izone)
        return zon


def _numpyvoid2numeric(a):
    # The budget record array has multiple dtypes and a slice returns
    # the flexible-type numpy.void which must be converted to a numeric
    # type prior to performing reducing functions such as sum() or
    # mean()
    return np.array([list(r) for r in a])


def sum_flux_tuples(fromzones, tozones, fluxes):
    tup = zip(fromzones, tozones, fluxes)
    sorted_tups = sort_tuple(tup)

    # Group the sorted tuples by (from zone, to zone)
    # itertools.groupby() returns the index (from zone, to zone) and
    # a list of the tuples with that index
    from_zones = []
    to_zones = []
    fluxes = []
    for (fz, tz), ftup in groupby(sorted_tups, lambda tup: tup[:2]):
        f = np.sum([tup[-1] for tup in list(ftup)])
        from_zones.append(fz)
        to_zones.append(tz)
        fluxes.append(f)
    return np.array(from_zones), np.array(to_zones), np.array(fluxes)


def sort_tuple(tup, n=2):
    """Sort a tuple by the first n values

    tup: tuple
        input tuple
    n : int
        values to sort tuple by (default is 2)

    Returns
    -------
    tup : tuple
        tuple sorted by the first n values

    """
    return tuple(sorted(tup, key=lambda t: t[:n]))


def _recarray_to_dataframe(
    recarray,
    zonenamedict,
    start_datetime=None,
    timeunit="D",
    index_key="totim",
    zones=None,
    pivot=False,
):
    """
    Method to convert zonebudget recarrays to pandas dataframes

    Parameters
    ----------
    recarray :
    zonenamedict :
    start_datetime :
    timeunit :
    index_key :
    names :
    zones :
    net :

    Returns
    -------

    pd.DataFrame
    """
    try:
        import pandas as pd
    except Exception as e:
        msg = f"ZoneBudget.get_dataframes() error import pandas: {e!s}"
        raise ImportError(msg)

    valid_index_keys = ["totim", "kstpkper"]
    s = f'index_key "{index_key}" is not valid.'
    assert index_key in valid_index_keys, s

    valid_timeunit = ["S", "M", "H", "D", "Y"]

    if timeunit.upper() == "SECONDS":
        timeunit = "S"
    elif timeunit.upper() == "MINUTES":
        timeunit = "M"
    elif timeunit.upper() == "HOURS":
        timeunit = "H"
    elif timeunit.upper() == "DAYS":
        timeunit = "D"
    elif timeunit.upper() == "YEARS":
        timeunit = "Y"

    errmsg = (
        f"Specified time units ({timeunit}) not recognized. Please use one of "
    )
    assert timeunit in valid_timeunit, errmsg + ", ".join(valid_timeunit) + "."

    df = pd.DataFrame().from_records(recarray)
    if start_datetime is not None and "totim" in list(df):
        totim = totim_to_datetime(
            df.totim,
            start=pd.to_datetime(start_datetime),
            timeunit=timeunit,
        )
        df["datetime"] = totim
        if pivot:
            return pd.DataFrame.from_records(recarray)

        index_cols = ["datetime", "name"]
    else:
        if pivot:
            return pd.DataFrame.from_records(recarray)

        if index_key == "totim" and "totim" in list(df):
            index_cols = ["totim", "name"]
        else:
            index_cols = ["time_step", "stress_period", "name"]

    df = df.set_index(index_cols)  # .sort_index(level=0)
    if zones is not None:
        keep_cols = zones
    else:
        keep_cols = zonenamedict.values()
    return df.loc[:, keep_cols]


def _get_budget(recarray, zonenamedict, names=None, zones=None, net=False):
    """
    Get a list of zonebudget record arrays.

    Parameters
    ----------
    recarray : np.recarray
        budget recarray
    zonenamedict : dict
        dictionary of zone names
    names : list of strings
        A list of strings containing the names of the records desired.
    zones : list of ints or strings
        A list of integer zone numbers or zone names desired.
    net : boolean
        If True, returns net IN-OUT for each record.

    Returns
    -------
    budget_list : list of record arrays
        A list of the zonebudget record arrays.

    """
    if isinstance(names, str):
        names = [names]
    if isinstance(zones, str):
        zones = [zones]
    elif isinstance(zones, int):
        zones = [zones]
    standard_fields = ["time_step", "stress_period", "name"]
    if "totim" in recarray.dtype.names:
        standard_fields.insert(0, "totim")
    select_fields = standard_fields + list(zonenamedict.values())
    select_records = np.where((recarray["name"] == recarray["name"]))
    if zones is not None:
        for idx, z in enumerate(zones):
            if isinstance(z, int):
                zones[idx] = zonenamedict[z]
        select_fields = standard_fields + zones

    if names is not None:
        names = _clean_budget_names(recarray, names)
        select_records = np.in1d(recarray["name"], names)
    if net:
        if names is None:
            names = _clean_budget_names(recarray, _get_record_names(recarray))
        net_budget = _compute_net_budget(recarray, zonenamedict)
        seen = []
        net_names = []
        for name in names:
            if name.endswith("_IN") or name.endswith("_OUT"):
                iname = "_".join(name.split("_")[:-1])
            else:
                iname = "_".join(name.split("_")[1:])
            if iname not in seen:
                seen.append(iname)
            else:
                net_names.append(iname)
        select_records = np.in1d(net_budget["name"], net_names)
        return net_budget[select_fields][select_records]
    else:
        return recarray[select_fields][select_records]


def _clean_budget_names(recarray, names):
    """
    Method to clean budget names

    Parameters
    ----------
    recarray : np.recarray

    names : list
        list of names in recarray

    Returns
    -------
        list
    """
    newnames = []
    mbnames = ["TOTAL_IN", "TOTAL_OUT", "IN-OUT", "PERCENT_DISCREPANCY"]
    for name in names:
        if name in mbnames:
            newnames.append(name)
        elif (
            not name.startswith("FROM_")
            and not name.startswith("TO_")
            and not name.endswith("_IN")
            and not name.endswith("_OUT")
        ):
            newname_in = "FROM_" + name.upper()
            newname_out = "TO_" + name.upper()
            if newname_in in recarray["name"]:
                newnames.append(newname_in)
            if newname_out in recarray["name"]:
                newnames.append(newname_out)
        else:
            if name in recarray["name"]:
                newnames.append(name)
    return newnames


def _get_record_names(recarray, stripped=False):
    """
    Get a list of water budget record names in the file.

    Returns
    -------
    out : list of strings
        List of unique text names in the binary file.

    """
    rec_names = np.unique(recarray["name"])
    if not stripped:
        return rec_names
    else:
        seen = []
        for recname in rec_names:
            if recname in ["IN-OUT", "TOTAL_IN", "TOTAL_OUT", "IN_OUT"]:
                continue
            if recname.endswith("_IN"):
                recname = recname[:-3]
            elif recname.endswith("_OUT"):
                recname = recname[:-4]
            if recname not in seen:
                seen.append(recname)
        seen.extend(["IN-OUT", "TOTAL", "IN_OUT"])
        return np.array(seen)


def _compute_net_budget(recarray, zonenamedict):
    """

    :param recarray:
    :param zonenamedict:
    :return:
    """
    recnames = _get_record_names(recarray)
    innames = [
        n for n in recnames if n.startswith("FROM_") or n.endswith("_IN")
    ]
    outnames = [
        n for n in recnames if n.startswith("TO_") or n.endswith("_OUT")
    ]
    select_fields = ["totim", "time_step", "stress_period", "name"] + list(
        zonenamedict.values()
    )
    if "totim" not in recarray.dtype.names:
        select_fields.pop(0)

    select_records_in = np.in1d(recarray["name"], innames)
    select_records_out = np.in1d(recarray["name"], outnames)
    in_budget = recarray[select_fields][select_records_in]
    out_budget = recarray[select_fields][select_records_out]
    net_budget = in_budget.copy()
    for f in [n for n in zonenamedict.values() if n in select_fields]:
        net_budget[f] = np.array([r for r in in_budget[f]]) - np.array(
            [r for r in out_budget[f]]
        )
    newnames = []
    for n in net_budget["name"]:
        if n.endswith("_IN") or n.endswith("_OUT"):
            newnames.append("_".join(n.split("_")[:-1]))
        else:
            newnames.append("_".join(n.split("_")[1:]))
    net_budget["name"] = newnames
    return net_budget


def _read_zb_zblst(fname):
    """Method to read zonebudget zblst output

    Parameters
    ----------
    fname : str
        zonebudget output file name

    Returns
    -------
        np.recarray
    """
    with open(fname) as foo:

        data = {}
        read_data = False
        flow_budget = False
        empty = 0
        prefix = ""
        while True:
            line = foo.readline().strip().upper()
            t = line.split()
            if t:
                if t[-1].strip() == "ZONES.":
                    line = foo.readline().strip()
                    zones = [int(i) for i in line.split()]
                    for zone in zones:
                        data[f"TO_ZONE_{zone}"] = []
                        data[f"FROM_ZONE_{zone}"] = []

            if "FLOW BUDGET FOR ZONE" in line:
                flow_budget = True
                read_data = False
                zlist = []
                empty = 0
                t = line.split()
                zone = int(t[4])
                if len(t[7]) > 4:
                    t.insert(8, t[7][4:])
                kstp = int(t[8]) - 1
                if len(t[11]) > 6:
                    t.append(t[11][6:])
                kper = int(t[12]) - 1
                if "ZONE" not in data:
                    data["ZONE"] = [zone]
                    data["KSTP"] = [kstp]
                    data["KPER"] = [kper]
                else:
                    data["ZONE"].append(zone)
                    data["KSTP"].append(kstp)
                    data["KPER"].append(kper)

            elif line in ("", " "):
                empty += 1

            elif read_data:
                if "=" in line:
                    t = line.split("=")
                    label = t[0].strip()
                    if "ZONE" in line:
                        if prefix == "FROM_":
                            zlist.append(int(label.split()[1]))
                            label = f"FROM_ZONE_{label.split()[1]}"
                        else:
                            label = f"TO_ZONE_{label.split()[-1]}"

                    elif "TOTAL" in line or "PERCENT DISCREPANCY" in line:
                        label = "_".join(label.split())

                    elif "IN - OUT" in line:
                        label = "IN-OUT"

                    else:
                        label = prefix + "_".join(label.split())

                    if label in data:
                        data[label].append(float(t[1]))
                    else:
                        data[label] = [float(t[1])]

                    if label == "PERCENT_DISCREPANCY":
                        # fill in non-connected zones with zeros...
                        for zone in zones:
                            if zone in zlist:
                                continue
                            data[f"FROM_ZONE_{zone}"].append(0)
                            data[f"TO_ZONE_{zone}"].append(0)

                elif "OUT:" in line:
                    prefix = "TO_"

                else:
                    pass

            elif flow_budget:
                if "IN:" in line:
                    prefix = "FROM_"
                    read_data = True
                    flow_budget = False

            else:
                pass

            if empty >= 30:
                break

    return _zb_dict_to_recarray(data)


def _read_zb_csv(fname):
    """Method to read zonebudget csv output

    Parameters
    ----------
    fname : str
        zonebudget output file name

    Returns
    -------
        np.recarray
    """
    with open(fname) as foo:
        data = {}
        zone_header = False
        read_data = False
        empty = 0
        while True:
            line = foo.readline().strip().upper()

            if "TIME STEP" in line:
                t = line.split(",")
                kstp = int(t[1]) - 1
                kper = int(t[3]) - 1
                totim = float(t[5])
                if "KSTP" not in data:
                    data["KSTP"] = []
                    data["KPER"] = []
                    data["TOTIM"] = []
                    data["ZONE"] = []

                zone_header = True
                empty = 0

            elif zone_header:
                t = line.split(",")
                zones = [int(i.split()[-1]) for i in t[1:] if i not in ("",)]

                for zone in zones:
                    data["KSTP"].append(kstp)
                    data["KPER"].append(kper)
                    data["ZONE"].append(zone)
                    data["TOTIM"].append(totim)

                zone_header = False
                read_data = True

            elif read_data:

                t = line.split(",")
                if "IN" in t[1]:
                    prefix = "FROM_"

                elif "OUT" in t[1]:
                    prefix = "TO_"

                else:
                    if "ZONE" in t[0] or "TOTAL" in t[0] or "IN-OUT" in t[0]:
                        label = "_".join(t[0].split())
                    elif "PERCENT ERROR" in line:
                        label = "_".join(t[0].split())
                        read_data = False
                    else:
                        label = prefix + "_".join(t[0].split())

                    if label not in data:
                        data[label] = []

                    for val in t[1:]:
                        if val in ("",):
                            continue

                        data[label].append(float(val))

            elif line in ("", " "):
                empty += 1

            else:
                pass

            if empty >= 25:
                break

    return _zb_dict_to_recarray(data)


def _read_zb_csv2(fname, add_prefix=True, aliases=None):
    """
    Method to read CSV2 output from zonebudget and CSV output
    from Zonebudget6

    Parameters
    ----------
    fname : str
        zonebudget output file name
    add_prefix : bool
        boolean flag to add "TO_", "FROM_" prefixes to column headings
    Returns
    -------
        np.recarray
    """
    with open(fname) as foo:
        # read the header and create the dtype
        h = foo.readline().upper().strip().split(",")
        h = [i.strip() for i in h if i]
        dtype = []
        prefix = "FROM_"
        for col in h:
            col = col.replace("-", "_")
            if not add_prefix:
                prefix = ""
            if col in ("TOTIM", "PERIOD", "STEP", "KSTP", "KPER", "ZONE"):
                if col in ("ZONE", "STEP", "KPER", "KSTP", "PERIOD"):
                    if col == "STEP":
                        col = "KSTP"
                    elif col == "PERIOD":
                        col = "KPER"
                    dtype.append((col, int))

                else:
                    dtype.append((col, float))

            elif col == "TOTAL IN":
                dtype.append(("_".join(col.split()), float))
                prefix = "TO_"
            elif col == "TOTAL OUT":
                dtype.append(("_".join(col.split()), float))
                prefix = ""
            elif col in ("FROM OTHER ZONES", "TO OTHER ZONES"):
                dtype.append(("_".join(col.split()), float))
            elif col == "IN_OUT":
                dtype.append(("IN-OUT", float))
            else:
                dtype.append((prefix + "_".join(col.split()), float))

        array = np.genfromtxt(foo, delimiter=",").T
        if len(array) != len(dtype):
            array = array[:-1]
        array.shape = (len(dtype), -1)
        data = {name[0]: list(array[ix]) for ix, name in enumerate(dtype)}
        data["KPER"] = list(np.array(data["KPER"]) - 1)
        data["KSTP"] = list(np.array(data["KSTP"]) - 1)
        return _zb_dict_to_recarray(data, aliases=aliases)


def _zb_dict_to_recarray(data, aliases=None):
    """
    Method to check the zonebudget dictionary and convert it to a
    numpy recarray.

    Parameters
    ----------
    data : dict
        dictionary of zonebudget data from CSV 1 or ZBLST files

    Returns
    -------
        np.recarray
    """
    # if steady state is used, storage will not be written
    if "FROM_STORAGE" in data:
        if len(data["FROM_STORAGE"]) < len(data["ZONE"]):
            adj = len(data["ZONE"]) - len(data["FROM_STORAGE"])
            adj = [0] * adj
            data["FROM_STORAGE"] = adj + data["FROM_STORAGE"]
            data["TO_STORAGE"] = adj + data["TO_STORAGE"]

    zones = list(np.unique(data["ZONE"]))
    zone_dtypes = []
    for zn in zones:
        if aliases is not None:
            if zn in aliases:
                zone_dtypes.append((aliases[zn], float))
            else:
                zone_dtypes.append((f"ZONE_{int(zn)}", float))
        else:
            zone_dtypes.append((f"ZONE_{int(zn)}", float))

    dtype = [
        ("totim", float),
        ("time_step", int),
        ("stress_period", int),
        ("name", object),
    ] + zone_dtypes

    if "TOTIM" not in data:
        dtype.pop(0)

    array = []
    allzones = data["ZONE"]
    for strt in range(0, len(data["ZONE"]), len(zones)):
        end = strt + len(zones)
        kstp = data["KSTP"][strt]
        kper = data["KPER"][strt]
        totim = None
        if "TOTIM" in data:
            totim = data["TOTIM"][strt]

        for name, values in data.items():
            if name in ("KSTP", "KPER", "TOTIM", "ZONE"):
                continue
            rec = [kstp, kper, name]
            if totim is not None:
                rec = [totim] + rec
            tmp = values[strt:end]
            tzones = allzones[strt:end]
            # check zone numbering matches header numbering, if not re-order
            if tzones != zones:
                idx = [zones.index(z) for z in tzones]
                tmp = [tmp[i] for i in idx]

            array.append(tuple(rec + tmp))

    array = np.array(array, dtype=dtype)
    return array.view(type=np.recarray)


def _pivot_recarray(recarray):
    """
    Method to pivot the zb output recarray to be compatible
    with the ZoneBudgetOutput method until the class is deprecated

    Returns
    -------

    """
    dtype = [("totim", float), ("kper", int), ("kstp", int), ("zone", int)]
    record_names = np.unique(recarray["name"])
    for rec_name in record_names:
        dtype.append((rec_name, float))

    rnames = recarray.dtype.names
    zones = {i: int(i.split("_")[-1]) for i in rnames if i.startswith("ZONE")}

    kstp_kper = np.vstack(
        sorted({(rec["time_step"], rec["stress_period"]) for rec in recarray})
    )
    pvt_rec = np.recarray((1,), dtype=dtype)
    n = 0
    for kstp, kper in kstp_kper:
        idxs = np.where(
            (recarray["time_step"] == kstp)
            & (recarray["stress_period"] == kper)
        )
        if len(idxs) == 0:
            pass
        else:
            temp = recarray[idxs]
            for zonename, zone in zones.items():
                if n != 0:
                    pvt_rec.resize((len(pvt_rec) + 1,), refcheck=False)
                pvt_rec["kstp"][-1] = kstp
                pvt_rec["kper"][-1] = kper
                pvt_rec["zone"][-1] = zone
                for rec in temp:
                    pvt_rec[rec["name"]][-1] = rec[zonename]

                if "totim" in rnames:
                    pvt_rec["totim"][-1] = temp["totim"][-1]
                else:
                    pvt_rec["totim"][-1] = 0

                n += 1
    return pvt_rec


def _volumetric_flux(recarray, modeltime, extrapolate_kper=False):
    """
    Method to generate a volumetric budget table based on flux information

    Parameters
    ----------
    recarray : np.recarray
        pivoted numpy recarray of zonebudget fluxes
    modeltime : flopy.discretization.ModelTime object
        flopy modeltime object
    extrapolate_kper : bool
        flag to determine if we fill in data gaps with other
        timestep information from the same stress period.
        if True, we assume that flux is constant throughout a stress period
        and the pandas dataframe returned contains a
        volumetric budget per stress period

        if False, calculates volumes from available flux data

    Returns
    -------
        pd.DataFrame

    """
    import pandas as pd

    nper = len(modeltime.nstp)
    volumetric_data = {}
    zones = np.unique(recarray["zone"])

    for key in recarray.dtype.names:
        volumetric_data[key] = []

    if extrapolate_kper:
        volumetric_data.pop("kstp")
        perlen = modeltime.perlen
        totim = np.add.accumulate(perlen)
        for per in range(nper):
            idx = np.where(recarray["kper"] == per)[0]

            if len(idx) == 0:
                continue

            temp = recarray[idx]

            for zone in zones:
                if zone == 0:
                    continue

                zix = np.where(temp["zone"] == zone)[0]

                if len(zix) == 0:
                    raise Exception

                for key in recarray.dtype.names:
                    if key == "totim":
                        volumetric_data[key].append(totim[per])

                    elif key == "tslen":
                        volumetric_data["perlen"].append(perlen[per])

                    elif key == "kstp":
                        continue

                    elif key == "kper":
                        volumetric_data[key].append(per)

                    elif key == "zone":
                        volumetric_data[key].append(zone)

                    else:
                        tmp = np.nanmean(temp[zix][key])
                        vol = tmp * perlen[per]
                        volumetric_data[key].append(vol)

    else:
        n = 0
        tslen = {}
        dtotim = {}
        totim = modeltime.totim
        for ix, nstp in enumerate(modeltime.nstp):
            for stp in range(nstp):
                idx = np.where(
                    (recarray["kper"] == ix) & (recarray["kstp"] == stp)
                )
                if len(idx[0]) == 0:
                    continue
                elif n == 0:
                    tslen[(stp, ix)] = totim[n]
                else:
                    tslen[(stp, ix)] = totim[n] - totim[n - 1]
                dtotim[(stp, ix)] = totim[n]
                n += 1

        ltslen = [tslen[(rec["kstp"], rec["kper"])] for rec in recarray]
        if len(np.unique(recarray["totim"])) == 1:
            ltotim = [dtotim[(rec["kstp"], rec["kper"])] for rec in recarray]
            recarray["totim"] = ltotim

        for name in recarray.dtype.names:
            if name in ("zone", "kstp", "kper", "tslen", "totim"):
                volumetric_data[name] = recarray[name]
            else:
                volumetric_data[name] = recarray[name] * ltslen

    return pd.DataFrame.from_dict(volumetric_data)


def dataframe_to_netcdf_fmt(df, zone_array, flux=True):
    """
    Method to transform a volumetric zonebudget dataframe into
    array format for netcdf.

    time is on axis 0
    zone is on axis 1

    Parameters
    ----------
    df : pd.DataFrame
    zone_array : np.ndarray
        zonebudget zones array
    flux : bool
        boolean flag to indicate if budget data is a flux "L^3/T" (True,
        default) or if the data have been processed to
        volumetric values "L^3" (False)

    Returns
    -------
        ZBNetOutput object

    """
    zones = np.sort(np.unique(df.zone.values))
    totim = np.sort(np.unique(df.totim.values))

    data = {}
    for col in df.columns:
        if col in ("totim", "zone", "kper", "kstp", "perlen"):
            pass
        else:
            data[col] = np.zeros((totim.size, zones.size), dtype=float)

    for i, time in enumerate(totim):
        tdf = df.loc[
            df.totim.isin(
                [
                    time,
                ]
            )
        ]
        tdf = tdf.sort_values(by=["zone"])

        for col in df.columns:
            if col in ("totim", "zone", "kper", "kstp", "perlen"):
                pass
            else:
                data[col][i, :] = tdf[col].values

    return ZBNetOutput(zones, totim, data, zone_array, flux=flux)


class ZBNetOutput:
    """
    Class that holds zonebudget netcdf output and allows export utilities
    to recognize the output data type.

    Parameters
    ----------
    zones : np.ndarray
        array of zone numbers
    time : np.ndarray
        array of totim
    arrays : dict
        dictionary of budget term arrays.
        axis 0 is totim,
        axis 1 is zones
    flux : bool
        boolean flag to indicate if budget data is a flux "L^3/T"(True,
        default) or if the data have been processed to
        volumetric values "L^3" (False)
    """

    def __init__(self, zones, time, arrays, zone_array, flux=True):
        self.zones = zones
        self.time = time
        self.arrays = arrays
        self.zone_array = zone_array
        self.flux = flux
