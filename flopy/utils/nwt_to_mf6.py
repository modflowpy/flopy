"""
Utilities for converting MODFLOW-NWT binary outputs to MODFLOW 6 format.
"""

import os
from pathlib import Path

import numpy as np


def get_icelltype_from_laytyp(laytyp):
    """
    Convert NWT LAYTYP values to MF6 ICELLTYPE values.

    Parameters
    ----------
    laytyp : array_like
        Layer type array from NWT LPF or UPW package.
        - 0: Confined
        - >0: Convertible (unconfined/water table)
        Shape can be (nlay,) or (nlay, nrow, ncol)

    Returns
    -------
    icelltype : ndarray
        Cell type array for MF6 (same shape as input):
        - 0: Confined
        - 1: Convertible

    Notes
    -----
    In MODFLOW-NWT, LAYTYP indicates layer properties:
    - LAYTYP = 0: Confined, transmissivity constant
    - LAYTYP > 0: Convertible, transmissivity varies with saturation

    In MODFLOW 6, ICELLTYPE similarly indicates cell properties:
    - ICELLTYPE = 0: Confined
    - ICELLTYPE > 0: Convertible

    For conversion, we use a simple mapping:
    - LAYTYP = 0 → ICELLTYPE = 0
    - LAYTYP > 0 → ICELLTYPE = 1

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils.nwt_to_mf6 import get_icelltype_from_laytyp
    >>> laytyp = np.array([1, 0, 0])  # Top layer convertible
    >>> icelltype = get_icelltype_from_laytyp(laytyp)
    >>> print(icelltype)
    [1 0 0]
    """
    laytyp = np.atleast_1d(laytyp).astype(np.int32)
    icelltype = np.where(laytyp > 0, 1, 0).astype(np.int32)
    return icelltype


class NwtToMf6Converter:
    """
    Convert MODFLOW-NWT binary outputs to MODFLOW 6 binary format.

    This class reads NWT head and budget files, applies necessary
    transformations, and writes MF6-compatible binary files that can
    be consumed by PRT or other MF6 post-processors.

    Parameters
    ----------
    hds_file : str
        Path to NWT head file (.hds)
    cbc_file : str
        Path to NWT cell budget file (.cbc)
    nlay : int
        Number of layers
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    delr : array_like
        Column spacing, shape (ncol,)
    delc : array_like
        Row spacing, shape (nrow,)
    top : array_like
        Top elevation, shape (nrow, ncol)
    botm : array_like
        Bottom elevation, shape (nlay, nrow, ncol)
    laytyp : array_like
        Layer type from LPF/UPW, shape (nlay,).
        0 = confined, >0 = convertible
    idomain : array_like, optional
        Domain array, shape (nlay, nrow, ncol).
        >0 = active, 0 = inactive, <0 = vertical pass-through
        If None, all cells are active.
    hdry : float, optional
        Head value for dry cells (default -999.0)
    hnoflo : float, optional
        Head value for inactive cells (default -9999.0)
    model_ws : str or PathLike, optional
        Model workspace for input files (default current directory)

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils import NwtToMf6Converter
    >>> # Set up grid parameters
    >>> nlay, nrow, ncol = 3, 10, 10
    >>> delr = np.ones(ncol) * 100.0
    >>> delc = np.ones(nrow) * 100.0
    >>> top = np.ones((nrow, ncol)) * 10.0
    >>> botm = np.zeros((nlay, nrow, ncol))
    >>> botm[0] = 5.0
    >>> botm[1] = 0.0
    >>> botm[2] = -5.0
    >>> laytyp = np.array([1, 0, 0])  # Top layer convertible
    >>>
    >>> # Create converter
    >>> converter = NwtToMf6Converter(
    ...     'model.hds',
    ...     'model.cbc',
    ...     nlay, nrow, ncol,
    ...     delr, delc, top, botm,
    ...     laytyp
    ... )
    >>>
    >>> # Convert files
    >>> converter.convert('mf6_output')
    """

    def __init__(
        self,
        hds_file,
        cbc_file,
        nlay,
        nrow,
        ncol,
        delr,
        delc,
        top,
        botm,
        laytyp,
        idomain=None,
        hdry=-999.0,
        hnoflo=-9999.0,
        model_ws=".",
    ):
        from ..mf6.utils import get_structured_connectivity
        from .binaryfile import CellBudgetFile, HeadFile

        self.hds_file = Path(model_ws) / hds_file
        self.cbc_file = Path(model_ws) / cbc_file
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.ncells = nlay * nrow * ncol

        # Store grid geometry
        self.delr = np.atleast_1d(delr).astype(np.float64)
        self.delc = np.atleast_1d(delc).astype(np.float64)
        self.top = np.atleast_2d(top).astype(np.float64)
        self.botm = np.atleast_3d(botm).astype(np.float64)

        # Convert LAYTYP to ICELLTYPE
        self.laytyp = np.atleast_1d(laytyp).astype(np.int32)
        self.icelltype = get_icelltype_from_laytyp(laytyp)

        # Expand ICELLTYPE to 3D if needed
        if self.icelltype.ndim == 1:
            # Expand (nlay,) to (nlay, nrow, ncol)
            # Use broadcasting to properly replicate each layer's value
            self.icelltype_3d = np.broadcast_to(
                self.icelltype[:, np.newaxis, np.newaxis],
                (nlay, nrow, ncol),
            ).copy()  # Copy to make it writable
        else:
            self.icelltype_3d = self.icelltype

        # Set IDOMAIN
        if idomain is None:
            self.idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
        else:
            self.idomain = np.atleast_3d(idomain).astype(np.int32)

        self.hdry = hdry
        self.hnoflo = hnoflo

        # Build connectivity arrays
        self.ia, self.ja, self.nja = get_structured_connectivity(
            nlay, nrow, ncol, self.idomain
        )

        # Open binary files
        self.hds_obj = HeadFile(str(self.hds_file))
        self.cbc_obj = CellBudgetFile(str(self.cbc_file))

        # Get time information
        self.times = self.hds_obj.get_times()
        self.kstpkper = self.hds_obj.get_kstpkper()

    def convert(
        self,
        output_dir,
        grb_name="gwf.grb",
        hds_name="gwf.hds",
        bud_name="gwf.bud",
        precision="double",
        verbose=False,
    ):
        """
        Convert NWT binary outputs to MF6 format.

        Parameters
        ----------
        output_dir : str or PathLike
            Directory for output files (will be created if doesn't exist)
        grb_name : str, optional
            Grid file name (default 'gwf.grb')
        hds_name : str, optional
            Head file name (default 'gwf.hds')
        bud_name : str, optional
            Budget file name (default 'gwf.bud')
        precision : str, optional
            'single' or 'double' for binary files (default 'double')
        verbose : bool, optional
            Print progress messages (default False)

        Returns
        -------
        dict
            Paths to created files:
            - 'grb': Path to grid file
            - 'hds': Path to head file
            - 'bud': Path to budget file
        """
        from ..mf6.utils import MfGrdFile
        from .binaryfile import CellBudgetFile, HeadFile

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grb_path = output_dir / grb_name
        hds_path = output_dir / hds_name
        bud_path = output_dir / bud_name

        if verbose:
            print("\nConverting NWT outputs to MF6 format")
            print(f"  Input head file: {self.hds_file}")
            print(f"  Input budget file: {self.cbc_file}")
            print(f"  Output directory: {output_dir}")
            print(f"  Grid dimensions: {self.nlay}L x {self.nrow}R x {self.ncol}C")
            print(f"  Time steps: {len(self.times)}")

        # Write GRB file
        if verbose:
            print(f"\nWriting grid file: {grb_path}")
        MfGrdFile.write_dis(
            str(grb_path),
            self.nlay,
            self.nrow,
            self.ncol,
            self.delr,
            self.delc,
            self.top,
            self.botm,
            self.ia,
            self.ja,
            idomain=self.idomain,
            icelltype=self.icelltype_3d,
            precision=precision,
            verbose=verbose,
        )

        # Write HDS file
        if verbose:
            print(f"\nWriting head file: {hds_path}")
        self._write_heads(hds_path, precision, verbose)

        # Write BUD file
        if verbose:
            print(f"\nWriting budget file: {bud_path}")
        self._write_budget(bud_path, precision, verbose)

        if verbose:
            print("\nConversion complete!")

        return {"grb": grb_path, "hds": hds_path, "bud": bud_path}

    def _write_heads(self, filename, precision, verbose):
        """Write head file with all time steps."""
        from .binaryfile import HeadFile

        # Read all heads
        head_dict = {}
        totim_dict = {}
        pertim_dict = {}

        for idx, kstpkper in enumerate(self.kstpkper):
            head = self.hds_obj.get_data(kstpkper=kstpkper)
            totim = self.times[idx]
            # Assume pertim = totim for now (could be refined)
            pertim = totim

            head_dict[kstpkper] = head
            totim_dict[kstpkper] = totim
            pertim_dict[kstpkper] = pertim

            if verbose:
                print(
                    f"  Read head for time step {kstpkper}: "
                    f"min={head.min():.2f}, max={head.max():.2f}"
                )

        # Write using HeadFile.write()
        HeadFile.write(
            str(filename),
            head_dict,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            precision=precision,
            totim=totim_dict,
            pertim=pertim_dict,
            verbose=verbose,
        )

    def _write_budget(self, filename, precision, verbose):
        """Write budget file with FLOW-JA-FACE, DATA-SPDIS, and DATA-SAT."""
        from ..mf6.utils import get_structured_flowja
        from .binaryfile import CellBudgetFile
        from .postprocessing import get_saturation

        if verbose:
            print(f"  Processing {len(self.kstpkper)} time steps...")

        # We'll write three terms per time step:
        # 1. FLOW-JA-FACE (imeth=1, array)
        # 2. DATA-SPDIS (imeth=6, list with qx, qy, qz)
        # 3. DATA-SAT (imeth=6, list)

        # Build list of records
        records = []

        for idx, kstpkper in enumerate(self.kstpkper):
            kstp, kper = kstpkper
            totim = self.times[idx]
            pertim = totim  # Simplified
            delt = 1.0  # Simplified - could calculate from times

            if verbose:
                print(f"  Processing time step {kstpkper}...")

            # Get head
            head = self.hds_obj.get_data(kstpkper=kstpkper)

            # Get face flows
            if verbose:
                print(
                    f"    Available budget terms: "
                    f"{self.cbc_obj.get_unique_record_names()}"
                )

            # Check which face flows are available
            # For 1D/2D models, not all face flows may exist
            available_terms = [
                t.decode().strip() for t in self.cbc_obj.get_unique_record_names()
            ]

            try:
                # FLOW RIGHT FACE (required for X-direction flow)
                if "FLOW RIGHT FACE" in available_terms:
                    frf_data = self.cbc_obj.get_data(
                        text="FLOW RIGHT FACE", kstpkper=kstpkper
                    )
                    if verbose:
                        print(
                            f"    FLOW RIGHT FACE: {type(frf_data)}, "
                            f"len={len(frf_data) if frf_data else 0}"
                        )
                    frf = frf_data[0] if frf_data and len(frf_data) > 0 else None
                else:
                    frf = None

                # FLOW FRONT FACE (required for Y-direction flow)
                if "FLOW FRONT FACE" in available_terms:
                    fff_data = self.cbc_obj.get_data(
                        text="FLOW FRONT FACE", kstpkper=kstpkper
                    )
                    if verbose:
                        print(
                            f"    FLOW FRONT FACE: {type(fff_data)}, "
                            f"len={len(fff_data) if fff_data else 0}"
                        )
                    fff = fff_data[0] if fff_data and len(fff_data) > 0 else None
                else:
                    fff = None

                # FLOW LOWER FACE (required for Z-direction flow)
                if "FLOW LOWER FACE" in available_terms:
                    flf_data = self.cbc_obj.get_data(
                        text="FLOW LOWER FACE", kstpkper=kstpkper
                    )
                    if verbose:
                        print(
                            f"    FLOW LOWER FACE: {type(flf_data)}, "
                            f"len={len(flf_data) if flf_data else 0}"
                        )
                    flf = flf_data[0] if flf_data and len(flf_data) > 0 else None
                else:
                    flf = None

                # Validate at least one face flow exists
                if frf is None and fff is None and flf is None:
                    raise ValueError("No face flows found in budget file")

                # Create zero arrays for missing face flows
                # For 1D/2D models, not all directions have flow
                shape_3d = (self.nlay, self.nrow, self.ncol)
                if frf is None:
                    frf = np.zeros(shape_3d, dtype=np.float64)
                if fff is None:
                    fff = np.zeros(shape_3d, dtype=np.float64)
                if flf is None:
                    flf = np.zeros(shape_3d, dtype=np.float64)

            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not read face flows: {e}")
                    print(f"    Skipping time step {kstpkper}")
                continue

            # 1. Convert to FLOW-JA-FACE
            flowja = get_structured_flowja(
                (frf, fff, flf),
                ia=self.ia,
                ja=self.ja,
                nlay=self.nlay,
                nrow=self.nrow,
                ncol=self.ncol,
            )

            records.append(
                {
                    "data": flowja,
                    "kstp": kstp,
                    "kper": kper,
                    "totim": totim,
                    "pertim": pertim,
                    "delt": delt,
                    "text": "FLOW-JA-FACE",
                    "imeth": 1,
                }
            )

            # 2. DATA-SPDIS (specific discharge) - SKIPPED for now
            # TODO: Requires refactoring get_specific_discharge() to work
            # without model object or implementing a minimal wrapper. PRT can
            # calculate specific discharge from FLOW-JA-FACE if needed.
            # See PHASE3_REMAINING_ISSUES.md for details.

            # 3. Calculate saturation
            sat = get_saturation(
                head, self.top, self.botm, self.icelltype_3d, self.hdry, self.hnoflo
            )

            # Build list data for DATA-SAT
            sat_flat = sat.flatten(order="F")
            active_sat = ~np.isnan(sat_flat)
            nlist_sat = np.sum(active_sat)

            if nlist_sat > 0:
                nodes_sat = np.arange(self.ncells)[active_sat] + 1  # 1-based

                # Create structured array for imeth=6
                dtype = np.dtype(
                    [
                        ("node", np.int32),
                        ("node2", np.int32),
                        ("sat", np.float64),
                    ]
                )
                sat_data = np.zeros(nlist_sat, dtype=dtype)
                sat_data["node"] = nodes_sat
                sat_data["node2"] = nodes_sat
                sat_data["sat"] = sat_flat[active_sat]

                records.append(
                    {
                        "data": sat_data,
                        "kstp": kstp,
                        "kper": kper,
                        "totim": totim,
                        "pertim": pertim,
                        "delt": delt,
                        "text": "DATA-SAT",
                        "imeth": 6,
                        "ndat": 1,
                    }
                )

        # Write all records
        if verbose:
            print(f"  Writing {len(records)} budget records...")

        CellBudgetFile.write(
            str(filename),
            records,
            precision=precision,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            verbose=verbose,
        )

    def __repr__(self):
        return (
            f"NwtToMf6Converter(\n"
            f"  hds_file={self.hds_file},\n"
            f"  cbc_file={self.cbc_file},\n"
            f"  grid={self.nlay}x{self.nrow}x{self.ncol},\n"
            f"  time_steps={len(self.times)}\n"
            f")"
        )
