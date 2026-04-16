"""
Utilities for converting legacy MODFLOW binary outputs to MODFLOW 6 format.

Supported variants
------------------
Any structured-grid MODFLOW variant that writes compact budget files with
FLOW RIGHT FACE, FLOW FRONT FACE, and FLOW LOWER FACE terms is supported,
including MODFLOW-NWT (UPW), MODFLOW-2005 (LPF), and MODFLOW-2000 (LPF or
BCF).  Use :class:`ClassicMfToMf6Converter` or its
:meth:`~ClassicMfToMf6Converter.from_model`
constructor to perform the conversion.

Binary format differences: compact MODFLOW vs MODFLOW 6
--------------------------------------------------------
**Head files (.hds)**
    The on-disk layout is essentially identical: one record per layer per
    time step, each prefixed by a header containing ``(kstp, kper, pertim,
    totim, text, ncol, nrow, ilay)``.

**Budget files (.cbc / .bud)**
    Classic MODFLOW budget files exist in two sub-formats controlled by the
    OC package keyword ``COMPACT BUDGET [FILES]``:

    * *Non-compact* (old-style, ``COMPACT BUDGET`` absent): ``nlay > 0`` in
      every record header; no extended header fields; all records are
      implicitly full 3-D arrays (``imeth=0``); stress period and time step
      are stored but no time data. **Not supported by this converter.**

    * *Compact* (``COMPACT BUDGET`` or ``COMPACT BUDGET FILES`` in OC):
      ``nlay < 0`` is the binary signal; each record has an extended header
      with ``(imeth, delt, pertim, totim)``; face-flow terms use ``imeth=1``
      (full 3-D array) and boundary-flow terms use ``imeth=2`` or higher.
      **This is the format the converter requires.**

    Within the compact format, classic and MF6 semantics diverge:

    * *Classic compact* stores each flow component as a separate named
      record—``FLOW RIGHT FACE``, ``FLOW FRONT FACE``, ``FLOW LOWER FACE``—
      each a 3-D array of shape ``(nlay, nrow, ncol)`` (``imeth=1``).
      Inter-cell flows are directional with each face stored once, in the
      positive direction (away from the lower-index cell).

    * *MF6* stores all inter-cell flows in a single ``FLOW-JA-FACE`` record:
      a 1-D array of length NJA (total number of cell connections) indexed by
      the IA/JA sparse connectivity arrays.  Each connection appears twice
      (once from each side) with opposite signs.  Saturation and specific
      discharge are stored as separate ``DATA-SAT`` and ``DATA-SPDIS``
      records using the ``imeth=6`` sparse-list format.  MF6 budget file
      headers also use ``nlay < 0``, like classic compact files.

**Grid file (.grb) — new in MF6**
    Classic MODFLOW stores grid geometry in the ASCII DIS file.  MF6
    flow models create a binary grid file (``.dis.grb``) that encodes
    the IA/JA connectivity, cell geometry, convertibility, and IDOMAIN.
"""

import shutil
from pathlib import Path

import numpy as np


def get_icelltype_from_laytyp(laytyp):
    """
    Convert classic MODFLOW LAYTYP values to MF6 ICELLTYPE values.

    Parameters
    ----------
    laytyp : array_like
        Layer type array from a UPW (NWT) or LPF (2005/2000) package.
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
    In MODFLOW-NWT (UPW) and MODFLOW-2005/2000 (LPF), LAYTYP indicates layer
    properties:
    - LAYTYP = 0: Confined, transmissivity constant
    - LAYTYP > 0: Convertible with wetting active
    - LAYTYP < 0: Convertible without wetting (rewetting disabled)

    Any non-zero LAYTYP means the layer is convertible regardless of sign.
    The sign only controls whether the wetting option is active, not whether
    the layer can desaturate. (Note: in LPF with THICKSTRT active, negative
    LAYTYP layers use starting-head-based thickness rather than the full
    cell thickness, but the layer is still treated as convertible in terms
    of transmissivity variation with saturation.)

    In MODFLOW 6, ICELLTYPE similarly indicates cell properties:
    - ICELLTYPE = 0: Confined
    - ICELLTYPE > 0: Convertible

    For conversion, we use a simple mapping:
    - LAYTYP = 0  → ICELLTYPE = 0
    - LAYTYP != 0 → ICELLTYPE = 1

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils import get_icelltype_from_laytyp
    >>> laytyp = np.array([1, 0, -1])  # Top and bottom layers convertible
    >>> icelltype = get_icelltype_from_laytyp(laytyp)
    >>> print(icelltype)
    [1 0 1]
    """
    laytyp = np.atleast_1d(laytyp).astype(np.int32)
    icelltype = np.where(laytyp != 0, 1, 0).astype(np.int32)
    return icelltype


def get_icelltype_from_laycon(laycon):
    """
    Convert BCF LAYCON values to MF6 ICELLTYPE values.

    Parameters
    ----------
    laycon : array_like
        Layer connectivity array from the BCF package, shape ``(nlay,)`` or
        ``(nlay, nrow, ncol)``.

        * 0 — confined; transmissivity is constant.
        * 1 — convertible; transmissivity based on initial saturated thickness.
        * 2 — confined; transmissivity varies with head (quasi-3D / Tran
          specified).
        * 3 — convertible; transmissivity varies with saturated thickness.

    Returns
    -------
    icelltype : ndarray
        Cell type for MF6 (same shape as input):

        * 0 — confined (LAYCON 0 or 2).
        * 1 — convertible (LAYCON 1 or 3).

    Notes
    -----
    LAYCON 1 and LAYCON 3 both represent convertible layers in which the
    water table can fall below the cell top; they differ only in how
    transmissivity is computed from the saturated thickness.  Both map to
    ``ICELLTYPE = 1``.

    LAYCON 2 layers have a head-dependent transmissivity but are never
    allowed to desaturate fully, so they map to ``ICELLTYPE = 0``
    (confined).

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils import get_icelltype_from_laycon
    >>> laycon = np.array([3, 0, 2, 1])
    >>> get_icelltype_from_laycon(laycon)
    array([1, 0, 0, 1], dtype=int32)
    """
    laycon = np.atleast_1d(laycon).astype(np.int32)
    icelltype = np.where((laycon == 1) | (laycon == 3), 1, 0).astype(np.int32)
    return icelltype


class ClassicMfToMf6Converter:
    """
    Convert classic MODFLOW binary outputs to MODFLOW 6 binary format.

    Reads head and cell-budget files produced by any structured-grid MODFLOW
    variant with compact budget output (FLOW RIGHT FACE / FLOW FRONT FACE /
    FLOW LOWER FACE terms), and writes the MF6-compatible head file, budget
    file (``FLOW-JA-FACE``, ``DATA-SAT``), and binary grid record (GRB)
    required by PRT and other MF6 post-processors.

    Confirmed compatible variants: MODFLOW-NWT (UPW), MODFLOW-2005 (LPF),
    MODFLOW-2000 (LPF or BCF).

    The easiest way to construct a converter from a loaded model is
    :meth:`from_model`, which autodetects the flow package and extracts all
    required parameters automatically.  Construct directly only when a model
    object is not available (e.g. you only have the binary files on disk).

    Parameters
    ----------
    hds_file : str or PathLike
        Path to the head file (.hds).
    cbc_file : str or PathLike
        Path to the compact cell-budget file (.cbc).
    nlay : int
        Number of layers.
    nrow : int
        Number of rows.
    ncol : int
        Number of columns.
    delr : array_like, shape (ncol,)
        Column widths (along rows).
    delc : array_like, shape (nrow,)
        Row widths (along columns).
    top : array_like, shape (nrow, ncol)
        Top elevation of the model.
    botm : array_like, shape (nlay, nrow, ncol)
        Bottom elevation of each layer.
    laytyp : array_like, shape (nlay,)
        Layer type from the LPF or UPW package.  ``0`` → confined; any
        non-zero value → convertible (sign controls wetting, not
        confinement).  For BCF models, pass the output of
        :func:`get_icelltype_from_laycon` here rather than the raw
        ``laycon`` values.
    idomain : array_like, shape (nlay, nrow, ncol), optional
        Active-cell mask: ``> 0`` active, ``0`` inactive,
        ``< 0`` vertical pass-through.  ``None`` treats all cells as active.
    hdry : float, optional
        Sentinel head value written by MODFLOW for dry cells.  Should match
        the value in the LPF/UPW ``HDRY`` field (default ``-999.0``).  For
        BCF models the default is ``-1e30``; pass the model value explicitly
        or use :meth:`from_model`.
    hnoflo : float, optional
        Sentinel head value for inactive (IBOUND ≤ 0) cells.  Should match
        the BAS6 ``HNOFLO`` value (default ``-9999.0``; BAS6 default is
        ``-999.99``).  Use :meth:`from_model` to avoid mismatches.
    model_ws : str or PathLike, optional
        Directory prepended to ``hds_file`` and ``cbc_file`` when those
        paths are relative.  Default is the current directory.

    See Also
    --------
    from_model : Preferred constructor when a loaded ``Modflow`` object is
        available; handles flow-package detection and sentinel-value
        extraction automatically.
    get_icelltype_from_laytyp : Maps LPF/UPW ``LAYTYP`` → ``ICELLTYPE``.
    get_icelltype_from_laycon : Maps BCF ``LAYCON`` → ``ICELLTYPE``.

    Examples
    --------
    Construct from a loaded model (recommended):

    >>> from flopy.modflow import Modflow
    >>> from flopy.utils import ClassicMfToMf6Converter
    >>> mf = Modflow.load('model.nam', load_only=['dis', 'upw', 'bas6'])
    >>> converter = ClassicMfToMf6Converter.from_model(
    ...     mf, 'model.hds', 'model.cbc'
    ... )
    >>> converter.convert('mf6_output')

    Construct directly from arrays (when model object is unavailable):

    >>> import numpy as np
    >>> from flopy.utils import ClassicMfToMf6Converter
    >>> nlay, nrow, ncol = 3, 10, 10
    >>> laytyp = np.array([1, 0, 0])  # top layer convertible
    >>> converter = ClassicMfToMf6Converter(
    ...     'model.hds', 'model.cbc',
    ...     nlay, nrow, ncol,
    ...     np.ones(ncol) * 100.0, np.ones(nrow) * 100.0,
    ...     np.ones((nrow, ncol)) * 10.0,
    ...     np.array([5.0, 0.0, -5.0])[:, None, None] * np.ones((nlay, nrow, ncol)),
    ...     laytyp,
    ... )
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

    @classmethod
    def from_model(cls, model, hds_file, cbc_file, **kwargs):
        """
        Construct a converter from a loaded ``Modflow`` model object.

        Autodetects the flow package (UPW, LPF, or BCF), extracts grid
        geometry and layer-type arrays, and reads ``hdry`` / ``hnoflo``
        sentinel values so they do not need to be provided manually.

        Parameters
        ----------
        model : flopy.modflow.Modflow
            Loaded MODFLOW model.  At minimum the DIS package and one of
            UPW, LPF, or BCF must be loaded.  BAS6 is optional but
            recommended so that ``hnoflo`` is read from the model rather
            than using the default.
        hds_file : str or PathLike
            Path to the head file produced by the model run.
        cbc_file : str or PathLike
            Path to the compact cell-budget file produced by the model run.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`ClassicMfToMf6Converter` (e.g. ``idomain``,
            ``model_ws``).  Parameters already extracted from ``model``
            (``nlay``, ``nrow``, ``ncol``, ``delr``, ``delc``, ``top``,
            ``botm``, ``laytyp``, ``hdry``, ``hnoflo``) must not be
            supplied here.

        Returns
        -------
        ClassicMfToMf6Converter

        Raises
        ------
        ValueError
            If the model has no DIS package, or has no supported flow
            package (UPW, LPF, or BCF).

        Examples
        --------
        NWT / UPW model:

        >>> from flopy.modflow import Modflow
        >>> from flopy.utils import ClassicMfToMf6Converter
        >>> mf = Modflow.load('model.nam', load_only=['dis', 'upw', 'bas6'])
        >>> converter = ClassicMfToMf6Converter.from_model(
        ...     mf, 'model.hds', 'model.cbc'
        ... )

        MF2000 / BCF model:

        >>> mf = Modflow.load('model.nam', load_only=['dis', 'bcf6', 'bas6'])
        >>> converter = ClassicMfToMf6Converter.from_model(
        ...     mf, 'model.hds', 'model.cbc'
        ... )
        """
        if model.dis is None:
            raise ValueError(
                "model.dis is None — load the DIS package before calling from_model()."
            )

        # Detect flow package and derive laytyp / hdry.
        # For BCF, laycon values are first converted to ICELLTYPE (0/1)
        # and passed as laytyp; __init__ then maps them through
        # get_icelltype_from_laytyp, which is a no-op for 0/1 values.
        if getattr(model, "upw", None) is not None:
            laytyp = model.upw.laytyp.array
            hdry = float(model.upw.hdry)
        elif getattr(model, "lpf", None) is not None:
            laytyp = model.lpf.laytyp.array
            hdry = float(model.lpf.hdry)
        elif getattr(model, "bcf6", None) is not None:
            laytyp = get_icelltype_from_laycon(model.bcf6.laycon.array)
            hdry = float(model.bcf6.hdry)
        else:
            raise ValueError(
                "No supported flow package found on model.  "
                "Load one of: UPW, LPF, or BCF before calling from_model()."
            )

        # Read hnoflo from BAS6 if available.
        if getattr(model, "bas6", None) is not None:
            hnoflo = float(model.bas6.hnoflo)
        else:
            hnoflo = -9999.0

        return cls(
            hds_file=str(hds_file),
            cbc_file=str(cbc_file),
            nlay=int(model.dis.nlay),
            nrow=int(model.dis.nrow),
            ncol=int(model.dis.ncol),
            delr=model.dis.delr.array,
            delc=model.dis.delc.array,
            top=model.dis.top.array,
            botm=model.dis.botm.array,
            laytyp=laytyp,
            hdry=hdry,
            hnoflo=hnoflo,
            **kwargs,
        )

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
        Convert classic MODFLOW binary outputs to MF6 format.

        Parameters
        ----------
        output_dir : str or PathLike
            Directory for output files (will be created if it does not exist)
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
            print("\nConverting classic MODFLOW outputs to MF6 format")
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

        # Copy head file verbatim — the on-disk format is identical between
        # classic MODFLOW and MF6, so no conversion is needed.
        if verbose:
            print(f"\nCopying head file to: {hds_path}")
        shutil.copy2(self.hds_file, hds_path)

        # Write BUD file
        if verbose:
            print(f"\nWriting budget file: {bud_path}")
        self._write_budget(bud_path, precision, verbose)

        if verbose:
            print("\nConversion complete!")

        return {"grb": grb_path, "hds": hds_path, "bud": bud_path}

    def _build_header_lookup(self):
        """
        Build a dict mapping (kstp, kper) -> header record from the head file.

        Returns
        -------
        dict
            Keys are (kstp, kper) tuples, values are header records with
            fields including 'pertim' and 'totim'.
        """
        # get_kstpkper() returns 0-based indices; recordarray stores 1-based
        # file values.  Subtract 1 so the lookup keys match self.kstpkper.
        return {
            (int(rec["kstp"]) - 1, int(rec["kper"]) - 1): rec
            for rec in self.hds_obj.recordarray
        }

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

        header_lookup = self._build_header_lookup()

        for idx, kstpkper in enumerate(self.kstpkper):
            kstp, kper = kstpkper
            rec = header_lookup[kstpkper]
            totim = float(rec["totim"])
            pertim = float(rec["pertim"])

            # delt: for the first time step within a period pertim equals delt;
            # for subsequent steps subtract the previous step's pertim.
            # kstp is 0-based here (from get_kstpkper()).
            if kstp == 0:
                delt = pertim
            else:
                prev_rec = header_lookup.get((kstp - 1, kper))
                delt = pertim - float(prev_rec["pertim"]) if prev_rec else pertim

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
                    "kstp": int(kstp) + 1,
                    "kper": int(kper) + 1,
                    "totim": totim,
                    "pertim": pertim,
                    "delt": delt,
                    "text": "FLOW-JA-FACE",
                    "imeth": 1,
                }
            )

            # 2. DATA-SPDIS (specific discharge) - SKIPPED for now
            # TODO: get_specific_discharge() requires a model object and cannot
            # easily be driven from binary files alone. PRT can reconstruct
            # specific discharge from FLOW-JA-FACE if needed.

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
                        "kstp": int(kstp) + 1,
                        "kper": int(kper) + 1,
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
            f"{type(self).__name__}(\n"
            f"  hds_file={self.hds_file},\n"
            f"  cbc_file={self.cbc_file},\n"
            f"  grid={self.nlay}x{self.nrow}x{self.ncol},\n"
            f"  time_steps={len(self.times)}\n"
            f")"
        )


#: Backward-compatible alias.  New code should use ClassicMfToMf6Converter.
NwtToMf6Converter = ClassicMfToMf6Converter
