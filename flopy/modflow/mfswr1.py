"""
mfswr1 module.  Contains the ModflowSwr1 class. Note that the user can access
the ModflowSwr1 class as `flopy.modflow.ModflowSwr1`.

The Surface-Water Routing (SWR1) Process is available in MODFLOW-NWT and
MODFLOW-2005. Every input item of SWR1 can be written and loaded.

Additional information for this MODFLOW process can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/swr.html>`_.

Sources used (and where they disagree, the Fortran code wins because it is
what MODFLOW actually reads):

* Data Input Instructions for the SWR1 Process, version 1.04
  (MODFLOW-NWT 1.3.0 ``doc/SWRProcessInputInstructions_v1.04.pdf``).
* SWR1_AdditionsForNWT_1.01.0.txt (SFR2 coupling, Newton correction).
* ``gwf2swr7.f`` from MODFLOW-NWT 1.3.0 (SWR1 1.05): GWF2SWR7AR, GWF2SWR7RP,
  SSWRLSTRD, SSWREXTRD, SSWR_RD_COMM, SSWR_RDTABDATA, SSWR_RDDRO.

Differences between manual and code that this module follows (code):

* Item 13a field order per ISTRTYPE (manual lists a superset):
    -2 : STRINV [STRLEN] [STRLEN2 if ISTRCONN>0]
     2 : [STRLEN] [STRLEN2 if ISTRCONN>0]
     3 : STRVAL
     4 : NSTRPTS (+ item 13c table immediately after the 13a line)
     5 : STRCD STRCD2 STRINV STRINV2 STRWID [STRWID2 if STRWID<0] STRLEN
         STRMAN ISTRDIR
     6,8 : STRCD STRCD3 STRINV STRWID STRVAL ISTRDIR
     7,9,10 : STRCD STRCD2 STRCD3 STRINV STRWID STRVAL ISTRDIR
     11 : (nothing)          12 : STRLEN STRVAL|TABDATAn
     13 : STRCD STRCD3 STRWSMO STRVAL ISTRDIR
  followed by ISFRSEG [ISFRRCH] when ISTRCONN = 0 and SFR is in the model.
* Item 13b: CSTROTYP, then (not for ISTRTYPE=1) ISTRORCH [ISTROQCON if
  FLOW] CSTRLO, then CSTRCRIT (number, TABDATAn or REACHn), then (ISTRTYPE>1)
  STRCRITC STRRT, then STRMAX, then (ISTRTYPE>1, optional) CSTRVAL.
* Item 4f: an extra integer (IOBSLOC2) is read for FLOW, STRUCTURE and
  BASEFLOW; for BASEFLOW it is the layer (0 = all layers).
* Extra CTABTYPE ``SWRSTAGE`` (stage read from a SWR stage output file) and
  options USE_STAGE_TRANSFORM, USE_WEIGHTED_HEADS,
  USE_ORIGINAL_2D_QM_FORMULATION, USE_IMPLICIT_INVARIATE_QM,
  USE_SOURCECELL_INVARIATE_QM, USE_MULTICORE n, USE_MULTICORE_VECTOR n.
* Every input line is read into a 200-character buffer, so written lines
  are checked against that limit.

Extras beyond the file format
-----------------------------
* :meth:`ModflowSwr1.validate` (run by ``write_file``) reports problems
  MODFLOW would stop on or silently misread: reach numbering and grid
  indices, symmetric connectivity, structures connected to reaches that are
  really connected, type rules (11/12 unconnected, 13 connected, 11 needs
  SFR), TABDATAn/REACHn references, geometry fields and tables, observation
  locations and first-stress-period requirements.
* :class:`SwrExternal` keeps lists and grids in their own OPEN/CLOSE files;
  loading preserves OPEN/CLOSE references (EXTERNAL units and whole-block
  redirections of items 4b, 11 and 13 are read but written inline).
* ASCII and binary time-series files are loaded into ``tabfiles[i]["data"]``.
* ``zero_based=True`` uses FloPy's zero-based KRCH/IRCH/JRCH.
* ``max_line_length`` can be raised for executables built with a longer
  SWR1 line buffer.

Index convention
----------------
All reach, layer, row, column, structure, geometry and table numbers are
**one-based, exactly as in the SWR1 manual** (unlike most FloPy packages),
unless ``zero_based=True`` (then KRCH/IRCH/JRCH only are zero-based).
KRCH keeps its manual meaning (negative = start layer of a multi-layer
reach).  Stress-period keys are zero-based, as everywhere in FloPy.
"""

import os
import warnings

import numpy as np

from ..pakbase import Package
from ..utils import Util2d

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
# option name -> kind of the value that follows it
#   None   : keyword only
#   "unit" : output unit (>0 ASCII, <0 binary)
#   "ascii": output unit that must be ASCII (>0)
#   "input": input unit (ASCII, >0)
#   "int"  : plain integer (number of cores)
OPTIONS = {
    "PRINT_SWR_TO_SCREEN": None,
    "SAVE_SWRDT": "unit",
    "SAVE_CONVERGENCE_HISTORY": "ascii",
    "SAVE_RIVER_PACKAGE": "ascii",
    "SAVE_RIVER_PACKAGE_ALL": "ascii",
    "SAVE_AVERAGE_RESULTS": None,
    "USE_TABFILES": None,
    "USE_NONCONVERGENCE_CONTINUE": None,
    "USE_UPSTREAM_WEIGHTING": None,
    "USE_INEXACT_NEWTON": None,
    "USE_STEADYSTATE_STORAGE": None,
    "USE_LAGGED_OPR_DATA": None,
    "USE_LINEAR_DEPTH_SCALING": None,
    "USE_DIAGONAL_SCALING": None,
    "USE_L2NORM_SCALING": None,
    "USE_RCMREORDERING": None,
    "USE_RCMREORDERING_IF_IMPROVEMENT": None,
    "USE_STAGE_TRANSFORM": None,
    "USE_WEIGHTED_HEADS": None,
    "USE_IMPLICIT_NEWTON_CORRECTION": None,
    "USE_EXPLICIT_NEWTON_CORRECTION": None,
    "USE_ORIGINAL_2D_QM_FORMULATION": None,
    "USE_IMPLICIT_INVARIATE_QM": None,
    "USE_SOURCECELL_INVARIATE_QM": None,
    "USE_FRACTIONAL_TOLR": None,
    "USE_L2NORM_TOLR": None,
    "SAVE_SWROBSERVATIONS": "unit",
    "SAVE_SWROBSERVATIONS_ALL": "unit",
    "USE_MULTICORE": "int",
    "USE_MULTICORE_VECTOR": "int",
    "USE_DIRECT_RUNOFF": "input",
    "USE_GLOBAL_TOLA": None,
    "USE_ABSOLUTE_TOLA": None,
}

# mutually exclusive options (the Fortran silently drops the second one)
CONFLICTS = [
    ("SAVE_RIVER_PACKAGE", "SAVE_RIVER_PACKAGE_ALL"),
    ("USE_DIAGONAL_SCALING", "USE_L2NORM_SCALING"),
    ("USE_RCMREORDERING", "USE_RCMREORDERING_IF_IMPROVEMENT"),
    ("USE_IMPLICIT_NEWTON_CORRECTION", "USE_EXPLICIT_NEWTON_CORRECTION"),
    ("USE_ORIGINAL_2D_QM_FORMULATION", "USE_IMPLICIT_INVARIATE_QM"),
    ("USE_FRACTIONAL_TOLR", "USE_L2NORM_TOLR"),
    ("SAVE_SWROBSERVATIONS", "SAVE_SWROBSERVATIONS_ALL"),
    ("USE_GLOBAL_TOLA", "USE_ABSOLUTE_TOLA"),
]

TABTYPES = ("TIME", "RAIN", "EVAP", "LATFLOW", "STAGE", "STRUCTURE", "SWRSTAGE")
# table types that take CTABRCH / ITABRCH
TABTYPES_WITH_REACHES = ("RAIN", "EVAP", "LATFLOW", "STAGE", "SWRSTAGE")
INTERPOLATION = ("NONE", "AVERAGE", "INTERPOLATE")
OBSTYPES = ("STAGE", "DEPTH", "RBOTTOM", "FLOW", "BASEFLOW", "STRUCTURE")
OBSTYPES_WITH_LOC2 = ("FLOW", "BASEFLOW", "STRUCTURE")

# item 13a fields that follow ISTRTYPE, per |ISTRTYPE| (code order)
STRUCT_FIELDS = {
    0: [],
    1: [],
    3: ["strval"],
    5: ["strcd", "strcd2", "strinv", "strinv2", "strwid"],  # + see writer
    6: ["strcd", "strcd3", "strinv", "strwid", "strval", "istrdir"],
    7: ["strcd", "strcd2", "strcd3", "strinv", "strwid", "strval", "istrdir"],
    8: ["strcd", "strcd3", "strinv", "strwid", "strval", "istrdir"],
    9: ["strcd", "strcd2", "strcd3", "strinv", "strwid", "strval", "istrdir"],
    10: ["strcd", "strcd2", "strcd3", "strinv", "strwid", "strval", "istrdir"],
    11: [],
    12: ["strlen", "strval"],
    13: ["strcd", "strcd3", "strwsmo", "strval", "istrdir"],
}
OPERABLE_TYPES = (1, 3, 8, 9, 10)
MAXLINE = 199  # Fortran reads into CHARACTER(LEN=200)

REACH_DTYPE = np.dtype(
    [
        ("reach", int),
        ("iroutetype", int),
        ("irgnum", int),
        ("krch", int),
        ("irch", int),
        ("jrch", int),
        ("rlen", float),
    ]
)


def _first_delt(model):
    """Length of the first MODFLOW time step (what SWR1 uses for RTMIN/RTMAX=0)."""
    dis = model.get_package("DIS")
    if dis is None:
        return None
    perlen = float(dis.perlen.array[0])
    nstp = int(dis.nstp.array[0])
    r = float(dis.tsmult.array[0])
    return perlen / nstp if r == 1.0 else perlen * (r - 1.0) / (r**nstp - 1.0)


def _is_adaptive(rtmin, rtmax, rtmult, model):
    """
    GWF2SWR7AR: zero RTMIN/RTMAX are replaced by the first DELT *before*
    deciding on adaptive time stepping (which makes DMAXRAI/DMAXSTG/DMAXINF
    be read). Adaptive if (RTMIN < RTMAX or both were zero) and RTMULT > 1.
    """
    if rtmult <= 1.0:
        return False
    if rtmin == 0 and rtmax == 0:
        return True
    delt = _first_delt(model)
    lo = delt if rtmin == 0 else rtmin
    hi = delt if rtmax == 0 else rtmax
    if lo is None or hi is None:
        raise ValueError("RTMIN or RTMAX = 0 needs the DIS package to know DELT")
    return lo < hi


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def _fmt(v):
    """Format one value for free-format Fortran input (float32 exact)."""
    if isinstance(v, str):
        return f"'{v}'" if (" " in v) else v
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.9G}"


def _line(*values, comment=None, maxlen=MAXLINE):
    s = " ".join(_fmt(v) for v in values)
    if len(s) > maxlen:
        raise ValueError(
            f"SWR1 input line longer than {maxlen} characters "
            f"(MODFLOW would truncate it): {s[:80]}... Use fewer values per "
            "line, or raise max_line_length for an executable built with a "
            "longer SWR1 line buffer."
        )
    if comment:
        pad = " " * max(1, 40 - len(s))
        if len(s) + len(pad) + 2 + len(comment) <= maxlen:
            s = f"{s}{pad}# {comment}"
    return s + "\n"


def _to_float(tok):
    return float(tok.replace("d", "e").replace("D", "E"))


def _to_int(tok):
    try:
        return int(tok)
    except ValueError:
        return int(_to_float(tok))


def _tokenize(line):
    """Split a line the way MODFLOW's URWORD does (spaces/commas, 'quotes')."""
    toks, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in " ,\t":
            i += 1
            continue
        if c == "'":
            j = line.find("'", i + 1)
            j = n if j < 0 else j
            toks.append(line[i + 1 : j])
            i = j + 1
            continue
        j = i
        while j < n and line[j] not in " ,\t":
            j += 1
        toks.append(line[i:j])
        i = j
    return toks


def _as_table(data, ncol, name):
    """Normalise list input (list of tuples / ndarray / recarray) to 2-D float."""
    if data is None:
        return None
    data = _plain(data)
    if isinstance(data, np.ndarray) and data.dtype.names:
        data = np.column_stack([data[n] for n in data.dtype.names])
    arr = np.atleast_2d(np.asarray(data, dtype=float))
    if arr.size == 0:
        return arr.reshape(0, ncol)
    if arr.shape[1] != ncol:
        raise ValueError(f"{name}: expected {ncol} columns, got {arr.shape[1]}")
    return arr


def _rec(arr, names, ints=()):
    """2-D float table -> record array with the given field names."""
    dt = [(n, int if n in ints else float) for n in names]
    out = np.zeros(len(arr), dtype=dt)
    for k, n in enumerate(names):
        out[n] = arr[:, k]
    return out.view(np.recarray)


class SwrExternal(np.ndarray):
    """
    Data stored in its own file and referenced with OPEN/CLOSE.

    Wrap any list (items 4a, 6, 7a, 8a, 9a, 10, 11b, 11c, 13c, 14a, 15) or
    grid (items 7b, 8b, 9b, 14b, direct-runoff arrays) to have it written to
    ``fname`` (relative to model_ws) instead of inline. Loading keeps
    OPEN/CLOSE references this way, so they survive a load/write cycle.
    """

    def __new__(cls, data, fname):
        obj = np.asanyarray(data).view(cls)
        obj.fname = str(fname)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.fname = getattr(obj, "fname", None)

    def __reduce__(self):  # keep fname when pickled/copied
        state = super().__reduce__()
        return state[0], state[1], (state[2], self.fname)

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.fname = state[1]


def _plain(data):
    """Strip the SwrExternal wrapper (keeps record arrays as they are)."""
    if isinstance(data, SwrExternal):
        return data.view(np.recarray if data.dtype.names else np.ndarray)
    return data


# --------------------------------------------------------------------------
# reading helpers
# --------------------------------------------------------------------------
class _Source:
    """Line source that mimics the comment handling of SSWR_RD_COMM/URDCOM."""

    def __init__(self, lines, name):
        self.lines = lines
        self.pos = 0
        self.name = name

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(f.read().splitlines(), path)

    @staticmethod
    def _is_comment(line):
        s = line.strip()
        return (not s) or s[0] in "#!" or s.startswith("//")

    def skip_comments(self):
        while self.pos < len(self.lines) and self._is_comment(self.lines[self.pos]):
            self.pos += 1

    def next_line(self):
        self.skip_comments()
        if self.pos >= len(self.lines):
            raise EOFError(f"unexpected end of SWR1 input in {self.name}")
        line = self.lines[self.pos]
        self.pos += 1
        return line

    def next_tokens(self):
        return _tokenize(self.next_line())

    def peek_tokens(self):
        self.skip_comments()
        if self.pos >= len(self.lines):
            return []
        return _tokenize(self.lines[self.pos])

    # file-like interface for Util2d.load / Util2d.load_txt
    def read(self):
        rest = "\n".join(self.lines[self.pos :])
        self.pos = len(self.lines)
        return rest

    def readline(self):
        if self.pos >= len(self.lines):
            return ""
        line = self.lines[self.pos]
        self.pos += 1
        return line + "\n"


class _Reader:
    """Holds the open sources (main file, EXTERNAL units, OPEN/CLOSE files)."""

    def __init__(self, main, model, ext_unit_dict):
        self.main = main
        self.model = model
        self.ext_unit_dict = ext_unit_dict or {}
        self.units = {}

    def _resolve(self, fname):
        fname = fname.replace("\\", os.sep)
        for base in (self.model.model_ws, os.getcwd(), os.path.dirname(self.main.name)):
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"SWR1 OPEN/CLOSE file not found: {fname}")

    def unit_source(self, iu):
        if iu not in self.units:
            if iu not in self.ext_unit_dict:
                raise KeyError(f"SWR1 EXTERNAL unit {iu} not in the name file")
            self.units[iu] = _Source.from_file(self.ext_unit_dict[iu].filename)
        return self.units[iu]

    def block_source(self, src, with_name=False):
        """SSWREXTRD: optional EXTERNAL/OPEN/CLOSE/INTERNAL for a whole block."""
        toks = src.peek_tokens()
        key = toks[0].upper() if toks else ""
        out, fname = src, None
        if key == "EXTERNAL":
            src.next_line()
            out = self.unit_source(_to_int(toks[1]))
        elif key == "OPEN/CLOSE":
            src.next_line()
            fname = toks[1].replace("\\", "/")
            out = _Source.from_file(self._resolve(toks[1]))
        elif key == "INTERNAL":
            src.next_line()
        return (out, fname) if with_name else out

    def read_list(self, src, nrow, ncol, keep_external=False):
        """SSWRLSTRD: read nrow records of ncol numbers."""
        out = np.zeros((nrow, ncol))
        if nrow == 0:
            return (out, None) if keep_external else out
        sub, fname = self.block_source(src, with_name=True)
        for i in range(nrow):
            toks = sub.next_tokens()
            out[i] = [_to_float(t) for t in toks[:ncol]]
        return (out, fname) if keep_external else out

    def read_array(self, src, dtype, name):
        nrow, ncol = self.model.nrow, self.model.ncol
        toks = src.peek_tokens()
        fname = None
        if toks and toks[0].upper() == "OPEN/CLOSE":
            fname = toks[1].replace("\\", "/")
        src.skip_comments()
        u2d = Util2d.load(
            src,
            self.model,
            (nrow, ncol),
            dtype,
            name,
            ext_unit_dict=self.ext_unit_dict or None,
        )
        arr = u2d.array
        return SwrExternal(arr, fname) if fname else arr

    def read_rec(self, src, nrow, names, ints=()):
        """read_list + field names; OPEN/CLOSE lists come back as SwrExternal."""
        arr, fname = self.read_list(src, nrow, len(names), keep_external=True)
        rec = _rec(arr, names, ints)
        return SwrExternal(rec, fname) if fname else rec


# --------------------------------------------------------------------------
# the package
# --------------------------------------------------------------------------
class ModflowSwr1(Package):
    """
    MODFLOW Surface-Water Routing (SWR1) Process -- complete implementation.

    Parameters
    ----------
    model : flopy.modflow.Modflow
        Must be a MODFLOW-NWT or MODFLOW-2005 model.
    nreaches : int
        NREACHES. If None it is taken from ``reach_data``.
    iswronly, iswrcbc, iswrprgf, iswrpstg, iswrpqaq, iswrpqm, iswrpstr,
    iswrpfrn : int
        Item 1a flags/units (see manual). Output units > 0 are ASCII,
        < 0 binary; they are added to the name file automatically.
    auxiliary : list of str
        AUX variable names (item 1a, max 20).
    options : dict or list
        Item 1b keywords, e.g. ``{"USE_TABFILES": None, "SAVE_SWRDT": -205}``
        or ``["USE_TABFILES", ("SAVE_SWRDT", -205)]``. The SWROPTIONS flag
        and the END keyword are written automatically.
    dlenconv ... dmaxinf : float/int
        Item 2. DMAXRAI/DMAXSTG/DMAXINF are only written when adaptive time
        stepping is active (RTMIN < RTMAX, or both zero, and RTMULT > 1).
    isolver ... ptolr : float/int
        Item 3. IPC is written if ISOLVER > 1, NLEVELS/DROPTOL if IPC = 4,
        IBTPRT if IBT > 1, PTOLR if USE_FRACTIONAL_TOLR or USE_L2NORM_TOLR.
    reach_data : array-like
        Item 4a, columns (reach, iroutetype, irgnum, krch, irch, jrch, rlen).
        Use :meth:`get_empty_reach_data`.
    connectivity : dict or list
        Item 4b. ``{reach: [connected reaches]}`` or a list ordered by reach.
    tabfiles : list of dict
        Item 4d (USE_TABFILES is added automatically). Keys: ``itab``,
        ``ctabtype``, ``itabunit``, ``cintp`` (default NONE), ``reaches``
        ("ALL" or list; RAIN/EVAP/LATFLOW/STAGE/SWRSTAGE only), and
        optionally ``data`` (N x 2 array of time, value) plus ``fname`` to
        have the time-series file written and registered in the name file.
        ``header`` (dict of TIMEOFFSET/TIMESCALE/OFFSET/SCALE) is optional.
    observations : list of tuple/dict
        Item 4f: (cobsname, cobstype, iobsloc[, iobsloc2]). Requires
        SAVE_SWROBSERVATIONS(_ALL) in ``options``.
    stress_period_data : dict
        ``{kper: spd}`` with zero-based kper; see :meth:`get_empty_spd`.
        Stress periods not given reuse the previous data (ITMP = -1).
        Keys of ``spd``:

        * ``iptflg`` : int
        * ``boundary`` : (reach, iswrbnd) list -> item 6
        * ``rain``, ``evap``, ``latflow`` : (reach, value) list -> items
          7a/8a/9a, or a (nrow, ncol) array -> items 7b/8b/9b
        * ``geometry_assignment`` : (reach, igeonumr, gzshift) -> item 10
        * ``geometries`` : ``{igeonum: geo}`` -> item 11, where ``geo``
          has ``igeotype``, ``igcndop``, ``gmanning`` and, as required,
          ``gwidth``, ``gbelev``, ``gsslope``, ``gcnd``, ``glk``,
          ``gcndln``, ``getextd``, ``xsec`` (N x 2: XB, ELEVB, item 11b)
          or ``table`` (N x 5: ELEV VOL WETPER SAREA XAREA, item 11c)
        * ``structures`` : ``{reach: [struct, ...]}`` -> items 12/13;
          an empty list means NSTRUCT=0, None means reuse (NSTRUCT=-1).
          ``struct`` is a dict with ``istrconn``, ``istrtype`` and the
          fields listed in STRUCT_FIELDS, plus ``rating`` (N x 2, type 4),
          ``strwid2`` (type 5 rectangular), ``strlen``/``strman`` (type 5),
          ``strinv``/``strlen``/``strlen2`` (types +-2), ``isfrseg``/
          ``isfrrch`` (SFR coupling), and for operable types 1,3,8,9,10:
          ``cstrotyp``, ``istrorch``, ``istroqcon``, ``cstrlo``,
          ``cstrcrit``, ``strcritc``, ``strrt``, ``strmax``, ``cstrval``.
        * ``stage`` : (reach, stage) list -> 14a, or (nrow, ncol) -> 14b
        * ``aux`` : (reach, aux1, ..., auxn) list -> item 15
    direct_runoff : dict
        ``{kper: {"map": int (nrow,ncol), "mult": (nrow,ncol),
        "val": (nrow,ncol)}}`` (any subset). Requires USE_DIRECT_RUNOFF.
    output_filenames : dict
        ``{unit: filename}`` for output files; defaults are generated.
    zero_based : bool
        If True, KRCH/IRCH/JRCH in ``reach_data`` are zero-based like
        other FloPy packages (SFR2). A negative KRCH keeps its manual
        meaning (multi-layer reach starting at layer |KRCH|, one-based).
        Reach, structure, geometry and table numbers stay one-based.
    max_line_length : int
        Longest input line allowed (default 199, the 200-character buffer
        of MODFLOW-NWT 1.3.0). Raise it only for an executable built with a
        longer SWR1 line buffer.
    extension, unitnumber, filenames : as in other FloPy packages.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow(version="mfnwt")
    >>> dis = flopy.modflow.ModflowDis(m, nrow=1, ncol=2)
    >>> rd = flopy.modflow.ModflowSwr1.get_empty_reach_data(2)
    >>> rd["iroutetype"], rd["irgnum"], rd["irch"] = 3, [1, 2], 1
    >>> rd["jrch"], rd["rlen"] = [1, 2], 100.0
    >>> swr = flopy.modflow.ModflowSwr1(
    ...     m, reach_data=rd, connectivity={1: [2], 2: [1]}
    ... )

    Loading an existing model reads the SWR1 file as well:

    >>> m = flopy.modflow.Modflow.load("model.nam", version="mfnwt")
    >>> swr = m.get_package("SWR")
    """

    def __init__(
        self,
        model,
        nreaches=None,
        iswronly=0,
        iswrcbc=0,
        iswrprgf=0,
        iswrpstg=0,
        iswrpqaq=0,
        iswrpqm=0,
        iswrpstr=0,
        iswrpfrn=0,
        auxiliary=None,
        options=None,
        dlenconv=1.0,
        timeconv=1.0,
        rtini=0.0,
        rtmin=0.0,
        rtmax=0.0,
        rtprn=0.0,
        rtmult=1.0,
        ntmult=1,
        dmingrad=0.0,
        dmindpth=0.0,
        dmaxrai=0.0,
        dmaxstg=0.0,
        dmaxinf=0.0,
        isolver=1,
        nouter=50,
        ninner=1,
        ibt=1,
        tols=1.0e-9,
        tolr=1.0e-1,
        tola=0.0,
        dampss=1.0,
        damptr=1.0,
        iprswr=0,
        mutswr=2,
        ipc=0,
        nlevels=7,
        droptol=1.0e-3,
        ibtprt=0,
        ptolr=0.0,
        reach_data=None,
        connectivity=None,
        tabfiles=None,
        observations=None,
        stress_period_data=None,
        direct_runoff=None,
        output_filenames=None,
        zero_based=False,
        max_line_length=MAXLINE,
        extension="swr",
        unitnumber=None,
        filenames=None,
        heading=None,
    ):
        if unitnumber is None:
            unitnumber = ModflowSwr1._defaultunit()
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )
        if model.version in ("mf2k", "mfusg", "mf6"):
            raise ValueError(f"SWR1 is not available for model version {model.version}")
        self._generate_heading()
        if heading:
            self.heading = heading
        self.url = "swr.html"

        # ---- item 1a / 1b
        self.reach_data = self._reach_array(reach_data)
        if nreaches is None:
            nreaches = 0 if self.reach_data is None else len(self.reach_data)
        self.nreaches = int(nreaches)
        self.iswronly = iswronly
        self.iswrcbc = iswrcbc
        self.iswrprgf = iswrprgf
        self.iswrpstg = iswrpstg
        self.iswrpqaq = iswrpqaq
        self.iswrpqm = iswrpqm
        self.iswrpstr = iswrpstr
        self.iswrpfrn = iswrpfrn
        self.auxiliary = list(auxiliary or [])
        if len(self.auxiliary) > 20:
            raise ValueError("SWR1 allows at most 20 auxiliary variables")
        self.options = self._normalise_options(options)
        self.tabfiles = [dict(t) for t in (tabfiles or [])]
        if any(int(t["itabunit"]) < 0 for t in self.tabfiles):
            warnings.warn(
                "SWR1 binary time-series files (ITABUNIT < 0) crash MODFLOW-NWT "
                "1.3.0: SSWR_RDTABDATA calls BACKSPACE on the negative unit "
                "('invalid logical unit number'). Use ASCII (ITABUNIT > 0).",
                stacklevel=2,
            )
        if self.tabfiles and "USE_TABFILES" not in self.options:
            self.options["USE_TABFILES"] = None

        # ---- item 2
        self.dlenconv, self.timeconv = dlenconv, timeconv
        self.rtini, self.rtmin, self.rtmax, self.rtprn = rtini, rtmin, rtmax, rtprn
        self.rtmult, self.ntmult = rtmult, ntmult
        self.dmingrad, self.dmindpth = dmingrad, dmindpth
        self.dmaxrai, self.dmaxstg, self.dmaxinf = dmaxrai, dmaxstg, dmaxinf
        # ---- item 3
        self.isolver, self.nouter, self.ninner, self.ibt = isolver, nouter, ninner, ibt
        self.tols, self.tolr, self.tola = tols, tolr, tola
        self.dampss, self.damptr = dampss, damptr
        self.iprswr, self.mutswr = iprswr, mutswr
        self.ipc, self.nlevels, self.droptol = ipc, nlevels, droptol
        self.ibtprt, self.ptolr = ibtprt, ptolr
        # ---- item 4
        self.connectivity = self._connectivity_dict(connectivity)
        self.observations = [self._obs_tuple(o) for o in (observations or [])]
        # ---- stress periods
        self.stress_period_data = dict(stress_period_data or {})
        self.direct_runoff = dict(direct_runoff or {})

        self.output_filenames = dict(output_filenames or {})
        self.zero_based = bool(zero_based)
        self.max_line_length = int(max_line_length)
        self._register_files()
        self.parent.add_package(self)

    # ------------------------------------------------------------------
    # flopy plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _ftype():
        return "SWR"

    @staticmethod
    def _defaultunit():
        return 36

    @staticmethod
    def get_empty_reach_data(nreaches):
        """Item 4a record array (one-based reach/layer/row/column)."""
        rd = np.zeros(nreaches, dtype=REACH_DTYPE).view(np.recarray)
        rd["reach"] = np.arange(1, nreaches + 1)
        rd["iroutetype"] = 3
        rd["krch"] = 1
        return rd

    @staticmethod
    def get_empty_spd():
        """Template for one stress period of SWR1 data."""
        return {
            "iptflg": 0,
            "boundary": None,
            "rain": None,
            "evap": None,
            "latflow": None,
            "geometry_assignment": None,
            "geometries": None,
            "structures": None,
            "stage": None,
            "aux": None,
        }

    # ------------------------------------------------------------------
    # normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reach_array(reach_data):
        if reach_data is None:
            return None
        if isinstance(reach_data, np.ndarray) and reach_data.dtype.names:
            rd = np.zeros(len(reach_data), dtype=REACH_DTYPE)
            for n in REACH_DTYPE.names:
                rd[n] = reach_data[n]
        else:
            arr = np.atleast_2d(np.asarray(reach_data, dtype=float))
            rd = np.zeros(len(arr), dtype=REACH_DTYPE)
            for k, n in enumerate(REACH_DTYPE.names):
                rd[n] = arr[:, k]
        rd = rd.view(np.recarray)
        fname = getattr(reach_data, "fname", None)
        return SwrExternal(rd, fname) if fname else rd

    @staticmethod
    def _normalise_options(options):
        out = {}
        if options is None:
            return out
        items = options.items() if isinstance(options, dict) else options
        for item in items:
            if isinstance(item, str):
                toks = item.split()
                key, val = toks[0], (int(toks[1]) if len(toks) > 1 else None)
            else:
                key, val = item[0], (item[1] if len(item) > 1 else None)
            key = key.upper()
            if key in ("END", "SWROPTIONS"):
                continue
            if key not in OPTIONS:
                raise ValueError(f"unknown SWR1 option {key}")
            kind = OPTIONS[key]
            if kind is not None and val is None:
                raise ValueError(f"SWR1 option {key} requires a value")
            if kind is None:
                val = None
            elif kind in ("ascii", "input") and int(val) <= 0:
                raise ValueError(f"SWR1 option {key} needs a positive (ASCII) unit")
            out[key] = None if val is None else int(val)
        for a, b in CONFLICTS:
            if a in out and b in out:
                raise ValueError(f"SWR1 options {a} and {b} are mutually exclusive")
        return out

    def _connectivity_dict(self, conn):
        if conn is None:
            return {}
        if isinstance(conn, dict):
            return {int(k): [int(c) for c in v] for k, v in conn.items()}
        return {i + 1: [int(c) for c in v] for i, v in enumerate(conn)}

    @staticmethod
    def _obs_tuple(o):
        if isinstance(o, dict):
            o = (o["cobsname"], o["cobstype"], o["iobsloc"], o.get("iobsloc2", 0))
        o = list(o)
        name, otype = str(o[0]), str(o[1]).upper()
        if otype not in OBSTYPES:
            raise ValueError(f"unknown SWR1 observation type {otype}")
        if len(name) > 20:
            raise ValueError(f"SWR1 observation name longer than 20 chars: {name}")
        loc2 = int(o[3]) if len(o) > 3 else 0
        return (name, otype, int(o[2]), loc2)

    # ------------------------------------------------------------------
    # name-file registration
    # ------------------------------------------------------------------
    def _default_out_name(self, tag, unit):
        ext = "bin" if unit < 0 else "csv"
        return f"{self.parent.name}.swr.{tag}.{ext}"

    def _add_out(self, unit, tag, binary=None):
        if not unit:
            return
        binflag = (unit < 0) if binary is None else binary
        fname = self.output_filenames.get(unit) or self.output_filenames.get(abs(unit))
        if fname is None:
            fname = self._default_out_name(tag, -1 if binflag else 1)
        self.parent.add_output_file(
            abs(unit), fname=fname, binflag=binflag, package=self._ftype()
        )

    def _register_files(self):
        if self.iswrcbc > 0:
            self._add_out(self.iswrcbc, "cbc", binary=True)
        self._add_out(self.iswrprgf, "rgflow")
        self._add_out(self.iswrpstg, "stage")
        self._add_out(self.iswrpqaq, "qaq")
        self._add_out(self.iswrpqm, "qm")
        self._add_out(self.iswrpstr, "str")
        tags = {
            "SAVE_SWRDT": "dt",
            "SAVE_CONVERGENCE_HISTORY": "cnv",
            "SAVE_RIVER_PACKAGE": "riv",
            "SAVE_RIVER_PACKAGE_ALL": "riv",
            "SAVE_SWROBSERVATIONS": "obs",
            "SAVE_SWROBSERVATIONS_ALL": "obs",
        }
        for key, tag in tags.items():
            if key in self.options:
                self._add_out(self.options[key], tag)
        if "USE_DIRECT_RUNOFF" in self.options:
            iu = self.options["USE_DIRECT_RUNOFF"]
            if self.direct_runoff:
                fname = f"{self.parent.name}.swr.dro"
                self.parent.add_external(fname, iu, binflag=False)
        for t in self.tabfiles:
            if t.get("data") is not None:
                fname = t.get("fname") or f"{self.parent.name}.swr.tab{int(t['itab'])}"
                t["fname"] = fname
                iu = int(t["itabunit"])
                self.parent.add_external(fname, abs(iu), binflag=iu < 0)

    # ------------------------------------------------------------------
    # derived properties
    # ------------------------------------------------------------------
    def _line(self, *values, comment=None):
        return _line(*values, comment=comment, maxlen=self.max_line_length)

    def _kij_one_based(self):
        """reach_data columns KRCH, IRCH, JRCH as MODFLOW expects them."""
        k = self.reach_data["krch"].astype(int).copy()
        i = self.reach_data["irch"].astype(int).copy()
        j = self.reach_data["jrch"].astype(int).copy()
        if self.zero_based:
            k[k >= 0] += 1
            i += 1
            j += 1
        return k, i, j

    @property
    def _sfr_active(self):
        return self.parent.get_package("SFR") is not None

    @property
    def _adaptive(self):
        return _is_adaptive(self.rtmin, self.rtmax, self.rtmult, self.parent)

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self):
        """
        Check the data for errors MODFLOW would stop on (or silently misread).

        Returns a list of messages; empty means no problem found. Covers
        reach numbering, grid indices, routing types, symmetric
        connectivity, structure connections and type rules, references to
        tables (TABDATAn) and reaches (REACHn), geometry fields and tables,
        observation locations, and first-stress-period requirements.
        """
        err = []
        m = self.parent
        nlay, nrow, ncol = m.nlay, m.nrow, m.ncol
        n = self.nreaches
        rd = self.reach_data
        if rd is None or len(rd) != n:
            return ["reach_data must have NREACHES rows"]
        reaches = sorted(int(r) for r in rd["reach"])
        if reaches != list(range(1, n + 1)):
            err.append("reach numbers in reach_data must be 1..NREACHES, once each")
        k, i, j = self._kij_one_based()
        for idx in range(n):
            r = int(rd["reach"][idx])
            if int(rd["iroutetype"][idx]) not in (1, 2, 3, 4):
                err.append(f"reach {r}: IROUTETYPE must be 1-4")
            if not (1 <= i[idx] <= nrow and 1 <= j[idx] <= ncol):
                err.append(
                    f"reach {r}: row/column ({i[idx]},{j[idx]}) outside the grid"
                )
            if k[idx] > nlay or abs(k[idx]) > nlay:
                err.append(f"reach {r}: layer {k[idx]} outside the grid")
        conn = self.connectivity
        for r, cl in conn.items():
            if not 1 <= r <= n:
                err.append(f"connectivity: reach {r} out of range")
            for c in cl:
                if not 1 <= c <= n:
                    err.append(f"connectivity: reach {r} -> {c} out of range")
                elif c == r:
                    err.append(f"connectivity: reach {r} is connected to itself")
                elif r not in conn.get(c, []):
                    err.append(
                        f"connectivity not symmetric: {r} -> {c} but not {c} -> {r}"
                    )
            if len(set(cl)) != len(cl):
                err.append(f"connectivity: reach {r} lists a connection twice")

        tabs = {int(t["itab"]): t for t in self.tabfiles}
        if len(tabs) != len(self.tabfiles):
            err.append("tabfiles: ITAB numbers must be unique")
        if sum(t["ctabtype"].upper() == "TIME" for t in self.tabfiles) > 1:
            err.append("tabfiles: CTABTYPE TIME can only be used once")
        for t in self.tabfiles:
            if int(t["itabunit"]) == 0:
                err.append(f"tabfile {t['itab']}: ITABUNIT must be non-zero")
            rch = t.get("reaches", "ALL")
            if not isinstance(rch, str):
                bad = [x for x in rch if not 1 <= int(x) <= n]
                if bad:
                    err.append(f"tabfile {t['itab']}: reaches out of range {bad[:5]}")

        def tabref(val, where, kind="STRUCTURE"):
            if isinstance(val, str) and "TABDATA" in val.upper():
                it = int(val.upper().split("TABDATA")[1])
                if it not in tabs:
                    err.append(f"{where}: {val} refers to an undefined table")
                elif tabs[it]["ctabtype"].upper() != kind:
                    err.append(f"{where}: {val} must be a {kind} table")

        nobs_ok = (
            "SAVE_SWROBSERVATIONS" in self.options
            or "SAVE_SWROBSERVATIONS_ALL" in self.options
        )
        if self.observations and not nobs_ok:
            err.append("observations need SAVE_SWROBSERVATIONS(_ALL) in options")
        for name, otype, loc, loc2 in self.observations:
            if not 1 <= loc <= n:
                err.append(f"observation {name}: reach {loc} out of range")
            if otype == "FLOW" and loc2 not in conn.get(loc, []):
                err.append(
                    f"observation {name}: reach {loc2} is not connected to {loc}"
                )
            if otype == "BASEFLOW" and not 0 <= loc2 <= nlay:
                err.append(f"observation {name}: layer {loc2} out of range")
        if self.direct_runoff and "USE_DIRECT_RUNOFF" not in self.options:
            err.append("direct_runoff needs USE_DIRECT_RUNOFF in options")

        sp0 = self.stress_period_data.get(0)
        if sp0 is None:
            err.append("stress_period_data must define stress period 0")
        else:

            def count(key):
                v = sp0.get(key)
                if v is None:
                    return 0
                if (
                    np.shape(v) == (nrow, ncol)
                    and getattr(v, "dtype", None) is not None
                    and v.dtype.names is None
                ):
                    return n
                return len(v)

            if count("boundary") != n:
                err.append("stress period 0: boundary must list every reach")
            asn = sp0.get("geometry_assignment")
            if asn is None or len(_as_table(asn, 3, "g")) != n:
                err.append("stress period 0: geometry_assignment must list every reach")
            if count("stage") != n:
                err.append(
                    "stress period 0: stage must be given for every reach "
                    "(initial stages)"
                )
            if self.auxiliary and count("aux") != n:
                err.append("stress period 0: aux must list every reach")
        sfr = self._sfr_active
        for kper, spd in self.stress_period_data.items():
            tag = f"stress period {kper}"
            for key in ("boundary", "rain", "evap", "latflow", "stage", "aux"):
                v = spd.get(key)
                if v is None or (
                    np.shape(v) == (nrow, ncol)
                    and getattr(v, "dtype", None) is not None
                    and v.dtype.names is None
                ):
                    continue
                t = _as_table(v, len(self.auxiliary) + 1 if key == "aux" else 2, key)
                bad = [int(x) for x in t[:, 0] if not 1 <= int(x) <= n]
                if bad:
                    err.append(f"{tag} {key}: reach numbers out of range {bad[:5]}")
                if len(set(t[:, 0].astype(int))) != len(t):
                    err.append(f"{tag} {key}: a reach is listed more than once")
            asn = spd.get("geometry_assignment")
            if asn is not None:
                t = _as_table(asn, 3, "geometry_assignment")
                geos = spd.get("geometries") or {}
                for g in sorted({int(x) for x in t[:, 1] if x >= 1}):
                    if g not in geos:
                        err.append(f"{tag}: geometry {g} is assigned but not defined")
                        continue
                    err += [
                        f"{tag} geometry {g}: {e}" for e in self._check_geo(geos[g])
                    ]
            for r, lst in (spd.get("structures") or {}).items():
                if not 1 <= int(r) <= n:
                    err.append(f"{tag}: structure reach {r} out of range")
                    continue
                for num, st in enumerate(lst or [], 1):
                    err += [
                        f"{tag} reach {r} structure {num}: {e}"
                        for e in self._check_struct(st, int(r), conn, n, sfr, tabref)
                    ]
        return err

    @staticmethod
    def _check_geo(g):
        err = []
        gtype, gcnd = int(g.get("igeotype", 0)), int(g.get("igcndop", -1))
        if gtype not in (1, 2, 3, 4, 5):
            return ["IGEOTYPE must be 1-5"]
        if gcnd not in (0, 1, 2, 3):
            return ["IGCNDOP must be 0-3"]
        need = ["gmanning"]
        need += {
            1: ["gwidth", "gbelev"],
            2: ["gwidth", "gbelev", "gsslope"],
            3: ["xsec"],
            4: ["table"],
            5: ["getextd"],
        }[gtype]
        need += {0: ["gcnd"], 1: ["glk"], 2: ["gcndln"], 3: ["glk", "gcndln"]}[gcnd]
        err += [f"missing {f}" for f in need if g.get(f) is None]
        if gtype in (1, 2) and g.get("gwidth") is not None and g["gwidth"] <= 0:
            err.append("GWIDTH must be > 0")
        if gtype == 4 and g.get("table") is not None:
            tb = _as_table(g["table"], 5, "table")
            if len(tb) < 2 or np.any(np.diff(tb[:, 0]) <= 0):
                err.append("table needs >= 2 rows with increasing ELEV")
            elif np.any(np.diff(tb[:, 1:], axis=0) < 0):
                err.append("table VOL/WETPER/SAREA/XAREA must not decrease")
        if gtype == 3 and g.get("xsec") is not None:
            if len(_as_table(g["xsec"], 2, "xsec")) < 2:
                err.append("xsec needs at least 2 points")
        return err

    @staticmethod
    def _check_struct(st, r, conn, n, sfr, tabref):
        err = []
        stype, sc = int(st.get("istrtype", -99)), int(st.get("istrconn", 0))
        atype = abs(stype)
        if atype not in STRUCT_FIELDS and atype not in (2, 4, 5):
            return [f"unknown ISTRTYPE {stype}"]
        if stype < 0 and stype != -2:
            err.append("only ISTRTYPE=-2 may be negative")
        if sc != 0 and sc not in conn.get(r, []):
            err.append(f"ISTRCONN {sc} is not connected to reach {r} (item 4b)")
        if atype in (11, 12) and sc != 0:
            err.append(f"ISTRTYPE={atype} requires ISTRCONN=0")
        if atype == 13 and sc == 0:
            err.append("ISTRTYPE=13 requires a connected reach")
        if atype == 11 and not sfr:
            err.append("ISTRTYPE=11 requires the SFR package")
        fields = {
            5: ["strcd", "strcd2", "strinv", "strinv2", "strwid", "strlen", "strman"],
            4: ["rating"],
        }.get(atype, STRUCT_FIELDS.get(atype, []))
        if stype == -2:
            fields = ["strinv"]
        err += [f"missing {f}" for f in fields if f != "istrdir" and st.get(f) is None]
        if atype == 5 and st.get("strwid", 0) < 0 and st.get("strwid2") is None:
            err.append("rectangular culvert (STRWID<0) needs strwid2")
        if atype == 4 and st.get("rating") is not None:
            if len(_as_table(st["rating"], 2, "rating")) < 2:
                err.append("rating needs at least 2 points")
        if atype in (3, 6, 7, 8, 9, 10, 13) and isinstance(st.get("strval"), str):
            err.append("STRVAL can only be TABDATAn for ISTRTYPE=12")
        tabref(st.get("strval"), "STRVAL")
        if atype in OPERABLE_TYPES:
            for f in ("cstrotyp", "cstrcrit", "strmax"):
                if st.get(f) is None:
                    err.append(f"operable structure missing {f}")
            otyp = str(st.get("cstrotyp", "")).upper()
            if atype == 1 and otyp == "FLOW":
                err.append("CSTROTYP=FLOW is not supported for ISTRTYPE=1")
            if atype != 1:
                orch = st.get("istrorch")
                if orch is None or not 1 <= int(orch) <= n:
                    err.append("ISTRORCH missing or out of range")
                elif otyp == "FLOW" and int(st.get("istroqcon", -1)) not in conn.get(
                    int(orch), []
                ):
                    err.append("ISTROQCON is not connected to ISTRORCH")
            crit, cval = st.get("cstrcrit"), st.get("cstrval")
            tabref(crit, "CSTRCRIT")
            tabref(cval, "CSTRVAL")
            if isinstance(crit, str) and "REACH" in crit.upper():
                rr = int(crit.upper().split("REACH")[1])
                if otyp != "STAGE":
                    err.append("CSTRCRIT=REACHn needs CSTROTYP=STAGE")
                if not 1 <= rr <= n:
                    err.append(f"CSTRCRIT {crit} out of range")
            if (
                isinstance(crit, str)
                and "TABDATA" in crit.upper()
                and isinstance(cval, str)
                and "TABDATA" in cval.upper()
            ):
                err.append("TABDATA cannot be used for both CSTRCRIT and CSTRVAL")
        return err

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def write_file(self, check=True):
        """Write the SWR1 input file (and DRO / time-series files if given).

        ``check=True`` runs :meth:`validate` first and raises ValueError
        listing every problem found.
        """
        nper = self.parent.nper
        if check:
            problems = self.validate()
            if problems:
                raise ValueError("SWR1 input errors:\n  - " + "\n  - ".join(problems))
        if self.reach_data is None or len(self.reach_data) != self.nreaches:
            raise ValueError("reach_data must have NREACHES rows")
        if 0 not in self.stress_period_data:
            raise ValueError("stress_period_data must define stress period 0")
        lines = [f"# {self.heading}\n"]
        w = lines.append

        # item 1a
        rec = [
            self.nreaches,
            self.iswronly,
            self.iswrcbc,
            self.iswrprgf,
            self.iswrpstg,
            self.iswrpqaq,
            self.iswrpqm,
            self.iswrpstr,
            self.iswrpfrn,
        ]
        for a in self.auxiliary:
            rec += ["AUX", a]
        if self.options:
            rec.append("SWROPTIONS")
        w(
            "# ITEM 1A: NREACHES ISWRONLY ISWRCBC ISWRPRGF ISWRPSTG ISWRPQAQ "
            "ISWRPQM ISWRPSTR ISWRPFRN [OPTIONS]\n"
        )
        w(self._line(*rec))
        # item 1b
        if self.options:
            w("# ITEM 1B: SWR1 OPTIONS\n")
            for key, val in self.options.items():
                w(self._line(key) if val is None else _line(key, int(val)))
            w("END\n")
        # item 2
        w(
            "# ITEM 2: DLENCONV TIMECONV RTINI RTMIN RTMAX RTPRN RTMULT NTMULT "
            "DMINGRAD DMINDPTH [DMAXRAI DMAXSTG DMAXINF]\n"
        )
        rec = [
            float(self.dlenconv),
            float(self.timeconv),
            float(self.rtini),
            float(self.rtmin),
            float(self.rtmax),
            float(self.rtprn),
            float(self.rtmult),
            int(self.ntmult),
            float(self.dmingrad),
            float(self.dmindpth),
        ]
        if self._adaptive:
            rec += [float(self.dmaxrai), float(self.dmaxstg), float(self.dmaxinf)]
        w(self._line(*rec))
        # item 3
        w(
            "# ITEM 3: ISOLVER NOUTER NINNER IBT TOLS TOLR TOLA DAMPSS DAMPTR "
            "IPRSWR MUTSWR [IPC] [NLEVELS DROPTOL] [IBTPRT] [PTOLR]\n"
        )
        rec = [
            int(self.isolver),
            int(self.nouter),
            int(self.ninner),
            int(self.ibt),
            float(self.tols),
            float(self.tolr),
            float(self.tola),
            float(self.dampss),
            float(self.damptr),
            int(self.iprswr),
            int(self.mutswr),
        ]
        if self.isolver > 1:
            rec.append(int(self.ipc))
            if self.ipc == 4:
                rec += [int(self.nlevels), float(self.droptol)]
        if self.ibt > 1:
            rec.append(int(self.ibtprt))
        if "USE_FRACTIONAL_TOLR" in self.options or "USE_L2NORM_TOLR" in self.options:
            rec.append(float(self.ptolr))
        w(self._line(*rec))
        # item 4a
        w("# ITEM 4A: IRCH4A IROUTETYPE IRGNUM KRCH IRCH JRCH RLEN\n")
        k, i, j = self._kij_one_based()
        rd = self.reach_data
        rows = [
            (
                int(rd.reach[n]),
                int(rd.iroutetype[n]),
                int(rd.irgnum[n]),
                int(k[n]),
                int(i[n]),
                int(j[n]),
                float(rd.rlen[n]),
            )
            for n in np.argsort(rd.reach, kind="stable")
        ]
        self._write_list(w, rows, getattr(self.reach_data, "fname", None))
        # item 4b
        w("# ITEM 4B: IRCH4B NCONN ICONN(1)...ICONN(NCONN)\n")
        for irch in range(1, self.nreaches + 1):
            conn = self.connectivity.get(irch, [])
            w(self._line(irch, len(conn), *conn))
        # items 4c/4d
        if "USE_TABFILES" in self.options:
            w("# ITEM 4C: NTABS\n")
            w(self._line(len(self.tabfiles)))
            if self.tabfiles:
                w("# ITEM 4D: ITAB CTABTYPE ITABUNIT CINTP [CTABRCH] [ITABRCH]\n")
            for t in self.tabfiles:
                ctype = t["ctabtype"].upper()
                if ctype not in TABTYPES:
                    raise ValueError(f"unknown SWR1 CTABTYPE {ctype}")
                cintp = t.get("cintp", "NONE").upper()
                if cintp not in INTERPOLATION:
                    raise ValueError(f"unknown SWR1 CINTP {cintp}")
                rec = [int(t["itab"]), ctype, int(t["itabunit"]), cintp]
                if ctype in TABTYPES_WITH_REACHES:
                    reaches = t.get("reaches", "ALL")
                    if isinstance(reaches, str):
                        rec.append(reaches.upper())
                    else:
                        reaches = [int(x) for x in reaches]
                        rec += [len(reaches)] + reaches
                w(self._line(*rec))
        # items 4e/4f
        if (
            "SAVE_SWROBSERVATIONS" in self.options
            or "SAVE_SWROBSERVATIONS_ALL" in self.options
        ):
            w("# ITEM 4E: NOBS\n")
            w(self._line(len(self.observations)))
            if self.observations:
                w("# ITEM 4F: COBSNAME COBSTYPE IOBSLOC [IOBSLOC2|IOBSLAY]\n")
            for name, otype, loc, loc2 in self.observations:
                rec = [name, otype, loc]
                if otype in OBSTYPES_WITH_LOC2:
                    rec.append(loc2)
                w(self._line(*rec))

        # stress periods
        dro_lines = []
        for kper in range(nper):
            spd = self.stress_period_data.get(kper)
            has_dro = kper in self.direct_runoff
            if spd is None and not has_dro:
                w(f"# STRESS PERIOD {kper + 1}: REUSE\n")
                w(self._line(-1, 0, 0, 0, 0, 0, 0, 0, 0))
                continue
            self._write_period(w, kper, spd or {})
            if "USE_DIRECT_RUNOFF" in self.options:
                dro_lines += self._dro_record(kper)

        with open(self.fn_path, "w") as f:
            f.writelines(lines)
        if "USE_DIRECT_RUNOFF" in self.options and self.direct_runoff:
            path = os.path.join(self.parent.model_ws, f"{self.parent.name}.swr.dro")
            with open(path, "w") as f:
                f.write("# SWR1 direct runoff file written by mfswr1\n")
                f.writelines(dro_lines)
        self._write_tabfiles()

    def _write_period(self, w, kper, spd):
        nrch = self.nreaches
        bnd = _as_table(spd.get("boundary"), 2, "boundary")
        geo_asn = _as_table(spd.get("geometry_assignment"), 3, "geometry_assignment")
        if geo_asn is not None:  # IGEONUMR < 1 means reuse: just leave it out
            geo_asn = geo_asn[geo_asn[:, 1] >= 1]
        structs = spd.get("structures") or {}
        aux = spd.get("aux")
        naux = len(self.auxiliary)
        aux = _as_table(aux, naux + 1, "aux") if aux is not None else None

        def flag(key):
            v = spd.get(key)
            if v is None:
                return 0, None
            # a plain (nrow, ncol) ndarray or a Util2d is a grid (items
            # 7b/8b/9b/14b); lists are python lists or record arrays
            if isinstance(v, Util2d) or (
                isinstance(v, np.ndarray)
                and v.dtype.names is None
                and v.shape == (self.parent.nrow, self.parent.ncol)
            ):
                return -1, v
            t = _as_table(v, 2, key)
            return len(t), t

        irdrai, rain = flag("rain")
        irdevp, evap = flag("evap")
        irdlin, lat = flag("latflow")
        irdstg, stage = flag("stage")
        irdbnd = 0 if bnd is None else len(bnd)
        irdgeo = 0 if geo_asn is None else len(geo_asn)
        irdstr = len(structs)
        irdaux = 0 if aux is None else len(aux)

        if kper == 0:
            if irdbnd != nrch:
                raise ValueError("boundary must list all reaches in stress period 0")
            if irdgeo != nrch:
                raise ValueError(
                    "geometry_assignment must list all reaches in stress period 0"
                )
            if naux and irdaux != nrch:
                raise ValueError("aux must list all reaches in stress period 0")

        w(f"# STRESS PERIOD {kper + 1}\n")
        w(
            "# ITEM 5: ITMP IRDBND IRDRAI IRDEVP IRDLIN IRDGEO IRDSTR IRDSTG "
            "IPTFLG [IRDAUX]\n"
        )
        rec = [
            1,
            irdbnd,
            irdrai,
            irdevp,
            irdlin,
            irdgeo,
            irdstr,
            irdstg,
            int(spd.get("iptflg", 0)),
        ]
        if naux:
            rec.append(irdaux)
        w(self._line(*rec))

        def ext(key):
            return getattr(spd.get(key), "fname", None)

        if irdbnd:
            w("# ITEM 6: IBNDRCH ISWRBND\n")
            self._write_list(w, [(int(r[0]), int(r[1])) for r in bnd], ext("boundary"))
        for item, nflag, data, key in (
            ("7", irdrai, rain, "rain"),
            ("8", irdevp, evap, "evap"),
        ):
            self._write_rate(w, item, nflag, data, key.upper(), ext(key))
        # item 9a has no comment skip in front of it in the Fortran
        # (SSWRLSTRD is called directly), so no comment line is written
        if irdlin > 0:
            self._write_list(w, [(int(r[0]), float(r[1])) for r in lat], ext("latflow"))
        elif irdlin < 0:
            w("# ITEM 9B: QLATFLOW2D\n")
            w(self._array_entry(lat, "QLATFLOW2D"))
        if irdgeo:
            w("# ITEM 10: IGMODRCH IGEONUMR GZSHIFT\n")
            self._write_list(
                w,
                [(int(r[0]), int(r[1]), float(r[2])) for r in geo_asn],
                ext("geometry_assignment"),
            )
            self._write_geometries(w, spd, geo_asn)
        if irdstr:
            self._write_structures(w, structs)
        if irdstg > 0:
            w("# ITEM 14A: IRCHSTG STAGE\n")
            self._write_list(w, [(int(r[0]), float(r[1])) for r in stage], ext("stage"))
        elif irdstg < 0:
            w("# ITEM 14B: STAGE2D\n")
            w(self._array_entry(stage, "STAGE2D"))
        if naux and irdaux:
            w("# ITEM 15: IRCHAUX " + " ".join(self.auxiliary) + "\n")
            self._write_list(
                w, [(int(r[0]), *[float(x) for x in r[1:]]) for r in aux], ext("aux")
            )

    def _u2d(self, data, name, dtype=np.float32):
        if isinstance(data, Util2d):
            return data
        return Util2d(
            self.parent,
            (self.parent.nrow, self.parent.ncol),
            dtype,
            _plain(data),
            name=name,
        )

    def _write_external_file(self, fname, text):
        path = os.path.join(self.parent.model_ws, fname)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def _write_list(self, w, rows, fname=None):
        """Write list rows inline, or to ``fname`` behind an OPEN/CLOSE record."""
        text = "".join(self._line(*r) for r in rows)
        if fname:
            self._write_external_file(fname, text)
            w(self._line("OPEN/CLOSE", fname))
        else:
            w(text)

    def _array_entry(self, data, name, dtype=np.float32):
        """U2DREL/U2DINT entry; SwrExternal grids go to their own file."""
        fname = getattr(data, "fname", None)
        if not fname:
            return self._u2d(data, name, dtype).get_file_entry()
        arr = np.asarray(_plain(data), dtype=dtype)
        fmt = "%d" if np.issubdtype(arr.dtype, np.integer) else "%.9G"
        text = "\n".join(" ".join(fmt % v for v in row) for row in arr) + "\n"
        self._write_external_file(fname, text)
        return self._line("OPEN/CLOSE", fname, 1, "(FREE)", -1)

    def _write_rate(self, w, item, nflag, data, label, fname=None):
        if nflag > 0:
            w(f"# ITEM {item}A: REACH {label}\n")
            self._write_list(w, [(int(r[0]), float(r[1])) for r in data], fname)
        elif nflag < 0:
            w(f"# ITEM {item}B: {label}2D\n")
            w(self._array_entry(data, f"{label}2D"))

    def _write_geometries(self, w, spd, geo_asn):
        geos = spd.get("geometries") or {}
        needed = sorted({int(g) for g in geo_asn[:, 1]})
        missing = [g for g in needed if g not in geos]
        if missing:
            raise ValueError(f"geometries missing for IGEONUM {missing}")
        for igeo in needed:
            g = geos[igeo]
            gtype, gcnd = int(g["igeotype"]), int(g["igcndop"])
            rec = [igeo, gtype, gcnd, float(g["gmanning"])]
            pts = None
            if gtype == 3:
                pts = _as_table(g["xsec"], 2, f"geometry {igeo} xsec")
            elif gtype == 4:
                pts = _as_table(g["table"], 5, f"geometry {igeo} table")
            if pts is not None:
                rec.append(len(pts))
            if gtype in (1, 2):
                rec += [float(g["gwidth"]), float(g["gbelev"])]
                if gtype == 2:
                    rec.append(float(g["gsslope"]))
            if gcnd == 0:
                rec.append(float(g["gcnd"]))
            if gcnd in (1, 3):
                rec.append(float(g["glk"]))
            if gcnd in (2, 3):
                rec.append(float(g["gcndln"]))
            if gtype == 5:
                rec.append(float(g["getextd"]))
            w(
                "# ITEM 11A: IGEONUM IGEOTYPE IGCNDOP GMANNING [NGEOPTS] "
                "[GWIDTH GBELEV [GSSLOPE]] [GCND|GLK|GCNDLN] [GETEXTD]\n"
            )
            w(self._line(*rec))
            if pts is not None:
                w(
                    "# ITEM 11B: XB ELEVB\n"
                    if gtype == 3
                    else "# ITEM 11C: ELEV VOL WETPER SAREA XAREA\n"
                )
                src = g["xsec"] if gtype == 3 else g["table"]
                self._write_list(
                    w,
                    [tuple(float(x) for x in p) for p in pts],
                    getattr(src, "fname", None),
                )

    def _write_structures(self, w, structs):
        w("# ITEM 12: ISMODRCH NSTRUCT\n")
        for irch, lst in structs.items():
            w(self._line(int(irch), -1 if lst is None else len(lst)))
        body = [
            (int(irch), k + 1, s)
            for irch, lst in structs.items()
            if lst
            for k, s in enumerate(lst)
        ]
        if not body:
            return
        w("# ITEM 13: STRUCTURE DATA\n")
        sfr = self._sfr_active
        for irch, istrnum, s in body:
            conn = int(s.get("istrconn", 0))
            stype = int(s["istrtype"])
            atype = abs(stype)
            rec = [irch, int(s.get("istrnum", istrnum)), conn, stype]
            if atype == 2:
                if stype < 0:
                    rec.append(float(s["strinv"]))
                rec.append(float(s.get("strlen", 0.0)))
                if conn > 0:
                    rec.append(float(s.get("strlen2", 0.0)))
            elif atype == 4:
                rating = _as_table(s["rating"], 2, "rating")
                rec.append(len(rating))
            elif atype == 5:
                rec += [
                    float(s["strcd"]),
                    float(s["strcd2"]),
                    float(s["strinv"]),
                    float(s["strinv2"]),
                    float(s["strwid"]),
                ]
                if s["strwid"] < 0:
                    rec.append(float(s["strwid2"]))
                rec += [
                    float(s["strlen"]),
                    float(s["strman"]),
                    int(s.get("istrdir", 0)),
                ]
            elif atype in STRUCT_FIELDS:
                for fld in STRUCT_FIELDS[atype]:
                    v = s.get(fld, 0)
                    if fld == "istrdir":
                        rec.append(int(v))
                    elif fld == "strval" and isinstance(v, str):
                        rec.append(v.upper())
                    else:
                        rec.append(float(v))
            else:
                raise ValueError(f"unknown ISTRTYPE {stype}")
            if conn == 0 and sfr:
                seg = int(s.get("isfrseg", 0))
                rec.append(seg)
                if seg > 0:
                    rec.append(int(s.get("isfrrch", 1)))
            w(self._line(*rec))
            if atype == 4:
                self._write_list(
                    w,
                    [(float(p[0]), float(p[1])) for p in rating],
                    getattr(s["rating"], "fname", None),
                )
            if atype in OPERABLE_TYPES:
                w(self._line(*self._operation_record(s, stype)))

    @staticmethod
    def _operation_record(s, stype):
        otyp = s["cstrotyp"].upper()
        if otyp not in ("STAGE", "FLOW"):
            raise ValueError("CSTROTYP must be STAGE or FLOW")
        rec = [otyp]
        if stype != 1:
            rec.append(int(s["istrorch"]))
            if otyp == "FLOW":
                rec.append(int(s["istroqcon"]))
            lo = s["cstrlo"].upper()
            if lo not in ("GE", "LT"):
                raise ValueError("CSTRLO must be GE or LT")
            rec.append(lo)
        crit = s["cstrcrit"]
        rec.append(crit.upper() if isinstance(crit, str) else float(crit))
        if stype > 1:
            rec += [float(s.get("strcritc", 0.0)), float(s.get("strrt", 0.0))]
        rec.append(float(s["strmax"]))
        if stype > 1 and s.get("cstrval"):
            rec.append(str(s["cstrval"]).upper())
        return rec

    def _dro_record(self, kper):
        d = self.direct_runoff.get(kper)
        if not d:
            return ["0\n"]
        flags = [int(d.get(k) is not None) for k in ("map", "mult", "val")]
        out = [f"# STRESS PERIOD {kper + 1}\n", self._line(1, *flags)]
        if flags[0]:
            out.append(self._array_entry(d["map"], "DROMAP2D", np.int32))
        if flags[1]:
            out.append(self._array_entry(d["mult"], "DROMULT2D"))
        if flags[2]:
            out.append(self._array_entry(d["val"], "DROVAL2D"))
        return out

    def _write_tabfiles(self):
        for t in self.tabfiles:
            data = t.get("data")
            if data is None:
                continue
            data = _as_table(data, 2, f"tabfile {t['itab']}")
            path = os.path.join(self.parent.model_ws, t["fname"])
            if int(t["itabunit"]) < 0:
                # MODFLOW-NWT opens DATA(BINARY) as ACCESS='STREAM' (openspec),
                # so the file is plain float32 (time, value) pairs, no markers
                data.astype("<f4").tofile(path)
            else:
                with open(path, "w") as f:
                    hdr = t.get("header") or {}
                    if hdr:
                        f.write(
                            " ".join(
                                f"{k.upper()} {_fmt(float(v))}" for k, v in hdr.items()
                            )
                            + "\n"
                        )
                    for tm, v in data:
                        f.write(_line(float(tm), float(v)))

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        f,
        model,
        ext_unit_dict=None,
        check=False,
        zero_based=False,
        load_tabfiles=True,
        max_line_length=MAXLINE,
    ):
        """
        Load an existing SWR1 file into a ModflowSwr1 object.

        OPEN/CLOSE lists and arrays come back as :class:`SwrExternal` so
        they are written to the same files again. ASCII and binary
        time-series files are read into ``tabfiles[i]["data"]`` (and
        ``["header"]``) unless ``load_tabfiles`` is False.
        """
        if model.verbose:
            print("loading swr1 process file...")
        if hasattr(f, "read"):
            path = f.name
            text = f.read().splitlines()
            src = _Source(text, path)
        else:
            src = _Source.from_file(f)
        rd = _Reader(src, model, ext_unit_dict)
        nper = model.nper

        # item 1a
        toks = src.next_tokens()
        ints = [_to_int(t) for t in toks[:9]]
        aux, has_opt, i = [], False, 9
        while i < len(toks):
            key = toks[i].upper()
            if key in ("AUX", "AUXILIARY") and i + 1 < len(toks):
                aux.append(toks[i + 1])
                i += 2
            elif key == "SWROPTIONS":
                has_opt = True
                i += 1
            else:
                break
        nreaches = ints[0]
        # item 1b
        options = {}
        if has_opt:
            while True:
                t = src.next_tokens()
                key = t[0].upper()
                if key == "END":
                    break
                if key not in OPTIONS:
                    continue  # unrecognised keywords are ignored by SWR1
                if key in options:
                    continue
                if any((key == b and a in options) for a, b in CONFLICTS):
                    continue
                options[key] = _to_int(t[1]) if OPTIONS[key] is not None else None
        # item 2
        t = src.next_tokens()
        kw = {
            "dlenconv": _to_float(t[0]),
            "timeconv": _to_float(t[1]),
            "rtini": _to_float(t[2]),
            "rtmin": _to_float(t[3]),
            "rtmax": _to_float(t[4]),
            "rtprn": _to_float(t[5]),
            "rtmult": _to_float(t[6]),
            "ntmult": _to_int(t[7]),
            "dmingrad": _to_float(t[8]),
            "dmindpth": _to_float(t[9]),
        }
        if _is_adaptive(kw["rtmin"], kw["rtmax"], kw["rtmult"], model):
            kw.update(
                dmaxrai=_to_float(t[10]),
                dmaxstg=_to_float(t[11]),
                dmaxinf=_to_float(t[12]),
            )
        # item 3
        t = src.next_tokens()
        kw.update(
            isolver=_to_int(t[0]),
            nouter=_to_int(t[1]),
            ninner=_to_int(t[2]),
            ibt=_to_int(t[3]),
            tols=_to_float(t[4]),
            tolr=_to_float(t[5]),
            tola=_to_float(t[6]),
            dampss=_to_float(t[7]),
            damptr=_to_float(t[8]),
            iprswr=_to_int(t[9]),
            mutswr=_to_int(t[10]),
        )
        k = 11
        if kw["isolver"] > 1:
            kw["ipc"] = _to_int(t[k])
            k += 1
            if kw["ipc"] == 4:
                kw["nlevels"], kw["droptol"] = _to_int(t[k]), _to_float(t[k + 1])
                k += 2
        if kw["ibt"] > 1:
            kw["ibtprt"] = _to_int(t[k])
            k += 1
        if "USE_FRACTIONAL_TOLR" in options or "USE_L2NORM_TOLR" in options:
            kw["ptolr"] = _to_float(t[k])
        # item 4a
        arr, fname = rd.read_list(src, nreaches, 7, keep_external=True)
        reach_data = cls._reach_array(arr)
        if zero_based:
            reach_data.krch[reach_data.krch > 0] -= 1
            reach_data.irch -= 1
            reach_data.jrch -= 1
        if fname:
            reach_data = SwrExternal(reach_data, fname)
        # item 4b
        sub = rd.block_source(src)
        connectivity = {}
        for _ in range(nreaches):
            t = sub.next_tokens()
            irch, nconn = _to_int(t[0]), _to_int(t[1])
            connectivity[irch] = [_to_int(x) for x in t[2 : 2 + nconn]]
        # items 4c/4d
        tabfiles = []
        if "USE_TABFILES" in options:
            ntabs = _to_int(src.next_tokens()[0])
            for _ in range(max(ntabs, 0)):
                t = src.next_tokens()
                entry = {
                    "itab": _to_int(t[0]),
                    "ctabtype": t[1].upper(),
                    "itabunit": _to_int(t[2]),
                    "cintp": t[3].upper(),
                }
                if entry["ctabtype"] in TABTYPES_WITH_REACHES:
                    if t[4].upper() == "ALL":
                        entry["reaches"] = "ALL"
                    else:
                        n = _to_int(t[4])
                        vals = t[5:]
                        while len(vals) < n:  # list-directed READ spans lines
                            vals += src.next_tokens()
                        entry["reaches"] = [_to_int(x) for x in vals[:n]]
                iu = abs(entry["itabunit"])
                if ext_unit_dict and iu in ext_unit_dict:
                    path = ext_unit_dict[iu].filename
                    entry["fname"] = os.path.relpath(path, model.model_ws)
                    if load_tabfiles:
                        entry["data"], entry["header"] = cls._read_tabfile(
                            path, entry["itabunit"] < 0
                        )
                tabfiles.append(entry)
        # items 4e/4f
        observations = []
        if "SAVE_SWROBSERVATIONS" in options or "SAVE_SWROBSERVATIONS_ALL" in options:
            nobs = _to_int(src.next_tokens()[0])
            for _ in range(max(nobs, 0)):
                t = src.next_tokens()
                otype = t[1].upper()
                loc2 = _to_int(t[3]) if otype in OBSTYPES_WITH_LOC2 else 0
                observations.append((t[0], otype, _to_int(t[2]), loc2))

        # stress periods
        sfr = model.get_package("SFR") is not None or bool(
            ext_unit_dict and any(v.filetype == "SFR" for v in ext_unit_dict.values())
        )
        dro_src = None
        if "USE_DIRECT_RUNOFF" in options and ext_unit_dict:
            iu = options["USE_DIRECT_RUNOFF"]
            if iu in ext_unit_dict:
                dro_src = _Source.from_file(ext_unit_dict[iu].filename)
                if model.verbose:
                    print(f"   loading SWR1 direct runoff from unit {iu}")
        spd_all, dro_all = {}, {}
        naux = len(aux)
        for kper in range(nper):
            try:
                t = src.next_tokens()
            except EOFError:
                break
            if _to_int(t[0]) < 1:
                continue
            flags = [_to_int(x) for x in t[1:9]]
            irdbnd, irdrai, irdevp, irdlin, irdgeo, irdstr, irdstg, iptflg = flags
            irdaux = _to_int(t[9]) if (naux and len(t) > 9) else 0
            spd = {"iptflg": iptflg}
            if irdbnd > 0:
                spd["boundary"] = rd.read_rec(
                    src, irdbnd, ("reach", "iswrbnd"), ("reach", "iswrbnd")
                )
            for key, n in (("rain", irdrai), ("evap", irdevp), ("latflow", irdlin)):
                if n > 0:
                    spd[key] = rd.read_rec(src, n, ("reach", key), ("reach",))
                elif n < 0:
                    spd[key] = rd.read_array(src, np.float32, key.upper() + "2D")
            if irdgeo > 0:
                asn_rec = rd.read_rec(
                    src, irdgeo, ("reach", "igeonumr", "gzshift"), ("reach", "igeonumr")
                )
                spd["geometry_assignment"] = asn_rec
                asn = _as_table(asn_rec, 3, "geometry_assignment")
                pos = asn[:, 1] >= 1
                nuniq = len(np.unique(asn[pos, 1])) + int((~pos).any())
                sub = rd.block_source(src)
                geos = {}
                for _ in range(nuniq):
                    g = cls._read_geometry(sub, rd)
                    geos[g.pop("igeonum")] = g
                spd["geometries"] = geos
            if irdstr > 0:
                spd["structures"] = cls._read_structures(src, rd, irdstr, sfr)
            if irdstg > 0:
                spd["stage"] = rd.read_rec(src, irdstg, ("reach", "stage"), ("reach",))
            elif irdstg < 0:
                spd["stage"] = rd.read_array(src, np.float32, "STAGE2D")
            if naux and irdaux > 0:
                spd["aux"] = rd.read_rec(src, irdaux, ["reach"] + list(aux), ("reach",))
            spd_all[kper] = spd
            if dro_src is not None:
                d = cls._read_dro(dro_src, rd)
                if d:
                    dro_all[kper] = d

        # unit numbers / file names from the name file
        unitnumber, filenames = None, None
        output_filenames = {}
        if ext_unit_dict is not None:
            unitnumber, fname = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )
            filenames = [fname]
            units = [ints[2], *ints[3:8]]
            units += [v for v in options.values() if v is not None]
            for u in units:
                if u and abs(u) in ext_unit_dict:
                    output_filenames[u] = os.path.basename(
                        ext_unit_dict[abs(u)].filename
                    )
                    model.add_pop_key_list(abs(u))
            for tf in tabfiles:
                if abs(tf["itabunit"]) in ext_unit_dict:
                    model.add_pop_key_list(abs(tf["itabunit"]))

        obj = cls(
            model,
            nreaches=nreaches,
            iswronly=ints[1],
            iswrcbc=ints[2],
            iswrprgf=ints[3],
            iswrpstg=ints[4],
            iswrpqaq=ints[5],
            iswrpqm=ints[6],
            iswrpstr=ints[7],
            iswrpfrn=ints[8],
            auxiliary=aux,
            options=options,
            reach_data=reach_data,
            connectivity=connectivity,
            tabfiles=tabfiles,
            observations=observations,
            stress_period_data=spd_all,
            direct_runoff=dro_all,
            output_filenames=output_filenames,
            unitnumber=unitnumber,
            filenames=filenames,
            zero_based=zero_based,
            max_line_length=max_line_length,
            **kw,
        )
        return obj

    @staticmethod
    def _read_tabfile(path, binary):
        """Time series as SSWR_RDTABDATA reads it: (N x 2 data, header dict)."""
        if binary:
            return np.fromfile(path, "<f4").reshape(-1, 2).astype(float), {}
        src = _Source.from_file(path)
        header = {}
        toks = src.peek_tokens()
        keys = ("TIMEOFFSET", "TIMESCALE", "OFFSET", "SCALE")
        if toks and toks[0].upper() in keys:
            src.next_line()
            for k in range(0, len(toks) - 1, 2):
                if toks[k].upper() not in keys:
                    break
                header[toks[k].upper()] = _to_float(toks[k + 1])
        rows = []
        while True:
            t = src.peek_tokens()
            if not t:
                break
            src.next_line()
            rows.append((_to_float(t[0]), _to_float(t[1])))
        return np.array(rows), header

    @staticmethod
    def _read_geometry(src, rd):
        t = src.next_tokens()
        g = {
            "igeonum": _to_int(t[0]),
            "igeotype": _to_int(t[1]),
            "igcndop": _to_int(t[2]),
            "gmanning": _to_float(t[3]),
        }
        k, gtype, gcnd = 4, g["igeotype"], g["igcndop"]
        npts = 0
        if gtype in (3, 4):
            npts = _to_int(t[k])
            k += 1
        if gtype in (1, 2):
            g["gwidth"], g["gbelev"] = _to_float(t[k]), _to_float(t[k + 1])
            k += 2
            if gtype == 2:
                g["gsslope"] = _to_float(t[k])
                k += 1
        if gcnd == 0:
            g["gcnd"] = _to_float(t[k])
            k += 1
        if gcnd in (1, 3):
            g["glk"] = _to_float(t[k])
            k += 1
        if gcnd in (2, 3):
            g["gcndln"] = _to_float(t[k])
            k += 1
        if gtype == 5:
            g["getextd"] = _to_float(t[k])
            k += 1
        if gtype in (3, 4):
            arr, fname = rd.read_list(
                src, npts, 2 if gtype == 3 else 5, keep_external=True
            )
            g["xsec" if gtype == 3 else "table"] = (
                SwrExternal(arr, fname) if fname else arr
            )
        return g

    @staticmethod
    def _read_structures(src, rd, irdstr, sfr):
        asn = rd.read_list(src, irdstr, 2)  # item 12 is rebuilt from the dict
        structs, total = {}, 0
        for irch, n in asn:
            irch, n = int(irch), int(n)
            if n > 0:
                structs[irch] = [None] * n
                total += n
            elif n == 0:
                structs[irch] = []
            else:
                structs[irch] = None
        if total == 0:
            return structs
        sub = rd.block_source(src)
        for _ in range(total):
            t = sub.next_tokens()
            irch, num = _to_int(t[0]), _to_int(t[1])
            conn, stype = _to_int(t[2]), _to_int(t[3])
            s = {"istrnum": num, "istrconn": conn, "istrtype": stype}
            atype, k = abs(stype), 4

            def nxt_f():
                nonlocal k
                v = _to_float(t[k])
                k += 1
                return v

            def opt_f():
                nonlocal k
                if k < len(t):
                    try:
                        v = _to_float(t[k])
                    except ValueError:
                        return 0.0
                    k += 1
                    return v
                return 0.0

            if atype == 2:
                if stype < 0:
                    s["strinv"] = nxt_f()
                s["strlen"] = opt_f()
                if conn > 0:
                    s["strlen2"] = opt_f()
            elif atype == 4:
                npts = _to_int(t[k])
                k += 1
            elif atype == 5:
                for fld in ("strcd", "strcd2", "strinv", "strinv2", "strwid"):
                    s[fld] = nxt_f()
                if s["strwid"] < 0:
                    s["strwid2"] = nxt_f()
                s["strlen"], s["strman"] = nxt_f(), nxt_f()
                s["istrdir"] = _to_int(t[k])
                k += 1
            elif atype in STRUCT_FIELDS:
                for fld in STRUCT_FIELDS[atype]:
                    if fld == "istrdir":
                        s[fld] = _to_int(t[k])
                        k += 1
                    elif fld == "strval" and "TABDATA" in t[k].upper():
                        s[fld] = t[k].upper()
                        k += 1
                    else:
                        s[fld] = nxt_f()
            if conn == 0 and sfr:
                s["isfrseg"] = _to_int(t[k]) if k < len(t) else 0
                k += 1
                if s["isfrseg"] > 0:
                    s["isfrrch"] = _to_int(t[k])
                    k += 1
            if atype == 4:
                arr, fname = rd.read_list(sub, npts, 2, keep_external=True)
                s["rating"] = SwrExternal(arr, fname) if fname else arr
            if atype in OPERABLE_TYPES:
                o = sub.next_tokens()
                s["cstrotyp"] = o[0].upper()
                j = 1
                if stype != 1:
                    s["istrorch"] = _to_int(o[j])
                    j += 1
                    if s["cstrotyp"] == "FLOW":
                        s["istroqcon"] = _to_int(o[j])
                        j += 1
                    s["cstrlo"] = o[j].upper()
                    j += 1
                crit = o[j]
                j += 1
                cu = crit.upper()
                s["cstrcrit"] = (
                    cu if ("TABDATA" in cu or "REACH" in cu) else _to_float(crit)
                )
                if stype > 1:
                    s["strcritc"], s["strrt"] = _to_float(o[j]), _to_float(o[j + 1])
                    j += 2
                s["strmax"] = _to_float(o[j])
                j += 1
                if stype > 1 and j < len(o) and "TABDATA" in o[j].upper():
                    s["cstrval"] = o[j].upper()
            structs[irch][num - 1] = s
        return structs

    @staticmethod
    def _read_dro(src, rd):
        t = src.next_tokens()
        if _to_int(t[0]) < 1:
            return None
        imap, imult, ival = (_to_int(x) for x in t[1:4])
        d = {}
        if imap > 0:
            d["map"] = rd.read_array(src, np.int32, "DROMAP2D")
        if imult > 0:
            d["mult"] = rd.read_array(src, np.float32, "DROMULT2D")
        if ival > 0:
            d["val"] = rd.read_array(src, np.float32, "DROVAL2D")
        return d

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def output_file(self, kind):
        """Path of an output file: 'rgflow', 'stage', 'qaq', 'qm', 'str'."""
        unit = {
            "rgflow": self.iswrprgf,
            "stage": self.iswrpstg,
            "qaq": self.iswrpqaq,
            "qm": self.iswrpqm,
            "str": self.iswrpstr,
        }[kind]
        if not unit:
            return None
        m = self.parent
        if abs(unit) in m.output_units:
            fn = m.output_fnames[m.output_units.index(abs(unit))]
            return os.path.join(m.model_ws, fn)
        return None
