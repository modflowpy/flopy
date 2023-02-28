# pylint: disable=E1101
"""
Mfusgcln module.

Contains the MfUsgCln class. Note that the user can
access the MfUsgCln class as `flopy.mfusg.MfUsgCln`.

Compatible with USG-Transport Version 1.7.0. which can be downloade from
https://www.gsi-net.com/en/software/free-software/modflow-usg.html

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
Panday, S., 2021; USG-Transport Version 1.7.0: The Block-Centered Transport
Process for MODFLOW-USG, GSI Environmental, March, 2021

Panday, Sorab, Langevin, C.D., Niswonger, R.G., Ibaraki, Motomu, and Hughes,
J.D., 2013, MODFLOW–USG version 1: An unstructured grid version of MODFLOW
for simulating groundwater flow and tightly coupled processes using a control
volume finite-difference formulation: U.S. Geological Survey Techniques and
Methods, book 6, chap. A45, 66 p.
"""
import numpy as np

from ..pakbase import Package
from ..utils import Util2d
from ..utils.utils_def import get_open_file_object
from .cln_dtypes import MfUsgClnDtypes
from .mfusg import MfUsg, fmt_string


class MfUsgCln(Package):
    """Connected Linear Network (CLN) Package class for MODFLOW-USG.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflowusg.mf.Modflow`) to which
        this package will be added.
    ncln : int
        is a flag or the number of CLN segments. If NCLN = 0, this flag
        indicates that the CLN domain connectivity is input in a general IA-JA
        manner as is used for the GWF Process.If NCLN > 0, linear CLN segments
        (for instance multi-aquifer wells) or simple CLN networks are simulated
        and NCLN is the total number of CLN segments in the domain.
    iclnnds : int
        is a flag or number of CLN-nodes simulated in the model. Multiple
        CLN-nodes constitute a segment.If ICLNNDS < 0, the CLN-nodes are
        ordered in a sequential manner from the first CLN node to the last
        CLN node. Therefore, only linear CLN segments are simulated since a
        CLN segment does not share any of its nodes with another CLN segment.
        If ICLNNDS > 0, CLN networks can be simulated and ICLNNDS is
        the total number of CLN-nodes simulated by the model (NCLNNDS). CLN
        nodes can be shared among CLN segments and therefore, the CLN-nodal
        connectivity for the network is also required as input.
    nndcln : list of int
        is the number of CLN-nodes that are associated with each CLN segment.
        Only read if NCLN > 0. If ICLNNDS < 0, sum of nndcln is the total number
        of CLN-nodes (NCLNNDS)
    clncon : list of list
        are the CLN-node numbers associated with each CLN segment. Only read
        if NCLN > 0 and ICLNNDS > 0. It is read NCLN times, once for each CLN
        segment. The number of entries for each sublist is the number of CLN
        cells (NNDCLN) associated with each CLN segment
    nja_cln : int
        is the total number of connections of the CLN domain. NJA_CLN is used
        to dimension the sparse matrix in a compressed row storage format.
    iac_cln : list of int
        is a matrix indicating the number of connections plus 1 for each CLN
        node to another CLN node. Note that the IAC_CLN array is only supplied
        for the CLN cells; the IAC_CLN array is internally expanded to include
        other domains if present in a simulation. sum(IAC)=NJAG
    ja_cln : list of list
        is a list of CLN cell number (n) followed by its connecting CLN cell
        numbers (m) for each of the m CLN cells connected to CLN cell n. This
        list is sequentially provided for the first to the last CLN cell.
        Note that the cell and its connections are only supplied for the CLN
        cells and their connections to the other CLN cells using the local CLN
        cell numbers.
    node_prop : matrix
        [IFNO IFTYP IFDIR FLENG FELEV FANGLE IFLIN ICCWADI X1 Y1 Z1 X2 Y2 Z2]
        is a table of the node properties. Total rows equal the total number
        of CLN-nodes (NCLNNDS). The first 6 fields is required for running
        model. Rest of fields have default value of 0.
    nclngwc : int
        is the number of CLN to porous-medium grid-block connections present
        in the model. A CLN node need not be connected to any groundwater node.
        Conversely, a CLN node may be connected to multiple groundwater nodes,
        or multiple CLN nodes may be connected to the same porous medium mode.
    cln_gwc : matrix
        unstructured: [IFNOD IGWNOD IFCON FSKIN FLENGW FANISO ICGWADI]
        structured: [IFNOD IGWLAY IGWROW IGWFCOL IFCON FSKIN FLENGW FANISO
                     ICGWADI]
        is a table define connections between CLN nodes and groundwater cells.
        Total rows of the table equals nclngwc.
    nconduityp : int
        is the number of circular conduit-geometry types.
    cln_circ :
        [ICONDUITYP FRAD CONDUITK TCOND TTHK TCFLUID TCONV]
        is a table define the circular conduit properties. Total rows of the
        table equals nconduityp. Last 4 fields only needed for heat transport
        simulation.
    ibound : 1-D array
        is the boundary array for CLN-nodes. Length equal NCLNNDS
    strt : 1-D array
        is initial head at the beginning of the simulation in CLN nodes.
        Length equal NCLNNDS
    transient : bool
        if there is transient IBOUND for each stress period
    printiaja : bool
        whether to print IA_CLN and JA_CLN to listing file
    nrectyp : int
        is the number of rectangular conduit-geometry types.
    cln_rect : rectangular fracture properties
        [IRECTYP FLENGTH FHEIGHT CONDUITK TCOND TTHK TCFLUID TCONV]
        is read for each rectangular conduit.  Total rows of the table equals
        nrectyp. Last 4 fields only needed for heat transport simulation.
    bhe : bool
        is a flag indicating that bhe details are also included in a heat transport
        model. Specifically, the thermal conductance and bhe tube thickness are
        included in transfer of heat between groundwater and CLN cells along with
        the heat conductivity of the bhe fluid and the convective heat transfer
        coefficient.
    grav : float
        is the gravitational acceleration constant in model simulation units.
        The value of the constant follows the keyword GRAVITY. Note that the
        constant value is 9.81 m/s2 in SI units; 32.2 ft/s2 in fps units.
    visk : float
        is the kinematic viscosity of water in model simulation units [L2/T].
        The value of kinematic viscosity follows the keyword VISCOSITY. Note
        that the constant value is 1.787 x 10-6 m2/s in SI units;
        1.924 x 10-5 ft2/s in fps units.
    extension : list of strings
        (default is ['cln','clncb','clnhd','clndd','clnib','clncn','clnmb']).
    unitnumber : list of int
        File unit number for the package and the output files.
        (default is [71, 0, 0, 0, 0, 0, 0] ).
    filenames : list of str
        Filenames to use for the package and the output files. If filenames
        = None the package name will be created using the model name and package
        extensions.

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
    >>> ml = flopy.mfusg.MfUsg()
    >>> node_prop = [[1,1,0,10.0,-110.0,1.57,0,0],[2,1,0,10.0,-130.0,1.57,0,0]]
    >>> cln_gwc = [[1,1,50,50,0,0,10.0,1.0,0],[2,2,50,50,0,0,10.0,1.0,0]]
    >>> cln = flopy.mfusg.MfUsgCln(ml, ncln=1, iclnnds=-1, nndcln=2,
            nclngwc = 2, node_prop =node_prop, cln_gwc =cln_gwc)"""

    def __init__(
        self,
        model,
        ncln=None,  # number of CLNs
        iclnnds=None,  # number of nodes
        nndcln=None,  # number of nodes in each CLN segments
        clncon=None,  # node IDs in each CLN segments
        nja_cln=None,  # total number of node-node connections (NJAG)
        iac_cln=None,  # number of connections for each node (sum(IAC)=NJAG
        ja_cln=None,  # node connections
        node_prop=None,  # node properties
        nclngwc=None,  # number of CLN-GW connections
        cln_gwc=None,  # CLN-GW connections
        nconduityp=1,  # number of circular conduit types
        cln_circ=None,  # circular conduit properties
        ibound=1,  # boundary condition types
        strt=1.0,  # initial head in CLN cells
        transient=False,  # OPTIONS: transient IBOUND for each stress period
        printiaja=False,  # OPTIONS: print IA_CLN and JA_CLN to listing file
        nrectyp=0,  # OPTIONS2: number of rectangular fracture types
        cln_rect=None,  # rectangular fracture properties
        bhe=False,  # OPTIONS2: borehole heat exchanger (BHE)
        grav=None,  # OPTIONS2: gravitational acceleration constant
        visk=None,  # OPTIONS2: kinematic viscosity of water
        extension=(
            "cln",
            "clncb",
            "clnhd",
            "clndd",
            "clnib",
            "clncn",
            "clnmb",
        ),
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
            self.unitnumber = self._defaultunit()
        elif isinstance(unitnumber, list):
            if len(unitnumber) < 7:
                for idx in range(len(unitnumber), 7):
                    unitnumber.append(0)

        # set filenames
        filenames = self._prepare_filenames(filenames, num=7)

        # Call ancestor's init to set self.parent, extension, name and unit number
        super().__init__(
            model,
            extension=list(extension),
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames,
        )

        self._generate_heading()

        # Options
        self.transient = transient
        self.printiaja = printiaja

        for idx, attr in enumerate(extension[1:]):
            setattr(self, f"i{attr}", int(unitnumber[idx + 1]))
            if getattr(self, f"i{attr}") > 0:
                model.add_output_file(
                    getattr(self, f"i{attr}"),
                    fname=filenames[idx + 1],
                    extension=attr,
                    binflag=True,
                    package=self._ftype(),
                )

        # Define CLN networks and connections
        self.ncln = ncln
        self.iclnnds = iclnnds
        self.nndcln = nndcln
        self.clncon = clncon
        self.iac_cln = iac_cln
        self.nja_cln = nja_cln
        self.ja_cln = ja_cln
        self._define_cln_networks(model)

        # Define CLN node properties
        if node_prop is None:
            raise Exception("mfcln: Node properties must be provided")

        if len(node_prop) != self.nclnnds:
            raise Exception(
                "mfcln: Length of Node properties must equal number of nodes"
            )

        self.node_prop = self._make_recarray(
            node_prop, dtype=MfUsgClnDtypes.get_clnnode_dtype()
        )

        # Define CLN groundwater connections
        if nclngwc is None:
            raise Exception("mfcln: Number of CLN-GW connections not defined")
        self.nclngwc = nclngwc

        if cln_gwc is None:
            raise Exception("mfcln: CLN-GW connections not provided")

        if len(cln_gwc) != nclngwc:
            raise Exception(
                "mfcln: Number of CLN-GW connections not equal to nclngwc"
            )

        structured = self.parent.structured

        self.cln_gwc = self._make_recarray(
            cln_gwc, dtype=MfUsgClnDtypes.get_gwconn_dtype(structured)
        )

        # Define CLN geometry types
        self.nconduityp = nconduityp
        self.cln_circ = cln_circ
        self.nrectyp = nrectyp
        self.cln_rect = cln_rect
        self.bhe = bhe
        self.grav = grav
        self.visk = visk

        self._define_cln_geometries()

        # Define CLN ibound and initial heads properties
        self.ibound = Util2d(
            model,
            (self.nclnnds,),
            np.int32,
            ibound,
            name="ibound",
            locat=self.unit_number[0],
        )

        self.strt = Util2d(
            model,
            (self.nclnnds,),
            np.float32,
            strt,
            name="strt",
            locat=self.unit_number[0],
        )

        self.parent.add_package(self)

    @staticmethod
    def _get_default_extension():
        """Gets default package file extensions."""
        return [
            "cln",
            "clncb",
            "clnhd",
            "clndd",
            "clnib",
            "clncn",
            "clnmb",
        ]

    def _define_cln_networks(self, model):
        """Initialises CLN networks."""
        if self.ncln is None:
            raise Exception("mfcln: CLN network not defined")

        if self.ncln < 0:
            raise Exception(
                "mfcln: negative number of CLN segments in CLN package"
            )

        if self.ncln > 0:  # Linear CLN segments
            if self.nndcln is None:
                raise Exception(
                    "mfcln: number of nodes for each CLN segment must be "
                    "provided"
                )
            self.nndcln = Util2d(
                model,
                (self.ncln,),
                np.int32,
                self.nndcln,
                name="nndcln",
                locat=self.unit_number[0],
            )

            # consecutive node number. No connection between segments
            if self.iclnnds < 0:
                self.nclnnds = self.nndcln.array.sum()
                self.nodeno = np.array(range(self.nclnnds), dtype=int) + 1
            # Node number provided for each segment to simulate CLN networks
            elif self.iclnnds > 0:
                self.nclnnds = self.iclnnds
                self.nodeno = (
                    np.asarray(set(self.clncon), dtype=object) + 1
                )  # can be jagged
            else:
                raise Exception("mfcln: Node number = 0")

        elif self.ncln == 0:  # CLN network defined by IA-JA connection matrix
            if self.iclnnds <= 0:
                raise Exception("mfcln: Negative or zero number of nodes")

            self.nclnnds = self.iclnnds

            self.nodeno = np.array(range(self.nclnnds), dtype=int) + 1

            if self.iac_cln is None:
                raise Exception("mfcln: iac_cln must be provided")
            self.iac_cln = Util2d(
                model,
                (self.nclnnds,),
                np.int32,
                self.iac_cln,
                name="iac_cln",
                locat=self.unit_number[0],
            )

            msg = "mfcln: The sum of iac_cln must equal nja_cln."
            assert self.iac_cln.array.sum() == self.nja_cln, msg

            if self.ja_cln is None:
                raise Exception("mfcln: ja_cln must be provided")
            if abs(self.ja_cln[0]) != 1:
                raise Exception(
                    "mfcln: first ja_cln entry (node 1) is not 1 or -1."
                )
            self.ja_cln = Util2d(
                model,
                (self.nja_cln,),
                np.int32,
                self.ja_cln,
                name="ja_cln",
                locat=self.unit_number[0],
            )

    def _define_cln_geometries(self):
        """Initialises CLN geometry types."""
        # Circular conduit geometry types
        if self.nconduityp <= 0 or self.cln_circ is None:
            raise Exception(
                "mfcln: Circular conduit properties must be provided"
            )

        if len(self.cln_circ) != self.nconduityp:
            raise Exception(
                "mfcln: Number of circular properties not equal nconduityp"
            )

        self.cln_circ = self._make_recarray(
            self.cln_circ, dtype=MfUsgClnDtypes.get_clncirc_dtype(self.bhe)
        )

        # Rectangular conduit geometry types
        if self.nrectyp > 0:
            if len(self.cln_rect) != self.nconduityp:
                raise Exception(
                    "mfcln: Number of rectangular properties not equal nrectyp"
                )
            self.cln_rect = self._make_recarray(
                self.cln_rect, dtype=MfUsgClnDtypes.get_clnrect_dtype(self.bhe)
            )

    @property
    def cln_nodes(self):
        """Returns the total number of CLN nodes."""

        return self.nclnnds

    def write_file(self, f=None, check=False):
        """
        Write the package file.

        Parameters
        ----------
        f : filename or file handle
            File to write to.

        Returns
        -------
        None

        """
        if f is None:
            f = self.fn_path
        f_cln = get_open_file_object(f, "w")

        if check:
            print("Warning: mfcln package check not yet implemented.")

        f_cln.write(f"{self.heading}\n")

        # write items 0 and 1
        self._write_items_0_1(f_cln)

        if self.ncln > 0:
            f_cln.write(self.nndcln.get_file_entry())
            if self.iclnnds > 0:
                for icln in range(self.ncln):
                    f_cln.write(self.clncon[icln])
        elif self.ncln == 0:
            f_cln.write(f" {self.nja_cln:9d}\n")
            f_cln.write(self.iac_cln.get_file_entry())
            f_cln.write(self.ja_cln.get_file_entry())

        np.savetxt(
            f_cln, self.node_prop, fmt=fmt_string(self.node_prop), delimiter=""
        )

        np.savetxt(
            f_cln, self.cln_gwc, fmt=fmt_string(self.cln_gwc), delimiter=""
        )

        if self.nconduityp > 0:
            np.savetxt(
                f_cln,
                self.cln_circ,
                fmt=fmt_string(self.cln_circ),
                delimiter="",
            )

        if self.nrectyp > 0:
            np.savetxt(
                f_cln,
                self.cln_rect,
                fmt=fmt_string(self.cln_rect),
                delimiter="",
            )

        f_cln.write(self.ibound.get_file_entry())
        f_cln.write(self.strt.get_file_entry())

        f_cln.close()

    def _write_items_0_1(self, f_cln):
        """Writes cln items 0 and 1."""
        if self.transient or self.printiaja:
            f_cln.write("OPTIONS   ")
            if self.transient:
                f_cln.write("TRANSIENT ")
            if self.printiaja:
                f_cln.write("PRINTIAJA ")
            f_cln.write("\n")

        f_cln.write(
            f" {self.ncln:9d} {self.iclnnds:9d} {self.iclncb:9d}"
            f" {self.iclnhd:9d} {self.iclndd:9d} {self.iclnib:9d}"
            f" {self.nclngwc:9d} {self.nconduityp:9d}"
        )

        if self.nrectyp > 0:
            f_cln.write(f"RECTANGULAR {self.nrectyp:d}")
        if self.bhe:
            f_cln.write("BHEDETAIL ")
        if self.iclncn != 0:
            f_cln.write(f"SAVECLNCON {self.iclncn:d}")
        if self.iclnmb != 0:
            f_cln.write(f"SAVECLNMAS {self.iclnmb:d}")
        if self.grav is not None:
            f_cln.write(f"GRAVITY {self.grav:f}")
        if self.visk is not None:
            f_cln.write(f"VISCOSITY {self.visk:f}")
        f_cln.write("\n")

    @classmethod
    def load(cls, f, model, pak_type="cln", ext_unit_dict=None, **kwargs):
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
        cln : MfUsgCln object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> cln = flopy.mfusg.MfUsgCln.load('test.cln', m)
        """
        msg = (
            "Model object must be of type flopy.mfusg.MfUsg\n"
            f"but received type: {type(model)}."
        )
        assert isinstance(model, MfUsg), msg

        if model.verbose:
            print("loading CLN package file...\n")

        if not hasattr(f, "read"):
            filename = f
            f = open(filename, "r")

        # Items 0 and 1
        (
            transient,
            printiaja,
            ncln,
            iclnnds,
            iclncb,
            iclnhd,
            iclndd,
            iclnib,
            nclngwc,
            nconduityp,
            nrectyp,
            cln_rect,
            bhe,
            iclncn,
            iclnmb,
            grav,
            visk,
        ) = cls._load_items_0_1(f, model)

        # Items 3, or 4/5/6
        (
            nndcln,
            clncon,
            nja_cln,
            iac_cln,
            ja_cln,
            nclnnds,
        ) = cls._load_items_3to6(f, model, ncln, iclnnds, ext_unit_dict)

        if model.verbose:
            print("  Reading node_prop...")
        node_prop = cls._read_prop(f, nclnnds)

        if model.verbose:
            print("   Reading cln_gwc...")
        cln_gwc = cls._read_prop(f, nclngwc)
        if model.verbose:
            print("   Reading cln_circ...")
        cln_circ = cls._read_prop(f, nconduityp)

        cln_rect = None
        if nrectyp > 0:
            if model.verbose:
                print("   Reading cln_rect...")
            cln_rect = cls._read_prop(f, nrectyp)

        if model.verbose:
            print("   Reading ibound...")
        ibound = Util2d.load(
            f, model, (nclnnds,), np.int32, "ibound", ext_unit_dict
        )

        if model.verbose:
            print("   Reading strt...")
        strt = Util2d.load(
            f, model, (nclnnds,), np.float32, "strt", ext_unit_dict
        )

        if hasattr(f, "read"):
            f.close()

        # set package unit number
        # reset unit numbers
        unitnumber = MfUsgCln._defaultunit()
        filenames = [None] * 7
        if ext_unit_dict is not None:
            unitnumber[0], filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=cls._ftype()
            )
            file_unit_items = [iclncb, iclnhd, iclndd, iclnib, iclncn, iclnmb]
            funcs = [abs] + [int] * 3 + [abs] * 2
            for idx, (item, func) in enumerate(zip(file_unit_items, funcs)):
                if item > 0:
                    (
                        unitnumber[idx + 1],
                        filenames[idx + 1],
                    ) = model.get_ext_dict_attr(ext_unit_dict, unit=func(item))
                    model.add_pop_key_list(func(item))

        # create dis object instance
        cln = cls(
            model,
            ncln=ncln,
            iclnnds=iclnnds,
            nndcln=nndcln,
            clncon=clncon,
            nja_cln=nja_cln,
            iac_cln=iac_cln,
            ja_cln=ja_cln,
            node_prop=node_prop,
            nclngwc=nclngwc,
            cln_gwc=cln_gwc,
            nconduityp=nconduityp,
            cln_circ=cln_circ,
            ibound=ibound,
            strt=strt,
            transient=transient,
            printiaja=printiaja,
            nrectyp=nrectyp,
            cln_rect=cln_rect,
            grav=grav,
            visk=visk,
            bhe=bhe,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        # return dis object instance
        return cln

    @staticmethod
    def _load_items_0_1(f_obj, model):
        """Loads items 0 and 1 from filehandle f."""
        # Options
        transient = False
        printiaja = False
        line = f_obj.readline().upper()
        while line.find("#") >= 0:
            line = f_obj.readline().upper()
        if line.startswith("OPTIONS"):
            line_text = line.strip().split()
            transient = bool("TRANSIENT" in line_text)
            printiaja = bool("PRINTIAJA" in line_text)
            line = f_obj.readline().upper()

        line_text = line.strip().split()

        line_text[:8] = [int(item) for item in line_text[:8]]
        (
            ncln,
            iclnnds,
            iclncb,
            iclnhd,
            iclndd,
            iclnib,
            nclngwc,
            nconduityp,
        ) = line_text[:8]

        # Options keywords
        nrectyp = 0
        cln_rect = None
        if "RECTANGULAR" in line_text:
            idx = line_text.index("RECTANGULAR")
            nrectyp = int(line_text[idx + 1])

        bhe = bool("BHEDETAIL" in line_text)

        iclncn = 0
        if "SAVECLNCON" in line_text:
            idx = line_text.index("SAVECLNCON")
            iclncn = int(line_text[idx + 1])

        iclnmb = 0
        if "SAVECLNMAS" in line_text:
            idx = line_text.index("SAVECLNMAS")
            iclnmb = int(line_text[idx + 1])

        grav = None
        if "GRAVITY" in line_text:
            idx = line_text.index("GRAVITY")
            grav = float(line_text[idx + 1])

        visk = None
        if "VISCOSITY" in line_text:
            idx = line_text.index("VISCOSITY")
            visk = float(line_text[idx + 1])

        if model.verbose:
            print(
                f"   ncln {ncln}\n   iclnnds {iclnnds}\n",
                f"   iclncb {iclncb}\n   iclnhd {iclnhd}\n",
                f"   iclndd {iclndd}\n   iclnib {iclnib}\n",
                f"   nclngwc {nclngwc}\n   TRANSIENT {transient}\n",
                f"   PRINTIAJA {printiaja}\n   RECTANGULAR {nrectyp}\n",
                f"   BHEDETAIL {bhe}\n   SAVECLNCON {iclncn}\n",
                f"   SAVECLNMAS {iclnmb}\n   GRAVITY {grav}\n",
                f"   VISCOSITY {visk}",
            )

        return (
            transient,
            printiaja,
            ncln,
            iclnnds,
            iclncb,
            iclnhd,
            iclndd,
            iclnib,
            nclngwc,
            nconduityp,
            nrectyp,
            cln_rect,
            bhe,
            iclncn,
            iclnmb,
            grav,
            visk,
        )

    @staticmethod
    def _load_items_3to6(f_obj, model, ncln, iclnnds, ext_unit_dict):
        """Loads cln items 3, or 4,5,6 from filehandle f."""
        nndcln = None
        clncon = None
        nja_cln = None
        iac_cln = None
        ja_cln = None
        if ncln > 0:
            if model.verbose:
                print("   Reading nndcln...")
            nndcln = Util2d.load(
                f_obj, model, (ncln,), np.int32, "nndcln", ext_unit_dict
            )
            nclnnds = nndcln.array.sum()
            if iclnnds > 0:
                if model.verbose:
                    print("   Reading clncon...")
                nclnnds = iclnnds
                clncon = []
                for icln in range(ncln):
                    line = f_obj.readline()
                    line_text = line.strip().split()
                    iclncon = []
                    for idx in range(nndcln[icln]):
                        iclncon.append(line_text[idx])
                    clncon.append(iclncon)
        elif ncln == 0:
            if model.verbose:
                print("   Reading nja_cln...")
            line = f_obj.readline()
            line_text = line.strip().split()
            nja_cln = int(line_text[0])

            if model.verbose:
                print("   Reading iac_cln...")
            nclnnds = abs(iclnnds)
            iac_cln = Util2d.load(
                f_obj, model, (nclnnds,), np.int32, "iac_cln", ext_unit_dict
            )

            if model.verbose:
                print("   Reading ja_cln...")
            ja_cln = Util2d.load(
                f_obj, model, (nja_cln,), np.int32, "ja_cln", ext_unit_dict
            )
        else:
            raise Exception("mfcln: negative number of CLN segments")

        return nndcln, clncon, nja_cln, iac_cln, ja_cln, nclnnds

    @staticmethod
    def _ftype():
        return "CLN"

    @staticmethod
    def _defaultunit():
        return [71, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def _is_float(string):
        """
        Test whether the string is a float number.
        """
        try:
            float(string)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def _make_recarray(array, dtype):
        """
        Returns a empty recarray based on dtype.
        """
        nprop = len(dtype.names)
        ptemp = []
        for item in array:
            if len(item) < nprop:
                item = item + (nprop - len(item)) * [0.0]
            else:
                item = item[:nprop]
            ptemp.append(tuple(item))

        return np.array(ptemp, dtype)

    @classmethod
    def _read_prop(cls, f_obj, nrec):
        """
        Read the property tables (node_prop, cln_gwc, cln_circ, cln_rect).

        Parameters
        ----------
        f_obj : package file handle
        nrec : number of rows in the table

        Returns
        -------
        A list of lists with length of nrec
        """
        ptemp = []

        for _ in range(nrec):
            line = f_obj.readline()
            line_text = line.strip().split()
            prop = [float(item) for item in line_text if cls._is_float(item)]
            ptemp.append(prop)

        return ptemp
