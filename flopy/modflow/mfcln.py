"""
mfcln module.  Contains the ModflowCln class. Note that the user can access
the ModflowCln class as `flopy.modflow.ModflowCln`.

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

import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util2d


class ModflowCln(Package):
    """
    Connected Linear Network class

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
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
        segment. The number of entries for each line is the number of CLN
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
    BHE : bool
        is a flag indicating that BHE details are also included in a heat transport
        model. Specifically, the thermal conductance and BHE tube thickness are
        included in transfer of heat between groundwater and CLN cells along with
        the heat conductivity of the BHE fluid and the convective heat transfer
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
    >>> ml = flopy.modflow.Modflow(version='mfusg', exe_name='mfusg.exe')
    >>> node_prop = [[1,1,0,10.0,-110.0,1.57,0,0],[2,1,0,10.0,-130.0,1.57,0,0]]
    >>> cln_gwc = [[1,1,50,50,0,0,10.0,1.0,0],[2,2,50,50,0,0,10.0,1.0,0]]
    >>> cln = flopy.modflow.ModflowCln(ml, ncln=1, iclnnds=-1, nndcln=2,
            nclngwc = 2, node_prop =node_prop, cln_gwc =cln_gwc)

    """

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
        cln_circ=[[1, 10.0, 3.23e10]],  # circular conduit properties
        ibound=1,  # boundary condition types
        strt=1.0,  # initial head in CLN cells
        transient=False,  # OPTIONS: transient IBOUND for each stress period
        printiaja=False,  # OPTIONS: print IA_CLN and JA_CLN to listing file
        nrectyp=0,  # OPTIONS2: number of rectangular fracture types
        cln_rect=None,  # rectangular fracture properties
        BHE=False,  # OPTIONS2: borehole heat exchanger (BHE)
        grav=None,  # OPTIONS2: gravitational acceleration constant
        visk=None,  # OPTIONS2: kinematic viscosity of water
        extension=[
            "cln",
            "clncb",
            "clnhd",
            "clndd",
            "clnib",
            "clncn",
            "clnmb",
        ],
        unitnumber=None,
        filenames=None,
    ):
        if model.version != "mfusg":
            err = "Error: model version must be mfusg to use CLN package"
            raise Exception(err)

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowCln._defaultunit()
        elif isinstance(unitnumber, list):
            if len(unitnumber) < 7:
                for idx in range(len(unitnumber), 7):
                    unitnumber.append(0)

        # set filenames
        if filenames is None:
            filenames = [None, None, None, None, None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None, None, None, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 7:
                for idx in range(len(filenames), 7):
                    filenames.append(None)

        # Fill namefile items
        name = [ModflowCln._ftype()]
        extra = [""]
        exten = [extension[0]]
        units = unitnumber[0]

        # set package name
        fname = [filenames[0]]
        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=exten,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.url = "Connected_Linear_Network.htm"

        self._generate_heading()

        # Options
        self.transient = transient
        self.printiaja = printiaja

        # Options2 is for Darcy-Weisbach equation. Not used.
        self.nrectyp = nrectyp
        self.BHE = BHE
        self.grav = grav
        self.visk = visk

        # CLN output files
        self.iclncb = unitnumber[1]
        # >0: File unit for CLN CBB; 0 No budget output; <0 budget in List file
        if self.iclncb > 0:
            model.add_output_file(
                self.iclncb,
                fname=filenames[1],
                extension=extension[1],
                binflag=True,
                package=ModflowCln._ftype(),
            )
        self.iclnhd = unitnumber[2]
        # >0: File unit for CLN HDS; 0: No head output;
        if self.iclnhd > 0:
            model.add_output_file(
                self.iclnhd,
                fname=filenames[2],
                extension=extension[2],
                binflag=True,
                package=ModflowCln._ftype(),
            )
        self.iclndd = unitnumber[3]
        # >0: File unit for CLN Drawdown; 0: No drawdown output;
        if self.iclndd > 0:
            model.add_output_file(
                self.iclndd,
                fname=filenames[3],
                extension=extension[3],
                binflag=True,
                package=ModflowCln._ftype(),
            )
        self.iclnib = unitnumber[4]
        # >0: File unit for CLN IBOUND; 0: No IBOUND output;
        if self.iclnib > 0:
            model.add_output_file(
                self.iclnib,
                fname=filenames[4],
                extension=extension[4],
                binflag=True,
                package=ModflowCln._ftype(),
            )
        self.iclncn = unitnumber[5]
        # >0: File unit for CLN concentration; 0: No concentration output;
        if self.iclncn > 0:
            model.add_output_file(
                self.iclncn,
                fname=filenames[5],
                extension=extension[5],
                binflag=True,
                package=ModflowCln._ftype(),
            )
        self.iclnmb = unitnumber[6]
        # >0: File unit for CLN mass flux; 0: No mass flux output;
        if self.iclnmb > 0:
            model.add_output_file(
                self.iclnmb,
                fname=filenames[6],
                extension=extension[6],
                binflag=True,
                package=ModflowCln._ftype(),
            )

        ## Define CLN networks
        if ncln is None:
            raise Exception("mfcln: CLN network not defined")

        self.ncln = ncln
        self.iclnnds = iclnnds

        if self.ncln > 0:  # Linear CLN segments
            if nndcln is None:
                raise Exception(
                    "mfcln: nodes for each CLN segment must be provided"
                )
            self.nndcln = Util2d(
                model,
                (self.ncln,),
                np.int32,
                nndcln,
                name="nndcln",
                locat=self.unit_number[0],
            )

            # consequtive node number. No connection between segments
            if self.iclnnds < 0:
                self.nclnnds = self.nndcln.array.sum()
                self.nodeno = [x + 1 for x in range(self.nclnnds)]
            # Node number provided for each segment to simulate CLN networks
            elif self.iclnnds > 0:
                self.nclnnds = self.iclnnds
                self.clncon = clncon
                self.nodeno = list(set(clncon))
            else:
                raise Exception("mfcln: Node number = 0")

        elif self.ncln == 0:  # CLN network defined by IA-JA connection matrix
            if self.iclnnds <= 0:
                raise Exception("mfcln: Negative or zero number of nodes")

            self.nclnnds = self.iclnnds

            self.nodeno = [x + 1 for x in range(self.nclnnds)]

            self.nja_cln = nja_cln

            if iac_cln is None:
                raise Exception("mfcln: iac_cln must be provided")
            self.iac_cln = Util2d(
                model,
                (self.nclnnds,),
                np.int32,
                iac_cln,
                name="iac_cln",
                locat=self.unit_number[0],
            )

            msg = "mfcln: The sum of iac_cln must equal nja_cln."
            assert self.iac_cln.array.sum() == nja_cln, msg

            if ja_cln is None:
                raise Exception("mfcln: ja_cln must be provided")
            if ja_cln[0] == 0:
                # convert from zero-based to one-based
                ja_cln += 1
            self.ja_cln = Util2d(
                model,
                (self.nja_cln,),
                np.int32,
                ja_cln,
                name="ja_cln",
                locat=self.unit_number[0],
            )

        else:
            raise Exception(
                "mfcln: negative number of CLN segments in CLN package"
            )

        if node_prop is None:
            raise Exception("mfcln: Node properties must be provided")

        if len(node_prop) != self.nclnnds:
            raise Exception(
                "mfcln: Length of Node properties must equal number of nodes"
            )

        self.node_prop = make_recarray(
            node_prop, dtype=ModflowCln.get_clnnode_dtype()
        )

        if nclngwc is None:
            raise Exception("mfcln: CLN-GW connections not defined")
        self.nclngwc = nclngwc  # number of CLN-GW connections

        if cln_gwc is None:
            raise Exception("mfcln: CLN-GW connection not provided")

        if len(cln_gwc) != self.nclngwc:
            raise Exception(
                "mfcln: Number of CLN-GW connections not equal nclngwc"
            )

        structured = self.parent.structured

        self.cln_gwc = make_recarray(
            cln_gwc, dtype=ModflowCln.get_gwconn_dtype(structured)
        )

        # Circular conduit geometry types

        self.nconduityp = nconduityp

        if self.nconduityp <= 0 or cln_circ is None:
            raise Exception(
                "mfcln: Circular conduit properties must be provided"
            )

        if len(cln_circ) != self.nconduityp:
            raise Exception(
                "mfcln: Number of circular properties not equal nconduityp"
            )

        self.cln_circ = make_recarray(
            cln_circ, dtype=ModflowCln.get_clncirc_dtype(self.BHE)
        )

        # Rectangular conduit geometry types
        if self.nrectyp > 0:
            if len(cln_rect) != self.nconduityp:
                raise Exception(
                    "mfcln: Number of rectangular properties not equal nrectyp"
                )
            self.cln_rect = make_recarray(
                cln_rect, dtype=ModflowCln.get_clnrect_dtype(self.BHE)
            )

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
    def get_clnnode_dtype():
        """Returns the dtype of CLN node properties"""
        dtype = np.dtype(
            [
                ("ifno", int),  ## node number
                ("iftyp", int),  ## type-index
                ("ifdir", int),  ## directional index
                ("fleng", np.float32),  ## length
                ("felev", np.float32),  ## elevation of the bottom
                ("fangle", np.float32),  ## angle
                ("iflin", int),  ## flag of flow conditions
                ("iccwadi", int),  ## flag of vertical flow correction
                ("x1", np.float32),  ## coordinates
                ("y1", np.float32),  ## coordinates
                ("z1", np.float32),  ## coordinates
                ("x2", np.float32),  ## coordinates
                ("y2", np.float32),  ## coordinates
                ("z2", np.float32),  ## coordinates
            ]
        )
        return dtype

    @staticmethod
    def get_gwconn_dtype(structured=True):
        """Returns the dtype of CLN node - GW node connection properties"""
        if structured:
            dtype = np.dtype(
                [
                    ("ifnod", int),  ##CLN node number
                    ("igwlay", int),  ##layer number of connecting gw node
                    ("igwrow", int),  ##row number of connecting gw node
                    ("igwfcol", int),  ##col number of connecting gw node
                    ("ifcon", int),  ##index of connectivity equation
                    ("fskin", np.float32),  ##leakance across a skin
                    ("flengw", np.float32),  ##length of connection
                    (
                        "faniso",
                        np.float32,
                    ),  ##anisotropy or thickness of sediments
                    ("icgwadi", int),  ##flag of vertical flow correction
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("ifnod", int),  ##CLN node number
                    ("igwnod", int),  ##node number of connecting gw node
                    ("ifcon", int),  ##index of connectivity equation
                    ("fskin", np.float32),  ##leakance across a skin
                    ("flengw", np.float32),  ##length of connection
                    (
                        "faniso",
                        np.float32,
                    ),  ##anisotropy or thickness of sediments
                    ("icgwadi", int),  ##flag of vertical flow correction
                ]
            )
        return dtype

    @staticmethod
    def get_clncirc_dtype(BHE=False):  # borehole heat exchanger (BHE)
        """Returns the dtype of CLN node circular conduit type properties"""
        if BHE:
            dtype = np.dtype(
                [
                    ("iconduityp", int),  ## index of circular conduit type
                    ("frad", np.float32),  ## radius
                    (
                        "conduitk",
                        np.float32,
                    ),  ## conductivity or resistance factor
                    ("tcond", np.float32),  ## thermal conductivity of BHE tube
                    ("tthk", np.float32),  ## thickness
                    (
                        "tcfluid",
                        np.float32,
                    ),  ## thermal conductivity of the fluid
                    ("tconv", np.float32),  ## thermal convective coefficient
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("iconduityp", int),  ## index of circular conduit type
                    ("frad", np.float32),  ## radius
                    (
                        "conduitk",
                        np.float32,
                    ),  ## conductivity or resistance factor
                ]
            )
        return dtype

    @staticmethod
    def get_clnrect_dtype(BHE=False):
        """Returns the dtype of CLN node rectangular conduit type properties"""
        if BHE:
            dtype = np.dtype(
                [
                    ("irectyp", int),  ## index of rectangular conduit type
                    ("flength", np.float32),  ## width
                    ("fheight", np.float32),  ## height
                    (
                        "conduitk",
                        np.float32,
                    ),  ## conductivity or resistance factor
                    ("tcond", np.float32),  ## thermal conductivity of BHE tube
                    ("tthk", np.float32),  ## thickness of BHE tube
                    ("tcfluid", np.float32),  ## thermal conductivity of fluid
                    ("tconv", np.float32),  ## thermal convective
                ]
            )
        else:
            dtype = np.dtype(
                [
                    ("irectyp", int),  ## index of rectangular conduit type
                    ("flength", np.float32),  ## width
                    ("fheight", np.float32),  ## height
                    (
                        "conduitk",
                        np.float32,
                    ),  ## conductivity or resistance factor
                ]
            )
        return dtype

    @staticmethod
    def _cln_nodes(self):
        """Returns the total number of CLN nodes"""
        return self.nclnnds

    def write_file(self, f=None):
        """
        Write the package file.

        Returns
        -------
        None

        """
        if f is not None:
            if isinstance(f, str):
                f_cln = open(f, "w")
            else:
                f_cln = f
        else:
            f_cln = open(self.fn_path, "w")

        f_cln.write(f"{self.heading}\n")

        if self.transient or self.printiaja:
            f_cln.write("OPTIONS   ")
            if self.transient:
                f_cln.write("TRANSIENT ")
            if self.printiaja:
                f_cln.write("PRINTIAJA ")
            f_cln.write("\n")

        f_cln.write(
            "%10d%10d%10d%10d%10d%10d%10d%10d"
            % (
                self.ncln,
                self.iclnnds,
                self.iclncb,
                self.iclnhd,
                self.iclndd,
                self.iclnib,
                self.nclngwc,
                self.nconduityp,
            )
        )

        if self.nrectyp > 0:
            f_cln.write("RECTANGULAR %d" % self.nrectyp)
        if self.BHE:
            f_cln.write("BHEDETAIL ")
        if self.iclncn != 0:
            f_cln.write("SAVECLNCON %d" % self.iclncn)
        if self.iclnmb != 0:
            f_cln.write("SAVECLNMAS %d" % self.iclnmb)
        if self.grav is not None:
            f_cln.write("GRAVITY %f" % self.grav)
        if self.visk is not None:
            f_cln.write("VISCOSITY %f" % self.visk)
        f_cln.write("\n")

        if self.ncln > 0:
            f_cln.write(self.nndcln.get_file_entry())
            if self.iclnnds > 0:
                for icln in range(self.ncln):
                    f_cln.write(self.clncon[icln])
        elif self.ncln == 0:
            f_cln.write("%10d\n" % self.nja_cln)
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

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
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
        cln : ModflowCln object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> cln = flopy.modflow.ModflowCln.load('test.cln', m)

        """

        if model.verbose:
            sys.stdout.write("loading CLN package file...\n")

        if model.version != "mfusg":
            msg = (
                "Warning: model version was reset from "
                + "'{}' to 'mfusg' in order to load a CLN file".format(
                    model.version
                )
            )
            print(msg)
            model.version = "mfusg"

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        line = f.readline()

        # Options
        transient = False
        printiaja = False
        if line.startswith("OPTIONS"):
            t = line.strip().split()
            if "TRANSIENT" in t:
                transient = True
            if "PRINTIAJA" in t:
                printiaja = True
            line = f.readline()

        t = line.strip().split()
        ncln, iclnnds, iclncb, iclnhd, iclndd, iclnib, nclngwc, nconduityp = (
            int(t[0]),
            int(t[1]),
            int(t[2]),
            int(t[3]),
            int(t[4]),
            int(t[5]),
            int(t[6]),
            int(t[7]),
        )
        # Options2
        if "RECTANGULAR" in t:
            idx = t.index("RECTANGULAR")
            nrectyp = int(t[idx + 1])
        else:
            nrectyp = 0
            cln_rect = None

        if "BHEDETAIL" in t:
            BHE = True
        else:
            BHE = False

        if "SAVECLNCON" in t:
            idx = t.index("SAVECLNCON")
            iclncn = int(t[idx + 1])
        else:
            iclncn = 0

        if "SAVECLNMAS" in t:
            idx = t.index("SAVECLNMAS")
            iclnmb = int(t[idx + 1])
        else:
            iclnmb = 0

        if "GRAVITY" in t:
            idx = t.index("GRAVITY")
            grav = float(t[idx + 1])
        else:
            grav = None

        if "VISCOSITY" in t:
            idx = t.index("VISCOSITY")
            visk = float(t[idx + 1])
        else:
            visk = None

        if model.verbose:
            print("   ncln {}".format(ncln))
            print("   iclnnds {}".format(iclnnds))
            print("   iclncb {}".format(iclncb))
            print("   iclnhd {}".format(iclnhd))
            print("   iclndd {}".format(iclndd))
            print("   iclnib {}".format(iclnib))
            print("   nclngwc {}".format(nclngwc))
            print("   TRANSIENT {}".format(transient))
            print("   PRINTIAJA {}".format(printiaja))
            print("   RECTANGULAR {}".format(nrectyp))
            print("   BHEDETAIL {}".format(BHE))
            print("   SAVECLNCON {}".format(iclncn))
            print("   SAVECLNMAS {}".format(iclnmb))
            print("   GRAVITY {}".format(grav))
            print("   VISCOSITY {}".format(visk))

        nndcln = None
        clncon = None
        nja_cln = None
        iac_cln = None
        ja_cln = None
        if ncln > 0:
            nndcln = Util2d.load(
                f, model, (ncln,), np.int32, "nndcln", ext_unit_dict
            )
            if model.verbose:
                print("   nndcln {}".format(nndcln))
            nclnnds = nndcln.array.sum()
            if iclnnds > 0:
                nclnnds = iclnnds
                clncon = []
                for icln in range(ncln):
                    line = f.readline()
                    t = line.strip().split()

                    iclncon = []
                    for i in range(nndcln[icln]):
                        iclncon.append(t[i])

                    clncon = clncon.append(iclncon)
                if model.verbose:
                    print("   clncon {}".format(clncon))

        elif ncln == 0:
            line = f.readline()
            t = line.strip().split()
            nja_cln = int(t[0])
            if model.verbose:
                print("   nja_cln {}".format(nja_cln))

            nclnnds = abs(iclnnds)
            iac_cln = Util2d.load(
                f, model, (nclnnds,), np.int32, "iac_cln", ext_unit_dict
            )
            if model.verbose:
                print("   iac_cln {}".format(iac_cln))

            ja_cln = Util2d.load(
                f, model, (nja_cln,), np.int32, "ja_cln", ext_unit_dict
            )
            if model.verbose:
                print("   ja_cln {}".format(ja_cln))
        else:
            raise Exception("mfcln: negative number of CLN segments")

        node_prop = read_prop(f, nclnnds)
        if model.verbose:
            print("   node_prop {}".format(node_prop))

        cln_gwc = read_prop(f, nclngwc)
        if model.verbose:
            print("   cln_gwc {}".format(cln_gwc))
        cln_circ = read_prop(f, nconduityp)
        if model.verbose:
            print("   cln_circ {}".format(cln_circ))

        if nrectyp > 0:
            cln_rect = read_prop(f, nrectyp)
            if model.verbose:
                print("   cln_rect {}".format(cln_circ))
        else:
            cln_rect = None

        ibound = Util2d.load(
            f, model, (nclnnds,), np.int32, "ibound", ext_unit_dict
        )
        if model.verbose:
            print("   ibound {}".format(ibound))

        strt = Util2d.load(
            f, model, (nclnnds,), np.float32, "strt", ext_unit_dict
        )
        if model.verbose:
            print("   strt {}".format(strt))

        if openfile:
            f.close()

        # set package unit number
        # reset unit numbers
        unitnumber = ModflowCln._defaultunit()
        filenames = [None, None, None, None, None, None, None]

        if ext_unit_dict is not None:
            unitnumber[0], filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowCln._ftype()
            )
            if iclncb > 0:
                unitnumber[1], filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iclncb
                )
                model.add_pop_key_list(iclncb)
            if iclnhd != 0:
                unitnumber[2], filenames[2] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(iclnhd)
                )
                model.add_pop_key_list(abs(iclnhd))
            if iclndd != 0:
                unitnumber[3], filenames[3] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(iclndd)
                )
                model.add_pop_key_list(abs(iclndd))
            if iclnib != 0:
                unitnumber[4], filenames[4] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=abs(iclnib)
                )
                model.add_pop_key_list(abs(iclnib))
            if iclncn > 0:
                unitnumber[5], filenames[5] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iclncn
                )
                model.add_pop_key_list(iclncn)
            if iclnmb > 0:
                unitnumber[6], filenames[6] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iclnmb
                )
                model.add_pop_key_list(iclnmb)

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
            BHE=BHE,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        # return dis object instance
        return cln

    @staticmethod
    def _ftype():
        return "CLN"

    @staticmethod
    def _defaultunit():
        return [71, 0, 0, 0, 0, 0, 0]


def fmt_string(array):
    """Returns a C-style fmt string for numpy savetxt that corresponds to
    the dtype"""
    fmts = []
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if vtype in ("i", "b"):
            fmts.append("%10d")
        elif vtype == "f":
            fmts.append("%10.2E")
        elif vtype == "o":
            fmts.append("%10s")
        elif vtype == "s":
            msg = (
                "mfcln.fmt_string error: 'str' type found in dtype. "
                "This gives unpredictable results when "
                "recarray to file - change to 'object' type"
            )
            raise TypeError(msg)
        else:
            raise TypeError(
                "mfcln.fmt_string error: unknown vtype in "
                "field: {}".format(field)
            )
    return "".join(fmts)


def is_float(s):
    """Test whether the string is a float number"""
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def make_recarray(array, dtype):
    """Returns a empty recarray based on dtype"""
    nprop = len(dtype.names)
    ptemp = []
    for t in array:
        if len(t) < nprop:
            t = t + (nprop - len(t)) * [0.0]
        else:
            t = t[:nprop]
        ptemp.append(tuple(t))

    return np.array(ptemp, dtype)


def read_prop(f, nrec):
    """Read the property tables (node_prop, cln_gwc, cln_circ, cln_rect)
    from file f. nrec = number of rows in the table"""
    ptemp = []

    for i in range(nrec):
        line = f.readline()
        t = line.strip().split()
        ra = [float(s) for s in t if is_float(s)]
        ptemp.append(ra)

    return ptemp
