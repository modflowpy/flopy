import numpy as np
from ..mbase import BaseModel
from ..pakbase import Package
from .mp6sim import Modpath6Sim
import os


class Modpath6List(Package):
    """
    List package class
    """

    def __init__(self, model, extension="list", listunit=7):
        """
        Package constructor.

        """
        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, "LIST", listunit)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def write_file(self):
        # Not implemented for list class
        return


class Modpath6(BaseModel):
    """
    Modpath6 class.

    Parameters
    ----------
    modelname : str, default "modpathtest"
        Basename for MODPATH 6 input and output files.
    simfile_ext : str, default "mpsim"
        Filename extension of the MODPATH 6 simulation file.
    namefile_ext : str, default mpnam"
        Filename extension of the MODPATH 6 namefile.
    version : str, default "modpath"
        String that defines the MODPATH version. Valid versions are
        "modpath" (default).
    exe_name : str, default "mp6.exe"
        The name of the executable to use.
    modflowmodel : flopy.modflow.Modflow
        MODFLOW model object with one of LPF, BCF6, or UPW packages.
    dis_file : str
        Required dis file name.
    dis_unit : int, default 87
        Optional dis file unit number.
    head_file : str
        Required filename of the MODFLOW output head file.
    budget_file : str
        Required filename of the MODFLOW output cell-by-cell budget file.
    model_ws : str, optional
        Model workspace.  Directory name to create model data sets.
        Default is the current working directory.
    external_path : str, optional
        Location for external files.
    verbose : bool, default False
        Print additional information to the screen.
    load : bool, default True
         Load model.
    listunit : int, default 7
        LIST file unit number.

    """

    def __init__(
        self,
        modelname="modpathtest",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name="mp6.exe",
        modflowmodel=None,
        dis_file=None,
        dis_unit=87,
        head_file=None,
        budget_file=None,
        model_ws=None,
        external_path=None,
        verbose=False,
        load=True,
        listunit=7,
    ):
        super().__init__(
            modelname,
            simfile_ext,
            exe_name,
            model_ws=model_ws,
            verbose=verbose,
        )

        self.version_types = {"modpath": "MODPATH"}
        self.set_version(version)

        self.__mf = modflowmodel
        self.lst = Modpath6List(self, listunit=listunit)
        self.mpnamefile = "{}.{}".format(self.name, namefile_ext)
        self.mpbas_file = "{}.mpbas".format(modelname)
        if self.__mf is not None:
            # ensure that user-specified files are used
            iu = self.__mf.oc.iuhead
            head_file = self.__mf.get_output(unit=iu)
            p = self.__mf.get_package("LPF")
            if p is None:
                p = self.__mf.get_package("BCF6")
            if p is None:
                p = self.__mf.get_package("UPW")
            if p is None:
                raise Exception(
                    "LPF, BCF6, or UPW packages must be included in the "
                    "passed MODFLOW model"
                )
            iu = p.ipakcb
            budget_file = self.__mf.get_output(unit=iu)
            dis_file = (
                self.__mf.dis.file_name[0] if dis_file is None else dis_file
            )
            dis_unit = self.__mf.dis.unit_number[0]
        self.head_file = head_file
        self.budget_file = budget_file
        self.dis_file = dis_file
        self.dis_unit = dis_unit
        # make sure the valid files are available
        if self.head_file is None:
            raise ValueError(
                "the head file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )
        if self.budget_file is None:
            raise ValueError(
                "the budget file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )
        if self.dis_file is None:
            raise ValueError(
                "the dis file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )

        # set the rest of the attributes
        self.__sim = None
        self.array_free_format = False
        self.array_format = "modflow"
        self.external_path = external_path
        self.external = False
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.load = load
        self.__next_ext_unit = 500
        if external_path is not None:
            assert os.path.exists(
                external_path
            ), "external_path does not exist"
            self.external = True
        return

    def __repr__(self):
        return "Modpath model"

    # function to encapsulate next_ext_unit attribute
    def next_ext_unit(self):
        self.__next_ext_unit += 1
        return self.__next_ext_unit

    def getsim(self):
        if self.__sim == None:
            for p in self.packagelist:
                if isinstance(p, Modpath6Sim):
                    self.__sim = p
        return self.__sim

    def getmf(self):
        return self.__mf

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        fn_path = os.path.join(self.model_ws, self.mpnamefile)
        f_nam = open(fn_path, "w")
        f_nam.write("%s\n" % (self.heading))
        if self.mpbas_file is not None:
            f_nam.write("%s %3i %s\n" % ("MPBAS", 86, self.mpbas_file))
        if self.dis_file is not None:
            f_nam.write("%s %3i %s\n" % ("DIS", self.dis_unit, self.dis_file))
        if self.head_file is not None:
            f_nam.write("%s %3i %s\n" % ("HEAD", 88, self.head_file))
        if self.budget_file is not None:
            f_nam.write("%s %3i %s\n" % ("BUDGET", 89, self.budget_file))
        for u, f in zip(self.external_units, self.external_fnames):
            f_nam.write("DATA  {0:3d}  ".format(u) + f + "\n")
        f_nam.close()

    sim = property(getsim)  # Property has no setter, so read-only
    mf = property(getmf)  # Property has no setter, so read-only

    def create_mpsim(
        self,
        simtype="pathline",
        trackdir="forward",
        packages="WEL",
        start_time=0,
        default_ifaces=None,
        ParticleColumnCount=4,
        ParticleRowCount=4,
        MinRow=0,
        MinColumn=0,
        MaxRow=None,
        MaxColumn=None,
    ):
        """
        Create a MODPATH simulation file using available MODFLOW boundary
        package data.

        Parameters
        ----------
        simtype : str
            Keyword defining the MODPATH simulation type. Available simtype's
             are 'endpoint', 'pathline', and 'timeseries'.
             (default is 'PATHLINE')
        trackdir : str
            Keyword that defines the MODPATH particle tracking direction.
            Available trackdir's are 'backward' and 'forward'.
            (default is 'forward')
        packages : str or list of strings
            Keyword defining the modflow packages used to create initial
            particle locations. Supported packages are 'WEL', 'MNW2' and 'RCH'.
            (default is 'WEL').
        start_time : float or tuple
            Sets the value of MODPATH reference time relative to MODFLOW time.
            float : value of MODFLOW simulation time at which to start the particle tracking simulation.
                    Sets the value of MODPATH ReferenceTimeOption to 1.
            tuple : (period, step, time fraction) MODFLOW stress period, time step and fraction
                    between 0 and 1 at which to start the particle tracking simulation.
                    Sets the value of MODPATH ReferenceTimeOption to 2.
        default_ifaces : list
            List of cell faces (1-6; see MODPATH6 manual, fig. 7) on which to start particles.
            (default is None, meaning ifaces will vary depending on packages argument above)
        ParticleRowCount : int
            Rows of particles to start on each cell index face (iface).
        ParticleColumnCount : int
            Columns of particles to start on each cell index face (iface).

        Returns
        -------
        mpsim : ModpathSim object

        """
        if isinstance(packages, str):
            packages = [packages]
        pak_list = self.__mf.get_package_list()

        # not sure if this is the best way to handle this
        ReferenceTimeOption = 1
        ref_time = 0
        ref_time_per_stp = (0, 0, 1.0)
        if isinstance(start_time, tuple):
            ReferenceTimeOption = 2  # 1: specify value for ref. time, 2: specify kper, kstp, rel. time pos
            ref_time_per_stp = start_time
        else:
            ref_time = start_time

        # set iface particle grids
        ptrow = ParticleRowCount
        ptcol = ParticleColumnCount
        side_faces = [
            [1, ptrow, ptcol],
            [2, ptrow, ptcol],
            [3, ptrow, ptcol],
            [4, ptrow, ptcol],
        ]
        top_face = [5, ptrow, ptcol]
        botm_face = [6, ptrow, ptcol]
        if default_ifaces is not None:
            default_ifaces = [[ifc, ptrow, ptcol] for ifc in default_ifaces]

        Grid = 1
        GridCellRegionOption = 1
        PlacementOption = 1
        ReleaseStartTime = 0.0
        ReleaseOption = 1
        CHeadOption = 1
        nper = self.__mf.dis.nper
        nlay, nrow, ncol = (
            self.__mf.dis.nlay,
            self.__mf.dis.nrow,
            self.__mf.dis.ncol,
        )
        arr = np.zeros((nlay, nrow, ncol), dtype=int)
        group_name = []
        group_region = []
        group_placement = []
        ifaces = []
        face_ct = []
        strt_file = None
        for package in packages:

            if package.upper() == "WEL":
                ParticleGenerationOption = 1
                if "WEL" not in pak_list:
                    raise Exception(
                        "Error: no well package in the passed model"
                    )
                for kper in range(nper):
                    mflist = self.__mf.wel.stress_period_data[kper]
                    idx = (mflist["k"], mflist["i"], mflist["j"])
                    arr[idx] = 1
                ngrp = arr.sum()
                icnt = 0
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            if arr[k, i, j] < 1:
                                continue
                            group_name.append("wc{}".format(icnt))
                            group_placement.append(
                                [
                                    Grid,
                                    GridCellRegionOption,
                                    PlacementOption,
                                    ReleaseStartTime,
                                    ReleaseOption,
                                    CHeadOption,
                                ]
                            )
                            group_region.append([k, i, j, k, i, j])
                            if default_ifaces is None:
                                ifaces.append(
                                    side_faces + [top_face, botm_face]
                                )
                                face_ct.append(6)
                            else:
                                ifaces.append(default_ifaces)
                                face_ct.append(len(default_ifaces))
                            icnt += 1
            # this is kind of a band aid pending refactoring of mpsim class
            elif "MNW" in package.upper():
                ParticleGenerationOption = 1
                if "MNW2" not in pak_list:
                    raise Exception(
                        "Error: no MNW2 package in the passed model"
                    )
                node_data = self.__mf.mnw2.get_allnode_data()
                node_data.sort(order=["wellid", "k"])
                wellids = np.unique(node_data.wellid)

                def append_node(ifaces_well, wellid, node_number, k, i, j):
                    """add a single MNW node"""
                    group_region.append([k, i, j, k, i, j])
                    if default_ifaces is None:
                        ifaces.append(ifaces_well)
                        face_ct.append(len(ifaces_well))
                    else:
                        ifaces.append(default_ifaces)
                        face_ct.append(len(default_ifaces))
                    group_name.append("{}{}".format(wellid, node_number))
                    group_placement.append(
                        [
                            Grid,
                            GridCellRegionOption,
                            PlacementOption,
                            ReleaseStartTime,
                            ReleaseOption,
                            CHeadOption,
                        ]
                    )

                for wellid in wellids:
                    nd = node_data[node_data.wellid == wellid]
                    k, i, j = nd.k[0], nd.i[0], nd.j[0]
                    if len(nd) == 1:
                        append_node(
                            side_faces + [top_face, botm_face],
                            wellid,
                            0,
                            k,
                            i,
                            j,
                        )
                    else:
                        append_node(
                            side_faces + [top_face], wellid, 0, k, i, j
                        )
                        for n in range(len(nd))[1:]:
                            k, i, j = nd.k[n], nd.i[n], nd.j[n]
                            if n == len(nd) - 1:
                                append_node(
                                    side_faces + [botm_face],
                                    wellid,
                                    n,
                                    k,
                                    i,
                                    j,
                                )
                            else:
                                append_node(side_faces, wellid, n, k, i, j)
            elif package.upper() == "RCH":
                ParticleGenerationOption = 1
                # for j in range(nrow):
                #    for i in range(ncol):
                #        group_name.append('rch')
                group_name.append("rch")
                group_placement.append(
                    [
                        Grid,
                        GridCellRegionOption,
                        PlacementOption,
                        ReleaseStartTime,
                        ReleaseOption,
                        CHeadOption,
                    ]
                )
                group_region.append([0, 0, 0, 0, nrow - 1, ncol - 1])
                if default_ifaces is None:
                    face_ct.append(1)
                    ifaces.append([[6, 1, 1]])
                else:
                    ifaces.append(default_ifaces)
                    face_ct.append(len(default_ifaces))

            else:
                model_ws = ""
                if self.__mf is not None:
                    model_ws = self.__mf.model_ws
                if os.path.exists(os.path.join(model_ws, package)):
                    print(
                        "detected a particle starting locations file in packages"
                    )
                    assert len(packages) == 1, (
                        "if a particle starting locations file is passed, "
                        "other packages cannot be specified"
                    )
                    ParticleGenerationOption = 2
                    strt_file = package
                else:
                    raise Exception(
                        "package '{0}' not supported".format(package)
                    )

        SimulationType = 1
        if simtype.lower() == "endpoint":
            SimulationType = 1
        elif simtype.lower() == "pathline":
            SimulationType = 2
        elif simtype.lower() == "timeseries":
            SimulationType = 3
        if trackdir.lower() == "forward":
            TrackingDirection = 1
        elif trackdir.lower() == "backward":
            TrackingDirection = 2
        WeakSinkOption = 2
        WeakSourceOption = 1

        StopOption = 2

        if SimulationType == 1:
            TimePointOption = 1
        else:
            TimePointOption = 3
        BudgetOutputOption = 1
        ZoneArrayOption = 1
        RetardationOption = 1
        AdvectiveObservationsOption = 1

        mpoptions = [
            SimulationType,
            TrackingDirection,
            WeakSinkOption,
            WeakSourceOption,
            ReferenceTimeOption,
            StopOption,
            ParticleGenerationOption,
            TimePointOption,
            BudgetOutputOption,
            ZoneArrayOption,
            RetardationOption,
            AdvectiveObservationsOption,
        ]

        return Modpath6Sim(
            self,
            ref_time=ref_time,
            ref_time_per_stp=ref_time_per_stp,
            option_flags=mpoptions,
            group_placement=group_placement,
            group_name=group_name,
            group_region=group_region,
            face_ct=face_ct,
            ifaces=ifaces,
            strt_file=strt_file,
        )
