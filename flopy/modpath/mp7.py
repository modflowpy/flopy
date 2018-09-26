import numpy as np
from ..mbase import BaseModel
from ..modflow import Modflow
from ..mf6 import ModflowGwf
from ..pakbase import Package
from .mp7sim import Modpath7Sim
from .mp7bas import Modpath7Bas
import os


class Modpath7List(Package):
    '''
    List package class
    '''

    def __init__(self, model, extension='list', unitnumber=None):
        """
        Package constructor.

        """
        if unitnumber is None:
            unitnumber = model.next_unit()

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, 'LIST', unitnumber)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def write_file(self):
        # Not implemented for list class
        return


class Modpath7(BaseModel):
    """
    Modpath 7 base class

    """

    def __init__(self, modelname='modpath7', simfile_ext='mpsim',
                 namefile_ext='mpnam', version='modpath7', exe_name='mp7.exe',
                 flowmodel=None, head_file=None, budget_file=None,
                 model_ws=None, verbose=False):
        """
        Model constructor.

        """
        BaseModel.__init__(self, modelname, simfile_ext, exe_name,
                           model_ws=model_ws)

        self.version_types = {'modpath7': 'MODPATH 7'}
        self.set_version(version)

        self.lst = Modpath7List(self)

        self.mpnamefile = '{}.{}'.format(self.name, namefile_ext)
        self.mpbas_file = '{}.mpbas'.format(modelname)

        self.flowmodel = flowmodel
        if isinstance(self.flowmodel, Modflow):
            self.flow_version = self.flowmodel.version
        elif isinstance(self.flowmodel, ModflowGwf):
            self.flow_version = self.flowmodel.version

        if self.flow_version == 'mf6':
            shape = None
            # get discretization package
            ibound = None
            dis = self.flowmodel.get_package('DIS')
            if dis is None:
                dis = self.flowmodel.get_package('DISV')
            else:
                nlay, nrow, ncol = dis.nlay.array, dis.nrow.array, \
                                   dis.ncol.array
                shape = (nlay, nrow, ncol)
            if dis is None:
                dis = self.flowmodel.get_package('DISU')
            elif dis is not None and shape is None:
                nlay, ncpl = dis.nlay.array, dis.ncpl.array
                shape = (nlay, ncpl)
            if dis is None:
                msg = 'DIS, DISV, or DISU packages must be ' + \
                      'included in the passed MODFLOW 6 model'
                raise Exception(msg)
            elif dis is not None and shape is None:
                nodes = dis.nodes.array
                shape = (nodes)

            # terminate (for now) if mf6 model does not use dis
            if len(shape) != 3:
                msg = 'DIS currently the only supported MODFLOW 6 ' + \
                      'discretization package that can be used with ' + \
                      'MODPATH 7'
                raise Exception(msg)


            # set dis and grbdis file name
            dis_file = None
            grbdis_file = dis.filename + '.grb'

            tdis = self.flowmodel.simulation.get_package('TDIS')
            if tdis is None:
                msg = 'TDIS package must be ' + \
                      'included in the passed MODFLOW 6 model'
                raise Exception(msg)
            tdis_file = tdis.filename

            # get stress period data
            nper = tdis.nper.array
            perlen = []
            nstp = []
            v = tdis.perioddata.array
            for pl, ns, tsmult in v:
                perlen.append(pl)
                nstp.append(ns)
            perlen = np.array(perlen, dtype=np.float32)
            nstp = np.array(nstp, dtype=np.int32)

            # get oc file
            oc = self.flowmodel.get_package('OC')
            if oc is not None:
                # set head file name
                if head_file is None:
                    head_file = oc.head_filerecord.array['headfile'][0]

                # set budget file name
                if budget_file is None:
                    budget_file = oc.budget_filerecord.array['budgetfile'][0]

            # set laytyp based on icelltype
            npf = self.flowmodel.get_package('NPF')
            if npf is None:
                msg = 'NPF package must be ' + \
                      'included in the passed MODFLOW 6 model'
                raise Exception(msg)
            icelltype = npf.icelltype.array.reshape(shape)
            laytyp = []
            for k in range(shape[0]):
                laytyp.append(icelltype[k].max())
            laytyp = np.array(laytyp, dtype=np.int32)


            # set default hdry and hnoflo
            hdry = None
            hnoflo = None

        else:
            shape = None
            # extract data from DIS or DISU files and set shape
            dis = self.flowmodel.get_package('DIS')
            if dis is None:
                dis = self.flowmodel.get_package('DISU')
            elif dis is not None and shape is None:
                nlay, nrow, ncol = dis.nlay, dis.nrow, dis.ncol
                shape = (nlay, nrow, ncol)
            if dis is None:
                msg = 'DIS, or DISU packages must be ' + \
                      'included in the passed MODFLOW model'
                raise Exception(msg)
            elif dis is not None and shape is None:
                nlay, nodes = dis.nlay, dis.nodes
                shape = (nodes)

            # terminate (for now) if mf6 model does not use dis
            if len(shape) != 3:
                msg = 'DIS currently the only supported MODFLOW ' + \
                      'discretization package that can be used with ' + \
                      'MODPATH 7'
                raise Exception(msg)

            # get stress period data
            nper = dis.nper
            perlen = dis.perlen.array
            nstp = dis.nstp.array

            # set dis_file
            dis_file = dis.file_name[0]

            # set grbdis_file
            grbdis_file = None

            # set tdis_file
            tdis_file = None

            # set head file name
            if head_file is None:
                iu = self.flowmodel.oc.iuhead
                head_file = self.flowmodel.get_output(unit=iu)

            # get discretization package
            p = self.flowmodel.get_package('LPF')
            if p is None:
                p = self.flowmodel.get_package('BCF6')
            if p is None:
                p = self.flowmodel.get_package('UPW')
            if p is None:
                msg = 'LPF, BCF6, or UPW packages must be ' + \
                      'included in the passed MODFLOW model'
                raise Exception(msg)

            # set budget file name
            if budget_file is None:
                iu = p.ipakcb
                budget_file = self.flowmodel.get_output(unit=iu)

            # set laytyp
            if p.name[0] == 'BCF6':
                laytyp = p.laycon.array
            else:
                laytyp = p.laytyp.array

            # set hdry from flow package
            hdry = p.hdry

            # set hnoflo and ibound from BAS6 package
            bas = self.flowmodel.get_package('BAS6')
            hnoflo = bas.hnoflo
            ib = bas.ibound.array
            # reset to constant values if possible
            ibound = []
            for k in range(shape[0]):
                i = ib[k].flatten()
                if np.all(i == i[0]):
                    kval = i[0]
                else:
                    kval = ib[k]
                ibound.append(kval)

        # set dis_file and tdis_file
        self.shape = shape
        self.dis_file = dis_file
        self.grbdis_file = grbdis_file
        self.tdis_file = tdis_file

        # set temporal data
        self.nper = nper
        self.time_end = perlen.sum()
        self.perlen = perlen
        self.nstp = nstp

        # set output file names
        self.head_file = head_file
        self.budget_file = budget_file

        # make sure the valid files are available
        if self.head_file is None:
            msg = 'the head file in the MODFLOW model or passed ' + \
                  'to __init__ cannot be None'
            raise ValueError(msg)
        if self.budget_file is None:
            msg = 'the budget file in the MODFLOW model or passed ' + \
                  'to __init__ cannot be None'
            raise ValueError(msg)
        if self.dis_file is None and self.grbdis_file is None:
            msg = 'the dis file in the MODFLOW model or passed ' + \
                  'to __init__ cannot be None'
            raise ValueError(msg)

        # set laytyp
        self.laytyp = laytyp

        # set hnoflo and hdry
        self.hnoflo = hnoflo
        self.hdry = hdry

        # set ibound
        self.ibound = ibound

        # set file attributes
        self.array_free_format = True
        self.array_format = 'modflow'
        self.external = False

        # # set the rest of the attributes
        # self.__sim = None
        # self.array_free_format = False
        # self.array_format = 'modflow'
        # self.external_path = external_path
        # self.external = False
        # self.external_fnames = []
        # self.external_units = []
        # self.external_binflag = []

        self.verbose = verbose
        return

    def __repr__(self):
        return 'MODPATH 7 model'

    def getsim(self):
        if (self.__sim == None):
            for p in (self.packagelist):
                if isinstance(p, Modpath7Sim):
                    self.__sim = p
        return self.__sim

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        fpth = os.path.join(self.model_ws, self.mpnamefile)
        f = open(fpth, 'w')
        f.write('{}\n'.format(self.heading))
        if self.mpbas_file is not None:
            f.write('{} {}\n'.format('MPBAS', self.mpbas_file))
        if self.dis_file is not None:
            f.write('{} {}\n'.format('DIS', self.dis_file))
        if self.grbdis_file is not None:
            f.write('{} {}\n'.format('GRBDIS', self.grbdis_file))
        if self.tdis_file is not None:
            f.write('{} {}\n'.format('TDIS', self.tdis_file))
        if self.head_file is not None:
            f.write('{} {}\n'.format('HEAD', self.head_file))
        if self.budget_file is not None:
            f.write('{} {}\n'.format('BUDGET', self.budget_file))
        f.close()

    #sim = property(getsim)  # Property has no setter, so read-only
    #mf = property(getmf)  # Property has no setter, so read-only

    def create_mp7sim(self, simtype='pathline', trackdir='forward',
                      packages='WEL', start_time=0, default_ifaces=None,
                      ParticleColumnCount=4, ParticleRowCount=4,
                      MinRow=0, MinColumn=0, MaxRow=None, MaxColumn=None,
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
            Keywork that defines the MODPATH particle tracking direction.
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
        pak_list = self.flowmodel.get_package_list()

        # not sure if this is the best way to handle this
        ReferenceTimeOption = 1
        ref_time = 0
        ref_time_per_stp = (0, 0, 1.)
        if isinstance(start_time, tuple):
            ReferenceTimeOption = 2  # 1: specify value for ref. time, 2: specify kper, kstp, rel. time pos
            ref_time_per_stp = start_time
        else:
            ref_time = start_time

        # set iface particle grids
        ptrow = ParticleRowCount
        ptcol = ParticleColumnCount
        side_faces = [[1, ptrow, ptcol], [2, ptrow, ptcol],
                      [3, ptrow, ptcol], [4, ptrow, ptcol]]
        top_face = [5, ptrow, ptcol]
        botm_face = [6, ptrow, ptcol]
        if default_ifaces is not None:
            default_ifaces = [[ifc, ptrow, ptcol] for ifc in default_ifaces]

        Grid = 1
        GridCellRegionOption = 1
        PlacementOption = 1
        ReleaseStartTime = 0.
        ReleaseOption = 1
        CHeadOption = 1
        nper = self.flowmodel.dis.nper
        nlay, nrow, ncol = self.flowmodel.dis.nlay, \
                           self.flowmodel.dis.nrow, \
                           self.flowmodel.dis.ncol
        arr = np.zeros((nlay, nrow, ncol), dtype=np.int)
        group_name = []
        group_region = []
        group_placement = []
        ifaces = []
        face_ct = []
        strt_file = None
        for package in packages:

            if package.upper() == 'WEL':
                ParticleGenerationOption = 1
                if 'WEL' not in pak_list:
                    raise Exception(
                        'Error: no well package in the passed model')
                for kper in range(nper):
                    mflist = self.flowmodel.wel.stress_period_data[kper]
                    idx = (mflist['k'], mflist['i'], mflist['j'])
                    arr[idx] = 1
                ngrp = arr.sum()
                icnt = 0
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            if arr[k, i, j] < 1:
                                continue
                            group_name.append('wc{}'.format(icnt))
                            group_placement.append([Grid, GridCellRegionOption,
                                                    PlacementOption,
                                                    ReleaseStartTime,
                                                    ReleaseOption,
                                                    CHeadOption])
                            group_region.append([k, i, j, k, i, j])
                            if default_ifaces is None:
                                ifaces.append(
                                    side_faces + [top_face, botm_face])
                                face_ct.append(6)
                            else:
                                ifaces.append(default_ifaces)
                                face_ct.append(len(default_ifaces))
                            icnt += 1
            # this is kind of a band aid pending refactoring of mpsim class
            elif 'MNW' in package.upper():
                ParticleGenerationOption = 1
                if 'MNW2' not in pak_list:
                    raise Exception(
                        'Error: no MNW2 package in the passed model')
                node_data = self.flowmodel.mnw2.get_allnode_data()
                node_data.sort(order=['wellid', 'k'])
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
                    group_name.append('{}{}'.format(wellid, node_number))
                    group_placement.append([Grid, GridCellRegionOption,
                                            PlacementOption,
                                            ReleaseStartTime,
                                            ReleaseOption,
                                            CHeadOption])

                for wellid in wellids:
                    nd = node_data[node_data.wellid == wellid]
                    k, i, j = nd.k[0], nd.i[0], nd.j[0]
                    if len(nd) == 1:
                        append_node(side_faces + [top_face, botm_face],
                                    wellid, 0, k, i, j)
                    else:
                        append_node(side_faces + [top_face],
                                    wellid, 0, k, i, j)
                        for n in range(len(nd))[1:]:
                            k, i, j = nd.k[n], nd.i[n], nd.j[n]
                            if n == len(nd) - 1:
                                append_node(side_faces + [botm_face],
                                            wellid, n, k, i, j)
                            else:
                                append_node(side_faces,
                                            wellid, n, k, i, j)
            elif package.upper() == 'RCH':
                ParticleGenerationOption = 1
                # for j in range(nrow):
                #    for i in range(ncol):
                #        group_name.append('rch')
                group_name.append('rch')
                group_placement.append([Grid, GridCellRegionOption,
                                        PlacementOption,
                                        ReleaseStartTime,
                                        ReleaseOption, CHeadOption])
                group_region.append([0, 0, 0, 0, nrow - 1, ncol - 1])
                if default_ifaces is None:
                    face_ct.append(1)
                    ifaces.append([[6, 1, 1]])
                else:
                    ifaces.append(default_ifaces)
                    face_ct.append(len(default_ifaces))


            else:
                model_ws = ''
                if self.flowmodel is not None:
                    model_ws = self.flowmodel.model_ws
                if os.path.exists(os.path.join(model_ws, package)):
                    print(
                        "detected a particle starting locations file in packages")
                    assert len(
                        packages) == 1, "if a particle starting locations file is passed" + \
                                        ", other packages cannot be specified"
                    ParticleGenerationOption = 2
                    strt_file = package
                else:
                    raise Exception(
                        "package '{0}' not supported".format(package))

        SimulationType = 1
        if simtype.lower() == 'endpoint':
            SimulationType = 1
        elif simtype.lower() == 'pathline':
            SimulationType = 2
        elif simtype.lower() == 'timeseries':
            SimulationType = 3
        if trackdir.lower() == 'forward':
            TrackingDirection = 1
        elif trackdir.lower() == 'backward':
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

        mpoptions = [SimulationType, TrackingDirection, WeakSinkOption,
                     WeakSourceOption, ReferenceTimeOption, StopOption,
                     ParticleGenerationOption, TimePointOption,
                     BudgetOutputOption, ZoneArrayOption, RetardationOption,
                     AdvectiveObservationsOption]

        return ModpathSim(self,
                          ref_time=ref_time,
                          ref_time_per_stp=ref_time_per_stp,
                          option_flags=mpoptions,
                          group_placement=group_placement,
                          group_name=group_name,
                          group_region=group_region,
                          face_ct=face_ct, ifaces=ifaces,
                          strt_file=strt_file)
