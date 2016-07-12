import numpy as np
from ..mbase import BaseModel
from ..pakbase import Package
from .mpsim import ModpathSim
from .mpbas import ModpathBas
import os


class ModpathList(Package):
    '''
    List package class
    '''

    def __init__(self, model, extension='list', listunit=7):
        """
        Package constructor.

        """
        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, 'LIST', listunit)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def write_file(self):
        # Not implemented for list class
        return


class Modpath(BaseModel):
    """
    Modpath base class

    """

    def __init__(self, modelname='modpathtest', simfile_ext='mpsim',
                 namefile_ext='mpnam',
                 version='modpath', exe_name='mp6.exe', modflowmodel=None,
                 dis_file=None, dis_unit=87, head_file=None, budget_file=None,
                 model_ws=None, external_path=None, verbose=False,
                 load=True, listunit=7):
        """
        Model constructor.

        """
        BaseModel.__init__(self, modelname, simfile_ext, exe_name,
                           model_ws=model_ws)

        self.version_types = {'modpath': 'MODPATH'}
        self.set_version(version)

        self.__mf = modflowmodel
        self.lst = ModpathList(self, listunit=listunit)
        self.mpnamefile = '{}.{}'.format(self.name, namefile_ext)
        self.mpbas_file = '{}.mpbas'.format(modelname)
        if self.__mf is not None:
            dis_file = self.__mf.dis.file_name[0]
            dis_unit = self.__mf.dis.unit_number[0]
        self.dis_file = dis_file
        self.dis_unit = dis_unit
        if self.__mf is not None:
            head_file = self.__mf.oc.file_name[1]
            budget_file = self.__mf.oc.file_name[3]
        self.head_file = head_file
        self.budget_file = budget_file
        self.__sim = None
        self.array_free_format = False
        self.array_format = 'modflow'
        self.external_path = external_path
        self.external = False
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.load = load
        self.__next_ext_unit = 500
        if external_path is not None:
            assert os.path.exists(
                external_path), 'external_path does not exist'
            self.external = True
        self.verbose = verbose
        return

    def __repr__(self):
        return 'Modpath model'

    # function to encapsulate next_ext_unit attribute
    def next_ext_unit(self):
        self.__next_ext_unit += 1
        return self.__next_ext_unit

    def getsim(self):
        if (self.__sim == None):
            for p in (self.packagelist):
                if isinstance(p, ModpathSim):
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
        f_nam = open(fn_path, 'w')
        f_nam.write('%s\n' % (self.heading))
        if self.mpbas_file is not None:
            f_nam.write('%s %3i %s\n' % ('MPBAS', 86, self.mpbas_file))
        if self.dis_file is not None:
            f_nam.write('%s %3i %s\n' % ('DIS', self.dis_unit, self.dis_file))
        if self.head_file is not None:
            f_nam.write('%s %3i %s\n' % ('HEAD', 88, self.head_file))
        if self.budget_file is not None:
            f_nam.write('%s %3i %s\n' % ('BUDGET', 89, self.budget_file))
        for u, f in zip(self.external_units, self.external_fnames):
            f_nam.write('DATA  {0:3d}  '.format(u) + f + '\n')
        f_nam.close()

    sim = property(getsim)  # Property has no setter, so read-only
    mf = property(getmf)  # Property has no setter, so read-only

    def create_mpsim(self, simtype='pathline', trackdir='forward',
                     packages='WEL'):
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
            particle locations. Supported packages are 'WEL' and 'RCH'.
            (default is 'WEL')

        Returns
        -------
        mpsim : ModpathSim object

        """
        if isinstance(packages, str):
            packages = [packages]
        pak_list = self.__mf.get_package_list()
        Grid = 1
        GridCellRegionOption = 1
        PlacementOption = 1
        ReleaseStartTime = 0.
        ReleaseOption = 1
        CHeadOption = 1
        nper = self.__mf.dis.nper
        nlay, nrow, ncol = self.__mf.dis.nlay, \
                           self.__mf.dis.nrow, \
                           self.__mf.dis.ncol
        arr = np.zeros((nlay, nrow, ncol), dtype=np.int)
        group_name = []
        group_region = []
        group_placement = []
        ifaces = []
        face_ct = []
        for package in packages:
            if package.upper() == 'WEL':
                if 'WEL' not in pak_list:
                    errmsg = 'Error: no well package in the passed model'
                    raise errmsg
                for kper in range(nper):
                    mflist = self.__mf.wel.stress_period_data[kper]
                    idx = [mflist['k'], mflist['i'], mflist['j']]
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
                            face_ct.append(6)
                            ifaces.append([[1, 4, 4], [2, 4, 4],
                                           [3, 4, 4], [4, 4, 4],
                                           [5, 4, 4], [6, 4, 4]])
                            icnt += 1
            elif package.upper() == 'RCH':
                for j in range(nrow):
                    for i in range(ncol):
                        group_name.append('rch')
                        group_placement.append([Grid, GridCellRegionOption,
                                                PlacementOption,
                                                ReleaseStartTime,
                                                ReleaseOption, CHeadOption])
                        group_region.append([0, i, j, 0, i, j])
                        face_ct.append(1)
                        ifaces.append([[6, 1, 1]])

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
        ReferenceTimeOption = 1
        StopOption = 2
        ParticleGenerationOption = 1
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

        return ModpathSim(self, option_flags=mpoptions,
                          group_placement=group_placement,
                          group_name=group_name,
                          group_region=group_region,
                          face_ct=face_ct, ifaces=ifaces)
