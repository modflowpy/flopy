import numpy as np
from ..mbase import BaseModel
from ..modflow import Modflow
from ..mf6 import MFModel
from ..pakbase import Package
from .mp7bas import Modpath7Bas
from .mp7sim import Modpath7Sim
from .mp7particle import ParticleCellData, NodeParticleTemplate
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

        Parameters
        ----------
        modelname
        simfile_ext
        namefile_ext
        version
        exe_name
        flowmodel
        head_file
        budget_file
        model_ws
        verbose
    """

    def __init__(self, modelname='modpath7test', simfile_ext='mpsim',
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

        if not isinstance(flowmodel, (Modflow, MFModel)):
            msg = 'Modpath7: flow model is not an instance of ' + \
                  'flopy.modflow.Modflow or flopy.mf6.MFModel. ' + \
                  'Passed object of type {}'.format(type(flowmodel))
            raise TypeError(msg)

        # if a MFModel instance ensure flowmodel is a MODFLOW 6 GWF model
        if isinstance(flowmodel, MFModel):
            if flowmodel.model_type != 'gwf' and \
                    flowmodel.model_type != 'gwf6':
                msg = 'Modpath7: flow model type must be gwf. ' + \
                      'Passed model_type is {}.'.format(flowmodel.model_type)
                raise TypeError(msg)

        # set flowmodel and flow_version attributes
        self.flowmodel = flowmodel
        self.flow_version = self.flowmodel.version

        if self.flow_version == 'mf6':
            shape = None
            # get discretization package
            ibound = None
            dis = self.flowmodel.get_package('DIS')
            if dis is None:
                msg = 'DIS, DISV, or DISU packages must be ' + \
                      'included in the passed MODFLOW 6 model'
                raise Exception(msg)
            else:
                if dis.package_name.lower() == 'dis':
                    nlay, nrow, ncol = dis.nlay.array, dis.nrow.array, \
                                       dis.ncol.array
                    shape = (nlay, nrow, ncol)
                elif dis.package_name.lower() == 'disv':
                    nlay, ncpl = dis.nlay.array, dis.ncpl.array
                    shape = (nlay, ncpl)
                elif dis.package_name.lower() == 'disu':
                    nodes = dis.nodes.array
                    shape = tuple(nodes, )
                else:
                    msg = 'DIS, DISV, or DISU packages must be ' + \
                          'included in the passed MODFLOW 6 model'
                    raise TypeError(msg)

            # terminate (for now) if mf6 model does not use dis or disv
            if len(shape) < 2:
                msg = 'DIS and DISV are currently the only supported ' + \
                      'MODFLOW 6 discretization packages that can be ' + \
                      'used with MODPATH 7'
                raise TypeError(msg)

            # set ib
            ib = dis.idomain.array
            # set all ib to active if ib is not defined
            if ib is None:
                ib = np.ones(shape, np.int32)

            # set dis and grbdis file name
            dis_file = None
            grbdis_file = dis.filename + '.grb'
            grbtag = 'GRB{}'.format(dis.package_name.upper())

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
            grbtag = None

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
        self.grbtag = grbtag
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

        # set ib and ibound
        self.ib = ib
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
            f.write('{:10s} {}\n'.format('MPBAS', self.mpbas_file))
        if self.dis_file is not None:
            f.write('{:10s} {}\n'.format('DIS', self.dis_file))
        if self.grbdis_file is not None:
            f.write('{:10s} {}\n'.format(self.grbtag, self.grbdis_file))
        if self.tdis_file is not None:
            f.write('{:10s} {}\n'.format('TDIS', self.tdis_file))
        if self.head_file is not None:
            f.write('{:10s} {}\n'.format('HEAD', self.head_file))
        if self.budget_file is not None:
            f.write('{:10s} {}\n'.format('BUDGET', self.budget_file))
        f.close()

    @staticmethod
    def create_mp7(modelname='modpath7test', trackdir='forward',
                   flowmodel=None, exe_name='mp7', model_ws='.',
                   verbose=False):
        """
        Create a default MODPATH 7 model using a passed flowmodel with
        8 particles in every active model cell.

        Parameters
        ----------
        modelname : string, optional
            Name of model.  This string will be used to name the MODFLOW input
            that are created with write_model. (the default is 'modpath7test')
        trackdir : str
            Keywork that defines the MODPATH particle tracking direction.
            Available trackdir's are 'backward' and 'forward'.
            (default is 'forward')
        flowmodel : flopy.modflow.Modflow or flopy.mf6.MFModel object
            MODFLOW model
        exe_name : string, optional
            The name of the executable to use (the default is 'mp7').
        model_ws : string, optional
            model workspace.  Directory name to create model data sets.
            (default is the present working directory).
        verbose : boolean, optional
            Print additional information to the screen (default is False).


        Returns
        -------
        mp : Modpath7 object

        """
        # create MODPATH 7 model instance
        mp = Modpath7(modelname=modelname, flowmodel=flowmodel,
                      exe_name=exe_name, model_ws=model_ws, verbose=verbose)

        # create MODPATH 7 basic file and add to the MODPATH 7
        # model instance (mp)
        Modpath7Bas(mp)

        # create particles
        nodes = []
        node = 0
        for ib in mp.ib.flatten():
            if ib > 0:
                nodes.append(node)
            node += 1
        p = ParticleCellData(columncelldivisions=2,
                             rowcelldivisions=2,
                             layercelldivisions=2,
                             nodes=nodes)
        pg = NodeParticleTemplate(particledata=p)

        # creat MODPATH 7 simulation file and add to the MODPATH 7
        # model instance (mp)
        Modpath7Sim(mp, simulationtype='combined',
                    trackingdirection=trackdir,
                    weaksinkoption='pass_through',
                    weaksourceoption='pass_through',
                    referencetime=0.,
                    stoptimeoption='extend',
                    particlegroups=pg)
        return mp
