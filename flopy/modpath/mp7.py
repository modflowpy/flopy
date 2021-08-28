"""
mp7 module.  Contains the Modpath7List and Modpath7 classes.


"""

import numpy as np
from ..mbase import BaseModel
from ..modflow import Modflow
from ..mf6 import MFModel
from ..pakbase import Package
from .mp7bas import Modpath7Bas
from .mp7sim import Modpath7Sim
from .mp7particledata import CellDataType, NodeParticleData
from .mp7particlegroup import ParticleGroupNodeTemplate
import os


class Modpath7List(Package):
    """
    List package class

    """

    def __init__(self, model, extension="list", unitnumber=None):
        """
        Package constructor.

        """
        if unitnumber is None:
            unitnumber = model.next_unit()

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, "LIST", unitnumber)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def write_file(self):
        # Not implemented for list class
        return


class Modpath7(BaseModel):
    """
    Modpath 7 class.

    Parameters
    ----------
    modelname : str, default "modpath7test"
        Basename for MODPATH 7 input and output files.
    simfile_ext : str, default "mpsim"
        Filename extension of the MODPATH 7 simulation file.
    namefile_ext : str, default mpnam"
        Filename extension of the MODPATH 7 namefile.
    version : str, default "modpath7"
        String that defines the MODPATH version. Valid versions are
        "modpath7" (default).
    exe_name : str, default "mp7.exe"
        The name of the executable to use.
    flowmodel : flopy.modflow.Modflow or flopy.mf6.MFModel object
        MODFLOW model object.
    headfilename : str, optional
        Filename of the MODFLOW output head file. If headfilename is
        not provided then it will be set from the flowmodel.
    budgetfilename : str, optional
        Filename of the MODFLOW output cell-by-cell budget file.
        If budgetfilename is not provided then it will be set
        from the flowmodel.
    model_ws : str, default "."
        Model workspace.  Directory name to create model data sets.
        Default is the current working directory.
    verbose : bool, default False
        Print additional information to the screen.

    Examples
    --------
    >>> import flopy
    >>> m = flopy.modflow.Modflow.load('mf2005.nam')
    >>> mp = flopy.modpath.Modpath7('mf2005_mp', flowmodel=m)

    """

    def __init__(
        self,
        modelname="modpath7test",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath7",
        exe_name="mp7.exe",
        flowmodel=None,
        headfilename=None,
        budgetfilename=None,
        model_ws=None,
        verbose=False,
    ):
        super().__init__(
            modelname,
            simfile_ext,
            exe_name,
            model_ws=model_ws,
            verbose=verbose,
        )

        self.version_types = {"modpath7": "MODPATH 7"}
        self.set_version(version)

        self.lst = Modpath7List(self)

        self.mpnamefile = f"{self.name}.{namefile_ext}"
        self.mpbas_file = f"{modelname}.mpbas"

        if not isinstance(flowmodel, (Modflow, MFModel)):
            raise TypeError(
                "Modpath7: flow model is not an instance of "
                "flopy.modflow.Modflow or flopy.mf6.MFModel. "
                "Passed object of type {}".format(type(flowmodel))
            )

        # if a MFModel instance ensure flowmodel is a MODFLOW 6 GWF model
        if isinstance(flowmodel, MFModel):
            if (
                flowmodel.model_type != "gwf"
                and flowmodel.model_type != "gwf6"
            ):
                raise TypeError(
                    "Modpath7: flow model type must be gwf. "
                    "Passed model_type is {}.".format(flowmodel.model_type)
                )

        # set flowmodel and flow_version attributes
        self.flowmodel = flowmodel
        self.flow_version = self.flowmodel.version

        if self.flow_version == "mf6":
            # get discretization package
            ibound = None
            dis = self.flowmodel.get_package("DIS")
            if dis is None:
                raise Exception(
                    "DIS, DISV, or DISU packages must be "
                    "included in the passed MODFLOW 6 model"
                )
            else:
                if dis.package_name.lower() == "dis":
                    nlay, nrow, ncol = (
                        dis.nlay.array,
                        dis.nrow.array,
                        dis.ncol.array,
                    )
                    shape = (nlay, nrow, ncol)
                elif dis.package_name.lower() == "disv":
                    nlay, ncpl = dis.nlay.array, dis.ncpl.array
                    shape = (nlay, ncpl)
                elif dis.package_name.lower() == "disu":
                    nodes = dis.nodes.array
                    shape = tuple(
                        nodes,
                    )
                else:
                    raise TypeError(
                        "DIS, DISV, or DISU packages must be "
                        "included in the passed MODFLOW 6 model"
                    )

            # terminate (for now) if mf6 model does not use dis or disv
            if len(shape) < 2:
                raise TypeError(
                    "DIS and DISV are currently the only supported "
                    "MODFLOW 6 discretization packages that can be "
                    "used with MODPATH 7"
                )

            # set ib
            ib = dis.idomain.array
            # set all ib to active if ib is not defined
            if ib is None:
                ib = np.ones(shape, np.int32)

            # set dis and grbdis file name
            dis_file = None
            grbdis_file = f"{dis.filename}.grb"
            grbtag = f"GRB{dis.package_name.upper()}"

            tdis = self.flowmodel.simulation.get_package("TDIS")
            if tdis is None:
                raise Exception(
                    "TDIS package must be "
                    "included in the passed MODFLOW 6 model"
                )
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
            oc = self.flowmodel.get_package("OC")
            if oc is not None:
                # set head file name
                if headfilename is None:
                    headfilename = oc.head_filerecord.array["headfile"][0]

                # set budget file name
                if budgetfilename is None:
                    budgetfilename = oc.budget_filerecord.array["budgetfile"][
                        0
                    ]
        else:
            shape = None
            # extract data from DIS or DISU files and set shape
            dis = self.flowmodel.get_package("DIS")
            if dis is None:
                dis = self.flowmodel.get_package("DISU")
            elif dis is not None and shape is None:
                nlay, nrow, ncol = dis.nlay, dis.nrow, dis.ncol
                shape = (nlay, nrow, ncol)
            if dis is None:
                raise Exception(
                    "DIS, or DISU packages must be "
                    "included in the passed MODFLOW model"
                )
            elif dis is not None and shape is None:
                nlay, nodes = dis.nlay, dis.nodes
                shape = (nodes,)

            # terminate (for now) if mf6 model does not use dis
            if len(shape) != 3:
                raise Exception(
                    "DIS currently the only supported MODFLOW "
                    "discretization package that can be used with MODPATH 7"
                )

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
            if headfilename is None:
                iu = self.flowmodel.oc.iuhead
                headfilename = self.flowmodel.get_output(unit=iu)

            # get discretization package
            p = self.flowmodel.get_package("LPF")
            if p is None:
                p = self.flowmodel.get_package("BCF6")
            if p is None:
                p = self.flowmodel.get_package("UPW")
            if p is None:
                raise Exception(
                    "LPF, BCF6, or UPW packages must be "
                    "included in the passed MODFLOW model"
                )

            # set budget file name
            if budgetfilename is None:
                iu = p.ipakcb
                budgetfilename = self.flowmodel.get_output(unit=iu)

            # set hnoflo and ibound from BAS6 package
            bas = self.flowmodel.get_package("BAS6")
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
        self.headfilename = headfilename
        self.budgetfilename = budgetfilename

        # make sure the valid files are available
        if self.headfilename is None:
            raise ValueError(
                "the head file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )
        if self.budgetfilename is None:
            raise ValueError(
                "the budget file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )
        if self.dis_file is None and self.grbdis_file is None:
            raise ValueError(
                "the dis file in the MODFLOW model or passed "
                "to __init__ cannot be None"
            )

        # set ib and ibound
        self.ib = ib
        self.ibound = ibound

        # set file attributes
        self.array_free_format = True
        self.array_format = "modflow"
        self.external = False

        return

    def __repr__(self):
        return "MODPATH 7 model"

    @property
    def laytyp(self):
        if self.flowmodel.version == "mf6":
            icelltype = self.flowmodel.npf.icelltype.array
            laytyp = [
                icelltype[k].max()
                for k in range(self.flowmodel.modelgrid.nlay)
            ]
        else:
            p = self.flowmodel.get_package("BCF6")
            if p is None:
                laytyp = self.flowmodel.laytyp
            else:
                laytyp = p.laycon.array
        return np.array(laytyp, dtype=np.int32)

    @property
    def hdry(self):
        if self.flowmodel.version == "mf6":
            return None
        else:
            return self.flowmodel.hdry

    @property
    def hnoflo(self):
        if self.flowmodel.version == "mf6":
            return None
        else:
            return self.flowmodel.hnoflo

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        fpth = os.path.join(self.model_ws, self.mpnamefile)
        f = open(fpth, "w")
        f.write(f"{self.heading}\n")
        if self.mpbas_file is not None:
            f.write(f"MPBAS      {self.mpbas_file}\n")
        if self.dis_file is not None:
            f.write(f"DIS        {self.dis_file}\n")
        if self.grbdis_file is not None:
            f.write(f"{self.grbtag:10s} {self.grbdis_file}\n")
        if self.tdis_file is not None:
            f.write(f"TDIS       {self.tdis_file}\n")
        if self.headfilename is not None:
            f.write(f"HEAD       {self.headfilename}\n")
        if self.budgetfilename is not None:
            f.write(f"BUDGET     {self.budgetfilename}\n")
        f.close()

    @classmethod
    def create_mp7(
        cls,
        modelname="modpath7test",
        trackdir="forward",
        flowmodel=None,
        exe_name="mp7",
        model_ws=".",
        verbose=False,
        columncelldivisions=2,
        rowcelldivisions=2,
        layercelldivisions=2,
        nodes=None,
    ):
        """
        Create a default MODPATH 7 model using a passed flowmodel with
        8 particles in user-specified node locations or every active model
        cell.

        Parameters
        ----------
        modelname : str
            Basename for MODPATH 7 input and output files (default is
            'modpath7test').
        trackdir : str
            Keyword that defines the MODPATH particle tracking direction.
            Available trackdir's are 'backward' and 'forward'.
            (default is 'forward')
        flowmodel : flopy.modflow.Modflow or flopy.mf6.MFModel object
            MODFLOW model
        exe_name : str
            The name of the executable to use (the default is 'mp7').
        model_ws : str
            model workspace.  Directory name to create model data sets.
            (default is the current working directory).
        verbose : bool
            Print additional information to the screen (default is False).
        columncelldivisions : int
            Number of particles in a cell in the column (x-coordinate)
            direction (default is 2).
        rowcelldivisions : int
            Number of particles in a cell in the row (y-coordinate)
            direction (default is 2).
        layercelldivisions : int
            Number of particles in a cell in the layer (z-coordinate)
            direction (default is 2).
        nodes : int, list of ints, tuple of ints, or np.ndarray
            Nodes (zero-based) with particles. If  (default is node 0).

        Returns
        -------
        mp : Modpath7 object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('mf2005.nam')
        >>> mp = flopy.modpath.Modpath7.create_mp7(flowmodel=m)

        """
        # create MODPATH 7 model instance
        mp = Modpath7(
            modelname=modelname,
            flowmodel=flowmodel,
            exe_name=exe_name,
            model_ws=model_ws,
            verbose=verbose,
        )

        # set default iface for recharge and et
        if mp.flow_version == "mf6":
            defaultiface = {"RCH": 6, "EVT": 6}
        else:
            defaultiface = {"RECHARGE": 6, "ET": 6}

        # create MODPATH 7 basic file and add to the MODPATH 7
        # model instance (mp)
        Modpath7Bas(mp, defaultiface=defaultiface)

        # create particles
        if nodes is None:
            nodes = []
            node = 0
            for ib in mp.ib.flatten():
                if ib > 0:
                    nodes.append(node)
                node += 1
        sd = CellDataType(
            columncelldivisions=columncelldivisions,
            rowcelldivisions=rowcelldivisions,
            layercelldivisions=layercelldivisions,
        )
        p = NodeParticleData(subdivisiondata=sd, nodes=nodes)
        pg = ParticleGroupNodeTemplate(particledata=p)

        # create MODPATH 7 simulation file and add to the MODPATH 7
        # model instance (mp)
        Modpath7Sim(
            mp,
            simulationtype="combined",
            trackingdirection=trackdir,
            weaksinkoption="pass_through",
            weaksourceoption="pass_through",
            referencetime=0.0,
            stoptimeoption="extend",
            particlegroups=pg,
        )
        return mp
