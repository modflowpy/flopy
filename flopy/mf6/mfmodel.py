import inspect
import os
import sys
from typing import Union

import numpy as np

from ..discretization.grid import Grid
from ..discretization.modeltime import ModelTime
from ..discretization.structuredgrid import StructuredGrid
from ..discretization.unstructuredgrid import UnstructuredGrid
from ..discretization.vertexgrid import VertexGrid
from ..mbase import ModelInterface
from ..utils import datautil
from ..utils.check import mf6check
from .coordinates import modeldimensions
from .data import mfstructure
from .data.mfdatautil import DataSearchOutput, iterable
from .mfbase import (
    ExtFileAction,
    FlopyException,
    MFDataException,
    MFFileMgmt,
    PackageContainer,
    PackageContainerType,
    ReadAsArraysException,
    VerbosityLevel,
)
from .mfpackage import MFPackage
from .utils.mfenums import DiscretizationType
from .utils.output_util import MF6Output


class MFModel(PackageContainer, ModelInterface):
    """
    MODFLOW-6 model base class.  Represents a single model in a simulation.

    Parameters
    ----------
    simulation_data : MFSimulationData
        Simulation data object of the simulation this model will belong to
    structure : MFModelStructure
        Structure of this type of model
    modelname : str
        Name of the model
    model_nam_file : str
        Relative path to the model name file from model working folder
    version : str
        Version of modflow
    exe_name : str
        Model executable name
    model_ws : str
        Model working folder path
    disfile : str
        Relative path to dis file from model working folder
    grid_type : str
        Type of grid the model will use (structured, unstructured, vertices)
    verbose : bool
        Verbose setting for model operations (default False)

    Attributes
    ----------
    name : str
        Name of the model
    exe_name : str
        Model executable name
    packages : dict of MFPackage
        Dictionary of model packages

    """

    def __init__(
        self,
        simulation,
        model_type="gwf6",
        modelname="model",
        model_nam_file=None,
        version="mf6",
        exe_name="mf6",
        add_to_simulation=True,
        structure=None,
        model_rel_path=".",
        verbose=False,
        **kwargs,
    ):
        super().__init__(simulation.simulation_data, modelname)
        self.simulation = simulation
        self.simulation_data = simulation.simulation_data
        self.name = modelname
        self.name_file = None
        self._version = version
        self.model_type = model_type
        self.type = "Model"

        if model_nam_file is None:
            model_nam_file = f"{modelname}.nam"

        if add_to_simulation:
            self.structure = simulation.register_model(
                self, model_type, modelname, model_nam_file
            )
        else:
            self.structure = structure
        self.set_model_relative_path(model_rel_path)
        self.exe_name = exe_name
        self.dimensions = modeldimensions.ModelDimensions(
            self.name, self.simulation_data
        )
        self.simulation_data.model_dimensions[modelname] = self.dimensions
        self._ftype_num_dict = {}
        self._package_paths = {}
        self._verbose = verbose

        if model_nam_file is None:
            self.model_nam_file = f"{modelname}.nam"
        else:
            self.model_nam_file = model_nam_file

        # check for spatial reference info in kwargs
        xll = kwargs.pop("xll", None)
        yll = kwargs.pop("yll", None)
        self._xul = kwargs.pop("xul", None)
        self._yul = kwargs.pop("yul", None)
        rotation = kwargs.pop("rotation", 0.0)
        crs = kwargs.pop("crs", None)
        # build model grid object
        self._modelgrid = Grid(crs=crs, xoff=xll, yoff=yll, angrot=rotation)

        self.start_datetime = None
        # check for extraneous kwargs
        if len(kwargs) > 0:
            kwargs_str = ", ".join(kwargs.keys())
            excpt_str = (
                f'Extraneous kwargs "{kwargs_str}" provided to MFModel.'
            )
            raise FlopyException(excpt_str)

        # build model name file
        # create name file based on model type - support different model types
        package_obj = self.package_factory("nam", model_type[0:3])
        if not package_obj:
            excpt_str = (
                f"Name file could not be found for model{model_type[0:3]}."
            )
            raise FlopyException(excpt_str)

        self.name_file = package_obj(
            self,
            filename=self.model_nam_file,
            pname=self.name,
            _internal_package=True,
        )

    def __init_subclass__(cls):
        """Register model type"""
        super().__init_subclass__()
        PackageContainer.modflow_models.append(cls)
        PackageContainer.models_by_type[cls.model_type] = cls

    def __getattr__(self, item):
        """
        __getattr__ - used to allow for getting packages as if they are
                      attributes

        Parameters
        ----------
        item : str
            3 character package name (case insensitive)


        Returns
        -------
        pp : Package object
            Package object of type :class:`flopy.pakbase.Package`

        """
        if item == "name_file" or not hasattr(self, "name_file"):
            raise AttributeError(item)

        package = self.get_package(item)
        if package is not None:
            return package
        raise AttributeError(item)

    def __repr__(self):
        return self._get_data_str(True)

    def __str__(self):
        return self._get_data_str(False)

    def _get_data_str(self, formal):
        file_mgr = self.simulation_data.mfpath
        data_str = (
            "name = {}\nmodel_type = {}\nversion = {}\nmodel_"
            "relative_path = {}"
            "\n\n".format(
                self.name,
                self.model_type,
                self.version,
                file_mgr.model_relative_path[self.name],
            )
        )

        for package in self.packagelist:
            pk_str = package._get_data_str(formal, False)
            if formal:
                if len(pk_str.strip()) > 0:
                    data_str = (
                        "{}###################\nPackage {}\n"
                        "###################\n\n"
                        "{}\n".format(data_str, package._get_pname(), pk_str)
                    )
            else:
                pk_str = package._get_data_str(formal, False)
                if len(pk_str.strip()) > 0:
                    data_str = (
                        "{}###################\nPackage {}\n"
                        "###################\n\n"
                        "{}\n".format(data_str, package._get_pname(), pk_str)
                    )
        return data_str

    @property
    def nper(self):
        """Number of stress periods.

        Returns
        -------
        nper : int
            Number of stress periods in the simulation.

        """
        try:
            return self.simulation.tdis.nper.array
        except AttributeError:
            return None

    @property
    def modeltime(self):
        """Model time discretization information.

        Returns
        -------
        modeltime : ModelTime
            FloPy object containing time discretization information for the
            simulation.

        """
        tdis = self.simulation.get_package("tdis")
        period_data = tdis.perioddata.get_data()

        # build steady state data
        sto = self.get_package("sto")
        if sto is None:
            steady = np.full((len(period_data["perlen"])), True, dtype=bool)
        else:
            steady = np.full((len(period_data["perlen"])), False, dtype=bool)
            ss_periods = sto.steady_state.get_active_key_dict()
            tr_periods = sto.transient.get_active_key_dict()
            if ss_periods:
                last_ss_value = False
                # loop through steady state array
                for index, value in enumerate(steady):
                    # resolve if current index is steady state or transient
                    if index in ss_periods:
                        last_ss_value = True
                    elif index in tr_periods:
                        last_ss_value = False
                    if last_ss_value == True:
                        steady[index] = True

        # build model time
        itmuni = tdis.time_units.get_data()
        start_date_time = tdis.start_date_time.get_data()
        if itmuni is None:
            itmuni = 0
        if start_date_time is None:
            start_date_time = "01-01-1970"
        data_frame = {
            "perlen": period_data["perlen"],
            "nstp": period_data["nstp"],
            "tsmult": period_data["tsmult"],
        }
        self._model_time = ModelTime(
            data_frame, itmuni, start_date_time, steady
        )
        return self._model_time

    @property
    def modeldiscrit(self):
        """Basic model spatial discretization information.  This is used
        internally prior to model spatial discretization information being
        fully loaded.

        Returns
        -------
        model grid : Grid subclass
            FloPy object containing basic spatial discretization information
            for the model.

        """
        if self.get_grid_type() == DiscretizationType.DIS:
            dis = self.get_package("dis")
            return StructuredGrid(
                nlay=dis.nlay.get_data(),
                nrow=dis.nrow.get_data(),
                ncol=dis.ncol.get_data(),
            )
        elif self.get_grid_type() == DiscretizationType.DISV:
            dis = self.get_package("disv")
            return VertexGrid(
                ncpl=dis.ncpl.get_data(), nlay=dis.nlay.get_data()
            )
        elif self.get_grid_type() == DiscretizationType.DISU:
            dis = self.get_package("disu")
            nodes = dis.nodes.get_data()
            ncpl = np.array([nodes], dtype=int)
            return UnstructuredGrid(ncpl=ncpl)

    @property
    def modelgrid(self):
        """Model spatial discretization information.

        Returns
        -------
        model grid : Grid subclass
            FloPy object containing spatial discretization information for the
            model.

        """

        if not self._mg_resync:
            return self._modelgrid
        if self.get_grid_type() == DiscretizationType.DIS:
            dis = self.get_package("dis")
            if not hasattr(dis, "_init_complete"):
                if not hasattr(dis, "delr"):
                    # dis package has not yet been initialized
                    return self._modelgrid
                else:
                    # dis package has been partially initialized
                    self._modelgrid = StructuredGrid(
                        delc=dis.delc.array,
                        delr=dis.delr.array,
                        top=None,
                        botm=None,
                        idomain=None,
                        lenuni=None,
                        crs=self._modelgrid.crs,
                        xoff=self._modelgrid.xoffset,
                        yoff=self._modelgrid.yoffset,
                        angrot=self._modelgrid.angrot,
                    )
            else:
                self._modelgrid = StructuredGrid(
                    delc=dis.delc.array,
                    delr=dis.delr.array,
                    top=dis.top.array,
                    botm=dis.botm.array,
                    idomain=dis.idomain.array,
                    lenuni=dis.length_units.array,
                    crs=self._modelgrid.crs,
                    xoff=self._modelgrid.xoffset,
                    yoff=self._modelgrid.yoffset,
                    angrot=self._modelgrid.angrot,
                )
        elif self.get_grid_type() == DiscretizationType.DISV:
            dis = self.get_package("disv")
            if not hasattr(dis, "_init_complete"):
                if not hasattr(dis, "cell2d"):
                    # disv package has not yet been initialized
                    return self._modelgrid
                else:
                    # disv package has been partially initialized
                    self._modelgrid = VertexGrid(
                        vertices=dis.vertices.array,
                        cell2d=dis.cell2d.array,
                        top=None,
                        botm=None,
                        idomain=None,
                        lenuni=None,
                        crs=self._modelgrid.crs,
                        xoff=self._modelgrid.xoffset,
                        yoff=self._modelgrid.yoffset,
                        angrot=self._modelgrid.angrot,
                    )
            else:
                self._modelgrid = VertexGrid(
                    vertices=dis.vertices.array,
                    cell2d=dis.cell2d.array,
                    top=dis.top.array,
                    botm=dis.botm.array,
                    idomain=dis.idomain.array,
                    lenuni=dis.length_units.array,
                    crs=self._modelgrid.crs,
                    xoff=self._modelgrid.xoffset,
                    yoff=self._modelgrid.yoffset,
                    angrot=self._modelgrid.angrot,
                )
        elif self.get_grid_type() == DiscretizationType.DISU:
            dis = self.get_package("disu")
            if not hasattr(dis, "_init_complete"):
                # disu package has not yet been fully initialized
                return self._modelgrid

            # check to see if ncpl can be constructed from ihc array,
            # otherwise set ncpl equal to [nodes]
            ihc = dis.ihc.array
            iac = dis.iac.array
            ncpl = UnstructuredGrid.ncpl_from_ihc(ihc, iac)
            if ncpl is None:
                ncpl = np.array([dis.nodes.get_data()], dtype=int)
            cell2d = dis.cell2d.array
            idomain = np.ones(dis.nodes.array, np.int32)
            if cell2d is None:
                if (
                    self.simulation.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: cell2d information missing. Functionality of "
                        "the UnstructuredGrid will be limited."
                    )
                iverts = None
                xcenters = None
                ycenters = None
            else:
                iverts = [list(i)[4:] for i in cell2d]
                xcenters = dis.cell2d.array["xc"]
                ycenters = dis.cell2d.array["yc"]
            vertices = dis.vertices.array
            if vertices is None:
                if (
                    self.simulation.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: vertices information missing. Functionality "
                        "of the UnstructuredGrid will be limited."
                    )
                vertices = None
            else:
                vertices = np.array(vertices)

            self._modelgrid = UnstructuredGrid(
                vertices=vertices,
                iverts=iverts,
                xcenters=xcenters,
                ycenters=ycenters,
                top=dis.top.array,
                botm=dis.bot.array,
                idomain=idomain,
                lenuni=dis.length_units.array,
                ncpl=ncpl,
                crs=self._modelgrid.crs,
                xoff=self._modelgrid.xoffset,
                yoff=self._modelgrid.yoffset,
                angrot=self._modelgrid.angrot,
                iac=dis.iac.array,
                ja=dis.ja.array,
            )
        elif self.get_grid_type() == DiscretizationType.DISL:
            dis = self.get_package("disl")
            if not hasattr(dis, "_init_complete"):
                if not hasattr(dis, "cell1d"):
                    # disv package has not yet been initialized
                    return self._modelgrid
                else:
                    # disv package has been partially initialized
                    self._modelgrid = VertexGrid(
                        vertices=dis.vertices.array,
                        cell1d=dis.cell1d.array,
                        top=None,
                        botm=None,
                        idomain=None,
                        lenuni=None,
                        crs=self._modelgrid.crs,
                        xoff=self._modelgrid.xoffset,
                        yoff=self._modelgrid.yoffset,
                        angrot=self._modelgrid.angrot,
                    )
            else:
                self._modelgrid = VertexGrid(
                    vertices=dis.vertices.array,
                    cell1d=dis.cell1d.array,
                    top=dis.top.array,
                    botm=dis.botm.array,
                    idomain=dis.idomain.array,
                    lenuni=dis.length_units.array,
                    crs=self._modelgrid.crs,
                    xoff=self._modelgrid.xoffset,
                    yoff=self._modelgrid.yoffset,
                    angrot=self._modelgrid.angrot,
                )
        else:
            return self._modelgrid

        if self.get_grid_type() != DiscretizationType.DISV:
            # get coordinate data from dis file
            xorig = dis.xorigin.get_data()
            yorig = dis.yorigin.get_data()
            angrot = dis.angrot.get_data()
        else:
            xorig = self._modelgrid.xoffset
            yorig = self._modelgrid.yoffset
            angrot = self._modelgrid.angrot

        # resolve offsets
        if xorig is None:
            xorig = self._modelgrid.xoffset
        if xorig is None:
            if self._xul is not None:
                xorig = self._modelgrid._xul_to_xll(self._xul)
            else:
                xorig = 0.0
        if yorig is None:
            yorig = self._modelgrid.yoffset
        if yorig is None:
            if self._yul is not None:
                yorig = self._modelgrid._yul_to_yll(self._yul)
            else:
                yorig = 0.0
        if angrot is None:
            angrot = self._modelgrid.angrot
        self._modelgrid.set_coord_info(
            xorig,
            yorig,
            angrot,
            self._modelgrid.crs,
        )
        self._mg_resync = not self._modelgrid.is_complete
        return self._modelgrid

    @property
    def packagelist(self):
        """List of model packages."""
        return self._packagelist

    @property
    def namefile(self):
        """Model namefile object."""
        return self.model_nam_file

    @property
    def model_ws(self):
        """Model file path."""
        file_mgr = self.simulation_data.mfpath
        return file_mgr.get_model_path(self.name)

    @property
    def exename(self):
        """MODFLOW executable name"""
        return self.exe_name

    @property
    def version(self):
        """Version of MODFLOW"""
        return self._version

    @property
    def solver_tols(self):
        """Returns the solver inner hclose and rclose values.

        Returns
        -------
        inner_hclose, rclose : float, float

        """
        ims = self.get_ims_package()
        if ims is not None:
            rclose = ims.rcloserecord.get_data()
            if rclose is not None:
                rclose = rclose[0][0]
            return ims.inner_hclose.get_data(), rclose
        return None

    @property
    def laytyp(self):
        """Layering type"""
        try:
            return self.npf.icelltype.array
        except AttributeError:
            return None

    @property
    def hdry(self):
        """Dry cell value"""
        return -1e30

    @property
    def hnoflo(self):
        """No-flow cell value"""
        return 1e30

    @property
    def laycbd(self):
        """Quasi-3D confining bed.  Not supported in MODFLOW-6.

        Returns
        -------
        None : None

        """
        return None

    @property
    def output(self):
        try:
            return self.oc.output
        except AttributeError:
            return MF6Output(self)

    def export(self, f, **kwargs):
        """Method to export a model to a shapefile or netcdf file

        Parameters
        ----------
        f : str
            File name (".nc" for netcdf or ".shp" for shapefile)
            or dictionary of ....
        **kwargs : keyword arguments
            modelgrid: flopy.discretization.Grid
                User supplied modelgrid object which will supercede the built
                in modelgrid object
            epsg : int
                EPSG projection code
            prj : str
                The prj file name
            if fmt is set to 'vtk', parameters of vtk.export_model

        """
        from ..export import utils

        return utils.model_export(f, self, **kwargs)

    @property
    def verbose(self):
        """Verbose setting for model operations (True/False)"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Verbose setting for model operations (True/False)"""
        self._verbose = verbose

    def check(self, f=None, verbose=True, level=1):
        """
        Check model data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        success : bool

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.check()
        """
        # check instance for model-level check
        chk = mf6check(self, f=f, verbose=verbose, level=level)

        return self._check(chk, level)

    @classmethod
    def load_base(
        cls,
        simulation,
        structure,
        modelname="NewModel",
        model_nam_file="modflowtest.nam",
        mtype="gwf",
        version="mf6",
        exe_name: Union[str, os.PathLike] = "mf6",
        strict=True,
        model_rel_path=os.curdir,
        load_only=None,
    ):
        """
        Class method that loads an existing model.

        Parameters
        ----------
        simulation : MFSimulation
            simulation object that this model is a part of
        simulation_data : MFSimulationData
            simulation data object
        structure : MFModelStructure
            structure of this type of model
        model_name : str
            name of the model
        model_nam_file : str
            relative path to the model name file from model working folder
        version : str
            version of modflow
        exe_name : str or PathLike
            model executable name or path
        strict : bool
            strict mode when loading files
        model_rel_path : str
            relative path of model folder to simulation folder
        load_only : list
            list of package abbreviations or package names corresponding to
            packages that flopy will load. default is None, which loads all
            packages. the discretization packages will load regardless of this
            setting. subpackages, like time series and observations, will also
            load regardless of this setting.
            example list: ['ic', 'maw', 'npf', 'oc', 'my_well_package_1']

        Returns
        -------
        model : MFModel

        Examples
        --------
        """
        instance = cls(
            simulation,
            mtype,
            modelname,
            model_nam_file=model_nam_file,
            version=version,
            exe_name=exe_name,
            add_to_simulation=False,
            structure=structure,
            model_rel_path=model_rel_path,
        )

        # build case consistent load_only dictionary for quick lookups
        load_only = instance._load_only_dict(load_only)

        # load name file
        instance.name_file.load(strict)

        # order packages
        vnum = mfstructure.MFStructure().get_version_string()
        # FIX: Transport - Priority packages maybe should not be hard coded
        priority_packages = {
            f"dis{vnum}": 1,
            f"disv{vnum}": 1,
            f"disu{vnum}": 1,
        }
        packages_ordered = []
        package_recarray = instance.simulation_data.mfdata[
            (modelname, "nam", "packages", "packages")
        ]
        for item in package_recarray.get_data():
            if item[0] in priority_packages:
                packages_ordered.insert(0, (item[0], item[1], item[2]))
            else:
                packages_ordered.append((item[0], item[1], item[2]))

        # load packages
        sim_struct = mfstructure.MFStructure().sim_struct
        instance._ftype_num_dict = {}
        for ftype, fname, pname in packages_ordered:
            ftype_orig = ftype
            ftype = ftype[0:-1].lower()
            if (
                ftype in structure.package_struct_objs
                or ftype in sim_struct.utl_struct_objs
            ):
                if (
                    load_only is not None
                    and not instance._in_pkg_list(
                        priority_packages, ftype_orig, pname
                    )
                    and not instance._in_pkg_list(load_only, ftype_orig, pname)
                ):
                    if (
                        simulation.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(f"    skipping package {ftype}...")
                    continue
                if model_rel_path and model_rel_path != ".":
                    # strip off model relative path from the file path
                    filemgr = simulation.simulation_data.mfpath
                    fname = filemgr.strip_model_relative_path(modelname, fname)
                if (
                    simulation.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(f"    loading package {ftype}...")
                # load package
                instance.load_package(ftype, fname, pname, strict, None)
                sim_data = simulation.simulation_data
                if ftype == "dis" and not sim_data.max_columns_user_set:
                    # set column wrap to ncol
                    dis = instance.get_package("dis")
                    if dis is not None and hasattr(dis, "ncol"):
                        sim_data.max_columns_of_data = dis.ncol.get_data()
                        sim_data.max_columns_user_set = False
                        sim_data.max_columns_auto_set = True
        # load referenced packages
        if modelname in instance.simulation_data.referenced_files:
            for ref_file in instance.simulation_data.referenced_files[
                modelname
            ].values():
                if (
                    ref_file.file_type in structure.package_struct_objs
                    or ref_file.file_type in sim_struct.utl_struct_objs
                ) and not ref_file.loaded:
                    instance.load_package(
                        ref_file.file_type,
                        ref_file.file_name,
                        None,
                        strict,
                        ref_file.reference_path,
                    )
                    ref_file.loaded = True

        # TODO: fix jagged lists where appropriate

        return instance

    def inspect_cells(
        self,
        cell_list,
        stress_period=None,
        output_file_path=None,
        inspect_budget=True,
        inspect_dependent_var=True,
    ):
        """
        Inspect model cells.  Returns model data associated with cells.

        Parameters
        ----------
        cell_list : list of tuples
            List of model cells.  Each model cell is a tuple of integers.
            ex: [(1,1,1), (2,4,3)]
        stress_period : int
            For transient data qnly return data from this stress period.  If
            not specified or None, all stress period data will be returned.
        output_file_path: str
            Path to output file that will contain the inspection results
        inspect_budget: bool
            Inspect budget file
        inspect_dependent_var: bool
            Inspect head file
        Returns
        -------
        output : dict
            Dictionary containing inspection results

        Examples
        --------

        >>> import flopy
        >>> sim = flopy.mf6.MFSimulation.load("name", "mf6", "mf6", ".")
        >>> model = sim.get_model()
        >>> inspect_list = [(2, 3, 2), (0, 4, 2), (0, 2, 4)]
        >>> out_file = os.path.join("temp", "inspect_AdvGW_tidal.csv")
        >>> model.inspect_cells(inspect_list, output_file_path=out_file)
        """
        # handle no cell case
        if cell_list is None or len(cell_list) == 0:
            return None

        output_by_package = {}
        # loop through all packages
        for pp in self.packagelist:
            # call the package's "inspect_cells" method
            package_output = pp.inspect_cells(cell_list, stress_period)
            if len(package_output) > 0:
                output_by_package[
                    f"{pp.package_name} package"
                ] = package_output
        # get dependent variable
        if inspect_dependent_var:
            try:
                if self.model_type == "gwf6":
                    heads = self.output.head()
                    name = "heads"
                elif self.model_type == "gwt6":
                    heads = self.output.concentration()
                    name = "concentration"
                else:
                    inspect_dependent_var = False
            except Exception:
                inspect_dependent_var = False
        if inspect_dependent_var and heads is not None:
            kstp_kper_lst = heads.get_kstpkper()
            data_output = DataSearchOutput((name,))
            data_output.output = True
            for kstp_kper in kstp_kper_lst:
                if stress_period is not None and stress_period != kstp_kper[1]:
                    continue
                head_array = np.array(heads.get_data(kstpkper=kstp_kper))
                # flatten output data in disv and disu cases
                if len(cell_list[0]) == 2:
                    head_array = head_array[0, :, :]
                elif len(cell_list[0]) == 1:
                    head_array = head_array[0, 0, :]
                # find data matches
                self.match_array_cells(
                    cell_list,
                    head_array.shape,
                    head_array,
                    kstp_kper,
                    data_output,
                )
            if len(data_output.data_entries) > 0:
                output_by_package[f"{name} output"] = [data_output]

        # get model dimensions
        model_shape = self.modelgrid.shape

        # get budgets
        if inspect_budget:
            try:
                bud = self.output.budget()
            except Exception:
                inspect_budget = False
        if inspect_budget and bud is not None:
            kstp_kper_lst = bud.get_kstpkper()
            rec_names = bud.get_unique_record_names()
            budget_matches = []
            for rec_name in rec_names:
                # clean up binary string name
                string_name = str(rec_name)[3:-1].strip()
                data_output = DataSearchOutput((string_name,))
                data_output.output = True
                for kstp_kper in kstp_kper_lst:
                    if (
                        stress_period is not None
                        and stress_period != kstp_kper[1]
                    ):
                        continue
                    budget_array = np.array(
                        bud.get_data(
                            kstpkper=kstp_kper,
                            text=rec_name,
                            full3D=True,
                        )[0]
                    )
                    if len(budget_array.shape) == 4:
                        # get rid of 4th "time" dimension
                        budget_array = budget_array[0, :, :, :]
                    # flatten output data in disv and disu cases
                    if len(cell_list[0]) == 2 and len(budget_array.shape) >= 3:
                        budget_array = budget_array[0, :, :]
                    elif (
                        len(cell_list[0]) == 1 and len(budget_array.shape) >= 2
                    ):
                        budget_array = budget_array[0, :]
                    # find data matches
                    if budget_array.shape != model_shape:
                        # no support yet for different shaped budgets like
                        # flow_ja_face
                        continue

                    self.match_array_cells(
                        cell_list,
                        budget_array.shape,
                        budget_array,
                        kstp_kper,
                        data_output,
                    )
                if len(data_output.data_entries) > 0:
                    budget_matches.append(data_output)
            if len(budget_matches) > 0:
                output_by_package["budget output"] = budget_matches

        if len(output_by_package) > 0 and output_file_path is not None:
            with open(output_file_path, "w") as fd:
                # write document header
                fd.write(f"Inspect cell results for model {self.name}\n")
                output = []
                for cell in cell_list:
                    output.append(" ".join([str(i) for i in cell]))
                output = ",".join(output)
                fd.write(f"Model cells inspected,{output}\n\n")

                for package_name, matches in output_by_package.items():
                    fd.write(f"Results from {package_name}\n")
                    for search_output in matches:
                        # write header line with data name
                        fd.write(
                            f",Results from "
                            f"{search_output.path_to_data[-1]}\n"
                        )
                        # write data header
                        if search_output.transient:
                            if search_output.output:
                                fd.write(",stress_period,time_step")
                            else:
                                fd.write(",stress_period/key")
                        if search_output.data_header is not None:
                            if len(search_output.data_entry_cellids) > 0:
                                fd.write(",cellid")
                            h_columns = ",".join(search_output.data_header)
                            fd.write(f",{h_columns}\n")
                        else:
                            fd.write(f",cellid,data\n")
                        # write data found
                        for index, data_entry in enumerate(
                            search_output.data_entries
                        ):
                            if search_output.transient:
                                sp = search_output.data_entry_stress_period[
                                    index
                                ]
                                if search_output.output:
                                    fd.write(f",{sp[1]},{sp[0]}")
                                else:
                                    fd.write(f",{sp}")
                            if search_output.data_header is not None:
                                if len(search_output.data_entry_cellids) > 0:
                                    cells = search_output.data_entry_cellids[
                                        index
                                    ]
                                    output = " ".join([str(i) for i in cells])
                                    fd.write(f",{output}")
                                fd.write(self._format_data_entry(data_entry))
                            else:
                                output = " ".join(
                                    [
                                        str(i)
                                        for i in search_output.data_entry_ids[
                                            index
                                        ]
                                    ]
                                )
                                fd.write(f",{output}")
                                fd.write(self._format_data_entry(data_entry))
                    fd.write(f"\n")
        return output_by_package

    def match_array_cells(
        self, cell_list, data_shape, array_data, key, data_output
    ):
        # loop through list of cells we are searching for
        for cell in cell_list:
            if len(data_shape) == 3 or data_shape[0] == "nodes":
                # data is by cell
                if array_data.ndim == 3 and len(cell) == 3:
                    data_output.data_entries.append(
                        array_data[cell[0], cell[1], cell[2]]
                    )
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)
                elif array_data.ndim == 2 and len(cell) == 2:
                    data_output.data_entries.append(
                        array_data[cell[0], cell[1]]
                    )
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)
                elif array_data.ndim == 1 and len(cell) == 1:
                    data_output.data_entries.append(array_data[cell[0]])
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)
                else:
                    if (
                        self.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        warning_str = (
                            'WARNING: CellID "{}" not same '
                            "number of dimensions as data "
                            "{}.".format(cell, data_output.path_to_data)
                        )
                        print(warning_str)
            elif len(data_shape) == 2:
                # get data based on ncpl/lay
                if array_data.ndim == 2 and len(cell) == 2:
                    data_output.data_entries.append(
                        array_data[cell[0], cell[1]]
                    )
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)
                elif array_data.ndim == 1 and len(cell) == 1:
                    data_output.data_entries.append(array_data[cell[0]])
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)
            elif len(data_shape) == 1:
                # get data based on nodes
                if len(cell) == 1 and array_data.ndim == 1:
                    data_output.data_entries.append(array_data[cell[0]])
                    data_output.data_entry_ids.append(cell)
                    data_output.data_entry_stress_period.append(key)

    @staticmethod
    def _format_data_entry(data_entry):
        output = ""
        if iterable(data_entry, True):
            for item in data_entry:
                if isinstance(item, tuple):
                    formatted = " ".join([str(i) for i in item])
                    output = f"{output},{formatted}"
                else:
                    output = f"{output},{item}"
            return f"{output}\n"
        else:
            return f",{data_entry}\n"

    def write(self, ext_file_action=ExtFileAction.copy_relative_paths):
        """
        Writes out model's package files.

        Parameters
        ----------
        ext_file_action : ExtFileAction
            Defines what to do with external files when the simulation path has
            changed.  defaults to copy_relative_paths which copies only files
            with relative paths, leaving files defined by absolute paths fixed.

        """

        # write name file
        if (
            self.simulation_data.verbosity_level.value
            >= VerbosityLevel.normal.value
        ):
            print("    writing model name file...")

        self.name_file.write(ext_file_action=ext_file_action)

        # write packages
        for pp in self.packagelist:
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(f"    writing package {pp._get_pname()}...")
            pp.write(ext_file_action=ext_file_action)

    def get_grid_type(self):
        """
        Return the type of grid used by model 'model_name' in simulation
        containing simulation data 'simulation_data'.

        Returns
        -------
        grid type : DiscretizationType
        """
        package_recarray = self.name_file.packages
        structure = mfstructure.MFStructure()
        if (
            package_recarray.search_data(
                f"dis{structure.get_version_string()}", 0
            )
            is not None
        ):
            return DiscretizationType.DIS
        elif (
            package_recarray.search_data(
                f"disv{structure.get_version_string()}", 0
            )
            is not None
        ):
            return DiscretizationType.DISV
        elif (
            package_recarray.search_data(
                f"disu{structure.get_version_string()}", 0
            )
            is not None
        ):
            return DiscretizationType.DISU
        elif (
            package_recarray.search_data(
                f"disl{structure.get_version_string()}", 0
            )
            is not None
        ):
            return DiscretizationType.DISL

        return DiscretizationType.UNDEFINED

    def get_ims_package(self):
        """Get the IMS package associated with this model.

        Returns
        -------
        IMS package : ModflowIms
        """
        solution_group = self.simulation.name_file.solutiongroup.get_data()
        for record in solution_group:
            for model_name in record[2:]:
                if model_name == self.name:
                    return self.simulation.get_ims_package(record[1])
        return None

    def get_steadystate_list(self):
        """Returns a list of stress periods that are steady state.

        Returns
        -------
        steady state list : list

        """
        ss_list = []
        tdis = self.simulation.get_package("tdis")
        period_data = tdis.perioddata.get_data()
        index = 0
        pd_len = len(period_data)
        while index < pd_len:
            ss_list.append(True)
            index += 1

        storage = self.get_package("sto")
        if storage is not None:
            tr_keys = storage.transient.get_keys(True)
            ss_keys = storage.steady_state.get_keys(True)
            for key in tr_keys:
                ss_list[key] = False
                for ss_list_key in range(key + 1, len(ss_list)):
                    for ss_key in ss_keys:
                        if ss_key == ss_list_key:
                            break
                        ss_list[key] = False
        return ss_list

    def is_valid(self):
        """
        Checks the validity of the model and all of its packages

        Returns
        -------
        valid : bool

        """

        # valid name file
        if not self.name_file.is_valid():
            return False

        # valid packages
        for pp in self.packagelist:
            if not pp.is_valid():
                return False

        # required packages exist
        for package_struct in self.structure.package_struct_objs.values():
            if (
                not package_struct.optional
                and not package_struct.file_type in self.package_type_dict
            ):
                return False

        return True

    def set_model_relative_path(self, model_ws):
        """
        Sets the file path to the model folder relative to the simulation
        folder and updates all model file paths, placing them in the model
        folder.

        Parameters
        ----------
        model_ws : str
            Model working folder relative to simulation working folder

        """
        # set all data internal
        self.set_all_data_internal(False)

        # update path in the file manager
        file_mgr = self.simulation_data.mfpath
        file_mgr.set_last_accessed_model_path()
        path = model_ws
        file_mgr.model_relative_path[self.name] = path

        if (
            model_ws
            and model_ws != "."
            and self.simulation.name_file is not None
        ):
            model_folder_path = file_mgr.get_model_path(self.name)
            if not os.path.exists(model_folder_path):
                # make new model folder
                os.makedirs(model_folder_path)
            # update model name file location in simulation name file
            models = self.simulation.name_file.models
            models_data = models.get_data()
            for index, entry in enumerate(models_data):
                old_model_file_name = os.path.split(entry[1])[1]
                old_model_base_name = os.path.splitext(old_model_file_name)[0]
                if (
                    old_model_base_name.lower() == self.name.lower()
                    or self.name == entry[2]
                ):
                    models_data[index][1] = os.path.join(
                        path, old_model_file_name
                    )
                    break
            models.set_data(models_data)

            if self.name_file is not None:
                # update listing file location in model name file
                list_file = self.name_file.list.get_data()
                if list_file:
                    path, list_file_name = os.path.split(list_file)
                    try:
                        self.name_file.list.set_data(
                            os.path.join(path, list_file_name)
                        )
                    except MFDataException as mfde:
                        message = (
                            "Error occurred while setting relative "
                            'path "{}" in model '
                            '"{}".'.format(
                                os.path.join(path, list_file_name), self.name
                            )
                        )
                        raise MFDataException(
                            mfdata_except=mfde,
                            model=self.model_name,
                            package=self.name_file._get_pname(),
                            message=message,
                        )
                # update package file locations in model name file
                packages = self.name_file.packages
                packages_data = packages.get_data()
                if packages_data is not None:
                    for index, entry in enumerate(packages_data):
                        # get package object associated with entry
                        package = None
                        if len(entry) >= 3:
                            package = self.get_package(entry[2])
                        if package is None:
                            package = self.get_package(entry[0])
                        if package is not None:
                            # combine model relative path with package path
                            packages_data[index][1] = os.path.join(
                                path, package.filename
                            )
                        else:
                            # package not found, create path based on
                            # information in name file
                            old_package_name = os.path.split(entry[1])[-1]
                            packages_data[index][1] = os.path.join(
                                path, old_package_name
                            )
                    packages.set_data(packages_data)
                # update files referenced from within packages
                for package in self.packagelist:
                    package.set_model_relative_path(model_ws)

    def _remove_package_from_dictionaries(self, package):
        # remove package from local dictionaries and lists
        if package.path in self._package_paths:
            del self._package_paths[package.path]
        self._remove_package(package)

    def remove_package(self, package_name):
        """
        Removes package and all child packages from the model.
        `package_name` can be the package's name, type, or package object to
        be removed from the model.

        Parameters
        ----------
        package_name : str
            Package name, package type, or package object to be removed from
            the model.

        """
        if isinstance(package_name, MFPackage):
            packages = [package_name]
        else:
            packages = self.get_package(package_name)
            if not isinstance(packages, list) and packages is not None:
                packages = [packages]
        if packages is None:
            return
        for package in packages:
            if package.model_or_sim.name != self.name:
                except_text = (
                    "Package can not be removed from model "
                    "{self.model_name} since it is not part of it."
                )
                raise mfstructure.FlopyException(except_text)

            self._remove_package_from_dictionaries(package)

            try:
                # remove package from name file
                package_data = self.name_file.packages.get_data()
            except MFDataException as mfde:
                message = (
                    "Error occurred while reading package names "
                    "from name file in model "
                    f'"{self.name}"'
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    model=self.model_name,
                    package=self.name_file._get_pname(),
                    message=message,
                )
            try:
                new_rec_array = None
                for item in package_data:
                    filename = os.path.basename(item[1])
                    if filename != package.filename:
                        if new_rec_array is None:
                            new_rec_array = np.rec.array(
                                [item.tolist()], package_data.dtype
                            )
                        else:
                            new_rec_array = np.hstack((item, new_rec_array))
            except:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "building package recarray",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    self._simulation_data.debug,
                )
            try:
                self.name_file.packages.set_data(new_rec_array)
            except MFDataException as mfde:
                message = (
                    "Error occurred while setting package names "
                    f'from name file in model "{self.name}".  Package name '
                    f"data:\n{new_rec_array}"
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    model=self.model_name,
                    package=self.name_file._get_pname(),
                    message=message,
                )

            # build list of child packages
            child_package_list = []
            for pkg in self.packagelist:
                if (
                    pkg.parent_file is not None
                    and pkg.parent_file.path == package.path
                ):
                    child_package_list.append(pkg)
            # remove child packages
            for child_package in child_package_list:
                self._remove_package_from_dictionaries(child_package)

    def update_package_filename(self, package, new_name):
        """
        Updates the filename for a package.  For internal flopy use only.

        Parameters
        ----------
        package : MFPackage
            Package object
        new_name : str
            New package name
        """
        try:
            # get namefile package data
            package_data = self.name_file.packages.get_data()
        except MFDataException as mfde:
            message = (
                "Error occurred while updating package names "
                "from name file in model "
                f'"{self.name}".'
            )
            raise MFDataException(
                mfdata_except=mfde,
                model=self.model_name,
                package=self.name_file._get_pname(),
                message=message,
            )
        try:
            file_mgr = self.simulation_data.mfpath
            model_rel_path = file_mgr.model_relative_path[self.name]
            # update namefile package data with new name
            new_rec_array = None
            old_leaf = os.path.split(package.filename)[1]
            for item in package_data:
                leaf = os.path.split(item[1])[1]
                if leaf == old_leaf:
                    item[1] = os.path.join(model_rel_path, new_name)

                if new_rec_array is None:
                    new_rec_array = np.rec.array(
                        [item.tolist()], package_data.dtype
                    )
                else:
                    new_rec_array = np.hstack((item, new_rec_array))
        except:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "updating package filename",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
            )
        try:
            self.name_file.packages.set_data(new_rec_array)
        except MFDataException as mfde:
            message = (
                "Error occurred while updating package names "
                f'from name file in model "{self.name}".  Package name '
                f"data:\n{new_rec_array}"
            )
            raise MFDataException(
                mfdata_except=mfde,
                model=self.model_name,
                package=self.name_file._get_pname(),
                message=message,
            )

    def rename_all_packages(self, name):
        """Renames all package files in the model.

        Parameters
        ----------
            name : str
                Prefix of package names.  Packages files will be named
                <name>.<package ext>.

        """
        nam_filename = f"{name}.nam"
        self.simulation.rename_model_namefile(self, nam_filename)
        self.name_file.filename = nam_filename
        self.model_nam_file = nam_filename
        package_type_count = {}
        for package in self.packagelist:
            if package.package_type not in package_type_count:
                base_filename, leaf = os.path.split(package.filename)
                lleaf = leaf.split(".")
                if len(lleaf) > 1:
                    # keep existing extension
                    ext = lleaf[-1]
                else:
                    # no extension found, create a new one
                    ext = package.package_type
                new_fileleaf = f"{name}.{ext}"
                if base_filename != "":
                    package.filename = os.path.join(
                        base_filename, new_fileleaf
                    )
                else:
                    package.filename = new_fileleaf
                package_type_count[package.package_type] = 1
            else:
                package_type_count[package.package_type] += 1
                package.filename = "{}_{}.{}".format(
                    name,
                    package_type_count[package.package_type],
                    package.package_type,
                )

    def set_all_data_external(
        self, check_data=True, external_data_folder=None
    ):
        """Sets the model's list and array data to be stored externally.

        Parameters
        ----------
            check_data : bool
                Determines if data error checking is enabled during this
                process.
            external_data_folder
                Folder, relative to the simulation path or model relative path
                (see use_model_relative_path parameter), where external data
                will be stored

        """
        for package in self.packagelist:
            package.set_all_data_external(check_data, external_data_folder)

    def set_all_data_internal(self, check_data=True):
        """Sets the model's list and array data to be stored externally.

        Parameters
        ----------
            check_data : bool
                Determines if data error checking is enabled during this
                process.

        """
        for package in self.packagelist:
            package.set_all_data_internal(check_data)

    def register_package(
        self,
        package,
        add_to_package_list=True,
        set_package_name=True,
        set_package_filename=True,
    ):
        """
        Registers a package with the model.  This method is used internally
        by FloPy and is not intended for use by the end user.

        Parameters
        ----------
        package : MFPackage
            Package to register
        add_to_package_list : bool
            Add package to lookup list
        set_package_name : bool
            Produce a package name for this package
        set_package_filename : bool
            Produce a filename for this package

        Returns
        -------
        path, package structure : tuple, MFPackageStructure

        """
        package.container_type = [PackageContainerType.model]
        if package.parent_file is not None:
            path = package.parent_file.path + (package.package_type,)
        else:
            path = (self.name, package.package_type)
        package_struct = self.structure.get_package_struct(
            package.package_type
        )
        if add_to_package_list and path in self._package_paths:
            if (
                package_struct is not None
                and not package_struct.multi_package_support
                and not isinstance(package.parent_file, MFPackage)
            ):
                # package of this type already exists, replace it
                self.remove_package(package.package_type)
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: Package with type {} already exists. "
                        "Replacing existing package"
                        ".".format(package.package_type)
                    )
            elif (
                not set_package_name
                and package.package_name in self.package_name_dict
            ):
                # package of this type with this name already
                # exists, replace it
                self.remove_package(
                    self.package_name_dict[package.package_name]
                )
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: Package with name {} already exists. "
                        "Replacing existing package"
                        ".".format(package.package_name)
                    )

        # make sure path is unique
        if path in self._package_paths:
            path_iter = datautil.PathIter(path)
            for new_path in path_iter:
                if new_path not in self._package_paths:
                    path = new_path
                    break
        self._package_paths[path] = 1

        if package.package_type.lower() == "nam":
            if not package.internal_package:
                excpt_str = (
                    "Unable to register nam file.  Do not create your own nam "
                    "files.  Nam files are automatically created and managed "
                    "for you by FloPy."
                )
                print(excpt_str)
                raise FlopyException(excpt_str)

            return path, self.structure.name_file_struct_obj

        package_extension = package.package_type
        if set_package_name:
            # produce a default package name
            if (
                package_struct is not None
                and package_struct.multi_package_support
            ):
                # check for other registered packages of this type
                name_iter = datautil.NameIter(package.package_type, False)
                for package_name in name_iter:
                    if package_name not in self.package_name_dict:
                        package.package_name = package_name
                        suffix = package_name.split("_")
                        if (
                            len(suffix) > 1
                            and datautil.DatumUtil.is_int(suffix[-1])
                            and suffix[-1] != "0"
                        ):
                            # update file extension to make unique
                            package_extension = (
                                f"{package_extension}_{suffix[-1]}"
                            )
                        break
            else:
                package.package_name = package.package_type

        if set_package_filename:
            # filename uses model base name
            package._filename = f"{self.name}.{package.package_type}"
            if package._filename in self.package_filename_dict:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(
                    package._filename, self.package_filename_dict
                )
                package._filename = file_name

        if add_to_package_list:
            self._add_package(package, path)

            # add obs file to name file if it does not have a parent
            if package.package_type in self.structure.package_struct_objs or (
                package.package_type == "obs" and package.parent_file is None
            ):
                # update model name file
                pkg_type = package.package_type.upper()
                if len(pkg_type) > 3 and pkg_type[-1] == "A":
                    pkg_type = pkg_type[0:-1]
                # Model Assumption - assuming all name files have a package
                # recarray
                file_mgr = self.simulation_data.mfpath
                model_rel_path = file_mgr.model_relative_path[self.name]
                if model_rel_path != ".":
                    package_rel_path = os.path.join(
                        model_rel_path, package.filename
                    )
                else:
                    package_rel_path = package.filename
                self.name_file.packages.update_record(
                    [
                        f"{pkg_type}6",
                        package_rel_path,
                        package.package_name,
                    ],
                    0,
                )
        if package_struct is not None:
            return (path, package_struct)
        else:
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "WARNING: Unable to register unsupported file type {} "
                    "for model {}.".format(package.package_type, self.name)
                )
        return None, None

    def load_package(
        self,
        ftype,
        fname,
        pname,
        strict,
        ref_path,
        dict_package_name=None,
        parent_package=None,
    ):
        """
        Loads a package from a file.  This method is used internally by FloPy
        and is not intended for the end user.

        Parameters
        ----------
        ftype : str
            the file type
        fname : str
            the name of the file containing the package input
        pname : str
            the user-defined name for the package
        strict : bool
            strict mode when loading the file
        ref_path : str
            path to the file. uses local path if set to None
        dict_package_name : str
            package name for dictionary lookup
        parent_package : MFPackage
            parent package

        Examples
        --------
        """
        if ref_path is not None:
            fname = os.path.join(ref_path, fname)
        sim_struct = mfstructure.MFStructure().sim_struct
        if (
            ftype in self.structure.package_struct_objs
            and self.structure.package_struct_objs[ftype].multi_package_support
        ) or (
            ftype in sim_struct.utl_struct_objs
            and sim_struct.utl_struct_objs[ftype].multi_package_support
        ):
            # resolve dictionary name for package
            if dict_package_name is not None:
                if parent_package is not None:
                    dict_package_name = f"{parent_package.path[-1]}_{ftype}"
                else:
                    # use dict_package_name as the base name
                    if ftype in self._ftype_num_dict:
                        self._ftype_num_dict[dict_package_name] += 1
                    else:
                        self._ftype_num_dict[dict_package_name] = 0
                    dict_package_name = "{}_{}".format(
                        dict_package_name,
                        self._ftype_num_dict[dict_package_name],
                    )
            else:
                # use ftype as the base name
                if ftype in self._ftype_num_dict:
                    self._ftype_num_dict[ftype] += 1
                else:
                    self._ftype_num_dict[ftype] = 1
                if pname is not None:
                    dict_package_name = pname
                else:
                    dict_package_name = (
                        f"{ftype}-{self._ftype_num_dict[ftype]}"
                    )
        else:
            dict_package_name = ftype

        # clean up model type text
        model_type = self.structure.model_type
        while datautil.DatumUtil.is_int(model_type[-1]):
            model_type = model_type[0:-1]

        # create package
        package_obj = self.package_factory(ftype, model_type)
        package = package_obj(
            self,
            filename=fname,
            pname=dict_package_name,
            loading_package=True,
            parent_file=parent_package,
            _internal_package=True,
        )
        try:
            package.load(strict)
        except ReadAsArraysException:
            #  create ReadAsArrays package and load it instead
            package_obj = self.package_factory(f"{ftype}a", model_type)
            package = package_obj(
                self,
                filename=fname,
                pname=dict_package_name,
                loading_package=True,
                parent_file=parent_package,
                _internal_package=True,
            )
            package.load(strict)

        # register child package with the model
        self._add_package(package, package.path)
        if parent_package is not None:
            # register child package with the parent package
            parent_package._add_package(package, package.path)

        return package

    def plot(self, SelPackList=None, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        model input data from a model instance

        Args:
            model: Flopy model instance
            SelPackList: (list) list of package names to plot, if none
                all packages will be plotted

            **kwargs : dict
                filename_base : str
                    Base file name that will be used to automatically generate file
                    names for output image files. Plots will be exported as image
                    files if file_name_base is not None. (default is None)
                file_extension : str
                    Valid matplotlib.pyplot file extension for savefig(). Only used
                    if filename_base is not None. (default is 'png')
                mflay : int
                    MODFLOW zero-based layer number to return.  If None, then all
                    all layers will be included. (default is None)
                kper : int
                    MODFLOW zero-based stress period number to return.
                    (default is zero)
                key : str
                    MfList dictionary key. (default is None)

        Returns:
            axes : list
                Empty list is returned if filename_base is not None. Otherwise
                a list of matplotlib.pyplot.axis are returned.
        """
        from ..plot.plotutil import PlotUtilities

        axes = PlotUtilities._plot_model_helper(
            self, SelPackList=SelPackList, **kwargs
        )

        return axes
