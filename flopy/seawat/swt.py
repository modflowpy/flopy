import os
from ..mbase import BaseModel
from ..pakbase import Package
from ..modflow import Modflow
from ..mt3d import Mt3dms
from .swtvdf import SeawatVdf
from .swtvsc import SeawatVsc
from ..discretization.structuredgrid import StructuredGrid
from flopy.discretization.modeltime import ModelTime


class SeawatList(Package):
    """
    List Package class
    """

    def __init__(self, model, extension="list", listunit=7):
        Package.__init__(self, model, extension, "LIST", listunit)
        return

    def __repr__(self):
        return "List package class"

    def write_file(self):
        # Not implemented for list class
        return


class Seawat(BaseModel):
    """
    SEAWAT Model Class.

    Parameters
    ----------
    modelname : str, default "swttest"
        Name of model.  This string will be used to name the SEAWAT input
        that are created with write_model.
    namefile_ext : str, default "nam"
        Extension for the namefile.
    modflowmodel : Modflow, default None
        Instance of a Modflow object.
    mt3dmodel : Mt3dms, default None
        Instance of a Mt3dms object.
    version : str, default "seawat"
        Version of SEAWAT to use. Valid versions are "seawat" (default).
    exe_name : str, default "swtv4"
        The name of the executable to use.
    structured : bool, default True
        Specify if model grid is structured (default) or unstructured.
    listunit : int, default 2
        Unit number for the list file.
    model_ws : str, default "."
        Model workspace.  Directory name to create model data sets.
        Default is the present working directory.
    external_path : str, optional
        Location for external files.
    verbose : bool, default False
        Print additional information to the screen.
    load : bool, default True
         Load model.
    silent : int, default 0
        Silent option.

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
    >>> m = flopy.seawat.swt.Seawat()

    """

    def __init__(
        self,
        modelname="swttest",
        namefile_ext="nam",
        modflowmodel=None,
        mt3dmodel=None,
        version="seawat",
        exe_name="swtv4",
        structured=True,
        listunit=2,
        model_ws=".",
        external_path=None,
        verbose=False,
        load=True,
        silent=0,
    ):
        super().__init__(
            modelname,
            namefile_ext,
            exe_name,
            model_ws,
            structured=structured,
            verbose=verbose,
        )

        # Set attributes
        self.version_types = {"seawat": "SEAWAT"}
        self.set_version(version)
        self.lst = SeawatList(self, listunit=listunit)
        self.glo = None
        self._mf = None
        self._mt = None

        # If a MODFLOW model was passed in, then add its packages
        self.mf = self
        if modflowmodel is not None:
            for p in modflowmodel.packagelist:
                self.packagelist.append(p)
            self._modelgrid = modflowmodel.modelgrid
        else:
            modflowmodel = Modflow()

        # If a MT3D model was passed in, then add its packages
        if mt3dmodel is not None:
            for p in mt3dmodel.packagelist:
                self.packagelist.append(p)
        else:
            mt3dmodel = Mt3dms()

        # external option stuff
        self.array_free_format = False
        self.array_format = "mt3d"
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.external = False
        self.load = load
        # the starting external data unit number
        self._next_ext_unit = 3000
        if external_path is not None:
            assert (
                model_ws == "."
            ), "ERROR: external cannot be used with model_ws"

            # external_path = os.path.join(model_ws, external_path)
            if os.path.exists(external_path):
                print(
                    "Note: external_path "
                    + str(external_path)
                    + " already exists"
                )
            # assert os.path.exists(external_path),'external_path does not exist'
            else:
                os.mkdir(external_path)
            self.external = True
        self.external_path = external_path
        self.verbose = verbose
        self.silent = silent

        # Create a dictionary to map package with package object.
        # This is used for loading models.
        self.mfnam_packages = {}
        for k, v in modflowmodel.mfnam_packages.items():
            self.mfnam_packages[k] = v
        for k, v in mt3dmodel.mfnam_packages.items():
            self.mfnam_packages[k] = v
        self.mfnam_packages["vdf"] = SeawatVdf
        self.mfnam_packages["vsc"] = SeawatVsc
        return

    @property
    def modeltime(self):
        # build model time
        data_frame = {
            "perlen": self.dis.perlen.array,
            "nstp": self.dis.nstp.array,
            "tsmult": self.dis.tsmult.array,
        }
        self._model_time = ModelTime(
            data_frame,
            self.dis.itmuni_dict[self.dis.itmuni],
            self.dis.start_datetime,
            self.dis.steady.array,
        )
        return self._model_time

    @property
    def modelgrid(self):
        if not self._mg_resync:
            return self._modelgrid

        if self.has_package("bas6"):
            ibound = self.bas6.ibound.array
        else:
            ibound = None
        # build grid
        # self.dis should exist if modflow model passed
        self._modelgrid = StructuredGrid(
            self.dis.delc.array,
            self.dis.delr.array,
            self.dis.top.array,
            self.dis.botm.array,
            idomain=ibound,
            lenuni=self.dis.lenuni,
            proj4=self._modelgrid.proj4,
            epsg=self._modelgrid.epsg,
            xoff=self._modelgrid.xoffset,
            yoff=self._modelgrid.yoffset,
            angrot=self._modelgrid.angrot,
            nlay=self.dis.nlay,
        )

        # resolve offsets
        xoff = self._modelgrid.xoffset
        if xoff is None:
            if self._xul is not None:
                xoff = self._modelgrid._xul_to_xll(self._xul)
            else:
                xoff = 0.0
        yoff = self._modelgrid.yoffset
        if yoff is None:
            if self._yul is not None:
                yoff = self._modelgrid._yul_to_yll(self._yul)
            else:
                yoff = 0.0
        self._modelgrid.set_coord_info(
            xoff,
            yoff,
            self._modelgrid.angrot,
            self._modelgrid.epsg,
            self._modelgrid.proj4,
        )
        self._mg_resync = not self._modelgrid.is_complete
        return self._modelgrid

    @property
    def nlay(self):
        if self.dis:
            return self.dis.nlay
        else:
            return 0

    @property
    def nrow(self):
        if self.dis:
            return self.dis.nrow
        else:
            return 0

    @property
    def ncol(self):
        if self.dis:
            return self.dis.ncol
        else:
            return 0

    @property
    def nper(self):
        if self.dis:
            return self.dis.nper
        else:
            return 0

    @property
    def nrow_ncol_nlay_nper(self):
        dis = self.get_package("DIS")
        if dis:
            return dis.nrow, dis.ncol, dis.nlay, dis.nper
        else:
            return 0, 0, 0, 0

    def get_nrow_ncol_nlay_nper(self):
        return self.nrow_ncol_nlay_nper

    def get_ifrefm(self):
        bas = self.get_package("BAS6")
        if bas:
            return bas.ifrefm
        else:
            return False

    @property
    def ncomp(self):
        if self.btn:
            return self.btn.ncomp
        else:
            return 1

    @property
    def mcomp(self):
        if self.btn:
            return self.btn.mcomp
        else:
            return 1

    def _set_name(self, value):
        # Overrides BaseModel's setter for name property
        super()._set_name(value)

        # for i in range(len(self.lst.extension)):
        #    self.lst.file_name[i] = self.name + '.' + self.lst.extension[i]
        # return

    def change_model_ws(self, new_pth=None, reset_external=False):
        # if hasattr(self,"_mf"):
        if self._mf is not None:
            self._mf.change_model_ws(
                new_pth=new_pth, reset_external=reset_external
            )
        # if hasattr(self,"_mt"):
        if self._mt is not None:
            self._mt.change_model_ws(
                new_pth=new_pth, reset_external=reset_external
            )
        super().change_model_ws(new_pth=new_pth, reset_external=reset_external)

    def write_name_file(self):
        """
        Write the name file

        Returns
        -------
        None

        """
        # open and write header
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, "w")
        f_nam.write("{}\n".format(self.heading))

        # Write global file entry
        if self.glo is not None:
            if self.glo.unit_number[0] > 0:
                f_nam.write(
                    "{:14s} {:5d}  {}\n".format(
                        self.glo.name[0],
                        self.glo.unit_number[0],
                        self.glo.file_name[0],
                    )
                )
        # Write list file entry
        f_nam.write(
            "{:14s} {:5d}  {}\n".format(
                self.lst.name[0],
                self.lst.unit_number[0],
                self.lst.file_name[0],
            )
        )

        # Write SEAWAT entries and close
        f_nam.write(str(self.get_name_file_entries()))

        if self._mf is not None:
            # write the external files
            for b, u, f in zip(
                self._mf.external_binflag,
                self._mf.external_units,
                self._mf.external_fnames,
            ):
                tag = "DATA"
                if b:
                    tag = "DATA(BINARY)"
                f_nam.write("{0:14s} {1:5d}  {2}\n".format(tag, u, f))

            # write the output files
            for u, f, b in zip(
                self._mf.output_units,
                self._mf.output_fnames,
                self._mf.output_binflag,
            ):
                if u == 0:
                    continue
                if b:
                    f_nam.write(
                        "DATA(BINARY)   {:5d}  {} REPLACE\n".format(u, f)
                    )
                else:
                    f_nam.write("DATA           {:5d}  {}\n".format(u, f))

        if self._mt is not None:
            # write the external files
            for b, u, f in zip(
                self._mt.external_binflag,
                self._mt.external_units,
                self._mt.external_fnames,
            ):
                tag = "DATA"
                if b:
                    tag = "DATA(BINARY)"
                f_nam.write("{0:14s} {1:5d}  {2}\n".format(tag, u, f))

            # write the output files
            for u, f, b in zip(
                self._mt.output_units,
                self._mt.output_fnames,
                self._mt.output_binflag,
            ):
                if u == 0:
                    continue
                if b:
                    f_nam.write(
                        "DATA(BINARY)   {:5d}  {} REPLACE\n".format(u, f)
                    )
                else:
                    f_nam.write("DATA           {:5d}  {}\n".format(u, f))

        # write the external files
        for b, u, f in zip(
            self.external_binflag, self.external_units, self.external_fnames
        ):
            tag = "DATA"
            if b:
                tag = "DATA(BINARY)"
            f_nam.write("{0:14s} {1:5d}  {2}\n".format(tag, u, f))

        # write the output files
        for u, f, b in zip(
            self.output_units, self.output_fnames, self.output_binflag
        ):
            if u == 0:
                continue
            if b:
                f_nam.write("DATA(BINARY)   {:5d}  {} REPLACE\n".format(u, f))
            else:
                f_nam.write("DATA           {:5d}  {}\n".format(u, f))

        f_nam.close()
        return

    @classmethod
    def load(
        cls,
        f,
        version="seawat",
        exe_name="swtv4",
        verbose=False,
        model_ws=".",
        load_only=None,
    ):
        """
        Load an existing model.

        Parameters
        ----------
        f : str
            Path to SEAWAT name file to load.
        version : str, default "seawat"
            Version of SEAWAT to use. Valid versions are "seawat" (default).
        exe_name : str, default "swtv4"
            The name of the executable to use.
        verbose : bool, default False
            Print additional information to the screen.
        model_ws : str, default "."
            Model workspace.  Directory name to create model data sets.
            Default is the present working directory.
        load_only : list of str, optional
            Packages to load (e.g. ["lpf", "adv"]). Default None
            means that all packages will be loaded.

        Returns
        -------
        flopy.seawat.swt.Seawat

        Examples
        --------
        >>> import flopy
        >>> m = flopy.seawat.swt.Seawat.load(f)

        """
        # test if name file is passed with extension (i.e., is a valid file)
        if os.path.isfile(os.path.join(model_ws, f)):
            modelname = f.rpartition(".")[0]
        else:
            modelname = f

        # create instance of a seawat model and load modflow and mt3dms models
        ms = cls(
            modelname=modelname,
            namefile_ext="nam",
            modflowmodel=None,
            mt3dmodel=None,
            version=version,
            exe_name=exe_name,
            model_ws=model_ws,
            verbose=verbose,
        )

        mf = Modflow.load(
            f,
            version="mf2k",
            exe_name=None,
            verbose=verbose,
            model_ws=model_ws,
            load_only=load_only,
            forgive=False,
            check=False,
        )

        mt = Mt3dms.load(
            f,
            version="mt3dms",
            exe_name=None,
            verbose=verbose,
            model_ws=model_ws,
            forgive=False,
        )

        # set listing and global files using mf objects
        ms.lst = mf.lst
        ms.glo = mf.glo

        for p in mf.packagelist:
            p.parent = ms
            ms.add_package(p)
        ms._mt = None
        if mt is not None:
            for p in mt.packagelist:
                p.parent = ms
                ms.add_package(p)
            mt.external_units = []
            mt.external_binflag = []
            mt.external_fnames = []
            ms._mt = mt
        ms._mf = mf

        # return model object
        return ms
