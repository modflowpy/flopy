"""
mf module.  Contains the ModflowGlobal, ModflowList, and Modflow classes.


"""

import os
import flopy
from inspect import getfullargspec
from ..mbase import BaseModel
from ..pakbase import Package
from ..utils import mfreadnam
from ..discretization.structuredgrid import StructuredGrid
from ..discretization.unstructuredgrid import UnstructuredGrid
from ..discretization.grid import Grid
from flopy.discretization.modeltime import ModelTime
from .mfpar import ModflowPar


class ModflowGlobal(Package):
    """
    ModflowGlobal Package class

    """

    def __init__(self, model, extension="glo"):
        Package.__init__(self, model, extension, "GLOBAL", 1)
        return

    def __repr__(self):
        return "Global Package class"

    def write_file(self):
        # Not implemented for global class
        return


class ModflowList(Package):
    """
    ModflowList Package class

    """

    def __init__(self, model, extension="list", unitnumber=2):
        Package.__init__(self, model, extension, "LIST", unitnumber)
        return

    def __repr__(self):
        return "List Package class"

    def write_file(self):
        # Not implemented for list class
        return


class Modflow(BaseModel):
    """
    MODFLOW Model Class.

    Parameters
    ----------
    modelname : str, default "modflowtest"
        Name of model.  This string will be used to name the MODFLOW input
        that are created with write_model.
    namefile_ext : str, default "nam"
        Extension for the namefile.
    version : str, default "mf2005"
        MODFLOW version. Choose one of: "mf2k", "mf2005" (default),
        "mfnwt", or "mfusg".
    exe_name : str, default "mf2005.exe"
        The name of the executable to use.
    structured : bool, default True
        Specify if model grid is structured (default) or unstructured.
    listunit : int, default 2
        Unit number for the list file.
    model_ws : str, default "."
        Model workspace.  Directory name to create model data sets.
        (default is the present working directory).
    external_path : str, optional
        Location for external files.
    verbose : bool, default False
        Print additional information to the screen.

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
    >>> m = flopy.modflow.Modflow()

    """

    def __init__(
        self,
        modelname="modflowtest",
        namefile_ext="nam",
        version="mf2005",
        exe_name="mf2005.exe",
        structured=True,
        listunit=2,
        model_ws=".",
        external_path=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(
            modelname,
            namefile_ext,
            exe_name,
            model_ws,
            structured=structured,
            verbose=verbose,
            **kwargs
        )
        self.version_types = {
            "mf2k": "MODFLOW-2000",
            "mf2005": "MODFLOW-2005",
            "mfnwt": "MODFLOW-NWT",
            "mfusg": "MODFLOW-USG",
        }

        self.set_version(version)

        if self.version == "mf2k":
            self.glo = ModflowGlobal(self)

        self.lst = ModflowList(self, unitnumber=listunit)
        # -- check if unstructured is specified for something
        # other than mfusg is specified
        if not self.structured:
            assert (
                "mfusg" in self.version
            ), "structured=False can only be specified for mfusg models"

        # external option stuff
        self.array_free_format = True
        self.array_format = "modflow"
        # self.external_fnames = []
        # self.external_units = []
        # self.external_binflag = []

        self.load_fail = False
        # the starting external data unit number
        self._next_ext_unit = 1000

        if external_path is not None:
            if os.path.exists(os.path.join(model_ws, external_path)):
                print(
                    "Note: external_path "
                    + str(external_path)
                    + " already exists"
                )
            else:
                os.makedirs(os.path.join(model_ws, external_path))
        self.external_path = external_path
        self.verbose = verbose
        self.mfpar = ModflowPar()

        # output file info
        self.hext = "hds"
        self.dext = "ddn"
        self.cext = "cbc"
        self.hpth = None
        self.dpath = None
        self.cpath = None

        # Create a dictionary to map package with package object.
        # This is used for loading models.
        self.mfnam_packages = {
            "zone": flopy.modflow.ModflowZon,
            "mult": flopy.modflow.ModflowMlt,
            "ag": flopy.modflow.ModflowAg,
            "pval": flopy.modflow.ModflowPval,
            "bas6": flopy.modflow.ModflowBas,
            "dis": flopy.modflow.ModflowDis,
            "disu": flopy.modflow.ModflowDisU,
            "bcf6": flopy.modflow.ModflowBcf,
            "lpf": flopy.modflow.ModflowLpf,
            "hfb6": flopy.modflow.ModflowHfb,
            "chd": flopy.modflow.ModflowChd,
            "fhb": flopy.modflow.ModflowFhb,
            "wel": flopy.modflow.ModflowWel,
            "mnw1": flopy.modflow.ModflowMnw1,
            "mnw2": flopy.modflow.ModflowMnw2,
            "mnwi": flopy.modflow.ModflowMnwi,
            "drn": flopy.modflow.ModflowDrn,
            "drt": flopy.modflow.ModflowDrt,
            "rch": flopy.modflow.ModflowRch,
            "evt": flopy.modflow.ModflowEvt,
            "ghb": flopy.modflow.ModflowGhb,
            "gmg": flopy.modflow.ModflowGmg,
            "lmt6": flopy.modflow.ModflowLmt,
            "lmt7": flopy.modflow.ModflowLmt,
            "riv": flopy.modflow.ModflowRiv,
            "str": flopy.modflow.ModflowStr,
            "swi2": flopy.modflow.ModflowSwi2,
            "pcg": flopy.modflow.ModflowPcg,
            "pcgn": flopy.modflow.ModflowPcgn,
            "nwt": flopy.modflow.ModflowNwt,
            "pks": flopy.modflow.ModflowPks,
            "sms": flopy.modflow.ModflowSms,
            "sfr": flopy.modflow.ModflowSfr2,
            "lak": flopy.modflow.ModflowLak,
            "gage": flopy.modflow.ModflowGage,
            "sip": flopy.modflow.ModflowSip,
            "sor": flopy.modflow.ModflowSor,
            "de4": flopy.modflow.ModflowDe4,
            "oc": flopy.modflow.ModflowOc,
            "uzf": flopy.modflow.ModflowUzf1,
            "upw": flopy.modflow.ModflowUpw,
            "sub": flopy.modflow.ModflowSub,
            "swt": flopy.modflow.ModflowSwt,
            "hyd": flopy.modflow.ModflowHyd,
            "hob": flopy.modflow.ModflowHob,
            "chob": flopy.modflow.ModflowFlwob,
            "gbob": flopy.modflow.ModflowFlwob,
            "drob": flopy.modflow.ModflowFlwob,
            "rvob": flopy.modflow.ModflowFlwob,
            "vdf": flopy.seawat.SeawatVdf,
            "vsc": flopy.seawat.SeawatVsc,
        }
        return

    def __repr__(self):
        nrow, ncol, nlay, nper = self.get_nrow_ncol_nlay_nper()
        if nrow is not None:
            # structured case
            s = (
                "MODFLOW {} layer(s) {} row(s) {} column(s) "
                "{} stress period(s)".format(nlay, nrow, ncol, nper)
            )
        else:
            # unstructured case
            nodes = ncol.sum()
            nodelay = " ".join(str(i) for i in ncol)
            print(nodelay, nlay, nper)
            s = (
                "MODFLOW unstructured\n"
                "  nodes = {}\n"
                "  layers = {}\n"
                "  periods = {}\n"
                "  nodelay = {}\n".format(nodes, nlay, nper, ncol)
            )
        return s

    #
    # def next_ext_unit(self):
    #     """
    #     Function to encapsulate next_ext_unit attribute
    #
    #     """
    #     next_unit = self.__next_ext_unit + 1
    #     self.__next_ext_unit += 1
    #     return next_unit

    @property
    def modeltime(self):
        if self.get_package("disu") is not None:
            dis = self.disu
        else:
            dis = self.dis
        # build model time
        data_frame = {
            "perlen": dis.perlen.array,
            "nstp": dis.nstp.array,
            "tsmult": dis.tsmult.array,
        }
        self._model_time = ModelTime(
            data_frame,
            dis.itmuni_dict[dis.itmuni],
            dis.start_datetime,
            dis.steady.array,
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

        if self.get_package("disu") is not None:
            # build unstructured grid
            self._modelgrid = UnstructuredGrid(
                grid_type="unstructured",
                vertices=self._modelgrid.vertices,
                ivert=self._modelgrid.iverts,
                xcenters=self._modelgrid.xcenters,
                ycenters=self._modelgrid.ycenters,
                ncpl=self.disu.nodelay.array,
                top=self.disu.top.array,
                botm=self.disu.bot.array,
                idomain=ibound,
                lenuni=self.disu.lenuni,
                proj4=self._modelgrid.proj4,
                epsg=self._modelgrid.epsg,
                xoff=self._modelgrid.xoffset,
                yoff=self._modelgrid.yoffset,
                angrot=self._modelgrid.angrot,
            )
            print(
                "WARNING: Model grid functionality limited for unstructured "
                "grid."
            )
        else:
            # build structured grid
            self._modelgrid = StructuredGrid(
                self.dis.delc.array,
                self.dis.delr.array,
                self.dis.top.array,
                self.dis.botm.array,
                ibound,
                self.dis.lenuni,
                proj4=self._modelgrid.proj4,
                epsg=self._modelgrid.epsg,
                xoff=self._modelgrid.xoffset,
                yoff=self._modelgrid.yoffset,
                angrot=self._modelgrid.angrot,
                nlay=self.dis.nlay,
                laycbd=self.dis.laycbd,
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

    @modelgrid.setter
    def modelgrid(self, value):
        self._mg_resync = False
        self._modelgrid = value

    @property
    def solver_tols(self):
        if self.pcg is not None:
            return self.pcg.hclose, self.pcg.rclose
        elif self.nwt is not None:
            return self.nwt.headtol, self.nwt.fluxtol
        elif self.sip is not None:
            return self.sip.hclose, -999
        elif self.gmg is not None:
            return self.gmg.hclose, self.gmg.rclose
        return None

    @property
    def nlay(self):
        if self.dis:
            return self.dis.nlay
        elif self.disu:
            return self.disu.nlay
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
        elif self.disu:
            return self.disu.nper
        else:
            return 0

    @property
    def ncpl(self):
        if self.dis:
            return self.dis.nrow * self.dis.ncol
        elif self.disu:
            return self.disu.ncpl
        else:
            return 0

    @property
    def nrow_ncol_nlay_nper(self):
        # structured dis
        dis = self.get_package("DIS")
        if dis:
            return dis.nrow, dis.ncol, dis.nlay, dis.nper
        # unstructured dis
        dis = self.get_package("DISU")
        if dis:
            return None, dis.nodelay.array[:], dis.nlay, dis.nper
        # no dis
        return 0, 0, 0, 0

    def get_nrow_ncol_nlay_nper(self):
        return self.nrow_ncol_nlay_nper

    def get_ifrefm(self):
        bas = self.get_package("BAS6")
        if bas:
            return bas.ifrefm
        else:
            return False

    def set_ifrefm(self, value=True):
        if not isinstance(value, bool):
            print("Error: set_ifrefm passed value must be a boolean")
            return False
        self.array_free_format = value
        bas = self.get_package("BAS6")
        if bas:
            bas.ifrefm = value
        else:
            return False

    def _set_name(self, value):
        # Overrides BaseModel's setter for name property
        super()._set_name(value)

        if self.version == "mf2k":
            for i in range(len(self.glo.extension)):
                self.glo.file_name[i] = self.name + "." + self.glo.extension[i]

        for i in range(len(self.lst.extension)):
            self.lst.file_name[i] = self.name + "." + self.lst.extension[i]

    def write_name_file(self):
        """
        Write the model name file.

        """
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, "w")
        f_nam.write("{}\n".format(self.heading))
        f_nam.write("#" + str(self.modelgrid))
        f_nam.write("; start_datetime:{0}\n".format(self.start_datetime))
        if self.version == "mf2k":
            if self.glo.unit_number[0] > 0:
                f_nam.write(
                    "{:14s} {:5d}  {}\n".format(
                        self.glo.name[0],
                        self.glo.unit_number[0],
                        self.glo.file_name[0],
                    )
                )
        f_nam.write(
            "{:14s} {:5d}  {}\n".format(
                self.lst.name[0],
                self.lst.unit_number[0],
                self.lst.file_name[0],
            )
        )
        f_nam.write(str(self.get_name_file_entries()))

        # write the external files
        for u, f, b, o in zip(
            self.external_units,
            self.external_fnames,
            self.external_binflag,
            self.external_output,
        ):
            if u == 0:
                continue
            replace_text = ""
            if o:
                replace_text = " REPLACE"
            if b:
                line = "DATA(BINARY)   {:5d}  {}{}\n".format(
                    u, f, replace_text
                )

                f_nam.write(line)
            else:
                f_nam.write("DATA           {:5d}  {}\n".format(u, f))

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

        # close the name file
        f_nam.close()
        return

    def set_model_units(self, iunit0=None):
        """
        Write the model name file.

        """
        if iunit0 is None:
            iunit0 = 1001

        # initialize starting unit number
        self.next_unit(iunit0)

        if self.version == "mf2k":
            # update global file unit number
            if self.glo.unit_number[0] > 0:
                self.glo.unit_number[0] = self.next_unit()

        # update lst file unit number
        self.lst.unit_number[0] = self.next_unit()

        # update package unit numbers
        for p in self.packagelist:
            p.unit_number[0] = self.next_unit()

        # update external unit numbers
        for i, iu in enumerate(self.external_units):
            if iu == 0:
                continue
            self.external_units[i] = self.next_unit()

        # update output files unit numbers
        oc = self.get_package("OC")
        output_units0 = list(self.output_units)
        for i, iu in enumerate(self.output_units):
            if iu == 0:
                continue
            iu1 = self.next_unit()
            self.output_units[i] = iu1
            # update oc files
            if oc is not None:
                if oc.iuhead == iu:
                    oc.iuhead = iu1
                elif oc.iuddn == iu:
                    oc.iuddn = iu1

        # replace value in ipakcb
        for p in self.packagelist:
            try:
                iu0 = p.ipakcb
                if iu0 in output_units0:
                    j = output_units0.index(iu0)
                    p.ipakcb = self.output_units[j]
            except:
                if self.verbose:
                    print("   could not replace value in ipakcb")

        return

    def load_results(self, **kwargs):

        # remove model if passed as a kwarg
        if "model" in kwargs:
            kwargs.pop("model")

        as_dict = False
        if "as_dict" in kwargs:
            as_dict = bool(kwargs.pop("as_dict"))

        savehead = False
        saveddn = False
        savebud = False

        # check for oc
        try:
            oc = self.get_package("OC")
            self.hext = oc.extension[1]
            self.dext = oc.extension[2]
            self.cext = oc.extension[3]
            if oc.chedfm is None:
                head_const = flopy.utils.HeadFile
            else:
                head_const = flopy.utils.FormattedHeadFile
            if oc.cddnfm is None:
                ddn_const = flopy.utils.HeadFile
            else:
                ddn_const = flopy.utils.FormattedHeadFile

            for k, lst in oc.stress_period_data.items():
                for v in lst:
                    if v.lower() == "save head":
                        savehead = True
                    if v.lower() == "save drawdown":
                        saveddn = True
                    if v.lower() == "save budget":
                        savebud = True
        except Exception as e:
            print(
                "error reading output filenames "
                + "from OC package: {}".format(str(e))
            )

        self.hpth = os.path.join(
            self.model_ws, "{}.{}".format(self.name, self.hext)
        )
        self.dpth = os.path.join(
            self.model_ws, "{}.{}".format(self.name, self.dext)
        )
        self.cpth = os.path.join(
            self.model_ws, "{}.{}".format(self.name, self.cext)
        )

        hdObj = None
        ddObj = None
        bdObj = None

        if savehead and os.path.exists(self.hpth):
            hdObj = head_const(self.hpth, model=self, **kwargs)

        if saveddn and os.path.exists(self.dpth):
            ddObj = ddn_const(self.dpth, model=self, **kwargs)
        if savebud and os.path.exists(self.cpth):
            bdObj = flopy.utils.CellBudgetFile(self.cpth, model=self, **kwargs)

        # get subsidence, if written
        subObj = None
        try:

            if self.sub is not None and "subsidence.hds" in self.sub.extension:
                idx = self.sub.extension.index("subsidence.hds")
                subObj = head_const(
                    os.path.join(self.model_ws, self.sub.file_name[idx]),
                    text="subsidence",
                )
        except Exception as e:
            print("error loading subsidence.hds:{0}".format(str(e)))

        if as_dict:
            oudic = {}
            if subObj is not None:
                oudic["subsidence.hds"] = subObj
            if savehead and hdObj:
                oudic[self.hpth] = hdObj
            if saveddn and ddObj:
                oudic[self.dpth] = ddObj
            if savebud and bdObj:
                oudic[self.cpth] = bdObj
            return oudic
        else:
            return hdObj, ddObj, bdObj

    @classmethod
    def load(
        cls,
        f,
        version="mf2005",
        exe_name="mf2005.exe",
        verbose=False,
        model_ws=".",
        load_only=None,
        forgive=False,
        check=True,
    ):
        """
        Load an existing MODFLOW model.

        Parameters
        ----------
        f : str
            Path to MODFLOW name file to load.
        version : str, default "mf2005"
            MODFLOW version. Choose one of: "mf2k", "mf2005" (default),
            "mfnwt", or "mfusg". Note that this can be modified on loading
            packages unique to different MODFLOW versions.
        exe_name : str, default "mf2005.exe"
            MODFLOW executable name.
        verbose : bool, default False
            Show messages that can be useful for debugging.
        model_ws : str, default "."
            Model workspace path. Default is the current directory.
        load_only : list, str or None
            List of case insensitive packages to load, e.g. ["bas6", "lpf"].
            One package can also be specified, e.g. "rch". Default is None,
            which attempts to load all files. An empty list [] will not load
            any additional packages than is necessary. At a minimum, "dis" or
            "disu" is always loaded.
        forgive : bool, optional
            Option to raise exceptions on package load failure, which can be
            useful for debugging. Default False.
        check : boolean, optional
            Check model input for common errors. Default True.

        Returns
        -------
        flopy.modflow.mf.Modflow

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('model.nam')

        """
        # similar to modflow command: if file does not exist , try file.nam
        namefile_path = os.path.join(model_ws, f)
        if not os.path.isfile(namefile_path) and os.path.isfile(
            namefile_path + ".nam"
        ):
            namefile_path += ".nam"
        if not os.path.isfile(namefile_path):
            raise IOError("cannot find name file: " + str(namefile_path))

        # Determine model name from 'f', without any extension or path
        modelname = os.path.splitext(os.path.basename(f))[0]

        # if model_ws is None:
        #    model_ws = os.path.dirname(f)
        if verbose:
            print(
                "\nCreating new model with name: {}\n{}\n".format(
                    modelname, 50 * "-"
                )
            )

        attribs = mfreadnam.attribs_from_namfile_header(
            os.path.join(model_ws, f)
        )

        ml = cls(
            modelname,
            version=version,
            exe_name=exe_name,
            verbose=verbose,
            model_ws=model_ws,
            **attribs
        )

        files_successfully_loaded = []
        files_not_loaded = []

        # read name file
        ext_unit_dict = mfreadnam.parsenamefile(
            namefile_path, ml.mfnam_packages, verbose=verbose
        )
        if ml.verbose:
            print(
                "\n{}\nExternal unit dictionary:\n{}\n{}\n".format(
                    50 * "-", ext_unit_dict, 50 * "-"
                )
            )

        # create a dict where key is the package name, value is unitnumber
        ext_pkg_d = {v.filetype: k for (k, v) in ext_unit_dict.items()}

        # reset version based on packages in the name file
        if "NWT" in ext_pkg_d or "UPW" in ext_pkg_d:
            version = "mfnwt"
        if "GLOBAL" in ext_pkg_d:
            if version != "mf2k":
                ml.glo = ModflowGlobal(ml)
            version = "mf2k"
        if "SMS" in ext_pkg_d:
            version = "mfusg"
        if "DISU" in ext_pkg_d:
            version = "mfusg"
            ml.structured = False
        # update the modflow version
        ml.set_version(version)

        # reset unit number for glo file
        if version == "mf2k":
            if "GLOBAL" in ext_pkg_d:
                unitnumber = ext_pkg_d["GLOBAL"]
                filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
                ml.glo.unit_number = [unitnumber]
                ml.glo.file_name = [filepth]
            else:
                # TODO: is this necessary? it's not done for LIST.
                ml.glo.unit_number = [0]
                ml.glo.file_name = [""]

        # reset unit number for list file
        if "LIST" in ext_pkg_d:
            unitnumber = ext_pkg_d["LIST"]
            filepth = os.path.basename(ext_unit_dict[unitnumber].filename)
            ml.lst.unit_number = [unitnumber]
            ml.lst.file_name = [filepth]

        # look for the free format flag in bas6
        bas_key = ext_pkg_d.get("BAS6")
        if bas_key is not None:
            bas = ext_unit_dict[bas_key]
            start = bas.filehandle.tell()
            line = bas.filehandle.readline()
            while line.startswith("#"):
                line = bas.filehandle.readline()
            if "FREE" in line.upper():
                ml.free_format_input = True
            bas.filehandle.seek(start)
        if verbose:
            print("ModflowBas6 free format:{0}\n".format(ml.free_format_input))

        # load dis
        dis_key = ext_pkg_d.get("DIS") or ext_pkg_d.get("DISU")
        if dis_key is None:
            raise KeyError("discretization entry not found in nam file")
        disnamdata = ext_unit_dict[dis_key]
        dis = disnamdata.package.load(
            disnamdata.filehandle, ml, ext_unit_dict=ext_unit_dict, check=False
        )
        files_successfully_loaded.append(disnamdata.filename)
        if ml.verbose:
            print("   {:4s} package load...success".format(dis.name[0]))
        assert ml.pop_key_list.pop() == dis_key
        ext_unit_dict.pop(dis_key).filehandle.close()

        dis.start_datetime = ml._start_datetime

        if load_only is None:
            # load all packages/files
            load_only = ext_pkg_d.keys()
        else:  # check items in list
            if not isinstance(load_only, list):
                load_only = [load_only]
            not_found = []
            for i, filetype in enumerate(load_only):
                load_only[i] = filetype = filetype.upper()
                if filetype not in ext_pkg_d:
                    not_found.append(filetype)
            if not_found:
                raise KeyError(
                    "the following load_only entries were not found "
                    "in the ext_unit_dict: " + str(not_found)
                )

        # zone, mult, pval
        if "PVAL" in ext_pkg_d:
            ml.mfpar.set_pval(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("PVAL")
        if "ZONE" in ext_pkg_d:
            ml.mfpar.set_zone(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("ZONE")
        if "MULT" in ext_pkg_d:
            ml.mfpar.set_mult(ml, ext_unit_dict)
            assert ml.pop_key_list.pop() == ext_pkg_d.get("MULT")

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                if item.filetype in load_only:
                    package_load_args = getfullargspec(item.package.load)[0]
                    if forgive:
                        try:
                            if "check" in package_load_args:
                                item.package.load(
                                    item.filehandle,
                                    ml,
                                    ext_unit_dict=ext_unit_dict,
                                    check=False,
                                )
                            else:
                                item.package.load(
                                    item.filehandle,
                                    ml,
                                    ext_unit_dict=ext_unit_dict,
                                )
                            files_successfully_loaded.append(item.filename)
                            if ml.verbose:
                                print(
                                    "   {:4s} package load...success".format(
                                        item.filetype
                                    )
                                )
                        except Exception as e:
                            ml.load_fail = True
                            if ml.verbose:
                                print(
                                    "   {:4s} package load...failed".format(
                                        item.filetype
                                    )
                                )
                                print("   {!s}".format(e))
                            files_not_loaded.append(item.filename)
                    else:
                        if "check" in package_load_args:
                            item.package.load(
                                item.filehandle,
                                ml,
                                ext_unit_dict=ext_unit_dict,
                                check=False,
                            )
                        else:
                            item.package.load(
                                item.filehandle,
                                ml,
                                ext_unit_dict=ext_unit_dict,
                            )
                        files_successfully_loaded.append(item.filename)
                        if ml.verbose:
                            print(
                                "   {:4s} package load...success".format(
                                    item.filetype
                                )
                            )
                else:
                    if ml.verbose:
                        print(
                            "   {:4s} package load...skipped".format(
                                item.filetype
                            )
                        )
                    files_not_loaded.append(item.filename)
            elif "data" not in item.filetype.lower():
                files_not_loaded.append(item.filename)
                if ml.verbose:
                    print(
                        "   {:4s} package load...skipped".format(item.filetype)
                    )
            elif "data" in item.filetype.lower():
                if ml.verbose:
                    print(
                        "   {:s} package load...skipped".format(item.filetype)
                    )
                    print("      {}".format(os.path.basename(item.filename)))
                if key not in ml.pop_key_list:
                    # do not add unit number (key) if it already exists
                    if key not in ml.external_units:
                        ml.external_fnames.append(item.filename)
                        ml.external_units.append(key)
                        ml.external_binflag.append(
                            "binary" in item.filetype.lower()
                        )
                        ml.external_output.append(False)
            else:
                raise KeyError("unhandled case: {}, {}".format(key, item))

        # pop binary output keys and any external file units that are now
        # internal
        for key in ml.pop_key_list:
            try:
                ml.remove_external(unit=key)
                item = ext_unit_dict.pop(key)
                if hasattr(item.filehandle, "close"):
                    item.filehandle.close()
            except KeyError:
                if ml.verbose:
                    print(
                        "\nWARNING:\n    External file unit {} does not "
                        "exist in ext_unit_dict.".format(key)
                    )

        # write message indicating packages that were successfully loaded
        if ml.verbose:
            print("")
            print(
                "   The following {} packages were successfully loaded.".format(
                    len(files_successfully_loaded)
                )
            )
            for fname in files_successfully_loaded:
                print("      " + os.path.basename(fname))
            if len(files_not_loaded) > 0:
                print(
                    "   The following {} packages were not loaded.".format(
                        len(files_not_loaded)
                    )
                )
                for fname in files_not_loaded:
                    print("      " + os.path.basename(fname))
        if check:
            ml.check(f="{}.chk".format(ml.name), verbose=ml.verbose, level=0)

        # return model object
        return ml
