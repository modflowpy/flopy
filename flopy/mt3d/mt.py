import os
import sys
import numpy as np
from ..mbase import BaseModel
from ..pakbase import Package
from ..utils import mfreadnam
from .mtbtn import Mt3dBtn
from .mtadv import Mt3dAdv
from .mtdsp import Mt3dDsp
from .mtssm import Mt3dSsm
from .mtrct import Mt3dRct
from .mtgcg import Mt3dGcg
from .mttob import Mt3dTob
from .mtphc import Mt3dPhc
from .mtuzt import Mt3dUzt
from .mtsft import Mt3dSft
from .mtlkt import Mt3dLkt
from ..discretization.structuredgrid import StructuredGrid
from flopy.discretization.modeltime import ModelTime


class Mt3dList(Package):
    """
    List package class
    """

    def __init__(self, model, extension="list", listunit=7):
        # Call ancestor's init to set self.parent, extension, name and
        # unit number
        Package.__init__(self, model, extension, "LIST", listunit)
        # self.parent.add_package(self) This package is not added to the base
        # model so that it is not included in get_name_file_entries()
        return

    def __repr__(self):
        return "List package class"

    def write_file(self):
        # Not implemented for list class
        return


class Mt3dms(BaseModel):
    """
    MT3DMS Model Class.

    Parameters
    ----------
    modelname : str, default "mt3dtest"
        Name of model.  This string will be used to name the MODFLOW input
        that are created with write_model.
    namefile_ext : str, default "nam"
        Extension for the namefile.
    modflowmodel : flopy.modflow.mf.Modflow
        This is a flopy Modflow model object upon which this Mt3dms model
        is based.
    ftlfilename : str, default "mt3d_link.ftl"
        Name of flow-transport link file.
    ftlfree : TYPE, default False
        If flow-link transport file is formatted (True) or unformatted
        (False, default).
    version : str, default "mt3dms"
        Mt3d version. Choose one of: "mt3dms" (default) or "mt3d-usgs".
    exe_name : str, default "mt3dms.exe"
        The name of the executable to use.
    structured : bool, default True
        Specify if model grid is structured (default) or unstructured.
    listunit : int, default 16
        Unit number for the list file.
    ftlunit : int, default 10
        Unit number for flow-transport link file.
    model_ws : str, optional
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
    >>> m = flopy.mt3d.mt.Mt3dms()

    """

    def __init__(
        self,
        modelname="mt3dtest",
        namefile_ext="nam",
        modflowmodel=None,
        ftlfilename="mt3d_link.ftl",
        ftlfree=False,
        version="mt3dms",
        exe_name="mt3dms.exe",
        structured=True,
        listunit=16,
        ftlunit=10,
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
        self.version_types = {"mt3dms": "MT3DMS", "mt3d-usgs": "MT3D-USGS"}

        self.set_version(version.lower())

        if listunit is None:
            listunit = 16

        if ftlunit is None:
            ftlunit = 10

        self.lst = Mt3dList(self, listunit=listunit)
        self.mf = modflowmodel
        self.ftlfilename = ftlfilename
        self.ftlfree = ftlfree
        self.ftlunit = ftlunit
        self.free_format = None

        # Check whether specified ftlfile exists in model directory; if not,
        # warn user
        if os.path.isfile(
            os.path.join(self.model_ws, str(modelname + "." + namefile_ext))
        ):
            with open(
                os.path.join(
                    self.model_ws, str(modelname + "." + namefile_ext)
                )
            ) as nm_file:
                for line in nm_file:
                    if line[0:3] == "FTL":
                        ftlfilename = line.strip().split()[2]
                        break
        if ftlfilename is None:
            print("User specified FTL file does not exist in model directory")
            print("MT3D will not work without a linker file")
        else:
            if os.path.isfile(os.path.join(self.model_ws, ftlfilename)):
                # Check that the FTL present in the directory is of the format
                # specified by the user, i.e., is same as ftlfree
                # Do this by checking whether the first non-blank character is
                # an apostrophe.
                # If code lands here, then ftlfilename exists, open and read
                # first 4 characters
                f = open(os.path.join(self.model_ws, ftlfilename), "rb")
                c = f.read(4)
                if isinstance(c, bytes):
                    c = c.decode()

                # if first non-blank char is an apostrophe, then formatted,
                # otherwise binary
                if (c.strip()[0] == "'" and self.ftlfree) or (
                    c.strip()[0] != "'" and not self.ftlfree
                ):
                    pass
                else:
                    print(
                        "Specified value of ftlfree conflicts with FTL "
                        "file format"
                    )
                    print(
                        "Switching ftlfree from "
                        "{} to {}".format(self.ftlfree, not self.ftlfree)
                    )
                    self.ftlfree = not self.ftlfree  # Flip the bool

        # external option stuff
        self.array_free_format = False
        self.array_format = "mt3d"
        self.external_fnames = []
        self.external_units = []
        self.external_binflag = []
        self.external = False
        self.load = load
        # the starting external data unit number
        self._next_ext_unit = 2000
        if external_path is not None:
            # assert model_ws == '.', "ERROR: external cannot be used " + \
            #                        "with model_ws"

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
        self.mfnam_packages = {
            "btn": Mt3dBtn,
            "adv": Mt3dAdv,
            "dsp": Mt3dDsp,
            "ssm": Mt3dSsm,
            "rct": Mt3dRct,
            "gcg": Mt3dGcg,
            "tob": Mt3dTob,
            "phc": Mt3dPhc,
            "lkt": Mt3dLkt,
            "sft": Mt3dSft,
            "uzt2": Mt3dUzt,
        }
        return

    def __repr__(self):
        return "MT3DMS model"

    @property
    def modeltime(self):
        # build model time
        data_frame = {
            "perlen": self.mf.dis.perlen.array,
            "nstp": self.mf.dis.nstp.array,
            "tsmult": self.mf.dis.tsmult.array,
        }
        self._model_time = ModelTime(
            data_frame,
            self.mf.dis.itmuni_dict[self.mf.dis.itmuni],
            self.dis.start_datetime,
            self.dis.steady.array,
        )
        return self._model_time

    @property
    def modelgrid(self):
        if not self._mg_resync:
            return self._modelgrid

        if self.btn is not None:
            ibound = self.btn.icbund.array
            delc = self.btn.delc.array
            delr = self.btn.delr.array
            top = self.btn.htop.array
            botm = np.subtract(top, self.btn.dz.array.cumsum(axis=0))
            nlay = self.btn.nlay
        else:
            delc = self.mf.dis.delc.array
            delr = self.mf.dis.delr.array
            top = self.mf.dis.top.array
            botm = self.mf.dis.botm.array
            nlay = self.mf.nlay
            if self.mf.bas6 is not None:
                ibound = self.mf.bas6.ibound.array
            else:
                ibound = None
        # build grid
        self._modelgrid = StructuredGrid(
            delc=delc,
            delr=delr,
            top=top,
            botm=botm,
            idomain=ibound,
            proj4=self._modelgrid.proj4,
            epsg=self._modelgrid.epsg,
            xoff=self._modelgrid.xoffset,
            yoff=self._modelgrid.yoffset,
            angrot=self._modelgrid.angrot,
            nlay=nlay,
        )

        # resolve offsets
        xoff = self._modelgrid.xoffset
        if xoff is None:
            if self._xul is not None:
                xoff = self._modelgrid._xul_to_xll(self._xul)
            else:
                xoff = self.mf._modelgrid.xoffset
            if xoff is None:
                # incase mf._modelgrid.xoffset is not set but mf._xul is
                if self.mf._xul is not None:
                    xoff = self._modelgrid._xul_to_xll(self.mf._xul)
                else:
                    xoff = 0.0
        yoff = self._modelgrid.yoffset
        if yoff is None:
            if self._yul is not None:
                yoff = self._modelgrid._yul_to_yll(self._yul)
            else:
                yoff = self.mf._modelgrid.yoffset
            if yoff is None:
                # incase mf._modelgrid.yoffset is not set but mf._yul is
                if self.mf._yul is not None:
                    yoff = self._modelgrid._yul_to_yll(self.mf._yul)
                else:
                    yoff = 0.0
        proj4 = self._modelgrid.proj4
        if proj4 is None:
            proj4 = self.mf._modelgrid.proj4
        epsg = self._modelgrid.epsg
        if epsg is None:
            epsg = self.mf._modelgrid.epsg
        angrot = self._modelgrid.angrot
        if angrot is None or angrot == 0.0:  # angrot normally defaulted to 0.0
            if self.mf._modelgrid.angrot is not None:
                angrot = self.mf._modelgrid.angrot
            else:
                angrot = 0.0

        self._modelgrid.set_coord_info(xoff, yoff, angrot, epsg, proj4)
        self._mg_resync = not self._modelgrid.is_complete
        return self._modelgrid

    @property
    def solver_tols(self):
        if self.gcg is not None:
            return self.gcg.cclose, -999
        return None

    @property
    def sr(self):
        if self.mf is not None:
            return self.mf.sr
        return None

    @property
    def nlay(self):
        if self.btn:
            return self.btn.nlay
        else:
            return 0

    @property
    def nrow(self):
        if self.btn:
            return self.btn.nrow
        else:
            return 0

    @property
    def ncol(self):
        if self.btn:
            return self.btn.ncol
        else:
            return 0

    @property
    def nper(self):
        if self.btn:
            return self.btn.nper
        else:
            return 0

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

    def get_nrow_ncol_nlay_nper(self):
        if self.btn:
            return self.btn.nrow, self.btn.ncol, self.btn.nlay, self.btn.nper
        else:
            return 0, 0, 0, 0

    # Property has no setter, so read-only
    nrow_ncol_nlay_nper = property(get_nrow_ncol_nlay_nper)

    def write_name_file(self):
        """
        Write the name file.

        """
        fn_path = os.path.join(self.model_ws, self.namefile)
        f_nam = open(fn_path, "w")
        f_nam.write("{}\n".format(self.heading))
        f_nam.write(
            "{:14s} {:5d}  {}\n".format(
                self.lst.name[0],
                self.lst.unit_number[0],
                self.lst.file_name[0],
            )
        )
        if self.ftlfilename is not None:
            ftlfmt = ""
            if self.ftlfree:
                ftlfmt = "FREE"
            f_nam.write(
                "{:14s} {:5d}  {} {}\n".format(
                    "FTL", self.ftlunit, self.ftlfilename, ftlfmt
                )
            )
        # write file entries in name file
        f_nam.write(str(self.get_name_file_entries()))

        # write the external files
        for u, f in zip(self.external_units, self.external_fnames):
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

        f_nam.close()
        return

    def load_results(self, **kwargs):
        return

    @classmethod
    def load(
        cls,
        f,
        version="mt3dms",
        exe_name="mt3dms.exe",
        verbose=False,
        model_ws=".",
        load_only=None,
        forgive=False,
        modflowmodel=None,
    ):
        """
        Load an existing model.

        Parameters
        ----------
        f : str
            Path to MT3D name file to load.
        version : str, default "mt3dms"
            Mt3d version. Choose one of: "mt3dms" (default) or "mt3d-usgs".
        exe_name : str, default "mt3dms.exe"
            The name of the executable to use.
        verbose : bool, default False
            Print information on the load process if True.
        model_ws : str, default "."
            Model workspace path. Default is the current directory.
        load_only : list of str, optional
            Packages to load (e.g. ['btn', 'adv']). Default None
            means that all packages will be loaded.
        forgive : bool, default False
            Option to raise exceptions on package load failure, which can be
            useful for debugging.
        modflowmodel : flopy.modflow.mf.Modflow, optional
            This is a flopy Modflow model object upon which this Mt3dms
            model is based.

        Returns
        -------
        flopy.mt3d.mt.Mt3dms

        Notes
        -----
        The load method does not retain the name for the MODFLOW-generated
        FTL file.  This can be added manually after the MT3D model has been
        loaded.  The syntax for doing this manually is
        ``mt.ftlfilename = 'example.ftl'``.

        Examples
        --------
        >>> import flopy
        >>> mt = flopy.mt3d.mt.Mt3dms.load('example.nam')
        >>> mt.ftlfilename = 'example.ftl'

        """
        modelname, ext = os.path.splitext(f)
        modelname_extension = ext[1:]  # without '.'

        if verbose:
            sys.stdout.write(
                "\nCreating new model with name: {}\n{}\n\n".format(
                    modelname, 50 * "-"
                )
            )
        mt = cls(
            modelname=modelname,
            namefile_ext=modelname_extension,
            version=version,
            exe_name=exe_name,
            verbose=verbose,
            model_ws=model_ws,
            modflowmodel=modflowmodel,
        )
        files_successfully_loaded = []
        files_not_loaded = []

        # read name file
        namefile_path = os.path.join(mt.model_ws, f)
        if not os.path.isfile(namefile_path):
            raise IOError("cannot find name file: " + str(namefile_path))
        try:
            ext_unit_dict = mfreadnam.parsenamefile(
                namefile_path, mt.mfnam_packages, verbose=verbose
            )
        except Exception as e:
            # print("error loading name file entries from file")
            # print(str(e))
            # return None
            raise Exception(
                "error loading name file entries from file:\n" + str(e)
            )

        if mt.verbose:
            print(
                "\n{}\nExternal unit dictionary:\n{}\n{}\n".format(
                    50 * "-", ext_unit_dict, 50 * "-"
                )
            )

        # reset unit number for list file
        unitnumber = None
        for key, value in ext_unit_dict.items():
            if value.filetype == "LIST":
                unitnumber = key
                filepth = os.path.basename(value.filename)
        if unitnumber == "LIST":
            unitnumber = 16
        if unitnumber is not None:
            mt.lst.unit_number = [unitnumber]
            mt.lst.file_name = [filepth]

        # set ftl information
        unitnumber = None
        for key, value in ext_unit_dict.items():
            if value.filetype == "FTL":
                unitnumber = key
                filepth = os.path.basename(value.filename)
        if unitnumber == "FTL":
            unitnumber = 10
        if unitnumber is not None:
            mt.ftlunit = unitnumber
            mt.ftlfilename = filepth

        # load btn
        btn = None
        btn_key = None
        for key, item in ext_unit_dict.items():
            if item.filetype.lower() == "btn":
                btn = item
                btn_key = key
                break

        if btn is None:
            return None

        try:
            pck = btn.package.load(
                btn.filename, mt, ext_unit_dict=ext_unit_dict
            )
        except Exception as e:
            raise Exception("error loading BTN: {0}".format(str(e)))
        files_successfully_loaded.append(btn.filename)
        if mt.verbose:
            sys.stdout.write(
                "   {:4s} package load...success\n".format(pck.name[0])
            )
        ext_unit_dict.pop(btn_key).filehandle.close()
        ncomp = mt.btn.ncomp
        # reserved unit numbers for .ucn, s.ucn, .obs, .mas, .cnf
        poss_output_units = set(
            list(range(201, 201 + ncomp))
            + list(range(301, 301 + ncomp))
            + list(range(401, 401 + ncomp))
            + list(range(601, 601 + ncomp))
            + [17]
        )
        if load_only is None:
            load_only = []
            for key, item in ext_unit_dict.items():
                load_only.append(item.filetype)
        else:
            if not isinstance(load_only, list):
                load_only = [load_only]
            not_found = []
            for i, filetype in enumerate(load_only):
                filetype = filetype.upper()
                if filetype != "BTN":
                    load_only[i] = filetype
                    found = False
                    for key, item in ext_unit_dict.items():
                        if item.filetype == filetype:
                            found = True
                            break
                    if not found:
                        not_found.append(filetype)
            if len(not_found) > 0:
                raise Exception(
                    "the following load_only entries were not found "
                    "in the ext_unit_dict: " + ",".join(not_found)
                )

        # try loading packages in ext_unit_dict
        for key, item in ext_unit_dict.items():
            if item.package is not None:
                if item.filetype in load_only:
                    if forgive:
                        try:
                            pck = item.package.load(
                                item.filehandle,
                                mt,
                                ext_unit_dict=ext_unit_dict,
                            )
                            files_successfully_loaded.append(item.filename)
                            if mt.verbose:
                                sys.stdout.write(
                                    "   {:4s} package load...success\n".format(
                                        pck.name[0]
                                    )
                                )
                        except BaseException as o:
                            if mt.verbose:
                                sys.stdout.write(
                                    "   {:4s} package load...failed\n   {!s}\n".format(
                                        item.filetype, o
                                    )
                                )
                            files_not_loaded.append(item.filename)
                    else:
                        pck = item.package.load(
                            item.filehandle, mt, ext_unit_dict=ext_unit_dict
                        )
                        files_successfully_loaded.append(item.filename)
                        if mt.verbose:
                            sys.stdout.write(
                                "   {:4s} package load...success\n".format(
                                    pck.name[0]
                                )
                            )
                else:
                    if mt.verbose:
                        sys.stdout.write(
                            "   {:4s} package load...skipped\n".format(
                                item.filetype
                            )
                        )
                    files_not_loaded.append(item.filename)
            elif "data" not in item.filetype.lower():
                files_not_loaded.append(item.filename)
                if mt.verbose:
                    sys.stdout.write(
                        "   {:4s} package load...skipped\n".format(
                            item.filetype
                        )
                    )
            elif "data" in item.filetype.lower():
                if mt.verbose:
                    sys.stdout.write(
                        "   {} file load...skipped\n      {}\n".format(
                            item.filetype, os.path.basename(item.filename)
                        )
                    )
                if key in poss_output_units:
                    # id files specified to output unit numbers and allow to
                    # pass through
                    mt.output_fnames.append(os.path.basename(item.filename))
                    mt.output_units.append(key)
                    mt.output_binflag.append("binary" in item.filetype.lower())
                elif key not in mt.pop_key_list:
                    mt.external_fnames.append(item.filename)
                    mt.external_units.append(key)
                    mt.external_binflag.append(
                        "binary" in item.filetype.lower()
                    )
                    mt.external_output.append(False)

        # pop binary output keys and any external file units that are now
        # internal
        for key in mt.pop_key_list:
            try:
                mt.remove_external(unit=key)
                item = ext_unit_dict.pop(key)
                if hasattr(item.filehandle, "close"):
                    item.filehandle.close()
            except KeyError:
                if mt.verbose:
                    sys.stdout.write(
                        "\nWARNING:\n    External file unit "
                        "{} does not exist in ext_unit_dict.\n".format(key)
                    )

        # write message indicating packages that were successfully loaded
        if mt.verbose:
            print(
                "\n   The following {0} packages were "
                "successfully loaded.".format(len(files_successfully_loaded))
            )
            for fname in files_successfully_loaded:
                print("      " + os.path.basename(fname))
            if len(files_not_loaded) > 0:
                print(
                    "   The following {0} packages were not loaded.".format(
                        len(files_not_loaded)
                    )
                )
                for fname in files_not_loaded:
                    print("      " + os.path.basename(fname))
                print("\n")

        # return model object
        return mt

    @staticmethod
    def load_mas(fname):
        """
        Load an mt3d mas file and return a numpy recarray

        Parameters
        ----------
        fname : str
            name of MT3D mas file

        Returns
        -------
        r : np.ndarray

        """
        if not os.path.isfile(fname):
            raise Exception("Could not find file: {}".format(fname))
        dtype = [
            ("time", float),
            ("total_in", float),
            ("total_out", float),
            ("sources", float),
            ("sinks", float),
            ("fluid_storage", float),
            ("total_mass", float),
            ("error_in-out", float),
            ("error_alt", float),
        ]
        r = np.loadtxt(fname, skiprows=2, dtype=dtype)
        r = r.view(np.recarray)
        return r

    @staticmethod
    def load_obs(fname):
        """
        Load an mt3d obs file and return a numpy recarray

        Parameters
        ----------
        fname : str
            name of MT3D obs file

        Returns
        -------
        r : np.ndarray

        """
        firstline = "STEP   TOTAL TIME             LOCATION OF OBSERVATION POINTS (K,I,J)"
        dtype = [("step", int), ("time", float)]
        nobs = 0
        obs = []

        if not os.path.isfile(fname):
            raise Exception("Could not find file: {}".format(fname))
        with open(fname, "r") as f:
            line = f.readline()
            if line.strip() != firstline:
                raise Exception(
                    "First line in file must be \n{}\nFound {}\n"
                    "{} does not appear to be a valid MT3D OBS file".format(
                        firstline, line.strip(), fname
                    )
                )

            # Read obs names (when break, line will have first data line)
            nlineperrec = 0
            while True:
                line = f.readline()
                if line[0:7].strip() == "1":
                    break
                nlineperrec += 1
                ll = line.strip().split()
                while len(ll) > 0:
                    k = int(ll.pop(0))
                    i = int(ll.pop(0))
                    j = int(ll.pop(0))
                    obsnam = "({}, {}, {})".format(k, i, j)
                    if obsnam in obs:
                        obsnam += str(len(obs) + 1)  # make obs name unique
                    obs.append(obsnam)

            icount = 0
            r = []
            while True:
                ll = []
                for n in range(nlineperrec):
                    icount += 1
                    if icount > 1:
                        line = f.readline()
                    ll.extend(line.strip().split())

                if not line:
                    break

                rec = [int(ll[0])]
                for val in ll[1:]:
                    rec.append(float(val))
                r.append(tuple(rec))

        # add obs names to dtype
        for nameob in obs:
            dtype.append((nameob, float))
        r = np.array(r, dtype=dtype)
        r = r.view(np.recarray)
        return r
