import sys

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d
from ..utils.flopy_io import line_parse


class ModflowBcf(Package):
    """
    MODFLOW Block Centered Flow Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 53)
    intercellt : int
        Intercell transmissivities, harmonic mean (0), arithmetic mean (1),
        logarithmic mean (2), combination (3). (default is 0)
    laycon : int
        Layer type, confined (0), unconfined (1), constant T, variable S (2),
        variable T, variable S (default is 3)
    trpy : float or array of floats (nlay)
        horizontal anisotropy ratio (default is 1.0)
    hdry : float
        head assigned when cell is dry - used as indicator(default is -1E+30)
    iwdflg : int
        flag to indicate if wetting is inactive (0) or not (non zero)
        (default is 0)
    wetfct : float
        factor used when cell is converted from dry to wet (default is 0.1)
    iwetit : int
        iteration interval in wetting/drying algorithm (default is 1)
    ihdwet : int
        flag to indicate how initial head is computed for cells that become
        wet (default is 0)
    tran : float or array of floats (nlay, nrow, ncol), optional
        transmissivity (only read if laycon is 0 or 2) (default is 1.0)
    hy : float or array of floats (nlay, nrow, ncol)
        hydraulic conductivity (only read if laycon is 1 or 3)
        (default is 1.0)
    vcont : float or array of floats (nlay-1, nrow, ncol)
        vertical leakance between layers (default is 1.0)
    sf1 : float or array of floats (nlay, nrow, ncol)
        specific storage (confined) or storage coefficient (unconfined),
        read when there is at least one transient stress period.
        (default is 1e-5)
    sf2 : float or array of floats (nrow, ncol)
        specific yield, only read when laycon is 2 or 3 and there is at least
        one transient stress period (default is 0.15)
    wetdry : float
        a combination of the wetting threshold and a flag to indicate which
        neighboring cells can cause a cell to become wet (default is -0.01)
    extension : string
        Filename extension (default is 'bcf')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output name will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> bcf = flopy.modflow.ModflowBcf(ml)

    """

    def __init__(
        self,
        model,
        ipakcb=None,
        intercellt=0,
        laycon=3,
        trpy=1.0,
        hdry=-1e30,
        iwdflg=0,
        wetfct=0.1,
        iwetit=1,
        ihdwet=0,
        tran=1.0,
        hy=1.0,
        vcont=1.0,
        sf1=1e-5,
        sf2=0.15,
        wetdry=-0.01,
        extension="bcf",
        unitnumber=None,
        filenames=None,
    ):

        if unitnumber is None:
            unitnumber = ModflowBcf._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                ipakcb, fname=fname, package=ModflowBcf._ftype()
            )
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowBcf._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.url = "bcf.htm"

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        # Set values of all parameters
        self.intercellt = Util2d(
            model,
            (nlay,),
            np.int32,
            intercellt,
            name="laycon",
            locat=self.unit_number[0],
        )
        self.laycon = Util2d(
            model,
            (nlay,),
            np.int32,
            laycon,
            name="laycon",
            locat=self.unit_number[0],
        )
        self.trpy = Util2d(
            model,
            (nlay,),
            np.float32,
            trpy,
            name="Anisotropy factor",
            locat=self.unit_number[0],
        )

        # item 1
        self.ipakcb = ipakcb
        self.hdry = hdry
        self.iwdflg = iwdflg
        self.wetfct = wetfct
        self.iwetit = iwetit
        self.ihdwet = ihdwet
        self.tran = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            tran,
            "Transmissivity",
            locat=self.unit_number[0],
        )
        self.hy = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            hy,
            "Horizontal Hydraulic Conductivity",
            locat=self.unit_number[0],
        )
        if model.nlay > 1:
            self.vcont = Util3d(
                model,
                (nlay - 1, nrow, ncol),
                np.float32,
                vcont,
                "Vertical Conductance",
                locat=self.unit_number[0],
            )
        else:
            self.vcont = None
        self.sf1 = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            sf1,
            "Primary Storage Coefficient",
            locat=self.unit_number[0],
        )
        self.sf2 = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            sf2,
            "Secondary Storage Coefficient",
            locat=self.unit_number[0],
        )
        self.wetdry = Util3d(
            model,
            (nlay, nrow, ncol),
            np.float32,
            wetdry,
            "WETDRY",
            locat=self.unit_number[0],
        )
        self.parent.add_package(self)
        return

    def write_file(self, f=None):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # get model information
        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
        dis = self.parent.get_package("DIS")
        if dis is None:
            dis = self.parent.get_package("DISU")

        ifrefm = self.parent.get_ifrefm()

        # Open file for writing
        if f is not None:
            f_bcf = f
        else:
            f_bcf = open(self.fn_path, "w")
        # Item 1: ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET
        f_bcf.write(
            "{:10d}{:10.6G}{:10d}{:10.3f}{:10d}{:10d}\n".format(
                self.ipakcb,
                self.hdry,
                self.iwdflg,
                self.wetfct,
                self.iwetit,
                self.ihdwet,
            )
        )

        # LAYCON array
        for k in range(nlay):
            if ifrefm:
                if self.intercellt[k] > 0:
                    f_bcf.write(
                        "{0:1d}{1:1d} ".format(
                            self.intercellt[k], self.laycon[k]
                        )
                    )
                else:
                    f_bcf.write("0{0:1d} ".format(self.laycon[k]))
            else:
                if self.intercellt[k] > 0:
                    f_bcf.write(
                        "{0:1d}{1:1d}".format(
                            self.intercellt[k], self.laycon[k]
                        )
                    )
                else:
                    f_bcf.write("0{0:1d}".format(self.laycon[k]))
        f_bcf.write("\n")
        f_bcf.write(self.trpy.get_file_entry())
        transient = not dis.steady.all()
        for k in range(nlay):
            if transient == True:
                f_bcf.write(self.sf1[k].get_file_entry())
            if (self.laycon[k] == 0) or (self.laycon[k] == 2):
                f_bcf.write(self.tran[k].get_file_entry())
            else:
                f_bcf.write(self.hy[k].get_file_entry())
            if k < nlay - 1:
                f_bcf.write(self.vcont[k].get_file_entry())
            if (transient == True) and (
                (self.laycon[k] == 2) or (self.laycon[k] == 3)
            ):
                f_bcf.write(self.sf2[k].get_file_entry())
            if (self.iwdflg != 0) and (
                (self.laycon[k] == 1) or (self.laycon[k] == 3)
            ):
                f_bcf.write(self.wetdry[k].get_file_entry())
        f_bcf.close()

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
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        wel : ModflowBcf object
            ModflowBcf object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflow.ModflowBcf.load('test.bcf', m)

        """

        if model.verbose:
            sys.stdout.write("loading bcf package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # determine problem dimensions
        nr, nc, nlay, nper = model.get_nrow_ncol_nlay_nper()
        dis = model.get_package("DIS")
        if dis is None:
            dis = model.get_package("DISU")

        # Item 1: ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET - line already read above
        if model.verbose:
            print("   loading ipakcb, HDRY, IWDFLG, WETFCT, IWETIT, IHDWET...")
        t = line_parse(line)
        ipakcb, hdry, iwdflg, wetfct, iwetit, ihdwet = (
            int(t[0]),
            float(t[1]),
            int(t[2]),
            float(t[3]),
            int(t[4]),
            int(t[5]),
        )

        # LAYCON array
        ifrefm = model.get_ifrefm()
        if model.verbose:
            print("   loading LAYCON...")
        line = f.readline()
        if ifrefm:
            t = []
            tt = line.strip().split()
            for iv in tt:
                t.append(iv)
            # read the rest of the laycon values
            if len(t) < nlay:
                while True:
                    line = f.readline()
                    tt = line.strip().split()
                    for iv in tt:
                        t.append(iv)
                    if len(t) == nlay:
                        break
        else:
            t = []
            istart = 0
            for k in range(nlay):
                lcode = line[istart : istart + 2]
                if lcode.strip() == "":
                    # hit end of line before expected end of data
                    # read next line
                    line = f.readline()
                    istart = 0
                    lcode = line[istart : istart + 2]
                lcode = lcode.replace(" ", "0")
                t.append(lcode)
                istart += 2
        intercellt = np.zeros(nlay, dtype=np.int32)
        laycon = np.zeros(nlay, dtype=np.int32)
        for k in range(nlay):
            if len(t[k]) > 1:
                intercellt[k] = int(t[k][0])
                laycon[k] = int(t[k][1])
            else:
                laycon[k] = int(t[k])

        # TRPY array
        if model.verbose:
            print("   loading TRPY...")
        trpy = Util2d.load(
            f, model, (nlay,), np.float32, "trpy", ext_unit_dict
        )

        # property data for each layer based on options
        transient = not dis.steady.all()
        sf1 = [0] * nlay
        tran = [0] * nlay
        hy = [0] * nlay
        if nlay > 1:
            vcont = [0] * (nlay - 1)
        else:
            vcont = [0] * nlay
        sf2 = [0] * nlay
        wetdry = [0] * nlay

        for k in range(nlay):

            # allow for unstructured changing nodes per layer
            if nr is None:
                nrow = 1
                ncol = nc[k]
            else:
                nrow = nr
                ncol = nc

            # sf1
            if transient:
                if model.verbose:
                    print("   loading sf1 layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "sf1", ext_unit_dict
                )
                sf1[k] = t

            # tran or hy
            if (laycon[k] == 0) or (laycon[k] == 2):
                if model.verbose:
                    print("   loading tran layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "tran", ext_unit_dict
                )
                tran[k] = t
            else:
                if model.verbose:
                    print("   loading hy layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "hy", ext_unit_dict
                )
                hy[k] = t

            # vcont
            if k < (nlay - 1):
                if model.verbose:
                    print("   loading vcont layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "vcont", ext_unit_dict
                )
                vcont[k] = t

            # sf2
            if transient and ((laycon[k] == 2) or (laycon[k] == 3)):
                if model.verbose:
                    print("   loading sf2 layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "sf2", ext_unit_dict
                )
                sf2[k] = t

            # wetdry
            if (iwdflg != 0) and ((laycon[k] == 1) or (laycon[k] == 3)):
                if model.verbose:
                    print("   loading sf2 layer {0:3d}...".format(k + 1))
                t = Util2d.load(
                    f, model, (nrow, ncol), np.float32, "wetdry", ext_unit_dict
                )
                wetdry[k] = t

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowBcf._ftype()
            )
            if ipakcb > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=ipakcb
                )
                model.add_pop_key_list(ipakcb)

        # create instance of bcf object
        bcf = cls(
            model,
            ipakcb=ipakcb,
            intercellt=intercellt,
            laycon=laycon,
            trpy=trpy,
            hdry=hdry,
            iwdflg=iwdflg,
            wetfct=wetfct,
            iwetit=iwetit,
            ihdwet=ihdwet,
            tran=tran,
            hy=hy,
            vcont=vcont,
            sf1=sf1,
            sf2=sf2,
            wetdry=wetdry,
            unitnumber=unitnumber,
            filenames=filenames,
        )

        # return bcf object
        return bcf

    @staticmethod
    def _ftype():
        return "BCF6"

    @staticmethod
    def _defaultunit():
        return 15
