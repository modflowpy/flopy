import re
import numpy as np
from ..pakbase import Package
from ..utils.flopy_io import line_parse, pop_item
from ..utils import MfList
from ..utils.recarray_utils import create_empty_recarray, recarray


class ModflowMnw1(Package):
    """
    MODFLOW Multi-Node Well 1 Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mxmnw : integer
        maximum number of multi-node wells to be simulated
    ipakcb : integer
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    iwelpt : integer
        verbosity flag
    nomoiter : integer
        the number of iterations for which flow in MNW wells is calculated
    kspref : string
        which set of water levels are to be used as reference values for
        calculating drawdown
    losstype : string
        head loss type for each well
    wel1_bynode_qsum : list of lists or None
        nested list containing file names, unit numbers, and ALLTIME flag for
        auxiliary output, e.g. [['test.ByNode',92,'ALLTIME']]
        if None, these optional external filenames and unit numbers are not written out
    itmp : array
        number of wells to be simulated for each stress period (shape : (NPER))
    lay_row_col_qdes_mn_multi : list of arrays
        lay, row, col, qdes, and MN or MULTI flag for all well nodes
        (length : NPER)
    mnwname : string
        prefix name of file for outputting time series data from MNW1
    extension : string
        Filename extension (default is 'mnw1')
    unitnumber : int
        File unit number (default is 33).
    filenames : string or list of strings
        File name of the package (with extension) or a list with the filename
        of the package and the cell-by-cell budget file for ipakcb. Default
        is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    The functionality of the ADD flag in data set 4 is not supported. Also
    not supported are all water-quality parameters (Qwval Iqwgrp), water-level
    limitations (Hlim, Href, DD), non-linear well losses, and pumping
    limitations (QCUT, Q-%CUT, Qfrcmn, Qfrcmx, DEFAULT).

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> mnw1 = flopy.modflow.ModflowMnw1(ml, ...)

    """

    def __init__(
        self,
        model,
        mxmnw=0,
        ipakcb=None,
        iwelpt=0,
        nomoiter=0,
        kspref=1,
        wel1_bynode_qsum=None,
        losstype="skin",
        stress_period_data=None,
        dtype=None,
        mnwname=None,
        extension="mnw1",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowMnw1._defaultunit()

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
                ipakcb, fname=fname, package=ModflowMnw1._ftype()
            )
        else:
            ipakcb = 0

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name, and unit number
        Package.__init__(
            self,
            model,
            extension,
            ModflowMnw1._ftype(),
            unitnumber,
            filenames=fname,
        )

        self.url = "mnw1.htm"
        self.nper = self.parent.nrow_ncol_nlay_nper[-1]
        self._generate_heading()
        self.mxmnw = (
            mxmnw  # -maximum number of multi-node wells to be simulated
        )
        self.ipakcb = ipakcb
        self.iwelpt = iwelpt  # -verbosity flag
        self.nomoiter = nomoiter  # -integer indicating the number of iterations for which flow in MNW wells is calculated
        self.kspref = kspref  # -alphanumeric key indicating which set of water levels are to be used as reference values for calculating drawdown
        self.losstype = (
            losstype  # -string indicating head loss type for each well
        )
        self.wel1_bynode_qsum = wel1_bynode_qsum  # -nested list containing file names, unit numbers, and ALLTIME flag for auxiliary output, e.g. [['test.ByNode',92,'ALLTIME']]
        # if stress_period_data is not None:
        #    for per, spd in stress_period_data.items():
        #        for n in spd.dtype.names:
        #            self.stress_period_data[per] = ModflowMnw1.get_empty_stress_period_data(len(spd),
        #                                                                                    structured=self.parent.structured)
        #            self.stress_period_data[per][n] = stress_period_data[per][n]
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured
            )
        self.stress_period_data = MfList(self, stress_period_data)

        self.mnwname = mnwname  # -string prefix name of file for outputting time series data from MNW1

        # -input format checks:
        lossTypes = ["skin", "linear", "nonlinear"]
        assert self.losstype.lower() in lossTypes, (
            "LOSSTYPE (%s) must be one of the following: skin, linear, nonlinear"
            % (self.losstype)
        )
        # auxFileExtensions = ['wl1','ByNode','Qsum']
        # for each in self.wel1_bynode_qsum:
        #    assert each[0].split('.')[1] in auxFileExtensions, 'File extensions in "wel1_bynode_qsum" must be one of the following: ".wl1", ".ByNode", or ".Qsum".'
        self.parent.add_package(self)

    @staticmethod
    def get_empty_stress_period_data(itmp, structured=True, default_value=0):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowMnw1.get_default_dtype(structured=structured)
        return create_empty_recarray(itmp, dtype, default_value=default_value)

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            return np.dtype(
                [
                    ("mnw_no", int),
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("qdes", np.float32),
                    ("mntxt", object),
                    ("qwval", np.float32),
                    ("rw", np.float32),
                    ("skin", np.float32),
                    ("hlim", np.float32),
                    ("href", np.float32),
                    ("dd", object),
                    ("iqwgrp", object),
                    ("cpc", object),
                    ("qcut", object),
                    ("qfrcmn", np.float32),
                    ("qfrcmx", np.float32),
                    ("label", object),
                ]
            )
        else:
            pass

    @classmethod
    def load(cls, f, model, nper=None, gwt=False, nsol=1, ext_unit_dict=None):

        if model.verbose:
            print("loading mnw1 package file...")

        structured = model.structured
        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()
            nper = (
                1 if nper == 0 else nper
            )  # otherwise iterations from 0, nper won't run

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 (header)
        line = skipcomments(next(f), f)

        # dataset 1
        mxmnw, ipakcb, iwelpt, nomoiter, kspref = _parse_1(line)

        # dataset 2
        line = skipcomments(next(f), f)
        losstype = _parse_2(line)

        # dataset 3
        wel1_bynode_qsum = []
        line = skipcomments(next(f), f)
        for txt in ["wel1", "bynode", "qsum"]:
            if txt in line.lower():
                wel1_bynode_qsum.append(_parse_3(line, txt))
                line = skipcomments(next(f), f)

        # dataset 4
        line = skipcomments(line, f)
        stress_period_data = {}
        dtype = ModflowMnw1.get_default_dtype(structured=structured)
        qfrcmn_default = None
        qfrcmx_default = None
        qcut_default = ""

        # not sure what 'add' means
        add = True if "add" in line.lower() else False

        for per in range(nper):
            if per > 0:
                line = skipcomments(next(f), f)
            add = True if "add" in line.lower() else False
            itmp = int(line_parse(line)[0])
            if itmp > 0:

                # dataset 5
                data, qfrcmn_default, qfrcmx_default, qcut_default = _parse_5(
                    f, itmp, qfrcmn_default, qfrcmx_default, qcut_default
                )

                # cast data (list) to recarray
                tmp = recarray(data, dtype)
                spd = ModflowMnw1.get_empty_stress_period_data(len(data))
                for n in dtype.descr:
                    spd[n[0]] = tmp[n[0]]
                stress_period_data[per] = spd

        if openfile:
            f.close()

        return cls(
            model,
            mxmnw=mxmnw,
            ipakcb=ipakcb,
            iwelpt=iwelpt,
            nomoiter=nomoiter,
            kspref=kspref,
            wel1_bynode_qsum=wel1_bynode_qsum,
            losstype=losstype,
            stress_period_data=stress_period_data,
        )

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """

        # -open file for writing
        # f_mnw1 = open( self.file_name[0], 'w' )
        f = open(self.fn_path, "w")

        # -write header
        f.write("%s\n" % self.heading)

        # -Section 1 - MXMNW ipakcb IWELPT NOMOITER REF:kspref
        f.write(
            "%10i%10i%10i%10i REF = %s\n"
            % (
                self.mxmnw,
                self.ipakcb,
                self.iwelpt,
                self.nomoiter,
                self.kspref,
            )
        )

        # -Section 2 - LOSSTYPE {PLossMNW}
        f.write("%s\n" % (self.losstype))

        if self.wel1_bynode_qsum is not None:
            # -Section 3a - {FILE:filename WEL1:iunw1}
            for each in self.wel1_bynode_qsum:
                if each[0].split(".")[1].lower() == "wl1":
                    f.write("FILE:%s WEL1:%-10i\n" % (each[0], int(each[1])))

            # -Section 3b - {FILE:filename BYNODE:iunby} {ALLTIME}
            for each in self.wel1_bynode_qsum:
                if each[0].split(".")[1].lower() == "bynode":
                    if len(each) == 2:
                        f.write(
                            "FILE:%s BYNODE:%-10i\n" % (each[0], int(each[1]))
                        )
                    elif len(each) == 3:
                        f.write(
                            "FILE:%s BYNODE:%-10i %s\n"
                            % (each[0], int(each[1]), each[2])
                        )

            # -Section 3C - {FILE:filename QSUM:iunqs} {ALLTIME}
            for each in self.wel1_bynode_qsum:
                if each[0].split(".")[1].lower() == "qsum":
                    if len(each) == 2:
                        f.write(
                            "FILE:%s QSUM:%-10i\n" % (each[0], int(each[1]))
                        )
                    elif len(each) == 3:
                        f.write(
                            "FILE:%s QSUM:%-10i %s\n"
                            % (each[0], int(each[1]), each[2])
                        )

        spd = self.stress_period_data.drop("mnw_no")
        # force write_transient to keep the list arrays internal because MNW1 doesn't allow open/close
        spd.write_transient(f, forceInternal=True)

        # -Un-numbered section PREFIX:MNWNAME
        if self.mnwname:
            f.write("PREFIX:%s\n" % (self.mnwname))

        f.close()

    @staticmethod
    def _ftype():
        return "MNW1"

    @staticmethod
    def _defaultunit():
        return 33


def skipcomments(line, f):
    if line.strip().startswith("#"):
        line = skipcomments(next(f), f)
    return line


def _parse_1(line):
    line = line_parse(line)
    mnwmax = pop_item(line, int)
    ipakcb = pop_item(line, int)
    mnwprint = pop_item(line, int)
    next_item = line.pop()
    nomoiter = 0
    kspref = 1
    if next_item.isdigit():
        nomoiter = int(next_item)
    elif "ref" in next_item:
        line = " ".join(line)
        kspref = re.findall(r"\d+", line)
        if len(kspref) > 0:
            kspref = int(kspref[0])
    return mnwmax, ipakcb, mnwprint, nomoiter, kspref


def _parse_2(line):
    line = line.split("!!")[0]
    options = ["SKIN", "NONLINEAR", "LINEAR"]
    losstype = "skin"
    for lt in options:
        if lt.lower() in line.lower():
            losstype = lt.lower()
    return losstype


def _parse_3(line, txt):
    def getitem(line, txt):
        return line.pop(0).replace(txt + ":", "").strip()

    line = line_parse(line.lower())
    items = [getitem(line, "file"), getitem(line, txt)]
    if "alltime" in " ".join(line):
        items.append("alltime")
    return items


def _parse_5(
    f, itmp, qfrcmn_default=None, qfrcmx_default=None, qcut_default=""
):
    data = []
    mnw_no = 0
    mn = False
    multi = False
    label = ""
    for n in range(itmp):

        linetxt = skipcomments(next(f), f).lower()
        line = line_parse(linetxt)

        # get the label; strip it out
        if "site:" in linetxt:
            label = linetxt.replace(",", " ").split("site:")[1].split()[0]
            label = "site:" + label
            txt = [t for t in line if "site:" in t]
            if len(txt) > 0:  # site: might have been in the comments section
                line.remove(txt[0])

        k = pop_item(line, int) - 1
        i = pop_item(line, int) - 1
        j = pop_item(line, int) - 1
        qdes = pop_item(line, float)

        # logic to create column of unique numbers for each MNW
        mntxt = ""
        if "mn" in line:
            if not mn:
                mnw_no -= 1  # this node has same number as previous
                if label == "":
                    label = data[n - 1][-1]
            mn = True
            mntxt = "mn"
            line.remove("mn")
        if "multi" in line:
            multi = True
            mntxt = "multi"
            line.remove("multi")
        if mn and not multi:
            multi = True

        # "The alphanumeric flags MN and DD can appear anywhere
        # between columns 41 and 256, inclusive."
        dd = ""
        if "dd" in line:
            line.remove("dd")
            dd = "dd"

        qwval = pop_item(line, float)
        rw = pop_item(line, float)
        skin = pop_item(line, float)
        hlim = pop_item(line, float)
        href = pop_item(line, float)
        iqwgrp = pop_item(line)

        cpc = ""
        if "cp:" in linetxt:
            cpc = re.findall(r"\d+", line.pop(0))
            # in case there is whitespace between cp: and the value
            if len(cpc) == 0:
                cpc = pop_item(line)
            cpc = "cp:" + cpc

        qcut = ""
        qfrcmn = 0.0
        qfrcmx = 0.0
        if "qcut" in linetxt:
            txt = [t for t in line if "qcut" in t][0]
            qcut = txt
            line.remove(txt)
        elif "%cut" in linetxt:
            txt = [t for t in line if "%cut" in t][0]
            qcut = txt
            line.remove(txt)
        if "qcut" in linetxt or "%cut" in linetxt:
            qfrcmn = pop_item(line, float)
            qfrcmx = pop_item(line, float)
        elif qfrcmn_default is not None and qfrcmx_default is not None:
            qfrcmn = qfrcmn_default
            qfrcmx = qfrcmx_default
            if "qcut" not in linetxt and "%cut" not in linetxt:
                qcut = qcut_default
        if "default" in line:
            qfrcmn_default = qfrcmn
            qfrcmx_default = qfrcmx
            qcut_default = qcut

        idata = [
            mnw_no,
            k,
            i,
            j,
            qdes,
            mntxt,
            qwval,
            rw,
            skin,
            hlim,
            href,
            dd,
            iqwgrp,
            cpc,
            qcut,
            qfrcmn,
            qfrcmx,
            label,
        ]
        data.append(idata)

        # reset MNW designators
        # if at the end of the well
        if mn and multi:
            mnw_no += 1
            mn = False
            multi = False
            label = ""
        elif not mn and not multi:
            mnw_no += 1
            label = ""

    return data, qfrcmn_default, qfrcmx_default, qcut_default


def _write_5(f, spd):
    f.write("{:d} {:d} {:d} {}")
    pass
