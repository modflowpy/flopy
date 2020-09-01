from ..pakbase import Package


class Mt3dPhc(Package):
    """
    PHC package class for PHT3D
    """

    unitnumber = 38

    def __init__(
        self,
        model,
        os=2,
        temp=25,
        asbin=0,
        eps_aqu=0,
        eps_ph=0,
        scr_output=1,
        cb_offset=0,
        smse=["pH", "pe"],
        mine=[],
        ie=[],
        surf=[],
        mobkin=[],
        minkin=[],
        surfkin=[],
        imobkin=[],
        extension="phc",
        unitnumber=None,
        filenames=None,
    ):

        if unitnumber is None:
            unitnumber = Mt3dPhc._defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dPhc._reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dPhc._ftype()]
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

        self.os = os
        self.temp = temp
        self.asbin = asbin
        self.eps_aqu = eps_aqu
        self.eps_ph = eps_ph
        self.scr_output = scr_output
        self.cb_offset = cb_offset
        self.smse = smse
        self.nsmse = len(self.smse)
        self.mine = mine
        self.nmine = len(self.mine)
        self.ie = ie
        self.nie = len(self.ie)
        self.surf = surf
        self.nsurf = len(self.surf)
        self.mobkin = mobkin
        self.nmobkin = len(self.mobkin)
        self.minkin = minkin[0]
        self.nminkin = len(self.minkin)
        self.minkin_parms = minkin[1]
        self.surfkin = surfkin
        self.nsurfkin = len(self.surfkin)
        self.imobkin = imobkin
        self.nimobkin = len(self.imobkin)
        self.parent.add_package(self)
        return

    def __repr__(self):
        return "PHC package class for PHT3D"

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_phc = open(self.fn_path, "w")
        f_phc.write(
            "%3d%10f%3d%10f%10f%3d\n"
            % (
                self.os,
                self.temp,
                self.asbin,
                self.eps_aqu,
                self.eps_ph,
                self.scr_output,
            )
        )
        f_phc.write("%10f\n" % (self.cb_offset))
        f_phc.write("%3d\n" % (self.nsmse))
        f_phc.write("%3d\n" % (self.nmine))
        f_phc.write("%3d\n" % (self.nie))
        f_phc.write("%3d\n" % (self.nsurf))
        f_phc.write(
            "%3d%3d%3d%3d\n"
            % (self.nmobkin, self.nminkin, self.nsurfkin, self.nimobkin)
        )
        for s in self.smse:
            f_phc.write("%s\n" % (s))
        i = 0
        for m in self.minkin:
            f_phc.write("%s %d\n" % (m, len(self.minkin_parms[i])))
            for n in self.minkin_parms[i]:
                f_phc.write("\t%10f\n" % (n))
            i = i + 1
        f_phc.close()
        return

    @staticmethod
    def _ftype():
        return "PHC"

    @staticmethod
    def _defaultunit():
        return 38

    @staticmethod
    def _reservedunit():
        return 38
