import numpy as np
from ..pakbase import Package


class ModflowFlwob(Package):
    """
    Head-dependent flow boundary Observation package class. Minimal working
    example that will be refactored in a future version.

    Parameters
    ----------
    nqfb : int
        Number of cell groups for the head-dependent flow boundary observations
    nqcfb : int
        Greater than or equal to the total number of cells in all cell groups
    nqtfb : int
        Total number of head-dependent flow boundary observations for all cell
        groups
    iufbobsv : int
        unit number where output is saved
    tomultfb : float
        Time-offset multiplier for head-dependent flow boundary observations.
        The product of tomultfb and toffset must produce a time value in units
        consistent with other model input. tomultfb can be dimensionless or can
        be used to convert the units of toffset to the time unit used in the
        simulation.
    nqobfb : int list of length nqfb
        The number of times at which flows are observed for the group of cells
    nqclfb : int list of length nqfb
        Is a flag, and the absolute value of nqclfb is the number of cells in
        the group.  If nqclfb is less than zero, factor = 1.0 for all cells in
        the group.
    obsnam : string list of length nqtfb
        Observation name
    irefsp : int of length nqtfb
        Stress period to which the observation time is referenced.
        The reference point is the beginning of the specified stress period.
    toffset : float list of length nqtfb
        Is the time from the beginning of the stress period irefsp to the time
        of the observation.  toffset must be in units such that the product of
        toffset and tomultfb are consistent with other model input.  For steady
        state observations, specify irefsp as the steady state stress period
        and toffset less than or equal to perlen of the stress period.  If
        perlen is zero, set toffset to zero.  If the observation falls within
        a time step, linearly interpolation is used between values at the
        beginning and end of the time step.
    flwobs : float list of length nqtfb
        Observed flow value from the head-dependent flow boundary into the
        aquifer (+) or the flow from the aquifer into the boundary (-)
    layer : int list of length(nqfb, nqclfb)
        layer index for the cell included in the cell group
    row : int list of length(nqfb, nqclfb)
        row index for the cell included in the cell group
    column : int list of length(nqfb, nqclfb)
        column index of the cell included in the cell group
    factor : float list of length(nqfb, nqclfb)
        Is the portion of the simulated gain or loss in the cell that is
        included in the total gain or loss for this cell group (fn of eq. 5).
    flowtype : string
        String that corresponds to the head-dependent flow boundary condition
        type (CHD, GHB, DRN, RIV)
    extension : list of string
        Filename extension. If extension is None, extension is set to
        ['chob','obc','gbob','obg','drob','obd', 'rvob','obr']
        (default is None).
    no_print : boolean
        When True or 1, a list of flow observations will not be
        written to the Listing File (default is False)
    options : list of strings
        Package options (default is None).
    unitnumber : list of int
        File unit number. If unitnumber is None, unitnumber is set to
        [40, 140, 41, 141, 42, 142, 43, 143] (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the flwob output name will be created using
        the model name and .out extension (for example,
        modflowtest.out), if iufbobsv is a number greater than zero.
        If a single string is passed the package will be set to the string
        and flwob output name will be created using the model name and .out
        extension, if iufbobsv is a number greater than zero. To define the
        names for all package files (input and output) the length of the list
        of strings should be 2. Default is None.


    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    This represents a minimal working example that will be refactored in a
    future version.

    """

    def __init__(self, model, nqfb=0, nqcfb=0, nqtfb=0, iufbobsv=0,
                 tomultfb=1.0, nqobfb=None, nqclfb=None, obsnam=None,
                 irefsp=None, toffset=None, flwobs=None, layer=None,
                 row=None, column=None, factor=None, flowtype=None,
                 extension=None, no_print=False, options=None,
                 filenames=None, unitnumber=None):

        """
        Package constructor
        """
        if nqobfb is None:
            nqobfb = []
        if nqclfb is None:
            nqclfb = []
        if obsnam is None:
            obsnam = []
        if irefsp is None:
            irefsp = []
        if toffset is None:
            toffset = []
        if flwobs is None:
            flwobs = []
        if layer is None:
            layer = []
        if row is None:
            row = []
        if column is None:
            column = []
        if factor is None:
            factor = []
        if extension is None:
            extension = ['chob', 'obc', 'gbob', 'obg', 'drob', 'obd',
                         'rvob', 'obr']
        if unitnumber is None:
            unitnumber = [40, 140, 41, 141, 42, 142, 43, 143]

        if flowtype.upper().strip() == 'CHD':
            name = ['CHOB', 'DATA']
            extension = extension[0:2]
            unitnumber = unitnumber[0:2]
            iufbobsv = unitnumber[1]
            self.url = 'chob.htm'
            self.heading = '# CHOB for MODFLOW, generated by Flopy.'
        elif flowtype.upper().strip() == 'GHB':
            name = ['GBOB', 'DATA']
            extension = extension[2:4]
            unitnumber = unitnumber[2:4]
            iufbobsv = unitnumber[1]
            self.url = 'gbob.htm'
            self.heading = '# GBOB for MODFLOW, generated by Flopy.'
        elif flowtype.upper().strip() == 'DRN':
            name = ['DROB', 'DATA']
            extension = extension[4:6]
            unitnumber = unitnumber[4:6]
            iufbobsv = unitnumber[1]
            self.url = 'drob.htm'
            self.heading = '# DROB for MODFLOW, generated by Flopy.'
        elif flowtype.upper().strip() == 'RIV':
            name = ['RVOB', 'DATA']
            extension = extension[6:8]
            unitnumber = unitnumber[6:8]
            iufbobsv = unitnumber[1]
            self.url = 'rvob.htm'
            self.heading = '# RVOB for MODFLOW, generated by Flopy.'
        else:
            msg = 'ModflowFlwob: flowtype must be CHD, GHB, DRN, or RIV'
            raise KeyError(msg)

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # call base package constructor
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=unitnumber,
                         allowDuplicates=True, filenames=filenames)

        self.nqfb = nqfb
        self.nqcfb = nqcfb
        self.nqtfb = nqtfb
        self.iufbobsv = iufbobsv
        self.tomultfb = tomultfb
        self.nqobfb = nqobfb
        self.nqclfb = nqclfb
        self.obsnam = obsnam
        self.irefsp = irefsp
        self.toffset = toffset
        self.flwobs = flwobs
        self.layer = layer
        self.row = row
        self.column = column
        self.factor = factor

        # -create empty arrays of the correct size
        self.layer = np.zeros((self.nqfb, max(self.nqclfb)), dtype='int32')
        self.row = np.zeros((self.nqfb, max(self.nqclfb)), dtype='int32')
        self.column = np.zeros((self.nqfb, max(self.nqclfb)), dtype='int32')
        self.factor = np.zeros((self.nqfb, max(self.nqclfb)), dtype='float32')
        self.nqobfb = np.zeros((self.nqfb), dtype='int32')
        self.nqclfb = np.zeros((self.nqfb), dtype='int32')
        self.irefsp = np.zeros((self.nqtfb), dtype='int32')
        self.toffset = np.zeros((self.nqtfb), dtype='float32')
        self.flwobs = np.zeros((self.nqtfb), dtype='float32')

        # -assign values to arrays

        self.nqobfb[:] = nqobfb
        self.nqclfb[:] = nqclfb
        self.obsnam[:] = obsnam
        self.irefsp[:] = irefsp
        self.toffset[:] = toffset
        self.flwobs[:] = flwobs
        for i in range(self.nqfb):
            self.layer[i, :len(layer[i])] = layer[i]
            self.row[i, :len(row[i])] = row[i]
            self.column[i, :len(column[i])] = column[i]
            self.factor[i, :len(factor[i])] = factor[i]

        # add more checks here

        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append('NOPRINT')
        self.options = options

        # add checks for input compliance (obsnam length, etc.)
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # open file for writing
        f_fbob = open(self.fn_path, 'w')

        # write header
        f_fbob.write('{}\n'.format(self.heading))

        # write sections 1 and 2 : NOTE- what about NOPRINT?
        line = '{:10d}'.format(self.nqfb)
        line += '{:10d}'.format(self.nqcfb)
        line += '{:10d}'.format(self.nqtfb)
        line += '{:10d}'.format(self.iufbobsv)
        if self.no_print or 'NOPRINT' in self.options:
            line += '{: >10}'.format('NOPRINT')
        line += '\n'
        f_fbob.write(line)
        f_fbob.write('{:10e}\n'.format(self.tomultfb))

        # write sections 3-5 looping through observations groups
        c = 0
        for i in range(self.nqfb):
            #        while (i < self.nqfb):
            # write section 3
            f_fbob.write('{:10d}{:10d}\n'.format(self.nqobfb[i],
                                                 self.nqclfb[i]))

            # Loop through observation times for the groups
            for j in range(self.nqobfb[i]):
                # write section 4
                line = '{}{:10d}{:10.4g} {:10.4g}\n'.format(self.obsnam[c],
                                                            self.irefsp[c],
                                                            self.toffset[c],
                                                            self.flwobs[c])
                f_fbob.write(line)
                c += 1  # index variable

                # write section 5 - NOTE- need to adjust factor for multiple
                # observations in the same cell
            for j in range(abs(self.nqclfb[i])):
                # set factor to 1.0 for all cells in group
                if self.nqclfb[i] < 0:
                    self.factor[i, :] = 1.0
                line = '{:10d}'.format(self.layer[i, j])
                line += '{:10d}'.format(self.row[i, j])
                line += '{:10d}'.format(self.column[i, j])
                line += ' '.format(self.factor[i, j])
                # note is 10f good enough here?
                line += '{:10f}\n'.format(self.factor[i, j])
                f_fbob.write(line)

        f_fbob.close()

        #
        # swm: BEGIN hack for writing standard file
        sfname = self.fn_path
        sfname += '_ins'

        # write header
        f_ins = open(sfname, 'w')
        f_ins.write('jif @\n')
        f_ins.write('StandardFile 0 1 {}\n'.format(self.nqtfb))
        for i in range(0, self.nqtfb):
            f_ins.write('{}\n'.format(self.obsnam[i]))

        f_ins.close()
        # swm: END hack for writing standard file

        return
