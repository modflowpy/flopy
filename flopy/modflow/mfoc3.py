"""
mfoc module.  Contains the ModflowOc class. Note that the user can access
the ModflowOc class as `flopy.modflow.ModflowOc`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?oc.htm>`_.

"""

import sys
from flopy.mbase import Package

class ModflowOc3(Package):
    """
    MODFLOW Output Control Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ihedfm : int
        is a code for the format in which heads will be printed.
        (default is 0).
    iddnfm : int
        is a code for the format in which heads will be printed.
        (default is 0).
    item2 : list of ints
        [incode, ihddfl, ibudfl, icbcfl], where incode is the code for reading
        Item 3. ihddfl is a head and drawdown output flag. This flag allows
        Item 3 flags to be specified in an early time step and then used or not
        used in subsequent time steps. Thus, it may be possible to use IHDDFL
        to avoid resetting Item 3 flags every time step.  ibudfl is a budget
        print flag. icbcfl is a flag for writing cell-by-cell flow data.
        (default is [[0, 1, 0, 1]]).
    item3 : list of ints
        [hdpr, ddpr, hdsv, ddsv]
        hdpr is the output flag for head printout.
        ddpr is the output flag for drawdown printout.
        hdsv is the output flag for head save.
        ddsv is the output flag for drawdown save.
        (default is [[0, 0, 1, 0]]).
    extension : list of strings
        (default is ['oc','hds','ddn','cbc']).
    unitnumber : list of ints
        (default is [14, 51, 52, 53]).
    save_head_every : int
        Time step interval for printing and/or saving results
        (default is None).
    words : list of instructions
        Can be specified as a 2d list of the following form:
            [[per,stp,'head','drawdown','budget','pbudget', 'phead']]
        In this 2d form, phead, pbudget will print the head and budget.
        Words can also be a 1d list of data items, such as
            ['head','drawdown','budget'].
        With a 1d list, the save_head_every option is used to determine the
        output frequency.
        (default is None).
    compact : boolean
        Save results in compact budget form. (default is False).
    chedfm : string
        is a character value that specifies the format for saving heads, and
        can only be specified if the word method of output control is used.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CHEDFM, then heads are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.
    cddnfm : string
        is a character value that specifies the format for saving drawdown, and
        can only be specified if the word method of output control is used.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CDDNFM, then drawdowns are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The "words" method for specifying output control is preferred in most
    cases.  Also, the "compact" budget should normally be used as it produces
    files that are typically much smaller.  The compact budget form is also
    a requirement for using the MODPATH particle tracking program.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> oc = flopy.modflow.ModflowOc(m, words=['head'], save_head_every=1)

    """
    def __init__(self, model,\
                 ihedfm=0, iddnfm=0, chedfm=None, cddnfm=None,\
                 ibndsav=None, compact=False,\
                 stress_period_data={(1,1):['save head']},\
                 extension=['oc','hds','ddn','cbc'],\
                 unitnumber=[14, 51, 52, 53]):

        '''
           words = list containing any of ['head','drawdown','budget']
           optionally, words in a 2-D list of shape:
           [[per,stp,'head','drawdown','budget']], where
           per,stp is the stress period,time step of output.
           To print heads/drawdowns, ihedfm/iddnfm must be non-zero

        '''

        # Call ancestor's init to set self.parent,
        # extension, name and unit number
        hds_fmt = 'DATA(BINARY)'
        ddn_fmt = 'DATA(BINARY)'
        if chedfm is not None:
            hds_fmt = 'DATA'
        if cddnfm is not None:
            ddn_fmt = 'DATA'

        ibouun = 0
        cboufm = None

        name = ['OC', hds_fmt, ddn_fmt, 'DATA(BINARY)']
        extra = ['', 'REPLACE', 'REPLACE', 'REPLACE']
        if ibndsav is not None:
            icont = True
            if ibndsav == True:
                name.append('DATA(BINARY)')
            else:
                try:
                    cboufm = ibndsav
                    name.append('DATA')
                except:
                    icont = False
            if icont == True:
                extension.append('ibo')
                unitsnumber.append(114)
                extra.append('REPLACE')

        Package.__init__(self, model, extension=extension, name=name, unit_number=unitnumber,
                         extra=extra)  # Call ancestor's init to set self.parent, extension, name and unit number


        self.heading = '# Output control package file'+\
                       ' for MODFLOW, generated by Flopy.'

        self.url = 'oc.htm'
        self.ihedfm = ihedfm
        self.iddnfm = iddnfm
        self.chedfm = chedfm
        self.cddnfm = cddnfm

        self.ibouun = ibouun
        self.cboufm = cboufm

        self.stress_period_data = stress_period_data

        self.parent.add_package(self)

    def __repr__( self ):
        return 'Output control package class'

    def write_file(self):
        """
        Write the file.

        """
        f_oc = open(self.fn_path, 'w')
        f_oc.write('{}\n'.format(self.heading))

        #--write options

        f_oc.write('HEAD PRINT FORMAT {0:3.0f}\n'\
                   .format(self.ihedfm))            
        if self.chedfm is not None:
            f_oc.write('HEAD SAVE FORMAT {0:20s} LABEL\n'\
                       .format(self.chedfm))            
        f_oc.write('HEAD SAVE UNIT {0:5.0f}\n'\
                   .format(self.unit_number[1]))            
            
        f_oc.write('DRAWDOWN PRINT FORMAT {0:3.0f}\n'\
                   .format(self.iddnfm))
        if self.cddnfm is not None:
            f_oc.write('DRAWDOWN SAVE FORMAT {0:20s} LABEL\n'\
                       .format(self.cddnfm))
        f_oc.write('DRAWDOWN SAVE UNIT {0:5.0f}\n'\
                   .format(self.unit_number[2]))

        if self.ibouun > 0:
            if self.cboufm is not None:
                f_oc.write('IBOUND SAVE FORMAT {0:20s} LABEL\n'\
                            .format(self.cboufm))
            f_oc.write('IBOUND SAVE UNIT {0:5.0f}\n'\
                       .format(self.unit_number[4]))

        if self.compact:
            f_oc.write('COMPACT BUDGET FILES')
        f_oc.write('\n')


        #write the transient sequence described by the data dict
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        nstp = self.parent.get_package('DIS').nstp

        keys = self.stress_period_data.keys()
        keys.sort()

        data = []
        lines = ''
        for kper in xrange(nper):
            for kstp in xrange(nstp[kper]):
                kperkstp = (kper, kstp)
                if kperkstp in keys:
                    data = self.stress_period_data[kperkstp]
                    lines = ''
                    if len(data) > 0:
                        for item in data:
                            lines += '{}\n'.format(item)
                if len(lines) > 0:
                    f_oc.write('period {} step {}\n'.format(kper+1, kstp+1))
                    f_oc.write(lines)

        #--close oc file
        f_oc.close()

    @staticmethod
    def load(f, model, nper=None, ext_unit_dict=None):
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
        oc : ModflowOc object
            ModflowOc object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> oc = flopy.modflow.ModflowOc.load('test.oc', m)

        """

        if model.verbose:
            sys.stdout.write('loading oc package file...\n')

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        #initialize
        ihedfm = 0
        iddnfm = 0
        ihedun = 0
        iddnun = 0
        compact = False
        chedfm = None
        cddnfm = None
        words = []
        wordrec = []

        stress_period_data = {}

        #open file
        if type(f) is not file:
            filename = f
            f = open(filename, 'r')

        #process each line
        lines = ''
        iperoc, itsoc = 0, 0 
        for line in f:
            lnlst = line.strip().split()
            if line[0] == '#':
                continue
            
            # added by JJS 12/12/14 to avoid error when there is a blank line in the OC file
            if lnlst == []:
                continue
            # end add

            #dataset 1 values
            elif ('HEAD' in lnlst[0].upper() and
                  'PRINT' in lnlst[1].upper() and
                  'FORMAT' in lnlst[2].upper()
                  ):
                ihedfm = int(lnlst[3])
            elif ('HEAD' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'FORMAT' in lnlst[2].upper()
                  ):
                chedfm = lnlst[3]
            elif ('HEAD' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'UNIT' in lnlst[2].upper()
                  ):
                ihedun = int(lnlst[3])
            elif ('DRAWDOWN' in lnlst[0].upper() and
                  'PRINT' in lnlst[1].upper() and
                  'FORMAT' in lnlst[2].upper()
                  ):
                iddnfm = int(lnlst[3])
            elif ('DRAWDOWN' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'FORMAT' in lnlst[2].upper()
                  ):
                cddnfm = lnlst[3]
            elif ('DRAWDOWN' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'UNIT' in lnlst[2].upper()
                  ):
                iddnun = int(lnlst[3])
            elif ('IBOUND' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'FORMAT' in lnlst[2].upper()
                  ):
                cboufm = lnlst[3]
            elif ('IBOUND' in lnlst[0].upper() and
                  'SAVE' in lnlst[1].upper() and
                  'UNIT' in lnlst[2].upper()
                  ):
                ibouun = int(lnlst[3])
            elif 'COMPACT' in lnlst[0].upper():
                compact = True

            #dataset 2
            elif 'PERIOD' in lnlst[0].upper():
                #--create period step tuple
                kperkstp = (iperoc-1, itsoc-1)
                #--save data
                stress_period_data[kperkstp] = lines
                #--reset lines
                lines = ''
                #--update iperoc and itsoc
                iperoc = int(lnlst[1])
                itsoc = int(lnlst[3])

            #dataset 3
            elif 'PRINT' in lnlst[0].upper():
                lines.append('{} {}'.format(lnlst[0].lower(), lnlst[1].lower()))
            elif 'SAVE' in lnlst[0].upper() :
                lines.append('{} {}'.format(lnlst[0].lower(), lnlst[1].lower()))
            else:
                print 'Old style oc files not supported for import.'
                print 'Convert to words.'
                return ModflowOc3(model)

        #store the last record in word
        if len(lines) > 0:
            #--create period step tuple
            kperkstp = (iperoc-1, itsoc-1)
            #--save data
            stress_period_data[kperkstp] = lines

        #--reset unit numbers
        unitnumber=[14, 51, 52, 53]
        if ihedun > 0:
            model.add_pop_key_list(ihedun)
            #unitnumber[1] = ihedun
        if iddnun > 0:
            model.add_pop_key_list(iddnun)
            #unitnumber[2] = iddnun

        #--create instance of oc class
        oc = ModflowOc3(model, ihedfm=ihedfm, iddnfm=iddnfm,
                 chedfm=chedfm, cddnfm=cddnfm, compact=compact,
                 stress_period_data=stress_period_data,
                 extension=['oc','hds','ddn','cbc'],
                 unitnumber=unitnumber)

        return oc