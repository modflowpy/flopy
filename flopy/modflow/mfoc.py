"""
mfoc module.  Contains the ModflowOc class. Note that the user can access
the ModflowOc class as `flopy.modflow.ModflowOc`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?oc.htm>`_.

"""

import sys
from flopy.mbase import Package

class ModflowOc(Package):
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
    chedfm : string
        is a character value that specifies the format for saving heads.
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
        (default is None)
    cddnfm : string
        is a character value that specifies the format for saving drawdown.
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
        (default is None)
    cboufm : string
        is a character value that specifies the format for saving ibound.
        The format must contain 20 characters or less and must be a valid
        Fortran format that is enclosed in parentheses. The format must be
        enclosed in apostrophes if it contains one or more blanks or commas.
        The optional word LABEL after the format is used to indicate that
        each layer of output should be preceded with a line that defines the
        output (simulation time, the layer being output, and so forth). If
        there is no record specifying CBOUFM, then ibounds are written to a
        binary (unformatted) file. Binary files are usually more compact than
        text files, but they are not generally transportable among different
        computer operating systems or different Fortran compilers.
        (default is None)
    stress_period_data : dictionary of of lists
        Dictionary key is a tuple with the zero-based period and step 
        (IPEROC, ITSOC) for each print/save option list. 
        (default is {(0,0):['save head']})
        
        The list can have any valid MODFLOW OC print/save option:
            PRINT HEAD
            PRINT DRAWDOWN
            PRINT BUDGET
            SAVE HEAD
            SAVE DRAWDOWN
            SAVE BUDGET
            SAVE IBOUND
            
            The lists can also include (1) DDREFERENCE in the list to reset 
            drawdown reference to the period and step and (2) a list of layers 
            for PRINT HEAD, SAVE HEAD, PRINT DRAWDOWN, SAVE DRAWDOWN, and
            SAVE IBOUND.
        
        The list is used for every stress period and time step after the 
        (IPEROC, ITSOC) tuple until a (IPEROC, ITSOC) tuple is entered with
        and empty list.
    compact : boolean
        Save results in compact budget form. (default is True).
    extension : list of strings
        (default is ['oc','hds','ddn','cbc']).
    unitnumber : list of ints
        (default is [14, 51, 52, 53]).

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The "words" method for specifying output control is the only option 
    available.  Also, the "compact" budget should normally be used as it 
    produces files that are typically much smaller.  The compact budget form is 
    also a requirement for using the MODPATH particle tracking program.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> spd = {(0, 0): ['print head'],
    ...   (0, 1): [],
    ...   (0, 249): ['print head'],
    ...   (0, 250): [],
    ...   (0, 499): ['print head', 'save ibound'],
    ...   (0, 500): [],
    ...   (0, 749): ['print head', 'ddreference'],
    ...   (0, 750): [],
    ...   (0, 999): ['print head']}
    >>> oc = flopy.modflow.ModflowOc3(m, stress_period_data=spd, cboufm='(20i5)')

    """
    def __init__(self, model,\
                 ihedfm=0, iddnfm=0, chedfm=None, cddnfm=None,\
                 cboufm=None, compact=True,\
                 stress_period_data={(0,0):['save head']},\
                 extension=['oc','hds','ddn','cbc'],\
                 unitnumber=[14, 51, 52, 53]):

        """
        Package constructor.

        """

        # Call ancestor's init to set self.parent,
        # extension, name and unit number
        hds_fmt = 'DATA(BINARY)'
        ddn_fmt = 'DATA(BINARY)'
        if chedfm is not None:
            hds_fmt = 'DATA'
        if cddnfm is not None:
            ddn_fmt = 'DATA'

        ibouun = 0
        ibndsav = False
        for key in list(stress_period_data.keys()):
            t = stress_period_data[key]
            if len(t) > 0:
                for option in t:
                    if 'ibound' in option.lower():
                        ibndsav = True
                        break

        name = ['OC', hds_fmt, ddn_fmt, 'DATA(BINARY)']
        extra = ['', 'REPLACE', 'REPLACE', 'REPLACE']
        if ibndsav == True:
            if cboufm == None:
                name.append('DATA(BINARY)')
            else:
                name.append('DATA')
            extension.append('ibo')
            unitnumber.append(114)
            ibouun = unitnumber[-1]
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
        
        self.compact = compact

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

        # write options

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
            f_oc.write('COMPACT BUDGET FILES\n')
        
        # add a line separator between header and stress
        #  period data
        f_oc.write('\n')


        #write the transient sequence described by the data dict
        nr, nc, nl, nper = self.parent.get_nrow_ncol_nlay_nper()
        nstp = self.parent.get_package('DIS').nstp

        keys = list(self.stress_period_data.keys())
        keys.sort()

        data = []
        lines = ''
        ddnref = ''
        for kper in range(nper):
            for kstp in range(nstp[kper]):
                kperkstp = (kper, kstp)
                if kperkstp in keys:
                    data = self.stress_period_data[kperkstp]
                    if not isinstance(data, list):
                        data = [data]
                    lines = ''
                    if len(data) > 0:
                        for item in data:
                            if 'DDREFERENCE' in item.upper():
                                ddnref = item.lower()
                            else:
                                lines += '{}\n'.format(item)
                if len(lines) > 0:
                    f_oc.write('period {} step {} {}\n'.format(kper+1, kstp+1, ddnref))
                    f_oc.write(lines)
                    f_oc.write('\n')
                    ddnref = ''

        # close oc file
        f_oc.close()

    @staticmethod
    def load(f, model, nper=None, nstp=None, nlay=None, ext_unit_dict=None):
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
        nstp : list
            List containing the number of time steps in each stress period.  
            If nstp is None, then nstp will be obtained from the DIS package 
            attached to the model object. (default is None).
        nlay : int
            The number of model layers.  If nlay is None, then nnlay will be
            obtained from the model object. nlay only needs to be specified
            if an empty model object is passed in and the oc file being loaded
            is defined using numeric codes. (default is None).
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
        
        if nstp is None:
            nstp = model.get_package('DIS').nstp.array


        #initialize
        ihedfm = 0
        iddnfm = 0
        ihedun = 0
        iddnun = 0
        ibouun = 0
        compact = False
        chedfm = None
        cddnfm = None
        cboufm = None
        words = []
        wordrec = []
        
        numericformat = False
        ihedfm, iddnfm = 0, 0

        stress_period_data = {}

        #open file
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # read header
        ipos = f.tell()
        while True:
            line = f.readline()
            if line[0] == '#':
                continue
            elif line[0] == []:
                continue
            else:
                lnlst = line.strip().split()
                try:
                    ihedfm, iddnfm = int(lnlst[0]), int(lnlst[1])
                    ihedun, iddnun = int(lnlst[2]), int(lnlst[3])
                    numericformat = True
                except:
                    f.seek(ipos)
                    pass
                # exit so the remaining data can be read
                #  from the file based on numericformat
                break
            # set pointer to current position in the OC file
            ipos = f.tell()
                 
        
        #process each line
        lines = []
        if numericformat == True:
            for iperoc in range(nper):
                for itsoc in range(nstp[iperoc]):
                    line = f.readline()
                    lnlst = line.strip().split()
                    incode, ihddfl = int(lnlst[0]), int(lnlst[1])
                    ibudfl, icbcfl = int(lnlst[2]), int(lnlst[3])
                    # new print and save flags are needed if incode is not
                    #  less than 0.
                    if incode >= 0:
                        lines = []
                    # use print options from the last time step
                    else:
                        if len(lines) > 0:
                            stress_period_data[(iperoc, itsoc)] = list(lines)
                        continue
                    # set print and save budget flags
                    if ibudfl != 0:
                        lines.append('PRINT BUDGET')
                    if icbcfl != 0:
                        lines.append('PRINT BUDGET')
                    if incode == 0:
                        line = f.readline()
                        lnlst = line.strip().split()
                        hdpr, ddpr = int(lnlst[0]), int(lnlst[1])
                        hdsv, ddsv = int(lnlst[2]), int(lnlst[3])
                        if hdpr != 0:
                            lines.append('PRINT HEAD')
                        if ddpr != 0:
                            lines.append('PRINT DRAWDOWN')
                        if hdsv != 0:
                            lines.append('SAVE HEAD')
                        if ddsv != 0:
                            lines.append('SAVE DRAWDOWN')
                    elif incode > 0:
                        headprint = ''
                        headsave = ''
                        ddnprint = ''
                        ddnsave = ''
                        for k in range(nlay):
                            line = f.readline()
                            lnlst = line.strip().split()
                            hdpr, ddpr = int(lnlst[0]), int(lnlst[1])
                            hdsv, ddsv = int(lnlst[2]), int(lnlst[3])
                            if hdpr != 0:
                                headprint += ' {}'.format(k+1)
                            if ddpr != 0:
                                ddnprint += ' {}'.format(k+1)
                            if hdsv != 0:
                                headsave += ' {}'.format(k+1)
                            if ddsv != 0:
                                ddnsave += ' {}'.format(k+1)
                        if len(headprint) > 0:
                            lines.append('PRINT HEAD'+headprint)
                        if len(ddnprint) > 0:
                            lines.append('PRINT DRAWDOWN'+ddnprint)
                        if len(headsave) > 0:
                            lines.append('SAVE HEAD'+headdave)
                        if len(ddnsave) > 0:
                            lines.append('SAVE DRAWDOWN'+ddnsave)
                    stress_period_data[(iperoc, itsoc)] = list(lines)
        else:
            iperoc, itsoc = 0, 0 
            while True:
                line = f.readline()
                if len(line) < 1:
                    break
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
                    if len(lines) > 0:
                        if iperoc > 0:
                            # create period step tuple
                            kperkstp = (iperoc-1, itsoc-1)
                            # save data
                            stress_period_data[kperkstp] = lines
                        # reset lines
                        lines = []
                    # turn off oc if required
                    if iperoc > 0:
                        if itsoc==nstp[iperoc-1]:
                            iperoc1 = iperoc + 1
                            itsoc1 = 1
                        else:
                            iperoc1 = iperoc
                            itsoc1 = itsoc + 1
                    else:
                        iperoc1, itsoc1 = iperoc, itsoc 
                    # update iperoc and itsoc
                    iperoc = int(lnlst[1])
                    itsoc = int(lnlst[3])
                    # do not used data that exceeds nper
                    if iperoc > nper:
                        break
                    # add a empty list if necessary
                    iempty = False
                    if iperoc != iperoc1:
                        iempty = True
                    else:
                        if itsoc != itsoc1:
                            iempty = True
                    if iempty == True:
                        kperkstp = (iperoc1-1, itsoc1-1)
                        stress_period_data[kperkstp] = []
                #dataset 3
                elif 'PRINT' in lnlst[0].upper():
                    lines.append('{} {}'.format(lnlst[0].lower(), lnlst[1].lower()))
                elif 'SAVE' in lnlst[0].upper() :
                    lines.append('{} {}'.format(lnlst[0].lower(), lnlst[1].lower()))
                else:
                    print('Error encountered in OC import.')
                    print('Creating default OC package.')
                    return ModflowOc(model)
    
            #store the last record in word
            if len(lines) > 0:
                # create period step tuple
                kperkstp = (iperoc-1, itsoc-1)
                # save data
                stress_period_data[kperkstp] = lines
                # add a empty list if necessary
                iempty = False
                if iperoc != iperoc1:
                    iempty = True
                else:
                    if itsoc != itsoc1:
                        iempty = True
                if iempty == True:
                    kperkstp = (iperoc1-1, itsoc1-1)
                    stress_period_data[kperkstp] = []
                    
        # reset unit numbers
        unitnumber=[14, 51, 52, 53]
        if ihedun > 0:
            model.add_pop_key_list(ihedun)
        if iddnun > 0:
            model.add_pop_key_list(iddnun)
        if ibouun > 0:
            model.add_pop_key_list(ibouun)
            if cboufm == None:
                cboufm = True
                

        # create instance of oc class
        oc = ModflowOc(model, ihedfm=ihedfm, iddnfm=iddnfm,
                 chedfm=chedfm, cddnfm=cddnfm, cboufm=cboufm,
                 compact=compact,
                 stress_period_data=stress_period_data)

        return oc