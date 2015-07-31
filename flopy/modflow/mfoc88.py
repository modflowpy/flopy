"""
mfoc module.  Contains the ModflowOc class. Note that the user can access
the ModflowOc class as `flopy.modflow.ModflowOc`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?oc.htm>`_.

"""

import sys
from flopy.mbase import Package

class ModflowOc88(Package):
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
    def __init__(self, model, ihedfm=0, iddnfm=0, item2=[[0,1,0,1]], \
                 item3=[[0,0,1,0]], extension=['oc','hds','ddn','cbc'],\
                 unitnumber=[14, 51, 52, 53], save_head_every=None,\
                 words=None, compact=False, chedfm=None, cddnfm=None):

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
        Package.__init__(self, model, extension, \
                         ['OC', hds_fmt, ddn_fmt,\
                         'DATA(BINARY)'], unitnumber, \
                          extra=['','REPLACE','REPLACE','REPLACE'])
        self.heading = '# Output control package file'+\
                       ' for MODFLOW, generated by Flopy.'
        if words is not None:
            self.heading += ' Output control by words option'
        
        self.heading += '\n# Deprecated flopy OC class'
        
        print('Warning: ModflowOc88 flopy class is deprecated')
        print('         Please use ModflowOc class')

        self.url = 'oc.htm'
        self.ihedfm = ihedfm
        self.iddnfm = iddnfm
        self.chedfm = chedfm
        self.cddnfm = cddnfm

        # using words
        if words is not None:

            hflag,dflag = False,False
            if 'head' in words and ihedfm != 0:
                hflag = True
            if 'drawdown' in words and iddnfm != 0:
                dflag = True

            self.words = []
            self.compact = compact

            # first try for simple 1-d list
            try:
                for w in words:
                    self.words.append(w.upper())

                # build a list of word output options
                word_list = []

                if save_head_every is None:
                    raise TypeError('to use the 1d words OC option, save_head_every must be used')

                nstp = self.parent.get_package('DIS').nstp
                for p in range(len(nstp)):
                    for s in range(nstp[p]):
                        if s % save_head_every == 0:
                            word_list.append('PERIOD {0:5.0f} STEP {1:5.0f}\n'\
                                                .format(p+1,s+1))
                            for w in words:
                                if 'PBUDGET' in w.upper():
                                    word_list.append('  PRINT BUDGET\n')
                                elif 'PHEAD' in w.upper():
                                    word_list.append('  PRINT HEAD\n')
                                else:
                                    word_list.append('  SAVE '+w.upper()+'\n')
                            if hflag:
                                word_list.append('  PRINT HEAD\n')
                            if dflag:
                                word_list.append('  PRINT DRAWDOWN\n')
                            word_list.append('\n')
                self.word_list = word_list



            # try for a 2-d list
            except:
                word_list = []
                self.words = []
                for i in words:
                    p,s = int(i[0]),int(i[1])
                    wwords = i[2:]
                    word_list.append('PERIOD {0:5.0f} STEP {1:45.0f}\n'\
                                                 .format(p,s))
                    for w in wwords:
                        if 'PBUDGET' in w.upper():
                            word_list.append('  PRINT BUDGET\n')
                        elif 'PHEAD' in w.upper():
                            word_list.append('  PRINT HEAD\n')
                        else:
                            word_list.append('  SAVE '+w.upper()+'\n')
                        if w.upper() not in self.words:
                            self.words.append(w.upper())
                    if hflag:
                        word_list.append('  PRINT HEAD\n')
                    if dflag:
                        word_list.append('  PRINT DRAWDOWN\n')
                    word_list.append('\n')
                self.word_list = (word_list)

        # numeric codes
        else:
            self.words = None
            #dummy, self.item2 = self.assign_layer_row_column_data(item2, 4, zerobase=False)  # misuse of this function - zerobase needs to be False
            if (item2 != None):
                error_message = 'item2 must have 4 columns'
                if (not isinstance(item2, list)):
                    item2 = [item2]
                for a in item2:
                    assert len(a) == 4, error_message
                self.item2 = item2
            if (item3 != None):
                error_message = 'item3 must have 4 columns'
                if (not isinstance(item3, list)):
                    item3 = [item3]
                for a in item3:
                    assert len(a) == 4, error_message
                self.item3 = item3
            if save_head_every is not None:
                nstp = self.parent.get_package('DIS').nstp
                self.item3 = []
                #len(nstp) is the number of stress periods
                for p in range(len(nstp)):
                    for s in range(1,nstp[p]+1):
                        if s % save_head_every == 0:
                            self.item3.append([0,0,1,0])
                        else:
                            self.item3.append([0,0,0,0])

        self.parent.add_package(self)

    def __repr__( self ):
        return 'Output control package class -- deprecated'

    def write_file(self):
        """
        Write the file.

        """
        f_oc = open(self.fn_path, 'w')
        f_oc.write('%s\n' % self.heading)
        nstp = self.parent.get_package('DIS').nstp
                
        # words option
        if self.words is not None:            
           
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
            if self.compact:
                f_oc.write('COMPACT BUDGET FILES')
            f_oc.write('\n')
            for i in self.word_list:
                f_oc.write(i)
        
        # numeric codes option
        else:                                  
            f_oc.write('%3i%3i%5i%5i\n' % \
                      (self.ihedfm, self.iddnfm, self.unit_number[1],\
                       self.unit_number[2]))
            
            ss = 0
            #len(nstp) is the number of stress periods
            for p in range(len(nstp)): 
                for s in range(nstp[p]):
                    if (ss < len(self.item2)):
                        a = self.item2[ss]
                    else:
                        a = self.item2[-1]
                    if (ss < len(self.item3)):
                        b = self.item3[ss]
                    else:
                        b = self.item3[-1]
                    f_oc.write('%3i%3i%3i%3i  Period %3i, step %3i\n' \
                               % (a[0], a[1], a[2], a[3], p + 1, s + 1) )

                    # incode > 0 means that item3 must have one record for each 
                    # layer, so perform check here
                    if (a[0] > 0): 
                        nr, nc = b.shape                
                        assert nr == nlay, 'item3 must have {0:1d} rows when incode > 0 ' % (nlay)                
                        for bb in b:
                            f_oc.write('%3i%3i%3i%3i\n' % (bb[0], bb[1], bb[2], bb[3]) )
                    else:
                        f_oc.write('%3i%3i%3i%3i\n' % (b[0], b[1], b[2], b[3]) )
                    ss = ss + 1
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

        #open file
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        #process each line
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
                if len(wordrec) > 3:
                    words.append(wordrec)
                iperoc = int(lnlst[1])
                itsoc = int(lnlst[3])
                wordrec = [iperoc, itsoc]

            #dataset 3
            elif 'PRINT' in lnlst[0].upper() and 'HEAD' in lnlst[1].upper():
                wordrec.append('PHEAD')
            elif ('PRINT' in lnlst[0].upper() and
                          'DRAWDOWN' in lnlst[1].upper()):
                wordrec.append('PDRAWDOWN')
            elif 'PRINT' in lnlst[0].upper() and 'BUDGET' in lnlst[1].upper():
                wordrec.append('PBUDGET')
            elif 'SAVE' in lnlst[0].upper() and 'HEAD' in lnlst[1].upper():
                wordrec.append('HEAD')
            elif ('SAVE' in lnlst[0].upper() and
                          'DRAWDOWN' in lnlst[1].upper()):
                wordrec.append('DRAWDOWN')
            elif 'SAVE' in lnlst[0].upper() and 'IBOUND' in lnlst[1].upper():
                wordrec.append('IBOUND')
            elif 'SAVE' in lnlst[0].upper() and 'BUDGET' in lnlst[1].upper():
                wordrec.append('BUDGET')
            else:
                print('Old style oc files not supported for import.')
                print('Convert to words.')
                return ModflowOc88(model)

        #store the last record in word
        if len(wordrec) > 3:
            words.append(wordrec)

        # reset unit numbers
        unitnumber=[14, 51, 52, 53]
        if ihedun > 0:
            model.add_pop_key_list(ihedun)
            #unitnumber[1] = ihedun
        if iddnun > 0:
            model.add_pop_key_list(iddnun)
            #unitnumber[2] = iddnun

        # create instance of oc class
        oc = ModflowOc88(model, ihedfm=ihedfm, iddnfm=iddnfm,
                 extension=['oc','hds','ddn','cbc'],
                 unitnumber=unitnumber, words=words, compact=compact,
                 chedfm=chedfm, cddnfm=cddnfm)

        return oc