from __future__ import print_function
import os
import sys
import copy
import numpy as np
from .binaryfile import CellBudgetFile
from itertools import groupby
from collections import OrderedDict
from ..utils.utils_def import totim_to_datetime


class ZoneBudget(object):
    """
    ZoneBudget class

    Parameters
    ----------
    cbc_file : str or CellBudgetFile object
        The file name or CellBudgetFile object for which budgets will be
        computed.
    z : ndarray
        The array containing to zones to be used.
    kstpkper : tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    totim : float
        The simulation time.
    aliases : dict
        A dictionary with key, value pairs of zones and aliases. Replaces
        the corresponding record and field names with the aliases provided.
        NOTE: When using this option in conjunction with a list of zones,
        the zone(s) passed may either be all strings (aliases), all
        integers, or mixed.

    Example usage:

    >>> from flopy.utils.zonbud import ZoneBudget, read_zbarray
    >>> zon = read_zbarray('zone_input_file')
    >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
    >>> zb.to_csv('zonebudtest.csv')
    >>> zb_mgd = zb * 7.48052 / 1000000
    """
    def __init__(self, cbc_file, z, kstpkper=None, totim=None, aliases=None, **kwargs):

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)
        else:
            raise Exception('Cannot load cell budget file: {}.'.format(cbc_file))

        # Zones must be passed as an array
        assert isinstance(z, np.ndarray), 'Please pass zones as type {}'.format(np.ndarray)

        # Check for negative zone values
        for zi in np.unique(z):
            if zi < 0:
                raise Exception('Negative zone value(s) found:', zi)

        self.dis = None
        self.sr = None
        if 'model' in kwargs.keys():
            self.model = kwargs.pop('model')
            self.sr = self.model.sr
            self.dis = self.model.dis
        if 'dis' in kwargs.keys():
            self.dis = kwargs.pop('dis')
            self.sr = self.dis.parent.sr
        if 'sr' in kwargs.keys():
            self.sr = kwargs.pop('sr')
        if len(kwargs.keys()) > 0:
            args = ','.join(kwargs.keys())
            raise Exception('LayerFile error: unrecognized kwargs: ' + args)

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.cbc.get_data(idx=0, full3D=True)[0].shape
        self.nlay, self.nrow, self.ncol = self.cbc_shape
        self.cbc_times = self.cbc.get_times()
        self.cbc_kstpkper = self.cbc.get_kstpkper()
        self.kstpkper = None
        self.totim = None

        if kstpkper is not None:
            if isinstance(kstpkper, tuple):
                kstpkper = [kstpkper]
            for kk in kstpkper:
                s = 'The specified time step/stress period ' \
                    'does not exist {}'.format(kk)
                assert kk in self.cbc.get_kstpkper(), s
            self.kstpkper = kstpkper
        elif totim is not None:
            if isinstance(totim, float):
                totim = [totim]
            elif isinstance(totim, int):
                totim = [float(totim)]
            for t in totim:
                s = 'The specified simulation time ' \
                    'does not exist {}'.format(t)
                assert t in self.cbc.get_times(), s
            self.totim = totim
        else:
            # No time step/stress period or simulation time pass
            self.kstpkper = self.cbc.get_kstpkper()

        # Set float and integer types
        self.float_type = np.float32
        self.int_type = np.int32
        is_64bit = sys.maxsize > 2**32
        if is_64bit:
            self.float_type = np.float64
            self.int_type = np.int64

        # Make sure the input zone array has the same shape as the cell budget file
        if len(z.shape) == 2 and self.nlay == 1:
            # Reshape a 2-D array to 3-D to match output from
            # the CellBudgetFile object.
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[0, :, :] = z[:, :]
        elif len(z.shape) == 2 and self.nlay > 1:
            # 2-D array specified, but model is more than 1 layer. Don't assume
            # user wants same zones for all layers.
            raise Exception('Zone array and CellBudgetFile shapes '
                            'do not match {} {}'.format(z.shape, self.cbc_shape))
        elif len(z.shape) == 3:
            izone = z.copy()
        else:
            raise Exception('Shape of the zone array is not recognized: {}'.format(z.shape))

        assert izone.shape == self.cbc_shape, \
            'Shape of input zone array {} does not ' \
            'match the cell by cell ' \
            'budget file {}'.format(izone.shape, self.cbc_shape)

        # self.kstpkper = kstpkper
        # self.totim = totim
        self.izone = izone
        self.allzones = [z for z in np.unique(self.izone)]
        self._zonefieldnamedict = OrderedDict([(z, 'ZONE_{}'.format(z))
                                               for z in self.allzones if z != 0])

        if aliases is not None:
            assert isinstance(aliases, dict), 'Input aliases not recognized. Please pass a dictionary ' \
                                              'with key,value pairs of zone/alias.'
            # Replace the relevant field names (ignore zone 0)
            seen = []
            for z, a in iter(aliases.items()):
                if z != 0 and z in self._zonefieldnamedict.keys():
                    if z in seen:
                        raise Exception('Zones may not have more than 1 alias.')
                    self._zonefieldnamedict[z] = '_'.join(a.split())
                    seen.append(z)

        self._iflow_from_recnames, self._iflow_to_recnames = self._get_internal_flow_record_names()
        self._zonefieldnames = list(self._zonefieldnamedict.values())

        # All record names in the cell-by-cell budget binary file
        self.record_names = [n.strip().decode("utf-8") for n in self.cbc.unique_record_names()]

        # Get imeth for each record in the CellBudgetFile record list
        self.imeth = {}
        for record in self.cbc.recordarray:
            self.imeth[record['text'].strip().decode("utf-8")] = record['imeth']

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        # FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO--- terms are used by the SWI2 groundwater flow process.
        internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

        # Source/sink/storage term record names
        # These are all of the terms that are not related to constant
        # head cells or face flow terms
        self.ssst_record_names = [n for n in self.record_names
                                  if n not in internal_flow_terms]

        # Build budget record array
        array_list = []
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                recordarray = self._compute_budget(kstpkper=kk)
                array_list.append(recordarray)
        elif self.totim is not None:
            for t in self.totim:
                recordarray = self._compute_budget(totim=t)
                array_list.append(recordarray)
        self._budget = np.concatenate(array_list, axis=0)
        return

    def get_model_shape(self):
        return self.nlay, self.nrow, self.ncol

    def get_record_names(self):
        """
        Get a list of water budget record names in the file.

        Returns
        -------
        out : list of strings
            List of unique text names in the binary file.

        Example usage:

        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> recnames = zb.get_record_names()

        """
        return np.unique(self._budget['name'])

    def get_budget(self, names=None, zones=None):
        """
        Get a list of zonebudget record arrays.

        Parameters
        ----------

        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.

        Returns
        -------
        budget_list : list of reecord arrays
            A list of the zonebudget record arrays.

        Example usage:

        >>> names = ['CONSTANT_HEAD_IN', 'RIVER_LEAKAGE_OUT']
        >>> zones = ['ZONE_1', 'ZONE_2']
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> bud = zb.get_budget(names=names, zones=zones)
        """
        if isinstance(names, str):
            names = [names]
        if isinstance(zones, str):
            zones = [zones]
        elif isinstance(zones, int):
            zones = [zones]
        if zones is not None:
            for idx, z in enumerate(zones):
                if isinstance(z, int):
                    zones[idx] = 'ZONE_{}'.format(z)
            select_fields = ['totim', 'time_step', 'stress_period', 'name'] + zones
        else:
            select_fields = ['totim', 'time_step', 'stress_period', 'name'] + self._zonefieldnames
        if names is not None:
            select_records = np.in1d(self._budget['name'], names)
        else:
            select_records = np.where((self._budget['name'] == self._budget['name']))
        return self._budget[select_fields][select_records]

    def to_csv(self, fname):
        """
        Saves the budget record arrays to a formatted
        comma-separated values file.

        Parameters
        ----------
        fname : str
            The name of the output comma-separated values file.

        Returns
        -------
        None

        """
        # Needs updating to handle the new budget list structure. Write out budgets for all kstpkper
        # if kstpkper is None or pass list of kstpkper/totim to save particular budgets.
        with open(fname, 'w') as f:
            # Write header
            f.write(','.join(self._budget_list[0].dtype.names) + '\n')
        with open(fname, 'a') as f:
            for bud in self._budget_list:

                for rowidx in range(bud.shape[0]):
                    s = ','.join([str(i) for i in list(bud[:][rowidx])])+'\n'
                    f.write(s)
        return

    def get_dataframes(self, start_datetime=None, timeunit='D', index_key='totim'):
        """
        Get pandas dataframes.
        Parameters
        ----------
        
        start_datetime : str
            Datetime string indicating the time at which the simulation starts.
        timeunit : str
            String that indicates the time units used in the model.
        index_key : str
            Indicates the fields to be used (in addition to "record") in the
            resulting DataFrame multi-index.

        Returns
        -------
        out : Pandas DataFrame
            Pandas DataFrame with the budget information.
        Examples
        --------
        >>> from flopy.utils.zonbud import ZoneBudget, read_zbarray
        >>> zon = read_zbarray('zone_input_file')
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> df = zb.get_dataframes()
        """
        try:
            import pandas as pd
        except Exception as e:
            raise Exception(
                "ZoneBudget.get_dataframes() error import pandas: " + \
                str(e))
        valid_index_keys = ['totim', 'kstpkper']
        assert index_key in valid_index_keys, 'index_key "{}" is not valid.'.format(index_key)

        valid_timeunit = ['S', 'M', 'H', 'D', 'Y']

        if timeunit.upper() == 'SECONDS':
            timeunit = 'S'
        elif timeunit.upper() == 'MINUTES':
            timeunit = 'M'
        elif timeunit.upper() == 'HOURS':
            timeunit = 'H'
        elif timeunit.upper() == 'DAYS':
            timeunit = 'D'
        elif timeunit.upper() == 'YEARS':
            timeunit = 'Y'

        errmsg = 'Specified time units ({}) not recognized. ' \
                 'Please use one of '.format(timeunit)
        assert timeunit in valid_timeunit, errmsg + ', '.join(valid_timeunit) + '.'

        out = pd.DataFrame()
        for bud in self.get_budget():
            out = out.append(pd.DataFrame(bud))
        if start_datetime is not None:
            totim = totim_to_datetime(out.totim,
                                      start=pd.to_datetime(start_datetime),
                                      timeunit=timeunit)
            out['datetime'] = totim
            index_cols = ['datetime', 'record']
        else:
            if index_key == 'totim':
                index_cols = ['totim', 'name']
            elif index_key == 'kstpkper':
                index_cols = ['time_step', 'stress_period', 'name']
        out = out.set_index(index_cols).sort_index()
        keep_cols = self._zonefieldnames
        return out[keep_cols]

    def copy(self):
        """
        Return a deepcopy of the object.
        """
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        """
        Over-rides the default deepcopy behavior. Copy all attributes except
        the CellBudgetFile object which does not copy nicely.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        ignore_attrs = ['cbc']
        for k, v in self.__dict__.items():
            if k not in ignore_attrs:
                setattr(result, k, copy.deepcopy(v, memo))

        # Set CellBudgetFile object attribute manually. This is object
        # read-only so should not be problems with pointers from
        # multiple objects.
        result.cbc = self.cbc
        return result

    def _compute_budget(self, kstpkper=None, totim=None):
        """
        Creates a budget for the specified zone array. This function only supports the
        use of a single time step/stress period or time.
        """
        # Initialize the budget record array
        recordarray = self._initialize_recordarray(kstpkper=kstpkper, totim=totim)

        # Create a throwaway list of all record names
        reclist = list(self.record_names)

        # Initialize an array to track where the constant head cells
        # are located.
        ich = np.zeros(self.cbc_shape, self.int_type)

        if 'CONSTANT HEAD' in reclist:
            reclist.remove('CONSTANT HEAD')
            chd = self.cbc.get_data(text='CONSTANT HEAD', full3D=True, kstpkper=kstpkper, totim=totim)[0]
            ich = np.zeros(self.cbc_shape, self.int_type)
            ich[chd != 0] = 1
        if 'FLOW RIGHT FACE' in reclist:
            reclist.remove('FLOW RIGHT FACE')
            recordarray = self._accumulate_flow_frf(recordarray, 'FLOW RIGHT FACE', ich, kstpkper, totim)
        if 'FLOW FRONT FACE' in reclist:
            reclist.remove('FLOW FRONT FACE')
            recordarray = self._accumulate_flow_fff(recordarray, 'FLOW FRONT FACE', ich, kstpkper, totim)
        if 'FLOW LOWER FACE' in reclist:
            reclist.remove('FLOW LOWER FACE')
            recordarray = self._accumulate_flow_flf(recordarray, 'FLOW LOWER FACE', ich, kstpkper, totim)
        if 'SWIADDTOCH' in reclist:
            reclist.remove('SWIADDTOCH')
            swichd = self.cbc.get_data(text='SWIADDTOCH', full3D=True, kstpkper=kstpkper, totim=totim)[0]
            swiich = np.zeros(self.cbc_shape, self.int_type)
            swiich[swichd != 0] = 1
        if 'SWIADDTOFRF' in reclist:
            reclist.remove('SWIADDTOFRF')
            recordarray = self._accumulate_flow_frf(recordarray, 'SWIADDTOFRF', swiich, kstpkper, totim)
        if 'SWIADDTOFFF' in reclist:
            reclist.remove('SWIADDTOFFF')
            recordarray = self._accumulate_flow_fff(recordarray, 'SWIADDTOFFF', swiich, kstpkper, totim)
        if 'SWIADDTOFLF' in reclist:
            reclist.remove('SWIADDTOFLF')
            recordarray = self._accumulate_flow_flf(recordarray, 'SWIADDTOFLF', swiich, kstpkper, totim)

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        # iterate over remaining items in the list
        for recname in reclist:
            imeth = self.imeth[recname]

            data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)
            if len(data) == 0:
                # Empty data, can occur during the first time step of a transient model when
                # storage terms are zero and not in the cell-budget file.
                continue
            else:
                data = data[0]

            if imeth == 2 or imeth == 5:
                # LIST
                qin = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
                qout = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
                for [node, q] in zip(data['node'], data['q']):
                    idx = node - 1
                    if q > 0:
                        qin.data[idx] += q
                    elif q < 0:
                        qout.data[idx] += q
                qin = np.ma.reshape(qin, (self.nlay, self.nrow, self.ncol))
                qout = np.ma.reshape(qout, (self.nlay, self.nrow, self.ncol))
            elif imeth == 0 or imeth == 1:
                # FULL 3-D ARRAY
                qin = np.ma.zeros(self.cbc_shape, self.float_type)
                qout = np.ma.zeros(self.cbc_shape, self.float_type)
                qin[data > 0] = data[data > 0]
                qout[data < 0] = data[data < 0]
            elif imeth == 3:
                # 1-LAYER ARRAY WITH LAYER INDICATOR ARRAY
                rlay, rdata = data[0], data[1]
                data = np.ma.zeros(self.cbc_shape, self.float_type)
                for (r, c), l in np.ndenumerate(rlay):
                    data[l - 1, r, c] = rdata[r, c]
                qin = np.ma.zeros(self.cbc_shape, self.float_type)
                qout = np.ma.zeros(self.cbc_shape, self.float_type)
                qin[data > 0] = data[data > 0]
                qout[data < 0] = data[data < 0]
            elif imeth == 4:
                # 1-LAYER ARRAY THAT DEFINES LAYER 1
                qin = np.ma.zeros(self.cbc_shape, self.float_type)
                qout = np.ma.zeros(self.cbc_shape, self.float_type)
                r, c = np.where(data > 0)
                qin[0, r, c] = data[r, c]
                r, c = np.where(data < 0)
                qout[0, r, c] = data[r, c]
            else:
                # Should not happen
                raise Exception('Unrecognized "imeth" for {} record: {}'.format(recname, imeth))
            recordarray = self._accumulate_flow_ssst(recordarray, recname, qin, qout)
        recordarray = self._compute_mass_balance(recordarray)
        return recordarray
    
    def _get_internal_flow_record_names(self):
        iflow_from_recnames = OrderedDict([])
        iflow_to_recnames = OrderedDict([])
        if 0 in self.allzones:
            iflow_from_recnames[0] = 'ZONE_0'
            iflow_to_recnames[0] = 'ZONE_0'
        for z, a in iter(self._zonefieldnamedict.items()):
            iflow_from_recnames[z] = '{}'.format(a)
            iflow_to_recnames[z] = '{}'.format(a)
        return iflow_from_recnames, iflow_to_recnames

    def _build_empty_record(self, recordarray, recname, kstpkper=None, totim=None):
        # Builds empty records based on the specified flow direction and
        # record name for the given list of zones.
        if kstpkper is not None:
            totim = self.cbc_times[self.cbc_kstpkper.index(kstpkper)]
        elif totim is not None:
            kstpkper = self.cbc_kstpkper[self.cbc_times.index(totim)]

        row = [totim, kstpkper[0], kstpkper[1], recname]
        row += [0. for _ in self._zonefieldnames]
        recs = np.array(tuple(row), dtype=recordarray.dtype)
        recordarray = np.append(recordarray, recs)
        return recordarray

    def _initialize_recordarray(self, kstpkper=None, totim=None):
        # Initialize the budget record array which will store all of the
        # fluxes in the cell-budget file.

        # Create empty array for the budget terms.
        dtype_list = [('totim', '<f4'), ('time_step', '<i4'), ('stress_period', '<i4'), ('name', (str, 50))]
        dtype_list += [(n, self.float_type) for n in self._zonefieldnames]
        dtype = np.dtype(dtype_list)
        recordarray = np.array([], dtype=dtype)

        # Add "in" records
        if 'STORAGE' in self.record_names:
            recordarray = self._build_empty_record(recordarray, 'STORAGE_IN', kstpkper, totim)
        if 'CONSTANT HEAD' in self.record_names:
            recordarray = self._build_empty_record(recordarray, 'CONSTANT_HEAD_IN', kstpkper, totim)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                recordarray = self._build_empty_record(recordarray, '_'.join(recname.split())+'_IN', kstpkper, totim)
        for n in self._iflow_from_recnames.values():
            recordarray = self._build_empty_record(recordarray, '_'.join(n.split())+'_IN', kstpkper, totim)
        recordarray = self._build_empty_record(recordarray, 'TOTAL_IN', kstpkper, totim)

        # Add "out" records
        if 'STORAGE' in self.record_names:
            recordarray = self._build_empty_record(recordarray, 'STORAGE_OUT', kstpkper, totim)
        if 'CONSTANT HEAD' in self.record_names:
            recordarray = self._build_empty_record(recordarray, 'CONSTANT_HEAD_OUT', kstpkper, totim)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                recordarray = self._build_empty_record(recordarray, '_'.join(recname.split())+'_OUT', kstpkper, totim)
        for n in self._iflow_to_recnames.values():
            recordarray = self._build_empty_record(recordarray, '_'.join(n.split())+'_OUT', kstpkper, totim)
        recordarray = self._build_empty_record(recordarray, 'TOTAL_OUT', kstpkper, totim)

        recordarray = self._build_empty_record(recordarray, 'IN-OUT', kstpkper, totim)
        recordarray = self._build_empty_record(recordarray, 'PERCENT_DISCREPANCY', kstpkper, totim)
        return recordarray

    def _update_record(self, recordarray, recname, colname, flux):
        # Update the budget record array with the flux for the specified
        # flow direction (in/out), record name, and column (exclusive of
        # ZONE 0).

        # Make sure the flux is between different zones
        a = ' '.join(recname.split('_')[1:])
        b = colname
        if a == b:
            errmsg = 'Circular flow detected: {}\t{}\t{}\t{}'.format(recname,
                                                                     colname,
                                                                     flux)
            raise Exception(errmsg)
        rowidx = np.where((recordarray['name'] == recname))
        recordarray[colname][rowidx] += flux
        return recordarray

    def _accumulate_flow_frf(self, recordarray, recname, ich, kstpkper, totim):
        """
        C
        C-----"FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
        C-----COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
        C-----ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        C-----1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K
        300   IF(NCOL.LT.2) RETURN
              DO 340 K=1,NLAY
              DO 340 I=1,NROW
              DO 340 J=2,NCOL
              NZ=IZONE(J,I,K)
              JL=J-1
              NZL=IZONE(JL,I,K)
              IF(NZL.LE.NZ) GO TO 340
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J-1,I,K).EQ.1) GO TO 340
              DBUFF=BUFFD(JL,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZL)=VBZNFL(2,NZ,NZL)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZL)=VBZNFL(1,NZ,NZL)+DBUFF
              END IF
          340 CONTINUE
        C
        C-----FLOW BETWEEN NODE J,I,K AND J+1,I,K
              DO 370 K=1,NLAY
              DO 370 I=1,NROW
              DO 370 J=1,NCOL-1
              NZ=IZONE(J,I,K)
              JR=J+1
              NZR=IZONE(JR,I,K)
              IF(NZR.LE.NZ) GO TO 370
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J+1,I,K).EQ.1) GO TO 370
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZR)=VBZNFL(1,NZ,NZR)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZR)=VBZNFL(2,NZ,NZR)+DBUFF
              END IF
          370 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 395 K=1,NLAY
              DO 395 I=1,NROW
              DO 395 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 395
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 395
              IF(J.EQ.NCOL) GO TO 380
              IF(ICH(J+1,I,K).EQ.1) GO TO 380
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        380   IF(J.EQ.1) GO TO 395
              IF(ICH(J-1,I,K).EQ.1) GO TO 395
              DBUFF=BUFFD(J-1,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        395   CONTINUE
              RETURN
        """
        if self.ncol >= 2:
            data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)[0]

            # "FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
            # COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
            # ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
            # 1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K

            l, r, c = np.where(self.izone[:, :, 1:] > self.izone[:, :, :-1])

            # Adjust column values to account for the starting position of "nz"
            c = np.copy(c) + 1

            # Define the zone from which flow is coming
            cl = c-1
            nzl = self.izone[l, r, cl]

            # Define the zone to which flow is going
            nz = self.izone[l, r, c]

            # Get the face flow
            q = data[l, r, cl]

            # Get indices where flow face values are positive (flow out of higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nzl[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # Get indices where flow face values are negative (flow into higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzl[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # FLOW BETWEEN NODE J,I,K AND J+1,I,K
            l, r, c = np.where(self.izone[:, :, :-1] > self.izone[:, :, 1:])

            # Define the zone from which flow is coming
            nz = self.izone[l, r, c]

            # Define the zone to which flow is going
            cr = c+1
            nzr = self.izone[l, r, cr]

            # Get the face flow
            q = data[l, r, c]

            # Get indices where flow face values are positive (flow out of higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzr[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # Get indices where flow face values are negative (flow into higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nzr[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[c > 0], r[c > 0], c[c > 0]
            cl = c - 1
            nzl = self.izone[l, r, cl]
            nz = self.izone[l, r, c]
            q = data[l, r, cl]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nzl[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzl[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[c < self.ncol-1], r[c < self.ncol-1], c[c < self.ncol-1]
            nz = self.izone[l, r, c]
            cr = c+1
            nzr = self.izone[l, r, cr]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzr[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nzr[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
        return recordarray

    def _accumulate_flow_fff(self, recordarray, recname, ich, kstpkper, totim):
        """
        C
        C-----"FLOW FRONT FACE"
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
        400   IF(NROW.LT.2) RETURN
              DO 440 K=1,NLAY
              DO 440 I=2,NROW
              DO 440 J=1,NCOL
              NZ=IZONE(J,I,K)
              IA=I-1
              NZA=IZONE(J,IA,K)
              IF(NZA.LE.NZ) GO TO 440
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I-1,K).EQ.1) GO TO 440
              DBUFF=BUFFD(J,IA,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZA)=VBZNFL(2,NZ,NZA)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZA)=VBZNFL(1,NZ,NZA)+DBUFF
              END IF
          440 CONTINUE
        C
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K
              DO 470 K=1,NLAY
              DO 470 I=1,NROW-1
              DO 470 J=1,NCOL
              NZ=IZONE(J,I,K)
              IB=I+1
              NZB=IZONE(J,IB,K)
              IF(NZB.LE.NZ) GO TO 470
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I+1,K).EQ.1) GO TO 470
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZB)=VBZNFL(1,NZ,NZB)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZB)=VBZNFL(2,NZ,NZB)+DBUFF
              END IF
          470 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 495 K=1,NLAY
              DO 495 I=1,NROW
              DO 495 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 495
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 495
              IF(I.EQ.NROW) GO TO 480
              IF(ICH(J,I+1,K).EQ.1) GO TO 480
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        480   IF(I.EQ.1) GO TO 495
              IF(ICH(J,I-1,K).EQ.1) GO TO 495
              DBUFF=BUFFD(J,I-1,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        495   CONTINUE
              RETURN
        """
        if self.nrow >= 2:
            data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)[0]

            # "FLOW FRONT FACE"
            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
            l, r, c = np.where(self.izone[:, 1:, :] < self.izone[:, :-1, :])
            r = np.copy(r) + 1
            ra = r - 1
            nza = self.izone[l, ra, c]
            nz = self.izone[l, r, c]
            q = data[l, ra, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
            l, r, c = np.where(self.izone[:, :-1, :] < self.izone[:, 1:, :])
            nz = self.izone[l, r, c]
            rb = r + 1
            nzb = self.izone[l, rb, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[r > 0], r[r > 0], c[r > 0]
            ra = r-1
            nza = self.izone[l, ra, c]
            nz = self.izone[l, r, c]
            q = data[l, ra, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[r < self.nrow-1], r[r < self.nrow-1], c[r < self.nrow-1]
            nz = self.izone[l, r, c]
            rb = r+1
            nzb = self.izone[l, rb, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray, self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
        return recordarray

    def _accumulate_flow_flf(self, recordarray, recname, ich, kstpkper, totim):
        """
        C
        C-----"FLOW LOWER FACE"
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
        500   IF(NLAY.LT.2) RETURN
              DO 540 K=2,NLAY
              DO 540 I=1,NROW
              DO 540 J=1,NCOL
              NZ=IZONE(J,I,K)
              KA=K-1
              NZA=IZONE(J,I,KA)
              IF(NZA.LE.NZ) GO TO 540
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I,K-1).EQ.1) GO TO 540
              DBUFF=BUFFD(J,I,KA)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZA)=VBZNFL(2,NZ,NZA)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZA)=VBZNFL(1,NZ,NZA)+DBUFF
              END IF
          540 CONTINUE
        C
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
              DO 570 K=1,NLAY-1
              DO 570 I=1,NROW
              DO 570 J=1,NCOL
              NZ=IZONE(J,I,K)
              KB=K+1
              NZB=IZONE(J,I,KB)
              IF(NZB.LE.NZ) GO TO 570
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I,K+1).EQ.1) GO TO 570
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZB)=VBZNFL(1,NZ,NZB)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZB)=VBZNFL(2,NZ,NZB)+DBUFF
              END IF
          570 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 595 K=1,NLAY
              DO 595 I=1,NROW
              DO 595 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 595
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 595
              IF(K.EQ.NLAY) GO TO 580
              IF(ICH(J,I,K+1).EQ.1) GO TO 580
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        580   IF(K.EQ.1) GO TO 595
              IF(ICH(J,I,K-1).EQ.1) GO TO 595
              DBUFF=BUFFD(J,I,K-1)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        595   CONTINUE
              RETURN
        """
        if self.nlay >= 2:
            data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)[0]

            # "FLOW LOWER FACE"
            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
            l, r, c = np.where(self.izone[1:, :, :] < self.izone[:-1, :, :])
            l = np.copy(l) + 1
            la = l - 1
            nza = self.izone[la, r, c]
            nz = self.izone[l, r, c]
            q = data[la, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
            l, r, c = np.where(self.izone[:-1, :, :] < self.izone[1:, :, :])
            nz = self.izone[l, r, c]
            lb = l + 1
            nzb = self.izone[lb, r, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_from_recnames[fz]+'_IN',
                                        self._zonefieldnamedict[tz], flux)
                if fz != 0:
                    recordarray = self._update_record(recordarray, self._iflow_to_recnames[tz]+'_OUT',
                                        self._zonefieldnamedict[fz], flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[l > 0], r[l > 0], c[l > 0]
            la = l - 1
            nza = self.izone[la, r, c]
            nz = self.izone[l, r, c]
            q = data[la, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[l < self.nlay - 1], r[l < self.nlay - 1], c[l < self.nlay - 1]
            nz = self.izone[l, r, c]
            lb = l + 1
            nzb = self.izone[lb, r, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if tz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_OUT', self._zonefieldnamedict[tz], flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                if fz != 0:
                    recordarray = self._update_record(recordarray, 'CONSTANT_HEAD_IN', self._zonefieldnamedict[fz], flux)
        return recordarray

    def _accumulate_flow_ssst(self, recordarray, recname, qin, qout):

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        for z in self.allzones:
            if z != 0:
                flux = np.abs(qin[(self.izone == z)].sum())
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.
                recordarray = self._update_record(recordarray,
                                                  '_'.join(recname.split())+'_IN',
                                                  self._zonefieldnamedict[z],
                                                  flux)

                flux = np.abs(qout[(self.izone == z)].sum())
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.
                recordarray = self._update_record(recordarray,
                                                  '_'.join(recname.split())+'_OUT',
                                                  self._zonefieldnamedict[z],
                                                  flux)
        return recordarray

    def _compute_mass_balance(self, recordarray):
        # Returns a record array with total inflow, total outflow,
        # and percent error summed by column.
        skipcols = ['time_step', 'stress_period', 'totim', 'name']

        # Compute inflows
        names = np.array([n for n in recordarray['name'] if '_IN' in n])
        idx = np.in1d(recordarray['name'], names)
        a = _numpyvoid2numeric(recordarray[self._zonefieldnames][idx])
        intot = np.array(a.sum(axis=0))
        for idx, colname in enumerate([n for n in recordarray.dtype.names if n not in skipcols]):
            flux = intot[idx]
            recordarray = self._update_record(recordarray, 'TOTAL_IN', colname, flux)

        # Compute outflows
        names = np.array([n for n in recordarray['name'] if '_OUT' in n])
        idx = np.in1d(recordarray['name'], names)
        a = _numpyvoid2numeric(recordarray[self._zonefieldnames][idx])
        outot = np.array(a.sum(axis=0))
        for idx, colname in enumerate([n for n in recordarray.dtype.names if n not in skipcols]):
            flux = outot[idx]
            recordarray = self._update_record(recordarray, 'TOTAL_OUT', colname, flux)

        # Compute in-out
        for idx, colname in enumerate([n for n in recordarray.dtype.names if n not in skipcols]):
            flux = intot[idx]-outot[idx]
            recordarray = self._update_record(recordarray, 'IN-OUT', colname, flux)

        # Compute percent discrepancy
        for idx, colname in enumerate([n for n in recordarray.dtype.names if n not in skipcols]):
            in_minus_out = intot[idx]-outot[idx]
            in_plus_out = intot[idx]+outot[idx]
            flux = 100 * in_minus_out / (in_plus_out / 2.)
            recordarray = self._update_record(recordarray, 'PERCENT_DISCREPANCY', colname, flux)

        return recordarray

    def __mul__(self, other):
        newbuds = list(self.get_budget())
        for idx, bud in enumerate(newbuds):
            for f in self._zonefieldnames:
                a = np.array([r for r in bud[f]]) * other
                bud[f] = a
            newbuds[idx] = bud
        newobj = self.copy()
        newobj._budget_list = newbuds
        return newobj

    def __truediv__(self, other):
        newbuds = list(self.get_budget())
        for idx, bud in enumerate(newbuds):
            for f in self._zonefieldnames:
                a = np.array([r for r in bud[f]]) / float(other)
                bud[f] = a
            newbuds[idx] = bud
        newobj = self.copy()
        newobj._budget_list = newbuds
        return newobj

    def __div__(self, other):
        newbuds = list(self.get_budget())
        for idx, bud in enumerate(newbuds):
            for f in self._zonefieldnames:
                a = np.array([r for r in bud[f]]) / float(other)
                bud[f] = a
            newbuds[idx] = bud
        newobj = self.copy()
        newobj._budget_list = newbuds
        return newobj

    def __add__(self, other):
        newbuds = list(self.get_budget())
        for idx, bud in enumerate(newbuds):
            for f in self._zonefieldnames:
                a = np.array([r for r in bud[f]]) + other
                bud[f] = a
            newbuds[idx] = bud
        newobj = self.copy()
        newobj._budget_list = newbuds
        return newobj

    def __sub__(self, other):
        newbuds = list(self.get_budget())
        for idx, bud in enumerate(newbuds):
            for f in self._zonefieldnames:
                a = np.array([r for r in bud[f]]) - other
                bud[f] = a
            newbuds[idx] = bud
        newobj = self.copy()
        newobj._budget_list = newbuds
        return newobj

    
def _numpyvoid2numeric(a):
    # The budget record array has multiple dtypes and a slice returns
    # the flexible-type numpy.void which must be converted to a numeric
    # type prior to performing reducing functions such as sum() or
    # mean()
    return np.array([list(r) for r in a])


def write_zbarray(fname, X, fmtin=None, iprn=None):
    """
    Saves a numpy array in a format readable by the zonebudget program executable.

    File format:
    line 1: nlay, nrow, ncol
    line 2: INTERNAL (format)
    line 3: begin data
    .
    .
    .

    example from NACP:
    19 250 500
    INTERNAL      (10I8)
         199     199     199     199     199     199     199     199     199     199
         199     199     199     199     199     199     199     199     199     199
         ...
    INTERNAL      (10I8)
         199     199     199     199     199     199     199     199     199     199
         199     199     199     199     199     199     199     199     199     199
         ...

    Parameters
    ----------
    X : array
        The array of zones to be written.
    fname :  str
        The path and name of the file to be written.
    width : int
        The number of values to write to each line.

    Returns
    -------

    """
    if len(X.shape) == 2:
        b = np.zeros((1, X.shape[0], X.shape[1]), dtype=np.int32)
        b[0, :, :] = X[:, :]
        X = b.copy()
    elif len(X.shape) < 2 or len(X.shape) > 3:
        raise Exception('Shape of the input array is not recognized: {}'.format(X.shape))

    nlay, nrow, ncol = X.shape

    if fmtin is not None:
        assert fmtin < ncol, 'The specified width is greater than the ' \
                             'number of columns in the array.'
    else:
        fmtin = ncol

    iprnmin = len(str(X.max()))
    if iprn is None or iprn <= iprnmin:
        iprn = iprnmin + 1

    formatter_str = '{{:>{iprn}}}'.format(iprn=iprn)
    formatter = formatter_str.format

    with open(fname, 'w') as f:
        header = '{nlay} {nrow} {ncol}\n'.format(nlay=nlay,
                                                 nrow=nrow,
                                                 ncol=ncol)
        f.write(header)
        for lay in range(nlay):
            record_2 = 'INTERNAL\t({fmtin}I{iprn})\n'.format(fmtin=fmtin, iprn=iprn)
            f.write(record_2)
            if fmtin < ncol:
                for row in range(nrow):
                    rowvals = X[lay, row, :].ravel()
                    start = 0
                    end = start + fmtin
                    vals = rowvals[start:end]
                    while len(vals) > 0:
                        s = ''.join([formatter(int(val)) for val in vals]) + '\n'
                        f.write(s)
                        start = end
                        end = start + fmtin
                        vals = rowvals[start:end]
                    # vals = rowvals[start:end]
                    # if len(vals) > 0:
                    #     s = ''.join([formatter(int(val)) for val in vals]) + '\n'
                    #     f.write(s)
            elif fmtin == ncol:
                for row in range(nrow):
                    vals = X[lay, row, :].ravel()
                    f.write(''.join([formatter(int(val)) for val in vals]) + '\n')
    return


def read_zbarray(fname):

    with open(fname, 'r') as f:
        lines = f.readlines()

    nlay, nrow, ncol = [int(v) for v in lines[0].strip().split()]
    zones = np.zeros((nlay, nrow, ncol), dtype=np.int64)

    # Initialize layer
    lay = 0

    # The number of values to read before placing
    # them into the zone array
    datalen = nrow * ncol

    # List of valid values for LOCAT
    locats = ['CONSTANT', 'INTERNAL', 'EXTERNAL']

    # ITERATE OVER THE ROWS
    for row in lines[1:]:
        rowitems = row.strip().split()

        # HEADER
        if rowitems[0] in locats:
            vals = []
            locat = rowitems[0]

            if locat == 'CONSTANT':
                iconst = int(rowitems[1])
            else:
                fmt = rowitems[1].strip('()')
                fmtin, iprn = [int(v) for v in fmt.split('I')]

        # ZONE DATA
        else:
            if locat == 'CONSTANT':
                zones[lay, :, :] = iconst
                lay += 1
            elif locat == 'INTERNAL':
                # READ ZONES
                rowvals = [int(v) for v in rowitems]
                vals.extend(rowvals)
                if len(vals) == datalen:
                    # place values for the previous layer into the zone array
                    vals = np.array(vals, dtype=np.int32).reshape((nrow, ncol))
                    zones[lay, :, :] = vals[:, :]
                    lay += 1
            elif locat == 'EXTERNAL':
                # READ EXTERNAL FILE
                fname = rowitems[0]
                if not os.path.isfile(fname):
                    errmsg = 'Could not find external file "{}"'.format(fname)
                    raise Exception(errmsg)
                with open(fname, 'r') as ext_f:
                    ext_flines = ext_f.readlines()
                for ext_frow in ext_flines:
                    ext_frowitems = ext_frow.strip().split()
                    rowvals = [int(v) for v in ext_frowitems]
                    vals.extend(rowvals)
                if len(vals) != datalen:
                    errmsg = 'The number of values read from external ' \
                             'file "{}" does not match the expected ' \
                             'number.'.format(len(vals))
                    raise Exception(errmsg)
                vals = np.array(vals, dtype=np.int32).reshape((nrow, ncol))
                zones[lay, :, :] = vals[:, :]
                lay += 1
            else:
                raise Exception('Locat not recognized: {}'.format(locat))

        # IGNORE COMPOSITE ZONES
    return zones


def sum_flux_tuples(fromzones, tozones, fluxes):
    tup = zip(fromzones, tozones, fluxes)
    sorted_tups = sort_tuple(tup)

    # Group the sorted tuples by (from zone, to zone)
    # itertools.groupby() returns the index (from zone, to zone) and
    # a list of the tuples with that index
    fluxes = []
    for (fz, tz), ftup in groupby(sorted_tups, lambda tup: tup[:2]):
        f = np.sum([tup[-1] for tup in list(ftup)])
        fluxes.append((fz, tz, f))
    return fluxes


def sort_tuple(tup, n=2):
    """
    Sort a tuple by the first n values
    :param tup:
    :param n:
    :return:
    """
    return tuple(sorted(tup, key=lambda t: t[:n]))