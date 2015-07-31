import sys
import numpy as np
import struct as strct
import string


class SwrBinaryStatements:
    integer = np.int32
    real = np.float64
    character = np.uint8
    integerbyte = 4
    realbyte = 8
    textbyte = 4

    def read_integer(self):
        intvalue = strct.unpack('i', self.file.read(1 * SwrBinaryStatements.integerbyte))[0]
        return intvalue

    def read_real(self):
        realvalue = strct.unpack('d', self.file.read(1 * SwrBinaryStatements.realbyte))[0]
        return realvalue

    def read_text(self):
        textvalue = np.fromfile(file=self.file, dtype=SwrBinaryStatements.character, count=16).tostring()
        return textvalue

    def read_obs_text(self, nchar=20):
        textvalue = np.fromfile(file=self.file, dtype=SwrBinaryStatements.character, count=nchar).tostring()
        return textvalue

    def read_record(self):
        if self.skip == True:
            lpos = self.file.tell() + ( self.nrecord * self.items * SwrBinaryStatements.realbyte )
            self.file.seek(lpos)
            x = np.zeros((self.nrecord * self.items), SwrBinaryStatements.real)
        else:
            x = np.fromfile(file=self.file, dtype=SwrBinaryStatements.real, count=self.nrecord * self.items)
        x.resize(self.nrecord, self.items)
        return x

    def read_items(self):
        if self.skip == True:
            lpos = self.file.tell() + ( self.items * SwrBinaryStatements.realbyte )
            self.file.seek(lpos)
            x = np.zeros((self.items), SwrBinaryStatements.real)
        else:
            x = np.fromfile(file=self.file, dtype=SwrBinaryStatements.real, count=self.items)
        return x

    def read_1dintegerarray(self):
        i = np.fromfile(file=self.file, dtype=SwrBinaryStatements.integer, count=self.nrecord)
        return i


class SwrObs(SwrBinaryStatements):
    'Reads binary SWR observations output from MODFLOW SWR Process binary observation files'

    def __init__(self, filename, verbose=False):
        # initialize class information
        self.skip = False
        self.verbose = verbose
        # open binary head file
        self.file = open(filename, 'rb')
        # NOBS
        self.nobs = self.read_integer()
        self.v = np.empty((self.nobs), dtype='float')
        self.v.fill(1.0E+32)
        # read obsnames
        obsnames = []
        for idx in range(0, self.nobs):
            cid = self.read_obs_text()
            obsnames.append(cid)
        self.obsnames = np.array(obsnames)
        # set position
        self.datastart = self.file.tell()
        #get times
        self.times = self.time_list()

    def get_time_list(self):
        return self.times

    def get_num_items(self):
        return self.nobs

    def get_obs_labels(self):
        return self.obsnames

    def rewind_file(self):
        self.file.seek(self.datastart)
        return True

    def time_list(self):
        self.skip = True
        self.file.seek(self.datastart)
        times = []
        while True:
            current_position = self.file.tell()
            totim, v, success = next(self)
            if success == True:
                times.append([totim, current_position])
            else:
                self.file.seek(self.datastart)
                times = np.array(times)
                self.skip = False
                return times


    def __iter__(self):
        return self

    def read_header(self):
        try:
            totim = self.read_real()
            return totim, True
        except:
            return -999., False

    def __next__(self):
        totim, success = self.read_header()
        if (success):
            for idx in range(0, self.nobs):
                self.v[idx] = self.read_real()
        else:
            if self.verbose == True:
                print('_BinaryObs object.next() reached end of file.')
            self.v.fill(1.0E+32)
        return totim, self.v, success

    def get_values(self, idx):
        iposition = int(self.times[idx, 1])
        self.file.seek(iposition)
        totim, v, success = next(self)
        if success == True:
            return totim, v, True
        else:
            self.v.fill(1.0E+32)
            return 0.0, self.v, False

    def get_time_gage(self, record):
        idx = -1
        try:
            idx = int(record) - 1
            if self.verbose == True:
                print('retrieving SWR observation record [{0}]'.format(idx + 1))
        except:
            for icnt, cid in enumerate(self.obsnames):
                if record.strip().lower() == cid.strip().lower():
                    idx = icnt
                    if self.verbose == True:
                        print('retrieving SWR observation record [{0}] {1}'.format(idx + 1, record.strip().lower()))
                    break
        gage_record = np.zeros((2))  # tottime plus observation
        if idx != -1 and idx < self.nobs:
            # --find offset to position
            ilen = self.get_point_offset(idx)
            # get data
            for time_data in self.times:
                self.file.seek(int(time_data[1]) + ilen)
                v = self.read_real()
                this_entry = np.array([float(time_data[0])])
                this_entry = np.hstack((this_entry, v))
                gage_record = np.vstack((gage_record, this_entry))
            #delete the first 'zeros' element
            gage_record = np.delete(gage_record, 0, axis=0)
        return gage_record

    def get_point_offset(self, ipos):
        self.file.seek(self.datastart)
        lpos0 = self.file.tell()
        point_offset = int(0)
        totim, success = self.read_header()
        idx = (ipos)
        lpos1 = self.file.tell() + idx * SwrBinaryStatements.realbyte
        self.file.seek(lpos1)
        point_offset = self.file.tell() - lpos0
        return point_offset


class SwrFile(SwrBinaryStatements):
    def __init__(self, swrtype, filename, verbose=False):
        # --swrtype =  0 = stage record
        # swrtype = -1 = reach group record
        # swrtype = -2 = reach group connection velocity record
        # swrtype >  0 = aq-reach exchange record type = nlay
        self.file = open(filename, 'rb')
        self.type = None
        try:
            ctype = swrtype.lower()
        except:
            ctype = None
            pass
        if ctype is not None:
            self.type = ctype
        else:
            try:
                itype = int(swrtype)
            except:
                print('SWR data type not defined')
                raise
            if itype == 0:
                self.type = 'stage'
            elif itype == -1:
                self.type = 'reachgroup'
            elif itype == -2:
                self.type = 'qm'
            elif itype > 0:
                self.type = 'qaq'
        if self.type is None:
            print('undefined SWR data type')
            raise

        self.verbose = verbose
        self.nrgout = 0
        if self.type == 'qm':
            self.nrgout = self.read_integer()
        self.nrecord = self.read_integer()
        self.items = self.get_num_items()
        self.null_record = np.zeros((self.nrecord, self.items)) + 1.0E+32
        #
        self.missingData = -9999.9
        self.dataAvailable = True
        self.skip = False
        #read connectivity for velocity data if necessary
        if self.type == 'qm':
            self.connectivity = self.read_connectivity()
            if self.verbose == True:
                print(self.connectivity)
        # initialize reachlayers and nqaqentries for qaq data
        if self.type == 'qaq':
            self.reachlayers = np.zeros((self.nrecord), np.int)
            self.nqaqentries = 0
        self.qaq_dtype = np.dtype([('layer', 'i4'),
                                   ('bottom', 'f8'), ('stage', 'f8'),
                                   ('depth', 'f8'), ('head', 'f8'),
                                   ('wetper', 'f8'), ('cond', 'f8'),
                                   ('headdiff', 'f8'), ('qaq', 'f8')])

        self.datastart = self.file.tell()
        #get times
        self.times = self.time_list()

    def get_nrecords(self):
        return self.nrgout, self.nrecord

    def get_time_list(self):
        return self.times

    def read_connectivity(self):
        conn = np.zeros((self.nrecord, 3), np.int)
        icount = 0
        for nrg in range(0, self.nrgout):
            nconn = self.read_integer()
            for ic in range(0, nconn):
                conn[icount, 0] = nrg
                conn[icount, 1] = self.read_integer()
                conn[icount, 2] = self.read_integer()
                icount += 1
        return conn

    def get_connectivity(self):
        if self.type == 'qm':
            return self.connectivity
        else:
            return None

    def get_num_items(self):
        if self.type == 'stage':
            return 1
        elif self.type == 'reachgroup':
            return 14
        elif self.type == 'qm':
            return 2
        elif self.type == 'qaq':
            return 10
        else:
            return -1

    def get_header_items(self):
        return ['totim', 'dt', 'kper', 'kstp', 'swrstp', 'success_flag']

    def get_item_list(self):
        if self.type == 'stage':
            list = ['stage']
        if self.type == 'reachgroup':
            list = ['stage', 'qsflow', 'qlatflow', 'quzflow', 'rain', 'evap',
                    'qbflow', 'qeflow', 'qexflow', 'qbcflow', 'qcrflow', 'dv', 'inf-out', 'volume']
        if self.type == 'qm':
            list = ['flow', 'velocity']
        if self.type == 'qaq':
            list = ['reach', 'layer', 'bottom', 'stage', 'depth', 'head',
                    'wetper', 'cond', 'headdiff', 'qaq']
        return list

    def get_temporal_list(self):
        list = ['totim', 'dt', 'kper', 'kstp', 'swrstp', 'success']
        return list

    def get_item_number(self, value, isTimeSeriesOutput=True):
        l = self.get_item_list()
        ioff = 6
        if isTimeSeriesOutput == False:
            ioff = 0
        try:
            i = l.index(value.lower())
            i += ioff
        except ValueError:
            l = self.get_temporal_list()
            try:
                i = l.index(value.lower())
            except ValueError:
                i = -1  # -no match
                print('no match to: ', value.lower())
        return i

    def return_gage_item_from_list(self, r, citem, scale=1.0):
        ipos = self.get_item_number(citem)
        n = r.shape[0]
        if n < 1:
            return self.null_record
        v = np.zeros((n), np.float)
        for i in range(0, n):
            v[i] = r[i, ipos] * scale
        return v

    def read_header(self):
        if self.type == 'qaq':
            try:
                self.nqaqentries = 0
                for i in range(0, self.nrecord):
                    self.reachlayers[i] = self.read_integer()
                    self.nqaqentries += self.reachlayers[i]
                    # print i+1, self.reachlayers[i]
                    #print self.nqaqentries
            except:
                if self.verbose == True:
                    sys.stdout.write('\nCould not read reachlayers')
                return 0.0, 0.0, 0, 0, 0, False
        try:
            totim = self.read_real()
            dt = self.read_real()
            kper = self.read_integer()
            kstp = self.read_integer()
            swrstp = self.read_integer()
            return totim, dt, kper, kstp, swrstp, True
        except:
            return 0.0, 0.0, 0, 0, 0, False

    def get_record(self, *args):
        # --pass a tuple of timestep,stress period
        try:
            kkspt = args[0]
            kkper = args[1]
            while True:
                totim, dt, kper, kstp, swrstp, success, r = next(self)
                if success == True:
                    if kkspt == kstp and kkper == kper:
                        if self.verbose == True:
                            print(totim, dt, kper, kstp, swrstp, True)
                        return totim, dt, kper, kstp, swrstp, True, r
                else:
                    return 0.0, 0.0, 0, 0, 0, False, self.null_record
        except:
            # pass a scalar of target totim -
            # returns either a match or the first
            # record that exceeds target totim
            try:
                ttotim = float(args[0])
                while True:
                    totim, dt, kper, kstp, swrstp, r, success = next(self)
                    if success == True:
                        if ttotim <= totim:
                            return totim, dt, kper, kstp, swrstp, True, r
                    else:
                        return 0.0, 0.0, 0, 0, 0, False, self.null_record
            except:
                # get the last successful record
                previous = next(self)
                while True:
                    this_record = next(self)
                    if this_record[-2] == False:
                        return previous
                    else:
                        previous = this_record

    def get_gage(self, rec_num=0, iconn=0, rec_lay=1):
        if self.type == 'qaq':
            gage_record = np.zeros((self.items + 6))  # items plus 6 header values, reach number, and layer value
        else:
            gage_record = np.zeros((self.items + 6))  # items plus 6 header values
        while True:
            totim, dt, kper, kstp, swrstp, success, r = next(self)
            if success == True:
                this_entry = np.array([totim, dt, kper, kstp, swrstp, success])
                irec = rec_num - 1
                # find correct entry for record and layer
                if self.type == 'qaq':
                    ifound = 0
                    ilay = rec_lay
                    ilen = np.shape(r)[0]
                    #print np.shape(r)
                    for i in range(0, ilen):
                        ir = int(r[i, 0])
                        il = int(r[i, 1])
                        if ir == rec_num and il == ilay:
                            ifound = 1
                            irec = i
                            break
                    if ifound < 1:
                        r[irec, :] = 0.0
                elif self.type == 'qm':
                    ifound = 0
                    for i in range(0, self.nrecord):
                        inode = self.connectivity[i, 1]
                        ic = self.connectivity[i, 2]
                        if rec_num == inode and ic == iconn:
                            ifound = 1
                            irec = i
                            break
                    if ifound < 1:
                        r[irec, :] = 0.0

                this_entry = np.hstack((this_entry, r[irec]))
                gage_record = np.vstack((gage_record, this_entry))

            else:
                gage_record = np.delete(gage_record, 0, axis=0)  # delete the first 'zeros' element
                return gage_record

    def __next__(self):
        totim, dt, kper, kstp, swrstp, success = self.read_header()
        if success == False:
            if self.verbose == True:
                print('SWR_Record.next() object reached end of file')
            return 0.0, 0.0, 0, 0, 0, False, self.null_record
        else:
            if self.type == 'qaq':
                r = self.read_qaq()
                return totim, dt, kper, kstp, swrstp, True, r
            else:
                r = self.read_record()
        return totim, dt, kper, kstp, swrstp, True, r

    def read_qaq(self):
        x = np.zeros((self.nqaqentries, self.items), SwrBinaryStatements.real)
        if self.skip == True:
            bytes = self.nqaqentries * (SwrBinaryStatements.integerbyte + 8 * SwrBinaryStatements.realbyte)
            lpos = self.file.tell() + ( bytes )
            self.file.seek(lpos)
        else:
            qaq_list = self.get_item_list()
            bd = np.fromfile(self.file, dtype=self.qaq_dtype, count=self.nqaqentries)
            ientry = 0
            for irch in range(self.nrecord):
                klay = self.reachlayers[irch]
                for k in range(klay):
                    x[ientry, 0] = irch + 1
                    ientry += 1
            for idx, k in enumerate(qaq_list[1:]):
                x[:, idx + 1] = bd[k]
        # print 'shape x: {}'.format(x.shape)
        return x


    def rewind_file(self):
        self.file.seek(self.datastart)
        return True

    def time_list(self):
        self.skip = True
        self.file.seek(self.datastart)
        idx = 0
        sys.stdout.write('Generating SWR binary data time list\n')
        times = []
        while True:
            # --output something to screen so it is possible to determine
            #  that the time list is being created
            idx += 1
            v = divmod(float(idx), 100.)
            if v[1] == 0.0:
                sys.stdout.write('.')
            # get current position
            current_position = self.file.tell()
            totim, dt, kper, kstp, swrstp, success, r = next(self)
            if success == True:
                times.append([totim, dt, kper, kstp, swrstp, current_position])
            else:
                self.file.seek(self.datastart)
                times = np.array(times)
                self.skip = False
                sys.stdout.write('\n')
                return times

    def get_time_record(self, time_index=0):
        self.file.seek(int(self.times[time_index][5]))
        totim, dt, kper, kstp, swrstp, success, r = next(self)
        if success == True:
            if self.verbose == True:
                print(totim, dt, kper, kstp, swrstp, True)
            return totim, dt, kper, kstp, swrstp, True, r
        else:
            return 0.0, 0.0, 0, 0, 0, False, self.null_record

    def get_point_offset(self, rec_num, iconn):
        self.file.seek(self.datastart)
        lpos0 = self.file.tell()
        point_offset = int(0)
        totim, dt, kper, kstp, swrstp, success = self.read_header()
        # --qaq terms
        if self.type == 'qaq':
            sys.stdout.write('MFBinaryClass::get_point_offset can not be used to extract QAQ data')
            sys.exit(1)
        # stage and reach group terms
        elif self.type == 'stage' or self.type == 'reachgroup':
            idx = (rec_num - 1) * self.items
            lpos1 = self.file.tell() + idx * SwrBinaryStatements.realbyte
            self.file.seek(lpos1)
            point_offset = self.file.tell() - lpos0
        # connection flux and velocity terms
        elif self.type == 'qm':
            frec = -999
            for i in range(0, self.nrecord):
                inode = self.connectivity[i, 1]
                ic = self.connectivity[i, 2]
                if rec_num == inode and ic == iconn:
                    frec = i
                    break
            if frec == -999:
                self.dataAvailable = False
            else:
                self.dataAvailable = True
                idx = (frec) * self.items
                lpos1 = self.file.tell() + idx * SwrBinaryStatements.realbyte
                self.file.seek(lpos1)
                point_offset = self.file.tell() - lpos0
        return point_offset

    def get_time_gage(self, rec_num=0, iconn=0):
        if self.type == 'qaq':
            sys.stdout.write('MFBinaryClass::get_time_gage can not be used to extract QAQ data\n')
            sys.exit(1)
        num_records = self.items + 6  # items plus 6 header values
        gage_record = np.zeros((num_records), np.float)
        # --find offset to position
        ilen = int(0)
        if rec_num > 0:
            ilen = self.get_point_offset(rec_num, iconn)
        else:
            self.dataAvailable = False
        if self.dataAvailable == False:
            sys.stdout.write('  Error: data is not available for reach {0} '.format(rec_num))
            if self.type == 'qm':
                sys.stdout.write('connected to reach {0}'.format(iconn))
            sys.stdout.write('\n')
        # get data
        if len(self.times) > 0:
            for time_data in self.times:
                totim = time_data[0]
                dt = time_data[1]
                kper = time_data[2]
                kstp = time_data[3]
                swrstp = time_data[4]
                success = True
                # get data
                if self.dataAvailable == True:
                    self.file.seek(int(time_data[5]) + ilen)
                    r = self.read_items()
                else:
                    r = np.empty((self.items), np.float)
                    r.fill(self.missingData)
                # push the data to the data structure
                this_entry = np.array([totim, dt, kper, kstp, swrstp, success])
                # update this_entry and current gage_record
                this_entry = np.hstack((this_entry, r))
                gage_record = np.vstack((gage_record, this_entry))
        # delete first empty entry and return gage_record
        gage_record = np.delete(gage_record, 0, axis=0)  #delete the first 'zeros' element
        return gage_record
