"""
This is a class for reading the mass budget from a (multi-component) mt3d(usgs) run.
Support SFT budget also

"""

import collections
import os
import re
import sys
from datetime import timedelta
import numpy as np

from ..utils.utils_def import totim_to_datetime

import pandas as pd

class MtListBudget(object):
    """
    MT3D mass budget reader

    Parameters
    ----------
    file_name : str
        the list file name
    timeunit : str
        the time unit to return in the recarray. (default is 'days')


    Examples
    --------
    >>> mt_list = MtListBudget("my_mt3d.list")
    >>> incremental, cumulative = mt_list.get_budget()
    >>> df_in, df_out = mt_list.get_dataframes(start_datetime="10-21-2015")

    """

    def __init__(self, file_name):

        # Set up file reading
        assert os.path.exists(file_name),"file_name {0} not found".format(file_name)
        self.file_name = file_name
        if sys.version_info[0] == 2:
            self.f = open(file_name, 'r')
        elif sys.version_info[0] == 3:
            self.f = open(file_name, 'r', encoding='ascii', errors='replace')

        self.tssp_lines = 0

        # Assign the budgetkey, which should have been overriden
        self.gw_budget_key = ">>>for component no."
        self.sw_budget_key = "STREAM MASS BUDGETS AT END OF TRANSPORT STEP".lower()
        self.time_key = "TOTAL ELAPSED TIME SINCE BEGINNING OF SIMULATION".lower()

        return


    def parse(self):
        """main entry point for parsing the list file.

        Returns
        -------
        df_gw,df_sw : pandas.DataFrame
            a dataframe for the groundwater mass and (optionally) surface-water mass budget.
            if the SFT process is not used, only one dataframe is returned.
        """
        try:
            import pandas as pd
        except:
            print("must use pandas")
            return
        self.gw_data = {}
        self.sw_data = {}
        with open(self.file_name) as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                line = line.lower()
                if self.gw_budget_key in line:
                    self._parse_gw(f,line)
                elif self.sw_budget_key in line:
                    self._parse_sw(f,line)
        #for i,l in self.gw_data.items():
        #    print(i,len(l))
        df_gw = pd.DataFrame(self.gw_data)

        if len(self.sw_data) > 0:
            df_sw = pd.DataFrame(self.sw_data)
            df_sw.loc[:,"totim"] = df_gw.totim_1
            return df_gw,df_sw
        return df_gw

    def _parse_gw(self,f,line):
        raw = line.strip().split()
        comp = int(raw[-1][:2])
        for _ in range(7):
            line = f.readline()
        totim = float(line.split()[-2])

        f.readline()
        f.readline()
        raw = f.readline().strip().split()
        kper = int(raw[-1])
        kstp = int(raw[-4][:-1])
        tkstp = int(raw[-7][:-1])
        for lab,val in zip(["totim","kper","kstp","tkstp"],[totim,kper,kstp,tkstp]):
            lab += '_{0}'.format(comp)
            if lab not in self.gw_data.keys():
                self.gw_data[lab] = []
            self.gw_data[lab].append(val)
        for _ in range(4):
            f.readline()
        while True:
            line = f.readline().lower()
            if line == '':
                raise Exception()
            elif '-----' in line:
                break
            item,ival,oval = self._parse_gw_line(line)
            item += "_{0}".format(comp)
            for lab, val in zip(["_in","_out"],[ival,oval]):
                iitem = item + lab
                if iitem not in self.gw_data.keys():
                    self.gw_data[iitem] = []
                self.gw_data[iitem].append(val)


    def _parse_gw_line(self,line):
        raw = line.lower().split(':')
        item = raw[0].strip().replace(' ','_')
        ival = float(raw[1].split()[0])
        oval = float(raw[1].split()[1])
        return item,ival,oval



    def _parse_sw(self,f,line):
        raw = line.split()
        comp = int(raw[-1])
        kper = int(raw[-4])
        kstp = int(raw[-7][:-1])
        tkstp = int(raw[-10][:-1])
        for lab,val in zip(["kper","kstp","tkstp"],[kper,kstp,tkstp]):
            lab += '_{0}'.format(comp)
            if lab not in self.gw_data.keys():
                self.sw_data[lab] = []
            self.sw_data[lab].append(val)
        for _ in range(4):
            f.readline()
        while True:
            line = f.readline().lower()
            if line == "":
                raise Exception()
            elif '------' in line:
                break
            item,cval,fval = self._parse_sw_line(line)
            item += '_{0}'.format(comp)
            for lab,val in zip(['_cum','_flx'],[cval,fval]):
                iitem = item+lab
                if iitem not in self.sw_data.keys():
                    self.sw_data[iitem] = []
                self.sw_data[iitem].append(val)
        in_tots = self._parse_sw_line(f.readline())
        f.readline()
        while True:
            line = f.readline().lower()
            if line == "":
                raise Exception()
            elif '------' in line:
                break
            outvals = self._parse_sw_line(line)
        out_tots = self._parse_sw_line(f.readline())
        f.readline()
        inout = self._parse_sw_line(f.readline())
        disc = self._parse_sw_line(f.readline())


    def _parse_sw_line(self,line):
        #print(line)
        raw = line.strip().split('=')
        citem = raw[0].strip().replace(" ", "_")
        cval = float(raw[1].split()[0])
        fitem = raw[1].split()[-1].replace(" ", "_")
        fval = float(raw[2])
        #assert citem == fitem,"{0}, {1}".format(citem,fitem)
        return citem,cval,fval






