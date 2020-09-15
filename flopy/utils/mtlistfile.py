"""
This is a class for reading the mass budget from a (multi-component)
mt3d(usgs) run. Also includes support for SFT budget.

"""
import os
import sys
import warnings
from datetime import timedelta
import numpy as np

from ..utils.utils_def import totim_to_datetime


class MtListBudget(object):
    """
    MT3D mass budget reader

    Parameters
    ----------
    file_name : str
        the list file name


    Examples
    --------
    >>> mt_list = MtListBudget("my_mt3d.list")
    >>> incremental, cumulative = mt_list.get_budget()
    >>> gw_df, sw_df = mt_list.parse(start_datetime="10-21-2015")

    """

    def __init__(self, file_name):
        """
        Class constructor
        """

        self.file_name = file_name
        self.tssp_lines = 0
        # in case transport step number goes above 99999
        # which might be outputted as *****
        self.tkstp_overflow = 100000

        # Assign the budgetkey, which should have been overridden
        self.gw_budget_key = ">>>for component no."
        line = "STREAM MASS BUDGETS AT END OF TRANSPORT STEP"
        self.sw_budget_key = line.lower()
        line = "TOTAL ELAPSED TIME SINCE BEGINNING OF SIMULATION"
        self.time_key = line.lower()
        line = "TRANSPORT TIME STEP"
        self.tkstp_key = line.lower()

        return

    def parse(
        self, forgive=True, diff=True, start_datetime=None, time_unit="d"
    ):
        """
        Main entry point for parsing the list file.

        Parameters
        ----------
        forgive : bool
            flag to raise exceptions when fail-to-read occurs. Default is True
        diff : bool
            flag to return dataframes with 'in minus out' columns.  Default
            is True
        start_datetime : str
            str that can be parsed by pandas.to_datetime.  Example: '1-1-1970'.
            Default is None.
        time_unit : str
            str to pass to pandas.to_timedelta.  Default is 'd' (days)

        Returns
        -------
        df_gw,df_sw : pandas.DataFrame
            a dataframe for the groundwater mass and
            (optionally) surface-water mass budget.
            If the SFT process is not used, df_sw is None.
        """
        try:
            import pandas as pd
        except:
            msg = "MtListBudget.parse: pandas not available"
            raise ImportError(msg)

        self.gw_data = {}
        self.sw_data = {}
        self.lcount = 0
        with open(self.file_name) as f:
            while True:
                line = self._readline(f)
                if line is None:
                    break
                if self.gw_budget_key in line:
                    if forgive:
                        try:
                            self._parse_gw(f, line)
                        except Exception as e:
                            warnings.warn(
                                "error parsing GW mass budget "
                                "starting on line {0}: {1} ".format(
                                    self.lcount, str(e)
                                )
                            )
                            break
                    else:
                        self._parse_gw(f, line)
                elif self.sw_budget_key in line:
                    if forgive:
                        try:
                            self._parse_sw(f, line)
                        except Exception as e:
                            warnings.warn(
                                "error parsing SW mass budget"
                                " starting on line {0}: {1} ".format(
                                    self.lcount, str(e)
                                )
                            )
                            break
                    else:
                        self._parse_sw(f, line)
                elif self.tkstp_key in line:
                    self.tkstp_overflow = int(line[51:58])

        if len(self.gw_data) == 0:
            raise Exception("no groundwater budget info found...")

        # trim the lists so that they are all the same length
        # in case of a read fail
        min_len = 1e10
        for i, lst in self.gw_data.items():
            min_len = min(min_len, len(lst))
        for i, lst in self.gw_data.items():
            self.gw_data[i] = lst[:min_len]
        df_gw = pd.DataFrame(self.gw_data)
        df_gw.loc[:, "totim"] = df_gw.pop("totim_1")

        # if cumulative:
        #     keep = [c for c in df_gw.columns if "_flx" not in c]
        #     df_gw = df_gw.loc[:,keep]
        # else:
        #     keep = [c for c in df_gw.columns if "_cum" not in c]
        #     df_gw = df_gw.loc[:, keep]

        if diff:
            df_gw = self._diff(df_gw)

        if start_datetime is not None:
            dts = pd.to_datetime(start_datetime) + pd.to_timedelta(
                df_gw.totim, unit=time_unit
            )
            df_gw.index = dts
        else:
            df_gw.index = df_gw.totim
        df_sw = None
        if len(self.sw_data) > 0:
            # trim the lists so that they are all the same length
            # in case of a read fail
            min_len = 1e10
            for i, lst in self.sw_data.items():
                min_len = min(min_len, len(lst))
            min_len = min(min_len, df_gw.shape[0])
            for i, lst in self.sw_data.items():
                self.sw_data[i] = lst[:min_len]
            df_sw = pd.DataFrame(self.sw_data)
            df_sw.loc[:, "totim"] = df_gw.totim.iloc[:min_len].values

            # if cumulative:
            #     keep = [c for c in df_sw.columns if "_flx" not in c]
            #     df_sw = df_sw.loc[:, keep]
            # else:
            #     keep = [c for c in df_sw.columns if "_cum" not in c]
            #     df_sw = df_sw.loc[:, keep]

            if diff:
                df_sw = self._diff(df_sw)
            if start_datetime is not None:
                dts = pd.to_datetime(start_datetime) + pd.to_timedelta(
                    df_sw.pop("totim"), unit=time_unit
                )
                df_sw.index = dts
            else:
                df_sw.index = df_sw.pop("totim")

        for col in df_gw.columns:
            if "totim" in col:
                df_gw.pop(col)
        return df_gw, df_sw

    def _diff(self, df):
        try:
            import pandas as pd
        except:
            msg = "MtListBudget._diff: pandas not available"
            raise ImportError(msg)

        out_cols = [
            c for c in df.columns if "_out" in c and not c.startswith("net_")
        ]
        in_cols = [
            c for c in df.columns if "_in" in c and not c.startswith("net_")
        ]
        add_cols = [
            c for c in df.columns if c not in out_cols + in_cols + ["totim"]
        ]
        out_base = [c.replace("_out_", "_") for c in out_cols]
        in_base = [c.replace("_in_", "_") for c in in_cols]
        map_names = {
            "stream_accumulation": "stream_depletion",
            "stream_outflow": "inflow_to_stream",
            "stream_to_gw": "gw_to_stream",
            "mass_loss": "mass_gain",
            "evaporation": "precipitation",
        }
        out_base_mapped = []
        for base in out_base:
            if np.any([key in base for key in map_names.keys()]):
                for key, new in map_names.items():
                    if key in base:
                        out_base_mapped.append(base.replace(key, new))
            else:
                out_base_mapped.append(base)
        out_base = out_base_mapped
        in_dict = {ib: ic for ib, ic in zip(in_base, in_cols)}
        out_dict = {ib: ic for ib, ic in zip(out_base, out_cols)}
        in_base = set(in_base)
        out_base = set(out_base)
        out_base.update(in_base)
        out_base = list(out_base)
        out_base.sort()
        new = {"totim": df.totim}
        for col in out_base:
            if col in out_dict:
                odata = df.loc[:, out_dict[col]]
            else:
                odata = 0.0
            if col in in_dict:
                idata = df.loc[:, in_dict[col]]
            else:
                idata = 0.0
            new[col] = idata - odata

        new_df = pd.concat(
            [pd.DataFrame(new, index=df.index), df.loc[:, add_cols]], axis=1
        )
        return new_df

    def _readline(self, f):
        line = f.readline().lower()
        self.lcount += 1
        if line == "":
            return None
        return line

    def _parse_gw(self, f, line):
        raw = line.strip().split()
        comp = int(raw[-1][:2])
        self.imm = False
        for _ in range(7):
            line = self._readline(f)
            if line is None:
                raise Exception(
                    "EOF while reading from component header to totim"
                )
        try:
            totim = float(line.split()[-2])
        except Exception as e:
            raise Exception(
                "error parsing totim on line {0}: {1}".format(
                    self.lcount, str(e)
                )
            )

        for _ in range(3):
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading from totim to time step")
        try:
            kper = int(line[-6:-1])
            kstp = int(line[-26:-21])
            tkstp_str = line[-42:-37]
            if tkstp_str == "*****":
                tkstp = self.tkstp_overflow
            else:
                tkstp = int(tkstp_str)
        except Exception as e:
            raise Exception(
                "error parsing time step info on line {0}: {1}".format(
                    self.lcount, str(e)
                )
            )
        for lab, val in zip(
            ["totim", "kper", "kstp", "tkstp"], [totim, kper, kstp, tkstp]
        ):
            lab += "_{0}".format(comp)
            if lab not in self.gw_data.keys():
                self.gw_data[lab] = []
            self.gw_data[lab].append(val)
        for _ in range(4):
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading from time step to budget")
        break_next = False
        while True:
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading budget")
            elif "-----" in line:
                self.imm = False
                break_next = True
                continue
            elif "....immobile" in line:
                self.imm = True
                continue
            try:
                item, ival, oval = self._parse_gw_line(line)
            except Exception as e:
                raise Exception(
                    "error parsing GW items on line {0}: {1}".format(
                        self.lcount, str(e)
                    )
                )
            self._add_to_gw_data(item, ival, oval, comp)
            if break_next:
                break
        # read extras (in-out and percent discrep.)
        blank_count = 0
        while True:
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading budget")
            elif "-----" in line:
                break
            elif line.strip() == "":
                blank_count += 1
                # two consecutive blank line is end of block
                # sadly this is not always the case
                if blank_count == 2:
                    break
                else:
                    continue
            else:
                blank_count = 0  #
            try:
                item, ival, oval = self._parse_gw_line(line)
            except Exception as e:
                raise Exception(
                    "error parsing GW items "
                    "on line {0}: {1}".format(self.lcount, str(e))
                )
            self._add_to_gw_data(item, ival, oval, comp)
            if "discrepancy" in item:
                # can't rely on blank lines following block
                break

    def _parse_gw_line(self, line):
        raw = line.lower().split(":")
        item = raw[0].strip().strip(r"[\|]").replace(" ", "_")
        idx_ival = 0
        idx_oval = 1
        if self.imm:
            item = "imm_" + item
        if "TOTAL" in item.upper():
            idx_oval += 1  # to deal with the units in the total string
        # net (in-out) and discrepancy will only have 1 entry
        if len(raw[1].split()) < 2:
            ival = float(raw[1])
            oval = None
        else:
            ival = float(raw[1].split()[idx_ival])
            oval = -1.0 * float(raw[1].split()[idx_oval])
        return item, ival, oval

    def _add_to_gw_data(self, item, ival, oval, comp):
        item += "_{0}".format(comp)
        if oval is None:
            lab_val = zip([""], [ival], [""])
        else:
            lab_val = zip(["_in", "_out"], [ival, oval], ["_cum", "_cum"])
        for lab, val, suf in lab_val:
            iitem = item + lab + suf
            if iitem not in self.gw_data.keys():
                self.gw_data[iitem] = []
            self.gw_data[iitem].append(val)

    def _parse_sw(self, f, line):
        try:
            comp = int(line[-5:-1])
            kper = int(line[-24:-19])
            kstp = int(line[-44:-39])
            tkstp_str = line[-60:-55]
            if tkstp_str == "*****":
                tkstp = self.tkstp_overflow
            else:
                tkstp = int(tkstp_str)
        except Exception as e:
            raise Exception(
                "error parsing time step info on line {0}: {1}".format(
                    self.lcount, str(e)
                )
            )
        for lab, val in zip(["kper", "kstp", "tkstp"], [kper, kstp, tkstp]):
            lab += "_{0}".format(comp)
            if lab not in self.sw_data.keys():
                self.sw_data[lab] = []
            self.sw_data[lab].append(val)
        for _ in range(4):
            line = self._readline(f)
            if line is None:
                msg = "EOF while reading from time step to SW budget"
                raise Exception(msg)
        break_next = False
        while True:
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading 'in' SW budget")
            elif "------" in line:
                break_next = True  # make sure we read total in
                continue
            try:
                item, cval, fval = self._parse_sw_line(line)
            except Exception as e:
                msg = "error parsing 'in' SW items on line {}: " + "{}".format(
                    self.lcount, str(e)
                )
                raise Exception(msg)
            self._add_to_sw_data("in", item, cval, fval, comp)
            if break_next:
                break
        # read net in-out and percent discrep for cumulative and flux for sw
        line = self._readline(f)  # blank line read
        if line is None:
            raise Exception("EOF while reading 'in' SW budget")
        break_next = False
        while True:  # read outs
            line = self._readline(f)
            if line is None:
                raise Exception()
            elif "------" in line:
                break_next = True  # make sure we read total out
                continue
            try:
                item, cval, fval = self._parse_sw_line(line)
            except Exception as e:
                raise Exception(
                    "error parsing 'out' SW items on line {0}: {1}".format(
                        self.lcount, str(e)
                    )
                )
            self._add_to_sw_data("out", item, cval, fval, comp)
            if break_next:
                break
        # read extras (in-out and percent discrep.)
        blank_count = 0
        while True:
            line = self._readline(f)
            if line is None:
                raise Exception("EOF while reading 'out' SW budget")
            elif line.strip() == "":
                blank_count += 1
                if blank_count == 2:
                    break  # two consecutive blank line is end of block
                else:
                    continue
            else:
                blank_count = 0
            try:
                item, cval, fval = self._parse_sw_line(line)
            except Exception as e:
                raise Exception(
                    "error parsing 'out' SW items on line {0}: {1}".format(
                        self.lcount, str(e)
                    )
                )
            self._add_to_sw_data("net", item, cval, fval, comp)
        # out_tots = self._parse_sw_line(line)

    def _parse_sw_line(self, line):
        # print(line)
        raw = line.strip().split("=")
        citem = raw[0].strip().strip(r"[\|]").replace(" ", "_")
        cval = float(raw[1].split()[0])
        if len(raw) < 3:  # deal with flow error if written
            fval = None
            citem += raw[1].split()[-1]
        else:
            fitem = raw[1].split()[-1].replace(" ", "_")
            fval = float(raw[2])
        # assert citem == fitem,"{0}, {1}".format(citem,fitem)
        return citem, cval, fval

    def _add_to_sw_data(self, inout, item, cval, fval, comp):
        item += "_{0}".format(comp)
        if inout.lower() in set(["in", "out"]):
            item += "_{0}".format(inout)
        if fval is None:
            lab_val = zip([""], [cval])
        else:
            lab_val = zip(["_cum", "_flx"], [cval, fval])
        for lab, val in lab_val:
            iitem = item + lab
            if iitem not in self.sw_data.keys():
                self.sw_data[iitem] = []
            self.sw_data[iitem].append(val)
