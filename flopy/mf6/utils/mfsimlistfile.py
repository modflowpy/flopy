import os
import pathlib as pl
import re
import warnings

import numpy as np


class MfSimulationList:
    def __init__(self, file_name: os.PathLike):
        # Set up file reading
        if isinstance(file_name, str):
            file_name = pl.Path(file_name)
        if not file_name.is_file():
            raise FileNotFoundError(f"file_name `{file_name}` not found")
        self.file_name = file_name
        self.f = open(file_name, "r", encoding="ascii", errors="replace")

        self.normal_termination = self._get_termination_message()
        self.memory_print_option = self._memory_print_option()

    @property
    def is_normal_termination(self) -> bool:
        """
        Return boolean indicating if the simulation terminated normally

        Returns
        -------
        success: bool
            Boolean indicating if the simulation terminated normally

        """
        return self.normal_termination

    def get_runtime(
        self, units: str = "seconds", simulation_timer: str = "elapsed"
    ) -> float:
        """
        Get model runtimes from the simulation list file.

        Parameters
        ----------
        units : str
            Units in which to return the timer. Acceptable values are
            'seconds', 'minutes', 'hours' (default is 'seconds')
        simulation_timer : str
            Timer to return. Acceptable values are 'elapsed', 'formulate',
            'solution' (default is 'elapsed')

        Returns
        -------
        out : float
            Floating point value with the runtime in requested units. Returns
            NaN if runtime not found in list file

        """
        UNITS = (
            "seconds",
            "minutes",
            "hours",
        )
        TIMERS = (
            "elapsed",
            "formulate",
            "solution",
        )
        TIMERS_DICT = {
            "elapsed": "Elapsed run time:",
            "formulate": "Total formulate time:",
            "solution": "Total solution time:",
        }

        simulation_timer = simulation_timer.lower()
        if simulation_timer not in TIMERS:
            msg = (
                "simulation_timers input variable must be "
                + " ,".join(TIMERS)
                + f": {simulation_timer} was specified."
            )
            raise ValueError(msg)

        units = units.lower()
        if units not in UNITS:
            msg = (
                "units input variable must be "
                + ", ".join(UNITS)
                + f": {units} was specified."
            )
            raise ValueError(msg)

        # rewind the file
        self._rewind_file()

        if simulation_timer == "elapsed":
            seekpoint = self._seek_to_string(TIMERS_DICT[simulation_timer])
            self.f.seek(seekpoint)
            line = self.f.readline().strip()
            if line == "":
                return np.nan

            # parse floating point values from the Elapsed run time string
            times = list(map(float, re.findall(r"[+-]?[0-9.]+", line)))

            # pad an array with zeros and times with
            # [days, hours, minutes, seconds]
            times = np.array([0 for _ in range(4 - len(times))] + times)

            # convert all to seconds
            time2sec = np.array([24 * 60 * 60, 60 * 60, 60, 1])
            times_sec = np.sum(times * time2sec)
        else:
            seekpoint = self._seek_to_string(TIMERS_DICT[simulation_timer])
            self.f.seek(seekpoint)
            line = self.f.readline().strip()
            if line == "":
                return np.nan
            times_sec = float(line.split()[3])

        # return time in the requested units
        if units == "seconds":
            return times_sec
        elif units == "minutes":
            return times_sec / 60.0
        elif units == "hours":
            return times_sec / 60.0 / 60.0

    def get_outer_iterations(self) -> int:
        """
        Get the total outer iterations from the simulation list file.

        Parameters
        ----------

        Returns
        -------
        outer_iterations : float
            Sum of all TOTAL ITERATIONS found in the list file

        """
        # initialize total_iterations
        outer_iterations = 0

        # rewind the file
        self._rewind_file()

        while True:
            seekpoint = self._seek_to_string("CALLS TO NUMERICAL SOLUTION IN")
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            outer_iterations += int(line.split()[0])

        return outer_iterations

    def get_total_iterations(self) -> int:
        """
        Get the total number of iterations from the simulation list file.

        Parameters
        ----------

        Returns
        -------
        total_iterations : float
            Sum of all TOTAL ITERATIONS found in the list file

        """
        # initialize total_iterations
        total_iterations = 0

        # rewind the file
        self._rewind_file()

        while True:
            seekpoint = self._seek_to_string("TOTAL ITERATIONS")
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            total_iterations += int(line.split()[0])

        return total_iterations

    def get_memory_usage(
        self,
        virtual: bool = False,
        units: str = "gigabytes",
    ) -> float:
        """
        Get the simulation memory usage from the simulation list file.

        Parameters
        ----------
        virtual : bool
            Return total or virtual memory usage (default is total)
        units : str
            Memory units for return results. Valid values are 'gigabytes',
            'megabytes', 'kilobytes', and 'bytes' (default is 'gigabytes').

        Returns
        -------
        memory_usage : float
            Total memory usage for a simulation (in Gigabytes)

        """
        # initialize memory_usage
        memory_usage = 0.0

        # rewind the file
        self._rewind_file()

        tags = (
            "MEMORY MANAGER TOTAL STORAGE BY DATA TYPE",
            "Total",
            "Virtual",
        )

        while True:
            seekpoint = self._seek_to_string(tags[0])

            self.f.seek(seekpoint)
            line = self.f.readline().strip()
            if line == "":
                break
            sim_units = line.split()[-1]
            unit_conversion = self._get_memory_unit_conversion(
                sim_units,
                return_units_str=units.upper(),
            )

            if virtual:
                tag = tags[2]
            else:
                tag = tags[1]
            seekpoint = self._seek_to_string(tag)
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            memory_usage = float(line.split()[-1]) * unit_conversion

        return memory_usage

    def get_non_virtual_memory_usage(self):
        """

        Returns
        -------
        non_virtual: float
            Non-virtual memory usage, which is the difference between the
            total and virtual memory usage

        """
        return self.get_memory_usage() - self.get_memory_usage(virtual=True)

    def get_memory_summary(self, units: str = "gigabytes") -> dict:
        """
        Get the summary memory information if it is available in the
        simulation list file. Summary memory information is only available
        if the memory_print_option is set to 'summary' in the simulation
        name file options block.

        Parameters
        ----------
        units : str
            Memory units for return results. Valid values are 'gigabytes',
            'megabytes', 'kilobytes', and 'bytes' (default is 'gigabytes').

        Returns
        -------
        memory_summary : dict
            dictionary with the total memory for each simulation component.
            None is returned if summary memory data is not present in the
            simulation listing file.


        """
        # initialize the return variable
        memory_summary = None

        if self.memory_print_option != "summary":
            msg = (
                "Cannot retrieve memory data using get_memory_summary() "
                + "since memory_print_option is not set to 'SUMMARY'. "
                + "Returning None."
            )
            warnings.warn(msg, category=Warning)

        else:
            # rewind the file
            self._rewind_file()

            seekpoint = self._seek_to_string(
                "SUMMARY INFORMATION ON VARIABLES "
                + "STORED IN THE MEMORY MANAGER"
            )
            self.f.seek(seekpoint)
            line = self.f.readline().strip()

            if line != "":
                sim_units = line.split()[-1]
                unit_conversion = self._get_memory_unit_conversion(
                    sim_units,
                    return_units_str=units.upper(),
                )
                # read the header
                for k in range(3):
                    _ = self.f.readline()
                terminator = 100 * "-"
                memory_summary = {}
                while True:
                    line = self.f.readline().strip()
                    if line == terminator:
                        break
                    data = line.split()
                    memory_summary[data[0]] = float(data[-1]) * unit_conversion

        return memory_summary

    def get_memory_all(self, units: str = "gigabytes") -> dict:
        """
        Get a dictionary of the memory table written if it is available in the
        simulation list file. The memory table is only available
        if the memory_print_option is set to 'all' in the simulation
        name file options block.

        Parameters
        ----------
        units : str
            Memory units for return results. Valid values are 'gigabytes',
            'megabytes', 'kilobytes', and 'bytes' (default is 'gigabytes').

        Returns
        -------
        memory_all : dict
            dictionary with the memory information for each variable in the
            MODFLOW 6 memory manager. The dictionary keys are the full memory
            path for a variable (the memory path and variable name). The
            dictionary entry for each key includes the memory path, the
            variable name, data type, size, and memory used for each variable.
            None is returned if the memory table is not present in the
            simulation listing file.


        """
        # initialize the return variable
        memory_all = None

        TYPE_SIZE = {
            "INTEGER": 4.0,
            "DOUBLE": 8.0,
            "LOGICAL": 4.0,
            "STRING": 1.0,
        }
        if self.memory_print_option != "all":
            msg = (
                "Cannot retrieve memory data using get_memory_all() since "
                + "memory_print_option is not set to 'ALL'. Returning None."
            )
            warnings.warn(msg, category=Warning)
        else:
            # rewind the file
            self._rewind_file()

            seekpoint = self._seek_to_string(
                "DETAILED INFORMATION ON VARIABLES "
                + "STORED IN THE MEMORY MANAGER"
            )
            self.f.seek(seekpoint)
            line = self.f.readline().strip()

            if line != "":
                sim_units = "BYTES"
                unit_conversion = self._get_memory_unit_conversion(
                    sim_units,
                    return_units_str=units.upper(),
                )
                # read the header
                for k in range(3):
                    _ = self.f.readline()
                terminator = 173 * "-"
                memory_all = {}
                # read the data
                while True:
                    line = self.f.readline().strip()
                    if line == terminator:
                        break
                    if "STRING LEN=" in line:
                        mempath = line[0:50].strip()
                        varname = line[51:67].strip()
                        data_type = line[68:84].strip()
                        no_items = float(line[84:105].strip())
                        assoc_var = line[106:].strip()
                        variable_bytes = (
                            TYPE_SIZE["STRING"]
                            * float(data_type.replace("STRING LEN=", ""))
                            * no_items
                        )
                    else:
                        data = line.split()
                        mempath = data[0]
                        varname = data[1]
                        data_type = data[2]
                        no_items = float(data[3])
                        assoc_var = data[4]
                        variable_bytes = TYPE_SIZE[data_type] * no_items

                    if assoc_var == "--":
                        size_bytes = variable_bytes * unit_conversion
                        memory_all[f"{mempath}/{varname}"] = {
                            "MEMPATH": mempath,
                            "VARIABLE": varname,
                            "DATATYPE": data_type,
                            "SIZE": no_items,
                            "MEMORYSIZE": size_bytes,
                        }

        return memory_all

    def _seek_to_string(self, s):
        """
        Parameters
        ----------
        s : str
            Seek through the file to the next occurrence of s.  Return the
            seek location when found.

        Returns
        -------
        seekpoint : int
            Next location of the string

        """
        while True:
            seekpoint = self.f.tell()
            line = self.f.readline()
            if line == "":
                break
            if s in line:
                break
        return seekpoint

    def _rewind_file(self):
        """
        Rewind the simulation list file

        Returns
        -------

        """
        self.f.seek(0)

    def _get_termination_message(self) -> bool:
        """
        Determine if the simulation terminated normally

        Returns
        -------
        success: bool
            Boolean indicating if the simulation terminated normally

        """
        # rewind the file
        self._rewind_file()

        seekpoint = self._seek_to_string("Normal termination of simulation.")
        self.f.seek(seekpoint)
        line = self.f.readline().strip()
        if line == "":
            success = False
        else:
            success = True
        return success

    def _get_memory_unit_conversion(
        self,
        sim_units_str: str,
        return_units_str,
    ) -> float:
        """
        Calculate memory unit conversion factor that converts from reported
        units to gigabytes

        Parameters
        ----------
        sim_units_str : str
            Memory Units in the simulation listing file. Valid values are
            'GIGABYTES', 'MEGABYTES', 'KILOBYTES', or 'BYTES'.

        Returns
        -------
        unit_conversion : float
            Unit conversion factor

        """
        valid_units = (
            "GIGABYTES",
            "MEGABYTES",
            "KILOBYTES",
            "BYTES",
        )
        if sim_units_str not in valid_units:
            raise ValueError(f"Unknown memory unit '{sim_units_str}'")

        factor = [1.0, 1e-3, 1e-6, 1e-9]
        if return_units_str == "MEGABYTES":
            factor = [v * 1e3 for v in factor]
        elif return_units_str == "KILOBYTES":
            factor = [v * 1e6 for v in factor]
        elif return_units_str == "BYTES":
            factor = [v * 1e9 for v in factor]
        factor_dict = {tag: factor[idx] for idx, tag in enumerate(valid_units)}

        return factor_dict[sim_units_str]

    def _memory_print_option(self) -> str:
        """
        Determine the memory print option selected

        Returns
        -------
        option: str
            memory_print_option ('summary', 'all', or None)

        """
        # rewind the file
        self._rewind_file()

        seekpoint = self._seek_to_string("MEMORY_PRINT_OPTION SET TO")
        self.f.seek(seekpoint)
        line = self.f.readline().strip()
        if line == "":
            option = None
        else:
            option_list = re.findall(r'"([^"]*)"', line)
            if len(option_list) < 1:
                raise LookupError(
                    "could not parse memory_print_option from" + f"'{line}'"
                )
            option = option_list[-1].lower()
            if option not in ("all", "summary"):
                raise ValueError(f"unknown memory print option {option}")
        return option
