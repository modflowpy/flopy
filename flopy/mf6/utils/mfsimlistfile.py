import os
import pathlib as pl
import re

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

    def is_normal_termination(self) -> bool:
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
        line = self.f.readline()
        if line == "":
            success = False
        else:
            success = True
        return success

    def get_model_runtime(self, units: str = "seconds") -> float:
        """
        Get the elapsed runtime of the model from the list file.

        Parameters
        ----------
        units : str
            Units in which to return the runtime. Acceptable values are
            'seconds', 'minutes', 'hours' (default is 'seconds')

        Returns
        -------
        out : float
            Floating point value with the runtime in requested units. Returns
            NaN if runtime not found in list file

        """
        # rewind the file
        self._rewind_file()

        units = units.lower()
        if (
            not units == "seconds"
            and not units == "minutes"
            and not units == "hours"
        ):
            raise AssertionError(
                '"units" input variable must be "minutes", "hours", '
                f'or "seconds": {units} was specified'
            )

        seekpoint = self._seek_to_string("Elapsed run time:")
        self.f.seek(seekpoint)
        line = self.f.readline()
        if line == "":
            return np.nan

        # yank out the floating point values from the Elapsed run time string
        times = list(map(float, re.findall(r"[+-]?[0-9.]+", line)))
        # pad an array with zeros and times with
        # [days, hours, minutes, seconds]
        times = np.array([0 for _ in range(4 - len(times))] + times)
        # convert all to seconds
        time2sec = np.array([24 * 60 * 60, 60 * 60, 60, 1])
        times_sec = np.sum(times * time2sec)
        # return in the requested units
        if units == "seconds":
            return times_sec
        elif units == "minutes":
            return times_sec / 60.0
        elif units == "hours":
            return times_sec / 60.0 / 60.0

    def get_formulate_time(self) -> float:
        """
        Get the formulate time for the solution from the list file.

        Returns
        -------
        out : float
            Floating point value with the formulate time,

        """
        # rewind the file
        self._rewind_file()

        try:
            seekpoint = self._seek_to_string("Total formulate time:")
        except:
            print(
                "'Total formulate time' not included in list file. "
                + "Returning NaN"
            )
            return np.nan

        self.f.seek(seekpoint)
        return float(self.f.readline().split()[3])

    def get_solution_time(self) -> float:
        """
        Get the solution time for the solution from the list file.

        Returns
        -------
        out : float
            Floating point value with the solution time,

        """
        # rewind the file
        self._rewind_file()

        try:
            seekpoint = self._seek_to_string("Total solution time:")
        except:
            print(
                "'Total solution time' not included in list file. "
                + "Returning NaN"
            )
            return np.nan

        self.f.seek(seekpoint)
        return float(self.f.readline().split()[3])

    def get_outer_iterations(self) -> int:
        """
        Get the total outer iterations from the list file.

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
        Get the total number of iterations from the list file.

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

    def get_memory_usage(self, virtual=False) -> float:
        """
        Get the simulation memory usage from the simulation list file.

        Parameters
        ----------

        Returns
        -------
        memory_usage : float
            Total memory usage for a simulation (in Gigabytes)

        """
        # initialize total_iterations
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
            line = self.f.readline()
            if line == "":
                break
            units = line.split()[-1]
            if units == "GIGABYTES":
                conversion = 1.0
            elif units == "MEGABYTES":
                conversion = 1e-3
            elif units == "KILOBYTES":
                conversion = 1e-6
            elif units == "BYTES":
                conversion = 1e-9
            else:
                raise ValueError(f"Unknown memory unit '{units}'")

            if virtual:
                tag = tags[2]
            else:
                tag = tags[1]
            seekpoint = self._seek_to_string(tag)
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            memory_usage = float(line.split()[-1]) * conversion

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
        Rewind the mfsim.lst file

        Returns
        -------

        """
        self.f.seek(0)


