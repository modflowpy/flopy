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

    @property
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

    def get_runtime(
        self, units: str = "seconds", simulation_timer: str = "elapsed"
    ) -> float:
        """
        Get the elapsed runtime of the model from the list file.

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
                + " ,".join(UNITS)
                + f": {units} was specified."
            )
            raise ValueError(msg)

        # rewind the file
        self._rewind_file()

        seekpoint = self._seek_to_string(TIMERS_DICT[simulation_timer])
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
        virtual : bool
            Return total or virtual memory usage (default is total)

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
