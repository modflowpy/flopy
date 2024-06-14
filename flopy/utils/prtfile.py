"""
Support for MODFLOW 6 PRT output files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from flopy.utils.particletrackfile import ParticleTrackFile


class PathlineFile(ParticleTrackFile):
    """Provides MODFLOW 6 prt output file support."""

    key_cols = ["imdl", "iprp", "irpt", "trelease"]
    """Columns making up each particle's composite key."""

    dtypes = {
        "base": np.dtype(
            [
                ("kper", np.int32),
                ("kstp", np.int32),
                ("imdl", np.int32),
                ("iprp", np.int32),
                ("irpt", np.int32),
                ("ilay", np.int32),
                ("icell", np.int32),
                ("izone", np.int32),
                ("istatus", np.int32),
                ("ireason", np.int32),
                ("trelease", np.float32),
                ("t", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("name", np.str_),
            ]
        ),
        "full": np.dtype(
            [
                ("kper", np.int32),
                ("kstp", np.int32),
                ("imdl", np.int32),
                ("iprp", np.int32),
                ("irpt", np.int32),
                ("ilay", np.int32),
                ("k", np.int32),  # canonical base dtype
                ("icell", np.int32),
                ("izone", np.int32),
                ("idest", np.int32),  # canonical full dtype
                ("dest", np.str_),  # canonical full dtype
                ("istatus", np.int32),
                ("ireason", np.int32),
                ("trelease", np.float32),
                ("t", np.float32),
                ("t0", np.float32),  # canonical full dtype
                ("tt", np.float32),  # canonical full dtype
                ("time", np.float32),  # canonical full dtype
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("particleid", np.int32),  # canonical base dtype
                ("name", np.str_),
            ]
        ),
    }
    """Base and full (extended) PRT pathline data dtypes."""

    @property
    def dtype(self):
        """
        PRT track file dtype. This is loaded dynamically from the binary header file or CSV file
        headers. A best-effort attempt is made to add extra columns to comply with the canonical
        `flopy.utils.particletrackfile.dtypes["base"]`, as well as convenience columns including
        release and termination time, and destination zone number and name, for the full dtype.
        If the loaded dtype is discovered to be different from `PrtPathlineFile.dtypes["base"]`,
        a warning will be issued.

        Consumers of this class which expect the canonical particle track file attributes should
        call `validate()` to make sure the attributes were successfully loaded.
        """
        return self._dtype

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        header_filename: Optional[Union[str, os.PathLike]] = None,
        destination_map: Optional[Dict[int, str]] = None,
        verbose: bool = False,
    ):
        super().__init__(filename, verbose)
        self._dtype, self._data = self._load(header_filename, destination_map)

    def _load(
        self,
        header_filename=None,
        destination_map=None,
    ) -> np.ndarray:
        # if a header file is present, we're reading a binary file
        hdr_fname = (
            None
            if header_filename is None
            else Path(header_filename).expanduser().absolute()
        )
        if hdr_fname is not None and hdr_fname.is_file():
            lines = open(hdr_fname).readlines()
            dtype = np.dtype(
                {
                    "names": lines[0].strip().split(","),
                    "formats": lines[1].strip().split(","),
                }
            )
            data = pd.DataFrame(np.fromfile(self.fname, dtype=dtype))
        else:
            # otherwise we're reading a CSV file
            data = pd.read_csv(self.fname)
            dtype = data.to_records(index=False).dtype

        # add particle id column
        data = data.sort_values(self.key_cols)
        data["particleid"] = data.groupby(self.key_cols).ngroup()

        # add time, release time and termination time columns
        data["time"] = data["t"]
        data["t0"] = (
            data.groupby("particleid")
            .apply(lambda x: x.t.min())
            .to_frame(name="t0")
            .t0
        )
        data["tt"] = (
            data.groupby("particleid")
            .apply(lambda x: x.t.max())
            .to_frame(name="tt")
            .tt
        )

        # add k column
        data["k"] = data["ilay"]

        # assign destinations if zone map is provided
        if destination_map is not None:
            data["idest"] = data[data.istatus > 1].izone
            data["dest"] = data.apply(
                lambda row: destination_map[row.idest],
                axis=1,
            )

        return data.to_records(index=False).dtype, data

    def intersect(
        self, cells, to_recarray=True
    ) -> Union[List[np.recarray], np.recarray]:
        """Find intersection of pathlines with cells."""

        if not all(isinstance(nn, int) for nn in cells):
            raise ValueError("Expected integer cell numbers")

        idxs = np.in1d(self._data[["icell"]], np.array(cells, dtype=np.int32))
        sect = self._data[idxs].copy()
        pids = np.unique(sect["particleid"])
        if to_recarray:
            idxs = np.in1d(sect["particleid"], pids)
            return sect[idxs].sort_values(by=["particleid", "time"])
        else:
            return [self.get_data(pid) for pid in pids]

    @staticmethod
    def get_track_dtype(path: Union[str, os.PathLike]):
        """Read a numpy dtype describing particle track
        data format from the ascii track header file."""

        hdr_lns = open(path).readlines()
        hdr_lns_spl = [[ll.strip() for ll in l.split(",")] for l in hdr_lns]
        return np.dtype(list(zip(hdr_lns_spl[0], hdr_lns_spl[1])))
