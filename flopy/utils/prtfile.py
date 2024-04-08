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
                ("k", np.int32),  # conform to canonical dtype
                ("icell", np.int32),
                ("izone", np.int32),
                ("idest", np.int32),  # destination zone, for convenience
                ("dest", np.str_),  # destination name, for convenience
                ("istatus", np.int32),
                ("ireason", np.int32),
                ("trelease", np.int32),
                ("t", np.int32),
                ("t0", np.int32),  # release time, for convenience
                ("tt", np.int32),  # termination time, convenience
                ("time", np.int32),  # conform to canonical dtype
                ("x", np.int32),
                ("y", np.int32),
                ("z", np.int32),
                ("particleid", np.int32),  # conform to canonical dtype
                ("name", np.str_),
            ]
        ),
    }

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        header_filename: Optional[Union[str, os.PathLike]] = None,
        destination_map: Optional[Dict[int, str]] = None,
        verbose: bool = False,
    ):
        super().__init__(filename, verbose)
        self.dtype, self._data = self._load(header_filename, destination_map)

    def _load(
        self,
        header_filename=None,
        destination_map=None,
    ) -> np.ndarray:
        # load dtype from header file if present, otherwise use default dtype
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
        else:
            dtype = self.dtypes["base"]

        # read as binary or csv
        try:
            data = pd.read_csv(self.fname, dtype=dtype)
        except UnicodeDecodeError:
            try:
                data = np.fromfile(self.fname, dtype=dtype)
            except:
                raise ValueError("Unreadable file, expected binary or CSV")

        # add particle id column
        data = data.sort_values(self.key_cols)
        data["particleid"] = data.groupby(self.key_cols).ngroup()

        # add release time and termination time columns
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

        # assign destinations if zone map is provided
        if destination_map is not None:
            data["idest"] = data[data.istatus > 1].izone
            data["dest"] = data.apply(
                lambda row: destination_map[row.idest],
                axis=1,
            )

        return dtype, data

    def intersect(
        self, cells, to_recarray
    ) -> Union[List[np.recarray], np.recarray]:
        if isinstance(cells, (list, tuple)):
            allint = all(isinstance(el, int) for el in cells)
            if allint:
                t = []
                for el in cells:
                    t.append((el,))
                    cells = t

        icell = self._data[["icell"]]
        cells = np.array(cells, dtype=icell.dtype)
        inds = np.in1d(icell, cells)
        epdest = self._data[inds].copy().view(np.recarray)
        if to_recarray:
            # use particle ids to get the rest of the paths
            inds = np.in1d(self._data["particleid"], epdest.particleid)
            series = self._data[inds].copy()
            series.sort(order=["particleid", "time"])
            series = series.view(np.recarray)
        else:
            # collect unique particleids in selection
            partids = np.unique(epdest["particleid"])
            series = [self.get_data(partid) for partid in partids]

        return series
