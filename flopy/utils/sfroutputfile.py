import numpy as np


class SfrFile:
    """
    Read SFR package results from text file (ISTCB2 > 0)

    Parameters
    ----------
    filename : str
        Name of the sfr output file
    geometries : any
        Ignored
    verbose : any
        Ignored

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Indexing starts at one for: layer, row, column, segment, reach.
    Indexing starts at zero for: i, j, k, and kstpkper.

    Examples
    --------

    >>> import flopy
    >>> sfq = flopy.utils.SfrFile('mymodel.sfq')

    """

    # non-float dtypes (default is float)
    dtypes = {
        "layer": int,
        "row": int,
        "column": int,
        "segment": int,
        "reach": int,
    }

    def __init__(self, filename, geometries=None, verbose=False):
        """
        Class constructor.
        """
        try:
            import pandas as pd

            self.pd = pd
        except ImportError:
            print("This method requires pandas")
            self.pd = None
            return

        # get the number of rows to skip at top, and the number of data columns
        self.filename = filename
        evaluated_format = False
        has_gradient = False
        has_delUzstor = False
        has_elevation = False
        with open(self.filename) as f:
            for i, line in enumerate(f):
                if "GRADIENT" in line:
                    has_gradient = True
                if "CHNG. UNSAT." in line:
                    has_delUzstor = True
                if "ELEVATION" in line:
                    has_elevation = True
                items = line.strip().split()
                if len(items) > 0 and items[0].isdigit():
                    evaluated_format = True
                    self.sr = i
                    self.ncol = len(items)
                    break
        if not evaluated_format:
            raise ValueError(
                "could not evaluate format of {!r} for SfrFile".format(
                    self.filename
                )
            )
        # all outputs start with the same 15 columns
        self.names = [
            "layer",
            "row",
            "column",
            "segment",
            "reach",
            "Qin",
            "Qaquifer",
            "Qout",
            "Qovr",
            "Qprecip",
            "Qet",
            "stage",
            "depth",
            "width",
            "Cond",
        ]
        if has_gradient and has_delUzstor:
            raise ValueError(
                "column 16 should be either 'gradient' or 'Qwt', not both"
            )
        elif has_gradient:
            self.names.append("gradient")
        elif has_delUzstor:
            self.names += ["Qwt", "delUzstor"]
            if self.ncol == 18:
                self.names.append("gw_head")
        if has_elevation:
            self.names.append("strtop")
        self.times = self.get_times()
        self.geoms = None  # not implemented yet
        self._df = None

    def get_times(self):
        """
        Parse the stress period/timestep headers.

        Returns
        -------
        kstpkper : tuple
            list of kstp, kper tuples

        """
        kstpkper = []
        with open(self.filename) as input:
            for line in input:
                if "STEP" in line:
                    line = line.strip().split()
                    kper, kstp = int(line[3]) - 1, int(line[5]) - 1
                    kstpkper.append((kstp, kper))
        return kstpkper

    @property
    def df(self):
        if self._df is None:
            self._df = self.get_dataframe()
        return self._df

    @staticmethod
    def get_nstrm(df):
        """
        Get the number of SFR cells from the results dataframe.

        Returns
        -------
        nrch : int
            Number of SFR cells

        """
        wherereach1 = np.where((df.segment == 1) & (df.reach == 1))[0]
        if len(wherereach1) == 1:
            return len(df)
        elif len(wherereach1) > 1:
            return wherereach1[1]

    def get_dataframe(self):
        """
        Read the whole text file into a pandas dataframe.

        Returns
        -------
        df : pandas dataframe
            SFR output as a pandas dataframe

        """

        df = self.pd.read_csv(
            self.filename,
            delim_whitespace=True,
            header=None,
            names=self.names,
            error_bad_lines=False,
            skiprows=self.sr,
            low_memory=False,
        )

        # drop text between stress periods; convert to numeric
        df["layer"] = self.pd.to_numeric(df.layer, errors="coerce")
        df.dropna(axis=0, inplace=True)

        # convert to proper dtypes
        for c in df.columns:
            df[c] = df[c].astype(self.dtypes.get(c, float))

        # add time, reachID, and reach geometry (if it exists)
        self.nstrm = self.get_nstrm(df)
        dftimes = []
        times = self.get_times()
        newper = df.segment.diff().fillna(0).values < 0
        kstpkper = times.pop(0)
        for per in newper:
            if per:
                kstpkper = times.pop(0)
            dftimes.append(kstpkper)
        df["kstpkper"] = dftimes
        df["k"] = df["layer"] - 1
        df["i"] = df["row"] - 1
        df["j"] = df["column"] - 1

        if self.geoms is not None:
            geoms = self.geoms * self.nstrm
            df["geometry"] = geoms
        self._df = df
        return df

    def _get_result(self, segment, reach):
        """

        Parameters
        ----------
        segment : int or sequence of ints
            Segment number for each location.
        reach : int or sequence of ints
            Reach number for each location

        Returns
        -------

        """
        return self.df.loc[
            (self.df.segment == segment) & (self.df.reach == reach)
        ].copy()

    def get_results(self, segment, reach):
        """
        Get results for a single reach or sequence of segments and reaches.

        Parameters
        ----------
        segment : int or sequence of ints
            Segment number for each location.
        reach : int or sequence of ints
            Reach number for each location

        Returns
        -------
        results : dataframe
            Dataframe of same format as SfrFile.df, but subset to input locations.

        """
        try:
            segment = int(segment)
            reach = int(reach)
            results = self._get_result(segment, reach)
        except:
            locsr = list(zip(segment, reach))
            results = self.pd.DataFrame()
            for s, r in locsr:
                srresults = self._get_result(s, r)
                if len(srresults) > 0:
                    results = results.append(srresults)
                else:
                    print("No results for segment {}, reach {}!".format(s, r))
        return results
