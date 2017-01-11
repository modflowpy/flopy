import numpy as np

class SfrFile():
    """
    Read SFR package results from text file (ISTCB2 > 0)

    Parameters
    ----------
    filename : string
        Name of the sfr output file
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> sfq = flopy.utils.SfrFile('mymodel.sfq')

    """

    names = ["layer", "row", "column", "segment", "reach", "Qin",
             "Qaquifer", "Qout", "Qovr", "Qprecip", "Qet",
             "stage", "depth", "width", "Cond", "gradient"]
    def __init__(self, filename, geometries=None, verbose=False):
        """
        Class constructor.
        """
        try:
            import pandas as pd
            self.pd = pd
        except:
            print('This method requires pandas')

        # get the number of rows to skip at top
        self.filename = filename
        self.sr = self.get_skiprows()
        self.times = self.get_times()
        self.geoms = None # not implemented yet
        self.df = None

    def get_skiprows(self):
        """Get the number of rows to skip at the top."""
        with open(self.filename) as input:
            for i, line in enumerate(input):
                line = line.strip().split()
                if len(line) > 0 and line[0].isdigit():
                    return i

    def get_times(self):
        """Parse the stress period/timestep headers."""
        kstpkper = []
        with open(self.filename) as input:
            for line in input:
                if 'STEP' in line:
                    line = line.strip().split()
                    kper, kstp = int(line[3]) - 1, int(line[5]) - 1
                    kstpkper.append((kper, kstp))
        return kstpkper

    @staticmethod
    def get_nstrm(df):
        """Get the number of SFR cells from the results dataframe."""
        wherereach1 = np.where((df.segment == 1) & (df.reach == 1))[0]
        if len(wherereach1) == 1:
            return len(df)
        elif len(wherereach1) > 1:
            return wherereach1[1]

    def get_dataframe(self):
        """Read the whole text file into a pandas dataframe."""

        df = self.pd.read_csv(self.filename, delim_whitespace=True,
                         header=None, names=self.names,
                         error_bad_lines=False, comment='S',
                         skiprows=self.sr)

        # add time, reachID, and reach geometry (if it exists)
        self.nstrm = self.get_nstrm(df)
        time = []
        times = []
        geoms = []
        reachID = []
        for ts in self.times:
            time += [ts[1]] * np.abs(self.nstrm)
            times.append(ts[1])
            reachID += list(range(self.nstrm))
        df['time'] = time
        df['reachID'] = reachID

        if self.geoms is not None:
            geoms = self.geoms * self.nstrm
            df['geometry'] = geoms
        self.df = df
        return df

    def _get_result(self, segment, reach):
        return self.df.loc[(self.df.segment == segment) & (self.df.reach == reach)].copy()

    def get_results(self, segment, reach):
        """Get results for a single reach or sequence of segments and reaches.

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
                    print('No results for segment {}, reach {}!'.format(s, r))
        return results




