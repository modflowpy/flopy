"""
mflmt module.  Contains the ModflowLmt class. Note that the user can access
the ModflowLmt class as `flopy.modflow.ModflowLmt`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?lmt6.htm>`_.

"""
import os

from ..pakbase import Package


class ModflowLmt(Package):
    """
    MODFLOW Link-MT3DMS Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    output_file_name : string
        Filename for output file (default is 'mt3d_link.ftl')
    unitnumber : int
        File unit number (default is 24).
    output_file_unit : int
        Output file unit number, pertaining to the file identified
        by output_file_name (default is 54).
    output_file_header : string
        Header for the output file (default is 'extended')
    output_file_format : {'formatted', 'unformatted'}
        Format of the output file (default is 'unformatted')
    package_flows : ['sfr', 'lak', 'uzf']
        Specifies which of the advanced package flows should be added to the
        flow-transport link (FTL) file. The addition of these flags may quickly
        increase the FTL file size. Thus, the user must specifically request
        their amendment within the FTL file. Default is not to add these
        terms to the FTL file by omitting the keyword package_flows from
        the LMT input file. One or multiple strings can be passed as a list to
        the argument.
    extension : string
        Filename extension (default is 'lmt6')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> lmt = flopy.modflow.ModflowLmt(m, output_file_name='mt3d_linkage.ftl')

    """

    def __init__(
        self,
        model,
        output_file_name="mt3d_link.ftl",
        output_file_unit=54,
        output_file_header="extended",
        output_file_format="unformatted",
        extension="lmt6",
        package_flows=[],
        unitnumber=None,
        filenames=None,
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowLmt._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowLmt._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self._generate_heading()
        self.url = "lmt.htm"
        self.output_file_name = output_file_name
        self.output_file_unit = output_file_unit
        self.output_file_header = output_file_header
        self.output_file_format = output_file_format
        self.package_flows = package_flows
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, "w")
        f.write(f"{self.heading}\n")
        f.write(f"OUTPUT_FILE_NAME     {self.output_file_name:20s}\n")
        f.write(f"OUTPUT_FILE_UNIT     {self.output_file_unit:10d}\n")
        f.write(f"OUTPUT_FILE_HEADER   {self.output_file_header:20s}\n")
        f.write(f"OUTPUT_FILE_FORMAT   {self.output_file_format:20s}\n")
        if self.package_flows:  # check that the list is not empty
            # Generate a string to write
            pckgs = ""
            if "sfr" in [x.lower() for x in self.package_flows]:
                pckgs += "SFR "
            if "lak" in [x.lower() for x in self.package_flows]:
                pckgs += "LAK "
            if "uzf" in [x.lower() for x in self.package_flows]:
                pckgs += "UZF "
            if "all" in [x.lower() for x in self.package_flows]:
                pckgs += "ALL"

            f.write(f"PACKAGE_FLOWS {pckgs}\n")

        f.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        lmt : ModflowLmt object
            ModflowLmt object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> lmt = flopy.modflow.ModflowGhb.load('test.lmt', m)

        """

        if model.verbose:
            print("loading lmt package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        elif hasattr(f, "name"):
            filename = f.name
        else:
            filename = None

        # set default values
        if filename:
            prefix = os.path.splitext(os.path.basename(filename))[0]
            output_file_name = f"{prefix}.ftl"
        else:
            output_file_name = f"{model.name}.ftl"
        output_file_unit = 333
        output_file_header = "standard"
        output_file_format = "unformatted"
        package_flows = []

        for line in f:
            if line[0] == "#":
                continue
            t = line.strip().split()
            if len(t) < 2:
                continue
            if t[0].lower() == "output_file_name":
                output_file_name = t[1]
            elif t[0].lower() == "output_file_unit":
                output_file_unit = int(t[1])
            elif t[0].lower() == "output_file_header":
                output_file_header = t[1]
            elif t[0].lower() == "output_file_format":
                output_file_format = t[1]
            elif t[0].lower() == "package_flows":
                # Multiple entries can follow 'package_flows'
                if len(t) > 1:
                    for i in range(1, len(t)):
                        package_flows.append(t[i])

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowLmt._ftype()
            )

        return cls(
            model,
            output_file_name=output_file_name,
            output_file_unit=output_file_unit,
            output_file_header=output_file_header,
            output_file_format=output_file_format,
            package_flows=package_flows,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "LMT6"

    @staticmethod
    def _defaultunit():
        return 30
