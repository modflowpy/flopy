# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 10, 2025 23:05:19 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowOlfcxs(mfpackage.MFPackage):
    """
    ModflowOlfcxs defines a cxs package within a olf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of stream
          reach information will be written to the listing file immediately
          after it is read.
    nsections : integer
        * nsections (integer) integer value specifying the number of cross
          sections that will be defined. There must be NSECTIONS entries in the
          PACKAGEDATA block.
    npoints : integer
        * npoints (integer) integer value specifying the total number of cross-
          section points defined for all reaches. There must be NPOINTS entries
          in the CROSSSECTIONDATA block.
    packagedata : [idcxs, nxspoints]
        * idcxs (integer) integer value that defines the cross section number
          associated with the specified PACKAGEDATA data on the line. IDCXS
          must be greater than zero and less than or equal to NSECTIONS.
          Information must be specified for every section or the program will
          terminate with an error. The program will also terminate with an
          error if information for a section is specified more than once. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * nxspoints (integer) integer value that defines the number of points
          used to define the define the shape of a section. NXSPOINTS must be
          greater than zero or the program will terminate with an error.
          NXSPOINTS defines the number of points that must be entered for the
          reach in the CROSSSECTIONDATA block. The sum of NXSPOINTS for all
          sections must equal the NPOINTS dimension.
    crosssectiondata : [xfraction, height, manfraction]
        * xfraction (double) real value that defines the station (x) data for
          the cross-section as a fraction of the width (WIDTH) of the reach.
          XFRACTION must be greater than or equal to zero but can be greater
          than one. XFRACTION values can be used to decrease or increase the
          width of a reach from the specified reach width (WIDTH).
        * height (double) real value that is the height relative to the top of
          the lowest elevation of the streambed (ELEVATION) and corresponding
          to the station data on the same line. HEIGHT must be greater than or
          equal to zero and at least one cross-section height must be equal to
          zero.
        * manfraction (double) real value that defines the Manning's roughness
          coefficient data for the cross-section as a fraction of the Manning's
          roughness coefficient for the reach (MANNINGSN) and corresponding to
          the station data on the same line. MANFRACTION must be greater than
          zero. MANFRACTION is applied from the XFRACTION value on the same
          line to the XFRACTION value on the next line.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    packagedata = ListTemplateGenerator(("olf6", "cxs", "packagedata", "packagedata"))
    crosssectiondata = ListTemplateGenerator(
        ("olf6", "cxs", "crosssectiondata", "crosssectiondata")
    )
    package_abbr = "olfcxs"
    _package_type = "cxs"
    dfn_file_name = "olf-cxs.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal iprpak",
        ],
        [
            "block dimensions",
            "name nsections",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block dimensions",
            "name npoints",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray idcxs nxspoints",
            "shape (nsections)",
            "reader urword",
        ],
        [
            "block packagedata",
            "name idcxs",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "numeric_index true",
        ],
        [
            "block packagedata",
            "name nxspoints",
            "type integer",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block crosssectiondata",
            "name crosssectiondata",
            "type recarray xfraction height manfraction",
            "shape (npoints)",
            "reader urword",
        ],
        [
            "block crosssectiondata",
            "name xfraction",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block crosssectiondata",
            "name height",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block crosssectiondata",
            "name manfraction",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional false",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        print_input=None,
        nsections=None,
        npoints=None,
        packagedata=None,
        crosssectiondata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "cxs", filename, pname, loading_package, **kwargs)

        # set up variables
        self.print_input = self.build_mfdata("print_input", print_input)
        self.nsections = self.build_mfdata("nsections", nsections)
        self.npoints = self.build_mfdata("npoints", npoints)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self.crosssectiondata = self.build_mfdata("crosssectiondata", crosssectiondata)
        self._init_complete = True
