# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 18, 2021 16:23:05 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


class ModflowGwtmst(mfpackage.MFPackage):
    """
    ModflowGwtmst defines a mst package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that MST flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    first_order_decay : boolean
        * first_order_decay (boolean) is a text keyword to indicate that first-
          order decay will occur. Use of this keyword requires that DECAY and
          DECAY_SORBED (if sorbtion is active) are specified in the GRIDDATA
          block.
    zero_order_decay : boolean
        * zero_order_decay (boolean) is a text keyword to indicate that zero-
          order decay will occur. Use of this keyword requires that DECAY and
          DECAY_SORBED (if sorbtion is active) are specified in the GRIDDATA
          block.
    sorption : string
        * sorption (string) is a text keyword to indicate that sorption will be
          activated. Valid sorption options include LINEAR, FREUNDLICH, and
          LANGMUIR. Use of this keyword requires that BULK_DENSITY and DISTCOEF
          are specified in the GRIDDATA block. If sorption is specified as
          FREUNDLICH or LANGMUIR then SP2 is also required in the GRIDDATA
          block.
    porosity : [double]
        * porosity (double) is the aquifer porosity.
    decay : [double]
        * decay (double) is the rate coefficient for first or zero-order decay
          for the aqueous phase of the mobile domain. A negative value
          indicates solute production. The dimensions of decay for first-order
          decay is one over time. The dimensions of decay for zero-order decay
          is mass per length cubed per time. decay will have no affect on
          simulation results unless either first- or zero-order decay is
          specified in the options block.
    decay_sorbed : [double]
        * decay_sorbed (double) is the rate coefficient for first or zero-order
          decay for the sorbed phase of the mobile domain. A negative value
          indicates solute production. The dimensions of decay_sorbed for
          first-order decay is one over time. The dimensions of decay_sorbed
          for zero-order decay is mass of solute per mass of aquifer per time.
          If decay_sorbed is not specified and both decay and sorbtion are
          active, then the program will terminate with an error. decay_sorbed
          will have no affect on simulation results unless the SORPTION keyword
          and either first- or zero-order decay are specified in the options
          block.
    bulk_density : [double]
        * bulk_density (double) is the bulk density of the aquifer in mass per
          length cubed. bulk_density is not required unless the SORBTION
          keyword is specified.
    distcoef : [double]
        * distcoef (double) is the distribution coefficient for the
          equilibrium-controlled linear sorption isotherm in dimensions of
          length cubed per mass. distcoef is not required unless the SORPTION
          keyword is specified.
    sp2 : [double]
        * sp2 (double) is the exponent for the Freundlich isotherm and the
          sorption capacity for the Langmuir isotherm.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    porosity = ArrayTemplateGenerator(("gwt6", "mst", "griddata", "porosity"))
    decay = ArrayTemplateGenerator(("gwt6", "mst", "griddata", "decay"))
    decay_sorbed = ArrayTemplateGenerator(
        ("gwt6", "mst", "griddata", "decay_sorbed")
    )
    bulk_density = ArrayTemplateGenerator(
        ("gwt6", "mst", "griddata", "bulk_density")
    )
    distcoef = ArrayTemplateGenerator(("gwt6", "mst", "griddata", "distcoef"))
    sp2 = ArrayTemplateGenerator(("gwt6", "mst", "griddata", "sp2"))
    package_abbr = "gwtmst"
    _package_type = "mst"
    dfn_file_name = "gwt-mst.dfn"

    dfn = [
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name first_order_decay",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name zero_order_decay",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name sorption",
            "type string",
            "valid linear freundlich langmuir",
            "reader urword",
            "optional true",
        ],
        [
            "block griddata",
            "name porosity",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block griddata",
            "name decay",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name decay_sorbed",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "optional true",
            "layered true",
        ],
        [
            "block griddata",
            "name bulk_density",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "optional true",
            "layered true",
        ],
        [
            "block griddata",
            "name distcoef",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name sp2",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        save_flows=None,
        first_order_decay=None,
        zero_order_decay=None,
        sorption=None,
        porosity=None,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distcoef=None,
        sp2=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super(ModflowGwtmst, self).__init__(
            model, "mst", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.first_order_decay = self.build_mfdata(
            "first_order_decay", first_order_decay
        )
        self.zero_order_decay = self.build_mfdata(
            "zero_order_decay", zero_order_decay
        )
        self.sorption = self.build_mfdata("sorption", sorption)
        self.porosity = self.build_mfdata("porosity", porosity)
        self.decay = self.build_mfdata("decay", decay)
        self.decay_sorbed = self.build_mfdata("decay_sorbed", decay_sorbed)
        self.bulk_density = self.build_mfdata("bulk_density", bulk_density)
        self.distcoef = self.build_mfdata("distcoef", distcoef)
        self.sp2 = self.build_mfdata("sp2", sp2)
        self._init_complete = True
