# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 11, 2025 01:24:12 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowGwtist(mfpackage.MFPackage):
    """
    ModflowGwtist defines a ist package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that IST flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the binary output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
    sorption : string
        * sorption (string) is a text keyword to indicate that sorption will be
          activated. Valid sorption options include LINEAR, FREUNDLICH, and
          LANGMUIR. Use of this keyword requires that BULK_DENSITY and DISTCOEF
          are specified in the GRIDDATA block. If sorption is specified as
          FREUNDLICH or LANGMUIR then SP2 is also required in the GRIDDATA
          block. The sorption option must be consistent with the sorption
          option specified in the MST Package or the program will terminate
          with an error.
    first_order_decay : boolean
        * first_order_decay (boolean) is a text keyword to indicate that first-
          order decay will occur. Use of this keyword requires that DECAY and
          DECAY_SORBED (if sorption is active) are specified in the GRIDDATA
          block.
    zero_order_decay : boolean
        * zero_order_decay (boolean) is a text keyword to indicate that zero-
          order decay will occur. Use of this keyword requires that DECAY and
          DECAY_SORBED (if sorption is active) are specified in the GRIDDATA
          block.
    cim_filerecord : [cimfile]
        * cimfile (string) name of the output file to write immobile
          concentrations. This file is a binary file that has the same format
          and structure as a binary head and concentration file. The value for
          the text variable written to the file is CIM. Immobile domain
          concentrations will be written to this file at the same interval as
          mobile domain concentrations are saved, as specified in the GWT Model
          Output Control file.
    cimprintrecord : [columns, width, digits, format]
        * columns (integer) number of columns for writing data.
        * width (integer) width for writing each number.
        * digits (integer) number of digits to use for writing a number.
        * format (string) write format can be EXPONENTIAL, FIXED, GENERAL, or
          SCIENTIFIC.
    sorbate_filerecord : [sorbatefile]
        * sorbatefile (string) name of the output file to write immobile
          sorbate concentration information. Immobile sorbate concentrations
          will be written whenever aqueous immobile concentrations are saved,
          as determined by settings in the Output Control option.
    porosity : [double]
        * porosity (double) porosity of the immobile domain specified as the
          immobile domain pore volume per immobile domain volume.
    volfrac : [double]
        * volfrac (double) fraction of the cell volume that consists of this
          immobile domain. The sum of all immobile domain volume fractions must
          be less than one.
    zetaim : [double]
        * zetaim (double) mass transfer rate coefficient between the mobile and
          immobile domains, in dimensions of per time.
    cim : [double]
        * cim (double) initial concentration of the immobile domain in mass per
          length cubed. If CIM is not specified, then it is assumed to be zero.
    decay : [double]
        * decay (double) is the rate coefficient for first or zero-order decay
          for the aqueous phase of the immobile domain. A negative value
          indicates solute production. The dimensions of decay for first-order
          decay is one over time. The dimensions of decay for zero-order decay
          is mass per length cubed per time. Decay will have no effect on
          simulation results unless either first- or zero-order decay is
          specified in the options block.
    decay_sorbed : [double]
        * decay_sorbed (double) is the rate coefficient for first or zero-order
          decay for the sorbed phase of the immobile domain. A negative value
          indicates solute production. The dimensions of decay_sorbed for
          first-order decay is one over time. The dimensions of decay_sorbed
          for zero-order decay is mass of solute per mass of aquifer per time.
          If decay_sorbed is not specified and both decay and sorption are
          active, then the program will terminate with an error. decay_sorbed
          will have no effect on simulation results unless the SORPTION keyword
          and either first- or zero-order decay are specified in the options
          block.
    bulk_density : [double]
        * bulk_density (double) is the bulk density of this immobile domain in
          mass per length cubed. Bulk density is defined as the immobile domain
          solid mass per volume of the immobile domain. bulk_density is not
          required unless the SORPTION keyword is specified in the options
          block. If the SORPTION keyword is not specified in the options block,
          bulk_density will have no effect on simulation results.
    distcoef : [double]
        * distcoef (double) is the distribution coefficient for the
          equilibrium-controlled linear sorption isotherm in dimensions of
          length cubed per mass. distcoef is not required unless the SORPTION
          keyword is specified in the options block. If the SORPTION keyword is
          not specified in the options block, distcoef will have no effect on
          simulation results.
    sp2 : [double]
        * sp2 (double) is the exponent for the Freundlich isotherm and the
          sorption capacity for the Langmuir isotherm. sp2 is not required
          unless the SORPTION keyword is specified in the options block and
          sorption is specified as FREUNDLICH or LANGMUIR. If the SORPTION
          keyword is not specified in the options block, or if sorption is
          specified as LINEAR, sp2 will have no effect on simulation results.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    budget_filerecord = ListTemplateGenerator(
        ("gwt6", "ist", "options", "budget_filerecord")
    )
    budgetcsv_filerecord = ListTemplateGenerator(
        ("gwt6", "ist", "options", "budgetcsv_filerecord")
    )
    cim_filerecord = ListTemplateGenerator(("gwt6", "ist", "options", "cim_filerecord"))
    cimprintrecord = ListTemplateGenerator(("gwt6", "ist", "options", "cimprintrecord"))
    sorbate_filerecord = ListTemplateGenerator(
        ("gwt6", "ist", "options", "sorbate_filerecord")
    )
    porosity = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "porosity"))
    volfrac = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "volfrac"))
    zetaim = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "zetaim"))
    cim = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "cim"))
    decay = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "decay"))
    decay_sorbed = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "decay_sorbed"))
    bulk_density = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "bulk_density"))
    distcoef = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "distcoef"))
    sp2 = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "sp2"))
    package_abbr = "gwtist"
    _package_type = "ist"
    dfn_file_name = "gwt-ist.dfn"

    dfn = [
        [
            "header",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name budget_filerecord",
            "type record budget fileout budgetfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budget",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name fileout",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name budgetfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name budgetcsv_filerecord",
            "type record budgetcsv fileout budgetcsvfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name budgetcsv",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name budgetcsvfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
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
            "name cim_filerecord",
            "type record cim fileout cimfile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name cim",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name cimfile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name cimprintrecord",
            "type record cim print_format formatrecord",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_format",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name formatrecord",
            "type record columns width digits format",
            "shape",
            "in_record true",
            "reader urword",
            "tagged",
            "optional false",
        ],
        [
            "block options",
            "name columns",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name width",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name digits",
            "type integer",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional",
        ],
        [
            "block options",
            "name format",
            "type string",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
        ],
        [
            "block options",
            "name sorbate_filerecord",
            "type record sorbate fileout sorbatefile",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
        ],
        [
            "block options",
            "name sorbate",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name sorbatefile",
            "type string",
            "preserve_case true",
            "shape",
            "in_record true",
            "reader urword",
            "tagged false",
            "optional false",
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
            "name volfrac",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block griddata",
            "name zetaim",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block griddata",
            "name cim",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "optional true",
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
            "optional true",
            "layered true",
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
        budget_filerecord=None,
        budgetcsv_filerecord=None,
        sorption=None,
        first_order_decay=None,
        zero_order_decay=None,
        cim_filerecord=None,
        cimprintrecord=None,
        sorbate_filerecord=None,
        porosity=None,
        volfrac=None,
        zetaim=None,
        cim=None,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distcoef=None,
        sp2=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "ist", filename, pname, loading_package, **kwargs)

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.budget_filerecord = self.build_mfdata(
            "budget_filerecord", budget_filerecord
        )
        self.budgetcsv_filerecord = self.build_mfdata(
            "budgetcsv_filerecord", budgetcsv_filerecord
        )
        self.sorption = self.build_mfdata("sorption", sorption)
        self.first_order_decay = self.build_mfdata(
            "first_order_decay", first_order_decay
        )
        self.zero_order_decay = self.build_mfdata("zero_order_decay", zero_order_decay)
        self.cim_filerecord = self.build_mfdata("cim_filerecord", cim_filerecord)
        self.cimprintrecord = self.build_mfdata("cimprintrecord", cimprintrecord)
        self.sorbate_filerecord = self.build_mfdata(
            "sorbate_filerecord", sorbate_filerecord
        )
        self.porosity = self.build_mfdata("porosity", porosity)
        self.volfrac = self.build_mfdata("volfrac", volfrac)
        self.zetaim = self.build_mfdata("zetaim", zetaim)
        self.cim = self.build_mfdata("cim", cim)
        self.decay = self.build_mfdata("decay", decay)
        self.decay_sorbed = self.build_mfdata("decay_sorbed", decay_sorbed)
        self.bulk_density = self.build_mfdata("bulk_density", bulk_density)
        self.distcoef = self.build_mfdata("distcoef", distcoef)
        self.sp2 = self.build_mfdata("sp2", sp2)
        self._init_complete = True
