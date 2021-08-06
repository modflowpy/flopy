# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on August 06, 2021 20:57:00 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwtist(mfpackage.MFPackage):
    """
    ModflowGwtist defines a ist package within a gwt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of.  Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that IST flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    sorption : boolean
        * sorption (boolean) is a text keyword to indicate that sorption will
          be activated. Use of this keyword requires that BULK_DENSITY and
          DISTCOEF are specified in the GRIDDATA block. The linear sorption
          isotherm is the only isotherm presently supported in the IST Package.
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
    cim : [double]
        * cim (double) initial concentration of the immobile domain in mass per
          length cubed. If CIM is not specified, then it is assumed to be zero.
    thetaim : [double]
        * thetaim (double) porosity of the immobile domain specified as the
          volume of immobile pore space per total volume (dimensionless).
    zetaim : [double]
        * zetaim (double) mass transfer rate coefficient between the mobile and
          immobile domains, in dimensions of per time.
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
        * bulk_density (double) is the bulk density of the aquifer in mass per
          length cubed. bulk_density will have no effect on simulation results
          unless the SORPTION keyword is specified in the options block.
    distcoef : [double]
        * distcoef (double) is the distribution coefficient for the
          equilibrium-controlled linear sorption isotherm in dimensions of
          length cubed per mass. distcoef will have no effect on simulation
          results unless the SORPTION keyword is specified in the options
          block.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    cim_filerecord = ListTemplateGenerator(
        ("gwt6", "ist", "options", "cim_filerecord")
    )
    cimprintrecord = ListTemplateGenerator(
        ("gwt6", "ist", "options", "cimprintrecord")
    )
    cim = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "cim"))
    thetaim = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "thetaim"))
    zetaim = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "zetaim"))
    decay = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "decay"))
    decay_sorbed = ArrayTemplateGenerator(
        ("gwt6", "ist", "griddata", "decay_sorbed")
    )
    bulk_density = ArrayTemplateGenerator(
        ("gwt6", "ist", "griddata", "bulk_density")
    )
    distcoef = ArrayTemplateGenerator(("gwt6", "ist", "griddata", "distcoef"))
    package_abbr = "gwtist"
    _package_type = "ist"
    dfn_file_name = "gwt-ist.dfn"

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
            "name sorption",
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
            "name thetaim",
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
            "layered true",
        ],
        [
            "block griddata",
            "name distcoef",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        save_flows=None,
        sorption=None,
        first_order_decay=None,
        zero_order_decay=None,
        cim_filerecord=None,
        cimprintrecord=None,
        cim=None,
        thetaim=None,
        zetaim=None,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distcoef=None,
        filename=None,
        pname=None,
        parent_file=None,
    ):
        super().__init__(
            model, "ist", filename, pname, loading_package, parent_file
        )

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.sorption = self.build_mfdata("sorption", sorption)
        self.first_order_decay = self.build_mfdata(
            "first_order_decay", first_order_decay
        )
        self.zero_order_decay = self.build_mfdata(
            "zero_order_decay", zero_order_decay
        )
        self.cim_filerecord = self.build_mfdata(
            "cim_filerecord", cim_filerecord
        )
        self.cimprintrecord = self.build_mfdata(
            "cimprintrecord", cimprintrecord
        )
        self.cim = self.build_mfdata("cim", cim)
        self.thetaim = self.build_mfdata("thetaim", thetaim)
        self.zetaim = self.build_mfdata("zetaim", zetaim)
        self.decay = self.build_mfdata("decay", decay)
        self.decay_sorbed = self.build_mfdata("decay_sorbed", decay_sorbed)
        self.bulk_density = self.build_mfdata("bulk_density", bulk_density)
        self.distcoef = self.build_mfdata("distcoef", distcoef)
        self._init_complete = True
