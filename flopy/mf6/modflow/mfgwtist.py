# autogenerated file, do not modify

from os import PathLike, curdir
from typing import Union

from flopy.mf6.data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator
from flopy.mf6.mfpackage import MFChildPackages, MFPackage


class ModflowGwtist(MFPackage):
    """
    ModflowGwtist defines a IST package.

    Parameters
    ----------
    save_flows : keyword
        keyword to indicate that ist flow terms will be written to the file specified
        with 'budget fileout' in output control.
    budget_filerecord : (budgetfile)
        * budgetfile : string
                name of the binary output file to write budget information.

    budgetcsv_filerecord : (budgetcsvfile)
        * budgetcsvfile : string
                name of the comma-separated value (CSV) output file to write budget summary
                information.  A budget summary record will be written to this file for each
                time step of the simulation.

    sorption : string
        is a text keyword to indicate that sorption will be activated.  valid sorption
        options include linear, freundlich, and langmuir.  use of this keyword requires
        that bulk_density and distcoef are specified in the griddata block.  if
        sorption is specified as freundlich or langmuir then sp2 is also required in
        the griddata block.  the sorption option must be consistent with the sorption
        option specified in the mst package or the program will terminate with an
        error.
    first_order_decay : keyword
        is a text keyword to indicate that first-order decay will occur.  use of this
        keyword requires that decay and decay_sorbed (if sorption is active) are
        specified in the griddata block.
    zero_order_decay : keyword
        is a text keyword to indicate that zero-order decay will occur.  use of this
        keyword requires that decay and decay_sorbed (if sorption is active) are
        specified in the griddata block.
    cim_filerecord : record
    cimprintrecord : (print_format)
        * print_format : keyword
                keyword to specify format for printing to the listing file.

    sorbate_filerecord : (sorbatefile)
        * sorbatefile : string
                name of the output file to write immobile sorbate concentration information.
                Immobile sorbate concentrations will be written whenever aqueous immobile
                concentrations are saved, as determined by settings in the Output Control
                option.

    porosity : [double precision]
        porosity of the immobile domain specified as the immobile domain pore volume
        per immobile domain volume.
    volfrac : [double precision]
        fraction of the cell volume that consists of this immobile domain.  the sum of
        all immobile domain volume fractions must be less than one.
    zetaim : [double precision]
        mass transfer rate coefficient between the mobile and immobile domains, in
        dimensions of per time.
    cim : [double precision]
        initial concentration of the immobile domain in mass per length cubed.  if cim
        is not specified, then it is assumed to be zero.
    decay : [double precision]
        is the rate coefficient for first or zero-order decay for the aqueous phase of
        the immobile domain.  a negative value indicates solute production.  the
        dimensions of decay for first-order decay is one over time.  the dimensions of
        decay for zero-order decay is mass per length cubed per time.  decay will have
        no effect on simulation results unless either first- or zero-order decay is
        specified in the options block.
    decay_sorbed : [double precision]
        is the rate coefficient for first or zero-order decay for the sorbed phase of
        the immobile domain.  a negative value indicates solute production.  the
        dimensions of decay_sorbed for first-order decay is one over time.  the
        dimensions of decay_sorbed for zero-order decay is mass of solute per mass of
        aquifer per time.  if decay_sorbed is not specified and both decay and sorption
        are active, then the program will terminate with an error.  decay_sorbed will
        have no effect on simulation results unless the sorption keyword and either
        first- or zero-order decay are specified in the options block.
    bulk_density : [double precision]
        is the bulk density of this immobile domain in mass per length cubed.  bulk
        density is defined as the immobile domain solid mass per volume of the immobile
        domain.  bulk_density is not required unless the sorption keyword is specified
        in the options block.  if the sorption keyword is not specified in the options
        block, bulk_density will have no effect on simulation results.
    distcoef : [double precision]
        is the distribution coefficient for the equilibrium-controlled linear sorption
        isotherm in dimensions of length cubed per mass.  distcoef is not required
        unless the sorption keyword is specified in the options block.  if the sorption
        keyword is not specified in the options block, distcoef will have no effect on
        simulation results.
    sp2 : [double precision]
        is the exponent for the freundlich isotherm and the sorption capacity for the
        langmuir isotherm.  sp2 is not required unless the sorption keyword is
        specified in the options block and sorption is specified as freundlich or
        langmuir. if the sorption keyword is not specified in the options block, or if
        sorption is specified as linear, sp2 will have no effect on simulation results.

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
        ["header"],
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
        """
        ModflowGwtist defines a IST package.

        Parameters
        ----------
        model
            Model that this package is a part of. Package is automatically
            added to model when it is initialized.
        loading_package : bool
            Do not set this parameter. It is intended for debugging and internal
            processing purposes only.
        save_flows : keyword
            keyword to indicate that ist flow terms will be written to the file specified
            with 'budget fileout' in output control.
        budget_filerecord : record
        budgetcsv_filerecord : record
        sorption : string
            is a text keyword to indicate that sorption will be activated.  valid sorption
            options include linear, freundlich, and langmuir.  use of this keyword requires
            that bulk_density and distcoef are specified in the griddata block.  if
            sorption is specified as freundlich or langmuir then sp2 is also required in
            the griddata block.  the sorption option must be consistent with the sorption
            option specified in the mst package or the program will terminate with an
            error.
        first_order_decay : keyword
            is a text keyword to indicate that first-order decay will occur.  use of this
            keyword requires that decay and decay_sorbed (if sorption is active) are
            specified in the griddata block.
        zero_order_decay : keyword
            is a text keyword to indicate that zero-order decay will occur.  use of this
            keyword requires that decay and decay_sorbed (if sorption is active) are
            specified in the griddata block.
        cim_filerecord : record
        cimprintrecord : (print_format)
            * print_format : keyword
                    keyword to specify format for printing to the listing file.

        sorbate_filerecord : record
        porosity : [double precision]
            porosity of the immobile domain specified as the immobile domain pore volume
            per immobile domain volume.
        volfrac : [double precision]
            fraction of the cell volume that consists of this immobile domain.  the sum of
            all immobile domain volume fractions must be less than one.
        zetaim : [double precision]
            mass transfer rate coefficient between the mobile and immobile domains, in
            dimensions of per time.
        cim : [double precision]
            initial concentration of the immobile domain in mass per length cubed.  if cim
            is not specified, then it is assumed to be zero.
        decay : [double precision]
            is the rate coefficient for first or zero-order decay for the aqueous phase of
            the immobile domain.  a negative value indicates solute production.  the
            dimensions of decay for first-order decay is one over time.  the dimensions of
            decay for zero-order decay is mass per length cubed per time.  decay will have
            no effect on simulation results unless either first- or zero-order decay is
            specified in the options block.
        decay_sorbed : [double precision]
            is the rate coefficient for first or zero-order decay for the sorbed phase of
            the immobile domain.  a negative value indicates solute production.  the
            dimensions of decay_sorbed for first-order decay is one over time.  the
            dimensions of decay_sorbed for zero-order decay is mass of solute per mass of
            aquifer per time.  if decay_sorbed is not specified and both decay and sorption
            are active, then the program will terminate with an error.  decay_sorbed will
            have no effect on simulation results unless the sorption keyword and either
            first- or zero-order decay are specified in the options block.
        bulk_density : [double precision]
            is the bulk density of this immobile domain in mass per length cubed.  bulk
            density is defined as the immobile domain solid mass per volume of the immobile
            domain.  bulk_density is not required unless the sorption keyword is specified
            in the options block.  if the sorption keyword is not specified in the options
            block, bulk_density will have no effect on simulation results.
        distcoef : [double precision]
            is the distribution coefficient for the equilibrium-controlled linear sorption
            isotherm in dimensions of length cubed per mass.  distcoef is not required
            unless the sorption keyword is specified in the options block.  if the sorption
            keyword is not specified in the options block, distcoef will have no effect on
            simulation results.
        sp2 : [double precision]
            is the exponent for the freundlich isotherm and the sorption capacity for the
            langmuir isotherm.  sp2 is not required unless the sorption keyword is
            specified in the options block and sorption is specified as freundlich or
            langmuir. if the sorption keyword is not specified in the options block, or if
            sorption is specified as linear, sp2 will have no effect on simulation results.

        filename : str
            File name for this package.
        pname : str
            Package name for this package.
        parent_file : MFPackage
            Parent package file that references this package. Only needed for
            utility packages (mfutl*). For example, mfutllaktab package must have
            a mfgwflak package parent_file.
        """

        super().__init__(model, "ist", filename, pname, loading_package, **kwargs)

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
