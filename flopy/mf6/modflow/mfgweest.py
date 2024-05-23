# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on May 23, 2024 14:30:07 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowGweest(mfpackage.MFPackage):
    """
    ModflowGweest defines a est package within a gwe6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that EST flow terms will be
          written to the file specified with "BUDGET FILEOUT" in Output
          Control.
    zero_order_decay : boolean
        * zero_order_decay (boolean) is a text keyword to indicate that zero-
          order decay will occur. Use of this keyword requires that DECAY and
          DECAY_SORBED (if sorption is active) are specified in the GRIDDATA
          block.
    latent_heat_vaporization : boolean
        * latent_heat_vaporization (boolean) is a text keyword to indicate that
          cooling associated with evaporation will occur. Use of this keyword
          requires that LATHEATVAP are specified in the GRIDDATA block. While
          the EST package does not simulate evaporation, multiple other
          packages in a GWE simulation may. For example, evaporation may occur
          from the surface of streams or lakes. Owing to the energy consumed by
          the change in phase, the latent heat of vaporization is required.
    porosity : [double]
        * porosity (double) is the mobile domain porosity, defined as the
          mobile domain pore volume per mobile domain volume. The GWE model
          does not support the concept of an immobile domain in the context of
          heat transport.
    decay : [double]
        * decay (double) is the rate coefficient for zero-order decay for the
          aqueous phase of the mobile domain. A negative value indicates heat
          (energy) production. The dimensions of decay for zero-order decay is
          energy per length cubed per time. Zero-order decay will have no
          effect on simulation results unless zero-order decay is specified in
          the options block.
    cps : [double]
        * cps (double) is the mass-based heat capacity of dry solids (aquifer
          material). For example, units of J/kg/C may be used (or equivalent).
    rhos : [double]
        * rhos (double) is a user-specified value of the density of aquifer
          material not considering the voids. Value will remain fixed for the
          entire simulation. For example, if working in SI units, values may be
          entered as kilograms per cubic meter.
    packagedata : [cpw, rhow, latheatvap]
        * cpw (double) is the mass-based heat capacity of the simulated fluid.
          For example, units of J/kg/C may be used (or equivalent).
        * rhow (double) is a user-specified value of the density of water.
          Value will remain fixed for the entire simulation. For example, if
          working in SI units, values may be entered as kilograms per cubic
          meter.
        * latheatvap (double) is the user-specified value for the latent heat
          of vaporization. For example, if working in SI units, values may be
          entered as kJ/kg.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    porosity = ArrayTemplateGenerator(("gwe6", "est", "griddata", "porosity"))
    decay = ArrayTemplateGenerator(("gwe6", "est", "griddata", "decay"))
    cps = ArrayTemplateGenerator(("gwe6", "est", "griddata", "cps"))
    rhos = ArrayTemplateGenerator(("gwe6", "est", "griddata", "rhos"))
    packagedata = ListTemplateGenerator(
        ("gwe6", "est", "packagedata", "packagedata")
    )
    package_abbr = "gweest"
    _package_type = "est"
    dfn_file_name = "gwe-est.dfn"

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
            "name zero_order_decay",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name latent_heat_vaporization",
            "type keyword",
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
            "name cps",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block griddata",
            "name rhos",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
        ],
        [
            "block packagedata",
            "name packagedata",
            "type recarray cpw rhow latheatvap",
            "shape",
            "reader urword",
        ],
        [
            "block packagedata",
            "name cpw",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name rhow",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block packagedata",
            "name latheatvap",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        save_flows=None,
        zero_order_decay=None,
        latent_heat_vaporization=None,
        porosity=None,
        decay=None,
        cps=None,
        rhos=None,
        packagedata=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "est", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.zero_order_decay = self.build_mfdata(
            "zero_order_decay", zero_order_decay
        )
        self.latent_heat_vaporization = self.build_mfdata(
            "latent_heat_vaporization", latent_heat_vaporization
        )
        self.porosity = self.build_mfdata("porosity", porosity)
        self.decay = self.build_mfdata("decay", decay)
        self.cps = self.build_mfdata("cps", cps)
        self.rhos = self.build_mfdata("rhos", rhos)
        self.packagedata = self.build_mfdata("packagedata", packagedata)
        self._init_complete = True
