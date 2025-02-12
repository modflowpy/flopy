# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 11, 2025 01:24:12 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator


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
    zero_order_decay_water : boolean
        * zero_order_decay_water (boolean) is a text keyword to indicate that
          zero-order decay will occur in the aqueous phase. That is, decay
          occurs in the water and is a rate per volume of water only, not per
          volume of aquifer (i.e., grid cell). Use of this keyword requires
          that DECAY_WATER is specified in the GRIDDATA block.
    zero_order_decay_solid : boolean
        * zero_order_decay_solid (boolean) is a text keyword to indicate that
          zero-order decay will occur in the solid phase. That is, decay occurs
          in the solid and is a rate per mass (not volume) of solid only. Use
          of this keyword requires that DECAY_SOLID is specified in the
          GRIDDATA block.
    density_water : double
        * density_water (double) density of water used by calculations related
          to heat storage and conduction. This value is set to 1,000 kg/m3 if
          no overriding value is specified. A user-specified value should be
          provided for models that use units other than kilograms and meters or
          if it is necessary to use a value other than the default.
    heat_capacity_water : double
        * heat_capacity_water (double) heat capacity of water used by
          calculations related to heat storage and conduction. This value is
          set to 4,184 J/kg/C if no overriding value is specified. A user-
          specified value should be provided for models that use units other
          than kilograms, joules, and degrees Celsius or it is necessary to use
          a value other than the default.
    latent_heat_vaporization : double
        * latent_heat_vaporization (double) latent heat of vaporization is the
          amount of energy that is required to convert a given quantity of
          liquid into a gas and is associated with evaporative cooling. While
          the EST package does not simulate evaporation, multiple other
          packages in a GWE simulation may. To avoid having to specify the
          latent heat of vaporization in multiple packages, it is specified in
          a single location and accessed wherever it is needed. For example,
          evaporation may occur from the surface of streams or lakes and the
          energy consumed by the change in phase would be needed in both the
          SFE and LKE packages. This value is set to 2,453,500 J/kg if no
          overriding value is specified. A user-specified value should be
          provided for models that use units other than joules and kilograms or
          if it is necessary to use a value other than the default.
    porosity : [double]
        * porosity (double) is the mobile domain porosity, defined as the
          mobile domain pore volume per mobile domain volume. The GWE model
          does not support the concept of an immobile domain in the context of
          heat transport.
    decay_water : [double]
        * decay_water (double) is the rate coefficient for zero-order decay for
          the aqueous phase of the mobile domain. A negative value indicates
          heat (energy) production. The dimensions of zero-order decay in the
          aqueous phase are energy per length cubed (volume of water) per time.
          Zero-order decay in the aqueous phase will have no effect on
          simulation results unless ZERO_ORDER_DECAY_WATER is specified in the
          options block.
    decay_solid : [double]
        * decay_solid (double) is the rate coefficient for zero-order decay for
          the solid phase. A negative value indicates heat (energy) production.
          The dimensions of zero-order decay in the solid phase are energy per
          mass of solid per time. Zero-order decay in the solid phase will have
          no effect on simulation results unless ZERO_ORDER_DECAY_SOLID is
          specified in the options block.
    heat_capacity_solid : [double]
        * heat_capacity_solid (double) is the mass-based heat capacity of dry
          solids (aquifer material). For example, units of J/kg/C may be used
          (or equivalent).
    density_solid : [double]
        * density_solid (double) is a user-specified value of the density of
          aquifer material not considering the voids. Value will remain fixed
          for the entire simulation. For example, if working in SI units,
          values may be entered as kilograms per cubic meter.
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
    decay_water = ArrayTemplateGenerator(("gwe6", "est", "griddata", "decay_water"))
    decay_solid = ArrayTemplateGenerator(("gwe6", "est", "griddata", "decay_solid"))
    heat_capacity_solid = ArrayTemplateGenerator(
        ("gwe6", "est", "griddata", "heat_capacity_solid")
    )
    density_solid = ArrayTemplateGenerator(("gwe6", "est", "griddata", "density_solid"))
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
            "name zero_order_decay_water",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name zero_order_decay_solid",
            "type keyword",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name density_water",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 1000.0",
            "mf6internal rhow",
        ],
        [
            "block options",
            "name heat_capacity_water",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 4184.0",
            "mf6internal cpw",
        ],
        [
            "block options",
            "name latent_heat_vaporization",
            "type double precision",
            "reader urword",
            "optional true",
            "default_value 2453500.0",
            "mf6internal latheatvap",
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
            "name decay_water",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name decay_solid",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "optional true",
        ],
        [
            "block griddata",
            "name heat_capacity_solid",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "mf6internal cps",
        ],
        [
            "block griddata",
            "name density_solid",
            "type double precision",
            "shape (nodes)",
            "reader readarray",
            "layered true",
            "mf6internal rhos",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        save_flows=None,
        zero_order_decay_water=None,
        zero_order_decay_solid=None,
        density_water=1000.0,
        heat_capacity_water=4184.0,
        latent_heat_vaporization=2453500.0,
        porosity=None,
        decay_water=None,
        decay_solid=None,
        heat_capacity_solid=None,
        density_solid=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(model, "est", filename, pname, loading_package, **kwargs)

        # set up variables
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self.zero_order_decay_water = self.build_mfdata(
            "zero_order_decay_water", zero_order_decay_water
        )
        self.zero_order_decay_solid = self.build_mfdata(
            "zero_order_decay_solid", zero_order_decay_solid
        )
        self.density_water = self.build_mfdata("density_water", density_water)
        self.heat_capacity_water = self.build_mfdata(
            "heat_capacity_water", heat_capacity_water
        )
        self.latent_heat_vaporization = self.build_mfdata(
            "latent_heat_vaporization", latent_heat_vaporization
        )
        self.porosity = self.build_mfdata("porosity", porosity)
        self.decay_water = self.build_mfdata("decay_water", decay_water)
        self.decay_solid = self.build_mfdata("decay_solid", decay_solid)
        self.heat_capacity_solid = self.build_mfdata(
            "heat_capacity_solid", heat_capacity_solid
        )
        self.density_solid = self.build_mfdata("density_solid", density_solid)
        self._init_complete = True
