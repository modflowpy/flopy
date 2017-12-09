from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowGwfnpf(mfpackage.MFPackage):
    """
    ModflowGwfnpf defines a npf package within a gwf6 model.

    Attributes
    ----------
    save_flows : (save_flows : boolean)
        save_flows : keyword to indicate that cell-by-cell flow terms will be
          written to the file specified with ``BUDGET SAVE FILE'' in Output
          Control.
    alternative_cell_averaging : (alternative_cell_averaging : string)
        alternative_cell_averaging : is a text keyword to indicate that an
          alternative method will be used for calculating the conductance for
          horizontal cell connections. The text value for
          alternative\_cell\_averaging can be ``LOGARITHMIC'', ``AMT-
          LMK'', or ``AMT-HMK''. ``AMT-LMK'' signifies that the conductance
          will be calculated using arithmetic-mean thickness and logarithmic-
          mean hydraulic conductivity. ``AMT-HMK'' signifies that the
          conductance will be calculated using arithmetic-mean thickness and
          harmonic-mean hydraulic conductivity. If the user does not specify a
          value for alternative\_cell\_averaging, then the harmonic-
          mean method will be used. This option cannot be used if the XT3D
          option is invoked.
    thickstrt : (thickstrt : boolean)
        thickstrt : indicates that cells having a negative icelltype
          are confined, and their cell thickness for conductance calculations
          will be computed as STRT-BOT rather than TOP-BOT.
    cvoptions : [(dewatered : string)]
        dewatered : If the DEWATERED keyword is specified, then the vertical
          conductance is calculated using only the saturated thickness and
          properties of the overlying cell if the head in the underlying cell
          is below its top.
    perched : (perched : boolean)
        perched : keyword to indicate that when a cell is overlying a dewatered
          convertible cell, the head difference used in Darcy's Law is equal to
          the head in the overlying cell minus the bottom elevation of the
          overlying cell. If not specified, then the default is to use the head
          difference between the two cells.
    rewet_record : [(wetfct : double), (iwetit : integer), (ihdwet : integer)]
        wetfct : is a keyword and factor that is included in the calculation of
          the head that is initially established at a cell when that cell is
          converted from dry to wet.
        iwetit : is a keyword and iteration interval for attempting to wet
          cells. Wetting is attempted every iwetit iteration. This
          applies to outer iterations and not inner iterations. If
          iwetit is specified as zero or less, then the value is
          changed to 1.
        ihdwet : is a keyword and integer flag that determines which equation
          is used to define the initial head at cells that become wet. If
          ihdwet is 0, $h = BOT + WETFCT (hm - BOT)$. If
          ihdwet is not 0, $h = BOT + WETFCT (THRESH)$.
    xt3doptions : [(rhs : string)]
        rhs : If the RHS keyword is also included, then the XT3D
          additional terms will be added to the right-hand side. If the
          RHS keyword is excluded, then the XT3D terms will be put
          into the coefficient matrix.
    save_specific_discharge : (save_specific_discharge : boolean)
        save_specific_discharge : keyword to indicate that x, y, and z
          components of specific discharge will be calculated at cell centers
          and written to the cell-by-cell flow file, which is specified with
          ``BUDGET SAVE FILE'' in Output Control.
    icelltype : [(icelltype : integer)]
        icelltype : flag for each cell that specifies how saturated thickness
          is treated. 0 means saturated thickness is held constant; $>$0 means
          saturated thickness varies with computed head when head is below the
          cell top; $<$0 means saturated thickness varies with computed head
          unless the THICKSTRT option is in effect. When THICKSTRT is in
          effect, a negative value of icelltype indicates that saturated
          thickness will be computed as STRT-BOT and held constant.
    k : [(k : double)]
        k : is the hydraulic conductivity. For the common case in which the
          user would like to specify the horizontal hydraulic conductivity and
          the vertical hydraulic conductivity, then K should be
          assigned as the horizontal hydraulic conductivity, K33
          should be assigned as the vertical hydraulic conductivity, and
          texttt{K22 and the three rotation angles should not be specified.
          When more sophisticated anisotropy is required, then K
          corresponds to the $K_{11$ hydraulic conductivity axis. All included
          cells ($IDOMAIN > 0$) must have a K value greater
          than zero.
    k22 : [(k22 : double)]
        k22 : is the hydraulic conductivity of the second ellipsoid axis; for
          an unrotated case this is the hydraulic conductivity in the y
          direction. If K22 is not included in the GRIDDATA block,
          then K22 is set equal to $K$. For a regular MODFLOW grid
          (DIS Package is used) in which no rotation angles are specified,
          K22 is the hydraulic conductivity along columns in the y
          direction. For an unstructured DISU grid, the user must assign
          principal x and y axes and provide the angle for each cell face
          relative to the assigned x direction. All included cells
          ($IDOMAIN > 0$) must have a K22 value greater than
          zero.
    k33 : [(k33 : double)]
        k33 : is the hydraulic conductivity of the third ellipsoid axis; for an
          unrotated case, this is the vertical hydraulic conductivity. When
          anisotropy is applied, K33 corresponds to the $K_{33$
          tensor component. All included cells ($IDOMAIN > 0$) must
          have a K33 value greater than zero.
    angle1 : [(angle1 : double)]
        angle1 : is a rotation angle of the hydraulic conductivity tensor in
          degrees. The angle represents the first of three sequential rotations
          of the hydraulic conductivity ellipsoid. With the $K_{11$, $K_{22$,
          and $K_{33$ axes of the ellipsoid initially aligned with the x, y,
          and z coordinate axes, respectively, angle1 rotates the
          ellipsoid about its $K_{33$ axis (within the x - y plane). A
          positive value represents counter-clockwise rotation when viewed from
          any point on the positive $K_{33$ axis, looking toward the center of
          the ellipsoid. A value of zero indicates that the $K_{11$ axis lies
          within the x - z plane. If angle1 is not specified, default
          values of zero are assigned to angle1, angle2, and
          angle3, in which case the $K_{11$, $K_{22$, and $K_{33$
          axes are aligned with the x, y, and z axes, respectively.
    angle2 : [(angle2 : double)]
        angle2 : is a rotation angle of the hydraulic conductivity tensor in
          degrees. The angle represents the second of three sequential
          rotations of the hydraulic conductivity ellipsoid. Following the
          rotation by angle1 described above, angle2 rotates
          the ellipsoid about its $K_{22$ axis (out of the x - y plane). An
          array can be specified for angle2 only if angle1 is
          also specified. A positive value of angle2 represents
          clockwise rotation when viewed from any point on the positive
          $K_{22$ axis, looking toward the center of the ellipsoid. A value of
          zero indicates that the $K_{11$ axis lies within the x - y plane. If
          angle2 is not specified, default values of zero are assigned
          to angle2 and angle3; connections that are not
          user-designated as vertical are assumed to be strictly horizontal
          (that is, to have no z component to their orientation); and
          connection lengths are based on horizontal distances.
    angle3 : [(angle3 : double)]
        angle3 : is a rotation angle of the hydraulic conductivity tensor in
          degrees. The angle represents the third of three sequential rotations
          of the hydraulic conductivity ellipsoid. Following the rotations by
          angle1 and angle2 described above, angle3
          rotates the ellipsoid about its $K_{11$ axis. An array can be
          specified for angle3 only if angle1 and
          angle2 are also specified. An array must be specified for
          angle3 if angle2 is specified. A positive value of
          angle3 represents clockwise rotation when viewed from any
          point on the positive $K_{11$ axis, looking toward the center of the
          ellipsoid. A value of zero indicates that the $K_{22$ axis lies
          within the x - y plane.
    wetdry : [(wetdry : double)]
        wetdry : is a combination of the wetting threshold and a flag to
          indicate which neighboring cells can cause a cell to become wet. If
          wetdry $<$ 0, only a cell below a dry cell can cause the
          cell to become wet. If wetdry $>$ 0, the cell below a dry
          cell and horizontally adjacent cells can cause a cell to become wet.
          If wetdry is 0, the cell cannot be wetted. The absolute
          value of wetdry is the wetting threshold. When the sum of
          BOT and the absolute value of wetdry at a dry cell is
          equaled or exceeded by the head at an adjacent cell, the cell is
          wetted. wetdry must be specified if ``REWET'' is specified
          in the OPTIONS block. If ``REWET'' is not specified in the options
          block, then wetdry can be entered, and memory will be
          allocated for it, even though it is not used.

    """
    rewet_record = ListTemplateGenerator(('gwf6', 'npf', 'options', 
                                          'rewet_record'))
    icelltype = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 
                                        'icelltype'))
    k = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 'k'))
    k22 = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 'k22'))
    k33 = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 'k33'))
    angle1 = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 
                                     'angle1'))
    angle2 = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 
                                     'angle2'))
    angle3 = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 
                                     'angle3'))
    wetdry = ArrayTemplateGenerator(('gwf6', 'npf', 'griddata', 
                                     'wetdry'))
    package_abbr = "gwfnpf"

    def __init__(self, model, add_to_package_list=True, save_flows=None,
                 alternative_cell_averaging=None, thickstrt=None,
                 cvoptions=None, perched=None, rewet_record=None,
                 xt3doptions=None, save_specific_discharge=None,
                 icelltype=None, k=None, k22=None, k33=None, angle1=None,
                 angle2=None, angle3=None, wetdry=None, fname=None, pname=None,
                 parent_file=None):
        super(ModflowGwfnpf, self).__init__(model, "npf", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.save_flows = self.build_mfdata("save_flows",  save_flows)
        self.alternative_cell_averaging = self.build_mfdata(
            "alternative_cell_averaging",  alternative_cell_averaging)
        self.thickstrt = self.build_mfdata("thickstrt",  thickstrt)
        self.cvoptions = self.build_mfdata("cvoptions",  cvoptions)
        self.perched = self.build_mfdata("perched",  perched)
        self.rewet_record = self.build_mfdata("rewet_record",  rewet_record)
        self.xt3doptions = self.build_mfdata("xt3doptions",  xt3doptions)
        self.save_specific_discharge = self.build_mfdata(
            "save_specific_discharge",  save_specific_discharge)
        self.icelltype = self.build_mfdata("icelltype",  icelltype)
        self.k = self.build_mfdata("k",  k)
        self.k22 = self.build_mfdata("k22",  k22)
        self.k33 = self.build_mfdata("k33",  k33)
        self.angle1 = self.build_mfdata("angle1",  angle1)
        self.angle2 = self.build_mfdata("angle2",  angle2)
        self.angle3 = self.build_mfdata("angle3",  angle3)
        self.wetdry = self.build_mfdata("wetdry",  wetdry)
