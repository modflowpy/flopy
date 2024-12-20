# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on December 20, 2024 02:43:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ArrayTemplateGenerator, ListTemplateGenerator


class ModflowUtlncf(mfpackage.MFPackage):
    """
    ModflowUtlncf defines a ncf package within a utl model.

    Parameters
    ----------
    parent_package : MFPackage
        Parent_package that this package is a part of. Package is automatically
        added to parent_package when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    wkt : [string]
        * wkt (string) is the CRS well-known text (WKT) string.
    deflate : integer
        * deflate (integer) is the variable deflate level (0=min, 9=max) in the
          netcdf file. Defining this parameter activates per-variable
          compression at the level specified.
    shuffle : boolean
        * shuffle (boolean) is the keyword used to turn on the netcdf variable
          shuffle filter when the deflate option is also set. The shuffle
          filter has the effect of storing the first byte of all of a
          variable's values in a chunk contiguously, followed by all the second
          bytes, etc. This can be an optimization for compression with certain
          types of data.
    chunk_time : integer
        * chunk_time (integer) is the keyword used to provide a data chunk size
          for the time dimension in a NETCDF_MESH2D or NETCDF_STRUCTURED output
          file. Must be used in combination with the the chunk_face parameter
          (NETCDF_MESH2D) or the chunk_z, chunk_y, and chunk_x parameter set
          (NETCDF_STRUCTURED) to have an effect.
    chunk_face : integer
        * chunk_face (integer) is the keyword used to provide a data chunk size
          for the face dimension in a NETCDF_MESH2D output file. Must be used
          in combination with the the chunk_time parameter to have an effect.
    chunk_z : integer
        * chunk_z (integer) is the keyword used to provide a data chunk size
          for the z dimension in a NETCDF_STRUCTURED output file. Must be used
          in combination with the the chunk_time, chunk_x and chunk_y parameter
          set to have an effect.
    chunk_y : integer
        * chunk_y (integer) is the keyword used to provide a data chunk size
          for the y dimension in a NETCDF_STRUCTURED output file. Must be used
          in combination with the the chunk_time, chunk_x and chunk_z parameter
          set to have an effect.
    chunk_x : integer
        * chunk_x (integer) is the keyword used to provide a data chunk size
          for the x dimension in a NETCDF_STRUCTURED output file. Must be used
          in combination with the the chunk_time, chunk_y and chunk_z parameter
          set to have an effect.
    modflow6_attr_off : boolean
        * modflow6_attr_off (boolean) is the keyword used to turn off internal
          input tagging in the model netcdf file. Tagging adds internal modflow
          6 attribute(s) to variables which facilitate identification.
          Currently this applies to gridded arrays.
    ncpl : integer
        * ncpl (integer) is the number of cells in a projected plane layer.
    latitude : [double]
        * latitude (double) cell center latitude.
    longitude : [double]
        * longitude (double) cell center longitude.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    wkt = ListTemplateGenerator(('ncf', 'options', 'wkt'))
    latitude = ArrayTemplateGenerator(('ncf', 'griddata', 'latitude'))
    longitude = ArrayTemplateGenerator(('ncf', 'griddata', 'longitude'))
    package_abbr = "utlncf"
    _package_type = "ncf"
    dfn_file_name = "utl-ncf.dfn"

    dfn = [
           ["header", ],
           ["block options", "name wkt", "type string", "shape lenbigline",
            "reader urword", "optional true"],
           ["block options", "name deflate", "type integer",
            "reader urword", "optional true"],
           ["block options", "name shuffle", "type keyword",
            "reader urword", "optional true"],
           ["block options", "name chunk_time", "type integer",
            "reader urword", "optional true"],
           ["block options", "name chunk_face", "type integer",
            "reader urword", "optional true"],
           ["block options", "name chunk_z", "type integer",
            "reader urword", "optional true"],
           ["block options", "name chunk_y", "type integer",
            "reader urword", "optional true"],
           ["block options", "name chunk_x", "type integer",
            "reader urword", "optional true"],
           ["block options", "name modflow6_attr_off", "type keyword",
            "reader urword", "optional true", "mf6internal attr_off"],
           ["block dimensions", "name ncpl", "type integer",
            "optional true", "reader urword"],
           ["block griddata", "name latitude", "type double precision",
            "shape (ncpl)", "optional true", "reader readarray"],
           ["block griddata", "name longitude", "type double precision",
            "shape (ncpl)", "optional true", "reader readarray"]]

    def __init__(self, parent_package, loading_package=False, wkt=None,
                 deflate=None, shuffle=None, chunk_time=None, chunk_face=None,
                 chunk_z=None, chunk_y=None, chunk_x=None,
                 modflow6_attr_off=None, ncpl=None, latitude=None,
                 longitude=None, filename=None, pname=None, **kwargs):
        super().__init__(parent_package, "ncf", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.wkt = self.build_mfdata("wkt", wkt)
        self.deflate = self.build_mfdata("deflate", deflate)
        self.shuffle = self.build_mfdata("shuffle", shuffle)
        self.chunk_time = self.build_mfdata("chunk_time", chunk_time)
        self.chunk_face = self.build_mfdata("chunk_face", chunk_face)
        self.chunk_z = self.build_mfdata("chunk_z", chunk_z)
        self.chunk_y = self.build_mfdata("chunk_y", chunk_y)
        self.chunk_x = self.build_mfdata("chunk_x", chunk_x)
        self.modflow6_attr_off = self.build_mfdata("modflow6_attr_off",
                                                   modflow6_attr_off)
        self.ncpl = self.build_mfdata("ncpl", ncpl)
        self.latitude = self.build_mfdata("latitude", latitude)
        self.longitude = self.build_mfdata("longitude", longitude)
        self._init_complete = True


class UtlncfPackages(mfpackage.MFChildPackages):
    """
    UtlncfPackages is a container class for the ModflowUtlncf class.

    Methods
    -------
    initialize
        Initializes a new ModflowUtlncf package removing any sibling child
        packages attached to the same parent package. See ModflowUtlncf init
        documentation for definition of parameters.
    append_package
        Adds a new ModflowUtlncf package to the container. See ModflowUtlncf
        init documentation for definition of parameters.
    """
    package_abbr = "utlncfpackages"

    def initialize(self, wkt=None, deflate=None, shuffle=None, chunk_time=None,
                   chunk_face=None, chunk_z=None, chunk_y=None, chunk_x=None,
                   modflow6_attr_off=None, ncpl=None, latitude=None,
                   longitude=None, filename=None, pname=None):
        new_package = ModflowUtlncf(self._cpparent, wkt=wkt, deflate=deflate,
                                    shuffle=shuffle, chunk_time=chunk_time,
                                    chunk_face=chunk_face, chunk_z=chunk_z,
                                    chunk_y=chunk_y, chunk_x=chunk_x,
                                    modflow6_attr_off=modflow6_attr_off,
                                    ncpl=ncpl, latitude=latitude,
                                    longitude=longitude, filename=filename,
                                    pname=pname, child_builder_call=True)
        self.init_package(new_package, filename)

    def append_package(self, wkt=None, deflate=None, shuffle=None,
                   chunk_time=None, chunk_face=None, chunk_z=None,
                   chunk_y=None, chunk_x=None, modflow6_attr_off=None,
                   ncpl=None, latitude=None, longitude=None, filename=None,
                   pname=None):
        new_package = ModflowUtlncf(self._cpparent, wkt=wkt, deflate=deflate,
                                    shuffle=shuffle, chunk_time=chunk_time,
                                    chunk_face=chunk_face, chunk_z=chunk_z,
                                    chunk_y=chunk_y, chunk_x=chunk_x,
                                    modflow6_attr_off=modflow6_attr_off,
                                    ncpl=ncpl, latitude=latitude,
                                    longitude=longitude, filename=filename,
                                    pname=pname, child_builder_call=True)
        self._append_package(new_package, filename)
