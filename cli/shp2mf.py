#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path
from argparse import RawTextHelpFormatter
from collections import OrderedDict

import mando
import numpy as np

import geopandas as gpd
from geopandas.tools import sjoin

import flopy


def isin(field, shpfilename, columns, error=False):
    if field not in columns:
        if error is True:
            raise ValueError("""
*
*   The field "{0}" is not in the shapefile
*   "{1}"
*   fields:
*   "{2}"
*
""".format(field, shpfilename, columns))
        else:
            return False
    return True


@mando.command(formatter_class=RawTextHelpFormatter)
def well(well_file_name, grid_shp_file, rc_fields, *args):
    """Take information from shapefiles and build a well file.

    :param str well_file_name: The name of the produced well file.  The
        basename of this string will be used as the name of \*.wel file
        written.

    :param str grid_shp_file: The polygon shapefile that represents the MODFLOW
        grid.

    :param str rc_fields: A single, comma separated list of
        "row_field_name,column_field_name".  The row and column field names
        must be in the "grid_shp_file".  No spaces.

    :param args: At least one comma separated list.  You can have several of
        these comma separated lists to import from different shapefiles or
        different stress periods from the same shapefile.  Separate the comma
        separated lists with spaces::

            well_point_shapefile,stress_per,layer,flux,id_field1,id_field2,...
            ...

        After the first two arguments the following are field names either in
        the polygon grid or well point shapefiles.

        The "id_field*" are optional field names whose contents will be
        appended to the line in the "wel" file.  This is useful to identify the
        well.

        The "flux" field name has some additional features to make it easier to
        use existing data that otherwise would have required modification of
        the well point shapefile table.

        The first "flux" field feature is that if prepended by a "-" will
        multiple the contents of the flux field by -1.0.  So if well pumping in
        the "flux" field of the shapefile is positive, using "-flux" will
        change the sign to indicate to MODFLOW that it is a withdrawal.

        Even though the mgd unit is a horror, it has been adopted by the water
        supply community.  The second special feature of the "flux" field is if
        appended with ".mgd_to_cfd" or ".cfs_to_cfd" it will convert units for
        you.

        Summary of "flux" field special behaviors:

        +--------------------------+-----------+-----------+-----------+
        | Command Line             | Shapefile | Shapefile | Shapefile |
        |                          | Pumping   | Injection | Units     |
        +==========================+===========+===========+===========+
        | ...,flux,...             | -         | +         | cfd       |
        +--------------------------+-----------+-----------+-----------+
        | ...,-flux,...            | +         | -         | cfd       |
        +--------------------------+-----------+-----------+-----------+
        | ...,-flux.mgd_to_cfd,... | +         | -         | mgd       |
        +--------------------------+-----------+-----------+-----------+
        | ...,flux.mgd_to_cfd,...  | -         | +         | mgd       |
        +--------------------------+-----------+-----------+-----------+
        | ...,-flux.cfs_to_cfd,... | +         | -         | cfs       |
        +--------------------------+-----------+-----------+-----------+
        | ...,flux.cfs_to_cfd,...  | -         | +         | cfs       |
        +--------------------------+-----------+-----------+-----------+

        """
    # Just need the basename
    basename = os.path.splitext(well_file_name)[0]

    row_field, column_field = rc_fields.split(',')

    mf = flopy.modflow.Modflow(basename)

    grid = gpd.read_file(grid_shp_file)

    _ = isin(row_field, grid_shp_file, grid.columns, error=True)
    _ = isin(column_field, grid_shp_file, grid.columns, error=True)

    stress_period_data = {}
    argsdict = OrderedDict()
    maxlength = 0
    for word in args:
        (pshp_file,
         stress_per,
         layer_id_field,
         flux_field,
         *id_fields) = word.split(",")
        if len(id_fields) > maxlength:
            maxlength = len(id_fields)

        argsdict.setdefault(pshp_file, []).append([stress_per,
                                                   layer_id_field,
                                                   flux_field] +
                                                   id_fields)

    for point_shp_file in argsdict:
        wellpnt = gpd.read_file(point_shp_file)
        pointinpolys = sjoin(wellpnt, grid, how='inner', op='intersects')

        for eachstressper in argsdict[point_shp_file]:
            stress_per, layer_id_field, flux_field, *id_fields = eachstressper

            print()
            print(','.join([point_shp_file] + eachstressper))
            if len(wellpnt) != len(pointinpolys):
                print('    WARNING: There are {0} wells outside of the grid.'.format(len(wellpnt) - len(pointinpolys)))

            # Change sign of flux field
            multi = 1.0
            if flux_field[0] == '-':
                multi = -1.0
                flux_field = flux_field[1:]

            # Change units of flux field
            nfields = flux_field.split('.')
            umulti = 1.0
            if len(nfields) == 2:
                flux_field = nfields[0]
                if nfields[1] == 'mgd_to_cfd':
                    umulti = 133680.56
                elif nfields[1] == 'cfs_to_cfd':
                    umulti = 86400

            for fname in [layer_id_field, flux_field] + id_fields:
                _ = isin(fname, point_shp_file, wellpnt.columns, error=True)

            stress_per = int(stress_per) - 1

            nlayer_id_field = layer_id_field
            nrow_field = row_field
            ncolumn_field = column_field
            nflux_field = flux_field

            if layer_id_field not in pointinpolys.columns:
                print("    Duplicate field '{0}'. Using field '{0}' from '{1}'.".format(layer_id_field, point_shp_file))
                nlayer_id_field = layer_id_field + "_right"
            if row_field not in pointinpolys.columns:
                print("    Duplicate field '{0}'. Using field '{0}' from '{1}'.".format(row_field, grid_shp_file))
                nrow_field = row_field + "_left"
            if column_field not in pointinpolys.columns:
                print("    Duplicate field '{0}'. Using field '{0}' from '{1}'.".format(column_field, grid_shp_file))
                ncolumn_field = column_field + "_left"
            if flux_field not in pointinpolys.columns:
                print("    Duplicate field '{0}'. Using field '{0}' from '{1}'.".format(flux_field, point_shp_file))
                nflux_field = flux_field + "_right"

            collect_id_fields = []
            for i in id_fields:
                if i not in pointinpolys.columns:
                    print("    Duplicate field '{0}'. Using field '{0}' from '{1}'.".format(i, point_shp_file))
                    collect_id_fields.append(i + "_right")
                else:
                    collect_id_fields.append(i)

            for p in pointinpolys.itertuples():

                layer = int(getattr(p, nlayer_id_field)) - 1
                row = int(getattr(p, nrow_field)) - 1
                column = int(getattr(p, ncolumn_field)) - 1
                flux = float(getattr(p, nflux_field))*multi*umulti
                stress_period_data.setdefault(stress_per, []).append([
                    layer,
                    row,
                    column,
                    flux] + [str(getattr(p, i)) for i in collect_id_fields])

    for key in stress_period_data:
        stress_period_data[key] = sorted(stress_period_data[key])
        for line in stress_period_data[key]:
            while len(line) < 4 + maxlength:
                line.append('')

    wel = flopy.modflow.ModflowWel(mf,
                                   stress_period_data=stress_period_data,
                                   dtype=np.dtype([("k", np.int),
                                                   ("i", np.int),
                                                   ("j", np.int),
                                                   ("flux", np.float32)] +
                                                  [('_{0}'.format(i), np.object) for i in range(maxlength)]))
    wel.write_file()


def main():
    """Main function."""
    mando.main()


if __name__ == '__main__':
    main()
