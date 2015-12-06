from __future__ import print_function
import os
import shutil
import numpy as np
import subprocess


def features_to_shapefile(features, featuretype, filename):
    """
    Write a shapefile for the features of type featuretype.

    Parameters
    ----------
    features : list
        List of point, line, or polygon features
    featuretype : str
        Must be 'point', 'line', or 'polygon'
    filename : string
        name of the shapefile to write

    Returns
    -------
    None

    """
    try:
        import shapefile
    except:
        raise Exception('Erorr importing shapefile: ' +
                        'try pip install pyshp')

    if featuretype.lower() not in ['point', 'line', 'polygon']:
        raise Exception('Unrecognized feature type: {}'.format(featuretype))

    if featuretype.lower() == 'line':
        wr = shapefile.Writer(shapeType=shapefile.POLYLINE)
        wr.field("number", "N", 20, 0)
        for i, line in enumerate(features):
            wr.line(line)
            wr.record(i)

    elif featuretype.lower() == 'point':
        wr = shapefile.Writer(shapeType=shapefile.POINT)
        wr.field("number", "N", 20, 0)
        for i, point in enumerate(features):
            wr.point(point[0], point[1])
            wr.record(i)

    elif featuretype.lower() == 'polygon':
        wr = shapefile.Writer(shapeType=shapefile.POLYGON)
        wr.field("number", "N", 20, 0)
        for i, polygon in enumerate(features):
            wr.poly(polygon)
            wr.record(i)

    wr.save(filename)
    return


class Gridgen(object):
    """
    Class to work with the gridgen program to create layered quadtree grids.

    Parameters
    ----------
    dis : flopy.modflow.ModflowDis
        Flopy discretization object
    exe_name : str
        path and name of the gridgen program. (default is gridgen)

    """
    def __init__(self, dis, model_ws='.', exe_name='gridgen'):
        self.dis = dis
        self.model_ws = model_ws
        self.exe_name = exe_name

        # Set up a blank _active_domain list with None for each layer
        self._addict = {}
        self._active_domain = []
        for k in range(dis.nlay):
            self._active_domain.append(None)

        # Set up a blank _refinement_features list with empty list for
        # each layer
        self._rfdict = {}
        self._refinement_features = []
        for k in range(dis.nlay):
            self._refinement_features.append([])

        return

    def add_active_domain(self, feature, layers):
        """
        Parameters
        ----------
        feature : str or list
            feature can be either a string containing the name of a polygon
            shapefile or it can be a list of polygons
        layers : list
            A list of layers (zero based) for which this active domain
            applies.

        Returns
        -------
        None

        """
        adname = 'ad{}'.format(len(self._addict))
        if isinstance(feature, list):
            # Create a shapefile
            features_to_shapefile(feature, 'polygon', adname)
            shapefile = adname
        else:
            shapefile = feature

        self._addict[adname] = shapefile
        sn = shapefile + '.shp'
        assert os.path.isfile(sn), 'Shapefile does not exist: {}'.format(sn)

        for k in layers:
            self._active_domain[k] = adname

        return

    def add_refinement_features(self, features, featuretype, level, layers):
        """
        Parameters
        ----------
        features : str or list
            features can be either a string containing the name of a shapefile
            or it can be a list of points, lines, or polygons
        featuretype : str
            Must be either 'point', 'line', or 'polygon'
        level : int
            The level of refinement for this features
        layers : list
            A list of layers (zero based) for which this refinement features
            applies.

        Returns
        -------
        None

        """
        rfname = 'rf{}'.format(len(self._rfdict))
        if isinstance(features, list):
            features_to_shapefile(features, featuretype, rfname)
            shapefile = rfname
        else:
            shapefile = features

        self._rfdict[rfname] = [shapefile, featuretype, level]
        sn = shapefile + '.shp'
        assert os.path.isfile(sn), 'Shapefile does not exist: {}'.format(sn)

        for k in layers:
            self._refinement_features[k].append(rfname)

        return

    def build(self):
        """
        Build the quadtree grid

        Returns
        -------
        None

        """
        fname = os.path.join(self.model_ws, '_gridgen_build.dfn')
        f = open(fname, 'w')

        # Write the basegrid information
        f.write(self._mfgrid_block())
        f.write(2 * '\n')

        # Write the quadtree builder block
        f.write(self._builder_block())
        f.write(2 * '\n')

        # Write the active domain blocks
        f.write(self._ad_blocks())
        f.write(2 * '\n')

        # Write the refinement features
        f.write(self._rf_blocks())
        f.write(2 * '\n')
        f.close()

        # Build the grid
        # Command: gridgen quadtreebuilder _gridgen_build.dfn
        qtgfname = os.path.join(self.model_ws, 'quadtreegrid.dfn')
        if os.path.isfile(qtgfname):
            os.remove(qtgfname)
        cmds = [self.exe_name, 'quadtreebuilder', '_gridgen_build.dfn']
        buff = subprocess.check_output(cmds, cwd=self.model_ws)
        assert os.path.isfile(qtgfname)

        # Make shapefiles
        self.export()

        return

    def export(self):
        """
        Export the quadtree grid to shapefiles, usgdata, and vtk

        Returns
        -------
        None

        """
        # Create the export definition file
        fname = os.path.join(self.model_ws, '_gridgen_export.dfn')
        f = open(fname, 'w')
        f.write('LOAD quadtreegrid.dfn\n')
        f.write('\n')
        f.write(self._grid_export_blocks())
        f.close()
        assert os.path.isfile(fname), \
            'Could not create export dfn file: {}'.format(fname)

        # Export shapefiles
        cmds = [self.exe_name, 'grid_to_shapefile_poly', '_gridgen_export.dfn']
        buff = []
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            assert os.path.isfile('qtgrid.shp')
        except:
            print('Error.  Failed to export polygon shapefile of grid', buff)

        cmds = [self.exe_name, 'grid_to_shapefile_point', '_gridgen_export.dfn']
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            assert os.path.isfile('qtgrid_pt.shp')
        except:
            print ('Error.  Failed to export polygon shapefile of grid', buff)

        # Export the usg data
        cmds = [self.exe_name, 'grid_to_usgdata', '_gridgen_export.dfn']
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            assert os.path.isfile('qtg.nod')
        except:
            print ('Error.  Failed to export usgdata', buff)

        # Export vtk
        cmds = [self.exe_name, 'grid_to_vtk', '_gridgen_export.dfn']
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            assert os.path.isfile('qtg.vtu')
        except:
            print ('Error.  Failed to export vtk file', buff)

        cmds = [self.exe_name, 'grid_to_vtk_sv', '_gridgen_export.dfn']
        try:
            buff = subprocess.check_output(cmds, cwd=self.model_ws)
            assert os.path.isfile('qtg_sv.vtu')
        except:
            print ('Error.  Failed to export shared vertex vtk file', buff)

        return

    def _mfgrid_block(self):
        s = ''
        s += 'BEGIN MODFLOW_GRID basegrid' + '\n'
        s += '  ROTATION_ANGLE = {}\n'.format(self.dis.sr.rotation)
        s += '  X_OFFSET = {}\n'.format(self.dis.sr.xul)
        s += '  Y_OFFSET = {}\n'.format(self.dis.sr.yedge[-1])
        s += '  NLAY = {}\n'.format(self.dis.nlay)
        s += '  NROW = {}\n'.format(self.dis.nrow)
        s += '  NCOL = {}\n'.format(self.dis.ncol)

        # delr
        delr = self.dis.delr.array
        if delr.min() == delr.max():
            s += '  DELR = CONSTANT {}\n'.format(delr.min())
        else:
            s += '  DELR = OPEN/CLOSE delr.dat\n'
            fname = os.path.join(self.model_ws, 'delr.dat')
            np.savetxt(fname, delr)

        # delc
        delc = self.dis.delc.array
        if delc.min() == delc.max():
            s += '  DELC = CONSTANT {}\n'.format(delc.min())
        else:
            s += '  DELC = OPEN/CLOSE delc.dat\n'
            fname = os.path.join(self.model_ws, 'delc.dat')
            np.savetxt(fname, delc)

        # top
        top = self.dis.top.array
        if top.min() == top.max():
            s += '  TOP = CONSTANT {}\n'.format(top.min())
        else:
            s += '  TOP = OPEN/CLOSE top.dat\n'
            fname = os.path.join(self.model_ws, 'top.dat')
            np.savetxt(fname, top)

        # bot
        botm = self.dis.botm
        for k in range(self.dis.nlay):
            bot = botm[k].array
            if bot.min() == bot.max():
                s += '  BOTTOM LAYER {} = CONSTANT {}\n'.format(k + 1,
                                                                bot.min())
            else:
                s += '  BOTTOM LAYER {0} = OPEN/CLOSE bot{0}.dat\n'.format(k + 1)
                fname = os.path.join(self.model_ws, 'bot{}.dat'.format(k + 1))
                np.savetxt(fname, bot)

        s += 'END MODFLOW_GRID' + '\n'
        return s

    def _rf_blocks(self):
        s = ''
        for rfname, rf in self._rfdict.items():
            shapefile, featuretype, level = rf
            s += 'BEGIN REFINEMENT_FEATURES {}\n'.format(rfname)
            s += '  SHAPEFILE = {}\n'.format(shapefile)
            s += '  FEATURE_TYPE = {}\n'.format(featuretype)
            s += '  REFINEMENT_LEVEL = {}\n'.format(level)
            s += 'END REFINEMENT_FEATURES\n'
            s += 2 * '\n'
        return s

    def _ad_blocks(self):
        s = ''
        for adname, shapefile in self._addict.items():
            s += 'BEGIN ACTIVE_DOMAIN {}\n'.format(adname)
            s += '  SHAPEFILE = {}\n'.format(shapefile)
            s += '  FEATURE_TYPE = {}\n'.format('polygon')
            s += '  INCLUDE_BOUNDARY = {}\n'.format('True')
            s += 'END ACTIVE_DOMAIN\n'
            s += 2 * '\n'
        return s

    def _builder_block(self):
        s = 'BEGIN QUADTREE_BUILDER quadtreebuilder\n'
        s += '  MODFLOW_GRID = basegrid\n'

        # Write active domain information
        for k, adk in enumerate(self._active_domain):
            if adk is None:
                continue
            s += '  ACTIVE_DOMAIN LAYER {0} = active_domain_layer_{0}\n'.format(k + 1)

        # Write refinement feature information
        for k, rfkl in enumerate(self._refinement_features):
            if len(rfkl) == 0:
                continue
            s += '  REFINEMENT_FEATURES LAYER {} = '.format(k + 1)
            for rf in rfkl:
                s += rf + ' '
            s += '\n'

        s += '  SMOOTHING = full\n'

        for k in range(self.dis.nlay):
            s += '  TOP LAYER {} = INTERPOLATE basegrid\n'.format(k + 1)

        for k in range(self.dis.nlay):
            s += '  BOTTOM LAYER {} = INTERPOLATE basegrid\n'.format(k + 1)

        s += '  GRID_DEFINITION_FILE = quadtreegrid.dfn\n'
        s += 'END QUADTREE_BUILDER\n'
        return s

    def _grid_export_blocks(self):
        s = 'BEGIN GRID_TO_SHAPEFILE grid_to_shapefile_poly\n'
        s += '  GRID = quadtreegrid\n'
        s += '  SHAPEFILE = qtgrid\n'
        s += '  FEATURE_TYPE = polygon\n'
        s += 'END GRID_TO_SHAPEFILE\n'
        s += '\n'
        s += 'BEGIN GRID_TO_SHAPEFILE grid_to_shapefile_point\n'
        s += '  GRID = quadtreegrid\n'
        s += '  SHAPEFILE = qtgrid_pt\n'
        s += '  FEATURE_TYPE = point\n'
        s += 'END GRID_TO_SHAPEFILE\n'
        s += '\n'
        s += 'BEGIN GRID_TO_USGDATA grid_to_usgdata\n'
        s += '  GRID = quadtreegrid\n'
        s += '  USG_DATA_PREFIX = qtg\n'
        s += 'END GRID_TO_USGDATA\n'
        s += '\n'
        s += 'BEGIN GRID_TO_VTKFILE grid_to_vtk\n'
        s += '  GRID = quadtreegrid\n'
        s += '  VTKFILE = qtg\n'
        s += '  SHARE_VERTEX = False\n'
        s += 'END GRID_TO_VTKFILE\n'
        s += '\n'
        s += 'BEGIN GRID_TO_VTKFILE grid_to_vtk_sv\n'
        s += '  GRID = quadtreegrid\n'
        s += '  VTKFILE = qtg_sv\n'
        s += '  SHARE_VERTEX = True\n'
        s += 'END GRID_TO_VTKFILE\n'
        return s

