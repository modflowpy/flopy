"""
pakbase module
  This module contains the base package class from which
  all of the other packages inherit from.

"""

from __future__ import print_function

import os
import sys
import platform
import webbrowser as wb

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from .modflow.mfparbc import ModflowParBc as mfparbc
from .utils import Util2d, Util3d, Transient2d, MfList, check


class Package(object):
    """
    Base package class from which most other packages are derived.

    """

    def __init__(self, parent, extension='glo', name='GLOBAL', unit_number=1,
                 extra='', filenames=None, allowDuplicates=False):
        """
        Package init

        """
        self.parent = parent  # To be able to access the parent modflow object's attributes
        if (not isinstance(extension, list)):
            extension = [extension]
        self.extension = []
        self.file_name = []
        for idx, e in enumerate(extension):
            self.extension.append(e)
            file_name = self.parent.name + '.' + e
            if filenames is not None:
                try:
                    if filenames[idx] is not None:
                        file_name = filenames[idx]
                except:
                    pass
            self.file_name.append(file_name)
        self.fn_path = os.path.join(self.parent.model_ws, self.file_name[0])
        if (not isinstance(name, list)):
            name = [name]
        self.name = name
        if (not isinstance(unit_number, list)):
            unit_number = [unit_number]
        self.unit_number = unit_number
        if (not isinstance(extra, list)):
            self.extra = len(self.unit_number) * [extra]
        else:
            self.extra = extra
        self.url = 'index.html'
        self.allowDuplicates = allowDuplicates

        self.acceptable_dtypes = [int, np.float32, str]
        return

    def __repr__(self):
        s = self.__doc__
        exclude_attributes = ['extension', 'heading', 'name', 'parent', 'url']
        for attr, value in sorted(self.__dict__.items()):
            if not (attr in exclude_attributes):
                if (isinstance(value, list)):
                    if (len(value) == 1):
                        s = s + ' {0:s} = {1:s}\n'.format(attr, str(value[0]))
                    else:
                        s = s + ' {0:s} (list, items = {1:d}\n'.format(attr,
                                                                       len(
                                                                           value))
                elif (isinstance(value, np.ndarray)):
                    s = s + ' {0:s} (array, shape = {1:s})\n'.format(attr,
                                                                     value.shape.__str__()[
                                                                     1:-1])
                else:
                    s = s + ' {0:s} = {1:s} ({2:s})\n'.format(attr, str(value),
                                                              str(type(value))[
                                                              7:-2])
        return s

    def __getitem__(self, item):
        if hasattr(self, 'stress_period_data'):
            # added this check because stress_period_data also used in Oc and Oc88 but is not a MfList
            if isinstance(item, MfList):
                if not isinstance(item, list) and not isinstance(item, tuple):
                    assert item in list(
                        self.stress_period_data.data.keys()), "package.__getitem__() kper " + str(
                        item) + " not in data.keys()"
                    return self.stress_period_data[item]
                else:
                    if item[1] not in self.dtype.names:
                        raise Exception(
                            "package.__getitem(): item \'" + item + "\' not in dtype names " + str(
                                self.dtype.names))
                    assert item[0] in list(
                        self.stress_period_data.data.keys()), "package.__getitem__() kper " + str(
                        item[0]) + " not in data.keys()"
                    if self.stress_period_data.vtype[item[0]] == np.recarray:
                        return self.stress_period_data[item[0]][item[1]]

    def __setitem__(self, key, value):
        raise NotImplementedError("package.__setitem__() not implemented")

    def __setattr__(self, key, value):
        var_dict = vars(self)
        if key in list(var_dict.keys()):
            old_value = var_dict[key]
            if isinstance(old_value, Util2d):
                value = Util2d(self.parent, old_value.shape,
                               old_value.dtype, value,
                               name=old_value.name,
                               fmtin=old_value.format.fortran,
                               locat=old_value.locat,
                               array_free_format=old_value.format.array_free_format)
            elif isinstance(old_value, Util3d):
                value = Util3d(self.parent, old_value.shape,
                               old_value.dtype, value,
                               name=old_value.name_base,
                               fmtin=old_value.fmtin,
                               locat=old_value.locat,
                               array_free_format=old_value.array_free_format)
            elif isinstance(old_value, Transient2d):
                value = Transient2d(self.parent, old_value.shape,
                                    old_value.dtype, value,
                                    name=old_value.name_base,
                                    fmtin=old_value.fmtin,
                                    locat=old_value.locat)
            elif isinstance(old_value, MfList):
                value = MfList(self, dtype=old_value.dtype,
                               data=value)
            elif isinstance(old_value, list):
                if len(old_value) > 0:
                    if isinstance(old_value[0], Util3d):
                        new_list = []
                        for vo, v in zip(old_value, value):
                            new_list.append(Util3d(self.parent, vo.shape,
                                                   vo.dtype, v,
                                                   name=vo.name_base,
                                                   fmtin=vo.fmtin,
                                                   locat=vo.locat))
                        value = new_list
                    elif isinstance(old_value[0], Util2d):
                        new_list = []
                        for vo, v in zip(old_value, value):
                            new_list.append(Util2d(self.parent, vo.shape,
                                                   vo.dtype, v,
                                                   name=vo.name,
                                                   fmtin=vo.fmtin,
                                                   locat=vo.locat))
                        value = new_list

        super(Package, self).__setattr__(key, value)

    def export(self, f, **kwargs):
        from flopy import export
        return export.utils.package_helper(f, self, **kwargs)

    @staticmethod
    def add_to_dtype(dtype, field_names, field_types):
        """
        Add one or more fields to a structured array data type

        Parameters
        ----------
        dtype : numpy.dtype
            Input structured array datatype to add to.
        field_names : str or list
            One or more field names.
        field_types : numpy.dtype or list
            One or more data types. If one data type is supplied, it is
            repeated for each field name.
        """
        if not isinstance(field_names, list):
            field_names = [field_names]
        if not isinstance(field_types, list):
            field_types = [field_types] * len(field_names)
        newdtypes = dtype.descr
        for field_name, field_type in zip(field_names, field_types):
            newdtypes.append((str(field_name), field_type))
        return np.dtype(newdtypes)

    def check(self, f=None, verbose=True, level=1):
        """
        Check package data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a sting is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.dis.check()

        """
        chk = None

        if self.__dict__.get('stress_period_data', None) is not None and \
                        self.name[0] != 'OC':
            spd_inds_valid = True
            chk = check(self, f=f, verbose=verbose, level=level)
            for per in self.stress_period_data.data.keys():
                if isinstance(self.stress_period_data.data[per], np.recarray):
                    spd = self.stress_period_data.data[per]
                    inds = (spd.k, spd.i, spd.j) if self.parent.structured \
                        else (spd.node)

                    # General BC checks
                    # check for valid cell indices
                    spd_inds_valid = chk._stress_period_data_valid_indices(spd)

                    # first check for and list nan values
                    chk._stress_period_data_nans(spd)

                    if spd_inds_valid:
                        # next check for BCs in inactive cells
                        chk._stress_period_data_inactivecells(spd)

                        # More specific BC checks
                        # check elevations in the ghb, drain, and riv packages
                        if self.name[0] in check.bc_stage_names.keys():
                            # check that bc elevations are above model cell bottoms
                            # also checks for nan values
                            elev_name = chk.bc_stage_names[self.name[0]]
                            botms = self.parent.dis.botm.array[inds]
                            chk.stress_period_data_values(spd, spd[
                                elev_name] < botms,
                                                          col=elev_name,
                                                          error_name='BC elevation below cell bottom',
                                                          error_type='Error')

            chk.summarize()

        # check property values in upw and lpf packages
        elif self.name[0] in ['UPW', 'LPF']:

            chk = check(self, f=f, verbose=verbose, level=level)
            active = chk.get_active()

            # check for confined layers above convertable layers
            confined = False
            thickstrt = False
            for option in self.options:
                if option.lower() == 'thickstrt':
                    thickstrt = True
            for i, l in enumerate(self.laytyp.array.tolist()):
                if l == 0 or l < 0 and thickstrt:
                    confined = True
                    continue
                if confined and l > 0:
                    chk._add_to_summary(type='Warning',
                                        desc='\r    LAYTYP: unconfined (convertible) ' + \
                                             'layer below confined layer')

            # check for zero or negative values of hydraulic conductivity, anisotropy,
            # and quasi-3D confining beds
            kparams = {'hk': 'horizontal hydraulic conductivity',
                       'vka': 'vertical hydraulic conductivity'}
            for kp, name in kparams.items():
                chk.values(self.__dict__[kp].array,
                           active & (self.__dict__[kp].array <= 0),
                           'zero or negative {} values'.format(name), 'Error')

            # check for negative hani
            chk.values(self.__dict__['hani'].array,
                       active & (self.__dict__['hani'].array < 0),
                       'negative horizontal anisotropy values', 'Error')

            def check_thresholds(array, active, thresholds, name):
                """Checks array against min and max threshold values."""
                mn, mx = thresholds
                chk.values(array, active & (array < mn),
                           '{} values below checker threshold of {}'
                           .format(name, mn), 'Warning')
                chk.values(array, active & (array > mx),
                           '{} values above checker threshold of {}'
                           .format(name, mx), 'Warning')

            # check for unusually high or low values of hydraulic conductivity
            if self.layvka.sum() > 0:  # convert vertical anistropy to Kv for checking
                vka = self.vka.array.copy()
                for l in range(vka.shape[0]):
                    vka[l] *= self.hk.array[l] if self.layvka.array[
                                                      l] != 0 else 1
                check_thresholds(vka, active,
                                 chk.property_threshold_values['vka'],
                                 kparams.pop('vka'))

            for kp, name in kparams.items():
                check_thresholds(self.__dict__[kp].array, active,
                                 chk.property_threshold_values[kp],
                                 name)

            # check vkcb if there are any quasi-3D layers
            if self.parent.dis.laycbd.sum() > 0:
                # pad non-quasi-3D layers in vkcb array with ones so they won't fail checker
                vkcb = self.vkcb.array.copy()
                for l in range(self.vkcb.shape[0]):
                    if self.parent.dis.laycbd[l] == 0:
                        vkcb[l, :,
                        :] = 1  # assign 1 instead of zero as default value that won't violate checker
                        # (allows for same structure as other checks)

                chk.values(vkcb, active & (vkcb <= 0),
                           'zero or negative quasi-3D confining bed Kv values',
                           'Error')
                check_thresholds(vkcb, active,
                                 chk.property_threshold_values['vkcb'],
                                 'quasi-3D confining bed Kv')

            # only check storage if model is transient
            if not np.all(self.parent.dis.steady.array):

                # do the same for storage if the model is transient
                sarrays = {'ss': self.ss.array, 'sy': self.sy.array}
                if 'STORAGECOEFFICIENT' in self.options:  # convert to specific for checking
                    chk._add_to_summary(type='Warning',
                                        desc='\r    STORAGECOEFFICIENT option is activated, \
                                              storage values are read storage coefficients')
                    tshape = (self.parent.nlay, self.parent.nrow,
                              self.parent.ncol)
                    sarrays['ss'].shape != tshape
                    sarrays['sy'].shape != tshape

                chk.values(sarrays['ss'], active & (sarrays['ss'] < 0),
                           'zero or negative specific storage values', 'Error')
                check_thresholds(sarrays['ss'], active,
                                 chk.property_threshold_values['ss'],
                                 'specific storage')

                # only check specific yield for convertible layers
                inds = np.array(
                    [True if l > 0 or l < 0 and 'THICKSRT' in self.options
                     else False for l in self.laytyp])
                sarrays['sy'] = sarrays['sy'][inds, :, :]
                active = active[inds, :, :]
                chk.values(sarrays['sy'], active & (sarrays['sy'] < 0),
                           'zero or negative specific yield values', 'Error')
                check_thresholds(sarrays['sy'], active,
                                 chk.property_threshold_values['sy'],
                                 'specific yield')
            chk.summarize()

        else:
            txt = 'check method not implemented for {} Package.'.format(
                self.name[0])
            if f is not None:
                if isinstance(f, str):
                    pth = os.path.join(self.parent.model_ws, f)
                    f = open(pth, 'w')
                    f.write(txt)
                    f.close()
            if verbose:
                print(txt)
        return chk

    def level1_arraylist(self, idx, v, name, txt):
        ndim = v.ndim
        if ndim == 3:
            kon = -1
            for [k, i, j] in idx:
                if k > kon:
                    kon = k
                    txt += '    {:>10s}{:>10s}{:>10s}{:>15s}\n'.format('layer',
                                                                       'row',
                                                                       'column',
                                                                       name[
                                                                           k].lower().replace(
                                                                           ' layer ',
                                                                           ''))
                txt += '    {:10d}{:10d}{:10d}{:15.7g}\n'.format(k + 1, i + 1,
                                                                 j + 1,
                                                                 v[k, i, j])
        elif ndim == 2:
            txt += '    {:>10s}{:>10s}{:>15s}\n'.format('row', 'column',
                                                        name[
                                                            0].lower().replace(
                                                            ' layer ', ''))
            for [i, j] in idx:
                txt += '    {:10d}{:10d}{:15.7g}\n'.format(i + 1, j + 1,
                                                           v[i, j])
        elif ndim == 1:
            txt += '    {:>10s}{:>15s}\n'.format('number', name[0])
            for i in idx:
                txt += '    {:10d}{:15.7g}\n'.format(i + 1, v[i])
        return txt

    def plot(self, **kwargs):
        """
        Plot 2-D, 3-D, transient 2-D, and stress period list (MfList)
        package input data

        Parameters
        ----------
        **kwargs : dict
            filename_base : str
                Base file name that will be used to automatically generate file
                names for output image files. Plots will be exported as image
                files if file_name_base is not None. (default is None)
            file_extension : str
                Valid matplotlib.pyplot file extension for savefig(). Only used
                if filename_base is not None. (default is 'png')
            mflay : int
                MODFLOW zero-based layer number to return.  If None, then all
                all layers will be included. (default is None)
            kper : int
                MODFLOW zero-based stress period number to return. (default is
                zero)
            key : str
                MfList dictionary key. (default is None)

        Returns
        ----------
        axes : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis are returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.dis.plot()

        """

        # valid keyword arguments
        if 'kper' in kwargs:
            kper = kwargs.pop('kper')
        else:
            kper = 0

        if 'filename_base' in kwargs:
            fileb = kwargs.pop('filename_base')
        else:
            fileb = None

        if 'mflay' in kwargs:
            mflay = kwargs.pop('mflay')
        else:
            mflay = None

        if 'file_extension' in kwargs:
            fext = kwargs.pop('file_extension')
            fext = fext.replace('.', '')
        else:
            fext = 'png'

        if 'key' in kwargs:
            key = kwargs.pop('key')
        else:
            key = None

        if 'initial_fig' in kwargs:
            ifig = int(kwargs.pop('initial_fig'))
        else:
            ifig = 0

        inc = self.parent.nlay
        if mflay is not None:
            inc = 1

        axes = []
        for item, value in self.__dict__.items():
            caxs = []
            if isinstance(value, MfList):
                if self.parent.verbose:
                    print('plotting {} package MfList instance: {}'.format(
                        self.name[0], item))
                if key is None:
                    names = ['{} location stress period {} layer {}'.format(
                        self.name[0], kper + 1, k + 1)
                             for k in range(self.parent.nlay)]
                    colorbar = False
                else:
                    names = ['{} {} data stress period {} layer {}'.format(
                        self.name[0], key, kper + 1, k + 1)
                             for k in range(self.parent.nlay)]
                    colorbar = True

                fignum = list(range(ifig, ifig + inc))
                ifig = fignum[-1] + 1
                caxs.append(value.plot(key, names, kper,
                                       filename_base=fileb,
                                       file_extension=fext, mflay=mflay,
                                       fignum=fignum, colorbar=colorbar,
                                       **kwargs))

            elif isinstance(value, Util3d):
                if self.parent.verbose:
                    print('plotting {} package Util3d instance: {}'.format(
                        self.name[0], item))
                # fignum = list(range(ifig, ifig + inc))
                fignum = list(range(ifig, ifig + value.shape[0]))
                ifig = fignum[-1] + 1
                caxs.append(
                    value.plot(filename_base=fileb, file_extension=fext,
                               mflay=mflay,
                               fignum=fignum, colorbar=True))
            elif isinstance(value, Util2d):
                if len(value.shape) == 2:
                    if self.parent.verbose:
                        print('plotting {} package Util2d instance: {}'.format(
                            self.name[0], item))
                    fignum = list(range(ifig, ifig + 1))
                    ifig = fignum[-1] + 1
                    caxs.append(
                        value.plot(filename_base=fileb,
                                   file_extension=fext,
                                   fignum=fignum, colorbar=True))
            elif isinstance(value, Transient2d):
                if self.parent.verbose:
                    print(
                        'plotting {} package Transient2d instance: {}'.format(
                            self.name[0], item))
                fignum = list(range(ifig, ifig + inc))
                ifig = fignum[-1] + 1
                caxs.append(
                    value.plot(filename_base=fileb, file_extension=fext,
                               kper=kper,
                               fignum=fignum, colorbar=True))
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, Util3d):
                        if self.parent.verbose:
                            print(
                                'plotting {} package Util3d instance: {}'.format(
                                    self.name[0], item))
                        fignum = list(range(ifig, ifig + inc))
                        ifig = fignum[-1] + 1
                        caxs.append(
                            v.plot(filename_base=fileb,
                                   file_extension=fext,
                                   mflay=mflay,
                                   fignum=fignum, colorbar=True))
            else:
                pass

            # unroll nested lists os axes into a single list of axes
            if isinstance(caxs, list):
                for c in caxs:
                    if isinstance(c, list):
                        for cc in c:
                            axes.append(cc)
                    else:
                        axes.append(c)
            else:
                axes.append(caxs)

        return axes

    def to_shapefile(self, filename, **kwargs):
        """
        Export 2-D, 3-D, and transient 2-D model data to shapefile (polygons).
        Adds an attribute for each layer in each data array

        Parameters
        ----------
        filename : str
            Shapefile name to write

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.lpf.to_shapefile('test_hk.shp')

        """
        import warnings
        warnings.warn("to_shapefile() is deprecated. use .export()")
        self.export(filename)

    def webdoc(self):
        if self.parent.version == 'mf2k':
            wb.open(
                'http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/' + self.url)
        elif self.parent.version == 'mf2005':
            wb.open(
                'http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/' + self.url)
        elif self.parent.version == 'ModflowNwt':
            wb.open(
                'http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/' + self.url)

    def write_file(self, check=False):
        """
        Every Package needs its own write_file function

        """
        print('IMPLEMENTATION ERROR: write_file must be overloaded')
        return

    @staticmethod
    def load(model, pack_type, f, nper=None, pop_key_list=None, check=True,
             unitnumber=None, ext_unit_dict=None):
        """
        The load method has not been implemented for this package.

        """

        bc_pack_types = []

        if not hasattr(f, 'read'):
            filename = f
            if platform.system().lower() == 'windows' and \
                            sys.version_info[0] < 3:
                import io
                f = io.open(filename, 'r')
            else:
                f = open(filename, 'r')
        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        # check for parameters
        nppak = 0
        if "parameter" in line.lower():
            t = line.strip().split()
            # assert int(t[1]) == 0,"Parameters are not supported"
            nppak = np.int(t[1])
            mxl = 0
            if nppak > 0:
                mxl = np.int(t[2])
                if model.verbose:
                    print('   Parameters detected. Number of parameters = ',
                          nppak)
            line = f.readline()
        # dataset 2a
        t = line.strip().split()
        ipakcb = 0
        try:
            ipakcb = int(t[1])
        except:
            pass
        options = []
        aux_names = []
        if len(t) > 2:
            it = 2
            while it < len(t):
                toption = t[it]
                if toption.lower() is 'noprint':
                    options.append(toption)
                elif 'aux' in toption.lower():
                    options.append(' '.join(t[it:it + 2]))
                    aux_names.append(t[it + 1].lower())
                    it += 1
                it += 1

        # set partype
        #  and read phiramp for modflow-nwt well package
        partype = ['cond']
        if 'modflowwel' in str(pack_type).lower():
            partype = ['flux']

        if 'nwt' in model.version.lower() and \
            'flopy.modflow.mfwel.modflowwel'.lower() in str(pack_type).lower():

            specify = False
            ipos = f.tell()
            line = f.readline()
            # test for specify keyword if a NWT well file
            if 'specify' in line.lower():
                specify = True
                t = line.strip().split()
                phiramp = np.float32(t[1])
                try:
                    phiramp_unit = np.int32(t[2])
                except:
                    phiramp_unit = 2
                options.append('specify {} {} '.format(phiramp, phiramp_unit))
            else:
                f.seek(ipos)
        elif 'flopy.modflow.mfchd.modflowchd'.lower() in str(
                pack_type).lower():
            partype = ['shead', 'ehead']

        # read parameter data
        if nppak > 0:
            dt = pack_type.get_empty(1, aux_names=aux_names,
                                     structured=model.structured).dtype
            pak_parms = mfparbc.load(f, nppak, dt, model.verbose)
            # pak_parms = mfparbc.load(f, nppak, len(dt.names))

        if nper is None:
            nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # read data for every stress period
        bnd_output = None
        stress_period_data = {}
        for iper in range(nper):
            if model.verbose:
                print(
                    "   loading " + str(
                        pack_type) + " for kper {0:5d}".format(
                        iper + 1))
            line = f.readline()
            if line == '':
                break
            t = line.strip().split()
            itmp = int(t[0])
            itmpp = 0
            try:
                itmpp = int(t[1])
            except:
                pass

            if itmp == 0:
                bnd_output = None
                current = pack_type.get_empty(itmp, aux_names=aux_names,
                                              structured=model.structured)
            elif itmp > 0:
                current = pack_type.get_empty(itmp, aux_names=aux_names,
                                              structured=model.structured)
                for ibnd in range(itmp):
                    line = f.readline()
                    if "open/close" in line.lower():
                        binary = False
                        if '(binary)' in line.lower():
                            binary = True
                        # need to strip out existing path seps and
                        # replace current-system path seps
                        raw = line.strip().split()
                        fname = raw[1]
                        if '/' in fname:
                            raw = fname.split('/')
                        elif '\\' in fname:
                            raw = fname.split('\\')
                        else:
                            raw = [fname]
                        fname = os.path.join(*raw)
                        oc_filename = os.path.join(model.model_ws, fname)
                        assert os.path.exists(
                            oc_filename), "Package.load() error: open/close filename " + \
                                          oc_filename + " not found"
                        try:
                            if binary:
                                dtype2 = []
                                for name in current.dtype.names:
                                    dtype2.append((name, np.float32))
                                dtype2 = np.dtype(dtype2)
                                d = np.fromfile(oc_filename,
                                                dtype=dtype2,
                                                count=itmp)
                                current = np.array(d, dtype=current.dtype)
                            else:
                                #current = np.genfromtxt(oc_filename,
                                #                         dtype=current.dtype)
                                #if len(current.shape) == 1:
                                cd = current.dtype
                                current = np.loadtxt(oc_filename).transpose()
                                if current.ndim == 1:
                                    current = np.atleast_2d(current).transpose()
                                #current = np.atleast_2d(np.loadtxt(oc_filename,
                                #                                   dtype=current.dtype)).transpose()
                                current = np.core.records.fromarrays(current,dtype=cd)
                            current = current.view(np.recarray)
                        except Exception as e:
                            raise Exception(
                                "Package.load() error loading open/close file " + oc_filename + \
                                " :" + str(e))
                        assert current.shape[
                                   0] == itmp, "Package.load() error: open/close rec array from file " + \
                                               oc_filename + " shape (" + str(current.shape) + \
                                               ") does not match itmp: {0:d}".format(
                                                   itmp)
                        break
                    try:
                        t = line.strip().split()
                        current[ibnd] = tuple(t[:len(current.dtype.names)])
                    except:
                        t = []
                        for ivar in range(len(current.dtype.names)):
                            istart = ivar * 10
                            istop = istart + 10
                            t.append(line[istart:istop])
                        current[ibnd] = tuple(t[:len(current.dtype.names)])

                # convert indices to zero-based
                if model.structured:
                    current['k'] -= 1
                    current['i'] -= 1
                    current['j'] -= 1
                else:
                    current['node'] -= 1
                bnd_output = np.recarray.copy(current)
            else:
                bnd_output = np.recarray.copy(current)

            for iparm in range(itmpp):
                line = f.readline()
                t = line.strip().split()
                pname = t[0].lower()
                iname = 'static'
                try:
                    tn = t[1]
                    c = tn.lower()
                    instance_dict = pak_parms.bc_parms[pname][1]
                    if c in instance_dict:
                        iname = c
                    else:
                        iname = 'static'
                except:
                    pass
                par_dict, current_dict = pak_parms.get(pname)
                data_dict = current_dict[iname]

                par_current = pack_type.get_empty(par_dict['nlst'],
                                                  aux_names=aux_names)

                #  get appropriate parval
                if model.mfpar.pval is None:
                    parval = np.float(par_dict['parval'])
                else:
                    try:
                        parval = np.float(model.mfpar.pval.pval_dict[pname])
                    except:
                        parval = np.float(par_dict['parval'])

                # fill current parameter data (par_current)
                for ibnd, t in enumerate(data_dict):
                    par_current[ibnd] = tuple(t[:len(par_current.dtype.names)])

                if model.structured:
                    par_current['k'] -= 1
                    par_current['i'] -= 1
                    par_current['j'] -= 1
                else:
                    par_current['node'] -= 1

                for ptype in partype:
                    par_current[ptype] *= parval

                if bnd_output is None:
                    bnd_output = np.recarray.copy(par_current)
                else:
                    bnd_output = stack_arrays((bnd_output, par_current),
                                              asrecarray=True, usemask=False)

            if bnd_output is None:
                stress_period_data[iper] = itmp
            else:
                stress_period_data[iper] = bnd_output

        dtype = pack_type.get_empty(0, aux_names=aux_names,
                                    structured=model.structured).dtype

        # set package unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=pack_type.ftype())
            if ipakcb > 0:
                iu, filenames[1] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
                model.add_pop_key_list(ipakcb)

        pak = pack_type(model, ipakcb=ipakcb,
                        stress_period_data=stress_period_data,
                        dtype=dtype, options=options,
                        unitnumber=unitnumber, filenames=filenames)
        if check:
            pak.check(f='{}.chk'.format(pak.name[0]),
                      verbose=pak.parent.verbose, level=0)
        return pak
