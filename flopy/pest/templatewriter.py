from __future__ import print_function
import flopy.pest.tplarray as tplarray

class TemplateWriter(object):
    """
    Class for writing PEST template files.

    Parameters
    ----------
    model : flopy.modflow object
        flopy model object.
    plist : list
        list of parameter objects of type flopy.pest.params.Params.
    """
    def __init__(self, model, plist):
        self.model = model
        self.plist = plist
        return

    def write_template(self):
        """
        Write the template files for all model files that have arrays that
        have been parameterized.

        """

        # Import and initialize
        import copy
        pakdict = {}

        # Create a copy of any flopy model packages that have arrays that have
        # been parameterized.
        for p in self.plist:
            pakdictkey = p.mfpackage.upper()
            if pakdictkey not in pakdict:
                paktpl = copy.copy(self.model.get_package(pakdictkey))
                pakdict[pakdictkey] = paktpl

        # Go through each copy of parameterized packages, replace parameterized
        # arrays with template arrays (Util3dTpl) and then replace the string
        # array values with the parameter name.
        for p in self.plist:
            paktpl = pakdict[p.mfpackage.upper()]
            if not hasattr(paktpl, p.type.lower()):
                msg = 'Parameter type {} not found in package.'.format(p.type.lower())
                raise Exception(msg)
            pakarray = getattr(paktpl, p.type.lower())
            if not isinstance(pakarray, tplarray.Util3dTpl):
                tpla = tplarray.Util3dTpl(pakarray.array, p.type.lower())
                setattr(paktpl, p.type.lower(), tpla)

            # Fill the template array with the string name
            tpla = getattr(paktpl, p.type.lower())
            tpla.chararray[p.idx] = '~{0:^13s}~'.format(p.name)

        # Go through each package template and write the template file
        for pakdictkey, paktpl in pakdict.items():
            paktpl.heading = 'ptf ~\n' + paktpl.heading
            paktpl.fn_path += '.tpl'
            paktpl.write_file()

        return

