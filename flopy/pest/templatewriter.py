from __future__ import print_function
from ..pest import tplarray as tplarray


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

        # Create a list of packages that have parameters applied to them.
        # Verify that the package exists
        ftypelist = []
        for p in self.plist:
            ftype = p.mfpackage.upper()
            if ftype not in ftypelist:

                # Verify package exists in model
                try:
                    pak = self.model.get_package(ftype)
                except:
                    raise Exception("Package type {} not found.".format(ftype))

                # Check to make sure pak has p.type as an attribute
                if not hasattr(pak, p.type.lower()):
                    msg = (
                        "Parameter named {} of type {} not found in "
                        "package {}".format(p.name, p.type.lower(), ftype)
                    )
                    raise Exception(msg)

                # Ftype is valid and package has attribute so store in list
                ftypelist.append(ftype)

        # Print a list of packages that will be parameterized
        print(
            "The following packages will be parameterized: "
            "{}\n".format(ftypelist)
        )

        # Go through each package, and then through each parameter and make
        # the substitution.  Then write the template file.
        for ftype in ftypelist:
            pak = self.model.get_package(ftype)
            paktpl = copy.copy(pak)

            for p in self.plist:

                # Skip if parameter doesn't apply to this package
                if p.mfpackage.upper() != ftype:
                    continue

                # Create a new template array from the package array first
                # time it is referenced.
                pakarray = getattr(paktpl, p.type.lower())
                tpla = tplarray.get_template_array(pakarray)

                # Replace the array with the new template array.  Use the
                # __dict__ instead of setattr to avoid setitem protection
                # in mbase.
                paktpl.__dict__[p.type.lower()] = tpla

                # Substitute the parameter name in the template array
                tpla = getattr(paktpl, p.type.lower())
                tpla.add_parameter(p)

            # Write the file
            paktpl.heading = "ptf ~\n" + paktpl.heading
            paktpl.fn_path += ".tpl"
            paktpl.write_file(
                check=False
            )  # fot now, turn off checks for template files

            # Destroy the template version of the package
            paktpl = None

        return
