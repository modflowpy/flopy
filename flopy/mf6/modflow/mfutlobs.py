from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator, ArrayTemplateGenerator


class ModflowUtlobs(mfpackage.MFPackage):
    """
    ModflowUtlobs defines a obs package within a utl model.

    Attributes
    ----------
    precision : (precision : double)
        precision : Keyword and precision specifier for output of binary data,
          which can be either SINGLE or DOUBLE. The default is DOUBLE. When
          simulated values are written to a file specified as file type
          DATA(BINARY) in the Name File, the precision specifier controls
          whether the data (including simulated values and, for continuous
          observations, time values) are written as single- or double-
          precision.
    digits : (digits : integer)
        digits : Keyword and an integer digits specifier used for conversion of
          simulated values to text on output. The default is 5 digits. When
          simulated values are written to a file specified as file type DATA in
          the Name File, the digits specifier controls the number of
          significant digits with which simulated values are written to the
          output file. The digits specifier has no effect on the number of
          significant digits with which the simulation time is written for
          continuous observations.
    print_input : (print_input : boolean)
        print_input : keyword to indicate that the list of observation
          information will be written to the listing file immediately after it
          is read.
    continuousrecarray : [(obsname : string), (obstype : string), (id : string),
      (id2 : string)]
        obsname : string of 1 to 40 nonblank characters used to identify the
          observation. The identifier need not be unique; however,
          identification and post-processing of observations in the output
          files are facilitated if each observation is given a unique name.
        obstype : a string of characters used to identify the observation type.
        id : Text identifying cell where observation is located. For packages
          other than NPF, if boundary names are defined in the corresponding
          package input file, ID can be a boundary name. Otherwise ID is a
          cellid. If the model discretization is type DIS, cellid is three
          integers (layer, row, column). If the discretization is DISV, cellid
          is two integers (layer, cell number). If the discretization is DISU,
          cellid is one integer (node number).
        id2 : Text identifying cell adjacent to cell identified by ID. The form
          of ID2 is as described for ID. ID2 is used for intercell-flow
          observations of a GWF model, for three observation types of the LAK
          Package, for two observation types of the MAW Package, and one
          observation type of the UZF Package.

    """
    continuousrecarray = ListTemplateGenerator(('obs', 'continuous', 
                                                'continuousrecarray'))
    package_abbr = "utlobs"

    def __init__(self, model, add_to_package_list=True, precision=None,
                 digits=None, print_input=None, continuousrecarray=None,
                 fname=None, pname=None, parent_file=None):
        super(ModflowUtlobs, self).__init__(model, "obs", fname, pname,
                                            add_to_package_list, parent_file)        

        # set up variables
        self.precision = self.build_mfdata("precision",  precision)
        self.digits = self.build_mfdata("digits",  digits)
        self.print_input = self.build_mfdata("print_input",  print_input)
        self.continuousrecarray = self.build_mfdata("continuousrecarray", 
                                                    continuousrecarray)
