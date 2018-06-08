"""
The FloPy package consists of a set of Python scripts to run MODFLOW, MT3D,
SEAWAT and other MODFLOW-related groundwater programs. FloPy enables you to
run all these programs with Python scripts. The FloPy project started in 2009
and has grown to a fairly complete set of scripts with a growing user base.

This version of Flopy (FloPy3) was released in December 2015 with a few great
enhancements that make FloPy3 backwards incompatible. The first significant
change is that FloPy3 uses zero-based indexing everywhere, which means that
all layers, rows, columns, and stress periods start numbering at zero. This
change was made for consistency as all array-indexing was already zero-based
(as are all arrays in Python). This may take a little getting-used-to, but
hopefully will avoid confusion in the future. A second significant enhancement
concerns the ability to specify time-varying boundary conditions that are
specified with a sequence of layer-row-column-values, like the WEL and GHB
packages. A variety of flexible and readable ways have been implemented to
specify these boundary conditions. FloPy is an open-source project and any
assistance is welcomed. Please email the development team if you want to
contribute.

"""

__name__ = 'flopy'
__author__ = 'Mark Bakker, Vincent Post, Christian D. Langevin, ' + \
             'Joseph D. Hughes, Jeremy T. White, Andrew T. Leaf, ' + \
             'Scott R. Paulinski, Jeffrey J. Starn, and Michael N. Fienen'
from .version import __version__, __build__, __git_commit__

# imports
from . import modflow
from . import mt3d
from . import seawat
from . import modpath
from . import modflowlgr
from . import utils
from . import plot
from . import export
from . import pest
from . import mf6
from .mbase import run_model, which
