#!/bin/bash
set -e

if [[ ! -d "$HOME/.local/bin" ]]; then
  mkdir "$HOME/.local/bin"
fi
echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"

echo "Adding directory for downloaded executables to path..."
ls -l "$HOME/.local/bin"

echo "Showing version information..."
gfortran --version
gcc --version
g++ --version
python -c "import sys; print('python sys.path: {}'.format(sys.path))"
python -c "import os; is_travis = 'TRAVIS' in os.environ; print('TRAVIS {}'.format(is_travis))"
python -c "import flopy; print('flopy: {}'.format(flopy.__version__))"
python -c "import numpy; print('numpy: {}'.format(numpy.version.version))"
python -c "import pandas as pd; print('pandas: {}'.format(pd.__version__))"
python -c "import requests; print('requests: {}'.format(requests.__version__))"
python -c "import shapefile; print('pyshp: {}'.format(shapefile.__version__))"
if [ "${RUN_TYPE}" = "misc" ]; then
  echo "jupyter: $(jupyter --version); $(jupyter --runtime-dir)"
  pylint --version
fi
nosetests --version
