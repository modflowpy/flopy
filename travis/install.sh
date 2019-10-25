#!/bin/bash
set -e

echo "Installing pip for Python ${TRAVIS_PYTHON_VERSION} ${RUN_TYPE} run"
pip install --upgrade pip
if [ "${TRAVIS_PYTHON_VERSION}" = "2.7" ]; then
  pip install -r requirements27.travis.txt
else
  pip install -r requirements.travis.txt
  pip install rasterio
  pip install --upgrade numpy
fi
if [ "${RUN_TYPE}" = "misc" ]; then
  pip install flake8 pylint pylint-exit
  pip install jupyter nbconvert
fi
pip install https://github.com/modflowpy/pymake/zipball/master
pip install shapely[vectorize]
pip install coveralls nose-timer
