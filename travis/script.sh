#!/bin/bash
set -e

echo "Downloading executables..."
python ./autotest/get_exes.py

if [ "${RUN_TYPE}" = "test" ]; then
  echo "Running flopy autotest suite..."
  nosetests -v --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
elif [ "${RUN_TYPE}" = "misc" ]; then
  echo "Checking Python code with flake8..."
  flake8 --exit-zero
  echo "Checking Python code with pylint..."
  pylint --jobs=2 --errors-only ./flopy || pylint-exit $?
  if [ $? -ne 0 ]; then
    echo "An error occurred while running pylint." >&2
    exit 1
  fi
  echo "Running notebook autotest suite..."
  nosetests -v autotest_scripts.py --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
  nosetests -v autotest_notebooks.py --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
else
  echo "Unhandled RUN_TYPE=${RUN_TYPE}" >&2
  exit 1
fi
