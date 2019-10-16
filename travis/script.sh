#!/bin/bash
set -e

echo "Building executables..."
nosetests -v build_exes.py --with-id --with-timer -w ./autotest

if [ "${RUN_TYPE}" = "test" ]; then
  echo "Running flopy autotest suite..."
  nosetests -v --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
elif [ "${RUN_TYPE}" = "misc" ]; then
  echo "Checking Python code..."
  pylint --jobs=2 --errors-only ./flopy || pylint-exit $?
  echo "TODO: exit code was $?"
  echo "Running notebook autotest suite..."
  nosetests -v autotest_scripts.py --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
  nosetests -v autotest_notebooks.py --with-id --with-timer -w ./autotest \
    --with-coverage --cover-package=flopy
else
  echo "Unhandled RUN_TYPE=${RUN_TYPE}"
  exit 1
fi
