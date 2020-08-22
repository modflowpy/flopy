#!/bin/bash

sphinx-apidoc -e -o source/ ../flopy/
make html
