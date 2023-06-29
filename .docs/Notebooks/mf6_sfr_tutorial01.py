# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   metadata:
#     section: mf6
# ---

# # SFR2 package loading and querying

import os

# +
import sys

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

print(sys.version)
print("flopy version: {}".format(flopy.__version__))
# -

# Create a model instance

m = flopy.modflow.Modflow()

# Read the SFR2 file

f = os.path.join(
    "..", "..", "examples", "data", "mf2005_test", "testsfr2_tab_ICALC2.sfr"
)
stuff = open(f).readlines()
stuff

# Load the SFR2 file

sfr = flopy.modflow.ModflowSfr2.load(f, m, nper=50)

sfr.segment_data.keys()

# Query the reach data in the SFR2 file

sfr.reach_data

# Query the channel flow data in the SFR2 file

sfr.channel_flow_data

# Query the channel geometry data in the SFR2 file

sfr.channel_geometry_data

# Query dataset 5 data in the SFR2 file

sfr.dataset_5

# Query the TABFILES dictionary in the SFR2 object to determine the TABFILES data in the SFR2 file

sfr.tabfiles_dict

# + pycharm={"name": "#%%\n"}
sfr.tabfiles
