# The ATTRIBUTES block is required.
# Number of names on NAME line indicates 
#     number of time series to be created.
# NAME must precede METHOD (or METHODS).
# Either METHOD or METHODS must be specified, but not both.
# If METHOD is specified, all time series in file
#     share the same method (either LINEAR or STEPWISE).
# IF METHODS is specified, a method is specified for each time series.
#
BEGIN ATTRIBUTES
  NAME    ghbh_1 ghbh_2 ghbc_1 ghbc_2
  METHODS linear linear linear linear
END ATTRIBUTES

BEGIN TIMESERIES
#   time  ghbh_1  ghbh_2 ghbc_1 ghbc_2
     0.0    15.0    -5.0    500    100 
    10.0    10.0     0.0    400    200
    20.0     5.0     5.0    300    300
    30.0     0.0    10.0    200    400
END TIMESERIES
