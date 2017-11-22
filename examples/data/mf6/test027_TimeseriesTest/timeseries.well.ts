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
  NAME     wrate_1   wrate_2
  METHODS  stepwise  linear
END ATTRIBUTES

BEGIN TIMESERIES
#   time  wrate_1  wrate_2
     0.0   -1000.   -2000.
    50.0   -1000.   -2050.
END TIMESERIES
