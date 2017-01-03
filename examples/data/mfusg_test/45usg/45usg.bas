#SAMPLE----3 LAYERS, 3 ROWS, 5 COLUMNS; STEADY STATE; CONSTANT HEADS COLUMN 1
#LAYERS 1 AND 2; RECHARGE, WELLS AND DRAINS
FREE UNSTRUCTURED
INTERNAL          1 (5I4)    3  IBOUND layer 1
  -1   1   1   1   1
  -1   1   1   1   1
  -1   1   1   1   1
INTERNAL          1 (5I4)    3  IBOUND layer 2
  -1   1   1   1   1
  -1   1   1   1   1
  -1   1   1   1   1
INTERNAL          1 (5I4)    3  IBOUND layer 3
   1   1   1   1   1
   1   1   1   1   1
   1   1   1   1   1
    999.99  HNOFLO
CONSTANT   0.000000E+00  SHEAD
CONSTANT   0.000000E+00  SHEAD
CONSTANT   0.000000E+00  SHEAD
