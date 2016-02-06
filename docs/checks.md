## List of available FloPy checks  

|Package  | Check | Implemented | Type |
| :-----------| :------------| :------------------ | :-------------|  
| mbase | unit number conflicts | :white_check_mark: | Error |
| mbase | compatible solver package | :white_check_mark: | Error |
| mbase | minimum packages needed to run the model | :x: | Error |
| mbase | overlapping boundary conditions | :x: | Error |
| all BC packages | NaN values in stress_period_data | :white_check_mark: | Error |
| all BC packages | valid indices for stress_period_data | :white_check_mark: | Error |
| LPF/UPW | hk or vka <=0 | :white_check_mark: | Error |
| LPF/UPW | hani < 0 | :white_check_mark: | Error |
| LPF/UPW | vkcb (quasi-3D kv values) <=0 | :white_check_mark: | Error |
| LPF/UPW | unusually high or low values in hk and vka arrays | :white_check_mark: | Warning |
| LPF/UPW | unusually high or low values in vkcb (quasi-3D kv values) | :white_check_mark: | Warning |
| LPF/UPW | storage values <=0 (transient only) | :white_check_mark: | Error |
| LPF/UPW | unusual values of storage (transient only) | :white_check_mark: | Error |
| RIV/SFR/STR | check for surface water BCs in confined layers | :x: | Warning |
| BAS | isolated cells | :white_check_mark: | Warning |
| BAS | NaN values | :white_check_mark: | Error |
| DIS | cell thicknesses <= 0 | :white_check_mark: | Error |
| DIS | cell thicknesses < thin_cell_threshold (default 1.0) | :white_check_mark: | Warning |
| DIS | NaN values in top and bottom arrays | :white_check_mark: | Error |
| DIS | discretization that violates the 1.5 rule | :x: | Warning |
| DIS | large changes in elevation | :x: | Warning |
| DISU | large changes in elevation | :x: | Warning |
| DISU | cell thicknesses <= 0 | :x: | Error |
| DISU | cell thicknesses < thin_cell_threshold (default 1.0) | :x: | Warning |
| DISU | NaN values in top and bottom arrays | :x: | Error |
| DISU | discretization that violates the 1.5 rule | :x: | Warning |
| DISU | large changes in elevation | :x: | Warning |
| RCH | unusually high or low R/T ratios | :white_check_mark: | Warning |
| RCH | NRCHOP not specified as 3 | :white_check_mark: | Warning |
| SFR | continuity in segment and reach numbering | :white_check_mark: | Error |
| SFR | segment number decreases in downstream direction | :white_check_mark: | Warning |
| SFR | circular routing | :white_check_mark: | Error |
| SFR | multiple non-zero conductances in a model cell | :white_check_mark: | Warning |
| SFR | elevation increases in the downstream direction | :white_check_mark: | Error |
| SFR | streambed elevations above model top | :white_check_mark: | Warning |
| SFR | streambed elevations below cell bottom | :white_check_mark: | Error |
| SFR | negative stream depth when icalc=0 | :x: | Error |
| SFR | slopes above or below specified threshold | :white_check_mark: | Warning |
| SFR | unusual values for manning's roughness and unit constant | :x: | Warning |
| SFR | gaps in segment and reach routing | :x: | Warning |
| SFR | outlets in interior of model domain | :x: | Warning |
| WEL | PHIRAMP is < 1 and should be close to recommended value of 0.001 | :x: | Warning |


## Visualizations

|Package  | Check | Implemented | Type |
| :-----------| :------------| :------------------ | :-------------|  
| All | Shapefile with detected errors | :x: | Information |
| All | Shapefile with detected warnings | :x: | Information |
| SFR/STR | Segment Connectivity | :x: | Information |
| SFR/STR | Identification of diversions | :x: | Information |
| SFR/STR | Identification of outlet tributaries | :x: | Information |


## Additional checks and visualizations 

Please submit additional proposed checks as issues on the FloPy developmen branch on [github](https://github.com/modflowpy/flopy/tree/develop).

