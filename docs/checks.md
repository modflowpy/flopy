##Working list of FloPy checks  
Please add checks and/or visualizations that you would like to see implemented!



|Package  | Check | Implemented | Type |
| :-----------| :------------| :------------------ | :-------------|  
| mbase | unit number conflicts | :white_check_mark: | Error |
| mbase | compatible solver package | :white_check_mark: | Error |
| mbase | minimum packages needed to run the model | :x: | Error |
| mbase | overlapping boundary conditions | :x: | Error |
| pakbase (all BC packages) | NaN values in stress_period_data | :white_check_mark: | Error |
| pakbase (all BC packages)| valid indices for stress_period_data | :white_check_mark: | Error |
| pakbase (all BC packages)| stress_period_data in inactive cells | :white_check_mark: | Warning |
| pakbase (LPF and UPW)| hk or vka <=0 | :white_check_mark: | Error |
| pakbase (LPF and UPW)| hani < 0 | :white_check_mark: | Error |
| pakbase (LPF and UPW)| vkcb (quasi-3D kv values) <=0 | :white_check_mark: | Error |
| pakbase (LPF and UPW)| unusually high or low values in hk and vka arrays | :white_check_mark: | Warning |
| pakbase (LPF and UPW)| unusually high or low values in vkcb (quasi-3D kv values) | :white_check_mark: | Warning |
| pakbase (LPF and UPW)| storage values <=0 (transient only) | :white_check_mark: | Error |
| pakbase (LPF and UPW)| unusual values of storage (transient only) | :white_check_mark: | Error |
| pakbase (LPF and UPW)| convertible layers below confined layers | :white_check_mark: | Warning |
| pakbase | check for surface water BCs in confined layers | :x: | Warning |
| bas6 | isolated cells | :white_check_mark: | Warning |
| bas6 | NaN values | :white_check_mark: | Error |
| DIS | cell thicknesses <= 0 | :white_check_mark: | Error |
| DIS | cell thicknesses < thin_cell_threshold (default 1.0) | :white_check_mark: | Warning |
| DIS | NaN values in top and bottom arrays | :white_check_mark: | Error |
| DIS | discretization that violates the 1.5 rule | :x: | Warning |
| DIS | large changes in elevation | :x: | Warning |
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



###Visualizations (not implemented yet)
shapefiles:  

* general method for writting check summary table to shapefile
* SFR/STR, and SWR?
	* 	segment linkages and outlets
	  
	* 	outlet tributaries by color