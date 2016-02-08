FloPy Model Checks
-----------------------------------------------

## List of available FloPy model checks  

|Package  | Check | Implemented | Type |
| :-----------| :------------| :------------------ | :-------------|  
| NAM | unit number conflicts | Supported | Error |
| NAM | compatible solver package | Supported | Error |
| NAM | minimum packages needed to run the model | Not supported | Error |
| all BC packages | overlapping boundary conditions | Not supported | Error |
| all BC packages | NaN values in stress_period_data | Supported | Error |
| all BC packages | valid indices for stress_period_data | Supported | Error |
| LPF/UPW | hk or vka <=0 | Supported | Error |
| LPF/UPW | hani < 0 | Supported | Error |
| LPF/UPW | vkcb (quasi-3D kv values) <=0 | Supported | Error |
| LPF/UPW | unusually high or low values in hk and vka arrays | Supported | Warning |
| LPF/UPW | unusually high or low values in vkcb (quasi-3D kv values) | Supported | Warning |
| LPF/UPW | storage values <=0 (transient only) | Supported | Error |
| LPF/UPW | unusual values of storage (transient only) | Supported | Error |
| RIV/SFR/STR | check for surface water BCs in confined layers | Not supported | Warning |
| BAS | isolated cells | Supported | Warning |
| BAS | NaN values | Supported | Error |
| DIS | cell thicknesses <= 0 | Supported | Error |
| DIS | cell thicknesses < thin_cell_threshold (default 1.0) | Supported | Warning |
| DIS | NaN values in top and bottom arrays | Supported | Error |
| DIS | discretization that violates the 1.5 rule | Not supported | Warning |
| DIS | large changes in elevation | Not supported | Warning |
| DISU | large changes in elevation | Not supported | Warning |
| DISU | cell thicknesses <= 0 | Not supported | Error |
| DISU | cell thicknesses < thin_cell_threshold (default 1.0) | Not supported | Warning |
| DISU | NaN values in top and bottom arrays | Not supported | Error |
| DISU | discretization that violates the 1.5 rule | Not supported | Warning |
| DISU | large changes in elevation | Not supported | Warning |
| RCH | unusually high or low R/T ratios | Supported | Warning |
| RCH | NRCHOP not specified as 3 | Supported | Warning |
| SFR | continuity in segment and reach numbering | Supported | Error |
| SFR | segment number decreases in downstream direction | Supported | Warning |
| SFR | circular routing | Supported | Error |
| SFR | multiple non-zero conductances in a model cell | Supported | Warning |
| SFR | elevation increases in the downstream direction | Supported | Error |
| SFR | streambed elevations above model top | Supported | Warning |
| SFR | streambed elevations below cell bottom | Supported | Error |
| SFR | negative stream depth when icalc=0 | Not supported | Error |
| SFR | slopes above or below specified threshold | Supported | Warning |
| SFR | unusual values for manning's roughness and unit constant | Not supported | Warning |
| SFR | gaps in segment and reach routing | Not supported | Warning |
| SFR | outlets in interior of model domain | Not supported | Warning |
| WEL | PHIRAMP is < 1 and should be close to recommended value of 0.001 | Not supported | Warning |


## Visualizations

|Package  | Check | Implemented | Type |
| :-----------| :------------| :------------------ | :-------------|  
| All | Shapefile with detected errors | Not supported | Information |
| All | Shapefile with detected warnings | Not supported | Information |
| SFR/STR | Segment Connectivity | Not supported | Information |
| SFR/STR | Identification of diversions | Not supported | Information |
| SFR/STR | Identification of outlet tributaries | Not supported | Information |


## Additional model checks and visualizations 

Please submit additional proposed model checks as issues on the FloPy development branch on [github](https://github.com/modflowpy/flopy/tree/develop).

