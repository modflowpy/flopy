##Working list of FloPy checks

###Model level checks  
**Errors** 

* unit numbers conflicts
* check for solver package compatible with model version
* minimum packages needed to run the model

###Package level checks
####All packages
**Errors**   

* Check for NaN values  
**Warnings**   


####All BC packages
**Errors**
  
* check that head/stage not set below cell bottom
* zero or negative conductance values
* overlapping boundary conditions

**Warnings**   

* BCs in inactive cells
* check that BCs are in all stress periods
* check if surface water BCs are in a confined layer

####BAS
**Errors**  
 

**Warnings**   


* check for isolated cells (surrounded by inactive)

####DIS
**Errors**   

* Active cells whose top elevations are less than or equal to their bottom elevations.

**Warnings**   

* check for large changes in elevation
* Warn if cell thicknesses are less than a specified threshold (default 1.0)
* Warn if adjacent cell spacings differ by more than 50% (1.5 rule, e.g. Anderson, Woessner and Hunt 2015, p189-191)


####LPF/UPW  
**Errors**  

* check all properties for zero or negative values  

**Warnings**   

* check that K and storage are approximately within their natural ranges
     * Are there published reasonable ranges for anisotropy? Is this worth putting in the checker?
* check for unconfined layer below confined layer

####MNW

####Recharge
**Warnings**   
* Warn if R/T ratio unusually high or low
* warn if NRCHOP != 3

####SFR, STR, SWR  
**Errors**  

* In the STR package, the segments must be numbered such that lower number segments only flow into higher number segments.  

**Warnings**  

* Streams that flow uphill
* Sequential reaches in a stream segment that are not in adjacent cells.
* Linked stream segments where the linked reaches are not in adjacent cells.
* geographic proximity of segments and reaches
* check for reasonable values of slope, Manning's roughness coefficient, and unit constant

####Well  
**Warnings**   

* PHIRAMP is < 1 and should be close to recommended value of 0.001 

###Visualizations  
shapefiles:  

* general method for writting check summary table to shapefile
* SFR/STR, and SWR?
	* 	segment linkages and outlets
	  
	* 	outlet tributaries by color