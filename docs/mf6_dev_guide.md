Introduction
-----------------------------------------------

This file provides an overview of how FloPy for MODFLOW 6 (FPMF6) works under the hood and is intended for anyone who wants to add a new package, new model type, or new features to this library.  FloPy library files that support MODFLOW 6 can be found in the flopy/mf6 folder and sub-folders. 

Package Meta-Data and Package Files
-----------------------------------------------

FPMF6 uses meta-data files located in flopy/mf6/data/dfn to define the model and package types supported by MODFLOW 6.  When additional model and package types are added to MODFLOW 6, additional meta-data files can be added to this folder and flopy/mf6/utils/createpackages.py can be run to add new packages to the FloPy library.  createpackages.py uses flopy/mf6/data/mfstructure.py to read meta-data files (*.dfn) and use that meta-data to create the package files found in flopy/mf6/modflow (do not directly modify any of the files in this folder, they are all automatically generated).  The automatically generated package files contain an interface for accessing package data and data documentation generated from the meta-data files.  Additionally, meta-data describing package data types and shapes is stored in the dfn attribute.  flopy/mf6/data/mfstructure.py can load structure information using the dfn attribute (instead of loading it from the meta-data files).  This allows for flopy to be installed without the dfn files.

All meta-data can be accessed from the flopy.mf6.data.mfstructure.MFStructure class.  This is a singleton class, meaning only one instance of this class can be created.  The class contains a sim_struct attribute (which is a flopy.mf6.data.mfstructure.MFSimulationStructure object) which contains all of the meta-data for all package files.  Meta-data is stored in a structured format. MFSimulationStructure contains MFModelStructure and MFInputFileStructure objects, which contain the meta-data for each model type and each "simulation-level" package (tdis, ims, ...).  MFModelStructure contains model specific meta-data and a MFInputFileStructure object for each package in that model.  MFInputFileStructure contains package specific meta-data and a MFBlockStructure object for each block contained in the package file.  MFBlockStructure contains block specific meta-data and a MFDataStructure object for each data structure defined in the block, and MFDataStructure contains data structure specific meta-data and a MFDataItemStructure object for each data item contained in the data structure.  Data structures define the structure of data that is naturally grouped together, for example, the data in a numpy recarray.  Data item structures define the structure of specific pieces of data, for example, a single column of a numpy recarray.  The meta-data defined in these classes provides all the information FloPy needs to read and write MODFLOW 6 package and name files, create the Flopy interface, and check the data for various constraints.


***
MFStructure --+ MFSimulationStructure --+ MFModelStructure --+ MFInputFileStructure --+ MFBlockStructure --+ MFDataStructure --+ MFDataItemStructure

Figure 1: FPMF6 generic data structure classes.  Lines connecting classes show a relationship defined between the two connected classes.  A "*" next to the class means that the  class is a sub-class of the connected class.  A "+" next to the class means that the class is contained within the connected class.
***

Package and Data Base Classes
-----------------------------------------------

The package and data classes are related as shown below in figure 2.  On the top of the figure 2 is the MFPackage class, which is the base class for all packages.  MFPackage contains generic methods for building data objects and reading and writing the package to a file.  MFPackage contains a MFInputFileStructure object that defines how the data is structured in the package file.  MFPackage also contains a dictionary of blocks (MFBlock).  The MFBlock class is a generic class used to represent a block within a package.  MFBlock contains a MFBlockStructure object that defines how the data in the block is structured.  MFBlock also contains a dictionary of data objects (subclasses of MFData) contained in the block and a list of block headers (MFBlockHeader) for that block.  Block headers contain the block's name and optionally data items (eg. iprn).


***
MFPackage --+ MFBlock --+ MFData

MFPackage --+ MFInputFileStructure

MFBlock --+ MFBlockStructure

MFData --+ MFDataStructure

MFData --* MFArray --* MFTransientArray

MFData --* MFList --* MFTransientList

MFData --* MFScalar --* MFTransientScalar

MFTransientData --* MFTransientArray, MFTransientList, MFTransientScalar
							 
Figure 2:  FPMF6 package and data classes.  Lines connecting classes show a relationship defined between the two connected classes.  A "*" next to the class means that the  class is a sub-class of the connected class.  A "+" next to the class means that the class is contained within the connected class.
***

There are three main types of data, MFList, MFArray, and MFScalar data.  All three of these data types are derived from the MFData abstract base class.  MFList data is the type of data stored in a spreadsheet with different column headings.  For example, the data describing a flow barrier are of type MFList.  MFList data is stored in numpy recarrays.  MFArray data is data of a single type (eg. all integer values).  For example, the model's HK values are of type MFArray.  MFArrays are stored in numpy ndarrays.  MFScalar data is a single data item.  Most MFScalar data are options.  All MFData subclasses contain an MFDataStructure object that defines the expected structure and types of the data.

Transient data, or data defined for each stress period (eg. data in the period blocks) is stored in MFTransientArray, MFTransientList, and MFTransientScalar.  These classes are sub-classes of MFArray, MFList, and MFScalar, respectively.  These classes are also subclasses of MFTransientData.
