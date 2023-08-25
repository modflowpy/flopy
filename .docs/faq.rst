Frequently asked questions
==========================

Executable does not exist
-------------------------

One common error is missing executables. FloPy can drive MODFLOW and a number of related progams. In order for you to run a model you have created or loaded using FloPy, the desired executable must be available in your path. Executable paths must either be provided explicitly, or if an executable name is provided, the binary must be on the system path.

You can test if the executable is available using the `which` function. For example, to test if the `mf2005` executable is available in your path, use:

.. code-block:: Python

    flopy.which("mf2005")  # equivalent to shutil.which("mf2005")

If the executable is found, its path is returned, otherwise the function returns an empty string. If you receive the latter, you should:

#. Check the that you have spelled the executable correctly.
#. If you have spelled the executable correctly then you need to move the executable into your working directory or into a directory in your path.
#. If you have spelled the executable correctly but don't have the executable. Your options are:
     * Download a precompiled version of the executable. Precompiled versions of MODFLOW-based codes are available from the U.S. Geological Survey for the Windows operating system. 
     * Compile the source code (available from the U.S. Geological Survey) for the Windows, OS X, Linux, and UNIX operating systems and place the compiled executable in the working directory or a directory contained in your path (for example, `/Users/jdhughes/.local/bin/` as indicated above).

You can get a list of the directories in your system path using:

.. code-block:: Python

    os.getenv("PATH")

There is a `get-modflow` command line utility that can be used to download MODFLOW executables. See the `get-modflow documentation <md/get_modflow>`_ for more information.
