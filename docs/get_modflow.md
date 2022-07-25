# Install MODFLOW and related programs

This method describes how to install USGS MODFLOW and related programs for Windows, Mac or Linux using a "get modflow" utility. If FloPy is installed, the utility is available in the Python environment as a `get-modflow` command. The same utility is also available as a Python script `get_modflow.py`, described later.

The utility uses a [GitHub releases API](https://docs.github.com/en/rest/releases) to download versioned archives of programs that have been compiled with modern Intel Fortran compilers. The utility is able to match the binary archive to the operating system, and extract the console programs to a user-defined directory. A prompt can also be used to assist where to install programs.

## Command-line interface

### Using `get-modflow` from FloPy

When FloPy is installed, a `get-modflow` (or `get-modflow.exe` for Windows) program is installed, which is usually installed to the PATH (depending on the Python setup). From a console:

```console
$ get-modflow --help
usage: get-modflow [-h]
...
```

### Using `get_modflow.py` as a script

The script requires Python 3.6 or later and does not have any dependencies, not even FloPy. It can be downloaded separately and used the same as the console program, except with a different invocation. For example:

```console
$ wget https://raw.githubusercontent.com/modflowpy/flopy/develop/flopy/utils/get_modflow.py
$ python3 get_modflow.py --help
usage: get_modflow.py [-h]
...
```

## FloPy module

The same functionality of the command-line interface is available from the FloPy module, as demonstrated below:

```python
from pathlib import Path
import flopy

bindir = Path("/tmp/bin")
bindir.mkdir()
flopy.utils.get_modflow_main(bindir)
list(bindir.iterdir())
```
