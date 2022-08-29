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
bindir.mkdir(exist_ok=True)
flopy.utils.get_modflow(bindir)
list(bindir.iterdir())

# Or use an auto-select option
flopy.utils.get_modflow(":flopy")
```

## Where to install?

A required `bindir` parameter must be supplied to the utility, which specifies where to install the programs. This can be any existing directory, usually which is on the users' PATH environment variable.

To assist the user, special values can be specified starting with the colon character. Use a single `:` to interactively select an option of paths.

Other auto-select options are only available if the current user can write files (some may require `sudo` for Linux or macOS):
 - `:prev` - if this utility was run by FloPy more than once, the first option will be the previously used `bindir` path selection
 - `:flopy` - special option that will create and install programs for FloPy
 - `:python` - use Python's bin (or Scripts) directory
 - `:local` - use `$HOME/.local/bin`
 - `:system` - use `/usr/local/bin`
 - `:windowsapps` - use `%LOCALAPPDATA%\Microsoft\WindowsApps`
