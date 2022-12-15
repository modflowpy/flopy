# Install MODFLOW and related programs

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Command-line interface](#command-line-interface)
  - [Using the `get-modflow` command](#using-the-get-modflow-command)
  - [Using `get_modflow.py` as a script](#using-get_modflowpy-as-a-script)
- [FloPy module](#flopy-module)
- [Where to install?](#where-to-install)
- [Selecting a distribution](#selecting-a-distribution)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

FloPy includes a `get-modflow` utility to install USGS MODFLOW and related programs for Windows, Mac or Linux. If FloPy is installed, the utility is available in the Python environment as a `get-modflow` command. The script `flopy/utils/get_modflow.py` has no dependencies and can be invoked independently.

The utility uses the [GitHub releases API](https://docs.github.com/en/rest/releases) to download versioned archives containing executables compiled with [Intel Fortran](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html). The utility is able to match the binary archive to the operating system and extract the console programs to a user-defined directory. A prompt can also be used to help the user choose where to install programs.

## Command-line interface

### Using the `get-modflow` command

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
 - `:home` - use `$HOME/.local/bin`
 - `:system` - use `/usr/local/bin`
 - `:windowsapps` - use `%LOCALAPPDATA%\Microsoft\WindowsApps`

## Selecting a distribution

By default the distribution from the [executables repository](https://github.com/MODFLOW-USGS/executables) is installed. This includes the MODFLOW 6 binary `mf6` and over 20 other related programs. The utility can also install from the main [MODFLOW 6 repo](https://github.com/MODFLOW-USGS/modflow6) or the [nightly build](https://github.com/MODFLOW-USGS/modflow6-nightly-build) distributions, which contain only:

- `mf6`
- `mf5to6`
- `zbud6`
- `libmf6.dylib`

To select a distribution, specify a repository name with the `--repo` command line option or the `repo` function argument. Valid names are:

- `executables` (default)
- `modflow6`
- `modflow6-nightly-build`
