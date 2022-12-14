#!/usr/bin/env python3
"""Download and install USGS MODFLOW and related programs.

This script originates from FloPy: https://github.com/modflowpy/flopy
This file can be downloaded and run independently outside FloPy.
It requires Python 3.6 or later, and has no dependencies.

See https://developer.github.com/v3/repos/releases/ for GitHub Releases API.
"""
import json
import os
import sys
import tempfile
import urllib
import urllib.request
import zipfile
from importlib.util import find_spec
from pathlib import Path

__all__ = ["run_main"]
__license__ = "CC0"

owner = "MODFLOW-USGS"
# key is the repo name, value is the renamed file prefix for the download
renamed_prefix = {
    "executables": "modflow_executables",
    "modflow6-nightly-build": "modflow6_nightly",
}
available_repos = list(renamed_prefix.keys())
available_ostags = ["linux", "mac", "win32", "win64"]
max_http_tries = 3

# Check if this is running from flopy
within_flopy = False
spec = find_spec("flopy")
if spec is not None:
    within_flopy = (
        Path(spec.origin).resolve().parent in Path(__file__).resolve().parents
    )
del spec


def get_ostag():
    """Determine operating system tag from sys.platform."""
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win"):
        return "win" + ("64" if sys.maxsize > 2**32 else "32")
    elif sys.platform.startswith("darwin"):
        return "mac"
    raise ValueError(f"platform {sys.platform!r} not supported")


def get_request(url):
    """Get urllib.request.Request, with headers.

    This bears GITHUB_TOKEN if it is set as an environment variable.
    """
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return urllib.request.Request(url, headers=headers)


def get_avail_releases(api_url):
    """Get list of available releases."""
    req_url = f"{api_url}/releases"
    request = get_request(req_url)
    num_tries = 0
    while True:
        num_tries += 1
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                result = resp.read()
                break
        except urllib.error.HTTPError as err:
            if err.code == 401 and os.environ.get("GITHUB_TOKEN"):
                raise ValueError("GITHUB_TOKEN env is invalid") from err
            elif err.code == 403 and "rate limit exceeded" in err.reason:
                raise ValueError(
                    f"use GITHUB_TOKEN env to bypass rate limit ({err})"
                ) from err
            elif err.code in (404, 503) and num_tries < max_http_tries:
                # GitHub sometimes returns this error for valid URLs, so retry
                print(f"URL request {num_tries} did not work ({err})")
                continue
            raise RuntimeError(f"cannot retrieve data from {req_url}") from err

    releases = json.loads(result.decode())
    avail_releases = ["latest"]
    avail_releases.extend(release["tag_name"] for release in releases)
    return avail_releases


def columns_str(items, line_chars=79):
    """Return str of columns of items, similar to 'ls' command."""
    item_chars = max(len(item) for item in items)
    num_cols = line_chars // item_chars
    num_rows = len(items) // num_cols
    if len(items) % num_cols != 0:
        num_rows += 1
    lines = []
    for row_num in range(num_rows):
        row_items = items[row_num::num_rows]
        lines.append(
            " ".join(item.ljust(item_chars) for item in row_items).rstrip()
        )
    return "\n".join(lines)


def run_main(
    bindir,
    repo="executables",
    release_id="latest",
    ostag=None,
    subset=None,
    downloads_dir=None,
    force=False,
    quiet=False,
    _is_cli=False,
):
    """Run main method to get MODFLOW and related programs.

    Parameters
    ----------
    bindir : str or Path
        Writable path to extract executables. Auto-select options start with a
        colon character. See error message or other documentation for further
        information on auto-select options.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default) or
        "modflow6-nightly-build".
    release_id : str, default "latest"
        GitHub release ID.
    ostag : str, optional
        Operating system tag; default is to automatically choose.
    subset : list, set or str, optional
        Optional subset of executables to extract, specified as a list (e.g.)
        ``["mfnwt", "mp6"]`` or a comma-separated string "mfnwt,mp6".
    downloads_dir : str or Path, optional
        Manually specify directory to download archives. Default is to use
        home Downloads, if available, otherwise a temporary directory.
    force : bool, default False
        If True, always download archive. Default False will use archive if
        previously downloaded in ``downloads_dir``.
    quiet : bool, default False
        If True, show fewer messages.
    _is_cli : bool, default False
        Control behavior of method if this is run as a command-line interface
        or as a Python function.
    """
    meta_path = False
    prev_bindir = None
    flopy_bin = False
    if within_flopy:
        meta_list = []
        # Store metadata and possibly 'bin' in a user-writable path
        if sys.platform.startswith("win"):
            flopy_appdata = Path(os.path.expandvars(r"%LOCALAPPDATA%\flopy"))
        else:
            flopy_appdata = Path.home() / ".local" / "share" / "flopy"
        if not flopy_appdata.exists():
            flopy_appdata.mkdir(parents=True, exist_ok=True)
        flopy_bin = flopy_appdata / "bin"
        meta_path = flopy_appdata / "get_modflow.json"
        meta_path_exists = meta_path.exists()
        if meta_path_exists:
            del_meta_path = False
            try:
                meta_list = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as err:
                print(f"cannot read flopy metadata file '{meta_path}': {err}")
                if isinstance(err, OSError):
                    meta_path = False
                if isinstance(err, json.JSONDecodeError):
                    del_meta_path = True
            try:
                prev_bindir = Path(meta_list[-1]["bindir"])
            except (KeyError, IndexError):
                del_meta_path = True
            if del_meta_path:
                try:
                    meta_path.unlink()
                    meta_path_exists = False
                    print(f"removed corrupt flopy metadata file '{meta_path}'")
                except OSError as err:
                    print(f"cannot remove flopy metadata file: {err!r}")
                    meta_path = False

    if ostag is None:
        ostag = get_ostag()
    exe_suffix = ""
    if ostag in ["win32", "win64"]:
        exe_suffix = ".exe"
        lib_suffix = ".dll"
    elif ostag == "linux":
        lib_suffix = ".so"
    elif ostag == "mac":
        lib_suffix = ".dylib"
    else:
        raise KeyError(
            f"unrecognized ostag {ostag!r}; choose one of {available_ostags}"
        )

    if isinstance(bindir, Path):
        pass
    elif bindir.startswith(":"):
        options = {}  # key is an option name, value is (optpath, optinfo)
        if prev_bindir is not None and os.access(prev_bindir, os.W_OK):
            # Make previous bindir as the first option
            options[":prev"] = (prev_bindir, "previously selected bindir")
        if within_flopy:  # don't check is_dir() or access yet
            options[":flopy"] = (flopy_bin, "used by FloPy")
        # Python bin (same for standard or conda varieties)
        py_bin = Path(sys.prefix) / (
            "Scripts" if ostag.startswith("win") else "bin"
        )
        if py_bin.is_dir() and os.access(py_bin, os.W_OK):
            options[":python"] = (py_bin, "used by Python")
        home_local_bin = Path.home() / ".local" / "bin"
        if home_local_bin.is_dir() and os.access(home_local_bin, os.W_OK):
            options[":home"] = (home_local_bin, "user-specific bindir")
        local_bin = Path("/usr") / "local" / "bin"
        if local_bin.is_dir() and os.access(local_bin, os.W_OK):
            options[":system"] = (local_bin, "system local bindir")
        # Windows user
        windowsapps_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps")
        )
        if windowsapps_dir.is_dir() and os.access(windowsapps_dir, os.W_OK):
            options[":windowsapps"] = (windowsapps_dir, "User App path")
            options.append(windowsapps_dir)
        # any other possible OS-specific hard-coded locations?
        if not options:
            raise RuntimeError("could not find any installable folders")
        opt_avail = ", ".join(
            f"'{opt}' for '{optpath}'" for opt, (optpath, _) in options.items()
        )
        if len(bindir) > 1:  # auto-select mode
            # match one option that starts with input, e.g. :Py -> :python
            sel = list(
                opt for opt in options if opt.startswith(bindir.lower())
            )
            if len(sel) != 1:
                if bindir == ":flopy":
                    raise ValueError("option ':flopy' is only for flopy")
                raise ValueError(f"invalid option, choose from: {opt_avail}")
            bindir = options[sel[0]][0]
            if not quiet:
                print(f"auto-selecting option {sel[0]!r} for '{bindir}'")
        elif not _is_cli:
            raise ValueError(f"specify the option, choose from: {opt_avail}")
        else:
            ioptions = dict(enumerate(options.keys(), 1))
            print("select a number to extract executables to a directory:")
            for iopt, opt in ioptions.items():
                optpath, optinfo = options[opt]
                print(f" {iopt}: '{optpath}' -- {optinfo} ('{opt}')")
            num_tries = 0
            while True:
                num_tries += 1
                res = input("> ")
                try:
                    opt = ioptions[int(res)]
                    print(f"selecting option {opt!r}")
                    bindir = options[opt][0]
                    break
                except (KeyError, ValueError):
                    if num_tries < 2:
                        print("invalid option, try choosing option again")
                    else:
                        raise RuntimeError(
                            "invalid option, too many attempts"
                        ) from None

    bindir = Path(bindir).resolve()
    if bindir == flopy_bin and not flopy_bin.exists():
        # special case option that can create non-existing directory
        flopy_bin.mkdir(parents=True, exist_ok=True)
    if not bindir.is_dir():
        raise OSError(f"extraction directory '{bindir}' does not exist")
    elif not os.access(bindir, os.W_OK):
        raise OSError(f"extraction directory '{bindir}' is not writable")

    if repo not in available_repos:
        raise KeyError(
            f"repo {repo!r} not supported; choose one of {available_repos}"
        )
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    if release_id == "latest":
        req_url = f"{api_url}/releases/latest"
    else:
        req_url = f"{api_url}/releases/tags/{release_id}"
    request = get_request(req_url)
    avail_releases = None
    num_tries = 0
    while True:
        num_tries += 1
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                result = resp.read()
                remaining = int(resp.headers["x-ratelimit-remaining"])
                if remaining <= 10:
                    print(
                        f"Only {remaining} GitHub API requests remaining "
                        "before rate-limiting"
                    )
                break
        except urllib.error.HTTPError as err:
            if err.code == 401 and os.environ.get("GITHUB_TOKEN"):
                raise ValueError("GITHUB_TOKEN env is invalid") from err
            elif err.code == 403 and "rate limit exceeded" in err.reason:
                raise ValueError(
                    f"use GITHUB_TOKEN env to bypass rate limit ({err})"
                ) from err
            elif err.code == 404:
                if avail_releases is None:
                    avail_releases = get_avail_releases(api_url)
                if release_id in avail_releases:
                    if num_tries < max_http_tries:
                        # GitHub sometimes returns 404 for valid URLs, so retry
                        print(f"URL request {num_tries} did not work ({err})")
                        continue
                else:
                    raise ValueError(
                        f"Release {release_id!r} not found -- "
                        f"choose from {avail_releases}"
                    ) from err
            elif err.code == 503 and num_tries < max_http_tries:
                # GitHub sometimes returns this error for valid URLs, so retry
                print(f"URL request {num_tries} did not work ({err})")
                continue
            raise RuntimeError(f"cannot retrieve data from {req_url}") from err

    release = json.loads(result.decode())
    tag_name = release["tag_name"]
    if not quiet:
        print(f"fetched release {tag_name!r} from {owner}/{repo}")

    assets = release.get("assets", [])
    for asset in assets:
        if ostag in asset["name"]:
            break
    else:
        raise ValueError(
            f"could not find ostag {ostag!r} from release {tag_name!r}; "
            f"see available assets here:\n{release['html_url']}"
        )
    asset_name = asset["name"]
    download_url = asset["browser_download_url"]
    # change local download name so it is more unique
    dst_fname = "-".join([renamed_prefix[repo], tag_name, asset_name])
    tmpdir = None
    if downloads_dir is None:
        downloads_dir = Path.home() / "Downloads"
        if not (downloads_dir.is_dir() and os.access(downloads_dir, os.W_OK)):
            tmpdir = tempfile.TemporaryDirectory()
            downloads_dir = Path(tmpdir.name)
    else:  # check user-defined
        downloads_dir = Path(downloads_dir)
        if not downloads_dir.is_dir():
            raise OSError(
                f"downloads directory '{downloads_dir}' does not exist"
            )
        elif not os.access(downloads_dir, os.W_OK):
            raise OSError(
                f"downloads directory '{downloads_dir}' is not writable"
            )
    download_pth = downloads_dir / dst_fname
    if download_pth.is_file() and not force:
        if not quiet:
            print(
                f"using previous download '{download_pth}' (use "
                f"{'--force' if _is_cli else 'force=True'!r} to re-download)"
            )
    else:
        if not quiet:
            print(f"downloading to '{download_pth}'")
        urllib.request.urlretrieve(download_url, download_pth)

    if subset:
        if isinstance(subset, str):
            subset = set(subset.replace(",", " ").split())
        elif not isinstance(subset, set):
            subset = set(subset)

    # Open archive and extract files
    extract = set()
    chmod = set()
    items = []
    if meta_path:
        from datetime import datetime

        meta = {
            "bindir": str(bindir),
            "owner": owner,
            "repo": repo,
            "release_id": tag_name,
            "name": asset_name,
            "updated_at": asset["updated_at"],
            "extracted_at": datetime.now().isoformat(),
        }
        if subset:
            meta["subset"] = sorted(subset)
    with zipfile.ZipFile(download_pth, "r") as zipf:
        files = set(zipf.namelist())
        code = False
        if "code.json" in files:
            # don't extract this file
            files.remove("code.json")
            code_bytes = zipf.read("code.json")
            code = json.loads(code_bytes.decode())
            if meta_path:
                import hashlib

                code_md5 = hashlib.md5(code_bytes).hexdigest()
                meta["code_json_md5"] = code_md5
        if subset:
            nosub = False
            subset_keys = files
            if code:
                subset_keys |= set(code.keys())
            not_found = subset.difference(subset_keys)
            if not_found:
                raise ValueError(
                    f"subset item{'s' if len(not_found) != 1 else ''} "
                    f"not found: {', '.join(sorted(not_found))}\n"
                    f"available items are:\n{columns_str(sorted(subset_keys))}"
                )
        else:
            nosub = True
            subset = set()

        if code:

            def add_item(key, fname, do_chmod):
                if fname in files:
                    extract.add(fname)
                    items.append(f"{fname} ({code[key]['version']})")
                    if do_chmod:
                        chmod.add(fname)
                else:
                    print(f"file {fname} does not exist")
                return

            for key in sorted(code):
                key_in_sub = key in subset
                if code[key].get("shared_object"):
                    fname = f"{key}{lib_suffix}"
                    if nosub or (subset and (key_in_sub or fname in subset)):
                        add_item(key, fname, do_chmod=False)
                else:
                    fname = f"{key}{exe_suffix}"
                    if nosub or (subset and (key_in_sub or fname in subset)):
                        add_item(key, fname, do_chmod=True)
                    # check if double version exists
                    fname = f"{key}dbl{exe_suffix}"
                    if (
                        code[key].get("double_switch", True)
                        and fname in files
                        and (
                            nosub
                            or (subset and (key_in_sub or fname in subset))
                        )
                    ):
                        add_item(key, fname, do_chmod=True)
        else:  # release 1.0 did not have code.json
            for fname in sorted(files):
                if nosub or (subset and fname in subset):
                    extract.add(fname)
                    items.append(fname)
                    if not fname.endswith(lib_suffix):
                        chmod.add(fname)
        if not quiet:
            print(
                f"extracting {len(extract)} "
                f"file{'s' if len(extract) != 1 else ''} to '{bindir}'"
            )
        zipf.extractall(bindir, members=extract)

    # If this is a TemporaryDirectory, then delete the directory and files
    del tmpdir

    if ostag in ["linux", "mac"]:
        # similar to "chmod +x fname" for each executable
        for fname in chmod:
            pth = bindir / fname
            pth.chmod(pth.stat().st_mode | 0o111)

    # Show listing
    if not quiet:
        print(columns_str(items))

        if not subset:
            unexpected = extract.difference(files)
            if unexpected:
                print(f"unexpected remaining {len(unexpected)} files:")
                print(columns_str(sorted(unexpected)))

    # Save metadata, only for flopy
    if meta_path:
        if "pytest" in str(bindir) or "pytest" in sys.modules:
            # Don't write metadata if this is part of pytest
            print("skipping writing flopy metadata for pytest")
            return
        meta_list.append(meta)
        if not flopy_appdata.exists():
            flopy_appdata.mkdir(parents=True, exist_ok=True)
        try:
            meta_path.write_text(json.dumps(meta_list, indent=4) + "\n")
        except OSError as err:
            print(f"cannot write flopy metadata file: '{meta_path}': {err!r}")
        if not quiet:
            if meta_path_exists:
                print(f"updated flopy metadata file: '{meta_path}'")
            else:
                print(f"wrote new flopy metadata file: '{meta_path}'")


def cli_main():
    """Command-line interface."""
    import argparse

    # Show meaningful examples at bottom of help
    prog = Path(sys.argv[0]).stem
    if sys.platform.startswith("win"):
        drv = Path("c:/")
    else:
        drv = Path("/")
    example_bindir = drv / "path" / "to" / "bin"
    examples = f"""\
Examples:

  Install executables into an existing '{example_bindir}' directory:
    $ {prog} {example_bindir}

  Install a development snapshot of MODFLOW 6 by choosing a repo:
    $ {prog} --repo modflow6-nightly-build {example_bindir}
    """
    if within_flopy:
        examples += f"""\

  FloPy users can install executables using a special option:
    $ {prog} :flopy
    """

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )

    bindir_help = (
        "Directory to extract executables. Use ':' to interactively select an "
        "option of paths. Other auto-select options are only available if the "
        "current user can write files. "
    )
    if within_flopy:
        bindir_help += (
            "Option ':prev' is the previously used 'bindir' path selection. "
            "Option ':flopy' will create and install programs for FloPy. "
        )
    if sys.platform.startswith("win"):
        bindir_help += (
            "Option ':python' is Python's Scripts directory. "
            "Option ':windowsapps' is "
            "'%%LOCALAPPDATA%%\\Microsoft\\WindowsApps'."
        )
    else:
        bindir_help += (
            "Option ':python' is Python's bin directory. "
            "Option ':local' is '$HOME/.local/bin'. "
            "Option ':system' is '/usr/local/bin'."
        )
    parser.add_argument("bindir", help=bindir_help)
    parser.add_argument(
        "--repo",
        choices=available_repos,
        default="executables",
        help="Name of GitHub repository; default is 'executables'.",
    )
    parser.add_argument(
        "--release-id",
        default="latest",
        help="GitHub release ID; default is 'latest'.",
    )
    parser.add_argument(
        "--ostag",
        choices=available_ostags,
        help="Operating system tag; default is to automatically choose.",
    )
    parser.add_argument(
        "--subset",
        help="Subset of executables to extract, specified as a "
        "comma-separated string, e.g. 'mfnwt,mp6'.",
    )
    parser.add_argument(
        "--downloads-dir",
        help="Manually specify directory to download archives.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download archive. Default behavior will use archive if "
        "previously downloaded in downloads-dir.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Show fewer messages."
    )
    args = vars(parser.parse_args())
    try:
        run_main(**args, _is_cli=True)
    except (EOFError, KeyboardInterrupt):
        sys.exit(f" cancelling '{sys.argv[0]}'")


if __name__ == "__main__":
    """Run command-line interface, if run as a script."""
    cli_main()
