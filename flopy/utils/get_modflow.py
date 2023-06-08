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
import warnings
import zipfile
from importlib.util import find_spec
from pathlib import Path

__all__ = ["run_main"]
__license__ = "CC0"

from typing import Dict, List, Tuple

owner = "MODFLOW-USGS"
# key is the repo name, value is the renamed file prefix for the download
renamed_prefix = {
    "modflow6": "modflow6",
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

# local flopy install location (selected with :flopy)
flopy_appdata_path = (
    Path(os.path.expandvars(r"%LOCALAPPDATA%\flopy"))
    if sys.platform.startswith("win")
    else Path.home() / ".local" / "share" / "flopy"
)


def get_ostag() -> str:
    """Determine operating system tag from sys.platform."""
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win"):
        return "win" + ("64" if sys.maxsize > 2**32 else "32")
    elif sys.platform.startswith("darwin"):
        return "mac"
    raise ValueError(f"platform {sys.platform!r} not supported")


def get_suffixes(ostag) -> Tuple[str, str]:
    if ostag in ["win32", "win64"]:
        return ".exe", ".dll"
    elif ostag == "linux":
        return "", ".so"
    elif ostag == "mac":
        return "", ".dylib"
    else:
        raise KeyError(
            f"unrecognized ostag {ostag!r}; choose one of {available_ostags}"
        )


def get_request(url, params={}):
    """Get urllib.request.Request, with parameters and headers.

    This bears GITHUB_TOKEN if it is set as an environment variable.
    """
    if isinstance(params, dict):
        if len(params) > 0:
            url += "?" + urllib.parse.urlencode(params)
    else:
        raise TypeError("data must be a dict")
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return urllib.request.Request(url, headers=headers)


def get_releases(repo, quiet=False, per_page=None) -> List[str]:
    """Get list of available releases."""
    req_url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    params = {}
    if per_page is not None:
        if per_page < 1 or per_page > 100:
            raise ValueError("per_page must be between 1 and 100")
        params["per_page"] = per_page

    request = get_request(req_url, params=params)
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
    if not quiet:
        print(f"found {len(releases)} releases for {owner}/{repo}")

    avail_releases = ["latest"]
    avail_releases.extend(release["tag_name"] for release in releases)
    return avail_releases


def get_release(repo, tag="latest", quiet=False) -> dict:
    """Get info about a particular release."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    req_url = (
        f"{api_url}/releases/latest"
        if tag == "latest"
        else f"{api_url}/releases/tags/{tag}"
    )
    request = get_request(req_url)
    releases = None
    num_tries = 0

    while True:
        num_tries += 1
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                result = resp.read()
                remaining = int(resp.headers["x-ratelimit-remaining"])
                if remaining <= 10:
                    warnings.warn(
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
                if releases is None:
                    releases = get_releases(repo, quiet)
                if tag not in releases:
                    raise ValueError(
                        f"Release {tag} not found (choose from {', '.join(releases)})"
                    )
            elif err.code == 503 and num_tries < max_http_tries:
                # GitHub sometimes returns this error for valid URLs, so retry
                warnings.warn(f"URL request {num_tries} did not work ({err})")
                continue
            raise RuntimeError(f"cannot retrieve data from {req_url}") from err

    release = json.loads(result.decode())
    tag_name = release["tag_name"]
    if not quiet:
        print(f"fetched release {tag_name!r} info from {owner}/{repo}")

    return release


def columns_str(items, line_chars=79) -> str:
    """Return str of columns of items, similar to 'ls' command."""
    item_chars = max(len(item) for item in items)
    num_cols = line_chars // item_chars
    if num_cols == 0:
        num_cols = 1
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


def get_bindir_options(previous=None) -> Dict[str, Tuple[Path, str]]:
    """Generate install location options based on platform and filesystem access."""
    options = {}  # key is an option name, value is (optpath, optinfo)
    if previous is not None and os.access(previous, os.W_OK):
        # Make previous bindir as the first option
        options[":prev"] = (previous, "previously selected bindir")
    if within_flopy:  # don't check is_dir() or access yet
        options[":flopy"] = (flopy_appdata_path / "bin", "used by FloPy")
    # Python bin (same for standard or conda varieties)
    py_bin = Path(sys.prefix) / (
        "Scripts" if get_ostag().startswith("win") else "bin"
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

    # any other possible OS-specific hard-coded locations?
    if not options:
        raise RuntimeError("could not find any installable folders")

    return options


def select_bindir(bindir, previous=None, quiet=False, is_cli=False) -> Path:
    """Resolve an install location if provided, or prompt interactive user to select one."""
    options = get_bindir_options(previous)

    if len(bindir) > 1:  # auto-select mode
        # match one option that starts with input, e.g. :Py -> :python
        sel = list(opt for opt in options if opt.startswith(bindir.lower()))
        if len(sel) != 1:
            opt_avail = ", ".join(
                f"'{opt}' for '{optpath}'"
                for opt, (optpath, _) in options.items()
            )
            raise ValueError(
                f"invalid option '{bindir}', choose from: {opt_avail}"
            )
        if not quiet:
            print(f"auto-selecting option {sel[0]!r} for '{bindir}'")
        return Path(options[sel[0]][0]).resolve()
    else:
        if not is_cli:
            opt_avail = ", ".join(
                f"'{opt}' for '{optpath}'"
                for opt, (optpath, _) in options.items()
            )
            raise ValueError(f"specify the option, choose from: {opt_avail}")

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
                return Path(options[opt][0]).resolve()
            except (KeyError, ValueError):
                if num_tries < 2:
                    print("invalid option, try choosing option again")
                else:
                    raise RuntimeError(
                        "invalid option, too many attempts"
                    ) from None


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
        Name of GitHub repository. Choose one of "executables" (default),
        "modflow6", or "modflow6-nightly-build".
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
        if not flopy_appdata_path.exists():
            flopy_appdata_path.mkdir(parents=True, exist_ok=True)
        flopy_bin = flopy_appdata_path / "bin"
        meta_path = flopy_appdata_path / "get_modflow.json"
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

    exe_suffix, lib_suffix = get_suffixes(ostag)

    # select bindir if path not provided
    if bindir.startswith(":"):
        bindir = select_bindir(
            bindir, previous=prev_bindir, quiet=quiet, is_cli=_is_cli
        )
    elif not isinstance(bindir, (str, Path)):
        raise ValueError(f"Invalid bindir option (expected string or Path)")
    bindir = Path(bindir).resolve()

    # make sure bindir exists
    if bindir == flopy_bin:
        if not within_flopy:
            raise ValueError("option ':flopy' is only for flopy")
        elif not flopy_bin.exists():
            # special case option that can create non-existing directory
            flopy_bin.mkdir(parents=True, exist_ok=True)
    if not bindir.is_dir():
        raise OSError(f"extraction directory '{bindir}' does not exist")
    elif not os.access(bindir, os.W_OK):
        raise OSError(f"extraction directory '{bindir}' is not writable")

    # make sure repo option is valid
    if repo not in available_repos:
        raise KeyError(
            f"repo {repo!r} not supported; choose one of {available_repos}"
        )

    # get the selected release
    release = get_release(repo, release_id, quiet)
    assets = release.get("assets", [])

    # Windows 64-bit asset in modflow6 repo release has no OS tag
    if repo == "modflow6" and ostag == "win64":
        asset = list(sorted(assets, key=lambda a: len(a["name"])))[0]
    else:
        for asset in assets:
            if ostag in asset["name"]:
                break
        else:
            raise ValueError(
                f"could not find ostag {ostag!r} from release {release['tag_name']!r}; "
                f"see available assets here:\n{release['html_url']}"
            )
    asset_name = asset["name"]
    download_url = asset["browser_download_url"]
    if repo == "modflow6":
        asset_pth = Path(asset_name)
        asset_stem = asset_pth.stem
        asset_suffix = asset_pth.suffix
        dst_fname = "-".join([repo, release["tag_name"], ostag]) + asset_suffix
    else:
        # change local download name so it is more unique
        dst_fname = "-".join(
            [renamed_prefix[repo], release["tag_name"], asset_name]
        )
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
            print(f"downloading '{download_url}' to '{download_pth}'")
        urllib.request.urlretrieve(download_url, download_pth)

    if subset:
        if isinstance(subset, str):
            subset = set(subset.replace(",", " ").split())
        else:
            subset = set(subset)

    # Open archive and extract files
    extract = set()
    chmod = set()
    items = []
    full_path = {}
    if meta_path:
        from datetime import datetime

        meta = {
            "bindir": str(bindir),
            "owner": owner,
            "repo": repo,
            "release_id": release["tag_name"],
            "name": asset_name,
            "updated_at": asset["updated_at"],
            "extracted_at": datetime.now().isoformat(),
        }
        if subset:
            meta["subset"] = sorted(subset)
    with zipfile.ZipFile(download_pth, "r") as zipf:
        if repo == "modflow6":
            # modflow6 release contains the whole repo with an internal bindir
            for pth in zipf.namelist():
                p = Path(pth)
                if p.parent.name == "bin":
                    full_path[p.name] = pth
            files = set(full_path.keys())
        else:
            # assume all files to be extracted
            files = set(zipf.namelist())

        code = False
        if "code.json" in files and repo == "executables":
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
                if code[key].get("shared_object"):
                    fname = f"{key}{lib_suffix}"
                    if nosub or (
                        subset and (key in subset or fname in subset)
                    ):
                        add_item(key, fname, do_chmod=False)
                else:
                    fname = f"{key}{exe_suffix}"
                    if nosub or (
                        subset and (key in subset or fname in subset)
                    ):
                        add_item(key, fname, do_chmod=True)
                    # check if double version exists
                    fname = f"{key}dbl{exe_suffix}"
                    if (
                        code[key].get("double_switch", True)
                        and fname in files
                        and (
                            nosub
                            or (subset and (key in subset or fname in subset))
                        )
                    ):
                        add_item(key, fname, do_chmod=True)

        else:
            # releases without code.json
            for fname in sorted(files):
                if nosub or (subset and fname in subset):
                    if full_path:
                        extract.add(full_path[fname])
                    else:
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

    if full_path:
        # move files that used a full path to bindir
        rmdirs = set()
        for fpath in extract:
            fpath = Path(fpath)
            bindir_path = bindir / fpath
            bindir_path.replace(bindir / fpath.name)
            rmdirs.add(fpath.parent)
        # clean up directories, starting with the longest
        for rmdir in reversed(sorted(rmdirs)):
            bindir_path = bindir / rmdir
            bindir_path.rmdir()
            for subdir in rmdir.parents:
                bindir_path = bindir / subdir
                if bindir_path == bindir:
                    break
                bindir_path.rmdir()

    if ostag in ["linux", "mac"]:
        # similar to "chmod +x fname" for each executable
        for fname in chmod:
            pth = bindir / fname
            pth.chmod(pth.stat().st_mode | 0o111)

    # Show listing
    if not quiet:
        if any(items):
            print(columns_str(items))

        if not subset:
            if full_path:
                extract = {Path(fpth).name for fpth in extract}
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
        if not flopy_appdata_path.exists():
            flopy_appdata_path.mkdir(parents=True, exist_ok=True)
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
            "Option ':home' is '$HOME/.local/bin'. "
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
