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
    with urllib.request.urlopen(get_request(f"{api_url}/releases")) as resp:
        result = resp.read()
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
        Writable path to extract executables.
    repo : str, default "executables"
        Name of GitHub repository. Choose one of "executables" (default) or
        "modflow6-nightly-build".
    release_id : str, default "latest"
        GitHub release ID.
    ostag : str, optional
        Operating system tag; default is to automatically choose.
    subset : str, optional
        Optional subset of executables to extract, e.g. "mfnwt,mp6"
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

    if _is_cli and bindir == "?":
        options = []
        # check if conda
        conda_bin = (
            Path(sys.prefix)
            / "conda-meta"
            / ".."
            / ("Scripts" if ostag.startswith("win") else "bin")
        )
        if conda_bin.exists() and os.access(conda_bin, os.W_OK):
            options.append(conda_bin.resolve())
        home_local_bin = Path.home() / ".local" / "bin"
        if home_local_bin.is_dir() and os.access(home_local_bin, os.W_OK):
            options.append(home_local_bin)
        local_bin = Path("/usr") / "local" / "bin"
        if local_bin.is_dir() and os.access(local_bin, os.W_OK):
            options.append(local_bin)
        # Windows user
        windowsapps_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps")
        )
        if windowsapps_dir.is_dir() and os.access(windowsapps_dir, os.W_OK):
            options.append(windowsapps_dir)
        # any other possible locations?
        if not options:
            raise RuntimeError("could not find any installable folders")
        print("select a directory to extract executables:")
        options_d = dict(enumerate(options, 1))
        for iopt, opt in options_d.items():
            print(f"{iopt:2d}: {opt}")
        num_tries = 0
        while True:
            num_tries += 1
            res = input("> ")
            try:
                bindir = options_d[int(res)]
                break
            except (KeyError, ValueError):
                if num_tries < 3:
                    print("invalid option, try choosing option again")
                else:
                    raise RuntimeError("invalid option, too many attempts")

    bindir = Path(bindir).resolve()
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
    try:
        with urllib.request.urlopen(get_request(req_url)) as resp:
            result = resp.read()
            remaining = int(resp.headers["x-ratelimit-remaining"])
            if remaining <= 10:
                print(
                    f"Only {remaining} GitHub API requests remaining "
                    "before rate-limiting"
                )
    except urllib.error.HTTPError as err:
        if err.code == 401 and os.environ.get("GITHUB_TOKEN"):
            raise ValueError(
                "environment variable GITHUB_TOKEN is invalid"
            ) from err
        if err.code == 403 and "rate limit exceeded" in err.reason:
            raise ValueError(
                "use environment variable GITHUB_TOKEN to bypass rate limit"
            ) from err
        elif err.code == 404:
            avail_releases = get_avail_releases(api_url)
            raise ValueError(
                f"Release {release_id!r} not found -- "
                f"choose from {avail_releases}"
            ) from err
        else:
            raise err
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
    download_url = asset["browser_download_url"]
    # change local download name so it is more unique
    dst_fname = "-".join([renamed_prefix[repo], tag_name, asset["name"]])
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

    # Open archive and extract files
    extract = set()
    chmod = set()
    items = list()
    with zipfile.ZipFile(download_pth, "r") as zipf:
        files = set(zipf.namelist())
        # print(f"{len(files)=}: {files}")
        code = False
        if "code.json" in files:
            # don't extract this file
            code = json.loads(zipf.read("code.json").decode())
            files.remove("code.json")
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


def cli_main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "bindir",
        help="directory to extract executables; use '?' to help choose",
    )
    parser.add_argument(
        "--repo",
        choices=available_repos,
        default="executables",
        help="name of GitHub repository; default is 'executables'",
    )
    parser.add_argument(
        "--release-id",
        default="latest",
        help="GitHub release ID (default: latest)",
    )
    parser.add_argument(
        "--ostag",
        choices=available_ostags,
        help="operating system tag; default is to automatically choose",
    )
    parser.add_argument("--subset", help="subset of executables")
    parser.add_argument(
        "--downloads-dir",
        help="manually specify directory to download archives",
    )
    parser.add_argument(
        "--force", action="store_true", help="force re-download"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="show fewer messages"
    )
    args = vars(parser.parse_args())
    if args["subset"]:
        args["subset"] = set(args["subset"].replace(",", " ").split())
    # print(args)
    try:
        run_main(**args, _is_cli=True)
    except KeyboardInterrupt:
        sys.exit(f" cancelling {sys.argv[0]}")
    except (KeyError, OSError, RuntimeError, ValueError) as err:
        sys.exit(err)


if __name__ == "__main__":
    """Run command-line interface, if run as a script."""
    cli_main()
