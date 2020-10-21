import os
import sys
import shutil

# path to notebooks
src_pths = (
    os.path.join("..", "examples", "Notebooks"),
    os.path.join("..", "examples", "groundwater_paper", "Notebooks"),
    os.path.join("..", "examples", "FAQ"),
)

# parse command line arguments for notebook to create
nb_files = None
for idx, arg in enumerate(sys.argv):
    if arg in ("-f", "--file"):
        file_name = sys.argv[idx + 1]
        if not file_name.endswith(".ipynb"):
            file_name += ".ipynb"
        for src_pth in src_pths:
            src = os.path.join(src_pth, file_name)
            if os.path.isfile(src):
                nb_files = [src]
                break

# get list of notebooks
if nb_files is None:
    nb_files = []
    for src_pth in src_pths:
        nb_files += [
            os.path.join(src_pth, file_name)
            for file_name in sorted(os.listdir(src_pth))
            if file_name.endswith(".ipynb")
        ]

failed_runs = []

# run the notebooks
for src in nb_files:
    arg = (
        "jupytext",
        "--from ipynb",
        "--execute",
        src,
    )
    print(" ".join(arg))
    return_code = os.system(" ".join(arg))
    if return_code != 0:
        failed_runs.append(src)

# write out failed runs
for idx, src in enumerate(failed_runs):
    print("{:2d}...{} FAILED".format(idx + 1, src))
