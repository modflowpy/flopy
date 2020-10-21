import os
import sys
import shutil

# path to notebooks
src_pths = (
    os.path.join("..", "examples", "Notebooks"),
    os.path.join("..", "examples", "groundwater_paper", "Notebooks"),
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

# create temporary directory
dst_pth = os.path.join("..", "examples", ".nb")
if os.path.isdir(dst_pth):
    shutil.rmtree(dst_pth)
os.makedirs(dst_pth)

failed_runs = []

# run the notebooks
for src in nb_files:
    file_name = os.path.basename(src)
    dst = os.path.join(dst_pth, file_name)
    arg = (
        "jupytext",
        "--to ipynb",
        "--from ipynb",
        "--execute",
        "-o",
        dst,
        src,
    )
    print(" ".join(arg))
    return_code = os.system(" ".join(arg))
    if return_code == 0:
        print("copy {} -> {}". format(dst, src))
        shutil.copyfile(dst, src)
    else:
        failed_runs.append(src)

# write out failed runs
for idx, src in enumerate(failed_runs):
    print("{:2d}...{} FAILED".format(idx + 1, src))


# clean up temporary files
print("cleaning up...'{}'".format(dst_pth))
shutil.rmtree(dst_pth)
