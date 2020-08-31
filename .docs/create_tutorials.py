import os
import shutil


def create_notebooks():
    wpth = ".working"
    if os.path.isdir(wpth):
        shutil.rmtree(wpth)
    os.makedirs(wpth)

    # copy the python files
    pth = os.path.join("..", "examples", "Tutorials")

    # get a list of python files
    py_files = []
    for dirpath, _, filenames in os.walk(pth):
        py_files += [os.path.join(dirpath, filename) for filename in
                     sorted(filenames) if filename.endswith(".py")]
    # copy the python files
    for src in py_files:
        dst = os.path.join(wpth, os.path.basename(src))
        print("{} -> {}".format(src, dst))
        shutil.copyfile(src, dst)

    # create and run notebooks
    py_pth = os.path.join(wpth, "*.py")
    cmd = (
        "jupytext",
        "--to ipynb",
        "--execute",
        py_pth,
    )
    print(" ".join(cmd))
    os.system(" ".join(cmd))

    npth = "_notebooks"
    # copy notebooks
    if os.path.isdir(npth):
        shutil.rmtree(npth)
    os.makedirs(npth)

    for filepath in py_files:
        src = os.path.join(wpth,
                           os.path.basename(filepath).replace(".py", ".ipynb"))
        dst = os.path.join(npth,
                           os.path.basename(filepath).replace(".py", ".ipynb"))
        shutil.copyfile(src, dst)
    shutil.rmtree(".working")


if __name__ == "__main__":
    create_notebooks()
