import os
import shutil


def create_notebooks(pth):
    pth = str(pth)

    # create a working directory
    wpth = ".working"
    if os.path.isdir(wpth):
        shutil.rmtree(wpth)
    os.makedirs(wpth)

    # find python files paired with notebooks
    py_files = []
    for dirpath, _, filenames in os.walk(pth):
        py_files += [
            os.path.join(dirpath, filename)
            for filename in sorted(filenames)
            if filename.endswith(".py")
        ]

    # sort the python files
    py_files = sorted(py_files)

    # copy the python files
    for src in py_files:
        dst = os.path.join(wpth, os.path.basename(src))
        print(f"{src} -> {dst}")
        shutil.copyfile(src, dst)
    
    # copy common utils
    src = os.path.join("..", "examples", "common")
    dst = os.path.join("common")
    print(f"{src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

    # copy example data
    src = os.path.join("..", "examples", "data")
    dst = os.path.join("data")
    print(f"{src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

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
        src = os.path.join(
            wpth, os.path.basename(filepath).replace(".py", ".ipynb")
        )
        dst = os.path.join(
            npth, os.path.basename(filepath).replace(".py", ".ipynb")
        )
        shutil.copyfile(src, dst)
    shutil.rmtree(".working")


if __name__ == "__main__":
    create_notebooks(os.path.join("..", "examples", "Tutorials"))
    create_notebooks(os.path.join("..", "examples", "Notebooks"))
    create_notebooks(os.path.join("..", "examples", "groundwater_paper"))
    create_notebooks(os.path.join("..", "examples", "FAQ"))
