import os
import shutil

wpth = ".working"
if os.path.isdir(wpth):
    shutil.rmtree(wpth)
os.makedirs(wpth)
# copy the python files
pth = os.path.join("..", "examples", "Tutorials")
py_files = [file_name for file_name in os.listdir(pth)
            if file_name.endswith(".py")]
for file_name in py_files:
    src = os.path.join(pth, file_name)
    dst = os.path.join(wpth, file_name)
    shutil.copyfile(src, dst)
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
for file_name in py_files:
    src = os.path.join(wpth, file_name.replace(".py", ".ipynb"))
    dst = os.path.join(npth, file_name.replace(".py", ".ipynb"))
    shutil.copyfile(src, dst)
shutil.rmtree(".working")

