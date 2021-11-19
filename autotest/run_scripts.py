# Remove the temp directory and then create a fresh one
import pytest
import os
import sys
import shutil
from subprocess import Popen, PIPE

from ci_framework import get_parent_path, FlopyTestSetup

parent_path = get_parent_path()
base_dir = os.path.join(parent_path, "temp")

# exclude files that take time on locanachine
exclude = ["flopy_swi2_ex2.py", "flopy_swi2_ex5.py"]
if "CI" in os.environ:
    exclude = []
else:
    for arg in sys.argv:
        if arg.lower() == "--all":
            exclude = []

sdir = os.path.join("..", "examples", "scripts")
tdir = os.path.join("..", "examples", "Tutorials")

scripts = []
for exdir in [sdir, tdir]:
    for dirName, subdirList, fileList in os.walk(exdir):
        for file_name in fileList:
            if file_name not in exclude:
                if file_name.endswith(".py"):
                    print(f"Found file: {file_name}")
                    scripts.append(
                        os.path.abspath(os.path.join(dirName, file_name))
                    )

scripts = sorted(scripts)

print("Scripts found:")
for script in scripts:
    print(f"  {script}")


def copy_script(dstDir, src):
    fileDir = os.path.dirname(src)
    fileName = os.path.basename(src)

    # set destination path with the file name
    dst = os.path.join(dstDir, fileName)

    # copy script
    print(f"copying {fileName} from {fileDir} to {dstDir}")
    shutil.copyfile(src, dst)

    return dst


def run_script(script):
    ws = os.path.dirname(script)
    filename = os.path.basename(script)
    args = ("python", filename)
    print(f"running...'{' '.join(args)}'")
    proc = Popen(args, stdout=PIPE, stderr=PIPE, cwd=ws)
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout.decode("utf-8"))
    if stderr:
        print(f"Errors:\n{stderr.decode('utf-8')}")

    return


@pytest.mark.parametrize(
    "script",
    scripts,
)
def test_scripts(script):
    script_name = os.path.basename(script).replace(".py", "")
    dstDir = os.path.join(f"{base_dir}", f"scripts_{script_name}")
    test_setup = FlopyTestSetup(verbose=True, test_dirs=dstDir)

    # copy script
    dst = copy_script(dstDir, script)

    # run script
    run_script(dst)


if __name__ == "__main__":
    for script in scripts:
        test_scripts(script)
