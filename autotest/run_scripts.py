# Remove the temp directory and then create a fresh one
import pytest
import os
import sys
import shutil
from subprocess import Popen, PIPE

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


# make working directories
out_dir = os.path.join("temp", "scripts")
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def copy_script(src):
    # copy files
    filedir = os.path.dirname(src)
    filename = os.path.basename(src)

    # set dstpth and clean if it exists
    dstpth = os.path.abspath(
        os.path.join(out_dir, filename.replace(".py", ""))
    )
    if not os.path.isdir(dstpth):
        os.makedirs(dstpth, exist_ok=True)

    # set destination path
    dst = os.path.join(out_dir, dstpth, filename)

    # copy script
    print(f"copying {filename} from {filedir} to {dstpth}")
    shutil.copyfile(src, dst)

    return dst


def run_script(script):
    ws = os.path.dirname(script)
    file_name = os.path.basename(script)
    args = ("python", file_name)
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
    # copy script
    dst = copy_script(script)

    # run script
    run_script(dst)


if __name__ == "__main__":
    for script in scripts:
        dst = copy_script(script)
        run_script(dst)
