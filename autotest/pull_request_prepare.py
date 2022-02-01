import os

try:
    import isort

    print(f"isort version: {isort.__version__}")
except ModuleNotFoundError:
    print("isort not installed\n\tInstall using pip install isort")

try:
    import black

    print(f"black version: {black.__version__}")
except ModuleNotFoundError:
    print("black not installed\n\tInstall using pip install black")

print("running isort...")
os.system("isort -v ../flopy")

print("running black...")
os.system("black -v ../flopy")
