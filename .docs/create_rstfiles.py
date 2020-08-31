import os


def create_section(f, title, filenames):
    title = "{} Tutorials".format(title.upper())
    line = "{}\n".format(title) + len(title) * "-" + "\n\n"
    line += "Contents:\n\n.. toctree::\n   :maxdepth: 2\n\n"
    for filename in filenames:
        line += "   _notebooks/{}\n".format(filename)
    line += "\n\n"
    f.write(line)


def create_tutorial_rst():
    print("creating 'tutorials.rst'")
    pth = os.path.join("..", "examples", "Tutorials")
    tutorial_dict = {}
    for dirpath, _, filenames in os.walk(pth):
        key = os.path.basename(os.path.normpath(dirpath))
        files = [filename.replace(".py", "") for filename in sorted(filenames)
                 if filename.endswith(".py")]
        if len(files) > 0:
            tutorial_dict[key] = files

    # open the file and write the header
    f = open("tutorials.rst", "w")
    f.write("Tutorials\n=========\n\n")

    keys = list(tutorial_dict.keys())
    key = "modflow6"
    if key in keys:
        create_section(f, "MODFLOW 6", tutorial_dict[key])
        keys.remove(key)

    key = "modflow"
    if key in keys:
        create_section(f, "MODFLOW", tutorial_dict[key])
        keys.remove(key)

    # create the remaining tutorial sections
    for key in keys:
        create_section(f, key, tutorial_dict[key])

    # close the file
    f.close()


if __name__ == "__main__":
    create_tutorial_rst()
