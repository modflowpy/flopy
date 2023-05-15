import itertools
import os
from pathlib import Path


project_root_path = Path(__file__).parent.parent


def create_section(f, title, stems, upper_case=False):
    if upper_case:
        title = title.upper()
    line = f"{title}\n" + len(title) * "-" + "\n\n"
    line += "Contents:\n\n.. toctree::\n   :maxdepth: 2\n\n"
    for stem in stems:
        line += f"   Notebooks/{stem}\n"
    line += "\n\n"
    f.write(line)


def create_tutorial_rst():
    print("creating 'tutorials.rst'")
    nbs_path = project_root_path / ".docs" / "Notebooks"
    filestems = [
        p.stem for p in nbs_path.glob("*.py")
        if "tutorial" in p.name
    ]
    filestems.sort()

    # write the file
    with open(project_root_path / ".docs" / "tutorials.rst", "w") as f:
        f.write("Tutorials\n=========\n\n")
        f.write("The following tutorials demonstrate basic FloPy usage with MODFLOW 2005, MODFLOW 6, and SEAWAT.\n\n")

        def get_group(stem):
            if "mf6" in stem:
                return "MODFLOW6"
            elif "mf" in stem:
                return "MODFLOW"
            elif "seawat" in stem:
                return "SEAWAT"
            else:
                return "Miscellaneous"

        groups = itertools.groupby(filestems, key=get_group)
        for group_name, group in groups:
            create_section(f, group_name, group, upper_case=True)


if __name__ == "__main__":
    create_tutorial_rst()
