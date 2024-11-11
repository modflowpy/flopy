from pathlib import Path

project_root_path = Path(__file__).parent.parent


def get_section(f):
    lines = Path(f).open().readlines()
    line = next(iter([l for l in lines if "#     section:" in l]), None)
    section = "misc" if not line else line.rpartition(":")[2].strip()
    return section


def create_toc_section(f, title, stems, upper_case=False):
    if upper_case:
        title = title.upper()
    line = f"{title}\n" + len(title) * "-" + "\n\n"
    line += ".. toctree::\n   :maxdepth: 2\n\n"
    for stem in stems:
        line += f"   Notebooks/{stem}\n"
    line += "\n\n"
    f.write(line)


def create_gallery_section(f, name, title, stems):
    line = f"{title}\n" + len(title) * "-" + "\n\n"
    line += f".. nbgallery::\n    :name: {name}\n\n"
    for stem in stems:
        line += f"   Notebooks/{stem}\n"
    line += "\n\n"
    f.write(line)


def create_tutorials_rst():
    rst_path = project_root_path / ".docs" / "tutorials.rst"
    nbs_path = project_root_path / ".docs" / "Notebooks"
    filenames = sorted(
        [path.name for path in nbs_path.rglob("*.py") if "tutorial" in path.name]
    )

    print(f"Creating {rst_path}")
    with open(rst_path, "w") as rst_file:
        rst_file.write("Tutorials\n=========\n\n")
        rst_file.write(
            "The following tutorials demonstrate basic FloPy features and usage with MODFLOW 2005, MODFLOW 6, and related programs.\n\n"
        )

        sections = {
            "flopy": {"title": "FloPy", "files": []},
            "mf6": {"title": "MODFLOW 6", "files": []},
            "mf2005": {"title": "MODFLOW-2005", "files": []},
            "lgr": {"title": "MODFLOW-LGR", "files": []},
            "nwt": {"title": "MODFLOW-NWT", "files": []},
            "mt3dms": {"title": "MT3DMS", "files": []},
            "pest": {"title": "PEST", "files": []},
            "misc": {"title": "Miscellaneous", "files": []},
        }

        for file_name in filenames:
            section_name = get_section(nbs_path / file_name)
            sections[section_name]["files"].append(file_name)

        for section_name, section in sections.items():
            file_names = section["files"]
            if any(file_names):
                create_toc_section(
                    f=rst_file,
                    title=section["title"],
                    stems=[fn.rpartition(".")[0] for fn in file_names],
                )


def create_examples_rst():
    rst_path = project_root_path / ".docs" / "examples.rst"
    nbs_path = project_root_path / ".docs" / "Notebooks"
    filenames = sorted(
        [path.name for path in nbs_path.rglob("*.py") if "example" in path.name]
    )

    print(f"Creating {rst_path}")
    with open(rst_path, "w") as rst_file:
        rst_file.write("Examples gallery\n================\n\n")
        rst_file.write(
            "The following examples illustrate the functionality of Flopy. After the `tutorials <https://flopy.readthedocs.io/en/latest/tutorials.html>`_, the examples are the best resource for learning the underlying capabilities of FloPy.\n\n"
        )

        sections = {
            "dis": {"title": "Preprocessing and Discretization", "files": []},
            "viz": {"title": "Postprocessing and Visualization", "files": []},
            "export": {"title": "Exporting data", "files": []},
            "flopy": {"title": "Other FloPy features", "files": []},
            "mf6": {"title": "MODFLOW 6 examples", "files": []},
            "mfusg": {"title": "MODFLOW USG examples", "files": []},
            "mf2005": {
                "title": "MODFLOW-2005/MODFLOW-NWT examples",
                "files": [],
            },
            "modpath": {"title": "MODPATH examples", "files": []},
            "mt3d": {"title": "MT3D and SEAWAT examples", "files": []},
            "2016gw-paper": {
                "title": "Examples from Bakker and others (2016)",
                "description": "Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, https://doi.org/10.1111/gwat.12413.",
                "files": [],
            },
            "2023gw-paper": {
                "title": "Examples from Hughes and others (2023)",
                "description": "Hughes, Joseph D., Langevin, Christian D., Paulinski, Scott R., Larsen, Joshua D., and Brakenhoff, David, 2023, FloPy Workflows for Creating Structured and Unstructured MODFLOW Models: Groundwater, https://doi.org/10.1111/gwat.13327.",
                "files": [],
            },
            "misc": {"title": "Miscellaneous examples", "files": []},
        }

        for file_name in filenames:
            section_name = get_section(nbs_path / file_name)
            sections[section_name]["files"].append(file_name)

        for section_name, section in sections.items():
            file_names = section["files"]
            if any(file_names):
                create_gallery_section(
                    f=rst_file,
                    name=section_name,
                    title=section["title"],
                    stems=[fn.rpartition(".")[0] for fn in file_names],
                )


if __name__ == "__main__":
    create_tutorials_rst()
    create_examples_rst()
