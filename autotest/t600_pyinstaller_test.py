import os
import pathlib

import flopy

root_folder = pathlib.Path(__file__).parent.parent
flopy_folder = pathlib.Path(flopy.__file__).parent
dfn_path = flopy_folder / "mf6" / "data" / "dfn"
rename_path = flopy_folder / "mf6" / "data" / "no-dfn"


def test_flopy_runs_without_dfn_folder():
    """Test to ensure that flopy can load a modflow 6 simulation without dfn files being present."""
    exists = dfn_path.exists()
    if exists:
        if rename_path.exists():
            os.rmdir(rename_path)
        os.rename(dfn_path, rename_path)
    try:
        # run built executable
        sim_path = root_folder / "examples" / "data" / "mf6" / "test006_gwf3"

        flopy.mf6.MFSimulation.load(sim_ws=str(sim_path))
    finally:
        if exists and rename_path.exists():
            os.rename(rename_path, dfn_path)
