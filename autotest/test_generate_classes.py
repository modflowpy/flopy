import pytest
from autotest.conftest import excludes_branch

from flopy.mf6.utils import generate_classes


@pytest.mark.mf6
@pytest.mark.skip(
    reason="TODO: use external copy of the repo, otherwise files are rewritten"
)
@excludes_branch("master")
def test_generate_classes_from_dfn():
    # maybe compute hashes of files before/after
    # generation to make sure they don't change?

    generate_classes(branch="develop", backup=False)
