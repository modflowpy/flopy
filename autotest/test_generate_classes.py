from autotest.conftest import ci_only, excludes_branch

from flopy.mf6.utils import generate_classes


@ci_only
@excludes_branch("master")
def test_generate_classes_from_dfn(tmpdir):
    generate_classes(branch="develop", backup=False)
    # TODO: what to check?
