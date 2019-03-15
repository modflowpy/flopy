# Build the executables that are used in the flopy autotests

try:
    import pymake
except:
    print('pymake is not installed...will not build executables')
    pymake = None


def get_targets():
    targets = pymake.usgs_prog_data().get_keys(current=True)
    targets.sort()
    targets.remove('vs2dt')
    return targets


def build_target(target):
    if pymake is not None:
        pymake.build_apps(targets=target)

    return


def test_build_all_apps():
    targets = get_targets()
    for target in targets:
        yield build_target, target
    return


if __name__ == '__main__':
    targets = get_targets()
    for target in targets:
        build_target(target)
