# Build the executables that are used in the flopy autotests

try:
    import pymake
except:
    print('pymake is not installed...will not build executables')
    pymake = None


def build_all_apps():
    if pymake is not None:
        targets = pymake.usgs_prog_data().get_keys(current=True)
        targets.sort()
        targets.remove('vs2dt')
        pymake.build_apps(targets=targets)

    return


if __name__ == '__main__':
    build_all_apps()
