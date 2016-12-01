
def test_import():
    try:
        import flopy
    except:
        fail = True
        assert fail is False, 'could not import flopy'
    return


if __name__ == '__main__':
    test_import()
