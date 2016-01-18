# Test modflow write adn run
import os
# import matplotlib.pyplot as plt

pth = os.path.join('..', 'examples', 'data', 'swr_test')
files = [('SWR004.stg', 'stage'),
         ('SWR004.flow', 'reachgroup'),
         ('SWR004.vel', 'qm')]


def test_swr_binary_stage(ipos=0):
    import flopy

    fpth = os.path.join(pth, files[ipos][0])
    swrtype = files[ipos][1]

    sobj = flopy.utils.SwrFile(fpth, swrtype=swrtype)
    assert isinstance(sobj, flopy.utils.SwrFile), 'SwrFile object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (0, 18), 'SwrFile records does not equal (0, 18)'

    times = sobj.get_times()
    assert times.shape == (336, 6), 'SwrFile times shape does not equal (336, 6)'

    ts = sobj.get_ts(rec_num=17)
    assert ts.shape == (336, 7), 'SwrFile timeseries shape does not equal (336, 7)'

    # plt.plot(ts[:, 0], ts[:, -1])
    # plt.show()

    return

if __name__ == '__main__':
    test_swr_binary_stage()
