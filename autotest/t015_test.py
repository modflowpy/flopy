__author__ = 'aleaf'

import os
import matplotlib

matplotlib.use('agg')
import flopy

print(os.getcwd())

if os.path.split(os.getcwd())[-1] == 'flopy3':
    path = os.path.join('examples', 'data', 'mf2005_test')
else:
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')

str_items = {0: {'mfnam': 'str.nam',
                 'sfrfile': 'str.str'}}


def test_str_plot():
    m = flopy.modflow.Modflow.load(str_items[0]['mfnam'], model_ws=path,
                                   verbose=True)
    assert isinstance(m.str.plot()[0], matplotlib.axes.Axes)


if __name__ == '__main__':
    test_str_plot()
