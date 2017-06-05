"""
Module containing helper functions for plotting model data
using ModelMap and ModelCrossSection. Functions for plotting
shapefiles are also included.

"""
from __future__ import print_function
import os
import sys
import math
import numpy as np
try:
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [[0.26700401, 0.00487433, 0.32941519],
               [0.26851048, 0.00960483, 0.33542652],
               [0.26994384, 0.01462494, 0.34137895],
               [0.27130489, 0.01994186, 0.34726862],
               [0.27259384, 0.02556309, 0.35309303],
               [0.27380934, 0.03149748, 0.35885256],
               [0.27495242, 0.03775181, 0.36454323],
               [0.27602238, 0.04416723, 0.37016418],
               [0.2770184, 0.05034437, 0.37571452],
               [0.27794143, 0.05632444, 0.38119074],
               [0.27879067, 0.06214536, 0.38659204],
               [0.2795655, 0.06783587, 0.39191723],
               [0.28026658, 0.07341724, 0.39716349],
               [0.28089358, 0.07890703, 0.40232944],
               [0.28144581, 0.0843197, 0.40741404],
               [0.28192358, 0.08966622, 0.41241521],
               [0.28232739, 0.09495545, 0.41733086],
               [0.28265633, 0.10019576, 0.42216032],
               [0.28291049, 0.10539345, 0.42690202],
               [0.28309095, 0.11055307, 0.43155375],
               [0.28319704, 0.11567966, 0.43611482],
               [0.28322882, 0.12077701, 0.44058404],
               [0.28318684, 0.12584799, 0.44496],
               [0.283072, 0.13089477, 0.44924127],
               [0.28288389, 0.13592005, 0.45342734],
               [0.28262297, 0.14092556, 0.45751726],
               [0.28229037, 0.14591233, 0.46150995],
               [0.28188676, 0.15088147, 0.46540474],
               [0.28141228, 0.15583425, 0.46920128],
               [0.28086773, 0.16077132, 0.47289909],
               [0.28025468, 0.16569272, 0.47649762],
               [0.27957399, 0.17059884, 0.47999675],
               [0.27882618, 0.1754902, 0.48339654],
               [0.27801236, 0.18036684, 0.48669702],
               [0.27713437, 0.18522836, 0.48989831],
               [0.27619376, 0.19007447, 0.49300074],
               [0.27519116, 0.1949054, 0.49600488],
               [0.27412802, 0.19972086, 0.49891131],
               [0.27300596, 0.20452049, 0.50172076],
               [0.27182812, 0.20930306, 0.50443413],
               [0.27059473, 0.21406899, 0.50705243],
               [0.26930756, 0.21881782, 0.50957678],
               [0.26796846, 0.22354911, 0.5120084],
               [0.26657984, 0.2282621, 0.5143487],
               [0.2651445, 0.23295593, 0.5165993],
               [0.2636632, 0.23763078, 0.51876163],
               [0.26213801, 0.24228619, 0.52083736],
               [0.26057103, 0.2469217, 0.52282822],
               [0.25896451, 0.25153685, 0.52473609],
               [0.25732244, 0.2561304, 0.52656332],
               [0.25564519, 0.26070284, 0.52831152],
               [0.25393498, 0.26525384, 0.52998273],
               [0.25219404, 0.26978306, 0.53157905],
               [0.25042462, 0.27429024, 0.53310261],
               [0.24862899, 0.27877509, 0.53455561],
               [0.2468114, 0.28323662, 0.53594093],
               [0.24497208, 0.28767547, 0.53726018],
               [0.24311324, 0.29209154, 0.53851561],
               [0.24123708, 0.29648471, 0.53970946],
               [0.23934575, 0.30085494, 0.54084398],
               [0.23744138, 0.30520222, 0.5419214],
               [0.23552606, 0.30952657, 0.54294396],
               [0.23360277, 0.31382773, 0.54391424],
               [0.2316735, 0.3181058, 0.54483444],
               [0.22973926, 0.32236127, 0.54570633],
               [0.22780192, 0.32659432, 0.546532],
               [0.2258633, 0.33080515, 0.54731353],
               [0.22392515, 0.334994, 0.54805291],
               [0.22198915, 0.33916114, 0.54875211],
               [0.22005691, 0.34330688, 0.54941304],
               [0.21812995, 0.34743154, 0.55003755],
               [0.21620971, 0.35153548, 0.55062743],
               [0.21429757, 0.35561907, 0.5511844],
               [0.21239477, 0.35968273, 0.55171011],
               [0.2105031, 0.36372671, 0.55220646],
               [0.20862342, 0.36775151, 0.55267486],
               [0.20675628, 0.37175775, 0.55311653],
               [0.20490257, 0.37574589, 0.55353282],
               [0.20306309, 0.37971644, 0.55392505],
               [0.20123854, 0.38366989, 0.55429441],
               [0.1994295, 0.38760678, 0.55464205],
               [0.1976365, 0.39152762, 0.55496905],
               [0.19585993, 0.39543297, 0.55527637],
               [0.19410009, 0.39932336, 0.55556494],
               [0.19235719, 0.40319934, 0.55583559],
               [0.19063135, 0.40706148, 0.55608907],
               [0.18892259, 0.41091033, 0.55632606],
               [0.18723083, 0.41474645, 0.55654717],
               [0.18555593, 0.4185704, 0.55675292],
               [0.18389763, 0.42238275, 0.55694377],
               [0.18225561, 0.42618405, 0.5571201],
               [0.18062949, 0.42997486, 0.55728221],
               [0.17901879, 0.43375572, 0.55743035],
               [0.17742298, 0.4375272, 0.55756466],
               [0.17584148, 0.44128981, 0.55768526],
               [0.17427363, 0.4450441, 0.55779216],
               [0.17271876, 0.4487906, 0.55788532],
               [0.17117615, 0.4525298, 0.55796464],
               [0.16964573, 0.45626209, 0.55803034],
               [0.16812641, 0.45998802, 0.55808199],
               [0.1666171, 0.46370813, 0.55811913],
               [0.16511703, 0.4674229, 0.55814141],
               [0.16362543, 0.47113278, 0.55814842],
               [0.16214155, 0.47483821, 0.55813967],
               [0.16066467, 0.47853961, 0.55811466],
               [0.15919413, 0.4822374, 0.5580728],
               [0.15772933, 0.48593197, 0.55801347],
               [0.15626973, 0.4896237, 0.557936],
               [0.15481488, 0.49331293, 0.55783967],
               [0.15336445, 0.49700003, 0.55772371],
               [0.1519182, 0.50068529, 0.55758733],
               [0.15047605, 0.50436904, 0.55742968],
               [0.14903918, 0.50805136, 0.5572505],
               [0.14760731, 0.51173263, 0.55704861],
               [0.14618026, 0.51541316, 0.55682271],
               [0.14475863, 0.51909319, 0.55657181],
               [0.14334327, 0.52277292, 0.55629491],
               [0.14193527, 0.52645254, 0.55599097],
               [0.14053599, 0.53013219, 0.55565893],
               [0.13914708, 0.53381201, 0.55529773],
               [0.13777048, 0.53749213, 0.55490625],
               [0.1364085, 0.54117264, 0.55448339],
               [0.13506561, 0.54485335, 0.55402906],
               [0.13374299, 0.54853458, 0.55354108],
               [0.13244401, 0.55221637, 0.55301828],
               [0.13117249, 0.55589872, 0.55245948],
               [0.1299327, 0.55958162, 0.55186354],
               [0.12872938, 0.56326503, 0.55122927],
               [0.12756771, 0.56694891, 0.55055551],
               [0.12645338, 0.57063316, 0.5498411],
               [0.12539383, 0.57431754, 0.54908564],
               [0.12439474, 0.57800205, 0.5482874],
               [0.12346281, 0.58168661, 0.54744498],
               [0.12260562, 0.58537105, 0.54655722],
               [0.12183122, 0.58905521, 0.54562298],
               [0.12114807, 0.59273889, 0.54464114],
               [0.12056501, 0.59642187, 0.54361058],
               [0.12009154, 0.60010387, 0.54253043],
               [0.11973756, 0.60378459, 0.54139999],
               [0.11951163, 0.60746388, 0.54021751],
               [0.11942341, 0.61114146, 0.53898192],
               [0.11948255, 0.61481702, 0.53769219],
               [0.11969858, 0.61849025, 0.53634733],
               [0.12008079, 0.62216081, 0.53494633],
               [0.12063824, 0.62582833, 0.53348834],
               [0.12137972, 0.62949242, 0.53197275],
               [0.12231244, 0.63315277, 0.53039808],
               [0.12344358, 0.63680899, 0.52876343],
               [0.12477953, 0.64046069, 0.52706792],
               [0.12632581, 0.64410744, 0.52531069],
               [0.12808703, 0.64774881, 0.52349092],
               [0.13006688, 0.65138436, 0.52160791],
               [0.13226797, 0.65501363, 0.51966086],
               [0.13469183, 0.65863619, 0.5176488],
               [0.13733921, 0.66225157, 0.51557101],
               [0.14020991, 0.66585927, 0.5134268],
               [0.14330291, 0.66945881, 0.51121549],
               [0.1466164, 0.67304968, 0.50893644],
               [0.15014782, 0.67663139, 0.5065889],
               [0.15389405, 0.68020343, 0.50417217],
               [0.15785146, 0.68376525, 0.50168574],
               [0.16201598, 0.68731632, 0.49912906],
               [0.1663832, 0.69085611, 0.49650163],
               [0.1709484, 0.69438405, 0.49380294],
               [0.17570671, 0.6978996, 0.49103252],
               [0.18065314, 0.70140222, 0.48818938],
               [0.18578266, 0.70489133, 0.48527326],
               [0.19109018, 0.70836635, 0.48228395],
               [0.19657063, 0.71182668, 0.47922108],
               [0.20221902, 0.71527175, 0.47608431],
               [0.20803045, 0.71870095, 0.4728733],
               [0.21400015, 0.72211371, 0.46958774],
               [0.22012381, 0.72550945, 0.46622638],
               [0.2263969, 0.72888753, 0.46278934],
               [0.23281498, 0.73224735, 0.45927675],
               [0.2393739, 0.73558828, 0.45568838],
               [0.24606968, 0.73890972, 0.45202405],
               [0.25289851, 0.74221104, 0.44828355],
               [0.25985676, 0.74549162, 0.44446673],
               [0.26694127, 0.74875084, 0.44057284],
               [0.27414922, 0.75198807, 0.4366009],
               [0.28147681, 0.75520266, 0.43255207],
               [0.28892102, 0.75839399, 0.42842626],
               [0.29647899, 0.76156142, 0.42422341],
               [0.30414796, 0.76470433, 0.41994346],
               [0.31192534, 0.76782207, 0.41558638],
               [0.3198086, 0.77091403, 0.41115215],
               [0.3277958, 0.77397953, 0.40664011],
               [0.33588539, 0.7770179, 0.40204917],
               [0.34407411, 0.78002855, 0.39738103],
               [0.35235985, 0.78301086, 0.39263579],
               [0.36074053, 0.78596419, 0.38781353],
               [0.3692142, 0.78888793, 0.38291438],
               [0.37777892, 0.79178146, 0.3779385],
               [0.38643282, 0.79464415, 0.37288606],
               [0.39517408, 0.79747541, 0.36775726],
               [0.40400101, 0.80027461, 0.36255223],
               [0.4129135, 0.80304099, 0.35726893],
               [0.42190813, 0.80577412, 0.35191009],
               [0.43098317, 0.80847343, 0.34647607],
               [0.44013691, 0.81113836, 0.3409673],
               [0.44936763, 0.81376835, 0.33538426],
               [0.45867362, 0.81636288, 0.32972749],
               [0.46805314, 0.81892143, 0.32399761],
               [0.47750446, 0.82144351, 0.31819529],
               [0.4870258, 0.82392862, 0.31232133],
               [0.49661536, 0.82637633, 0.30637661],
               [0.5062713, 0.82878621, 0.30036211],
               [0.51599182, 0.83115784, 0.29427888],
               [0.52577622, 0.83349064, 0.2881265],
               [0.5356211, 0.83578452, 0.28190832],
               [0.5455244, 0.83803918, 0.27562602],
               [0.55548397, 0.84025437, 0.26928147],
               [0.5654976, 0.8424299, 0.26287683],
               [0.57556297, 0.84456561, 0.25641457],
               [0.58567772, 0.84666139, 0.24989748],
               [0.59583934, 0.84871722, 0.24332878],
               [0.60604528, 0.8507331, 0.23671214],
               [0.61629283, 0.85270912, 0.23005179],
               [0.62657923, 0.85464543, 0.22335258],
               [0.63690157, 0.85654226, 0.21662012],
               [0.64725685, 0.85839991, 0.20986086],
               [0.65764197, 0.86021878, 0.20308229],
               [0.66805369, 0.86199932, 0.19629307],
               [0.67848868, 0.86374211, 0.18950326],
               [0.68894351, 0.86544779, 0.18272455],
               [0.69941463, 0.86711711, 0.17597055],
               [0.70989842, 0.86875092, 0.16925712],
               [0.72039115, 0.87035015, 0.16260273],
               [0.73088902, 0.87191584, 0.15602894],
               [0.74138803, 0.87344918, 0.14956101],
               [0.75188414, 0.87495143, 0.14322828],
               [0.76237342, 0.87642392, 0.13706449],
               [0.77285183, 0.87786808, 0.13110864],
               [0.78331535, 0.87928545, 0.12540538],
               [0.79375994, 0.88067763, 0.12000532],
               [0.80418159, 0.88204632, 0.11496505],
               [0.81457634, 0.88339329, 0.11034678],
               [0.82494028, 0.88472036, 0.10621724],
               [0.83526959, 0.88602943, 0.1026459],
               [0.84556056, 0.88732243, 0.09970219],
               [0.8558096, 0.88860134, 0.09745186],
               [0.86601325, 0.88986815, 0.09595277],
               [0.87616824, 0.89112487, 0.09525046],
               [0.88627146, 0.89237353, 0.09537439],
               [0.89632002, 0.89361614, 0.09633538],
               [0.90631121, 0.89485467, 0.09812496],
               [0.91624212, 0.89609127, 0.1007168],
               [0.92610579, 0.89732977, 0.10407067],
               [0.93590444, 0.8985704, 0.10813094],
               [0.94563626, 0.899815, 0.11283773],
               [0.95529972, 0.90106534, 0.11812832],
               [0.96489353, 0.90232311, 0.12394051],
               [0.97441665, 0.90358991, 0.13021494],
               [0.98386829, 0.90486726, 0.13689671],
               [0.99324789, 0.90615657, 0.1439362]]

    viridis = LinearSegmentedColormap.from_list(__file__, cm_data)
except:
    pass



bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
                 'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy',
                 'STR': 'purple', 'SFR': 'blue'}


def _plot_array_helper(plotarray, model=None, sr=None, axes=None,
                       names=None, filenames=None, fignum=None,
                       mflay=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except:
        s = 'Could not import matplotlib.  Must install matplotlib ' +\
            ' in order to plot LayerFile data.'
        raise Exception(s)

    import flopy.plot.map as map
    

    # reshape 2d arrays to 3d for convenience
    if len(plotarray.shape) == 2:
        plotarray = plotarray.reshape((1, plotarray.shape[0],
                                       plotarray.shape[1]))

    # parse keyword arguments
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
    else:
        figsize = None

    if 'masked_values' in kwargs:
        masked_values = kwargs.pop('masked_values')
    else:
        masked_values = None

    if 'pcolor' in kwargs:
        pcolor = kwargs.pop('pcolor')
    else:
        pcolor = True

    if 'inactive' in kwargs:
        inactive = kwargs.pop('inactive')
    else:
        inactive = True

    if 'contour' in kwargs:
        contourdata = kwargs.pop('contour')
    else:
        contourdata = False

    if 'clabel' in kwargs:
        clabel = kwargs.pop('clabel')
    else:
        clabel = False

    if 'colorbar' in kwargs:
        cb = kwargs.pop('colorbar')
    else:
        cb = False

    if 'grid' in kwargs:
        grid = kwargs.pop('grid')
    else:
        grid = False

    if 'levels' in kwargs:
        levels = kwargs.pop('levels')
    else:
        levels = None

    if 'colors' in kwargs:
        colors = kwargs.pop('colors')
    else:
        colors = 'black'
    
    if 'dpi' in kwargs:
        dpi = kwargs.pop('dpi')
    else:
        dpi = None
    
    if 'fmt' in kwargs:
        fmt = kwargs.pop('fmt')
    else:
        fmt = '%1.3f'
    
    if mflay is not None:
        i0 = int(mflay)
        if i0+1 >= plotarray.shape[0]:
            i0 = plotarray.shape[0] - 1
        i1 = i0 + 1
    else:
        i0 = 0
        i1 = plotarray.shape[0]
    
    if names is not None:
        if not isinstance(names, list):
            names = [names]
        assert len(names) == plotarray.shape[0]
    
    if filenames is not None:
        if not isinstance(filenames, list):
            filenames = [filenames]
        assert len(filenames) == plotarray.shape[0]
    
    if fignum is not None:
        if not isinstance(fignum, list):
            fignum = [fignum]
        assert len(fignum) == plotarray.shape[0]
        # check for existing figures
        f0 = fignum[0]
        for i in plt.get_fignums():
            if i >= f0:
                f0 = i + 1
        finc = f0 - fignum[0]
        for idx in range(len(fignum)):
            fignum[idx] += finc
    else:
        #fignum = np.arange(i0, i1)
        # check for existing figures
        f0 = 0
        for i in plt.get_fignums():
            if i >= f0:
                f0 += 1
        f1 = f0 + (i1 - i0)
        fignum = np.arange(f0, f1)


    if axes is not None:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == plotarray.shape[0]
    # prepare some axis objects for use
    else:
        axes = []
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(figsize=figsize, num=fignum[idx])
            ax = plt.subplot(1, 1, 1, aspect='equal')
            if names is not None:
                title = names[k]
            else:
                klay = k
                if mflay is not None:
                    klay = int(mflay)
                title = '{} Layer {}'.format('data', klay+1)
            ax.set_title(title)
            axes.append(ax)
   
    for idx, k in enumerate(range(i0, i1)):
        fig = plt.figure(num=fignum[idx])
        mm = map.ModelMap(ax=axes[idx], model=model, sr=sr, layer=k)
        if pcolor:
            cm = mm.plot_array(plotarray[k], masked_values=masked_values,
                               ax=axes[idx], **kwargs)
            if cb:
                label = ''
                if not isinstance(cb,bool):
                    label = str(cb)
                plt.colorbar(cm, ax=axes[idx], shrink=0.5,label=label)

        if contourdata:
            cl = mm.contour_array(plotarray[k], masked_values=masked_values,
                                  ax=axes[idx], colors=colors, levels=levels, **kwargs)
            if clabel:
                axes[idx].clabel(cl, fmt=fmt,**kwargs)

        if grid:
            mm.plot_grid(ax=axes[idx])

        if inactive:
            try:
                ib = model.bas6.ibound.array
                mm.plot_inactive(ibound=ib, ax=axes[idx])
            except:
                pass

    if len(axes) == 1:
        axes = axes[0]
    if filenames is not None:
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(num=fignum[idx])
            fig.savefig(filenames[idx], dpi=dpi)
            print('    created...{}'.format(os.path.basename(filenames[idx])))
        # there will be nothing to return when done
        axes = None
        plt.close('all')
    return axes


def _plot_bc_helper(package, kper,
                    axes=None, names=None, filenames=None, fignum=None,
                    mflay=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except:
        s = 'Could not import matplotlib.  Must install matplotlib ' +\
            ' in order to plot boundary condition data.'
        raise Exception(s)

    import flopy.plot.map as map

    # reshape 2d arrays to 3d for convenience
    ftype = package.name[0]

    nlay = package.parent.nlay

    # parse keyword arguments
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
    else:
        figsize = None

    if 'inactive' in kwargs:
        inactive = kwargs.pop('inactive')
    else:
        inactive = True

    if 'grid' in kwargs:
        grid = kwargs.pop('grid')
    else:
        grid = False

    if 'dpi' in kwargs:
        dpi = kwargs.pop('dpi')
    else:
        dpi = None

    if 'masked_values' in kwargs:
        kwargs.pop('masked_values ')

    if mflay is not None:
        i0 = int(mflay)
        if i0+1 >= nlay:
            i0 = nlay - 1
        i1 = i0 + 1
    else:
        i0 = 0
        i1 = nlay

    if names is not None:
        if not isinstance(names, list):
            names = [names]
        assert len(names) == nlay

    if filenames is not None:
        if not isinstance(filenames, list):
            filenames = [filenames]
        assert len(filenames) == (i1 - i0)

    if fignum is not None:
        if not isinstance(fignum, list):
            fignum = [fignum]
        assert len(fignum) == (i1 - i0)
        # check for existing figures
        f0 = fignum[0]
        for i in plt.get_fignums():
            if i >= f0:
                f0 = i + 1
        finc = f0 - fignum[0]
        for idx in range(len(fignum)):
            fignum[idx] += finc
    else:
        #fignum = np.arange(i0, i1)
        # check for existing figures
        f0 = 0
        for i in plt.get_fignums():
            if i >= f0:
                f0 += 1
        f1 = f0 + (i1 - i0)
        fignum = np.arange(f0, f1)

    if axes is not None:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == i1 - i0
    # prepare some axis objects for use
    else:
        axes = []
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(figsize=figsize, num=fignum[idx])
            ax = plt.subplot(1, 1, 1, aspect='equal')
            if names is not None:
                title = names[k]
            else:
                klay = k
                if mflay is not None:
                    klay = int(mflay)
                title = '{} Layer {}'.format('data', klay+1)
            ax.set_title(title)
            axes.append(ax)

    for idx, k in enumerate(range(i0, i1)):
        mm = map.ModelMap(ax=axes[idx], model=package.parent, layer=k)
        fig = plt.figure(num=fignum[idx])
        qm = mm.plot_bc(ftype=ftype, package=package, kper=kper, ax=axes[idx])

        if grid:
            mm.plot_grid(ax=axes[idx])

        if inactive:
            try:
                ib = package.parent.bas6.ibound.array
                mm.plot_inactive(ibound=ib, ax=axes[idx])
            except:
                pass

    if len(axes) == 1:
        axes = axes[0]

    if filenames is not None:
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(num=fignum[idx])
            fig.savefig(filenames[idx], dpi=dpi)
            plt.close(fignum[idx])
            print('    created...{}'.format(os.path.basename(filenames[idx])))
        # there will be nothing to return when done
        axes = None
        plt.close('all')
    return axes


class SwiConcentration():
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files

    """
    def __init__(self, model=None, botm=None, istrat=1, nu=None):
        if model is None:
            if isinstance(botm, list):
                botm = np.array(botm)
            self.__botm = botm
            if isinstance(nu, list):
                nu = np.array(nu)
            self.__nu = nu
            self.__istrat = istrat
            if istrat == 1:
                self.__nsrf = self.nu.shape - 1
            else:
                self.__nsrf = self.nu.shape - 2
        else:
            try:
                dis = model.get_package('DIS')
            except:
                sys.stdout.write('Error: DIS package not available.\n')
            self.__botm = np.zeros((dis.nlay+1, dis.nrow, dis.ncol), np.float)
            self.__botm[0, :, :] = dis.top.array
            self.__botm[1:, :, :] = dis.botm.array
            try:
                swi = model.get_package('SWI2')
                self.__nu = swi.nu.array
                self.__istrat = swi.istrat
                self.__nsrf = swi.nsrf
            except:
                sys.stdout.write('Error: SWI2 package not available...\n')
        self.__nlay = self.__botm.shape[0] - 1
        self.__nrow = self.__botm[0, :, :].shape[0]
        self.__ncol = self.__botm[0, :, :].shape[1]
        self.__b = self.__botm[0:-1, :, :] - self.__botm[1:, :, :] 
     
    def calc_conc(self, zeta, layer=None):
        """
        Calculate concentrations for a given time step using passed zeta.

        Parameters
        ----------
        zeta : dictionary of numpy arrays
            Dictionary of zeta results. zeta keys are zero-based zeta surfaces.
        layer : int
            Concentration will be calculated for the specified layer.  If layer 
            is None, then the concentration will be calculated for all layers. 
            (default is None).

        Returns
        -------
        conc : numpy array
            Calculated concentration.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('test')
        >>> c = flopy.plot.SwiConcentration(model=m)
        >>> conc = c.calc_conc(z, layer=0)

        """
        conc = np.zeros((self.__nlay, self.__nrow, self.__ncol), np.float)
        
        pct = {}
        for isrf in range(self.__nsrf):
            z = zeta[isrf]
            pct[isrf] = (self.__botm[:-1, :, :] - z[:, :, :]) / self.__b[:, :, :]
        for isrf in range(self.__nsrf):
            p = pct[isrf]
            if self.__istrat == 1:
                conc[:, :, :] += self.__nu[isrf] * p[:, :, :]
                if isrf+1 == self.__nsrf:
                    conc[:, :, :] += self.__nu[isrf+1] * (1. - p[:, :, :])
            #TODO linear option
        if layer is None:
            return conc
        else:
            return conc[layer, :, :]



def shapefile_extents(shp):
    """
    Determine the extents of a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.

    Returns
    -------
    extents : tuple
        tuple with xmin, xmax, ymin, ymax from shapefile.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> extent = flopy.plot.plotutil.shapefile_extents(fshp)

    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    xmin, xmax, ymin, ymax = 1.e20, -1.e20, 1.e20, -1.e20
    ptchs = []
    for n in range(nshp):
        for p in shapes[n].points:
            xmin, xmax = min(xmin, p[0]), max(xmax, p[0])
            ymin, ymax = min(ymin, p[1]), max(ymax, p[1])
    return xmin, xmax, ymin, ymax


def shapefile_get_vertices(shp):
    """
    Get vertices for the features in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to extract shapefile feature vertices.

    Returns
    -------
    vertices : list
        Vertices is a list with vertices for each feature in the shapefile. 
        Individual feature vertices are x, y tuples and contained in a list.
        A list with a single x, y tuple is returned for point shapefiles. A
        list with multiple x, y tuples is returned for polyline and polygon
        shapefiles.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> lines = flopy.plot.plotutil.shapefile_get_vertices(fshp)
    
    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    vertices = []
    for n in range(nshp):
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            #points
            for p in shapes[n].points:
                vertices.append([(p[0], p[1])])
        elif st in [3, 13, 23]:
            #line
            line = []
            for p in shapes[n].points:
                line.append((p[0], p[1]))
            line = np.array(line)
            vertices.append(line)
        elif st in [5, 25, 31]:
            #polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                vertices.append(pts[par[pij]:par[pij+1]])
    return vertices
    

def shapefile_to_patch_collection(shp, radius=500., idx=None):
    """
    Create a patch collection from the shapes in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.
    radius : float
        Radius of circle for points in the shapefile.  (Default is 500.)
    idx : iterable int
        A list or array that contains shape numbers to include in the
        patch collection.  Return all shapes if not specified.

    Returns
    -------
        pc : matplotlib.collections.PatchCollection
            Patch collection of shapes in the shapefile

    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    from matplotlib.patches import Polygon, Circle, Path, PathPatch
    from matplotlib.collections import PatchCollection
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    ptchs = []
    if idx is None:
        idx = range(nshp)
    for n in idx:
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            # points
            for p in shapes[n].points:
                ptchs.append(Circle( (p[0], p[1]), radius=radius))
        elif st in [3, 13, 23]:
            # line
            vertices = []
            for p in shapes[n].points:
                vertices.append([p[0], p[1]])
            vertices = np.array(vertices)
            path = Path(vertices)
            ptchs.append(PathPatch(path, fill=False))
        elif st in [5, 25, 31]:
            # polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
    pc = PatchCollection(ptchs)
    return pc


def plot_shapefile(shp, ax=None, radius=500., cmap='Dark2',
                   edgecolor='scaled', facecolor='scaled',
                   a=None, masked_values=None, idx=None, **kwargs):
    """
    Generic function for plotting a shapefile.

    Parameters
    ----------
    shp : string
        Name of the shapefile to plot.
    radius : float
        Radius of circle for points.  (Default is 500.)
    linewidth : float
        Width of all lines. (default is 1)
    cmap : string
        Name of colormap to use for polygon shading (default is 'Dark2')
    edgecolor : string
        Color name.  (Default is 'scaled' to scale the edge colors.)
    facecolor : string
        Color name.  (Default is 'scaled' to scale the face colors.)
    a : numpy.ndarray
        Array to plot.
    masked_values : iterable of floats, ints
        Values to mask.
    idx : iterable int
        A list or array that contains shape numbers to include in the
        patch collection.  Return all shapes if not specified.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------

    """

    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    import numpy as np
    import matplotlib.pyplot as plt

    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = None

    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = None

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc = shapefile_to_patch_collection(shp, radius=radius, idx=idx)
    pc.set(**kwargs)
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1. * np.arange(nshp) / nshp)
        if facecolor == 'scaled':
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == 'scaled':
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == 'scaled':
            pc.set_edgecolor('none')
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a)
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    return pc


def cvfd_to_patch_collection(verts, iverts):
    """
    Create a patch collection from control volume vertices and incidence list

    Parameters
    ----------
    verts : ndarray
        2d array of x and y points.
    iverts : list of lists
        should be of len(ncells) with a list of vertex numbers for each cell

    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    ptchs = []
    for icell, ivertlist in enumerate(iverts):
        points = []
        for iv in ivertlist:
            points.append((verts[iv, 0], verts[iv, 1]))
        # close the polygon, if necessary
        if ivertlist[0] != ivertlist[-1]:
            iv = ivertlist[0]
            points.append((verts[iv, 0], verts[iv, 1]))
        ptchs.append(Polygon(points))
    pc = PatchCollection(ptchs)
    return pc


def plot_cvfd(verts, iverts, ax=None, layer=0, cmap='Dark2',
              edgecolor='scaled', facecolor='scaled', a=None,
              masked_values=None, **kwargs):
    """
    Generic function for plotting a control volume finite difference grid of
    information.

    Parameters
    ----------
    verts : ndarray
        2d array of x and y points.
    iverts : list of lists
        should be of len(ncells) with a list of vertex number for each cell
    ax : matplotlib.pylot axis
        matplotlib.pyplot axis instance. Default is None
    layer : int
        layer to extract. Used in combination to the optional ncpl
        parameter. Default is 0
    linewidth : float
        Width of all lines. (default is 1)
    cmap : string
        Name of colormap to use for polygon shading (default is 'Dark2')
    edgecolor : string
        Color name.  (Default is 'scaled' to scale the edge colors.)
    facecolor : string
        Color name.  (Default is 'scaled' to scale the face colors.)
    a : numpy.ndarray
        Array to plot.
    masked_values : iterable of floats, ints
        Values to mask.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------

    """
    import numpy as np
    import matplotlib.pyplot as plt

    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = None

    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = None

    if 'ncpl' in kwargs:
        nlay = layer + 1
        ncpl = kwargs.pop('ncpl')
        if isinstance(ncpl, int):
            i = int(ncpl)
            ncpl = np.ones((nlay), dtype=np.int) * i
        elif isinstance(ncpl, list) or isinstance(ncpl, tuple):
            ncpl = np.array(ncpl)
        i0 = 0
        i1 = 0
        for k in range(nlay):
            i0 = i1
            i1 = i0 + ncpl[k]
        # retain iverts in selected layer
        iverts = iverts[i0:i1]
        # retain vertices in selected layer
        tverts = []
        for iv in iverts:
            for iloc in iv:
                tverts.append((verts[iloc, 0], verts[iloc, 1]))
        verts = np.array(tverts)
        # calculate offset for starting vertex in layer based on
        # global vertex numbers
        iadj = iverts[0][0]
        # reset iverts to relative vertices in selected layer
        tiverts = []
        for iv in iverts:
            i = []
            for t in iv:
                i.append(t-iadj)
            tiverts.append(i)
        iverts = tiverts
    else:
        i0 = 0
        i1 = len(iverts)

    # get current axis
    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)

    pc = cvfd_to_patch_collection(verts, iverts)
    pc.set(**kwargs)

    # set colors
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1. * np.arange(nshp) / nshp)
        if facecolor == 'scaled':
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == 'scaled':
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == 'scaled':
            pc.set_edgecolor('none')
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a[i0:i1])
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    return pc


def saturated_thickness(head, top, botm, laytyp, mask_values=None):
    """
    Calculate the saturated thickness.

    Parameters
    ----------
    head : numpy.ndarray
        head array
    top : numpy.ndarray
        top array of shape (nrow, ncol)
    botm : numpy.ndarray
        botm array of shape (nlay, nrow, ncol)
    laytyp : numpy.ndarray
        confined (0) or convertible (1) of shape (nlay)
    mask_values : list of floats
        If head is one of these values, then set sat to top - bot

    Returns
    -------
    sat_thk : numpy.ndarray
        Saturated thickness of shape (nlay, nrow, ncol).

    """
    nlay, nrow, ncol = head.shape
    sat_thk = np.empty(head.shape, dtype=head.dtype)
    for k in range(nlay):
        if k == 0:
            t = top
        else:
            t = botm[k-1, :, :]
        sat_thk[k, :, :] = t - botm[k, :, :]
    for k in range(nlay):
        if laytyp[k] != 0:
            dh = np.zeros((nrow, ncol), dtype=head.dtype)
            s = sat_thk[k, :, :]

            for mv in mask_values:
                idx = (head[k, :, :] == mv)
                dh[idx] = s[idx]

            if k == 0:
                t = top
            else:
                t = botm[k-1, :, :]
            t = np.where(head[k, :, :] > t, t, head[k, :, :])
            dh = np.where(dh == 0, t - botm[k, :, :], dh)
            sat_thk[k, :, :] = dh[:, :]
    return sat_thk


def centered_specific_discharge(Qx, Qy, Qz, delr, delc, sat_thk):
    """
    Using the MODFLOW discharge, calculate the cell centered specific discharge
    by dividing by the flow width and then averaging to the cell center.

    Parameters
    ----------
    Qx : numpy.ndarray
        MODFLOW 'flow right face'
    Qy : numpy.ndarray
        MODFLOW 'flow front face'.  The sign on this array will be flipped
        by this function so that the y axis is positive to north.
    Qz : numpy.ndarray
        MODFLOW 'flow lower face'.  The sign on this array will be flipped by
        this function so that the z axis is positive in the upward direction.
    delr : numpy.ndarray
        MODFLOW delr array
    delc : numpy.ndarray
        MODFLOW delc array
    sat_thk : numpy.ndarray
        Saturated thickness for each cell

    Returns
    -------
    (qx, qy, qz) : tuple of numpy.ndarrays
        Specific discharge arrays that have been interpolated to cell centers.

    """
    qx = None
    qy = None
    qz = None

    if Qx is not None:

        nlay, nrow, ncol = Qx.shape
        qx = np.zeros(Qx.shape, dtype=Qx.dtype)

        for k in range(nlay):
            for j in range(ncol-1):
                area = delc[:] * 0.5 * (sat_thk[k, :, j] + sat_thk[k, :, j + 1])
                idx = area > 0.
                qx[k, idx, j] = Qx[k, idx, j] / area[idx]

        qx[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol-1] + qx[:, :, 1:ncol])
        qx[:, :, 0] = 0.5 * qx[:, :, 0]

    if Qy is not None:

        nlay, nrow, ncol = Qy.shape
        qy = np.zeros(Qy.shape, dtype=Qy.dtype)

        for k in range(nlay):
            for i in range(nrow-1):
                area = delr[:] * 0.5 * (sat_thk[k, i, :] + sat_thk[k, i + 1, :])
                idx = area > 0.
                qy[k, i, idx] = Qy[k, i, idx] / area[idx]

        qy[:, 1:, :] = 0.5 * (qy[:, 0:nrow-1, :] + qy[:, 1:nrow, :])
        qy[:, 0, :] = 0.5 * qy[:, 0, :]
        qy = -qy


    if Qz is not None:
        qz = np.zeros(Qz.shape, dtype=Qz.dtype)
        dr = delr.reshape((1, delr.shape[0]))
        dc = delc.reshape((delc.shape[0], 1))
        area = dr * dc
        for k in range(nlay):
            qz[k, :, :] = Qz[k, :, :] / area[:, :]
        qz[1:, :, :] = 0.5 * (qz[0:nlay-1, :, :] + qz[1:nlay, :, :])
        qz[0, :, :] = 0.5 * qz[0, :, :]
        qz = -qz


    return (qx, qy, qz)
    


def findrowcolumn(pt, xedge, yedge):
    """
    Find the MODFLOW cell containing the x- and y- point provided.

    Parameters
    ----------
    pt : list or tuple
        A list or tuple containing a x- and y- coordinate
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.

    Returns
    -------
    irow, jcol : int
        Row and column location containing x- and y- point passed to function.

    Examples
    --------
    >>> import flopy
    >>> irow, jcol = flopy.plotutil.findrowcolumn(pt, xedge, yedge)

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # find column
    jcol = -100
    for jdx, xmf in enumerate(xedge):
        if xmf > pt[0]:
            jcol = jdx - 1
            break

    # find row
    irow = -100
    for jdx, ymf in enumerate(yedge):
        if ymf < pt[1]:
            irow = jdx - 1
            break
    return irow, jcol


def line_intersect_grid(ptsin, xedge, yedge, returnvertices=False):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    ptsin : list
        A list of x, y points defining the vertices of a polyline that will be
        intersected with the rectilinear MODFLOW grid
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    returnvertices: bool
        Return the original polyline vertices in the list of numpy.ndarray
        containing vertices resulting from intersection of the provided
        polygon and the MODFLOW model grid if returnvertices=True.
        (default is False).

    Returns
    -------
    (x, y, dlen) : numpy.ndarray of tuples
        numpy.ndarray of tuples containing the x, y, and segment length of the
        intersection of the provided polyline with the rectilinear MODFLOW
        grid.

    Examples
    --------
    >>> import flopy
    >>> ptsout = flopy.plotutil.line_intersect_grid(ptsin, xedge, yedge)

    """

    small_value = 1.0e-4

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # build list of points along current line
    pts = []
    npts = len(ptsin)
    dlen = 0.
    for idx in range(1, npts):
        x0 = ptsin[idx - 1][0]
        x1 = ptsin[idx][0]
        y0 = ptsin[idx - 1][1]
        y1 = ptsin[idx][1]
        a = x1 - x0
        b = y1 - y0
        c = math.sqrt(math.pow(a, 2.) + math.pow(b, 2.))
        # find cells with (x0, y0) and (x1, y1)
        irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
        irow1, jcol1 = findrowcolumn((x1, y1), xedge, yedge)
        # determine direction to go in the x- and y-directions
        jx = 0
        incx = abs(small_value * a / c)
        iy = 0
        incy = -abs(small_value * b / c)
        if a == 0.:
            incx = 0.
        # go to the right
        elif a > 0.:
            jx = 1
            incx *= -1.
        if b == 0.:
            incy = 0.
        # go down
        elif b < 0.:
            iy = 1
            incy *= -1.
        # process data
        if irow0 >= 0 and jcol0 >= 0:
            iadd = True
            if idx > 1 and returnvertices: 
                iadd = False
            if iadd: 
                pts.append((x0, y0, dlen))
        icnt = 0
        while True:
            icnt += 1
            dx = xedge[jcol0 + jx] - x0
            dlx = 0.
            if a != 0.:
                dlx = c * dx / a
            dy = yedge[irow0 + iy] - y0
            dly = 0.
            if b != 0.:
                dly = c * dy / b
            if dlx != 0. and dly != 0.:
                if abs(dlx) < abs(dly):
                    dy = dx * b / a
                else:
                    dx = dy * a / b
            xt = x0 + dx + incx
            yt = y0 + dy + incy
            dl = math.sqrt(math.pow((xt - x0), 2.) + math.pow((yt - y0), 2.))
            dlen += dl
            if not returnvertices: 
                pts.append((xt, yt, dlen))
            x0, y0 = xt, yt
            xt = x0 - 2. * incx
            yt = y0 - 2. * incy
            dl = math.sqrt(math.pow((xt - x0), 2.) + math.pow((yt - y0), 2.))
            dlen += dl
            x0, y0 = xt, yt
            irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
            if irow0 >= 0 and jcol0 >= 0:
                if not returnvertices: 
                    pts.append((xt, yt, dlen))
            elif irow1 < 0 or jcol1 < 0:
                dl = math.sqrt(math.pow((x1 - x0), 2.) + math.pow((y1 - y0), 2.))
                dlen += dl
                break
            if irow0 == irow1 and jcol0 == jcol1:
                dl = math.sqrt(math.pow((x1 - x0), 2.) + math.pow((y1 - y0), 2.))
                dlen += dl
                pts.append((x1, y1, dlen))
                break
    return np.array(pts)


def cell_value_points(pts, xedge, yedge, vdata):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    pts : list
        A list of x, y points and polyline length to extract defining the
        vertices of a polyline that
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. The shape of xedge is
        (NCOL + 1). If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. The shape of yedge is
        (NROW + 1). If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    vdata : numpy.ndarray
        Data (i.e., head, hk, etc.) for a rectilinear MODFLOW model grid. The
        shape of vdata is (NROW, NCOL). If vdata is not a numpy.ndarray it is
        converted to a numpy.ndarray.

    Returns
    -------
    vcell : numpy.ndarray
        numpy.ndarray of of data values from the vdata numpy.ndarray at x- and
        y-coordinate locations in pts.

    Examples
    --------
    >>> import flopy
    >>> vcell = flopy.plotutil.cell_value_points(xpts, xedge, yedge, head[0, :, :])

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)
    if not isinstance(vdata, np.ndarray):
        vdata = np.array(vdata)

    vcell = []
    for idx, [xt, yt, dlen] in enumerate(pts):
        # find the modflow cell containing point
        irow, jcol = findrowcolumn((xt, yt), xedge, yedge)
        if irow >= 0 and jcol >= 0:
            if np.isnan(vdata[irow, jcol]):
                vcell.append(np.nan)
            else:
                v = np.asarray(vdata[irow, jcol])
                vcell.append(v) 

    return np.array(vcell)





