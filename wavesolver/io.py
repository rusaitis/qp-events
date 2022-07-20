import os
import shutil
import matplotlib as mpl
from wavesolver.linalg import *
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import colorcet as cc
import colorsys
# ----------------------------------------------------------------------------

# COLORS
clist = ['limegreen', 'firebrick', 'slateblue', 'brown', 'chocolate',
         'coral', 'hotpink', 'royalblue', 'mediumaquamarine', 'darkslategrey']
cxkcd = ['xkcd:electric blue', 'xkcd:barbie pink', 'xkcd:emerald',
         'xkcd:dandelion', 'xkcd:barney purple', 'xkcd:salmon', 'xkcd:blurple']
cxkcd = ['xkcd:electric blue', 'xkcd:barbie pink', 'xkcd:emerald',
         'xkcd:dandelion', 'xkcd:pale orange', 'xkcd:grey blue',
         'xkcd:blurple']

# teal light brown mauve grey blue
cgreys = ['black', 'dimgrey', 'darkgrey', 'lightgrey', 'gainsboro',
          'whitesmoke']
cdarks = ['seagreen', 'navy', 'indigo', 'maroon', 'darkorange', 'olive']
clights = ['mediumseagreen', 'mediumblue', 'darkviolet', 'firebrick',
           'orange', 'yellow']

# ----------------------------------------------------------------------------

import json



def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

class Config(dict):
    """
    Store configuration data in dictionary and access via dot notation.
    https://github.com/maet3608/config-dict
    """
    def __init__(self, *args, **kwargs):
        """ Construct the same way a plain Python dict is created. """
        wrap = lambda v: Config(v) if type(v) is dict else v
        kvdict = {k: wrap(v) for k, v in dict(*args, **kwargs).items()}
        super(Config, self).__init__(kvdict)
        self.__dict__ = self

    @staticmethod
    def load(filepath):
        """Load configuration from given JSON filepath"""
        with open(filepath) as f:
            return Config(json.load(f))

    def save(self, filepath):
        """Save configuration to given JSON filepath"""
        with open(filepath, 'w') as f:
            json.dump(self, f, indent=2)

    def __repr__(self):
        """Pretty string representation of configuration"""
        return json.dumps(self, sort_keys=True, indent=2)


class io():
    """Input/Ouput Configuration"""
    def __init__(self,
                 path='',
                 data_dir='',
                 movie=False,
                 save=True,
                 id=0,
                 name=None,
                 format='.png',
                 plotCoord='th',
                 plotx=None,
                 ploty=None,
                 Lmax=21,
                 plotRefEigs=False,
                 time=None) -> dict:
        self.path = path
        self.data_dir = data_dir
        self.movie = movie
        self.save = save
        self.id = id
        self.name = name
        self.plotCoord = plotCoord
        self.format = format
        self.plotx = plotx
        self.ploty = ploty
        self.Lmax = Lmax
        self.plotRefEigs = plotRefEigs
        self.time = time

#  class io():
    #  """ Input/Output Configuration """
    #  def __init__(self,



# ----------------------------------------------------------------------------


def getopts(argv):
    """ Get command line arguments (made redundant by argparse package) """
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            if argv[1]:
                val = argv[1]
            opts[argv[0]] = val  # Add key and value to the dictionary.
        # Reduce the argument list by copying it starting from index 1.
        argv = argv[1:]
    return opts
# ----------------------------------------------------------------------------

def createDirectory(DIR):
    """ Create a new directory if it doesn't already exist """
    os.makedirs(DIR, exist_ok=True)

# ----------------------------------------------------------------------------


def deleteFiles(DIR, fileformat='png'):
    """ Delete files of a given format in a directory (USE CAUTION!) """
    for filename in os.listdir(DIR):
        file_path = os.path.join(DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                if os.path.splitext(file_path)[1] == fileformat:
                    os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# ----------------------------------------------------------------------------

    print('Bins = {:d} by {:d} by {:d}'.format(Nbins_Lat, Nbins_LT, Nbins_Mlat))

# ----------------------------------------------------------------------------
def dataDisplayConfig(data, planet='earth'):
    scale = None
    vmin = None
    vmax = None
    vmin2 = None
    vmax2 = None
    ticks = None
    ticklabels = None
    barlabel = None
    cmap = None

    cmap = mpl.cm.get_cmap('rainbow') #'BuPu', 'magma', 'cividis', 'jet', 'rainbow4'
    cmap = cc.cm.rainbow

    # Adjust colors for values beyond the range
    cl_under = scale_lightness(list(cmap(0))[0:3], 0.5)
    cl_over = scale_lightness(list(cmap(255))[0:3], 0.5)
    cmap.set_over(cl_over)
    cmap.set_under(cl_under)

    if data == 'days':
        scale = 1
        vmin = 0.1
        vmax = 4
        vmin2 = 0.1
        vmax2 = 4
        norm = Normalize(vmin=vmin, vmax=vmax)
        ticks = [0.1, 2, 4]
        ticklabels = ['0.1', '2', '4']
        barlabel = r'Days [h]'

    if planet == 'earth':
        if data == 'n':
            scale = 1E-6
            vmin = 1
            vmax = 12
            vmin2 = 0
            vmax2 = 30
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticks = [1, 5, 10]
            ticklabels = ['1', '5', '10']
            barlabel = r'Electron Density [$\mathrm{cm}^{-3}]$'
        if data == 'm':
            scale = 1
            vmin = 1
            vmax = 16
            vmin2 = 1
            vmax2 = 16
            norm = Normalize(vmin=vmin, vmax=vmax)
            ticks = [4, 8, 12, 16]
            ticklabels = ['4', '8', '12', '16']
            barlabel = r'Average Ion Mass [amu]'
        if data == 'rho':
            scale = 1E-6
            vmin = 10
            vmax = 110
            vmin2 = 0
            vmax2 = 140
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticks = [10, 100]
            ticklabels = ['10', '100']
            barlabel = r'Mass Density [$\mathrm{amu cm}^{-3}]$'
        if data == 'BT':
            scale = 1E-9
            scale = 1
            vmin = 10
            vmax = 1000
            vmin2 = 10
            vmax2 = 1000
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticks = [10, 100, 1000]
            ticklabels = ['10', '100', '1000']
            barlabel = r'Magnetic Field Strength [nT]'
        if data == 'vA':
            scale = 1E-3
            vmin = 10.
            vmax = 10000
            vmin2 = 10
            vmax2 = 10000
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticks = [10, 100, 1000, 10000]
            ticklabels = ['10', '100', '1000', '10000']
            ticklabels = [r'$10$', r'$10^2$', r'$10^3$', r'$10^4$']
            barlabel = r'Alfven Velocity [$\mathrm{km s}^{-1}]$'
    else:
        if data == 'n':
            scale = 1E-6
            # scale = 1
            vmin = 0.5
            vmax = 50.
            vmin = 0.1
            vmax = 70.
            vmin = 0.005
            vmax = 70.
            vmin = 0.01
            vmax = 70.
            norm = Normalize(vmin=vmin, vmax=vmax)
            ticks = [0.5, 1., 5., 10., 50.]
            ticks = [0.1, 0.5, 1., 5., 10., 70.]
            ticks = [0.005, 0.5, 1., 5., 10., 70.]
            ticks = [0.01, 0.1, 1., 10., 70.]
            ticklabels = ['0.5', '1', '5', '10', '50']
            ticklabels = ['0.1', '0.5', '1', '5', '10', '70']
            ticklabels = ['0.005', '0.5', '1', '5', '10', '70']
            ticklabels = ['0.01', '0.1', '1', '10', '70']
            barlabel = r'Water Group Ion Density ($cm^{-3})$'
            barlabel = r'Plasma Density ($cm^{-3})$'
        if data == 'BT':
            scale = 1E9
            vmin = 1
            vmax = 10000
            ticks = [1, 10, 100, 1000, 10000]
            norm = LogNorm(vmin=vmin, vmax=vmax)
            # ticklabels = ['10', '100', '1000', '10000']
            ticklabels = [r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$']
            barlabel = r'Magnetic Field (nT)'
        if data == 'vA':
            scale = 1E-3
            vmin = 1.
            vmax = 10000
            vmin = 10.
            vmax = 10000
            ticks = [1, 10, 100, 1000, 10000]
            ticks = [10, 100, 1000, 10000]
            norm = LogNorm(vmin=vmin, vmax=vmax)
            ticklabels = ['1', '10', '100', '1000', '10000']
            ticklabels = ['10', '100', '1000', '10000']
            ticklabels = [r'$10$', r'$10^2$', r'$10^3$', r'$10^4$']
            barlabel = r'Alfven Velocity ($kms^{-1})$'
        if data == 'w':
            scale = 1.
            vmin = 1.
            vmax = 75
            ticks = [1., 15., 30, 45, 60, 75]
            norm = Normalize(vmin=vmin, vmax=vmax)
            ticklabels = ['1', '15', '30', '45', '60', '75']
            barlabel = r'Period ($min)$'
    return scale, vmin, vmax, vmin2, vmax2, norm, ticks, ticklabels, barlabel, cmap

def fancyVarName(var):
    ''' Fancy text for plotting '''
    if var == 'E':
        return r'$E_{\perp}$'
    elif var == 'b':
        return r'$b_{\perp}$'
    elif var == 'b/B':
        return r'$b_{\perp}/B$'
    elif var == 'xi':
        return r'$\xi_{\perp}$'
    else:
        return ''

# ----------------------------------------------------------------------------


def selectSolutionVar(var, solution, SIM, norm=True):
    ''' Select an appropriate variable of the simulation '''
    if var == 'E':
        y = solution.E
    elif var == 'b':
        y = solution.b
    elif var == 'b/B':
        y = solution.b / SIM.B
    elif var == 'xi':
        y = solution.xi
    return normalize(y) if norm else y

# ----------------------------------------------------------------------------


def toPlotCoords(z, SIM):
    ''' Convert simulation coordinates to plot coordinates '''
    if SIM.coords == 'cartesian':
        return z / SIM.units
    if SIM.coords == 'ds':
        return z / SIM.units
    if SIM.coords == 'cos':
        return np.arccos(z) * 180. / np.pi
    if SIM.coords == 'deg':
        return z * 180. / np.pi

# ----------------------------------------------------------------------------


def fromPlotCoords(z, SIM):
    ''' Convert from plot coordinates to simulation coordinates'''
    if SIM.coords == 'cartesian':
        return z * SIM.units
    if SIM.coords == 'ds':
        return z * SIM.units
    if SIM.coords == 'cos':
        return np.cos(z / 180. * np.pi)
    if SIM.coords == 'deg':
        return z / 180. * np.pi

