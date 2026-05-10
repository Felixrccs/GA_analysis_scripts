import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

def get_black():
    return [0., 0., 0., 0.95]

def get_white():
    return [1., 1., 1., 1.]

def get_mpg_green():
    return [0., 108./255., 102./255., 1.]

def get_mpg_green_dark():
    return [0., 85./255., 85./255., 1.]

def get_mpg_green_light():
    return [198./255., 211./255., 37./255., 1.]

def get_mpg_grey():
    return [167./255., 167./255., 168./255., 1.]

def get_mpg_grey_dark():
    return [119./255., 119./255., 119./255., 1.]

def get_mpg_grey_light():
    return [238./255., 238./255., 238./255., 1.]

def get_mpg_blue_dark():
    return [41./255., 72./255., 93./255., 1.]

def get_mpg_blue_light():
    return [0./255., 177./255., 234./255., 1.]

def get_mpg_blue_dark():
    return [239./255., 124./255., 0./255., 1.]

def get_tumblau():
    return [0, 101./255., 189./255., 1.]

def get_tumblau_light():
    return [152./255., 198./255., 234./255., 1]

def get_tumblau_dark():
    return [0, 82./255., 147./255., 1.]

def get_tumorange():
    return [227./255., 114./255., 34./255., 1.]

def get_tumgreen():
    return [162./255., 173./255., 0., 1.]

def get_mpg_cmp(reverse=False, include_black=True, include_dark_green=True):
    black, darkgreen  = get_black(), get_mpg_green_dark()
    green, lightgreen = get_mpg_green(), get_mpg_green_light()
    if reverse==False:
        if include_black and include_dark_green:
            colors = [black, darkgreen, green, lightgreen]
        elif include_black and not include_dark_green:
            colors = [black, green, lightgreen]
        elif not include_black and not include_dark_green:
            colors = [green, lightgreen]
        else:
            colors = [darkgreen, green, lightgreen]
    else:
        if include_black==True:
            colors = [lightgreen, green, darkgreen, black]
        else:
            colors = [lightgreen, green, darkgreen]
    mpgcmp = LinearSegmentedColormap.from_list('mpgcmap', colors, 512)
    return mpgcmp

def get_mpg_cmp2(reverse=False, include_black=True):
    black, darkblue  = get_black(), get_mpg_blue_dark()
    darkgrey, grey = get_mpg_grey_dark(), get_mpg_grey()
    lightblue, lightgreen = get_mpg_blue_light(), get_mpg_green_light()

    if reverse==False:
        if include_black==True:
            colors = [black, darkblue, darkgrey, grey, lightblue, lightgreen]
        else:
            colors = [darkblue, darkgrey, grey, lightblue, lightgreen]
    else:
        if include_black==True:
            colors = [lightgreen, lightblue, grey, darkgrey, darkblue, black]
        else:
            colors = [lightgreen, lightblue, grey, darkgrey, darkblue]
    mpgcmp = LinearSegmentedColormap.from_list('mpgcmap', colors, 512)
    return mpgcmp






def set_default_params(font_size=12):

    rcdict =    {
                    'figure.figsize':       (5.2, 3.9),
                    'font.size':            font_size	,
                    'lines.linewidth':      1,
                    'lines.markersize':     5,
                    'xtick.direction':      'in',
                    'xtick.major.size':     3.5,
                    'xtick.minor.size':     2.,
                    'xtick.minor.visible':  True,
                    'xtick.top':            True,

                    'ytick.direction':      'in',
                    'ytick.major.size':     3.5,
                    'ytick.minor.size':     2.,
                    'ytick.minor.visible':  True,
                    'ytick.right':          True,

                    'image.cmap':           'coolwarm',
                }

    plt.rcParams.update(rcdict)
    return None