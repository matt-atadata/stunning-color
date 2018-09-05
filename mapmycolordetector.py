import io
import sys
import csv
import math
import numpy as np
from scipy import spatial
from urllib.request import urlopen

from colorme import ColorMe
from colortranslate import *


def _load_settings(file_):
    try:
        with open(file_, 'r') as f:
            return json.load(f)
    except Exception as e:
        stderr.write('Failed to load settings file: `{}`\n'.format(e))
        exit(1)   


def polar_dist(combined_hsi_arrays, input_hsi):
    """
        K-NN for polar distance calculation of 
        Hue as theta, 
        Saturation as radius, 
        and 
        Intensity as vertical distance.
        Input: 
            idx [h,s,i]  (color index and hsi value of index)
    """
    vv = []
    for idx, [h,s,i] in enumerate(combined_hsi_arrays):
        hh = ( min(abs(h-input_hsi[0]), 360-abs(h-input_hsi[0])) )/180.
        ss = ( abs(s-input_hsi[1]) )/100.
        ii = ( abs(i-input_hsi[2]) )/100.
        vv.append([np.linalg.norm(np.array([hh,ss,ii])), idx])
    vvs = sorted(vv, key=lambda x: x[0])

    # Return polar distince to all points of named colors from input_hsi
    return vvs

def main():
    url_to_open = sys.argv[1]
    print("Opening: ", url_to_open)
    print()
    fd = urlopen(url_to_open)
    f = io.BytesIO(fd.read())


    color_me = ColorMe(f)

    colors = color_me.get_color(quality=3)
    print()
    print("Dominant color successfully extracted")
    input_hsis = []
    input_rgbs = []

    for color in colors:
        r, g, b = color
        input_rgbs.append([r,g,b])
        h,s,i = rgb2hsi(r,g,b)
        s = s*100
        i = i*100
        input_hsis.append(np.asarray([h,s,i]))
        
    # Dominant colors in HSI and RGB
    input_hsi = input_hsis[0]
    input_rgb = input_rgbs[0]
    print()
    print('Found dominant color in hsi:', input_hsi)
    print('Found dominant color in rgb:', input_rgb)
    print()

    with open('./color_palette_hsi.tsv','r') as tsvin:
        cc = csv.reader(tsvin, delimiter='\t')

        H = []
        S = []
        I = []
        hex_colors = []
        colors = []

        for row in cc:
            colors.append(row[0])

            H.append( float(row[1]))		
            S.append( float(row[2]))
            I.append( float(row[3]))

            hex_colors.append(row[4])

    combined_hsi_arrays = np.stack([H,S,I]).transpose()

    # Color Naming
    vv = polar_dist(combined_hsi_arrays, input_hsi)
    print()
    print('Nearest Neighbor Color Rankings by polor distance')
    print('-------------------------------------------------')
    for d,i in vv:
        print('dist, color', d, colors[i])

    print('-------------------------------------------------')
    print()    
    print('Found a dominant color!', colors[vv[0][1]])

if __name__ == "__main__":
    main()

