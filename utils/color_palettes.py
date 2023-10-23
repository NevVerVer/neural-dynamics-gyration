"""
Created on Sat September 16 21:15:06 2023

Description: Color palettes for:
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v1

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import matplotlib.colors as color

bg_gray_color = '#fafafa'

journ_color_dict = {'blue': '#087bb2',
                    'yellow': '#f1ac3e',
                    'red': '#d1232a',
                    'green': '#66a972'} 

alt_colors = {'blue': ['#0876ab', '#06547a'],
              'green': ['#67ab72', '#0d602c'],
              'yellow':['#f3ad3e', '#cf720b'],
              'brown':['#e0c985', '#c0822f'],
              'red':['#d52228', '#b31d22'],
              'purple':['#834592', '#624285'],
              'sea': ['#72c2b7', '#01655b'],}

# jPCA colors
# original, as in Churchland's paper
jpca_palette_orig = lambda : color.LinearSegmentedColormap.from_list("", 
                                                                    ['#9c0a0b',
                                                                    '#de0001',
                                                                    '#0bea0c',
                                                                    '#06b00d'])
#bright colors at the end, dark in the middle
jpca_palette_alternative1 = lambda : color.LinearSegmentedColormap.from_list("",
                                                                            ['#d1232a',
                                                                            '#9c0a0b',
                                                                            '#06b00d',
                                                                            '#0bea0c'])
# took red and green from journal colors
# journal colors in middle, churhc at ends
jpca_palette_journ1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                                      ['#9c0a0b',
                                                                      '#d1232a',
                                                                      '#66a972',
                                                                      '#06b00d'])

# all journ colors
GR1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                      [alt_colors['green'][1],
                                                      alt_colors['green'][0],
                                                      alt_colors['red'][0],
                                                      alt_colors['red'][1],])

RB1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                      [alt_colors['blue'][1],
                                                      alt_colors['blue'][0],
                                                      alt_colors['red'][0],
                                                      alt_colors['red'][1],])

YP1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                      [alt_colors['yellow'][1],
                                                      alt_colors['yellow'][0],
                                                      alt_colors['purple'][0],
                                                      alt_colors['purple'][1],])

YS1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                      ['#e79c3b',
                                                      '#f9f720',
                                                      alt_colors['sea'][0],
                                                      alt_colors['sea'][1],])

BG1 = lambda : color.LinearSegmentedColormap.from_list("", 
                                                      [alt_colors['blue'][1],
                                                      alt_colors['blue'][0],
                                                      alt_colors['green'][0],
                                                      alt_colors['green'][1],])

blue_yellow_cmap = lambda : color.LinearSegmentedColormap.from_list("", ['#382db0',
                                                                          '#0c88cd',
                                                                          '#e79c3b',
                                                                          '#f9f720'])

cmaps = {'lfp': YP1(),
        'church': GR1(),
        'grasp': RB1(),
        'kalid': YS1(),
        'pfc': BG1()}