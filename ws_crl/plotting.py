# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Plotting functions """

import matplotlib
import seaborn as sns

# Global constants
FONTSIZE = 11  # pt
PAGEWIDTH = 11  # inches


def init_plt():
    """Initialize matplotlib's rcparams to look good"""

    sns.set_style("whitegrid")

    matplotlib.rcParams.update(
        {
            # Font sizes
            "font.size": FONTSIZE,  # controls default text sizes
            "axes.titlesize": FONTSIZE,  # fontsize of the axes title
            "axes.labelsize": FONTSIZE,  # fontsize of the x and y labels
            "xtick.labelsize": FONTSIZE,  # fontsize of the tick labels
            "ytick.labelsize": FONTSIZE,  # fontsize of the tick labels
            "legend.fontsize": FONTSIZE,  # legend fontsize
            "figure.titlesize": FONTSIZE,  # fontsize of the figure title
            # Figure size and DPI
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
            # colors
            "lines.markeredgewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "grid.color": "0.9",
            "axes.grid.which": "both",
            # x-axis ticks and grid
            "xtick.bottom": True,
            "xtick.direction": "out",
            "xtick.color": "black",
            "xtick.major.bottom": True,
            "xtick.major.size": 4,
            "xtick.minor.bottom": True,
            "xtick.minor.size": 2,
            # y-axis ticks and grid
            "ytick.left": True,
            "ytick.direction": "out",
            "ytick.color": "black",
            "ytick.major.left": True,
            "ytick.major.size": 4,
            "ytick.minor.left": True,
            "ytick.minor.size": 2,
        }
    )
