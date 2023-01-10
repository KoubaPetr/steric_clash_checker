from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple

def compute_histogram_args(bins_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param bins_array: borders of the bins
    :return: centers of bins, widths of bins
    """
    intervals = np.hstack((bins_array[:-1,None], bins_array[1:,None]))
    bin_widths = intervals[:,1] - intervals[:,0]
    bin_centers = intervals.mean(axis=1)
    return bin_centers, bin_widths

def plot_hists(atom_dists: np.ndarray, subtitle: str = ''):
    """

    Wrapper for the histogram plotting

    :param atom_dists: distances to compute and evaluate the histograms for
    :param subtitle: how to distinguish the data in the plot title
    :return:
    """
    #Compute hists
    freq, bins = np.histogram(atom_dists, bins=20, range=(0,5))
    #Plot hists
    centers, widths = compute_histogram_args(bins)
    fig,ax = plt.subplots()
    bar = ax.bar(x=centers, height=freq, width=widths)
    ax.set_title(f'Histogram of ligand-target distances - {subtitle}')
    ax.vlines(x=0.4, ymin=0,ymax=max(freq), color='red', linewidth=3)
    ax.vlines(x=1, ymin=0,ymax=max(freq), color='lightgreen', linewidth=3, linestyles='dashed')
    ax.vlines(x=1.2, ymin=0,ymax=max(freq), color='lightgreen', linewidth=3)
    plt.show()

#Read data
heavy_atom_dists = np.load('../logs/heavy_dists_test_pdbbind_treshold_1.npy')
hydrogen_atom_dists = np.load('../logs/hydrogen_dists_test_pdbbind_treshold_1.npy')
overall_mindists = np.minimum(heavy_atom_dists, hydrogen_atom_dists)

#Plot data
plot_hists(heavy_atom_dists, subtitle='Heavy atoms')
plot_hists(hydrogen_atom_dists, subtitle='Hydrogen atoms')
plot_hists(overall_mindists, subtitle='Overall')