import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import glob

def exp(x, a, b):
    return a * np.exp(-b*x)


def indicator(state, num_particles):
    if np.array(state).sum() == num_particles:
        return 1
    else:
        return 0
    
def indicator_filtering(states, n, L):
    num_preserved = 0
    for state in states:
        num_preserved += indicator(state, n)
    return num_preserved / L

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font family
    "font.serif": "Computer Modern Roman",  # Use LaTeX's standard font
    "text.latex.preamble": r"\usepackage{amsmath}"  # Load additional packages if needed
})

"""
    DECAY PLOT WITH FIXED IRREP, DIFFERENT VALUES OF TRANSMITTIVITY
"""
m = 4
n = 4
k = 4
sequence_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
L = 10000

# Enable LaTeX font rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font family
    "font.serif": "Computer Modern Roman",  # Use LaTeX's standard font
    "text.latex.preamble": r"\usepackage{amsmath}"  # Load additional packages if needed
})


colors = [
    '#0072B2',  # Blue
    '#66CCEE',  # Soft Cyan
    '#009E73',  # Bluish Green
    '#E69F00',  # Orange
    '#F5A0C0',  # Soft Pink
    '#CC79A7',  # Reddish Purple
    '#56B4E9',  # Sky Blue
    '#F0F032',  # Bright Yellow
    '#999999',  # Gray
    '#999933',  # Olive
    '#D55E00',  # Vermilion
    '#AA4499',  # Strong Purple
    '#882255',  # Strong Magenta
    '#E69F00',  # Orange (reused, very distinguishable)
    '#009E73',  # Bluish Green (reused, very distinguishable)
    '#F0E442',  # Yellow
]
c = 0
fig = plt.figure()    
ax = fig.add_axes([0.17,0.17,0.75,0.75])
default_x_ticks = range(0, sequence_lengths[-1]+1)
plt.xlabel("sequence length $l$", fontsize = 14)
plt.xticks(sequence_lengths, fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylabel(r"$\hat F_I$", fontsize = 14)
plt.yscale("log")
# plt.ylim([0.01,1])

start_path = "./SU(4)/raw_data/n4/"
folders = ['t0.99/', 't0.975/', 't0.95/', 'tm0.9_tM1/']
transmittivities = [0.99, 0.975, 0.95, 10]

for folder in folders:
    indicator_filters = []
    for r in sequence_lengths:    
        file_pattern = f"r{r}/states*.npz"
        files = glob.glob(os.path.join(start_path, folder, file_pattern)) 
        # print(os.path.join(start_path, folder, file_pattern))
        for file in files:
            data = np.load(file)
            states = [*data['arr_0']][:10000]
            print(indicator_filtering(states, n, L))
            indicator_filters.append(indicator_filtering(states, n, L))
    print("\n")
    plt.plot(sequence_lengths, indicator_filters, color=colors[c], 
             marker='o', linestyle='none')
    
    popt, pcov = curve_fit(exp, sequence_lengths, indicator_filters)
    print(f"parameters = {popt}")
    t_estimated = np.e**(-popt[1]/(2*n))
    print(t_estimated)
    sequence_lengths = np.array(sequence_lengths)
    y = exp(sequence_lengths, *popt)
    t = transmittivities[c]
    if t == 10:
        plt.plot(sequence_lengths[:10], y[:10], color=colors[c], alpha=0.5, 
                  label = f"${round(popt[0], 2)} \cdot {round(t_estimated, 3)}^{{l}}  \quad (\sqrt{{p}} \in [0.9,1])$")
    else:
        plt.plot(sequence_lengths[:10], y[:10], color=colors[c], alpha=0.5, 
                  label = f"${round(popt[0], 2)} \cdot {round(t_estimated, 3)}^{{l}} \quad (\sqrt{{p}}={t})$")
    
    c += 1


#     print("\n")
plt.legend()
save_path = "./plots/decay_indicator"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.savefig(save_path+".pdf", format="pdf", bbox_inches="tight")
plt.savefig(save_path+".png", bbox_inches="tight", dpi=600)

plt.show()




















