import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import glob

def exp(x, a, b):
    return a * np.exp(-b*x)



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
plt.ylabel(r"$\hat F_{\lambda_{4}}$", fontsize = 14)
plt.yscale("log")
plt.ylim([0.01,1])

start_path = "./data_filtered/"
folders = ['t0.99/', 't0.975/', 't0.95/', 'tm0.9_tM1/']
transmittivities = [0.99, 0.975, 0.95, 10]
file_pattern = f"m{m}_n{n}_k{k}_L*.npz"

for folder in folders:
    files = glob.glob(os.path.join(start_path, folder, file_pattern))
    
    for file in files:
        print(file)
        data = np.load(file)

    keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    filtered_data = []
    empirical_vars = []
    std_devs = []
    for r in keys:
        arr = data[r][:L]
        # print(r, len(arr))
        estimator = np.average(arr)
        var = np.var(arr, ddof=1) / L
        # print(r, estimator, np.sqrt(var))
        filtered_data.append(estimator)
        empirical_vars.append(var)
        std_devs.append(np.sqrt(var))

    plt.errorbar(sequence_lengths, filtered_data, yerr = std_devs,
                  fmt='o', color=colors[c], markersize=3, elinewidth=1, capsize=6)
    # plt.plot(sequence_lengths, filtered_data, color=colors[c], marker='o', label=f"n = m={m}")
    

    popt, pcov = curve_fit(exp, sequence_lengths, filtered_data)
    print(f"parameters = {popt}")
    t_estimated = np.e**(-popt[1]/(2*n))
    print(t_estimated)
    sequence_lengths = np.array(sequence_lengths)
    y = exp(sequence_lengths, *popt)
    t = transmittivities[c]
    if t == 10:
        plt.plot(sequence_lengths[:10], y[:10], color=colors[c], alpha=0.65, 
                  label = f"${round(popt[0], 2)} \cdot {round(t_estimated, 3)}^{{l}}  \quad (\sqrt{{p}} \in [0.9,1])$")
    else:
        plt.plot(sequence_lengths[:10], y[:10], color=colors[c], alpha=0.65, 
                  label = f"${round(popt[0], 2)} \cdot {round(t_estimated, 3)}^{{l}} \quad (\sqrt{{p}}={t})$")
    # print(popt[1]**(1/(2*n)))
    
    c += 1


    print("\n")
plt.legend()
save_path = "./plots/decay"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.savefig(save_path+".pdf", format="pdf", bbox_inches="tight")
plt.savefig(save_path+".png", bbox_inches="tight", dpi=600)

plt.show()


"""
    DECAY PLOT FIXED TRANSMITTIVITY, ALL ALLOWED IRREPS
"""
m = 4
n = 4
k = 4
sequence_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
L = 10000
t = 0.95

# colors = ['#0072B2', '#D55E00', '#CC79A7', '#E79F00', '#9467BD', '#8C564B', '#009E73', '#F0E442']
c = 0
fig = plt.figure()    
ax = fig.add_axes([0.17,0.17,0.75,0.75])
default_x_ticks = range(0, sequence_lengths[-1]+1)
plt.xlabel("sequence length $l$", fontsize = 14)
plt.xticks(sequence_lengths, fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylabel(r"$\hat F_{\lambda_{k}}$", fontsize = 14)
plt.yscale("log")
# plt.ylim([0.01,1])

start_path = "./data_filtered/"
folder = f't{t}/'
transmittivities = [0.99, 0.975, 0.95, 10]

for k in range(0, n+1): 
    if k == 1:
        continue
    file_pattern = f"m{m}_n{n}_k{k}_L*.npz"
    files = glob.glob(os.path.join(start_path, folder, file_pattern)) 
    # print(os.path.join(start_path, folder, file_pattern))
    for file in files:
        print(file)
        data = np.load(file)
    keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    filtered_data = []
    empirical_vars = []
    std_devs = []
    for r in keys:
        arr = data[r][:L]
        # print(r, len(arr))
        estimator = np.average(arr)
        var = np.var(arr, ddof=1) / L
        # print(r, estimator, np.sqrt(var))
        filtered_data.append(estimator)
        empirical_vars.append(var)
        std_devs.append(np.sqrt(var))

    plt.errorbar(sequence_lengths, filtered_data, yerr = std_devs,
                  fmt='o', color=colors[c], markersize=3, elinewidth=1, capsize=6)
    # # plt.plot(sequence_lengths, filtered_data, color=colors[c], marker='o', label=f"n = m={m}")
    

    popt, pcov = curve_fit(exp, sequence_lengths, filtered_data)
    print(f"parameters = {popt}")
    # print(f"expected noise = {0.95**(2*n)}")
    # print(-np.log(0.95**(2*n)))
    t_estimated = np.e**(-popt[1]/(2*n))
    print(t_estimated)
    sequence_lengths = np.array(sequence_lengths)
    y = exp(sequence_lengths, *popt)
    
    plt.plot(sequence_lengths[:10], y[:10], color=colors[c], alpha=0.65, 
              label = f"${round(popt[0], 2)} \cdot {round(t_estimated, 3)}^{{l}} \quad (k={k})$")
    
    c += 1


#     print("\n")
plt.legend()
save_path = f"./plots/decay_t{t}"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.savefig(save_path+".pdf", format="pdf", bbox_inches="tight")
plt.savefig(save_path+".png", bbox_inches="tight", dpi=600)

plt.show()


























