import numpy as np
from julia import Main
Main.include("tools.jl")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def poly(x, A, B, C):
    return A * x**B + C

def log(x, A, B):
    return A * np.log(x) + B

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font family
    "font.serif": "Computer Modern Roman",  # Use LaTeX's standard font
    "text.latex.preamble": r"\usepackage{amsfonts}"  # Load additional packages if needed
})

"""
    PLOT with state |1...10...0>. Plot all allowed k
"""
# fig = plt.figure()
plt.xlabel("irrep", fontsize = 14)
# plt.ylabel(r'$\mathbb{E}[f_{\lambda}^2]_{\mathrm{ideal}}$', fontsize=14)
plt.ylabel("variance bound", fontsize=14)

plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
                                    r'$\lambda_5$', r'$\lambda_6$'], fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid()
colors = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermilion
    '#009E73',  # Bluish Green
    '#F5A0C0',  # Soft Pink
    '#CC79A7',  # Reddish Purple
    '#56B4E9',  # Sky Blue
    '#E69F00',  # Orange
    '#F0F032',  # Bright Yellow
    '#999999',  # Gray
    '#999933',  # Olive
    '#66CCEE',  # Soft Cyan
    '#AA4499',  # Strong Purple
    '#882255',  # Strong Magenta
    '#E69F00',  # Orange (reused, very distinguishable)
    '#009E73',  # Bluish Green (reused, very distinguishable)
    '#F0E442',  # Yellow
]

# Print n first, then m
c = 0
points_for_fit = []
for n in range(2, 7):
    for m in range(2, 6):
        if n > m:
            continue
        state = [1]*n + [0]*(m-n)
        N = Main.FockState(state)
        print(state)
        second_moments = []
        for k in range(0, n+1):
            file = f"./ideal/filtered_data_m{m}_n{n}_k{k}.npz"
            filtered_data = [*np.load(file)['arr_0']]
            estimator = np.average(filtered_data)
            second_moment = np.average(np.square(filtered_data))
            second_moments.append(np.real(second_moment))
            print(estimator, Main.moment1(k, N), second_moment)
            if n == m == k:
                points_for_fit.append(np.real(second_moment))
        plt.plot(range(n+1), second_moments, color=colors[c], marker='o', label=f"n = {n}, m={m}")
        # plt.plot(range(n+1), second_moments, color=colors[c], marker='o')

        c += 1

# Print m first, then n
# c = 0
# points_for_fit = []
# for m in range(2, 6):
#     for n in range(2, m+1):
#         state = [1]*n + [0]*(m-n)
#         N = Main.FockState(state)
#         print(state)
#         second_moments = []
#         for k in range(0, n+1):
#             file = f"./ideal/filtered_data_m{m}_n{n}_k{k}.npz"
#             filtered_data = [*np.load(file)['arr_0']]
#             estimator = np.average(filtered_data)
#             second_moment = np.average(np.square(filtered_data))
#             second_moments.append(np.real(second_moment))
#             print(estimator, Main.moment1(k, N), second_moment)
#             if n == m == k:
#                 points_for_fit.append(np.real(second_moment))
#         plt.plot(range(n+1), second_moments, color=colors[c], marker='o', label=f"m={m}, n = {n}")
#         # plt.plot(range(n+1), second_moments, color=colors[c], marker='o')

#         c += 1
# plt.legend()
# plt.show()


print("\n", points_for_fit)
popt, pcov = curve_fit(log, np.array(range(2, 6)), points_for_fit)
print(popt)
y = log(np.array(range(2, 6)), *popt)
plt.plot(np.array(range(2, 6)), y, color='black', 
         alpha=1, label=f"${round(popt[0], 1)} \cdot \log(m) + {round(popt[1], 1)}$")


plt.legend()

save_path = "./plots/"
file_name = "poster_second_moment"
# exists = os.path.exists(save_path)
# if not exists:
#     os.makedirs(save_path)
# plt.legend()
plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
plt.show()


"""
    Comparison with exact values, case n=m up to 3
"""
# colors = [
#     '#0072B2',  # Blue
#     '#F5A0C0',  # Soft Pink
#     '#009E73',  # Bluish Green
#     '#AA4499',  # Strong Purple
#     '#CC79A7',  # Reddish Purple
#     '#56B4E9',  # Sky Blue
#     '#E69F00',  # Orange
#     '#F0F032',  # Bright Yellow
#     '#999999',  # Gray
#     '#999933',  # Olive
#     '#66CCEE',  # Soft Cyan
#     '#D55E00',  # Vermilion
#     '#882255',  # Strong Magenta
#     '#E69F00',  # Orange (reused, very distinguishable)
#     '#009E73',  # Bluish Green (reused, very distinguishable)
#     '#F0E442',  # Yellow
# ]
# fig = plt.figure()
# plt.xlabel("irrep", fontsize = 14)
# plt.ylabel(r'$\mathbb{E}[f_{\lambda}^2]_{\mathrm{ideal}}$', fontsize=14)
# plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
#                                     r'$\lambda_5$', r'$\lambda_6$'], fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.grid()
# c = 0
# for m in range(2, 4):
#     exact_values = []
#     empirical_values = []
#     state = [1]*m
#     N = Main.FockState(state)
#     print(state)
#     for k in range(0, m+1):
#         exact = Main.moment2(k, N)
#         exact_values.append(exact)
#         file = f"./ideal/filtered_data_m{m}_n{m}_k{k}.npz"
#         filtered_data = [*np.load(file)['arr_0']]
#         empirical = np.real(np.average(np.square(filtered_data)))
#         empirical_values.append(empirical)
#         print(exact, empirical)
#     plt.plot(range(0, m+1), exact_values, color=colors[c], marker='o', label=f'n = m = {m} (exact)')
#     plt.plot(range(0, m+1), empirical_values, color=colors[c+1], marker='o', label=f'n = m = {m} (empirical)')
#     c += 2

# plt.legend()
# save_path = "./plots/"
# file_name = "second_moment_empirical_vs_exact"
# # # exists = os.path.exists(save_path)
# # # if not exists:
# # #     os.makedirs(save_path)
# # # plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", bbox_inches="tight", dpi=600)

# plt.show()




"""
    UB comparison
"""
# bounds = []
# moments = []
# points_for_fit = []
# for m in range(2, 6):
#     state = [1]*m
#     N = Main.FockState(state)
#     bound = Main.inv_sλ(m, m)**2
#     bounds.append(bound)
    
#     file = f"./ideal/filtered_data_m{m}_n{m}_k{m}.npz"
#     filtered_data = [*np.load(file)['arr_0']]
#     moment = np.average(np.square(filtered_data))
#     moments.append(np.real(moment))
# print(moments)
# for i in range(1, len(moments)):
#     print(moments[i]-moments[i-1])
# for i in range(1, len(moments)):
#     print(moments[i]/moments[i-1])
    
# plt.xlabel("number of modes m", fontsize = 14)
# # plt.ylabel(r'$\mathbb{E}[f_{\lambda_k}^2]_{\mathrm{ideal}}$ (empirical)', fontsize=14)
# plt.xticks([0, 1, 2, 3, 4, 5, 6])    
# plt.yscale('log')
# plt.plot(range(2, 6), bounds, color=colors[1], marker='o', label=r'$s_{\lambda_k}^{-2}$')
# plt.plot(range(2, 6), moments, color=colors[0], marker='o', label=r'$\mathbb{E}[f_{\lambda_k}^2]_{\mathrm{ideal}}$ (empirical)')    

# save_path = "./plots/"
# file_name = "second_moment_bounds"
# # # exists = os.path.exists(save_path)
# # # if not exists:
# # #     os.makedirs(save_path)
# # # plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.legend()
# plt.show()










"""
    PLOT with m fixed and n changing up to m. Plot all allowed k
"""
# fig = plt.figure()    
# # Creating axes instance
# # ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # # ax.set_title("SU(2) average moments")
# plt.xlabel("Irrep")
# plt.ylabel("second moment (ideal")
# plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
#                                     r'$\lambda_5$', r'$\lambda_6$'])
# plt.ylim(bottom=-0.01, top=2)
# plt.grid()
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
# c = 0
# m = 3
# for n in range(1, m+1):
#     state = [1]*n + [0]*(m-n)
#     num_particles = np.array(state).sum()
#     N = Main.FockState(state)
#     print(state)
#     moments = []
#     # print(num_particles)
#     for k in range(0, num_particles+1):
#         if k >= 4:
#             break
#         second_moment = Main.moment2(k, N)
#         moments.append(second_moment)
#         # print(f"k = {k}")
#         print(f"k = {k}\t{second_moment}")
#     plt.plot(range(num_particles+1), moments, color=colors[c], marker='o', label=f"n = {num_particles}")
#     c += 1
    
# save_path = "./plots/"
# file_name = "second_moment_m_fixed"
# exists = os.path.exists(save_path)
# if not exists:
#     os.makedirs(save_path)
# plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.show()


"""
    PLOT with n fixed and m = 3,...,6. Plot all allowed k
"""
# fig = plt.figure()    
# # Creating axes instance
# # ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # # ax.set_title("SU(2) average overlaps")
# plt.xlabel("Irrep")
# plt.ylabel("second moment (ideal)")
# # plt.ylim(bottom=-0.01, top=0.8)
# plt.grid()
# plt.xticks([0, 1, 2, 3, 4, 5], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_5$'])
# # colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
# # colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00']
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00', '#9467BD', '#8C564B']
# c = 0
# for n in range(2, 4):
#     # n = 3
#     for m in range(n, 5):
#         state = [1]*n + [0]*(m-n)
#         # state = [2] + [1]*(m-2) + [0]
#         num_particles = np.array(state).sum()
#         N = Main.FockState(state)
#         print(state)
#         second_moments = []
#         for k in range(0, num_particles+1):    
#             second_moment = Main.moment2(k, N)
#             second_moments.append(second_moment)
#             # print(f"k = {k}")
#             # print(f"k = {k}\t{first_moment}")
#         plt.plot(range(num_particles+1), second_moments, color=colors[c], marker='o', label=f"n = {n}, m = {m}")
#         c += 1
    
# save_path = "./plots/"
# file_name = f"second moment_n_fixed_many"
# exists = os.path.exists(save_path)
# if not exists:
#     os.makedirs(save_path)
# plt.legend()
# # plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# # plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.show()

    