import numpy as np
from julia import Main
Main.include("cgc_rb.jl")
import matplotlib.pyplot as plt
import os


# Enable LaTeX font rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font family
    "font.serif": "Computer Modern Roman",  # Use LaTeX's standard font
    "text.latex.preamble": r"\usepackage{amsmath}"  # Load additional packages if needed
})

"""
    PLOT with m = n, state is |111...1>. Plot all allowed k
"""
fig = plt.figure()    
# Creating axes instance
# ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # ax.set_title("SU(2) average overlaps")
plt.xlabel("irrep", fontsize=14)
plt.ylabel("overlap", fontsize=14)
plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
                                    r'$\lambda_5$', r'$\lambda_6$'], fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim(bottom=-0.01, top=0.8)
# plt.yscale("log")
plt.grid()
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']
c = 0
for m in range(2, 7):
    state = [1]*m
    num_particles = np.array(state).sum()
    N = Main.FockState(state)
    print(state)
    overlaps = []
    for k in range(0, num_particles+1):    
        first_moment = Main.moment1(k, N)
        overlaps.append(first_moment)    
        # print(f"k = {k}\t{first_moment}")
    plt.plot(range(num_particles+1), overlaps, color=colors[c], marker='o', label=f"n = m={m}")
    c += 1
    
save_path = "./plots/"
file_name = "overlap_ones"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.legend()
plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
plt.show()


"""
    PLOT with m fixed and n changing up to m. Plot all allowed k
"""
fig = plt.figure()    
# Creating axes instance
# ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # ax.set_title("SU(2) average overlaps")
plt.xlabel("irrep", fontsize=14)
plt.ylabel("overlap", fontsize=14)
plt.ylim(bottom=-0.01, top=0.85)
plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
                                    r'$\lambda_5$', r'$\lambda_6$'], fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid()
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
c = 0
m = 6
for n in range(1, m+1):
    state = [1]*n + [0]*(m-n)
    # state = [2] + [1]*(m-2) + [0]
    num_particles = np.array(state).sum()
    N = Main.FockState(state)
    print(state)
    overlaps = []
    print(num_particles)
    for k in range(0, num_particles+1):    
        first_moment = Main.moment1(k, N)
        overlaps.append(first_moment)
        # print(f"k = {k}")
        # print(f"k = {k}\t{first_moment}")
    plt.plot(range(num_particles+1), overlaps, color=colors[c], marker='o', label=f"n = {num_particles}")
    c += 1
    
save_path = "./plots/"
file_name = "overlap_m_fixed"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
# plt.legend()
plt.legend(bbox_to_anchor=(0, 0.8), loc='upper left')
plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
plt.show()


"""
    PLOT with n fixed and m = 3,...,6. Plot all allowed k
"""
# fig = plt.figure()    
# # Creating axes instance
# # ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # # ax.set_title("SU(2) average overlaps")
# plt.xlabel("Irrep")
# plt.ylabel("overlap")
# plt.ylim(bottom=-0.01, top=0.8)
# plt.grid()
# plt.xticks([0, 1, 2, 3, 4, 5], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_5$'])
# # colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
# # colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00']
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00', '#9467BD', '#8C564B']
# c = 0
# for n in range(3, 6):
#     # n = 3
#     for m in range(n, 7):
#         state = [1]*n + [0]*(m-n)
#         # state = [2] + [1]*(m-2) + [0]
#         num_particles = np.array(state).sum()
#         N = Main.FockState(state)
#         print(state)
#         overlaps = []
#         for k in range(0, num_particles+1):    
#             first_moment = Main.moment1(k, N)
#             overlaps.append(first_moment)
#             # print(f"k = {k}")
#             # print(f"k = {k}\t{first_moment}")
#         plt.plot(range(num_particles+1), overlaps, color=colors[c], marker='o', label=f"n = {n}, m = {m}")
#         c += 1
    
# save_path = "./plots/"
# file_name = f"overlap_n_fixed_many"
# exists = os.path.exists(save_path)
# if not exists:
#     os.makedirs(save_path)
# plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.show()


"""
    PLOT m=n max overlaps. k = n too, as it is the one maximizing the overlap.
"""
# fig = plt.figure()    
# # Creating axes instance
# # ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # # ax.set_title("SU(2) average overlaps")
# plt.xlabel("number of modes")
# plt.ylabel("max overlaps")
# plt.ylim(bottom=0.65, top=0.75)
# # plt.yscale("log")
# plt.grid()
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
# c = 0
# max_overlaps = []
# for m in range(2, 7):
#     state = [1]*m
#     num_particles = np.array(state).sum()
#     N = Main.FockState(state)
#     print(state)
    
#     k = m
#     max_overlap = Main.moment1(k, N)
#     max_overlaps.append(max_overlap)
# plt.plot(range(2, 7), max_overlaps, color=colors[c], marker='o', label=f"n = m={m}")
# c += 1
    
# save_path = "./plots/"
# file_name = "max_overlaps_n=m"
# exists = os.path.exists(save_path)
# if not exists:
#     os.makedirs(save_path)
# # plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.show()

    