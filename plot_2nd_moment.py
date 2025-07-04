import numpy as np
from julia import Main
Main.include("cgc_rb.jl")
import matplotlib.pyplot as plt
import os


"""
    PLOT with m = n, state is |111...1>. Plot all allowed k
"""
fig = plt.figure()    
# Creating axes instance
# ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # ax.set_title("SU(2) average moments")
plt.xlabel("irrep")
plt.ylabel('second moment (ideal)')
# plt.ylim(bottom=-0.01, top=0.85)
# plt.yscale("log")
plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
                                    r'$\lambda_5$', r'$\lambda_6$'])
plt.grid()
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']
c = 0
for m in range(2, 4):
    state = [1]*m
    num_particles = np.array(state).sum()
    N = Main.FockState(state)
    print(state)
    moments = []
    for k in range(0, num_particles+1):    
        second_moment = Main.moment2(k, N)
        moments.append(second_moment)    
        print(f"k = {k}\t{second_moment}")
    plt.plot(range(num_particles+1), moments, color=colors[c], marker='o', label=f"n = m={m}")
    c += 1
    
save_path = "./plots/"
file_name = "second_moment_ones"
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
# # ax.set_title("SU(2) average moments")
plt.xlabel("Irrep")
plt.ylabel("second moment (ideal")
plt.xticks([0, 1, 2, 3, 4, 5, 6], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
                                    r'$\lambda_5$', r'$\lambda_6$'])
plt.ylim(bottom=-0.01, top=2)
plt.grid()
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
c = 0
m = 3
for n in range(1, m+1):
    state = [1]*n + [0]*(m-n)
    num_particles = np.array(state).sum()
    N = Main.FockState(state)
    print(state)
    moments = []
    # print(num_particles)
    for k in range(0, num_particles+1):
        if k >= 4:
            break
        second_moment = Main.moment2(k, N)
        moments.append(second_moment)
        # print(f"k = {k}")
        print(f"k = {k}\t{second_moment}")
    plt.plot(range(num_particles+1), moments, color=colors[c], marker='o', label=f"n = {num_particles}")
    c += 1
    
save_path = "./plots/"
file_name = "second_moment_m_fixed"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.legend()
plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
plt.show()


"""
    PLOT with n fixed and m = 3,...,6. Plot all allowed k
"""
fig = plt.figure()    
# Creating axes instance
# ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # ax.set_title("SU(2) average overlaps")
plt.xlabel("Irrep")
plt.ylabel("second moment (ideal)")
# plt.ylim(bottom=-0.01, top=0.8)
plt.grid()
plt.xticks([0, 1, 2, 3, 4, 5], [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_5$'])
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
# colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00']
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#4C72B0', '#E79F00', '#9467BD', '#8C564B']
c = 0
for n in range(2, 4):
    # n = 3
    for m in range(n, 5):
        state = [1]*n + [0]*(m-n)
        # state = [2] + [1]*(m-2) + [0]
        num_particles = np.array(state).sum()
        N = Main.FockState(state)
        print(state)
        second_moments = []
        for k in range(0, num_particles+1):    
            second_moment = Main.moment2(k, N)
            second_moments.append(second_moment)
            # print(f"k = {k}")
            # print(f"k = {k}\t{first_moment}")
        plt.plot(range(num_particles+1), second_moments, color=colors[c], marker='o', label=f"n = {n}, m = {m}")
        c += 1
    
save_path = "./plots/"
file_name = f"second moment_n_fixed_many"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)
plt.legend()
# plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF
# plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
plt.show()

    