import numpy as np
from julia import Main
Main.include("cgc_rb.jl")
from datetime import datetime
# import matplotlib.pyplot as plt
import passive_rb as prb


m = 3
state = [1, 1, 1, 1, 1, 1]
k = 0
L = 1000
sequence_lengths = [1]
gate_dependent = False
t_lbound = 1 # Set to 1 to get ideal data

num_particles = np.array(state).sum()
N = Main.FockState(state)
print(state)
for k in range(0, num_particles+1):    
    first_moment = Main.moment1(k, N)
    print(f"k = {k}\tfirst moment = {first_moment}")



# print(f"input state = {state}, k = {k}, number of samples = {L}, t = {t_lbound}")
# estimators = []
# for r in sequence_lengths:
#     start_time = datetime.now()
#     outcomes_RB = prb.RB_data_lossy(state, k, m, r, L, t_lbound)
#     estimator = np.array(np.real(outcomes_RB)).sum() / L
#     print(f"Estimator = {estimator}")
#     estimators.append(estimator)
#     # empirical_var = np.var(np.array(outcomes_RB), ddof = 1)
#     # print(f"empirical var = {empirical_var}")
#     # print(outcomes_RB)
#     end_time = datetime.now()
#     simulation_time = (end_time - start_time).total_seconds()
#     print(f"time = {simulation_time}, r = {r}, F = {estimator}")


# fig = plt.figure()    
# # Creating axes instance
# ax = fig.add_axes([0.12,0.12,0.75,0.75])
# # ax.set_title("SU(2) average overlaps")
# # Creating plot
# plt.xlabel("sequence length")
# # plt.ylim(bottom=0.01, top=0.3)
# plt.yscale("log")
# plt.plot(sequence_lengths, estimators, 'bo', label=f"k={k}, loss={t_lbound}, state={state}, L={L}")

# # save_path = "./su(2)/decays/"
# # file_name = f"{state}_{J}"
# # exists = os.path.exists(save_path)
# # if not exists:
# #     os.makedirs(save_path)
# # # plt.savefig(save_path+file_name+".pdf", format="pdf", bbox_inches="tight") # Save as PDF

# plt.legend()
# # plt.savefig(save_path+file_name+".png", format="png", bbox_inches="tight", dpi=600) # Save as PNG
# plt.show()


# second_moment = prb.second_moment_PNR_su2(state, J)
# variance = second_moment - first_moment**2





















