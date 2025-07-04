import numpy as np
import passive_rb as prb
import os
from datetime import datetime





# state = [1, 1, 1, 1]
# num_particles = np.array(state).sum()
# m = len(state)
# L = 10000 #number of samples
# t_lbound = 1
# t_ubound = 1
# gate_dependent = False
# sequence_lengths = [1]


# t_start = datetime.now()
# for r in sequence_lengths:
    
#     t_in_start = datetime.now()
#     hyperparameters = np.array([num_particles, m, r, L, t_lbound])
#     sampled_matrices, sampled_states = prb.RB_sampled_data(state, r, L, t_lbound, t_ubound, gate_dependent)
#     t_in_end = datetime.now()
#     print(f"r={r} time = {(t_in_end-t_in_start).total_seconds()}")
    
#     if gate_dependent == False:
#         save_path_hyper = f"./SU({m})/raw_data/n{num_particles}/t{t_lbound}/r{r}/"
#         file_name_hyper = "hyperparameters"
#         exists = os.path.exists(save_path_hyper)
#         if not exists:
#             os.makedirs(save_path_hyper)
#         np.save(save_path_hyper+file_name_hyper+".npy",hyperparameters)
        
#         save_path = f"./SU({m})/raw_data/n{num_particles}/t{t_lbound}/r{r}/batches/"
#         file_name_states = f"states_L-{L}"
#         file_name_matrices = f"matrices_L-{L}"
#         exists = os.path.exists(save_path)
#         if not exists:
#             os.makedirs(save_path)
#     else:
#         save_path_hyper = f"./SU({m})/raw_data/n{num_particles}/tm{t_lbound}_tM{t_ubound}/r{r}/"
#         file_name_hyper = "hyperparameters"
#         exists = os.path.exists(save_path_hyper)
#         if not exists:
#             os.makedirs(save_path_hyper)
#         np.save(save_path_hyper+file_name_hyper+".npy",hyperparameters)
        
#         save_path = f"./SU({m})/raw_data/n{num_particles}/tm{t_lbound}_tM{t_ubound}/r{r}/batches/"
#         file_name_states = f"states_L-{L}"
#         file_name_matrices = f"matrices_L-{L}"
#         exists = os.path.exists(save_path)
#         if not exists:
#             os.makedirs(save_path)
    
#     np.savez(save_path+file_name_states+".npz",sampled_states)
#     np.savez(save_path+file_name_matrices+".npz",sampled_matrices)

    
# t_end = datetime.now()

# time = (t_end-t_start).total_seconds()
# print(f"Time collecting L={L} samples for sequence lengths up to 10 = {time}") 


"""
    FOR IDEAL DATA
"""   
state = [1, 1, 1, 1, 1, 1, 1]
num_particles = np.array(state).sum()
m = len(state)
L = 10 #number of samples
r = 1
t_lbound = 1
t_ubound = 1
gate_dependent = False
sequence_lengths = [1]


t_start = datetime.now()


t_in_start = datetime.now()
hyperparameters = np.array([num_particles, m, r, L, t_lbound])
sampled_matrices, sampled_states = prb.RB_sampled_data(state, r, L, t_lbound, t_ubound, gate_dependent)
t_in_end = datetime.now()
print(f"r={r} time = {(t_in_end-t_in_start).total_seconds()}")

save_path = "./ideal/"
file_name_states = f"states_m{m}_n{num_particles}_L{L}"
file_name_matrices = f"matrices_m{m}_n{num_particles}_L{L}"
exists = os.path.exists(save_path)
if not exists:
    os.makedirs(save_path)

# np.savez(save_path+file_name_states+".npz",sampled_states)
# np.savez(save_path+file_name_matrices+".npz",sampled_matrices)
    

    
t_end = datetime.now()

time = (t_end-t_start).total_seconds()
print(f"Time collecting L={L} samples for sequence lengths up to 10 = {time}")  


"""
    COMPUTE FILTER FUNCTION FOR IDEAL DATA
"""

# import numpy as np
# from julia import Main
# Main.include("tools.jl")

# def call_FockState(lst):
#     # Convert Python list to Julia array
#     julia_lst = Main.eval('x -> x')([int(i) for i in lst])
#     # Call the Julia function
#     return Main.FockState(julia_lst)

# m = 6
# n = 6
# L = 10000
# batch = 10
# for k in range(n, n+1):
#     input_state = [1]*n + [0]*(m-n)
#     N = Main.FockState(input_state)
#     states_file = f"./ideal/states_m{m}_n{n}_L{L}.npz"
#     matrices_file = f"./ideal/matrices_m{m}_n{n}_L{L}.npz"
#     states = [*np.load(states_file)['arr_0']][:batch]
#     matrices = [*np.load(matrices_file)['arr_0']][:batch]
    
#     overlap = Main.moment1(k, N)
#     filtered_data = []
#     for i in range(batch):
#         x = states[i].tolist()
#         X = call_FockState(x)
#         filter_function = Main.filter_function(k, N, X, matrices[i])
#         filtered_data.append(filter_function)
#     estimator = np.array(filtered_data).sum()/batch
    
#     name = f"./ideal/filtered_data_{m}m_n{n}_k{k}.npz"
#     np.savez(name, filtered_data)
  
