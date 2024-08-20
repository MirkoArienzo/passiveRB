import numpy as np
import piquasso as pq
from scipy.stats import unitary_group
from julia import Main
Main.include("cgc_rb.jl")

def call_FockState(lst):
    julia_lst = Main.eval('x -> x')([int(i) for i in lst])
    return Main.FockState(julia_lst)

def boson_sampler(state, unitaries, m, t_lbound=1, t_ubound=1, gate_dependent=False):
    modes = list(range(m))
    with pq.Program() as program:
        pq.Q(all) | pq.StateVector(state)
        for g in unitaries:
            pq.Q(all) | pq.Interferometer(g)
            if gate_dependent:
                transmittivity = np.random.uniform(t_lbound, t_ubound)
            else:
                transmittivity = t_lbound
            for i in modes:
                pq.Q(i) | pq.Loss(transmittivity)
        pq.Q(all) | pq.ParticleNumberMeasurement()
    
    simulator = pq.SamplingSimulator(d=m)
    result = simulator.execute(program, shots=1)
    return result

def RB_sampled_data(state, r, L, t_lbound=1, t_ubound=1, gate_dependent=False):

    m = len(state)
    sampled_matrices = []
    sampled_states = []    
    for _ in range(L):
        unitaries = []
        g_tot = np.identity(m)
        for i in range(r):
            g = unitary_group.rvs(m)
            g_tot = g @ g_tot
            unitaries.append(g)
        sampled_matrices.append(g_tot)
        x = np.array( list(boson_sampler(state, unitaries, m, t_lbound, t_ubound, gate_dependent).samples[0]) )
        sampled_states.append(x)

    return sampled_matrices, sampled_states


def RB_data(state, k, r, L, t_lbound=1, t_ubound=1, gate_dependent=False):
    
    m = len(state)
    number_particles = int(np.array(state).sum())
    frameop = Main.sλ(k, number_particles, m)  
    N = call_FockState(state)
    
    outcomes_filter_function = []
    for _ in range(L):
        unitaries = []
        g_tot = np.identity(m)
        for i in range(r):
            g = unitary_group.rvs(m)
            g_tot = g @ g_tot
            unitaries.append(g)
        x = list(boson_sampler(state, unitaries, m, t_lbound, t_ubound, gate_dependent).samples[0])
        X = call_FockState(x)
        if np.array(x).sum() != number_particles:
            continue
        filter_function = Main.filter(k, N, X, g_tot)
        assert np.abs(filter_function) <= 1/frameop, "filter function out of bounds" 
        outcomes_filter_function.append(filter_function)
    
    return outcomes_filter_function
