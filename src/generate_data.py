import numpy as np
import time
# from tqdm import tqdm

from tqdm import tqdm
import swendsen_wang as sw

import storage


def simulation(spins, bonds, T, N_measure=100):
    """Perform a Monte-carlo simulation at given temperature"""
    # no thermalization here
    Es = []
    Ms = []
    for n in range(N_measure):
        spins = sw.swendsen_wang_update(spins, bonds, T)
        Es.append(sw.energy(spins, bonds))
        Ms.append(sw.magnetization(spins))
    return np.array(Es), np.array(Ms)



# The full simulation at different temperatures
def gen_data_L(Ts, L, N_measure=10000, N_bins=10):
    print("generate data for L = {L: 3d}".format(L=L), flush=True)
    assert(N_measure//N_bins >= 10)
    spins, bonds, N = sw.init_system(L, L, L)
    obs = ['E', 'C', 'M', 'absM', 'chi', 'UB']
    data = dict((key, []) for key in obs)
    t0 = time.time()
    # for T in tqdm(Ts):
    for T in tqdm(Ts):
        if N_measure > 1000:
            print("simulating L={L: 3d}, T={T:.3f}".format(L=L, T=T), flush=True)
        # thermalize. Rule of thumb: spent ~10-20% of the simulation time without measurement
        simulation(spins, bonds, T, N_measure//3)
        # Simulate with measurements
        bins = dict((key, []) for key in obs)
        for b in range(N_bins):
            E, M = simulation(spins, bonds, T, N_measure//N_bins)
            bins['E'].append(np.mean(E)/N)
            bins['C'].append(np.var(E)/(T**2*N))
            bins['M'].append(np.mean(M)/N)
            bins['absM'].append(np.mean(np.abs(M))/N)
            bins['chi'].append(np.var(np.abs(M))/(T*N))
            bins['UB'].append(1.5*(1.-np.mean((M/N)**4)/(3.*np.mean((M/N)**2)**2)))
        for key in obs:
            bin = bins[key]
            data[key].append((np.mean(bin), np.std(bin)/np.sqrt(N_bins)))
    print("generating data for L ={L: 3d} took {t: 6.1f}s".format(L=L, t=time.time()-t0))
    # convert into arrays
    for key in obs:
        data[key] = np.array(data[key])
    # good practice: save meta-data along with the original data
    data['L'] = L
    data['observables'] = obs
    data['Ts'] = Ts
    data['N_measure'] = N_measure
    data['N_bins'] = N_bins
    return data

