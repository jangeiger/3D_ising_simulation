import numpy as np
import time
# from tqdm import tqdm

import swendsen_wang as sw

import pickle  # for input/output


def simulation(spins, bonds, T, N_measure=100):
    """Perform a Monte-carlo simulation at given temperature"""
    # no thermalization here
    Es = []
    Ms = []
    for n in range(N_measure):
        sw.swendsen_wang_update(spins, bonds, T)
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
    for T in Ts:
        if N_measure > 1000:
            print("simulating L={L: 3d}, T={T:.3f}".format(L=L, T=T), flush=True)
        # thermalize. Rule of thumb: spent ~10-20% of the simulation time without measurement
        simulation(spins, bonds, T, N_measure//10)
        # Simlulate with measurements
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


def save_data(filename, data, folder="data"):
    """Save an (almost) arbitrary python object to disc."""
    with open(folder+"/"+filename, 'wb') as f:
        pickle.dump(data, f)
    # done


def load_data(filename, folder="data"):
    """Load and return data saved to disc with the function `save_data`."""
    with open(folder+"/"+filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    #Tc_guess = None
    Tc_guess = 3.64   # good guess for the 2D Ising model; uncomment this to get
    #                    # many T-points around this value for large L (-> long runtime!)
    if Tc_guess is None:
        N_measure = 1000  # just a quick guess
        Ls = [4, 8, 16, 32]
        output_filename = 'data_ising_square.pkl'
    else:
        N_measure = 50000
        Ls = [4, 8, 16, 32, 64, 128]
        output_filename = 'data_ising_square_largeL.pkl'
    data = dict()
    for L in Ls:
        if Tc_guess is None:
            # no guess for Tc available -> scan a wide range to get a first guess
            Ts = np.linspace(1., 4., 50)
        else:
            # choose T-values L-dependent: more points around Tc
            Ts = np.linspace(Tc_guess - 0.5, Tc_guess + 0.5, 25)
            Ts = np.append(Ts, np.linspace(Tc_guess - 8./L, Tc_guess + 8./L, 50))
            Ts = np.sort(Ts)[::-1]
        data[L] = gen_data_L(Ts, L, N_measure)
    data['Ls'] = Ls
    save_data(output_filename, data)
    # data structure:
    #  data = {'Ls': [8, 16, ...],
    #          8: {'observables': ['E', 'M', 'C', ...],
    #              'Ts': (np.array of temperature values),
    #              'E': (np.array of mean & error, shape (len(Ts), 2)),
    #              'C': (np.array of mean & error, shape (len(Ts), 2)),
    #              ... (further observables & metadata)
    #             }
    #          ... (further L values with same structure)
    #         }
