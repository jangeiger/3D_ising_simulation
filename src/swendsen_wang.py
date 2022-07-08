import numpy as np

import scipy
import scipy.sparse as sparse

from numba import jit


def xyz_2_idx(x, y, z, Lx, Ly, Lz):
    return z*Lx*Ly + y*Lx + x

def idx_2_xyz(idx, Lx, Ly, Lz):
    # assert idx < Lx*Ly*Lz, "Conversion from id to x,y,z wasn't possible, since the id was too big!"
    # x, y, z = idx%Ly, (idx%(Lx*Ly))//Ly, idx//(Lx*Ly)
    # assert 0 <= x < Lx, "Wrong x! (x:%d, Lx:%d)" % (x, Lx)
    # assert 0 <= y < Ly, "Wrong y! (y:%d, Ly:%d)" % (y, Ly)
    # assert 0 <= z < Lz, "Wrong z! (z:%d, Lz:%d)" % (z, Lz)
    return idx%Ly, (idx%(Lx*Ly))//Ly, idx//(Lx*Ly)

def get_bond_indices(Lx, Ly, Lz):
    N = Lx * Ly * Lz
    ver_bond_list = []
    hor_bond_list = []
    for idx in range(N):
        x, y, z = idx_2_xyz(idx, Lx, Ly, Lz)
        hor_idx = xyz_2_idx((x+1)%Lx, y, z, Lx, Ly, Lz)
        ver_idx = xyz_2_idx(x, (y+1)%Ly, z, Lx, Ly, Lz)
        dep_idx = xyz_2_idx(x, y, (z+1)%Lz, Lx, Ly, Lz)
        hor_bond_list.append([idx, hor_idx])
        ver_bond_list.append([idx, ver_idx])
        ver_bond_list.append([idx, dep_idx])


    bond_list = hor_bond_list + ver_bond_list
    bond_indices = np.array(bond_list)
    return bond_indices





def init_system(Lx, Ly, Lz):
    N = Lx * Ly * Lz
    spin_config = np.random.choice([-1,1], N)
    bond_indices = get_bond_indices(Lx, Ly, Lz)
    return spin_config, bond_indices, N




def get_bond_config(spin_config, bond_indices, T):
    '''
    get the bond_config wb for each of the bonds
    '''
    beta = 1./T
    N = np.size(spin_config)

    ZZ = spin_config[bond_indices[:,0]] * spin_config[bond_indices[:,1]]
    bond_config = np.zeros([3*N])
    rand_array = np.array(np.random.rand(3*N) > np.exp(-2.*beta)).astype(int)
    bond_config[ZZ == 1] = rand_array[ZZ == 1]
    return bond_config




@jit(nopython=True)
def flip_spin_config(spin_config, N_components, labels):
    flip_cluster = np.random.random(N_components) < 0.5   # N_components True/False values with 50/50 chance
    for n in range(len(spin_config)):
        cluster = labels[n]
        if flip_cluster[cluster]:
            spin_config[n] = - spin_config[n]


def swendsen_wang_update(spin_config, bond_indices, T):
    """
    Perform one update of the Swendsen-Wang algorithm
    """
    N = np.size(spin_config)
    bond_config = get_bond_config(spin_config, bond_indices, T=T)
    graph = sparse.csr_matrix((bond_config, (bond_indices[:, 0], bond_indices[:, 1])), shape=(N, N))
    graph = graph + graph.T
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)
    cluster_flip_or_not = np.random.randint(0, 2, size=[n_components])
    site_flip_or_not = (-1) ** cluster_flip_or_not[labels]
    newconfig = spin_config * site_flip_or_not
    return newconfig








"""
measurements
"""


def energy(spin_config, bond_indices):
    """returns energy of entire system"""
    return -1. * np.sum(spin_config[bond_indices[:, 0]] * spin_config[bond_indices[:, 1]])


def magnetization(spin_config):
    """returns magnetization of entire system"""
    return np.sum(spin_config)



"""
simulation part
"""


def simulation(spin_config, bond_indices, T, N_measure=100):
    """Perform a Monte-carlo simulation at given temperature"""
    # thermalization: without measurement
    for _ in range(N_measure//10):
        spin_config = swendsen_wang_update(spin_config, bond_indices, T)
        
    Es = []
    Ms = []
    for n in range(N_measure):
        spin_config = swendsen_wang_update(spin_config, bond_indices, T)
        Es.append(energy(spin_config, bond_indices))
        Ms.append(magnetization(spin_config))
        
    return np.array(Es), np.array(Ms)



def run(Ts, L, N_measure=100):
    spin_config, bond_indices, N = init_system(L, L, L)
    Ms = []
    absMs = []
    Es = []
    Cs = []
    for T in Ts:
        #print("simulating T = ", T, flush=True)
        E, M = simulation(spin_config, bond_indices, T, N_measure)
        Es.append(np.mean(E)/N)
        Cs.append(np.var(E)/(T**2*N))
        Ms.append(np.mean(M)/N)
        absMs.append(np.mean(np.abs(M))/N)
        
    return Es, Cs, Ms, absMs 