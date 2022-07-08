import numpy as np

import generate_data as gd
import sys

import log




if __name__ == "__main__":
    """
    expects argument for L
    """
    assert len(sys.argv)-1 == 3, log.warnStr("Unexpected amount of command line parameters, got %d, expected %d" % (len(sys.argv)-1, 3))
    L = np.array(sys.argv[1], dtype=float)
    L = int(L)
    log.green("Simulating L=%d" % (L))

    a = int(np.array(sys.argv[2], dtype=float))
    b = int(np.array(sys.argv[3], dtype=float))
    assert a < b
    log.debug("Simulating temperatures from %d/%d to %d/%d" % (a, b, a+1, b))


    #Tc_guess = None
    Tc_guess = 3.64   # good guess for the 2D Ising model; uncomment this to get
    #                    # many T-points around this value for large L (-> long runtime!)
    if Tc_guess is None:
        N_measure = 1000  # just a quick guess
        output_filename = 'data_ising_square_L=%d.pkl' % L
    else:
        N_measure = 50000
        output_filename = 'data_ising_square_largeL_L=%d.pkl' % L

    data = dict()
    if Tc_guess is None:
        # no guess for Tc available -> scan a wide range to get a first guess
        Ts = np.linspace(1., 4., 50)
    else:
        # choose T-values L-dependent: more points around Tc
        Ts = np.linspace(Tc_guess - 0.5, Tc_guess + 0.5, 25)
        Ts = np.append(Ts, np.linspace(Tc_guess - 8./L, Tc_guess + 8./L, 50))
        Ts = np.sort(Ts)[::-1]

    # cut temperatures
    Ts = Ts[int(a/b * len(Ts)):int((a+1)/b * len(Ts))]
    log.debug(Ts)

    data[L] = gd.gen_data_L(Ts, L, N_measure)
    data['Ls'] = L
    gd.save_data(output_filename+("%d_%d" % (a, b)), data)
