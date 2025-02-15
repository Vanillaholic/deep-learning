""" frequency domain adaptive filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

def fdaf(x, d, M, mu=0.05, beta=0.9):
    H = np.zeros(M+1,dtype=float)
    norm = np.full(M+1,1e-8)

    window =  np.hanning(M)
    x_old = np.zeros(M)

    num_block = len(x) // M
    e = np.zeros(num_block*M)

    for n in range(num_block):
        x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
        d_n = d[n*M:(n+1)*M]
        x_old = x[n*M:(n+1)*M]
 
        X_n = np.fft.rfft(x_n)
        y_n = ifft(H*X_n)[M:]
        e_n = d_n-y_n

        e_fft = np.concatenate([np.zeros(M),e_n*window])
        E_n = fft(e_fft)

        norm = beta*norm + (1-beta)*np.abs(X_n)**2
        G = X_n.conj()*E_n/norm
        H = H + mu*G

        h = ifft(H)
        h[M:] = 0
        H = fft(h)

        e[n*M:(n+1)*M] = e_n
    return e