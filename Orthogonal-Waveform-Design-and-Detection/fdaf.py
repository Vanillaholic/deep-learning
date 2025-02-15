import math
import numpy as np
import scipy.signal as sig
import scipy.fft as fft

from utils import * 

def BFDF(X, H, S):
    """
    A Block-Frequency-domain filter implementation.

    Parameters
    ----------
    X : ndarray (in STFT-domain)
        shape: (N_block x N_DFT)
        Input signal.
    H : ndarray (in STFT-domain)
        shape: (1 x N_DFT)
        Impulse response.
    S : int
        Shift sample size (block length / # of shifts)

    Returns (yields)
    ----------
    y : filter output
    """
    Y = np.zeros_like(X)

    for i in range(X.shape[0]):
        Yi = np.sum(H * X[i, :], axis=0)
        Y[i, :] = Yi
    y = fft.ifft(Y)
    y = y[:, S:].ravel()[:, None]
    return y.real

def FDAF_OS(x, d, M=2400, S=1200, alpha=0.85, delta=1e-8, mu=0.3):
    """
    A Frequency-domain adaptive filter based on overlap-add method.
    (双端对讲检测功能已删除，可用于自适应噪声抵消（ANC）等应用)

    Parameters
    ----------
    x : ndarray
        Far end signal, e.g., the sound played from the speaker.
    d : ndarray
        Near end signal, e.g., the microphone signal.
    M : int
        Block size.
    S : int
        Shift sample size.
    alpha : number
        Forgetting factor.
    delta : number
        Regularization parameter.
    mu : number
        Step size.

    Returns
    -------
    e : ndarray
        Error signal (filter output error).
    y : ndarray
        Filtered output signal.
    H : ndarray
        Final frequency-domain filter.
    p : ndarray
        PSD estimate.
    """
    x_ = get_shifted_blocks(x, M, S)
    X = fft.fft(x_, n=M)
    
    H = np.zeros((1, M))
    y = np.zeros_like(x)
    e = np.zeros_like(y)
    p = np.zeros((1, M))

    # 构造 k 矩阵和 g 矩阵（Overlap-Add 系统相关矩阵）
    k = np.zeros((S, S))
    kp = np.diagflat(np.ones(S))
    k = np.concatenate((k, kp)).T
    kp = np.zeros((1, M))
    kp[:, :S] = 1
    g = np.diagflat(kp)

    nb_iterations = len(X) - 3

    for i in range(nb_iterations):  # per block
        Xm = np.diagflat(X[i, :])

        Y = H @ Xm
        yk = (k @ (fft.ifft(Y).T)).real
        y[S*(i+1):S*(i+2)] = yk
        e[S*(i+1):S*(i+2)] = d[S*(i+1):S*(i+2)] - yk

        # 自适应更新：始终进行更新（删除了双端对讲检测部分）
        e_ = k.T @ e[S*(i+1):S*(i+2)]
        E = fft.fft(e_, axis=0, n=M)

        # PSD 估计
        p = (1 - alpha) * p + alpha * (np.abs(np.diag(Xm)) ** 2)
        mu_a = mu * np.diagflat(np.reciprocal(p + delta))
        
        # 滤波器更新
        H_upd = 2 * fft.fft(
            g @ fft.ifft(mu_a @ (np.conj(Xm).T @ E), axis=0, n=M),
            axis=0, n=M
        )
        H = H + H_upd.T

    return e, y, H, p