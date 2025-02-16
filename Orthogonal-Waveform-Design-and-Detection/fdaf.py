""" frequency domain adaptive filter """

import numpy as np
from numpy.fft import rfft 
from numpy.fft import irfft 

def fdaf(x, d, M, hop, mu=0.05, beta=0.9):
    """
    自适应滤波函数，支持可自定义的帧移长度（hop size）。

    参数：
    x -- 输入信号
    d -- 期望信号
    M -- 滤波器长度（每个数据块的长度）
    hop -- 帧移长度（每次处理的新样本数）
    mu -- 步长因子，控制滤波器系数更新的速度（默认值：0.05）
    beta -- 平滑因子，控制归一化因子的更新速度（默认值：0.9）

    返回：
    e -- 误差信号
    """
    # 初始化滤波器系数和归一化因子
    H = np.zeros(M + 1, dtype=complex)
    norm = np.full(M + 1, 1e-8)

    # 窗函数
    window = np.hanning(M)
    # 重叠缓存
    x_old = np.zeros(M)

    # 计算数据块的数量
    num_blocks = (len(x) - M) // hop + 1
    e = np.zeros(num_blocks * hop)

    for n in range(num_blocks):
        # 获取当前数据块
        start_idx = n * hop
        end_idx = start_idx + M
        x_n = np.concatenate([x_old, x[start_idx:end_idx]])
        d_n = d[start_idx:end_idx]
        # 更新重叠缓存
        x_old = x[start_idx:end_idx]

        # 频域处理
        X_n = rfft(x_n)
        y_n = irfft(H * X_n)[M:]
        e_n = d_n - y_n

        # 更新归一化因子
        e_fft = np.concatenate([np.zeros(M), e_n * window])
        E_n = rfft(e_fft)
        norm = beta * norm + (1 - beta) * np.abs(X_n) ** 2

        # 更新滤波器系数
        G = X_n.conj() * E_n / norm
        H += mu * G

        # 强制滤波器系数的后半部分为零
        h = irfft(H)
        h[M:] = 0
        H = rfft(h)

        # 存储误差信号
        e[start_idx:start_idx + hop] = e_n[:hop]

    return e
'''

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
    '''