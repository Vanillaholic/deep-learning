import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class PFDAF:
    def __init__(self, N, filter_length, hop, mu, partial_constrain):
        self.N = N
        self.filter_length = int(filter_length)  # 滤波器长度（通常对应于 overlap 部分的长度+1）
        if self.filter_length <= 0:
            raise ValueError("filter_length 必须是正整数")
        self.hop = hop                    # 帧移长度（每帧新输入样本数）
        # FFT 长度 = (filter_length - 1) + hop
        self.N_fft = self.filter_length - 1 + self.hop
        self.N_freq = self.N_fft // 2 + 1
        self.mu = mu
        self.partial_constrain = partial_constrain
        self.p = 0
        # 初始化重叠缓存，长度为 filter_length - 1
        self.x_old = np.zeros(self.filter_length - 1, dtype=float)
        # 存储过去 N 帧频域数据及滤波器系数
        self.X = np.zeros((N, self.N_freq), dtype=complex)
        self.H = np.zeros((N, self.N_freq), dtype=complex)
        # 窗函数用于误差信号，长度与 hop 相同
        self.window = np.hanning(self.hop)

    def filt(self, x, d):
        # 输入 x 长度应为 hop
        assert len(x) == self.hop, "输入帧长度必须等于 hop"
        # 拼接重叠部分与当前新帧：长度 = (filter_length - 1) + hop = N_fft
        x_now = np.concatenate([self.x_old, x])
        X = fft(x_now)
        # 更新频域数据缓冲区：最新的一帧放在第一位
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        # 更新重叠缓存：取 x_now 最后 (filter_length - 1) 个样本
        self.x_old = x_now[-(self.filter_length - 1):]
        # 计算输出（频域滤波后求和）
        Y = np.sum(self.H * self.X, axis=0)
        y_time = ifft(Y)
        # 取时域输出的最后 hop 个样本作为本帧输出
        y = y_time[-self.hop:]
        e = d - y
        return e

    def update(self, e):
        # 误差信号 e 长度应为 hop
        X2 = np.sum(np.abs(self.X)**2, axis=0)
        e_fft = np.zeros(self.N_fft, dtype=float)
        # 将窗口加权后的误差放置在 e_fft 的后 hop 个样本位置
        e_fft[-self.hop:] = e * self.window
        E = fft(e_fft)
        G = self.mu * E / (X2 + 1e-10)
        self.H += self.X.conj() * G

        if self.partial_constrain:
            h = ifft(self.H[self.p])
            # 将无效部分置零：保留最后 hop 个样本
            h[:-self.hop] = 0
            self.H[self.p] = fft(h)
            self.p = (self.p + 1) % self.N
        else:
            for p in range(self.N):
                h = ifft(self.H[p])
                h[:-self.hop] = 0
                self.H[p] = fft(h)

def pfdaf(x, d, N=4, filter_length=64, hop=None, mu=0.2, partial_constrain=True):
    # 默认帧移等于滤波器长度（即原始代码中的情况）
    if hop is None:
        hop = filter_length
    ft = PFDAF(N, filter_length, hop, mu, partial_constrain)
    # 根据 hop 来划分帧数
    num_frames = min(len(x), len(d)) // hop
    e = np.zeros(num_frames * hop)
    for n in range(num_frames):
        x_n = x[n*hop : n*hop + hop]
        d_n = d[n*hop : n*hop + hop]
        e_n = ft.filt(x_n, d_n)
        ft.update(e_n)
        e[n*hop : n*hop + hop] = e_n
    return e