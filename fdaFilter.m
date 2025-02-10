function [en, yk, W] = fdaFilter(d, x, mu, M)
% 参数说明:
% d: 输入信号（期望信号，如麦克风采集的语音信号）
% x: 参考信号（如远端发送的语音信号）
% mu: 自适应滤波器的步长，控制滤波器系数更新的速度
% M: 滤波器的阶数

% 初始化变量
N = length(d); % 输入信号的长度
W = zeros(2*M, 1); % 初始化滤波器权重，长度为 2*M
en = zeros(N, 1); % 初始化误差信号向量
yk = zeros(N, 1); % 初始化滤波器输出向量

x_block = zeros(2*M, 1); % 输入信号块，长度为 2*M
power = ones(2*M, 1); % 信号功率估计，初始化为全 1

% 开始迭代
for n = 1:N
    % 更新输入信号块
    x_block = [x_block(2:end); x(n)];
    
    % 计算输入信号块的傅里叶变换
    Xk = fft(x_block);
    
    % 计算滤波器输出的频域表示
    Yk = Xk .* W;
    
    % 计算滤波器输出的时域表示
    y = real(ifft(Yk));
    
    % 提取当前时刻的滤波器输出
    yk(n) = y(M+1);
    
    % 计算误差信号
    en(n) = d(n) - yk(n);
    
    % 构造误差信号的频域表示
    Ek = zeros(2*M, 1);
    Ek(M+1) = en(n);
    Ek = fft(Ek);
    
    % 更新信号功率估计
    power = 0.9 * power + 0.1 * abs(Xk).^2;
    
    % 计算梯度
    gradient = real(ifft(conj(Xk) .* Ek ./ power));
    
    % 更新滤波器权重
    W = W + mu * fft(gradient);
end
end