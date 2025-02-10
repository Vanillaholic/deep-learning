%模糊函数
clc;clear;close all;

fs = 4e3;
fc = 1.5e3;
TIME = 120;
% 定义时间参数
t = (0:1/fs:TIME - 1/fs).'; % 时间向量，这里生成0到100秒的时间序列，采样间隔为1/fs
N = length(t);
f = -fs/2:fs/N:fs/2-fs/N;
Ts = 1/fs;
B = 200;    % 带宽（Hz）
c=1500;

n = 10;               % 跳频点数
lp = 2;               % 跳频周期（秒）
l = lp*fs;            % 跳频周期对应的采样点数 信号总采样数

% 计算每个跳频点的时间间隔和频率步长
tps = lp / n;          % 每个跳频点的时间间隔（秒）
tp = tps * fs;         % 每个跳频点的时间间隔对应的采样点数

deltaf = B / n;        % 频率步长（Hz）
costas = [1,6,4,3,9,2,8,7,5,10]; % 定义Costas 序列
fm = (fc) + 1 * (costas - n/2) * deltaf - deltaf/2; % 信号的跳频点频率

% 上采样至采样频率，从连续域生成采样信号
t_sub = (0:(tp-1))'/fs :1/fs:(tp)/fs-1/fs; % 时间序列，采样间隔为 1/fs
Nt = length(t_sub); % 采样点数
x = zeros(Nt*length(costas),1); % 初始化costas信号的数组

for kk = 1:n % 遍历每个跳频点
    Nx1 = (kk-1)*Nt+1; % 当前跳频点的起始索引
    Nx11 = (kk)*Nt;    % 当前跳频点的结束索引
    x(Nx1:Nx11) = exp(-1i*2*pi*fm(kk)*t_sub); % 生成信号的复包络
end

temp = [x;zeros(round(2*fs),1);x;zeros(round(2*fs),1);x;zeros(round(2*fs),1)];
P = [temp;zeros(round((TIME-12)*fs),1)];
matchedFilter = x(1:round(2*fs));%匹配滤波器系数
figure 
% 扩展时间向量 t 以匹配 x 的长度
%t = 0:1/fs:Nt*length(costas)/fs-1/fs;
%plot(t,real(x));ylim([-1.5,1.5])
t = (0:1/fs:TIME - 1/fs).'; % 时间向量，这里生成0到100秒的时间序列，采样间隔为1/fs
plot(t, real(P));ylim([-1.5,1.5]);
title(sprintf('跳频信号,周期: %fs',lp));xlabel('Time (s)') ;ylabel('Am');grid on;

%% 
r=50; 
b=2; 
len=7; 
velt= 0*c/(r*(b^len-1)); %速度分辨率
etat = 1+(velt/c);% 多普勒因子对应的速度值
[pt,qt] = rat(etat);
rs_bsig1 = resample(x,pt,qt); % 通过重采样模拟信号的拉伸或压缩
% 加入传播时延
N = size(rs_bsig1,2); 
delay = round((1000/c)*fs); % 1000m为传播距离
obsvN = delay+N; % 观测信号的总长度
s_n = [rs_bsig1 zeros(1,obsvN-N)]; % 在信号末尾填充零
sig = s_n(1:obsvN-delay); % 提取延迟信号
bsig_no = [zeros(1,delay) sig]; % 在信号前添加延迟零

clear obsvN sig N; % 清除临时变量

vel_del = c/(r*(b^len-1));% 速度分辨率
vel = 0:vel_del:vmax;
vel = [-vel(end:-1:2) vel];
eta = 1+(vel/c);
[p,q]= rat(eta);% 将多普勒因子分解为分数形式

ambig1 = cell(1, length(vel));
ambig2 = cell(1, length(vel));

for i =1:length(vel)   % 遍历每个速度分辨率单元
    
    
    re_samp_bsig1= resample(bsig_e1,p(i),q(i)); % bsig_e1 经过重采样后生成的模拟目标返回信号
    re_samp_bsig2= resample(bsig_e2,p(i),q(i));  % bsig_e2 经过重采样后生成的模拟目标返回信号
    
    % --------时域线性相关处理----------------------------------- 
    % 如果第信号的重采样长度大于观测信号长度，进行零填充
    if length(re_samp_bsig1) > length(bsig_no)    
        na = length(re_samp_bsig1) - length(bsig_no); 
        bsig_no = [bsig_no zeros(1, na+1)]; 
    end

    % 计算第一个信号的模糊函数
    ambig1{i} = abs(matchFilter(bsig_no, re_samp_bsig1, 'none'));
    
    ambigh1(1:(nl), i) = (ambig1{i}); % 存储第一个信号的模糊函数结果


    nl = length(bsig_no); % 当前观测信号的长度

end


Max1 = max(max(ambigh1)); % 计算第一个信号模糊函数的最大值
A1 = (abs(ambigh1 ./ Max1)); % 归一化第一个信号的模糊函数
EA1 = sum(sum(A1.^2)); % 计算第一个信号模糊函数的能量
EAA1 = fc * EA1 * (1/fs) * (2 * vel_del / c); % 计算第一个信号模糊函数的归一化总能量


% 定义时间方向的宽度和速度方向的宽度
k_width = 7853 - 7793; % 时间方向的宽度
v_width = 46 - 42; % 速度方向的宽度
DD = k_width.^2 + v_width.^2; % 定义模糊区域的圆形边界

% 中心点坐标（时间方向和速度方向的索引）
k0 = 7793; % 时间方向中心点索引
% v0 = 42; % 对应 Vmax = 10 m/s 的速度方向索引
% v0 = 21; % 对应 Vmax = 5 m/s 的速度方向索引
v0 = 83; % 对应 Vmax = 20 m/s 的速度方向索引

% 初始化矩阵 AAA1，用于保存主瓣区域的模糊函数数据
% AAA1 = ones(15402,83); % 对应 Vmax = 10 m/s 的速度索引范围
% AAA1 = ones(15402,41); % 对应 Vmax = 5 m/s 的速度索引范围
AAA1 = ones(15402,165); % 对应 Vmax = 5 m/s 的速度索引范围

% 遍历定义的模糊区域

for k=-k_width:1:k_width
    for m=-v_width:1:v_width        
      aa=k^2+m^2;
        if aa <= DD
            AAA1(k0+k,v0+m)=0;
        end                  
    end    
end


% 提取 AAA1 中非主瓣区域的模糊函数数据
AAA1 = AAA1 .* A1;

Max_A1=max(max(AAA1));

%% 重采样模糊函数

% 对模糊函数矩阵进行重采样，减少采样点数以提高处理效率
for i = 1:length(vel)
    % 重采样第一个模糊函数矩阵，降低采样率到原来的 1/6
    re_ambigh1(:,i) = resample(A1(:,i), 1, 6);
    % 重采样第二个模糊函数矩阵，降低采样率到原来的 1/6
    re_ambigh2(:,i) = resample(A2(:,i), 1, 6);
end

% 更新采样频率
fs = fs / 6;

% 更新重采样后的模糊函数矩阵
A1 = re_ambigh1;
A2 = re_ambigh2;

figure;
N = length(A1); % 获取模糊函数的长度
delay = 1:N; % 定义延迟索引
mesh(vel, ((delay - N/2) .* c) ./ (2 * fs), A1); % 三维网格图
xlabel('velocity (m/s, scale)'); % 速度轴标签
ylabel('range (m, delay)'); % 距离轴标签
zlabel('magnitude'); % 幅度轴标签
title('Ambiguity surface'); % 图标题
axis tight; % 紧凑显示轴范围