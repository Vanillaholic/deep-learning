function IR = get_IR(Arr, fs, duration)
% generateImpulseResponse: 根据到达幅度和时延计算单位脉冲响应和频谱
% 
% 输入参数：
%   - Arr: 包含到达幅度和时延的结构体，Arr.A 为幅度，Arr.delay 为时延。
%   - fs: 采样频率 (Hz)
%   - duration: 时间长度 (秒)，定义脉冲响应的总时间范围。
% 
% 输出参数：
%   - IR: 单位脉冲响应 (时域信号)。
%   - TF1: 单位脉冲响应的频域变换。


  % 输入检查
    if ~isstruct(Arr) || ~isfield(Arr, 'A') || ~isfield(Arr, 'delay')
        error('Arr 必须是一个包含 "A" 和 "delay" 字段的结构体。');
    end
    if fs <= 0 || duration <= 0
        error('采样频率和时长必须是正数。');
    end

  % 读取到达幅度和时延
    Amp = real(Arr.A);   % 到达幅度
    delay = Arr.delay;   % 到达时延

    t = (0:1/fs:duration - 1/fs).';
    % 去除幅度为 0 的项
    validIndices = Amp ~= 0; 
    Amp = Amp(validIndices);
    delay = delay(validIndices);

    % 确保时延和幅度按时延升序排列
    [delay, sortIdx] = sort(delay);
    Amp = Amp(sortIdx);

    % 生成频域数据
    nFFT = 2^nextpow2(fs * duration); % 使用采样频率和持续时间确定 FFT 点数
    freq = (0:nFFT-1) * (fs / nFFT);  % 频率轴 (Hz)
    omega = 2 * pi * freq;            % 角频率 (rad/s)
    
    % 初始化频域响应
    TF1 = zeros(1, nFFT); 

    % 引入每个到达的时延和幅度
    for i = 1:length(Amp)
        TF1 = TF1 + Amp(i) * exp(-1j * omega * delay(i));
    end

    % 对频域数据执行逆傅里叶变换
    IR = real(ifft(TF1, nFFT)); % 得到时域脉冲响应，取实部避免复数噪声

    IR = IR(1:length(t));      % 截取前 duration 的时域数据


    % % 合并时延和幅度，去掉幅度为 0 的条目
    % Amp_Delay = [delay; Amp];
    % Amp_Delay(:, all(Amp_Delay == 0, 1)) = []; % 去掉幅度为 0 的条目
    % 
    % % 按时延从小到大排序
    % Amp_Delay = sortrows(Amp_Delay', 1);
    % 
    % % 时间向量：从 0 到 duration，以 1/fs 为采样间隔
    % t = 0:1/fs:duration - 1/fs; 
    % IR = zeros(1, length(t)); % 初始化单位脉冲响应
    % 
    % % 计算时延对应的采样索引
    % Ts  = 1/fs;
    % delayIndex = round(Amp_Delay(:, 1) ./ Ts);
    % 
    % % 将到达幅度插入到单位脉冲响应中
    % for i = 1:length(delayIndex)
    %     if delayIndex(i) > 0 && delayIndex(i) <= length(IR) % 确保索引有效
    %         IR(delayIndex(i)) = Amp_Delay(i, 2);
    %     end
    % end
end