function beamformed_signals_time_domain = beamform(s,t,fs,d,c,eleNum,angles,alpha)
L = length(t);

element_signals = zeros(eleNum, L); % 初始化阵元信号矩阵
alpha = alpha-90;
for i = 1:eleNum
    delay = -(i-1)*d*sind(alpha)/c; % 计算时延
    delay_samples = round(delay * fs); % 时延转换为样本数
    element_signals(i,:) = applyDelay(s, delay_samples);
end


%% 时延波束形成
beamformed_signals_time_domain = gpuArray( zeros(length(angles), L)); % 初始化时域波束形成信号矩阵
for j = 1:length(angles)
    alpha = angles(j);
    beamformed_signal_time_domain = zeros(1, L);
    for i = 1:eleNum
        delay = (i-1)*d*sind(alpha)/c; % 计算每个阵元时延
        delay_samples = round(delay * fs); % 时延转换为样本数
        element_signal = applyDelay(element_signals(i,:), delay_samples);
        beamformed_signal_time_domain = beamformed_signal_time_domain + element_signal;
    end
    beamformed_signals_time_domain(j,:) = beamformed_signal_time_domain(1:L); % 截取有效部分
end

end 