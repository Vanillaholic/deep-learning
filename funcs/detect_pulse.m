function [vad_signal,vad_start_index,vad_end_index,frame_energies] = detect_pulse(s2,frame_length,frame_shift,threshold)
     % frame_length帧长（样本点数）        
     % frame_shift帧移（样本点数）
     % threshold能量阈值（根据实验调整）

     %vad_signal为提取的信号
     %vad_start_index为脉冲开始的索引 

num_frames = floor(length(s2 ) / frame_shift) - 1;  % 计算帧的数量

% 计算每帧的能量
frame_energies = zeros(num_frames, 1);
for i = 1:num_frames
    start_index = (i - 1) * frame_shift + 1;
    end_index = start_index + frame_length - 1;
    frame = s2(start_index:end_index);       % 提取当前帧
    frame_energies(i) = sum(abs(frame).^2);  % 计算帧的能量
end

% 判断语音活动
vad_output = frame_energies > threshold;  % 如果能量大于阈值，则为语音活动


% 提取语音信号
vad_signal = zeros(size(s2));  % 初始化信号
for i = 1:num_frames
    start_index = (i - 1) * frame_shift + 1;
    end_index = start_index + frame_length - 1;
    if vad_output(i) == 1  % 如果是语音活动
        vad_signal(start_index:end_index) = s2(start_index:end_index);
    end
end

% 初始化起始和结束索引数组
    vad_start_index = [];
    vad_end_index = [];

    % 遍历 vad_output，找到连续的 1 区间
    i = 1;
    while i <= num_frames
        if vad_output(i) == 1
            % 找到起始索引
            start_frame = i;
            % 找到结束索引
            while i <= num_frames && vad_output(i) == 1
                i = i + 1;
            end
            end_frame = i - 1;

            % 转换为原始信号的索引
            vad_start_index(end + 1) = (start_frame - 1) * frame_shift + 1;
            vad_end_index(end + 1) = end_frame * frame_shift;
        else
            i = i + 1;
        end
    end
end
