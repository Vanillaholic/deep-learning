function frame_energies= get_frame_energy(s2,frame_length,frame_shift)
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


end
