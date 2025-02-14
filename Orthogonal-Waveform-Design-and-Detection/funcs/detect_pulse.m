function vad_sig= detect_pulse(s,frame_energies,frame_shift,threshold)
%DETECT_PULSE 此处显示有关此函数的摘要
%   此处显示详细说明
vad_sig = [];
start_idx = (find(frame_energies>threshold,1)-1)*frame_shift;
end_idx = find(frame_energies>threshold,1 ,'last')*frame_shift;
vad_sig = s(start_idx:end_idx);
    %此处理方法存在问题
    % for j  = 1:length(frame_energies)
    %     if frame_energies(j)>threshold
    %         vad_sig =[vad_sig,s((j-1)*frame_length:j*frame_length)] ;
    %     end
    % end
end

