function zcr = get_zcr(frames,frame_length,num_frames)
%DETECT_PULSE_ZCR 此处显示有关此函数的摘要
%   此处显示详细说明
zcr = zeros(num_frames, 1);
    for i = 1:num_frames
        frame = frames(:, i);
        signChanges = sum(abs(diff(sign(frame))) / 2);
        zcr(i) = signChanges / frame_length;
    end
end

