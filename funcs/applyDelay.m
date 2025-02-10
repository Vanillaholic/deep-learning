function element_signal = applyDelay(signal_padded, delay_samples)
    % 根据delay_samples的值调整信号
    if delay_samples > 0
        % 延迟为正时，在左边加入零
        element_signal = [zeros(1, delay_samples), signal_padded(1:end-delay_samples)];
    elseif delay_samples < 0
        % 延迟为负时，在右边加入零
        element_signal = [signal_padded(-delay_samples+1:end), zeros(1, -delay_samples)];
    else
        % 延迟为零时，不做改变
        element_signal = signal_padded;
    end
end
