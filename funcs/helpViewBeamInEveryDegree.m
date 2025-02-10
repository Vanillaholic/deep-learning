function helpViewBeamInEveryDegree(t_total,beamformed_signals_time_domain,angles,scale)

temp = flipud(beamformed_signals_time_domain);
for i = 1:length(angles)
    plot(t_total,real(temp(i,:)).'+scale*i*2);hold on;
end

l = legend('180°','170°','160°','150°','140°','130°','120°','110°','100°', ...
    '90°','80°','70','60°','50°','40°','30°','20°','10°','0°');
l.Direction = 'reverse';
xlabel('Time (s)');
ylabel('Amplitude');
title('Time-domain Beamformed Signal');
axis([50,100,-inf,inf]);

end