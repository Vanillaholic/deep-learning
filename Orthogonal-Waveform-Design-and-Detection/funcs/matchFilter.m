function xco = matchFilter(data,signal,type)

%MATCHFILTER Matched filters the template against the signal
%
% xco = matchFilter(data,signal,type)
%   data is the data to be searched
%   signal is the template signal
%   default type is 'signal', see below for other types
%
% Matches template against the signal and returns a signal sized vector of
% correlation coefficients, each representing the correlation for a template
% matching signal beginning at that position.
%
% The type controls the estimator used. 'none' applies a linear correlator with
% no scaling, 'window' scales by the energy in the overlap window while
% cross-correlating, 'total' scales using the total energy in the data.
% 'signal' scales using the signal energy giving the best estimate of signal
% strength. 'sign' uses a sign correlator rather than a linear correlator,
% scaled such that an identity value is 1.
%
% Author: Mandar Chitre
% Last modified: Aug 17, 2004
% $Revision: 1.1 $

% check inputs
if length(data) < length(signal)
  error('Template signal is longer than the data');
end
if nargin < 3
  type = 'signal';
end

% do the cross correlation and chop the data
M = length(data);
if strcmp(type,'sign') == 1
  data = sign(data);
end
xco = xcorr(data,signal);
xco = xco(M:end);
if strcmp(type,'total') == 1
  xco = xco/sqrt(sum(data.^2)+eps);
  xco = xco/sqrt(sum(signal.^2)+eps);
elseif strcmp(type,'signal') == 1
  xco = xco/(sum(signal.^2)+eps);
elseif strcmp(type,'window') == 1
  rwin = rectwin(length(signal));
  sc = xcorr(data.^2,rwin);
  sc = sc(M:end);
  xco = xco./sqrt(sc+eps);
  xco = xco/sqrt(sum(signal.^2)+eps);
elseif strcmp(type,'sign') == 1
  xco = xco/sum(abs(signal));
elseif strcmp(type,'none') == 0
  disp('*** Warning: unknown type, defaulting to none');
end
