function seq = m_seq(oct,gen)

% function seq = m_seq(oct)
% Generates an m-sequence using generator given by oct.
% Reference:  Dilip V. Sarwate and Michael B. Pursley, 1980. pp. 599. Fig.1.


s = min(find(gen));
gen = gen(s+1:end);
n = size(gen, 2);
N = 2^n-1;
gen = fliplr(gen);

seq = zeros(1, n); seq(n) = 1;
for i=1:N-n
        next_bit = mod(sum(seq(i:i+n-1)&gen), 2);
        seq = [seq, next_bit];
end
