 function seqs = gold(oct1,gen)
% 此函数用于生成gold序列，输入参数为oct1：和gen：生成多项式系数
% function seqs = gold(oct1)
% This function generates N+2=2^n+1 Gold sequences when (n mod 4 != 0);
% and N+1=2^n Gold-like sequences when (n mod 4 == 0).
% oct1 is the generator polynomial in oct form.
% reference: Dilip V. Sarwate and Michael B. Pursley, 1980.

% some equavalent generators.
% m=3, 13
% m=4, 23
% m=5, 45
% m=6, 103
% m=7, 211

% Qinghua Zhao
% Aug. 2001 at UCSD


u = m_seq(oct1,gen);
N = size(u, 2);
n = log2(N+1);
t = 1 + 2^(floor(n/2+1));
v = u(mod(0:t:N*t-1, N)+1);

if mod(n, 4) ~=0
        seqs = [u; v];
        for i = 1:N
                v = rshift(v);
                seqs = [seqs; xor(u, v)];
        end
        fprintf(1, 'Gold sequences\n');
else
        seqs = [u];
        v1 = u(mod(0:t:N*t-1, N)+2);
        v2 = u(mod(0:t:N*t-1, N)+3);
        for i = 1:N/3
                seqs = [seqs; xor(u, v); xor(u, v1); xor(u, v2)];      
                v = rshift(v); v1 = rshift(v1); v2 = rshift(v2);
        end
        fprintf(1, 'Gold-like sequences\n');
end


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

function y = rshift(x)
y = [x(:, end), x(:, 1:end-1)];
 

